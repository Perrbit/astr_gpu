"""AIR5 full-state non-exact recovery from actual pre-adaptive CPU/GPU batches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import h5py
import numpy as np

from run_output_air5_restart_validation import prepare
from run_output_restart_validation import compare_fields
from run_adaptive_tgv_gate import STEP

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/input'))
from convert_legacy_input import convert_case,namelist

MODELS=dict(ASTR_AIR5_SOURCE_MODE='coupled',ASTR_AIR5_COMPENSATION='on',
    ASTR_AIR5_CONVECTION_LIMITER='symmetric_species',ASTR_AIR5_DIFFUSION_LIMITER='layered',
    ASTR_AIR5_TOP_MODE='prescribed',ASTR_AIR5_FILTER_MODE='conservative-flux',
    ASTR_AIR5_STEP_HISTORY='previous')


def output(restore='',exact=True,checkpoint=True):
    return ('&output\n directory="outdat/new",host_budget_bytes=-1,device_budget_bytes=-1,\n'
        f' restore_directory="{restore}",exact_restart={"t" if exact else "f"},'
        f'restart_output="{"override" if restore else "saved"}"\n/\n'
        f'&checkpoint\n enabled={"t" if checkpoint else "f"},interval_steps=1,keep=2\n/\n'
        '&volume\n enabled=f\n/\n&slices\n enabled=f\n/\n')


def launch(args,case,old=False,probe=False,reject=None):
    env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_coll='^cuda,hcoll,ucc',
        OMPI_MCA_opal_cuda_support='0',ASTR_FORCE_MPI_TOPOLOGY='2,1,1',ASTR_GPU_SYNC_MODE='explicit',
        ASTR_AIR5_FILTER_VALIDATION='on',ASTR_AIR5_TOP_GPU_VALIDATION='on')
    if old:
        env.update(MODELS)
    if probe:
        env['ASTR_CHECKPOINT_TEST_RESTORE_PROBE']='1'
    executable=args.old_executable if old else args.executable
    with (case/'run.log').open('w') as log:
        result=subprocess.run([str(args.mpiexec),'-n','2',str(executable),'run',
            'datin/input.air5_c4' if old else 'datin/input.case'],cwd=case,env=env,
            stdout=log,stderr=subprocess.STDOUT)
    text=(case/'run.log').read_text()
    if reject:
        if result.returncode==0 or reject not in text:
            raise AssertionError(f'expected {reject}: {case}/run.log')
    elif result.returncode or 'The job is done!' not in text:
        raise AssertionError(f'failed: {case}/run.log')
    return text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('old-executable','executable','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--old-root',type=Path,help='explicitly reuse the completed old_cpu/old_gpu cases')
    args=parser.parse_args()
    for name in ('old_executable','executable','mpiexec','output'):
        setattr(args,name,getattr(args,name).resolve())
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(old_executable=str(args.old_executable),
        old_executable_sha256=hashlib.sha256(args.old_executable.read_bytes()).hexdigest(),cases={})
    for backend in ('cpu','gpu'):
        if args.old_root is None:
            old=args.output/f'old_{backend}'
            prepare(old,backend,3,mean_statistics=True,sample_interval=1)
            (old/'outdat/new').mkdir(parents=True)
            # The old parser predates -1 spelling; use the same unlimited-budget constant as output_config.
            old_output=output().replace('=-1','='+str((2**63-1)//4096))
            (old/'datin/input.output').write_text(old_output)
            launch(args,old,old=True)
        else:
            old=args.old_root.resolve()/f'old_{backend}'
            if 'The job is done!' not in (old/'run.log').read_text():
                raise AssertionError('old reference has not completed')
        source=old/'outdat/new/checkpoints/step000000000003'
        control=(source/'control.bin').read_bytes()
        if b'ASTRDT01' in control or b'ASTRMS01' in control:
            raise AssertionError('source is not a pre-adaptive batch')
        if (source/'insitu_control.bin').read_bytes()[:8]!=b'ASTRIR07':
            raise AssertionError('source predates the approved dynamic-product checkpoint format')
        with h5py.File(source/'state.h5') as f:
            if len([key for key in f if re.fullmatch('q[0-9]{4}',key)])!=35:
                raise AssertionError('source has no full chemistry substep history')
            carry_nonzero=sum(np.count_nonzero(f[f'q{index:04d}'][:]) for index in range(12,23))
            if not carry_nonzero or not np.any(f['q0035'][:]>0):
                raise AssertionError('old reference cannot test nonzero carry/history preservation')
        for stop in (3,5):
            case=args.output/f'{backend}_stop{stop}'
            shutil.copytree(old/'datin',case/'datin')
            (case/'outdat/new').mkdir(parents=True)
            groups=convert_case((old/'datin/input.air5_c4').read_text(),model_environment=MODELS)
            (case/'datin/input.case').write_text(namelist(groups))
            (case/'datin/input.run').write_text(f'&run_control max_steps={stop} /\n'
                '&timestep mode="fixed",fixed_dt=2e-10 /\n&logging list_steps=0 /\n'
                '&statistics enabled=t,sample_steps=1 /\n')
            (case/'datin/input.output').write_text(output(source,False,stop>3))
            log=launch(args,case,probe=stop==3)
            clocks=[[int(m[1]),*map(float,m.groups()[1:])] for m in STEP.finditer(log)]
            if [row[0] for row in clocks]!=list(range(4,stop+1)) or any(row[3]!=2e-10 for row in clocks):
                raise AssertionError('current fixed dt/cumulative step was not applied')
            record=dict(clocks=clocks,nonzero_carry_values=int(carry_nonzero))
            if stop==3:
                compare_fields(source/'state.h5',case/'outdat/restore_probe.h5')
                record['all_full_state_datasets_preserved']=True
            else:
                final=case/'outdat/new/checkpoints/step000000000005'
                if not (case/'outdat/new/resources/geometry.h5').is_file():
                    raise AssertionError('AIR5 non-exact batch dropped full geometry')
                with h5py.File(final/'state.h5') as f:
                    if f['identity'][-1]!=7 or 'q0035' not in f:
                        raise AssertionError('AIR5 non-exact checkpoint became lightweight')
                with h5py.File(final/'statistics.h5') as f:
                    meta=f['metadata'][:]
                    if list(meta[:4])!=[3,3,3,5]:
                        raise AssertionError(f'statistics reused the old equal-sample window: {meta}')
                    duration=meta[8:9].view(np.float64)[0]
                    np.testing.assert_allclose(duration,clocks[-1][2]-clocks[0][1],rtol=64*np.finfo(float).eps,atol=0)
                record.update(full_geometry_retained=True,statistics_new_window=True)
            report['cases'][case.name]=record
        bad=args.output/f'{backend}_changed_chemistry'
        shutil.copytree((args.output/f'{backend}_stop3/datin'),bad/'datin')
        (bad/'outdat/new').mkdir(parents=True)
        primary=bad/'datin/input.case'
        primary.write_text(primary.read_text().replace("source_mode = 'coupled'","source_mode = 'vt'"))
        launch(args,bad,reject='AIR5 source/limiter/compensation/top contract mismatch')
    report['status']='passed'
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
