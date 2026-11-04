"""Recover an actual pre-adaptive batch; never synthesize its binary metadata."""
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

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/input'))
from convert_legacy_input import convert_case,namelist
from prepare_s1_flatplate_case import write_controller


def launch(executable,case,mpiexec,reject=None):
    env={key:value for key,value in os.environ.items() if not key.startswith('ASTR_')}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_coll='^cuda,hcoll,ucc',
        OMPI_MCA_opal_cuda_support='0',ASTR_FORCE_MPI_TOPOLOGY='2,1,1',ASTR_GPU_SYNC_MODE='explicit')
    primary='datin/input.flatplate' if case.name=='old_cpu' else 'datin/input.case'
    with (case/'run.log').open('w') as log:
        result=subprocess.run([str(mpiexec),'-n','2',str(executable),'run',primary],cwd=case,env=env,
            stdout=log,stderr=subprocess.STDOUT)
    text=(case/'run.log').read_text()
    if reject:
        if result.returncode==0 or reject not in text:
            raise AssertionError(f'expected {reject}: {case}/run.log')
    elif result.returncode or 'The job is done!' not in text:
        raise AssertionError(f'run failed: {case}/run.log')
    return text


def output_config(restore='',exact=True,initial=False):
    return ('&output\n directory="outdat/new",host_budget_bytes=-1,device_budget_bytes=-1,\n'
        f' restore_directory="{restore}",exact_restart={"t" if exact else "f"},'
        f'restart_output="{"override" if restore else "saved"}"\n/\n'
        f'&checkpoint\n enabled=t,interval_steps=1,keep=2,initial_frame={"t" if initial else "f"}\n/\n'
        '&volume\n enabled=f\n/\n&slices\n enabled=f\n/\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('old-cpu','cpu','gpu','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--old-case',type=Path,help='explicitly reuse an already completed pre-adaptive reference')
    args=parser.parse_args()
    for name in ('old_cpu','cpu','gpu','mpiexec','output'):
        setattr(args,name,getattr(args,name).resolve())
    args.output.mkdir(parents=True,exist_ok=False)
    if args.old_case is not None:
        old=args.old_case.resolve()
        if 'The job is done!' not in (old/'run.log').read_text():
            raise AssertionError('explicit old reference did not finish')
    else:
        old=args.output/'old_cpu'
        subprocess.run([sys.executable,str(ROOT/'tests/gpu_validation/prepare_s1_flatplate_case.py'),
            '--dst-case',str(old),'--use-gpu','f','--im','16','--jm','16','--km','16',
            '--conschm','543e','--lfilter','t','--maxstep','2','--deltat','1e-6'],
            check=True,capture_output=True,text=True)
        controller=old/'datin/controller'
        write_controller(controller,2,1,1e-6)
        controller.write_text(controller.read_text().replace('\nf,f,f,f\n','\nf,f,t,f\n')
            .replace('2,1,9999,9999,1,9999','2,1,9999,9999,1,1'))
        (old/'outdat/new').mkdir(parents=True)
        (old/'datin/input.output').write_text(output_config())
        launch(args.old_cpu,old,args.mpiexec)
    source=old/'outdat/new/checkpoints/step000000000003'
    control=(source/'control.bin').read_bytes()
    if b'ASTRDT01' in control or b'ASTRMS01' in control:
        raise AssertionError('reference is not a pre-adaptive batch')
    with h5py.File(source/'statistics.h5') as f:
        old_metadata=f['metadata'][:]
        if int(old_metadata[1])<=0 or not np.any(f['q0001'][:]):
            raise AssertionError('old reference must contain nonzero equal-sample statistics')
    report=dict(old_executable=str(args.old_cpu),old_executable_sha256=hashlib.sha256(args.old_cpu.read_bytes()).hexdigest(),
        old_control_magic=control[:8].decode(),old_statistics_metadata=old_metadata.tolist(),cases={})
    for backend in ('cpu','gpu'):
        for stop in (4,5):
            case=args.output/f'{backend}_stop{stop}'
            shutil.copytree(old/'datin',case/'datin')
            (case/'outdat/new').mkdir(parents=True)
            groups=convert_case((old/'datin/input.flatplate').read_text())
            groups['case_options']['use_gpu']=backend=='gpu'
            (case/'datin/input.case').write_text(namelist(groups))
            (case/'datin/input.run').write_text(f'&run_control max_steps={stop} /\n'
                '&timestep mode="auto",initial_dt=1e-6,growth_factor=1.1 /\n'
                '&statistics enabled=t,sample_steps=1,write_steps=0 /\n&logging list_steps=0 /\n')
            (case/'datin/input.output').write_text(output_config(source,False,True))
            log=launch(getattr(args,backend),case,args.mpiexec)
            final=case/f'outdat/new/checkpoints/step{stop:012d}'
            with h5py.File(final/'statistics.h5') as f:
                metadata=f['metadata'][:]
                # CPU stores 44 moments; the compact GPU provider stores z-integrated moments.
                if backend=='cpu':
                    count=int(metadata[1])
                    first,last=int(metadata[2]),int(metadata[3])
                    duration=float(metadata[8:9].view(np.float64)[0])
                    if metadata[0]!=3 or count!=stop-2 or first!=3 or last!=stop:
                        raise AssertionError(f'old equal-sample history was reused: {metadata}')
                    if duration<=0:
                        raise AssertionError('continued CPU window has no elapsed time')
                else:
                    duration=float(metadata[13:14].view(np.float64)[0])
                    if metadata[0] not in (3,4) or list(metadata[7:10])!=[stop-2,3,stop]:
                        raise AssertionError(f'GPU window retained old equal-sample history: {metadata}')
                    if duration<=0:
                        raise AssertionError('GPU window duration did not reset and restart')
            clocks=re.findall(r'ASTR_TIMESTEP complete_step=(\d+).*?dt_used=\s*(\S+)',log)
            if [int(step) for step,_ in clocks]!=list(range(4,stop+1)):
                raise AssertionError('restored cumulative step was reset or an extra step was advanced')
            if clocks and float(clocks[0][1])!=1e-6:
                raise AssertionError('non-exact initial_dt was not applied')
            report['cases'][case.name]=dict(metadata=metadata.tolist(),advanced_steps=[int(step) for step,_ in clocks])
    case=args.output/'exact_rejected'
    shutil.copytree(args.output/'cpu_stop4/datin',case/'datin')
    (case/'outdat/new').mkdir(parents=True)
    (case/'datin/input.output').write_text(output_config(source,True,True))
    launch(args.cpu,case,args.mpiexec,reject='missing or invalid timestep history')
    report.update(status='passed',exact_missing_history_rejected=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
