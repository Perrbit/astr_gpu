"""Existing H1 viscous-flux storage candidate; ordinary solver and LP1 history."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from run_cpp_tgv_gate import namelist,compare,snapshot,checkpoint,logged_steps
from run_cpp_multi_tgv_gate import raw_field
from prepare_ldc_reference import compare_stages
from run_cpp_restart_gate import exact_state
from run_native_precision_restart_gate import check_state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('binary','original','fixture','output','mpiexec'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--ranks',type=int,choices=(1,2),default=2)
    p.add_argument('--axis',choices=('x','y','z'),default='x')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binaries={}
    for name,path in (('cpp',a.binary),('old',a.original)):
        binary=out/(name+'_astr');shutil.copy2(path,binary);binary.chmod(0o555);binaries[name]=binary
    topology=[1,1,1];topology['xyz'.index(a.axis)]=a.ranks
    env={k:v for k,v in os.environ.items() if not k.startswith(('ASTR_','OMPI_MCA_','UCX_'))}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_GPU_SYNC_MODE='explicit',
        ASTR_GPU_HALO_TRANSPORT='pinned',ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)),
        ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_RHS_STEP_END='4')
    final=Path('outdat/new/checkpoints/step000000000004')
    def run(name,implementation,mixed,steps=4,writing=False,restore=None,observe=True):
        case=out/name
        shutil.copytree(a.fixture/'gpu/datin',case/'datin',copy_function=shutil.copyfile)
        (case/'datin').chmod(0o755);(case/'outdat/new').mkdir(parents=True)
        text=(case/'datin/input.case').read_text()
        options="gpu_mp_positivity='off',gpu_precision_mode='"+('mixed_workspace' if mixed else 'fp64')+"'"
        if mixed:options+=",gpu_mixed_candidate='viscous_flux'"
        text,count=re.subn(r'&precision_options\b.*?/',lambda m:'&precision_options\n '+options+'\n/',text,flags=re.S)
        if count!=1:raise AssertionError('one existing precision-options group')
        (case/'datin/input.case').write_text(text)
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=steps),
            timestep=dict(mode='fixed',fixed_dt=1e-5),logging=dict(list_steps=1),repair=dict(crashfix=False),
            statistics=dict(enabled=False))))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
            restore_directory=str(restore or ''),exact_restart=True,checkpoint_statistics=False,
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=writing,mode='steps',interval_steps=2,keep=2,initial_frame=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        execution=dict(env)
        if not observe:execution.update(ASTR_VALIDATION_RHS_STEP='4',ASTR_VALIDATION_RHS_STEP_END='4')
        with (case/'run.log').open('x') as log:
            subprocess.run([str(a.mpiexec.resolve()),'--bind-to','none','-n',str(a.ranks),
                str(binaries[implementation]),'run'],cwd=case,env=execution,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        log=(case/'run.log').read_text()
        if 'sbli-grcbc-h1 high-order boundary chain' not in log:raise AssertionError('H1 inactive')
        if implementation=='cpp':
            if log.count('backend=cuda')!=a.ranks:raise AssertionError('C++ backend inactive')
            marker='ASTR_GPU_CPP mixed_workspace=viscous_flux'
        else:marker='ASTR_GPU_ACTIVE_MIXED_WORKSPACE=viscous_flux'
        if (marker in log)!=mixed:raise AssertionError('requested viscous storage mode inactive')
        return case
    references={mode:run('reference_'+mode,'old',mode=='mixed') for mode in ('fp64','mixed')}
    report=dict(topology=topology,dt=1e-5,binary_sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in binaries.items()},cases={})
    fields={};scalars={}
    for mode in ('fp64','mixed'):
        mixed=mode=='mixed';old=references[mode]
        full=run(mode+'_full','cpp',mixed,writing=True)
        # Same-mode trajectories retain their established physical-phase gate.
        phases=compare_stages(old,full,ranks=a.ranks)
        errors=np.zeros(5)
        for rank in range(a.ranks):
            old_name=f'outdat/rhs.post_update.step00000003.rk03.rank{rank:08d}.bin'
            new_name=f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
            errors=np.maximum(errors,compare(snapshot(full/new_name),snapshot(old/old_name)))
        actual=np.loadtxt(full/'flowstate.dat',skiprows=1,ndmin=2)
        expected=np.loadtxt(old/'flowstate.dat',skiprows=1,ndmin=2)
        scalar_errors=compare(actual,expected)
        if logged_steps(full).tolist()!=logged_steps(old).tolist():raise AssertionError('same-mode H1 clocks')
        quiet=run(mode+'_quiet','cpp',mixed,writing=False,observe=False)
        for rank in range(a.ranks):
            name=f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
            if raw_field(full/name)[1].tobytes()!=raw_field(quiet/name)[1].tobytes():
                raise AssertionError('H1 mixed output/observation changed full q')
        prefix=run(mode+'_prefix','cpp',mixed,steps=2,writing=True)
        tail=run(mode+'_tail','cpp',mixed,writing=True,restore=prefix/'outdat/new/checkpoints/step000000000002')
        count=exact_state(full/final/'state.h5',tail/final/'state.h5')
        if (full/final/'control.bin').read_bytes()!=(tail/final/'control.bin').read_bytes():
            raise AssertionError('same-mode H1 mixed control history')
        import struct
        flags=struct.unpack('<8q',(full/final/'control.bin').read_bytes()[8:72])
        if bool(flags[6]&4)!=mixed:raise AssertionError('LP1 viscous-storage identity bit')
        resumed=np.loadtxt(tail/'flowstate.dat',skiprows=1,ndmin=2)
        if not np.array_equal(resumed,actual[np.isin(actual[:,0],resumed[:,0])]):raise AssertionError('same-mode H1 scalar continuation')
        fields[mode]=checkpoint(full/final/'state.h5');scalars[mode]=actual
        report['cases'][mode]=dict(reference_phases=phases,reference_field_scaled=errors.tolist(),
            reference_scalar_scaled=scalar_errors,exact_state_datasets=count,exact_control_scalars=True,
            output_observation_q_words=True,physical=check_state(full/final/'state.h5','fp64'))
    delta=np.max(abs(fields['mixed']-fields['fp64']),axis=(0,1,2))
    scalar_delta=np.max(abs(scalars['mixed']-scalars['fp64']),axis=0)
    if max(delta)>1e-9 or max(scalar_delta)>1e-10:raise AssertionError('existing viscous-flux candidate field/scalar limits')
    report.update(mixed_vs_fp64_field_max_abs=delta.tolist(),mixed_vs_fp64_scalar_max_abs=scalar_delta.tolist(),
        field_atol=1e-9,scalar_atol=1e-10)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('H1 viscous storage: original phases, LP1 continuation and observation isolation checked')


if __name__=='__main__':main()
