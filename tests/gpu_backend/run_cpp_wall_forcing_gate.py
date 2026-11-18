"""Actual C++ wall-forcing stages and same-configuration output/restart checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import subprocess

from prepare_ldc_reference import compare_stages
from run_cpp_tgv_gate import compare,snapshot,logged_steps,namelist
from run_cpp_multi_tgv_gate import layouts
from run_cpp_insitu_gate import compare_hdf


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('binary','reference','output','mpiexec'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--transport',choices=['pinned','device'],default='pinned')
    parser.add_argument('--sync',choices=['explicit','dependency'],default='explicit')
    parser.add_argument('--checkpoint',action='store_true');parser.add_argument('--memcheck',action='store_true')
    args=parser.parse_args();meta=json.loads((args.reference/'reference.json').read_text())
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    topology=meta['topology'];ranks=int(np.prod(topology));dt=meta['dt']
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_GPU_HALO_TRANSPORT=args.transport,
        ASTR_GPU_SYNC_MODE=args.sync,ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)),
        ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_RHS_STEP_END='4')
    if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
    def run(name,steps=4,writing=args.checkpoint,restore=None,observe=True):
        case=root/name;case.mkdir();shutil.copytree(args.reference/'gpu/datin',case/'datin',copy_function=shutil.copyfile)
        (case/'datin').chmod(0o755)
        (case/'outdat/new').mkdir(parents=True)
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=steps),
            timestep=dict(mode='fixed',fixed_dt=dt),logging=dict(list_steps=1),repair=dict(crashfix=False),statistics=dict(enabled=False))))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',exact_restart=True,
            restore_directory=str(restore or ''),checkpoint_statistics=False,host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=writing,mode='steps',interval_steps=2,keep=2,initial_frame=False),volume=dict(enabled=False),slices=dict(enabled=False))))
        execution=dict(env)
        if not observe:execution.update(ASTR_VALIDATION_RHS_STEP='4',ASTR_VALIDATION_RHS_STEP_END='4')
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(ranks)]
        if args.memcheck and name=='cpp':command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
            '--tool','memcheck','--leak-check','full','--error-exitcode','99','--log-file',str(case/'memcheck.%p.log')]
        with (case/'run.log').open('w') as log:
            subprocess.run(command+[str(args.binary.resolve()),'run'],cwd=case,env=execution,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        text=(case/'run.log').read_text()
        if 'backend=cuda' not in text or 'ASTR_GPU_CPP wall_forcing rank=' not in text:raise AssertionError('actual forced GPU wall not selected')
        if list(layouts(case)[0]['topology'])!=topology:raise AssertionError('stochastic reference requires the same topology')
        return case
    current=run('cpp');report=dict(family=meta['family'],forcing=meta['forcing'],topology=topology,
        transport=args.transport,sync=args.sync,formal_output_tested=args.checkpoint,references={},
        binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest())
    for backend in ('cpu','gpu'):
        ref=args.reference/backend;updated=0.
        for p in (ref/'outdat').glob('rhs.post_update.*.bin'):
            updated=max(updated,max(compare(snapshot(current/'outdat'/p.name),snapshot(p))))
        report['references'][backend]=dict(stages=compare_stages(ref,current,ranks=ranks),post_update=updated,
            scalars=compare(np.loadtxt(current/'flowstate.dat',skiprows=1,ndmin=2),np.loadtxt(ref/'flowstate.dat',skiprows=1,ndmin=2)))
        if not np.array_equal(logged_steps(current),logged_steps(ref)):raise AssertionError('wall forcing clock')
    if args.memcheck:
        logs=list(current.glob('memcheck.*.log'))
        if len(logs)!=ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() or 'LEAK SUMMARY: 0 bytes leaked' not in p.read_text() for p in logs):raise AssertionError('wall memory evidence')
    quiet=run('observation-off',observe=False)
    for rank in range(ranks):
        name=f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
        if snapshot(current/name).tobytes()!=snapshot(quiet/name).tobytes():raise AssertionError('observer changed forced q')
    if not np.array_equal(logged_steps(current),logged_steps(quiet)):raise AssertionError('observer changed clock')
    report['observation_state_clock_bits']=True
    if args.checkpoint:
        off=run('output-off',writing=False)
        for rank in range(ranks):
            name=f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
            if snapshot(current/name).tobytes()!=snapshot(off/name).tobytes():raise AssertionError('output changed forced q')
        if not np.array_equal(logged_steps(current),logged_steps(off)):raise AssertionError('output changed clock')
        prefix=run('prefix',steps=2);tail=run('tail',restore=prefix/'outdat/new/checkpoints/step000000000002')
        final=Path('outdat/new/checkpoints/step000000000004')
        compare_hdf(current/final/'state.h5',tail/final/'state.h5',exact=True)
        if (current/final/'control.bin').read_bytes()!=(tail/final/'control.bin').read_bytes():raise AssertionError('forced control/restart history differs')
        a=np.loadtxt(current/'flowstate.dat',skiprows=1,ndmin=2);b=np.loadtxt(tail/'flowstate.dat',skiprows=1,ndmin=2)
        if not np.array_equal(b,a[np.isin(a[:,0],b[:,0])]):raise AssertionError('forced scalar/driver restart differs')
        report.update(output_state_clock_bits=True,exact_state_control_history_scalars=True)
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
