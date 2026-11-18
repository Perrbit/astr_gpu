"""Real C++ time-series inlet: stage fields, ring history and exact continuation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import h5py
import numpy as np

from prepare_ldc_reference import compare_stages
from run_cpp_tgv_gate import compare,snapshot,logged_steps,namelist
from run_cpp_insitu_gate import compare_hdf
from run_cpp_multi_tgv_gate import layouts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('binary','reference','output','mpiexec'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--transport',choices=['pinned','device'],default='pinned')
    parser.add_argument('--sync',choices=['explicit','dependency'],default='explicit')
    parser.add_argument('--checkpoint',action='store_true');parser.add_argument('--memcheck',action='store_true')
    parser.add_argument('--restart-mode',choices=['exact','nonexact'],default='exact')
    parser.add_argument('--prefix-binary',type=Path,help='explicit producer for a non-exact migration check')
    args=parser.parse_args();meta=json.loads((args.reference/'reference.json').read_text())
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    topology=meta['topology'];ranks=int(np.prod(topology));dt=meta['dt']
    env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_GPU_HALO_TRANSPORT=args.transport,
        ASTR_GPU_SYNC_MODE=args.sync,ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)),
        ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_RHS_STEP_END='4')
    if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
    def run(name,steps=4,writing=args.checkpoint,restore=None,observe=True):
        binary=args.prefix_binary if name=='prefix' and args.prefix_binary is not None else args.binary
        case=root/name;case.mkdir()
        for source in ('datin','inflow'):
            shutil.copytree(args.reference/'gpu'/source,case/source,copy_function=shutil.copyfile)
            (case/source).chmod(0o755)
        (case/'outdat/new').mkdir(parents=True)
        timestep=(dict(mode='auto',initial_dt=dt,max_dt=1.6e-5,cfl_target=.5,growth_factor=1.1)
            if meta.get('automatic',False) else dict(mode='fixed',fixed_dt=dt))
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=steps),
            timestep=timestep,logging=dict(list_steps=1),repair=dict(crashfix=False),statistics=dict(enabled=False))))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',exact_restart=args.restart_mode=='exact',
            restore_directory=str(restore or ''),checkpoint_statistics=False,host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=writing,mode='steps',interval_steps=2,keep=2,initial_frame=False),volume=dict(enabled=False),slices=dict(enabled=False))))
        execution=dict(env)
        if not observe:execution.update(ASTR_VALIDATION_RHS_STEP='4',ASTR_VALIDATION_RHS_STEP_END='4')
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(ranks)]
        if args.memcheck and name=='cpp':command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
            '--tool','memcheck','--leak-check','full','--error-exitcode','99','--log-file',str(case/'memcheck.%p.log')]
        with (case/'run.log').open('w') as log:
            subprocess.run(command+[str(binary.resolve()),'run'],cwd=case,env=execution,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        if binary==args.binary and 'ASTR_GPU_CPP dynamic_inflow device_bytes=' not in (case/'run.log').read_text():raise AssertionError('C++ inlet not selected')
        if binary==args.binary:
            actual_topology=list(layouts(case)[0]['topology'])
        else:
            match=re.search(r'forced mpi topology=\s+(\d+)\s+(\d+)\s+(\d+)',(case/'run.log').read_text())
            if match is None:raise AssertionError('original producer topology record missing')
            actual_topology=list(map(int,match.groups()))
        if actual_topology!=topology:raise AssertionError('reference topology differs')
        return case
    current=run('cpp');report=dict(topology=topology,transport=args.transport,sync=args.sync,restart_mode=args.restart_mode,
        formal_output_tested=args.checkpoint,binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),references={})
    for backend in ('cpu','gpu'):
        ref=args.reference/backend;updated=0.;paths=list((ref/'outdat').glob('rhs.post_update.*.bin'))
        if len(paths)!=12*ranks:raise AssertionError('reference stage count')
        for path in paths:updated=max(updated,max(compare(snapshot(current/'outdat'/path.name),snapshot(path))))
        report['references'][backend]=dict(stages=compare_stages(ref,current,ranks=ranks),post_update=updated,
            scalars=compare(np.loadtxt(current/'flowstate.dat',skiprows=1,ndmin=2),np.loadtxt(ref/'flowstate.dat',skiprows=1,ndmin=2)))
        if not np.array_equal(logged_steps(current),logged_steps(ref)):raise AssertionError('inlet clock')
    if args.memcheck:
        logs=list(current.glob('memcheck.*.log'))
        if len(logs)!=ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() or 'LEAK SUMMARY: 0 bytes leaked' not in p.read_text() for p in logs):raise AssertionError('inlet memory evidence')
    for name,writing,observe in [('observation-off',args.checkpoint,False),('output-off',False,True)]:
        quiet=run(name,writing=writing,observe=observe)
        for rank in range(ranks):
            path=f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
            if snapshot(current/path).tobytes()!=snapshot(quiet/path).tobytes():raise AssertionError(name+' changed inlet q')
        if not np.array_equal(logged_steps(current),logged_steps(quiet)):raise AssertionError(name+' changed clock')
    report['observation_output_bits']=True
    if args.checkpoint:
        prefix=run('prefix',steps=2);tail=run('tail',restore=prefix/'outdat/new/checkpoints/step000000000002')
        if args.prefix_binary is not None:report['prefix_binary_sha256']=hashlib.sha256(args.prefix_binary.read_bytes()).hexdigest()
        final=Path('outdat/new/checkpoints/step000000000004')
        exact=args.restart_mode=='exact'
        report['restart_state_scaled']=compare_hdf(current/final/'state.h5',tail/final/'state.h5',exact=exact)
        compare_hdf(current/final/'inflow.h5',tail/final/'inflow.h5',exact=True)
        if exact and (current/final/'control.bin').read_bytes()!=(tail/final/'control.bin').read_bytes():raise AssertionError('inlet control differs')
        a=np.loadtxt(current/'flowstate.dat',skiprows=1,ndmin=2);b=np.loadtxt(tail/'flowstate.dat',skiprows=1,ndmin=2)
        expected=a[np.isin(a[:,0],b[:,0])]
        if exact:
            if not np.array_equal(b,expected):raise AssertionError('inlet scalar restart differs')
        else:
            report['restart_scalar_scaled']=compare(b,expected)
        if not np.array_equal(logged_steps(tail,count=2,first_step=3),logged_steps(current)[2:]):raise AssertionError('inlet restart clock')
        with h5py.File(current/final/'inflow.h5') as h:
            metadata=h['metadata'][...].reshape(-1)
            report['inflow_metadata']=metadata.tolist()
            if metadata[2]<=3:raise AssertionError('checkpoint did not save rolled inflow cursor')
        report['inflow_history_bits']=True
        report['exact_state_inflow_control_scalars']=exact
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
