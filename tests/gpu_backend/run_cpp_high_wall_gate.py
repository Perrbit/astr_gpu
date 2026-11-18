"""Ordinary evolved-density wall stages against immutable original trajectories."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from run_cpp_tgv_gate import compare,checkpoint,logged_steps,namelist,snapshot
from run_cpp_physical_gate import stages
from run_cpp_multi_tgv_gate import layouts
from run_cpp_insitu_gate import compare_hdf


def rhs_phase(current,previous,phase,limit=2e-10,componentwise=False):
    partitions=layouts(current);maximum=np.zeros(5)
    paths=sorted((previous/'outdat').glob('rhs.'+phase+'.*.bin'))
    if len(paths)!=12:raise AssertionError('missing original high-wall RHS phases')
    def read(path):
        with path.open('rb') as f:
            header=np.fromfile(f,np.int32,4)
            return np.fromfile(f,np.float64).reshape(tuple(header[:3]+1)+(header[3],),order='F')
    for path in paths:
        whole=read(path)
        for rank,layout in partitions.items():
            actual=read(current/'outdat'/path.name.replace('rank00000000',f'rank{rank:08d}'))
            expected=whole[tuple(slice(o,o+n+1) for o,n in zip(layout['origin'],layout['upper']))+(slice(None),)]
            maximum=np.maximum(maximum,compare(actual,expected,limit=limit))
    return maximum.tolist() if componentwise else float(max(maximum))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','reference','output','mpiexec'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--ranks',type=int,choices=[1,2],default=1);parser.add_argument('--axis',choices=list('xyz'),default='y')
    parser.add_argument('--transport',choices=['pinned','device'],default='pinned')
    parser.add_argument('--sync',choices=['explicit','dependency'],default='explicit');parser.add_argument('--memcheck',action='store_true')
    parser.add_argument('--workspace',choices=['scalar','full'],default='scalar')
    parser.add_argument('--wall-mode',choices=['hermite7','keep_sbp63'],default='hermite7')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_GPU_HALO_TRANSPORT=args.transport,
        ASTR_GPU_SYNC_MODE=args.sync,ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',
        ASTR_VALIDATION_RHS_STEP_END='4',ASTR_GPU_FILTER_WORKSPACE=args.workspace)
    if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
    def run(name,steps=4,restore=None,writing=True):
        case=out/name;case.mkdir();shutil.copytree(args.reference/'gpu/datin',case/'datin',copy_function=shutil.copyfile)
        (case/'datin').chmod(0o755);(case/'outdat/new').mkdir(parents=True)
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=steps),
            timestep=dict(mode='fixed',fixed_dt=1e-5),logging=dict(list_steps=1),repair=dict(crashfix=False),
            statistics=dict(enabled=False))))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
            restore_directory=str(restore or ''),exact_restart=True,checkpoint_statistics=False,
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=writing,mode='steps',interval_steps=2,keep=2,initial_frame=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        topology=[1,1,1];topology['xyz'.index(args.axis)]=args.ranks
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks)]
        if args.memcheck and name=='continuous':
            command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer','--tool','memcheck',
                '--error-exitcode','99','--log-file',str(case/'memcheck.%p.log')]
        with (case/'run.log').open('w') as log:
            subprocess.run(command+[str(args.binary.resolve()),'run'],cwd=case,
                env=env|dict(ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology))),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        text=(case/'run.log').read_text()
        marker='ASTR_GPU_CPP sbp63_wall=on' if args.wall_mode=='keep_sbp63' else 'ASTR_GPU_CPP high_wall_axis='
        if 'backend=cuda' not in text or marker not in text:
            raise AssertionError('new C++ high-wall main was not selected')
        return case
    current=run('continuous');final=Path('outdat/new/checkpoints/step000000000004')
    report=dict(ranks=args.ranks,axis=args.axis,transport=args.transport,sync=args.sync,workspace=args.workspace,
        wall_mode=args.wall_mode,references={})
    for backend in ('cpu','gpu'):
        previous=args.reference/backend
        report['references'][backend]=dict(stages=stages(current,previous),
            convection=rhs_phase(current,previous,'conv'),full_rhs=rhs_phase(current,previous,'full'),
            terminal=compare(checkpoint(current/final/'state.h5'),checkpoint(previous/final/'state.h5')),
            scalars=compare(np.loadtxt(current/'flowstate.dat',skiprows=1,ndmin=2),np.loadtxt(previous/'flowstate.dat',skiprows=1,ndmin=2)))
        if not np.array_equal(logged_steps(current),logged_steps(previous)):raise AssertionError('high-wall clock differs')
    quiet=run('without-checkpoint',writing=False)
    for rank in layouts(current):
        name=f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
        if snapshot(current/name).tobytes()!=snapshot(quiet/name).tobytes():raise AssertionError('high-wall output changed q')
    if not np.array_equal(logged_steps(current),logged_steps(quiet)):raise AssertionError('high-wall output changed clock')
    prefix=run('prefix',2);tail=run('tail',restore=prefix/'outdat/new/checkpoints/step000000000002')
    compare_hdf(current/final/'state.h5',tail/final/'state.h5',exact=True)
    if (current/final/'control.bin').read_bytes()!=(tail/final/'control.bin').read_bytes():raise AssertionError('high-wall control restart')
    full=np.loadtxt(current/'flowstate.dat',skiprows=1,ndmin=2);resumed=np.loadtxt(tail/'flowstate.dat',skiprows=1,ndmin=2)
    if not np.array_equal(resumed,full[np.isin(full[:,0],resumed[:,0])]):raise AssertionError('high-wall scalar/driver restart')
    if args.memcheck:
        logs=list(current.glob('memcheck.*.log'))
        if len(logs)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in logs):
            raise AssertionError('missing clean high-wall memory evidence')
    report.update(output_state_clock_bits=True,exact_state_controls_scalars=True,
        binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest())
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
