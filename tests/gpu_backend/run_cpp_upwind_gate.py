"""Ordinary TGV upwind advancement, frozen sources, output and exact restore."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from run_cpp_tgv_gate import prepare_case,compare,checkpoint,logged_steps,namelist,snapshot
from run_cpp_physical_gate import stages
from run_cpp_insitu_gate import compare_hdf
from run_cpp_multi_tgv_gate import layouts


def main():
    parser=argparse.ArgumentParser()
    for name in ('binary','reference','template','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--freeze-only',action='store_true')
    parser.add_argument('--reference-results',type=Path)
    parser.add_argument('--variant',choices=['inviscid','me6_filter'],default='inviscid')
    parser.add_argument('--ranks',type=int,choices=[1,2],default=1)
    parser.add_argument('--axis',choices=['x','y','z'],default='x')
    parser.add_argument('--transport',choices=['pinned','device'],default='pinned')
    parser.add_argument('--memcheck',action='store_true')
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_GPU_HALO_TRANSPORT=args.transport,
        ASTR_GPU_FILTER_WORKSPACE='scalar',ASTR_GPU_SYNC_MODE='explicit',ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',
        ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_RHS_STEP_END='4')
    if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
    final=Path('outdat/new/checkpoints/step000000000004')
    viscous=args.variant=='me6_filter'
    def run(name,binary,gpu,ranks,steps=4,restore=None,writing=True):
        case=root/name
        prepare_case(case,args.template.resolve(),gpu,False,False,steps=steps,dt=1e-3,
            numerics=dict(conschm='543e',difschm='643e',recon_schem=3,lchardecomp=False,lfilter=viscous,
                viscous_form='me6_opti' if viscous else 'standard'),physics=dict(diffterm=viscous))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
            restore_directory=str(restore or ''),exact_restart=True,checkpoint_statistics=False,
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=writing,mode='steps',interval_steps=2,keep=2,initial_frame=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        topology=[1,1,1];topology['xyz'.index(args.axis)]=ranks
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(ranks)]
        if args.memcheck and name=='cpp':
            command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
                '--tool','memcheck','--error-exitcode','99','--log-file',str(case/'memcheck.%p.log')]
        command += [str(binary.resolve()),'run']
        with (case/'run.log').open('w') as log:
            subprocess.run(command,cwd=case,env=env|dict(ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology))),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        return case
    if args.freeze_only:
        cpu=run('cpu',args.reference/'bin/astr_cpu',False,1)
        gpu=run('old_gpu',args.reference/'bin/astr_cuda_fortran',True,1)
        report=dict(variant=args.variant,terminal=compare(checkpoint(gpu/final/'state.h5'),checkpoint(cpu/final/'state.h5')),
            scalars=compare(np.loadtxt(gpu/'flowstate.dat',skiprows=1),np.loadtxt(cpu/'flowstate.dat',skiprows=1)))
        (root/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
        for path in root.rglob('*'):path.chmod(0o555 if path.is_dir() else 0o444)
        root.chmod(0o555);print(json.dumps(report,indent=2));return
    if args.reference_results is None:parser.error('--reference-results is required after freezing')
    current=run('cpp',args.binary,True,args.ranks)
    if 'backend=cuda' not in (current/'run.log').read_text():raise AssertionError('native C++ main not selected')
    report=dict(variant=args.variant,ranks=args.ranks,axis=args.axis,references={})
    for backend in ('cpu','old_gpu'):
        previous=args.reference_results/backend
        report['references'][backend]=dict(stages=stages(current,previous),
            terminal=compare(checkpoint(current/final/'state.h5'),checkpoint(previous/final/'state.h5')),
            scalars=compare(np.loadtxt(current/'flowstate.dat',skiprows=1),np.loadtxt(previous/'flowstate.dat',skiprows=1)))
    quiet=run('cpp-off',args.binary,True,args.ranks,writing=False)
    for rank in layouts(current):
        name=f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
        if snapshot(current/name).tobytes()!=snapshot(quiet/name).tobytes():raise AssertionError('output changed physical q')
    if not np.array_equal(logged_steps(current),logged_steps(quiet)):raise AssertionError('output changed time control')
    prefix=run('cpp-prefix',args.binary,True,args.ranks,steps=2)
    tail=run('cpp-tail',args.binary,True,args.ranks,restore=prefix/'outdat/new/checkpoints/step000000000002')
    compare_hdf(current/final/'state.h5',tail/final/'state.h5',exact=True)
    if (current/final/'control.bin').read_bytes()!=(tail/final/'control.bin').read_bytes():raise AssertionError('restored controls changed')
    full=np.loadtxt(current/'flowstate.dat',skiprows=1,ndmin=2);resumed=np.loadtxt(tail/'flowstate.dat',skiprows=1,ndmin=2)
    if not np.array_equal(resumed,full[np.isin(full[:,0],resumed[:,0])]):raise AssertionError('restored scalars changed')
    report.update(output_state_and_clock_bits_equal=True,exact_state_and_control=True,exact_scalars=True)
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
