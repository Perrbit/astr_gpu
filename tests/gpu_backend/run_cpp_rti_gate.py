"""Original RTI initialization, fixed end states and gravity in the ordinary main."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from run_cpp_tgv_gate import convert_case,namelist,compare,checkpoint,logged_steps,snapshot
from run_cpp_physical_gate import stages
from run_cpp_multi_tgv_gate import layouts
from run_cpp_insitu_gate import compare_hdf
from run_cpp_native_tgv import field as native_field,stage_report as native_stage_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','template','output','mpiexec'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--binary',type=Path);parser.add_argument('--reference-results',type=Path)
    parser.add_argument('--freeze-only',action='store_true')
    parser.add_argument('--precision',choices=['fp64','fp32'],default='fp64')
    parser.add_argument('--native-reference',type=Path)
    parser.add_argument('--variant',choices=['explicit','compact-filter'],default='explicit')
    parser.add_argument('--ranks',type=int,choices=[1,2],default=1);parser.add_argument('--axis',choices=list('xyz'),default='y')
    parser.add_argument('--transport',choices=['pinned','device'],default='pinned')
    parser.add_argument('--sync',choices=['explicit','dependency'],default='explicit')
    parser.add_argument('--memcheck',action='store_true')
    args=parser.parse_args()
    if not args.freeze_only:
        if args.binary is None or args.reference_results is None:parser.error('comparison requires binary and reference-results')
        frozen=json.loads((args.reference_results/'reference.json').read_text())
        if frozen['variant']!=args.variant or frozen.get('precision','fp64')!=args.precision:
            parser.error('reference numerical kind or discretization/filter differs')
    if args.precision=='fp32' and args.freeze_only and args.native_reference is None:
        parser.error('FP32 requires an original native reference executable')
    if args.native_reference is not None and (not args.freeze_only or args.precision!='fp32'):
        parser.error('native-reference is an explicit FP32 freezing input')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    if args.native_reference:shutil.copy2(args.native_reference,root/'original_astr')
    if not args.freeze_only:shutil.copy2(args.binary,root/'tested_astr')
    def compare_data(a,b):
        return compare(a.astype(np.float64),b.astype(np.float64),limit=None if args.precision=='fp32' else 2e-10)
    def field(path):return native_field(path)[1] if args.precision=='fp32' else snapshot(path)
    filtered=args.variant=='compact-filter';scheme='643c' if filtered else '643e'
    final=Path('outdat/new/checkpoints/step000000000004')
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_GPU_HALO_TRANSPORT=args.transport,
        ASTR_GPU_SYNC_MODE=args.sync,ASTR_GPU_FILTER_WORKSPACE='scalar',
        ASTR_GPU_COMPACT_TRANSPORT='device-aware' if args.transport=='device' else 'pinned',
        ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_RHS_STEP_END='4')
    if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
    def run(name,binary,gpu,ranks,steps=4,restore=None,writing=True):
        case=root/name;(case/'datin').mkdir(parents=True);(case/'outdat/new').mkdir(parents=True)
        groups=convert_case((args.template/'input.rti').read_text())
        groups['case_options'].update(use_gpu=gpu)
        groups['mesh'].update(ia=16,ja=16,ka=16,lreadgrid=False,lihomo=True,ljhomo=False,lkhomo=True)
        groups['physics'].update(diffterm=False)
        groups['numerics'].update(conschm=scheme,difschm=scheme,convection_form='keep',viscous_form='standard',
            lfilter=filtered,filter_scheme='compact10' if filtered else 'explicit10',lchardecomp=False)
        groups['precision_options']=dict(gpu_precision_mode='fp64')
        (case/'datin/input.case').write_text(namelist(groups))
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=steps),
            timestep=dict(mode='fixed',fixed_dt=1e-6),logging=dict(list_steps=1),
            repair=dict(crashfix=False),statistics=dict(enabled=False))))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
            restore_directory=str(restore or ''),exact_restart=True,checkpoint_statistics=False,
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=writing,mode='steps',interval_steps=2,keep=2,initial_frame=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        topology=[1,1,1];topology['xyz'.index(args.axis)]=ranks
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(ranks)]
        if args.memcheck and name=='cpp':
            command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer','--tool','memcheck',
                '--error-exitcode','99','--log-file',str(case/'memcheck.%p.log')]
        with (case/'run.log').open('w') as log:
            subprocess.run(command+[str(binary.resolve()),'run'],cwd=case,
                env=env|dict(ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology))),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        return case
    if args.freeze_only:
        cpu_binary=root/'original_astr' if args.native_reference else args.reference/'bin/astr_cpu'
        gpu_binary=root/'original_astr' if args.native_reference else args.reference/'bin/astr_cuda_fortran'
        cpu=run('cpu_y',cpu_binary,False,1)
        gpu=run('gpu_y',gpu_binary,True,1)
        report=dict(variant=args.variant,precision=args.precision,
            terminal=compare_data(checkpoint(gpu/final/'state.h5'),checkpoint(cpu/final/'state.h5')),
            scalars=compare_data(np.loadtxt(gpu/'flowstate.dat',skiprows=1,ndmin=2),np.loadtxt(cpu/'flowstate.dat',skiprows=1,ndmin=2)),
            binaries={name:hashlib.sha256(path.read_bytes()).hexdigest()
                      for name,path in [('astr_cpu',cpu_binary),('astr_cuda_fortran',gpu_binary)]})
        (root/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
        shutil.copyfile(__file__,root/'preparation.py')
        for path in root.rglob('*'):path.chmod(0o555 if path.is_dir() else 0o444)
        root.chmod(0o555);print(json.dumps(report,indent=2));return
    current=run('cpp',root/'tested_astr',True,args.ranks)
    if 'backend=cuda' not in (current/'run.log').read_text():raise AssertionError('C++ ordinary path was not selected')
    report=dict(variant=args.variant,precision=args.precision,ranks=args.ranks,axis=args.axis,transport=args.transport,sync=args.sync,
        reference_results=str(args.reference_results.resolve()),references={})
    for backend in ('cpu','gpu'):
        previous=args.reference_results/(backend+'_y')
        phase=(native_stage_report(current,previous,layouts(current)) if args.precision=='fp32' else stages(current,previous))
        report['references'][backend]=dict(stages=phase,
            terminal=compare_data(checkpoint(current/final/'state.h5'),checkpoint(previous/final/'state.h5')),
            scalars=compare_data(np.loadtxt(current/'flowstate.dat',skiprows=1,ndmin=2),np.loadtxt(previous/'flowstate.dat',skiprows=1,ndmin=2)))
        if not np.array_equal(logged_steps(current),logged_steps(previous)):
            raise AssertionError('RTI reference full-step time/dt sequence differs')
        report['references'][backend]['complete_step_clock_equal']=True
    quiet=run('cpp-off',root/'tested_astr',True,args.ranks,writing=False)
    for rank in layouts(current):
        name=f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
        if field(current/name).tobytes()!=field(quiet/name).tobytes():raise AssertionError('RTI output changed q')
    if not np.array_equal(logged_steps(current),logged_steps(quiet)):raise AssertionError('RTI output changed clock')
    prefix=run('cpp-prefix',root/'tested_astr',True,args.ranks,steps=2)
    tail=run('cpp-tail',root/'tested_astr',True,args.ranks,restore=prefix/'outdat/new/checkpoints/step000000000002')
    compare_hdf(current/final/'state.h5',tail/final/'state.h5',exact=True)
    if (current/final/'control.bin').read_bytes()!=(tail/final/'control.bin').read_bytes():raise AssertionError('RTI restart controls')
    full=np.loadtxt(current/'flowstate.dat',skiprows=1,ndmin=2);resumed=np.loadtxt(tail/'flowstate.dat',skiprows=1,ndmin=2)
    if not np.array_equal(resumed,full[np.isin(full[:,0],resumed[:,0])]):raise AssertionError('RTI restart extrema')
    if args.memcheck:
        logs=list(current.glob('memcheck.*.log'))
        if len(logs)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in logs):
            raise AssertionError('missing clean RTI memory evidence')
    report.update(output_state_clock_bits=True,exact_state_controls_extrema=True,
        binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest())
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
