"""Existing GRCBC/KEEP main coupling with the approved LA3-14 corner ownership."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from run_keep_incident_gate import setup,FINAL,characteristic_closure,compare_open_rhs
from run_cpp_tgv_gate import compare,checkpoint,logged_steps,namelist
from run_cpp_physical_gate import stages
from run_cpp_multi_tgv_gate import layouts,raw_field
from run_cpp_restart_gate import exact_state


def original_rhs_phases(cpu,gpu):
    paths=sorted((cpu/'outdat').glob('rhs.post_source.*.bin'))
    if len(paths)!=12:raise AssertionError('original four-step reference requires 12 complete RHS phases')
    errors={}
    for path in paths:
        arrays=[];headers=[]
        for case in (cpu,gpu):
            with (case/'outdat'/path.name).open('rb') as stream:
                header=np.fromfile(stream,np.int32,4)
                headers.append(header)
                arrays.append(np.fromfile(stream,np.float64).reshape(tuple(header[:3]+1)+(header[3],),order='F'))
        if not np.array_equal(*headers):raise AssertionError('original RHS phase layouts differ')
        errors[path.name]=compare(arrays[1],arrays[0])
    return errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','mpiexec','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--cpu',type=Path)
    parser.add_argument('--old-gpu',type=Path)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--freeze-only',action='store_true')
    parser.add_argument('--viscosity',choices=('off','standard','me6_opti'),default='off')
    parser.add_argument('--ranks',type=int,choices=(1,2),default=1)
    parser.add_argument('--axis',choices=tuple('xyz'),default='x')
    parser.add_argument('--transport',choices=('pinned','device'),default='pinned')
    parser.add_argument('--sync',choices=('explicit','dependency'),default='explicit')
    parser.add_argument('--scheme',choices=('643e','643c'),default='643e')
    parser.add_argument('--diffusion',choices=('643e','643c'),default='643e')
    parser.add_argument('--entry-x',type=float,default=.25)
    parser.add_argument('--memcheck',action='store_true')
    args=parser.parse_args()
    if not args.reference and (not args.cpu or not args.old_gpu):
        parser.error('freeze requires both original executables; comparison may reuse --reference')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=out/'tested_astr';shutil.copy2(args.binary.resolve(),binary)
    shutil.copy2(__file__,out/'runner.py')
    topology=[1,1,1];topology['xyz'.index(args.axis)]=args.ranks
    identity=dict(scheme=args.scheme,diffusion=args.diffusion,entry_x=args.entry_x,viscosity=args.viscosity)
    report=dict(scope='16-cubed four-step low-order KEEP/GRCBC coupling; not production SBLI physics',
        **identity,topology=topology,transport=args.transport,sync=args.sync,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),references={})
    sources=['src_gpu/fluid_runtime.F90','src_gpu/boundary_state.cu','src_gpu/boundary_state.h',
        'src_gpu/boundary_state.F90','src_gpu/nscbc_incident_math.cuh','src_gpu/compact_operator.F90',
        'src_gpu/device_state_internal.cuh','src_gpu/device_state.cu','src/nscbc_incident_body.inc',
        'src/bc.F90','src/output_runtime.F90','tests/gpu_validation/run_keep_incident_gate.py',
        'tests/gpu_backend/run_cpp_incident_gate.py']
    report['source_sha256']={}
    for name in sources:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
        report['source_sha256'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()

    def run(name,backend,executable,local_topology,steps=4,restore=None,writing=True,observe=True,instrument=False):
        case=out/name
        env=setup(case,backend,args.entry_x,restore=restore,scheme=args.scheme,diffusion=args.diffusion,
            viscosity=args.viscosity)
        env.update(ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,local_topology)),ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',
                   ASTR_VALIDATION_RHS_STEP='0' if observe else '4',ASTR_VALIDATION_RHS_STEP_END='4')
        if name not in ('cpu_np1','old_gpu_np1'):
            env.update(ASTR_GPU_HALO_TRANSPORT=args.transport,ASTR_GPU_SYNC_MODE=args.sync,
                ASTR_GPU_COMPACT_TRANSPORT='device-aware' if args.transport=='device' else 'pinned',
                ASTR_GPU_COMPACT_SYNC=args.sync)
            if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=steps),
            timestep=dict(mode='fixed',fixed_dt=1e-5),logging=dict(list_steps=1),
            repair=dict(crashfix=False),statistics=dict(enabled=False))))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
            restore_directory=str(restore or ''),restart_output='override' if restore else 'saved',exact_restart=True,
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0,checkpoint_statistics=False),
            checkpoint=dict(enabled=writing,interval_steps=2,keep=2,initial_frame=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(np.prod(local_topology))]
        if instrument:
            command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
                '--tool','memcheck','--leak-check','full','--error-exitcode','99','--log-file',str(case/'memcheck.%p.log')]
        command += [str(executable),'run']
        with (case/'run.log').open('w') as log:
            subprocess.run(command,cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if name not in ('cpu_np1','old_gpu_np1') and (case/'run.log').read_text().count('backend=cuda')!=args.ranks:
            raise AssertionError('ordinary C++ backend not selected')
        if instrument:
            logs=list(case.glob('memcheck.*.log'))
            if len(logs)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() or
                    'LEAK SUMMARY: 0 bytes leaked in 0 allocations' not in p.read_text() for p in logs):
                raise AssertionError('missing per-rank zero-error/zero-leak memcheck')
        if writing and b'sbli-grcbc-v3' not in (case/f'outdat/new/checkpoints/step{steps:012d}/control.bin').read_bytes():
            raise AssertionError('new low-order GRCBC numerical identity missing')
        report.setdefault('commands',{})[name]=command
        return case

    if args.reference:
        reference=args.reference.resolve();previous=json.loads((reference/'report.json').read_text())
        if any(previous[key]!=value for key,value in identity.items()):raise AssertionError('reference fixture differs')
    else:
        reference=out
        for name,backend,source in [('cpu_np1','cpu',args.cpu),('old_gpu_np1','gpu',args.old_gpu)]:
            frozen=out/(name+'_astr');shutil.copy2(source.resolve(),frozen)
            report.setdefault('reference_binary_sha256',{})[name]=hashlib.sha256(frozen.read_bytes()).hexdigest()
            run(name,backend,frozen,[1,1,1])
        report['original_state_difference']=compare(checkpoint(out/'old_gpu_np1'/FINAL/'state.h5'),
            checkpoint(out/'cpu_np1'/FINAL/'state.h5'))
    if args.freeze_only:
        if args.reference:parser.error('--freeze-only requires actual original executables, not another reference')
        report['original_same_phase']=compare_open_rhs(out/'cpu_np1',out/'old_gpu_np1',out)
        report['original_all_rk_phases']=original_rhs_phases(out/'cpu_np1',out/'old_gpu_np1')
        report['independent_characteristic_closure']={name:characteristic_closure(out/name,args.entry_x,args.viscosity!='off')
            for name in ('cpu_np1','old_gpu_np1')}
        report['frozen_reference']=True
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        for p in out.rglob('*'):p.chmod(0o555 if p.is_dir() or p.name.endswith('_astr') else 0o444)
        out.chmod(0o555)
        print(json.dumps({key:report[key] for key in ('original_state_difference','original_same_phase','independent_characteristic_closure')},indent=2));return
    current=run('cpp','gpu',binary,topology)
    for name in ('cpu_np1','old_gpu_np1'):
        previous=reference/name
        report['references'][name]=dict(stages=stages(current,previous),
            terminal=compare(checkpoint(current/FINAL/'state.h5'),checkpoint(previous/FINAL/'state.h5')),
            scalar=compare(np.loadtxt(current/'flowstate.dat',skiprows=1,ndmin=2),
                           np.loadtxt(previous/'flowstate.dat',skiprows=1,ndmin=2)))
    report['top_density_response']=float(np.max(abs(checkpoint(current/FINAL/'state.h5')[:,-1,:,0]-1)))
    if args.entry_x<1 and report['top_density_response']<=1e-12:raise AssertionError('no incident response')
    if args.entry_x>1:
        state=checkpoint(current/FINAL/'state.h5')
        uniform=np.broadcast_to(np.array([1.,1.,0.,0.,1/(5.6*.4)+.5]),state.shape)
        report['uniform_preservation_scaled']=compare(state,uniform)
    if args.ranks==1:
        report['independent_characteristic_closure']={name:characteristic_closure(case,args.entry_x,args.viscosity!='off')
            for name,case in [('cpu',reference/'cpu_np1'),('old_gpu',reference/'old_gpu_np1'),('cpp',current)]}
    quiet=run('quiet','gpu',binary,topology,writing=False,observe=False)
    if not np.array_equal(logged_steps(current),logged_steps(quiet)):raise AssertionError('observation changed step clock')
    for rank in layouts(current):
        suffix=f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
        header,active=raw_field(current/suffix);other_header,inactive=raw_field(quiet/suffix)
        if not np.array_equal(header,other_header) or active.tobytes()!=inactive.tobytes():
            raise AssertionError('observation/output changed physical q or conservative halo')
    report['output_observation_isolation']=True
    tail=run('tail','gpu',binary,topology,restore=current/'outdat/new/checkpoints/step000000000002',
             instrument=args.memcheck)
    report['exact_state_datasets']=exact_state(tail/FINAL/'state.h5',current/FINAL/'state.h5')
    for name in ('control.bin','archives.bin','insitu_control.bin'):
        if (tail/FINAL/name).read_bytes()!=(current/FINAL/name).read_bytes():raise AssertionError('exact control '+name)
    baseline=np.loadtxt(current/'flowstate.dat',skiprows=1,ndmin=2)
    resumed=np.loadtxt(tail/'flowstate.dat',skiprows=1,ndmin=2)
    if not np.array_equal(resumed,baseline[np.isin(baseline[:,0],resumed[:,0])]):
        raise AssertionError('exact scalar continuation')
    if not np.array_equal(logged_steps(tail,2,3),logged_steps(current)[2:]):raise AssertionError('exact restart step clock')
    report['exact_control_scalars_clock']=True
    report['memcheck']=args.memcheck
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('topology','references','top_density_response','exact_state_datasets')},indent=2))


if __name__=='__main__':main()
