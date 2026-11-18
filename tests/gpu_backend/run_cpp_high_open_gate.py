"""Original production H1 stages and ordinary C++ coupling, not H1 robustness promotion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from run_cpp_tgv_gate import compare,checkpoint,logged_steps,namelist,snapshot
from run_cpp_multi_tgv_gate import raw_field,layouts
from run_cpp_physical_gate import stages
from run_cpp_restart_gate import exact_state

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from test_grcbc_oblique_shock import prepare as prepare_oblique
from run_keep_incident_gate import setup as prepare_weak_profile
from named_fixture import prepare_named_run

FINAL=Path('outdat/new/checkpoints/step000000000004')


def prepare(case,backend,fixture,steps=4,restore=None,writing=True,automatic=False,exact=True):
    if fixture=='viscous':
        prepare_weak_profile(case,backend,.25,viscosity='standard')
    else:
        prepare_oblique(case,backend,16,steps,initial=fixture,entry_x=2. if fixture=='entropy' else .25,z_intervals=16)
    env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',OMP_NUM_THREADS='1',
        ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_HALO_TRANSPORT='pinned',ASTR_FORCE_MPI_TOPOLOGY='1,1,1',
        ASTR_NSCBC_FARFIELD_MODE='sbli_grcbc',ASTR_NSCBC_FARFIELD_SHOCK_X='2' if fixture=='entropy' else '.25',
        ASTR_NSCBC_FARFIELD_SHOCK_ANGLE_DEG='40',ASTR_NSCBC_FARFIELD_RHO='1',ASTR_NSCBC_FARFIELD_U='1',
        ASTR_NSCBC_FARFIELD_V='0',ASTR_NSCBC_FARFIELD_W='0',ASTR_NSCBC_FARFIELD_T='1',
        ASTR_GPU_MP_POSITIVITY='off',ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',
        ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_RHS_STEP_END=str(steps),ASTR_VALIDATION_GRADIENTS='1')
    boundary=dict(bctype=[11,22,41,52,1,1],twall=[0,0,1,0,0,0],pout=1.) if fixture=='viscous' else None
    prepare_named_run(case,'datin/input.flatplate',env,numerics=dict(convection_form='standard',viscous_form='standard',
        conschm='743e',difschm='643e',recon_schem=5,bfacmpld=.1,lchardecomp=True,lfilter=False),
        physics=dict(diffterm=fixture=='viscous',reynolds=1000. if fixture=='viscous' else 1e6),boundary=boundary)
    timestep=(dict(mode='auto',cfl_target=.5,initial_dt=1e-5,max_dt=2e-5,growth_factor=1.1)
              if automatic else dict(mode='fixed',fixed_dt=1e-5))
    (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=steps),
        timestep=timestep,logging=dict(list_steps=1),
        repair=dict(crashfix=False),statistics=dict(enabled=False))))
    (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
        restore_directory=str(restore or ''),restart_output='override' if restore else 'saved',exact_restart=exact,
        checkpoint_statistics=False,host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
        checkpoint=dict(enabled=writing,interval_steps=2,keep=2,initial_frame=False),
        volume=dict(enabled=False),slices=dict(enabled=False))))
    if restore:
        for name in ('grid.flatplate.h5','inlet.prof','flowini3d.h5'):(case/'datin'/name).unlink()
    return env


def original_phases(gpu,cpu):
    errors={}
    for name,primitive,halo in [('pre_rhs',False,True),('pre_rhs_primitives',True,True),
                                ('post_update',False,True),('post_source',False,False)]:
        paths=sorted((cpu/'outdat').glob(f'rhs.{name}.*.bin'))
        if len(paths)!=12:raise AssertionError('original H1 phases missing '+name)
        maximum=0.
        for path in paths:
            values=[]
            for p in (gpu/'outdat'/path.name,path):
                if halo:
                    header,array=raw_field(p,primitive=primitive);h=header[3]
                    values.append(array[tuple(slice(h,h+n+1) for n in header[:3])+(slice(None),)])
                else:
                    with p.open('rb') as stream:
                        header=np.fromfile(stream,np.int32,4)
                        values.append(np.fromfile(stream,np.float64).reshape(tuple(header[:3]+1)+(header[3],),order='F'))
            maximum=max(maximum,max(compare(values[0],values[1])))
        errors[name]=maximum
    return errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('mpiexec','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--fixture',choices=('entropy','shock','viscous'),required=True)
    parser.add_argument('--cpu',type=Path);parser.add_argument('--old-gpu',type=Path)
    parser.add_argument('--reference',type=Path);parser.add_argument('--binary',type=Path)
    parser.add_argument('--ranks',type=int,choices=(1,2),default=1)
    parser.add_argument('--axis',choices=tuple('xyz'),default='x')
    parser.add_argument('--transport',choices=('pinned','device'),default='pinned')
    parser.add_argument('--sync',choices=('explicit','dependency'),default='explicit')
    parser.add_argument('--memcheck',action='store_true')
    parser.add_argument('--automatic',action='store_true')
    parser.add_argument('--nonexact-source',type=Path)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(fixture=args.fixture,timestep='automatic' if args.automatic else 'fixed',
        scope='16-cubed four-step production H1 short coupling, no candidate or long-window stability claim')
    shutil.copy2(__file__,out/'runner.py')

    def run(name,backend,binary,topology,steps=4,restore=None,writing=True,observe=True,instrument=False,exact=True):
        case=out/name;env=prepare(case,backend,args.fixture,steps,restore,writing,args.automatic,exact)
        env.update(ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)))
        if name not in ('cpu','gpu'):
            env.update(ASTR_GPU_HALO_TRANSPORT=args.transport,ASTR_GPU_SYNC_MODE=args.sync)
            if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
        if not observe:env.update(ASTR_VALIDATION_RHS_STEP='4',ASTR_VALIDATION_RHS_STEP_END='4')
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(np.prod(topology))]
        if instrument:command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
            '--tool','memcheck','--leak-check','full','--error-exitcode','99','--log-file',str(case/'memcheck.%p.log')]
        command += [str(binary),'run']
        with (case/'run.log').open('w') as log:
            subprocess.run(command,cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if 'sbli-grcbc-h1 high-order boundary chain' not in (case/'run.log').read_text():
            raise AssertionError('H1 configuration inactive')
        if name not in ('cpu','gpu') and (case/'run.log').read_text().count('backend=cuda')!=args.ranks:
            raise AssertionError('C++ device backend inactive')
        if name not in ('cpu','gpu') and 'ASTR_GPU_CPP high_open=production-h1' not in (case/'run.log').read_text():
            raise AssertionError('C++ production H1 assembly inactive')
        if instrument:
            logs=list(case.glob('memcheck.*.log'))
            if len(logs)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() or
                'LEAK SUMMARY: 0 bytes leaked in 0 allocations' not in p.read_text() for p in logs):
                raise AssertionError('missing clean per-rank H1 Memcheck')
        report.setdefault('commands',{})[name]=command
        return case

    if not args.reference:
        if not args.cpu or not args.old_gpu:parser.error('freezing requires original CPU and CUDA Fortran executables')
        (out/'bin').mkdir()
        report['binaries']={}
        for name,source in [('cpu',args.cpu),('gpu',args.old_gpu)]:
            binary=out/'bin'/name;shutil.copy2(source.resolve(),binary)
            report['binaries'][name]=hashlib.sha256(binary.read_bytes()).hexdigest()
            run(name,name,binary,[1,1,1])
        report['phases']=original_phases(out/'gpu',out/'cpu')
        report['terminal']=compare(checkpoint(out/'gpu'/FINAL/'state.h5'),checkpoint(out/'cpu'/FINAL/'state.h5'))
        report['scalars']=compare(np.loadtxt(out/'gpu/flowstate.dat',skiprows=1,ndmin=2),
                                  np.loadtxt(out/'cpu/flowstate.dat',skiprows=1,ndmin=2))
    else:
        if not args.binary:parser.error('comparison requires --binary')
        reference=args.reference.resolve()
        if json.loads((reference/'report.json').read_text())['fixture']!=args.fixture:raise AssertionError('different H1 fixture')
        binary=out/'tested_astr';shutil.copy2(args.binary.resolve(),binary)
        topology=[1,1,1];topology['xyz'.index(args.axis)]=args.ranks
        current=run('cpp','gpu',binary,topology)
        report.update(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),topology=topology,
                      transport=args.transport,sync=args.sync,reference={})
        for name in ('cpu','gpu'):
            previous=reference/name
            report['reference'][name]=dict(stages=stages(current,previous),
                terminal=compare(checkpoint(current/FINAL/'state.h5'),checkpoint(previous/FINAL/'state.h5')),
                scalars=compare(np.loadtxt(current/'flowstate.dat',skiprows=1,ndmin=2),
                                np.loadtxt(previous/'flowstate.dat',skiprows=1,ndmin=2)))
            if not np.array_equal(logged_steps(current),logged_steps(previous)):raise AssertionError('different H1 step clock')
        quiet=run('quiet','gpu',binary,topology,writing=False,observe=False)
        for rank in layouts(current):
            suffix=f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
            a,x=raw_field(current/suffix);b,y=raw_field(quiet/suffix)
            if not np.array_equal(a,b) or x.tobytes()!=y.tobytes():raise AssertionError('H1 observation changed q/halo')
        if not np.array_equal(logged_steps(current),logged_steps(quiet)):raise AssertionError('H1 observation changed clock')
        tail=run('tail','gpu',binary,topology,restore=current/'outdat/new/checkpoints/step000000000002',instrument=args.memcheck)
        report['exact_state_datasets']=exact_state(tail/FINAL/'state.h5',current/FINAL/'state.h5')
        for name in ('control.bin','archives.bin','insitu_control.bin'):
            if (tail/FINAL/name).read_bytes()!=(current/FINAL/name).read_bytes():raise AssertionError('H1 exact controls '+name)
        if not np.array_equal(logged_steps(tail,2,3),logged_steps(current)[2:]):raise AssertionError('H1 restored step clock')
        full=np.loadtxt(current/'flowstate.dat',skiprows=1,ndmin=2);resumed=np.loadtxt(tail/'flowstate.dat',skiprows=1,ndmin=2)
        if not np.array_equal(resumed,full[np.isin(full[:,0],resumed[:,0])]):raise AssertionError('H1 restored scalar values')
        report.update(output_q_halo_clock_isolation=True,exact_controls_scalars_clock=True,memcheck=args.memcheck)
        if args.nonexact_source:
            own=run('nonexact-own','gpu',binary,topology,
                restore=current/'outdat/new/checkpoints/step000000000002',exact=False)
            recovered=run('nonexact','gpu',binary,topology,restore=args.nonexact_source.resolve(),exact=False)
            if not np.array_equal(logged_steps(recovered,2,3),logged_steps(own,2,3)):
                raise AssertionError('cross-backend nonexact H1 clocks differ')
            report['nonexact_terminal_scaled']=compare(checkpoint(recovered/FINAL/'state.h5'),checkpoint(own/FINAL/'state.h5'))
            report['nonexact_reference']='same-build nonexact restart with the same current timestep configuration'
            report['continuous_end_time']=float(logged_steps(current)[-1,2])
            report['nonexact_end_time']=float(logged_steps(recovered,2,3)[-1,2])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
