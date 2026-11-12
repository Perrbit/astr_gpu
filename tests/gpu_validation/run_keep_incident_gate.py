"""Existing GRCBC boundary coupling with new numerical forms, not shock validation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np

from named_fixture import prepare_named_run,namelist
from test_grcbc_oblique_shock import prepare as prepare_oblique,analytic_states
from run_keep_open_gate import compare_open_rhs
from run_keep_simple_boundary_gate import frozen_rhs
from run_keep_periodic_gate import compare_q
from run_output_restart_validation import compare_fields
from compare_q_validation_snapshots import read_q_snapshot
from run_keep_y_farfield_gate import phase,incoming_projector

FINAL=Path('outdat/new/checkpoints/step000000000004')


def setup(case,backend,entry,restore=None,form='keep',scheme='643e',diffusion='643e',
          viscosity='off',original_scheme='743e'):
    prepare_oblique(case,backend,16,4,initial='uniform',entry_x=entry,
                    checkpoint_interval=2,restore=restore,z_intervals=16)
    env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',OMP_NUM_THREADS='1',
        ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_HALO_TRANSPORT='pinned',
        ASTR_NSCBC_FARFIELD_MODE='sbli_grcbc',ASTR_NSCBC_FARFIELD_SHOCK_X=str(entry),
        ASTR_NSCBC_FARFIELD_SHOCK_ANGLE_DEG='40',ASTR_NSCBC_FARFIELD_RHO='1',
        ASTR_NSCBC_FARFIELD_U='1',ASTR_NSCBC_FARFIELD_V='0',ASTR_NSCBC_FARFIELD_W='0',
        ASTR_NSCBC_FARFIELD_T='1')
    viscous=viscosity!='off'
    convection=scheme if form=='keep' else original_scheme
    if viscous:
        with h5py.File(case/'datin/grid.flatplate.h5') as grid:
            x,y,z=[grid[name][:] for name in ('x','y','z')]
        profile=1-np.exp(-(y/.15)**2);ramp=1-np.exp(-x)
        angle=2*np.pi*z/.125
        temperature=1+1e-4*ramp*profile**2*np.cos(angle)
        fields=dict(ro=1/temperature,t=temperature,u1=profile*(1+1e-4*ramp*np.cos(angle)),
                    u2=1e-4*ramp*profile**2*np.sin(angle),u3=1e-4*ramp*profile**2*np.cos(angle))
        with h5py.File(case/'datin/flowini3d.h5','w') as initial:
            for name,value in fields.items():
                initial[name]=value
        with (case/'datin/inlet.prof').open('w') as stream:
            stream.write('Viscous incident fixture density=provided\nrho u v temperature\n1 1 1 1\n'
                         'y rho u v temperature\n')
            for yy in y[0,:,0]:
                stream.write(f'1.0000000000000000e+00 {(1-np.exp(-(yy/.15)**2)):.16e} '
                             '0.0000000000000000e+00 1.0000000000000000e+00\n')
    prepare_named_run(case,'datin/input.flatplate',env,numerics=dict(
        convection_form=form,viscous_form='me6_opti' if viscosity=='me6_opti' else 'standard',
        conschm=convection,difschm=diffusion,lchardecomp=form=='standard' and convection=='743e',
        recon_schem=5 if convection=='743e' else 3),
        physics=dict(diffterm=True,reynolds=1000.) if viscous else None,
        boundary=dict(bctype=[11,22,41,52,1,1],twall=[0,0,1,0,0,0],pout=1.) if viscous else None)
    (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=4),
        timestep=dict(mode='fixed',fixed_dt=1e-5),logging=dict(list_steps=1),
        repair=dict(crashfix=False),statistics=dict(enabled=False))))
    (case/'datin/input.output').write_text(namelist(dict(
        output=dict(directory='outdat/new',restore_directory=str(restore or ''),restart_output='override' if restore else 'saved',
                    exact_restart=True,host_budget_bytes=-1,device_budget_bytes=-1,
                    device_reserve_bytes=0,buffer_bytes=67108864,checkpoint_statistics=False),
        checkpoint=dict(enabled=True,interval_steps=2,keep=2,initial_frame=False),
        volume=dict(enabled=False),slices=dict(enabled=False))))
    parameters=json.loads((case/'parameters.json').read_text())
    parameters.update(dt=1e-5,convection_form=form,conschm=convection,
                      difschm=diffusion,viscosity=viscosity,main_filter=False,crashfix=False,
                      nominal_bottom_crossing=entry+1/np.tan(np.deg2rad(40.)))
    if viscous:
        parameters.update(reynolds=1000.,lower_bctype=41,wall_temperature=1.,outlet_pressure_ratio=1.,
                          initial='weak 3D perturbed profile, thickness parameter 0.15')
    (case/'parameters.json').write_text(json.dumps(parameters,indent=2)+'\n')
    if restore is not None:
        for name in ('grid.flatplate.h5','flowini3d.h5','inlet.prof'):
            (case/'datin'/name).unlink()
    return env


def characteristic_closure(case,entry,viscous):
    snapshot=read_q_snapshot(case/'outdat/rhs.pre_rhs.step00000000.rk01.rank00000000.bin')
    im,jm,km,hm,nq=snapshot.header
    q=snapshot.values.reshape((im+2*hm+1,jm+2*hm+1,km+2*hm+1,nq),order='F')
    q=q[hm:hm+im+1,hm:hm+jm+1,hm:hm+km+1]
    before=phase(case,'incident_pre');after=phase(case,'full')
    viscous_source=before-phase(case,'conv')
    with h5py.File(case/'outdat/new/resources/geometry.h5') as geometry:
        coordinates=np.stack([geometry[f'q{c+1:04d}'][:].transpose(2,1,0) for c in range(3)],axis=-1)
        jacobian=geometry['q0004'][:].transpose(2,1,0)
        metric=np.stack([geometry[f'q{6+3*c:04d}'][:].transpose(2,1,0) for c in range(3)],axis=-1)
    pre,post=analytic_states()
    conservative=[]
    for primitive in (pre,post):
        rho=primitive[0];velocity=primitive[1:4]
        conservative.append(np.r_[rho,rho*velocity,rho*primitive[4]/5.6/.4+.5*rho*np.dot(velocity,velocity)])
    acoustic=0.;outgoing=0.;idempotency=0.;counts=set()
    for i in range(1,im+1):
        for k in range(km+1):
            state=q[i,jm,k];rho=state[0];velocity=state[1:4]/rho
            pressure=.4*(state[4]-.5*np.dot(state[1:4],velocity))
            sound=np.sqrt(1.4*pressure/rho)
            normal=metric[i,jm,k]/np.linalg.norm(metric[i,jm,k])
            hn=np.dot(coordinates[i,jm,k]-coordinates[i,jm-1,k],normal)
            ghost=coordinates[i,jm,k]+hn*normal
            external=conservative[int((ghost[0]-entry)*np.sin(np.deg2rad(40.))+
                                      (ghost[1]-1)*np.cos(np.deg2rad(40.))>0)]
            external_velocity=external[1:4]/external[0]
            external_pressure=.4*(external[4]-.5*np.dot(external[1:4],external_velocity))
            drho=rho-external[0];du=velocity-external_velocity
            increment=np.r_[drho,velocity*drho+rho*du,
                (pressure-external_pressure)/.4+.5*np.dot(velocity,velocity)*drho+rho*np.dot(velocity,du)]
            un=np.dot(velocity,normal)
            dp=np.r_[.2*np.dot(velocity,velocity),-.4*velocity,.4]
            dun=np.r_[-un,normal,0.]
            right=np.r_[1.,velocity-sound*normal,(state[4]+pressure)/rho-sound*un]
            minus=np.outer(right,(dp-sound*dun)/(2*sound**2))
            expected=-abs(un-sound)/hn*(minus@increment)
            actual=minus@after[i,jm,k]/jacobian[i,jm,k]
            acoustic=max(acoustic,float(np.max(abs(actual-expected))))
            incoming,count=incoming_projector(state,normal);counts.add(count)
            outgoing=max(outgoing,float(np.max(abs((np.eye(5)-incoming)@(after[i,jm,k]-before[i,jm,k])))))
            idempotency=max(idempotency,float(np.max(abs(minus@minus-minus))))
    source_max=float(np.max(abs(viscous_source)))
    if max(acoustic,outgoing,idempotency)>2e-10:
        raise AssertionError(f'GRCBC characteristic closure failed: {acoustic} {outgoing} {idempotency}')
    if viscous and source_max<=1e-14:
        raise AssertionError('Viscous incident fixture did not exercise a nonzero diffusion increment')
    return dict(incoming_acoustic_residual=acoustic,outgoing_change=outgoing,
                projector_residual=idempotency,incoming_wave_counts=sorted(counts),viscous_source_max=source_max)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cpu','gpu','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--entry-x',type=float,default=2.)
    parser.add_argument('--scheme',choices=['643e','643c'],default='643e')
    parser.add_argument('--diffusion',choices=['643e','643c'],default='643e')
    parser.add_argument('--convection',choices=['keep','standard'],default='keep')
    parser.add_argument('--viscosity',choices=['off','standard','me6_opti'],default='off')
    parser.add_argument('--axes',nargs='*',choices=['x','y','z'],default=['x','y','z'])
    parser.add_argument('--multi-axis',action='store_true')
    parser.add_argument('--memcheck',action='store_true')
    parser.add_argument('--memcheck-stack-bytes',type=int,default=0,
                        help='Explicit instrumented GRCBC stack setting; testing builds only')
    args=parser.parse_args()
    if args.convection=='standard' and args.viscosity!='me6_opti':
        parser.error('The original-convection gate requires ME6 viscosity')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    topologies=[(1,1,1)]
    for axis in args.axes:
        values=[1,1,1];values['xyz'.index(axis)]=2;topologies.append(tuple(values))
    if args.multi_axis:
        topologies.append((2,2,1))
    pre,post=analytic_states()
    report=dict(status='running',scope='constant state or incident-boundary response; no KEEP shock-capturing claim',
                boundaries=[11,22,421 if args.viscosity=='off' else 41,52,1,1],intervals=[16]*3,steps=4,dt=1e-5,
                reference_mach=2.,beta_deg=40.,entry_x=args.entry_x,pre=pre.tolist(),post=post.tolist(),
                scheme=args.scheme,auxiliary=args.diffusion,convection=args.convection,viscosity=args.viscosity,cases={})
    report['memcheck']=dict(enabled=args.memcheck,explicit_stack_bytes=args.memcheck_stack_bytes,
                           stack_scope='instrumented GPU NP=1 only; no production default change')
    singles={}
    for topology in topologies:
        label=''.join(map(str,topology));ranks=int(np.prod(topology));runs={}
        for backend in ('cpu','gpu'):
            case=out/(backend+'_'+label);env=setup(case,backend,args.entry_x,
                scheme=args.scheme,diffusion=args.diffusion,form=args.convection,
                viscosity=args.viscosity,original_scheme=args.scheme)
            env['ASTR_FORCE_MPI_TOPOLOGY']=','.join(map(str,topology))
            if ranks==1:
                env.update(ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0')
            exe=[str(getattr(args,backend).resolve())]
            if args.memcheck and backend=='gpu' and ranks==1:
                exe=[shutil.which('compute-sanitizer'),'--tool','memcheck','--error-exitcode','99',*exe]
                if args.memcheck_stack_bytes:
                    env['ASTR_VALIDATION_GRCBC_STACK_BYTES']=str(args.memcheck_stack_bytes)
            command=[str(args.mpiexec.absolute()),'--mca','coll_hcoll_enable','0','-np',str(ranks),*exe,'run','datin/input.case']
            with (case/'run.log').open('w') as log:
                subprocess.run(command,cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            if args.memcheck and backend=='gpu' and ranks==1 and 'ERROR SUMMARY: 0 errors' not in (case/'run.log').read_text():
                raise AssertionError('Missing successful incident GPU memcheck result')
            if args.memcheck and backend=='gpu' and ranks==1 and args.memcheck_stack_bytes:
                if f'requested={args.memcheck_stack_bytes} ' not in (case/'run.log').read_text():
                    raise AssertionError('Instrumented GRCBC stack request was not applied')
            if b'/inc1' not in (case/FINAL/'control.bin').read_bytes():
                raise AssertionError('KEEP incident numerical identity missing')
            if 'sbli_grcbc: exterior reference, complete-RHS closure' not in (case/'run.log').read_text():
                raise AssertionError('Incident complete-RHS reference was never activated')
            if args.scheme=='643c' or args.diffusion=='643c':
                if 'COMPACT GLOBAL: KEEP physical open ends; MPI interfaces remain globally coupled' not in (case/'run.log').read_text():
                    raise AssertionError('Incident global compact resource configuration missing')
            if ranks==1:
                singles[backend]=case
            else:
                report['cases'][backend+'_'+label+'_decomposition']=compare_q(singles[backend]/FINAL/'state.h5',case/FINAL/'state.h5')
                resumed=out/(backend+'_'+label+'_exact')
                resume_env=setup(resumed,backend,args.entry_x,case/'outdat/new/checkpoints/step000000000002',
                    scheme=args.scheme,diffusion=args.diffusion,form=args.convection,
                    viscosity=args.viscosity,original_scheme=args.scheme)
                resume_env['ASTR_FORCE_MPI_TOPOLOGY']=env['ASTR_FORCE_MPI_TOPOLOGY']
                with (resumed/'run.log').open('w') as log:
                    subprocess.run(command,cwd=resumed,env=resume_env,stdout=log,stderr=subprocess.STDOUT,check=True)
                compare_fields(case/FINAL/'state.h5',resumed/FINAL/'state.h5')
                for name in ('control.bin','archives.bin','insitu_control.bin'):
                    if (case/FINAL/name).read_bytes()!=(resumed/FINAL/name).read_bytes():
                        raise AssertionError(f'Incident exact continuation changed {name}')
                report['cases'][backend+'_'+label+'_exact']='all state/control values identical; frozen resources only'
            with h5py.File(case/FINAL/'state.h5') as state:
                top_density_change=float(np.max(abs(state['q0001'][:,-1]-1.)))
                if args.entry_x>1 and args.viscosity=='off':
                    expected=(1.,1.,0.,0.,1/(5.6*.4)+.5)
                    error=max(float(np.max(abs(state[f'q{c+1:04d}'][:]-value))) for c,value in enumerate(expected))
                    if error>2e-10:
                        raise AssertionError(f'Uniform incident reference changed state: {error}')
                    report.setdefault('uniform_error',{})[backend+'_'+label]=error
                elif top_density_change<=1e-12:
                    raise AssertionError('Incident reference did not generate a nonzero top response')
            report.setdefault('top_density_change',{})[backend+'_'+label]=top_density_change
            runs[backend]=case
        if ranks==1:
            report['rhs']=compare_open_rhs(runs['cpu'],runs['gpu'],out)
            report['stages']=frozen_rhs(runs['cpu'],runs['gpu'],[0,1],mach=2.)
            if args.viscosity!='off':
                report['characteristic_closure']={backend:characteristic_closure(case,args.entry_x,True)
                    for backend,case in runs.items()}
        report['cases']['cpu_gpu_'+label]=compare_q(runs['cpu']/FINAL/'state.h5',runs['gpu']/FINAL/'state.h5')
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('KEEP incident',topology,'compared',flush=True)
    report['status']='passed_four_step_incident_coupling'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
