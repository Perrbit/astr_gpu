"""Native weak-wall SBP6/3 KEEP: frozen RHS, global wall rows and exact restart."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import h5py
import numpy as np

from compare_q_validation_snapshots import read_q_snapshot
from run_keep_periodic_gate import compare_q
from run_keep_wall_gate import wall_rhs
from run_output_restart_validation import compare_fields
from run_keep_sbp63_component_gate import scaled
from study_keep_sbp63_family import derive
from study_keep_sbp63_sat_wall import matrices,error_minimizer
from analyze_keep_rk_budgets import RK,SOURCE_WEIGHTS,DEFECT_WEIGHTS,NAMES
from run_keep_sbp63_wall_convergence import symbolic_steady_mms_sources
import sympy as sp

ROOT=Path(__file__).resolve().parents[2]
FINAL=Path('outdat/new/checkpoints/step000000000004')


def derivative(matrix,field,axis):
    return np.moveaxis(np.tensordot(matrix,field,axes=(1,axis)),0,axis)


def complete_step_budget(case,n,steady_mms=False):
    """Algebraic accounting of archived native RK stages, not a second solver."""
    def state(label,stage):
        snap=read_q_snapshot(case/'outdat'/f'rhs.{label}.step00000000.rk{stage:02d}.rank00000000.bin')
        im,jm,km,hm,nq=snap.header
        return snap.values.reshape((im+2*hm+1,jm+2*hm+1,km+2*hm+1,nq),order='F')[hm:hm+n,hm:hm+n+1,hm:hm+n]
    def rhs(label,stage):
        with (case/'outdat'/f'rhs.{label}.step00000000.rk{stage:02d}.rank00000000.bin').open('rb') as stream:
            np.fromfile(stream,np.int32,4)
            return np.fromfile(stream,np.float64).reshape((n+1,n+1,n+1,5),order='F')[:-1,:,:-1]
    with h5py.File(case/'outdat/new/resources/geometry.h5') as geometry:
        jac=geometry['q0004'][:].transpose(2,1,0)[:-1,:,:-1]
    norms,*_=derive();hy=np.ones(n+1)
    for i,h in enumerate(norms):
        hy[i]=hy[n-i]=float(h)
    weights=jac*hy[None,:,None];volume=float(weights.sum())
    mean=lambda values:np.sum(weights[...,None]*values,axis=(0,1,2))/volume
    def quantities(q):
        rho=q[...,0];v=q[...,1:4]/rho[...,None];kinetic=.5*np.sum(q[...,1:4]*v,axis=-1)
        internal=q[...,4]-kinetic
        if not np.isfinite(q).all() or not (rho>0).all() or not (internal>0).all():
            raise AssertionError('Invalid RK budget state')
        entropy=(np.log(.4*internal)-1.4*np.log(rho))/.4
        return mean(np.concatenate([q,kinetic[...,None],internal[...,None],(rho*entropy)[...,None]],axis=-1))
    def rates(q,mapped):
        rho=q[...,0];v=q[...,1:4]/rho[...,None];kinetic=.5*np.sum(q[...,1:4]*v,axis=-1)
        internal=q[...,4]-kinetic;entropy=(np.log(.4*internal)-1.4*np.log(rho))/.4
        f=mapped/jac[...,None]
        krate=np.sum(v*f[...,1:4],axis=-1)-.5*np.sum(v*v,axis=-1)*f[...,0]
        irate=f[...,4]-krate;srate=(entropy-1.4/.4)*f[...,0]+rho*irate/(.4*internal)
        return mean(np.concatenate([f,krate[...,None],irate[...,None],srate[...,None]],axis=-1))
    pre=[state('pre_rhs',s) for s in (1,2,3)];post=[state('post_update',s) for s in (1,2,3)]
    initial=state('initialized',1);base=pre[0];dt=1e-5
    conv=[];visc=[];sources=[];defects=[];stages=[];preparation=[]
    for index,(a,b,c) in enumerate(RK):
        fc=rhs('conv',index+1);ff=rhs('full',index+1)
        previous=initial if index==0 else post[index-1]
        recurrence=scaled(post[index],(a*base*jac[...,None]+b*pre[index]*jac[...,None]+c*dt*ff)/jac[...,None])
        prep=scaled(pre[index],previous)
        if max(recurrence,prep)>2e-10:
            raise AssertionError('Native weak-wall RK recurrence or stage preparation differs')
        before=rhs('sbp63_pre_source',index+1) if steady_mms else ff
        rc=rates(pre[index],fc);rv=rates(pre[index],before-fc);rs=rates(pre[index],ff-before)
        defect=quantities(post[index])-a*quantities(base)-b*quantities(pre[index])-c*dt*(rc+rv+rs)
        conv.append(rc);visc.append(rv);sources.append(rs);defects.append(defect)
        preparation.append(quantities(pre[index])-quantities(previous))
        stages.append(dict(stage=index+1,recurrence_max_scaled=recurrence,preparation_max_scaled=prep,
            rk_nonlinear_defect=defect.tolist()))
    terms=dict(convection=dt*SOURCE_WEIGHTS@np.array(conv),viscosity=dt*SOURCE_WEIGHTS@np.array(visc),
        manufactured_source=dt*SOURCE_WEIGHTS@np.array(sources),
        rk_nonlinear_defect=DEFECT_WEIGHTS@np.array(defects),preparation=np.array([1.,1/6,2/3])@np.array(preparation))
    change=quantities(post[-1])-quantities(initial);residual=change-sum(terms.values())
    if max(abs(residual))>2e-10 or max(abs(np.array(defects)[:,:5]).ravel())>2e-10:
        raise AssertionError('Native complete-step weak-wall budget does not close')
    return dict(quantities=NAMES,volume=volume,rule='SBP y norm times stored J; unique periodic x/z nodes',
        change=change.tolist(),terms={key:value.tolist() for key,value in terms.items()},
        residual=residual.tolist(),stages=stages,
        interpretation='weak wall with separately measured manufactured source when selected; numerical heat and nonlinear RK remainder retained')


def frozen_reference(case,n,stage=1,steady_mms=False):
    """Independent tensor matrices; never advances a CFD time step."""
    norms,left,centered,theta=derive();chosen,_=error_minimizer(norms,left,theta)
    h,dy,*_=matrices(n,norms,left,centered)
    # matrices() already scales to a unit physical interval, not integer spacing.
    hy=np.diag(np.array(h,dtype=float))*np.pi
    dy=np.array(dy.subs(theta,chosen),dtype=float)/np.pi
    dx=np.zeros((n,n));spacing=2*np.pi/n
    for row in range(n):
        for offset,c in ((1,.75),(2,-.15),(3,1/60)):
            dx[row,(row+offset)%n]=c/spacing;dx[row,(row-offset)%n]=-c/spacing
    operators=(dx,dy,dx)
    suffix=f'.step00000000.rk{stage:02d}.rank00000000.bin'
    snapshot=read_q_snapshot(case/'outdat'/('rhs.pre_rhs'+suffix))
    im,jm,km,hm,nq=snapshot.header
    shape=(im+2*hm+1,jm+2*hm+1,km+2*hm+1)
    with (case/'outdat'/('rhs.pre_rhs_primitives'+suffix)).open('rb') as stream:
        np.fromfile(stream,np.int32,4)
        primitive=np.fromfile(stream,np.float64).reshape((*shape,6),order='F')[hm:hm+n,hm:hm+n+1,hm:hm+n]
    rho=primitive[...,0];v=primitive[...,1:4];p=primitive[...,4];t=primitive[...,5]
    q=snapshot.values.reshape((*shape,nq),order='F')[hm:hm+n,hm:hm+n+1,hm:hm+n]
    gu=np.stack([derivative(d,v,a) for a,d in enumerate(operators)],axis=-1)
    gt=np.stack([derivative(d,t,a) for a,d in enumerate(operators)],axis=-1)
    for row,s in ((0,-1),(n,1)):
        gu[:,row,:,:,1]-=s*v[:,row,:,:]/hy[row]
        gt[:,row,:,1]+=s*(1-t[:,row,:])/hy[row]
    mu=t*np.sqrt(t)*(1+110.4/273.15)/(t+110.4/273.15)/3000
    kappa=(mu/.72)/(.4*.3**2)
    stress=mu[...,None,None]*(gu+np.swapaxes(gu,-1,-2)-(2/3)*np.trace(gu,axis1=-2,axis2=-1)[...,None,None]*np.eye(3))
    heat=kappa[...,None]*gt+np.einsum('...c,...cd->...d',v,stress)
    conv=np.zeros_like(q);visc=np.zeros_like(q)
    for axis,d in enumerate(operators):
        for row in range(d.shape[0]):
            sl=[slice(None)]*3;sl[axis]=row;sl=tuple(sl)
            rl=rho[sl];vl=v[sl];pl=p[sl]
            for column in np.flatnonzero(d[row]):
                sr=list(sl);sr[axis]=column;sr=tuple(sr)
                rr=rho[sr];vr=v[sr];pr=p[sr];average=.5*(vl+vr)
                mass=.5*(rl+rr)*average[...,axis]
                f=np.zeros_like(q[sl]);f[...,0]=mass;f[...,1:4]=mass[...,None]*average
                f[...,axis+1]+=.5*(pl+pr)
                f[...,4]=mass*(.5*np.sum(vl*vr,axis=-1)+.5*(pl/rl+pr/rr)/.4)+.5*(vl[...,axis]*pr+vr[...,axis]*pl)
                conv[sl]-=2*d[row,column]*f
        visc[...,1:4]+=derivative(d,stress[...,axis],axis)
        visc[...,4]+=derivative(d,heat[...,axis],axis)
    faces={}
    for row,s in ((0,-1),(n,1)):
        normal=np.array([0,s,0]);f=np.zeros_like(q[:,row,:]);f[...,0]=rho[:,row,:]*v[:,row,:,1]
        f[...,1:4]=f[...,0,None]*v[:,row,:,:];f[...,2]+=p[:,row,:]
        f[...,4]=(q[:,row,:,4]+p[:,row,:])*v[:,row,:,1]
        wall=np.zeros_like(f);wall[...,2]=p[:,row,:]
        conv[:,row,:,:]-=s*(wall-f)/hy[row]
        traction=stress[:,row,:,:,:]@normal-mu[:,row,:,None]/hy[row]*(v[:,row,:,:]+normal*v[:,row,:,1,None]*s/3)
        wallheat=kappa[:,row,:]*gt[:,row,:,1]*s-kappa[:,row,:]/hy[row]*(t[:,row,:]-1)
        visc[:,row,:,1:4]+=(traction-stress[:,row,:,:,:]@normal)/hy[row]
        visc[:,row,:,4]+=(wallheat-heat[:,row,:,1]*s)/hy[row]
        faces[str(row)]=dict(heat=wallheat,traction=traction)
    with h5py.File(case/'outdat/new/resources/geometry.h5') as geometry:
        jac=geometry['q0004'][:].transpose(2,1,0)[:-1,:,:-1]
        xyz=[geometry[f'q{c:04d}'][:].transpose(2,1,0)[:-1,:,:-1] for c in range(1,4)]
    source=np.zeros_like(q)
    if steady_mms:
        coordinates,_,direct=symbolic_steady_mms_sources()
        for c,expression in enumerate(direct):
            source[...,c]=sp.lambdify(coordinates,expression,'numpy')(*xyz)
    errors={}
    native={}
    for phase in ('conv','full'):
        with (case/'outdat'/('rhs.'+phase+suffix)).open('rb') as stream:
            header=np.fromfile(stream,np.int32,4)
            native[phase]=np.fromfile(stream,np.float64).reshape((n+1,n+1,n+1,5),order='F')[:-1,:,:-1]/jac[...,None]
        errors[phase]=scaled(native[phase],conv if phase=='conv' else conv+visc+source)
    if steady_mms:
        with (case/'outdat'/('rhs.sbp63_pre_source'+suffix)).open('rb') as stream:
            np.fromfile(stream,np.int32,4)
            before=np.fromfile(stream,np.float64).reshape((n+1,n+1,n+1,5),order='F')[:-1,:,:-1]/jac[...,None]
        errors['manufactured_source']=scaled(native['full']-before,source)
    with (case/'outdat'/('rhs.gradients'+suffix)).open('rb') as stream:
        np.fromfile(stream,np.int32,5)
        native_gu=np.fromfile(stream,np.float64,(n+1)**3*9).reshape((n+1,n+1,n+1,3,3),order='F')[:-1,:,:-1]
        native_gt=np.fromfile(stream,np.float64,(n+1)**3*3).reshape((n+1,n+1,n+1,3),order='F')[:-1,:,:-1]
    errors.update(gradient_u=scaled(native_gu,gu),gradient_T=scaled(native_gt,gt))
    weights=np.broadcast_to(spacing**2*hy[None,:,None],rho.shape)
    visc_native=native['full']-native['conv']-source
    kinetic_rate=np.sum(weights*(np.sum(v*visc_native[...,1:4],axis=-1)-.5*np.sum(v*v,axis=-1)*visc_native[...,0]))
    dissipation=-np.sum(weights*np.sum(stress*gu,axis=(-1,-2)))
    penalty=0.
    for row in (0,n):
        penalty-=spacing**2*np.sum(mu[:,row,:]/hy[row]*(np.sum(v[:,row,:,:]**2,axis=-1)+v[:,row,:,1]**2/3))
    energy_rate=np.sum(weights*visc_native[...,4]);wall_heat=spacing**2*sum(np.sum(face['heat']) for face in faces.values())
    balances=dict(convective_mass=float(np.sum(weights*native['conv'][...,0])),
                  convective_energy=float(np.sum(weights*native['conv'][...,4])),
                  viscous_kinetic=float(kinetic_rate),negative_stress=float(dissipation),wall_penalty=float(penalty),
                  viscous_work_residual=float(kinetic_rate-dissipation-penalty),
                  viscous_energy=float(energy_rate),numerical_wall_heat=float(wall_heat),
                  heat_residual=float(energy_rate-wall_heat))
    if max(errors.values())>2e-10 or max(abs(balances[key]) for key in
        ('convective_mass','convective_energy','viscous_work_residual','heat_residual'))>2e-10:
        raise AssertionError(f'Frozen tensor reference or SBP wall balances differ: {errors}: {balances}')
    return dict(errors=errors,balances=balances)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('cpu','gpu','mpiexec','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--backends',nargs='+',choices=['cpu','gpu'],default=['cpu','gpu'])
    parser.add_argument('--sizes',nargs='+',type=int,choices=[16,20],default=[16,20])
    parser.add_argument('--single-only',action='store_true')
    parser.add_argument('--memcheck',action='store_true')
    parser.add_argument('--steady-mms',action='store_true')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',dt=1e-5,steps=4,wall_mode='keep_sbp63',force=0,
        steady_mms=args.steady_mms,cases={},frozen={})
    for n in args.sizes:
        paths={}
        for ranks in ([1] if args.single_only else [1,2 if n==16 else 4]):
            for backend in args.backends:
                name=f'{backend}_n{n}_np{ranks}';case=out/name;datin=case/'datin'
                subprocess.run([sys.executable,str(ROOT/'examples/KEEP_SBP63_Walls/prepare.py'),
                    '--backend',backend,'--intervals',str(n),'--output',str(case)],check=True)
                env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
                env.update(OMP_NUM_THREADS='1',ASTR_FORCE_MPI_TOPOLOGY=f'1,{ranks},1',ASTR_GPU_SYNC_MODE='explicit',
                    ASTR_GPU_HALO_TRANSPORT='pinned',OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',
                    OMPI_MCA_osc='pt2pt',OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',
                    ASTR_VALIDATION_RK_STATE='1',ASTR_VALIDATION_PROFILE_STAGE_STATE='1')
                if ranks==1:
                    env.update(ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_GRADIENTS='1')
                if args.steady_mms:
                    env['ASTR_TEST_KEEP_SBP63_STEADY_MMS']='1'
                command=[str(args.mpiexec.absolute()),'--mca','coll_hcoll_enable','0','-np',str(ranks)]
                if args.memcheck and backend=='gpu':
                    command+=['compute-sanitizer','--tool','memcheck','--target-processes','all','--error-exitcode','99',
                        '--log-file',str(case/'memcheck.%p.log')]
                command+=[str(getattr(args,backend).resolve()),'run','datin/input.case']
                with (case/'run.log').open('w') as log:
                    subprocess.run(command,cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                checkpoint=case/FINAL
                if b'/s63sat-v1' not in (checkpoint/'control.bin').read_bytes():
                    raise AssertionError('New wall operator identity is absent')
                if (b'/mms-v1' in (checkpoint/'control.bin').read_bytes())!=args.steady_mms:
                    raise AssertionError('Manufactured source identity differs from its selected mode')
                with h5py.File(checkpoint/'state.h5') as state:
                    q=np.stack([state[f'q{c:04d}'][:].transpose(2,1,0) for c in range(1,6)],axis=-1)
                internal=q[...,4]-.5*np.sum(q[...,1:4]**2,axis=-1)/q[...,0]
                t=.126*.4*internal/q[...,0]
                if not np.isfinite(q).all() or not (q[...,0]>0).all() or not (internal>0).all():
                    raise AssertionError('Invalid native physical state')
                wall_q=q[:,[0,-1],:,:]
                entry=dict(path=str(case),rho_min=float(q[...,0].min()),pressure_min=float(.4*internal.min()),
                    temperature_min=float(t.min()),wall_velocity_max=float(np.max(abs(wall_q[...,1:4]/wall_q[...,0,None]))),
                    wall_temperature_error=float(np.max(abs(t[:,[0,-1],:]-1))))
                if args.memcheck and backend=='gpu':
                    counts=[int(x) for f in case.glob('memcheck.*.log') for x in re.findall(r'ERROR SUMMARY: (\d+) errors',f.read_text())]
                    if len(counts)<ranks or any(counts):
                        raise AssertionError('Native SBP63 memcheck errors or absent rank summaries')
                    entry['memcheck_errors']=counts
                if ranks==1:
                    report['frozen'][name]={stage:frozen_reference(case,n,stage,args.steady_mms) for stage in (1,2,3)}
                    entry['complete_step_budget']=complete_step_budget(case,n,args.steady_mms)
                else:
                    entry['vs_np1']=compare_q(paths[(backend,1)]/FINAL/'state.h5',checkpoint/'state.h5')
                resumed=out/(name+'_resumed');shutil.copytree(datin,resumed/'datin',symlinks=True)
                (resumed/'outdat/new').mkdir(parents=True)
                output=resumed/'datin/input.output'
                output.write_text(output.read_text().replace("directory='outdat/new',",f"directory='outdat/new',restore_directory='{case}/outdat/new/checkpoints/step000000000002',"))
                resume_env={k:v for k,v in env.items() if not k.startswith('ASTR_VALIDATION_')}
                resume_command=[str(args.mpiexec.absolute()),'--mca','coll_hcoll_enable','0','-np',str(ranks),str(getattr(args,backend).resolve()),'run','datin/input.case']
                with (resumed/'run.log').open('w') as log:
                    subprocess.run(resume_command,cwd=resumed,env=resume_env,stdout=log,stderr=subprocess.STDOUT,check=True)
                compare_fields(checkpoint/'state.h5',resumed/FINAL/'state.h5')
                for filename in ('control.bin','archives.bin','insitu_control.bin'):
                    if (checkpoint/filename).read_bytes()!=(resumed/FINAL/filename).read_bytes():
                        raise AssertionError('Exact restart differs: '+filename)
                entry['exact_2_plus_2']='all state datasets and three control files bit-identical'
                if args.steady_mms and backend=='cpu' and ranks==1:
                    changed=out/(name+'_source_off_exact');shutil.copytree(datin,changed/'datin')
                    (changed/'outdat/new').mkdir(parents=True)
                    output=changed/'datin/input.output'
                    output.write_text(output.read_text().replace("directory='outdat/new',",f"directory='outdat/new',restore_directory='{case}/outdat/new/checkpoints/step000000000002',"))
                    changed_env={key:value for key,value in resume_env.items() if key!='ASTR_TEST_KEEP_SBP63_STEADY_MMS'}
                    with (changed/'run.log').open('w') as log:
                        result=subprocess.run(resume_command,cwd=changed,env=changed_env,stdout=log,stderr=subprocess.STDOUT)
                    if result.returncode==0 or 'compact operator identity mismatch' not in (changed/'run.log').read_text():
                        raise AssertionError('Removing the manufactured source was accepted as an exact continuation')
                    if list((changed/'outdat/new').rglob('COMPLETE')):
                        raise AssertionError('Changed-source exact rejection published a checkpoint')
                    entry['source_off_exact']='rejected different MMS source identity before advancement'
                report['cases'][name]=entry;paths[(backend,ranks)]=case
                (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
                print(name,'frozen/reference/finite/exact checks done',flush=True)
            if all((b,ranks) in paths for b in ('cpu','gpu')):
                report['cases'][f'cpu_gpu_n{n}_np{ranks}']=compare_q(paths[('cpu',ranks)]/FINAL/'state.h5',paths[('gpu',ranks)]/FINAL/'state.h5')
                if ranks==1:
                    report['frozen'][f'cpu_gpu_n{n}']=wall_rhs(paths[('cpu',1)],paths[('gpu',1)])
    report['status']='passed_selected_native_sbp63_wall_scope'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
