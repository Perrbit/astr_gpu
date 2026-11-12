"""Run approved native weak-wall fixed-time windows and analyze final checkpoints."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import h5py
import numpy as np
import sympy as sp

from run_keep_periodic_gate import compare_q
from run_keep_wall_time_convergence import state_at
from study_keep_sbp63_family import derive
from study_keep_sbp63_sat_wall import matrices,error_minimizer

ROOT=Path(__file__).resolve().parents[2]
NAMES=['rho','rho_u','rho_v','rho_w','total_energy','u','v','w','p','T']


def sbp_matrices(n):
    norms,left,centered,theta=derive();chosen,_=error_minimizer(norms,left,theta)
    h,d,*_=matrices(n,norms,left,centered)
    # The exact reference is already scaled to a unit physical interval.
    return np.diag(np.array(h,dtype=float))*n,np.array(d.subs(theta,chosen),dtype=float)/n


def observables(q):
    rho=q[...,0];v=q[...,1:4]/rho[...,None]
    p=.4*(q[...,4]-.5*np.sum(q[...,1:4]*v,axis=-1));t=.126*p/rho
    if not np.isfinite(q).all() or not np.isfinite(t).all() or not (rho>0).all() or not (p>0).all() or not (t>0).all():
        raise AssertionError('Final weak-wall state is not finite and strictly admissible')
    return np.concatenate([q,v,p[...,None],t[...,None]],axis=-1)


def wall_diagnostics(case,data):
    q,xyz,jac,metadata=data;n=q.shape[0]-1
    hy,dy=sbp_matrices(n);values=observables(q);temperature=values[:-1,:,:-1,-1]
    gradients=[]
    for axis in (0,2):
        d=np.zeros_like(temperature)
        for radius,weight in ((1,.75),(2,-.15),(3,1/60)):
            d+=weight*(np.roll(temperature,-radius,axis=axis)-np.roll(temperature,radius,axis=axis))
        gradients.append(d)
    dy_t=np.einsum('ij,kjl->kil',dy,temperature)
    derivatives=(gradients[0],dy_t,gradients[1]);faces={}
    with h5py.File(case/'outdat/new/resources/geometry.h5') as geometry:
        metric=np.stack([geometry[f'q{c:04d}'][:].transpose(2,1,0) for c in range(5,14)],axis=-1)
    # The checkpoint packs a fastest inside b: Fortran dxi(a,b), not C row order.
    metric=metric.reshape((*q.shape[:3],3,3)).swapaxes(-1,-2)
    for label,row,sign in (('lower',0,-1.),('upper',n,1.)):
        g=metric[:-1,row,:-1,1,:];length_inv=np.sqrt(np.sum(g*g,axis=-1))
        normal=sign*g/length_inv[...,None];area=jac[:-1,row,:-1]*length_inv
        if not np.isfinite(area).all() or not (area>0).all():
            raise AssertionError('Invalid numerical wall measure')
        inverse_ell=length_inv/hy[row]
        gradient=sum(derivatives[a][:,row,:,None]*metric[:-1,row,:-1,a,:] for a in range(3))
        t=temperature[:,row,:];delta=t-1
        lifted=gradient-normal*delta[...,None]*inverse_ell[...,None]
        mu=t*np.sqrt(t)*(1+110.4/273.15)/(t+110.4/273.15)/3000
        kappa=mu/.72/.036
        raw=kappa*np.sum(gradient*normal,axis=-1)
        numerical=kappa*np.sum(lifted*normal,axis=-1)-kappa*inverse_ell*delta
        weighted_mean=lambda field:float(np.sum(area*field)/np.sum(area))
        speed=np.linalg.norm(values[:-1,row,:-1,5:8],axis=-1)
        faces[label]=dict(area=float(area.sum()),velocity_norm_max=float(speed.max()),
            velocity_norm_rms=float(np.sqrt(weighted_mean(speed**2))),temperature_error_max=float(abs(delta).max()),
            temperature_error_L1=weighted_mean(abs(delta)),unlifted_energy_flux_mean=weighted_mean(raw),
            numerical_energy_flux_mean=weighted_mean(numerical),numerical_energy_flux_min=float(numerical.min()),
            numerical_energy_flux_max=float(numerical.max()),numerical_energy_flux_integral=float(np.sum(area*numerical)),
            outward_heat_mean=-weighted_mean(numerical))
    return dict(faces=faces,
        interpretation='checkpoint-derived original SBP gradient and approved SAT, using stored metrics; positive viscous energy flux contributes to fluid energy, outward heat has the opposite sign',
        provenance='not a new native output field or an independent flow integrator'),faces


def norm_difference(coarse,fine):
    q,xyz,jac,_=coarse;r,xyz_f,_,_=fine
    n=q.shape[0]-1;ratio=(r.shape[0]-1)//n
    r=r[::ratio,::ratio,::ratio];xyz_f=xyz_f[::ratio,::ratio,::ratio]
    if q.shape!=r.shape or not np.isfinite(xyz).all() or np.max(abs(xyz-xyz_f))>2e-10:
        raise AssertionError('Mesh/time comparison does not use matching physical nodes')
    a=observables(q);b=observables(r);error=abs(a-b)
    hy,_=sbp_matrices(n);weights=jac[:-1,:,:-1]*hy[None,:,None];measure=float(weights.sum())
    result=dict(coordinate_max_abs=float(np.max(abs(xyz-xyz_f))),measure=measure)
    for label,delta in (('absolute',error),('scaled',error/np.maximum(1,np.maximum(abs(a),abs(b))))):
        physical=delta[:-1,:,:-1]
        result[label]=dict(L1=(np.sum(weights[...,None]*physical,axis=(0,1,2))/measure).tolist(),
            L2=np.sqrt(np.sum(weights[...,None]*physical**2,axis=(0,1,2))/measure).tolist(),
            Linf=np.max(delta,axis=(0,1,2)).tolist())
    return result


def exact_steady_reference(data):
    q,xyz,jac,metadata=data;x=xyz[...,0];y=xyz[...,1];z=xyz[...,2]
    velocity=np.stack([.05*np.sin(y)*(1+.1*np.cos(x)*np.cos(z)),
        .002*np.sin(x)*np.sin(y)**2,.001*np.cos(z)*np.sin(y)**2],axis=-1)
    velocity[:,0,:,:]=0;velocity[:,-1,:,:]=0
    gamma=1.4;pressure=1/(gamma*.3**2)
    reference=np.empty_like(q);reference[...,0]=1;reference[...,1:4]=velocity
    reference[...,4]=pressure/(gamma-1)+.5*np.sum(velocity**2,axis=-1)
    return reference,xyz,jac,metadata


def localize_mesh_difference(coarse,fine):
    q,xyz,jac,_=coarse;r=fine[0]
    n=q.shape[0]-1;ratio=(r.shape[0]-1)//n
    error=abs(observables(q)-observables(r[::ratio,::ratio,::ratio]))
    y=xyz[...,1];indices=np.arange(n+1)
    band=(indices<6)|(n-indices<6)
    core=(y>=3*np.pi/8)&(y<=5*np.pi/8)
    result=dict(boundary_band='first/last six coarse-grid nodes',core='3*pi/8 <= y <= 5*pi/8',fields={})
    for c,name in enumerate(NAMES):
        peak=np.unravel_index(np.argmax(error[...,c]),error[...,c].shape)
        result['fields'][name]=dict(maximum=float(error[(*peak,c)]),index=list(map(int,peak)),
            position=xyz[peak].tolist(),boundary_band_Linf=float(error[:,band,:,c].max()),
            fixed_core_Linf=float(error[...,c][core].max()))
    return result


def symbolic_initial_velocity():
    x,y,z=sp.symbols('x y z',real=True);r=sp.Rational
    velocity=sp.Matrix([r(1,20)*sp.sin(y)*(1+r(1,10)*sp.cos(x)*sp.cos(z)),
        r(1,500)*sp.sin(x)*sp.sin(y)**2,r(1,1000)*sp.cos(z)*sp.sin(y)**2])
    return (x,y,z),velocity


def initial_wall_compatibility():
    """Differentiate the specified initial field analytically; no flow advancement."""
    (x,y,z),velocity=symbolic_initial_velocity();r=sp.Rational
    coordinates=(x,y,z);gradient=velocity.jacobian(coordinates);divergence=sp.trace(gradient)
    mu=r(1,3000);gamma=r(7,5);mach=r(3,10);pressure=1/(gamma*mach**2)
    stress=mu*(gradient+gradient.T-r(2,3)*divergence*sp.eye(3))
    acceleration=-gradient*velocity+sp.Matrix([sum(sp.diff(stress[i,j],coordinates[j]) for j in range(3)) for i in range(3)])
    heating=gamma*(gamma-1)*mach**2*sum(stress[i,j]*gradient[i,j] for i in range(3) for j in range(3))
    temperature_rate=-(gamma-1)*divergence+heating
    result=dict(scope='analytic initial rho=T=1, constant pressure and Sutherland mu(1)/Re; fixed wall target time derivatives are zero',faces={})
    for label,wall in (('lower',0),('upper',sp.pi)):
        momentum=[sp.simplify(value.subs(y,wall)) for value in acceleration]
        thermal=sp.simplify(temperature_rate.subs(y,wall))
        acoustic=sp.simplify((gamma*pressure*sp.diff(divergence,y)).subs(y,wall))
        if momentum==[0,0,0] or thermal==0:
            raise AssertionError('The initial compatibility diagnostic unexpectedly became homogeneous')
        result['faces'][label]=dict(initial_velocity_time_derivative=list(map(str,momentum)),
            initial_temperature_time_derivative=str(thermal),
            linear_pressure_gradient_contribution_to_v_tt=str(acoustic))
    result['interpretation']='the initial values match the wall targets, but their first PDE time derivatives do not; this establishes a compatibility defect, not by itself the cause of every observed mesh difference'
    return result


def symbolic_steady_mms_sources():
    """Compact source definition and independent conservative-flux derivative."""
    coordinates,velocity=symbolic_initial_velocity();x,y,z=coordinates;r=sp.Rational
    gradient=velocity.jacobian(coordinates);divergence=sp.trace(gradient)
    mu=r(1,3000);gamma=r(7,5);mach=r(3,10);pressure=1/(gamma*mach**2)
    stress=mu*(gradient+gradient.T-r(2,3)*divergence*sp.eye(3))
    laplacian=sp.Matrix([sum(sp.diff(value,c,2) for c in coordinates) for value in velocity])
    viscous=mu*(laplacian+sp.Matrix([sp.diff(divergence,c) for c in coordinates])/3)
    energy=pressure/(gamma-1)+velocity.dot(velocity)/2
    dissipation=sum(stress[i,j]*gradient[i,j] for i in range(3) for j in range(3))
    momentum=gradient*velocity+velocity*divergence-viscous
    source=sp.Matrix([divergence,*momentum,
        (energy+pressure)*divergence+velocity.dot(gradient*velocity)-dissipation-velocity.dot(viscous)])
    direct=sp.zeros(5,1)
    for axis,c in enumerate(coordinates):
        convection=sp.Matrix([velocity[axis],*(velocity*velocity[axis]+pressure*sp.eye(3)[:,axis]),
            (energy+pressure)*velocity[axis]])
        diffusion=sp.Matrix([0,*stress[:,axis],velocity.dot(stress[:,axis])])
        direct+=(convection-diffusion).diff(c)
    return coordinates,source,direct


def steady_mms_source_candidate():
    """Analytic source definition; this result alone is not native MMS acceptance."""
    coordinates,source,direct=symbolic_steady_mms_sources();x,y,z=coordinates
    residual=[sp.simplify(value) for value in direct-source]
    if residual!=[0]*5:
        raise AssertionError('Proposed steady source differs from direct conservative flux divergence')
    return dict(status='analytic_source_identity_checked',
        exact_state='same prescribed velocity; rho=T=1, constant p; steady in time',
        source_definition='div(F_inviscid - F_viscous) evaluated on the exact steady reference, not on the evolving solution',
        five_component_conservative_residuals=list(map(str,residual)),
        wall_sources={label:[str(sp.simplify(value.subs(y,wall))) for value in source]
                      for label,wall in (('lower',0),('upper',sp.pi))},
        meaning='five residuals concern the analytic definition only; native implementation and solution errors require their separate evidence')


def composed_viscous_consistency():
    """Exact local moments of D_SB P squared on a constant-material reference."""
    norms,left,centered,theta=derive();chosen,_=error_minimizer(norms,left,theta)
    _,d,*_=matrices(32,norms,left,centered)
    d=d.subs(theta,chosen)/32;second=d*d
    boundary={}
    for i in range(9):
        moments=[sum(second[i,j]*(j-i)**k for j in range(33)) for k in range(5)]
        if moments[:4]!=[0,0,2,0]:
            raise AssertionError('Composed derivative failed a degree <=3 consistency moment')
        boundary[i]=str(sp.factor(moments[4]/sp.factorial(4)))
    if not any(sp.sympify(value)!=0 for value in boundary.values()):
        raise AssertionError('Composed operator unexpectedly lost its quadratic boundary defect')
    interior=[sum(second[16,j]*(j-16)**k for j in range(33)) for k in range(9)]
    if interior[:8]!=[0,0,2,0,0,0,0,0]:
        raise AssertionError('Interior composed operator is not sixth-order')
    return dict(scope='exact reference trace: primitive lift and viscous SAT jumps vanish; constant material; first derivative composed twice',
        representation='integer-coordinate D=D_SBP63; physical second derivative D^2/h^2',
        boundary_moments_0_to_3=['0','0','2','0'],
        h_squared_fourth_derivative_coefficients_left=boundary,
        interior_moments_0_to_7=list(map(str,interior[:8])),
        interior_h_sixth_eighth_derivative_coefficient=str(sp.factor(interior[8]/sp.factorial(8))),
        interpretation='the first derivative is 6/3; its gradient-divergence composition has a second-order near-wall term and a six-order unaffected interior; this is not a new D2 discretization',
        communication='solution halos precede local gradients; viscous-flux halos precede divergence; composition broadens the effective closure footprint without increasing the raw-field halo')


def analyze_existing(directory,steady_mms=False):
    directory=directory.resolve();source=directory/'report.json'
    report=json.loads(source.read_text())
    if report['status']!='measured_approved_native_windows':
        raise AssertionError('The native fixed-time matrix is not complete')
    for name,entry in report['runs'].items():
        data=state_at(Path(entry['path']),entry['step'])
        control=Path(entry['path'])/f"outdat/new/checkpoints/step{entry['step']:012d}/control.bin"
        if b'/s63sat-v1' not in control.read_bytes():
            raise AssertionError('Final checkpoint does not identify the weak-wall operator')
        if (b'/mms-v1' in control.read_bytes())!=steady_mms:
            raise AssertionError('Selected analysis does not match the checkpoint source identity')
        observables(data[0])
        entry['walls'],_=wall_diagnostics(Path(entry['path']),data)
        if steady_mms:
            reference=exact_steady_reference(data)
            entry['exact_error']=norm_difference(data,reference)
            entry['exact_error_localization']=localize_mesh_difference(data,reference)
    states={n:state_at(directory/f'gpu_n{n}_steps1000',1000) for n in (16,32,64)}
    report['mesh_localization']={str(n):localize_mesh_difference(states[n],states[2*n]) for n in (16,32)}
    report['unforced_initial_wall_compatibility']=initial_wall_compatibility()
    report['steady_mms_source_definition']=steady_mms_source_candidate()
    report['composed_viscous_consistency']=composed_viscous_consistency()
    if steady_mms:
        report['interpretation']='known steady manufactured-solution errors and same-time mesh differences; the MMS source cancels the separately reported unforced compatibility defect; no prescribed global order'
    report['analysis_revision']='Fortran dxi(a,b) pack order used explicitly; existing native trajectories reused'
    report['source_report']=str(source)
    report['cost_scope']='recorded elapsed wall time includes native launch, solve, file output, final checkpoint reads and test diagnostics; not RK-only or a performance result'
    target=directory/'report_analysis_v2.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(target,flush=True)
    for name,entry in report['runs'].items():
        faces=entry['walls']['faces']
        print(name,'wall max |u|',max(f['velocity_norm_max'] for f in faces.values()),
            'max |T-1|',max(f['temperature_error_max'] for f in faces.values()),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('cpu','gpu','mpiexec','output'):
        parser.add_argument('--'+key,type=Path,required=key=='output')
    parser.add_argument('--analyze-existing',action='store_true')
    parser.add_argument('--steady-mms',action='store_true')
    args=parser.parse_args()
    if args.analyze_existing:
        analyze_existing(args.output,args.steady_mms)
        return
    if any(getattr(args,key) is None for key in ('cpu','gpu','mpiexec')):
        parser.error('Native runs require --cpu, --gpu and --mpiexec')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',scope='approved section 14.10; uniform weak isothermal walls',
        physical_time=.01,fields=NAMES,ranks=2,topology=[1,2,1],filter=False,crashfix=False,force=0,
        quadrature='stored J times SBP y norm; unique periodic x/z nodes',steady_mms=args.steady_mms,
        runs={},mesh_differences={})
    if args.steady_mms:
        report['scope']='approved section 14.11; exact steady manufactured solution'
    env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(OMP_NUM_THREADS='1',ASTR_FORCE_MPI_TOPOLOGY='1,2,1',ASTR_GPU_SYNC_MODE='explicit',
        ASTR_GPU_HALO_TRANSPORT='pinned',OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',
        OMPI_MCA_osc='pt2pt',OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',
        ASTR_VALIDATION_RK_STATE='1',ASTR_VALIDATION_PROFILE_STAGE_STATE='1')
    if args.steady_mms:
        env['ASTR_TEST_KEEP_SBP63_STEADY_MMS']='1'
    def save():
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def run(backend,n,dt):
        steps=round(.01/dt);name=f'{backend}_n{n}_steps{steps}';case=out/name
        subprocess.run([sys.executable,str(ROOT/'examples/KEEP_SBP63_Walls/prepare.py'),
            '--backend',backend,'--intervals',str(n),'--output',str(case)],check=True)
        (case/'datin/input.run').write_text(f'''&run_control max_steps={steps} /
&timestep mode='fixed',fixed_dt={dt:.17e} /
&logging list_steps=1 /
&repair crashfix=f /
&statistics enabled=f /
''')
        path=case/'datin/input.output';path.write_text(path.read_text().replace('interval_steps=2,keep=2',f'interval_steps={steps},keep=1'))
        command=[str(args.mpiexec.absolute()),'--mca','coll_hcoll_enable','0','-np','2',
            str(getattr(args,backend).resolve()),'run','datin/input.case']
        report['active_case']=name;save();started=perf_counter()
        with (case/'run.log').open('w') as log:
            subprocess.run(command,cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        data=state_at(case,steps)
        if b'/s63sat-v1' not in (case/f'outdat/new/checkpoints/step{steps:012d}/control.bin').read_bytes():
            raise AssertionError('Native fixed-time run did not select the weak-wall operator')
        if (b'/mms-v1' in (case/f'outdat/new/checkpoints/step{steps:012d}/control.bin').read_bytes())!=args.steady_mms:
            raise AssertionError('Native fixed-time source identity differs from the requested mode')
        if data[0].shape!=(n+1,n+1,n+1,5) or data[3]['dt_used']!=dt or data[3]['dt_next']!=dt:
            raise AssertionError('Native weak-wall window changed the declared grid or fixed dt')
        values=observables(data[0]);walls,_=wall_diagnostics(case,data)
        cfl=[float(line.split('=')[1].split()[-1]) for line in (case/'run.log').read_text().splitlines()
            if line.startswith('ASTR_CFL directional/local_sum/upper_bound=')]
        if len(cfl)!=steps or not np.isfinite(cfl).all():
            raise AssertionError('Missing finite complete-step CFL measurements')
        report['runs'][name]=dict(path=str(case),command=command,elapsed_run_and_analysis_seconds=perf_counter()-started,
            c_upper_max=max(cfl),pressure_min=float(values[...,-2].min()),temperature_min=float(values[...,-1].min()),
            walls=walls,**data[3]);save()
        if args.steady_mms:
            reference=exact_steady_reference(data)
            report['runs'][name]['exact_error']=norm_difference(data,reference)
            report['runs'][name]['exact_error_localization']=localize_mesh_difference(data,reference)
            save()
        print(name,'native final state and wall diagnostics recorded; C_upper',max(cfl),flush=True)
        return case,data
    cpu,cpu_data=run('cpu',16,1e-5)
    gpu,data16=run('gpu',16,1e-5)
    final=Path('outdat/new/checkpoints/step000000001000/state.h5')
    report['cpu_gpu_n16']=compare_q(cpu/final,gpu/final);save()
    paths={16:gpu};data={16:data16}
    for n in (32,64):
        paths[n],data[n]=run('gpu',n,1e-5)
    for n in (16,32):
        report['mesh_differences'][str(n)]=norm_difference(data[n],data[2*n])
    report['apparent_orders']={}
    for norm in ('L1','L2','Linf'):
        a=report['mesh_differences']['16']['absolute'][norm];b=report['mesh_differences']['32']['absolute'][norm]
        report['apparent_orders'][norm]=[float(np.log2(x/y)) if x>0 and y>0 else None for x,y in zip(a,b)]
    if args.steady_mms:
        report['exact_error_orders']={}
        for n in (16,32):
            report['exact_error_orders'][str(n)]={}
            for norm in ('L1','L2','Linf'):
                a=report['runs'][f'gpu_n{n}_steps1000']['exact_error']['absolute'][norm]
                b=report['runs'][f'gpu_n{2*n}_steps1000']['exact_error']['absolute'][norm]
                report['exact_error_orders'][str(n)][norm]=[float(np.log2(x/y)) if x>0 and y>0 else None for x,y in zip(a,b)]
    _,half=run('gpu',32,5e-6)
    report['time_step_difference_n32']=norm_difference(data[32],half)
    report['status']='measured_approved_native_windows'
    report['interpretation']=('known steady manufactured-solution errors plus same-time grid differences; no prescribed global order'
        if args.steady_mms else 'same-time grid differences and wall traces, not analytic solution errors or a prescribed global order; step-halving sensitivity reported separately')
    report.pop('active_case');save()


if __name__=='__main__':
    main()
