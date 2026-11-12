"""Exact candidate analysis for the approved SBP/SAT design route; no CFD advancement or port."""
import argparse
import json
from pathlib import Path

import numpy as np
import sympy as sp

from study_keep_sbp63_family import derive


def matrices(n,norms,left,centered):
    q=sp.zeros(n+1)
    for i in range(6):
        for j in range(9):
            q[i,j]=left[i,j];q[n-i,n-j]=-left[i,j]
    for i in range(6,n-5):
        for radius,value in centered.items():
            q[i,i+radius]=value;q[i,i-radius]=-value
    weights=[sp.S.One]*(n+1)
    for i,value in enumerate(norms):
        weights[i]=value;weights[n-i]=value
    # Unit physical interval, n intervals; the production computational spacing is one.
    h=sp.diag(*(value/n for value in weights))
    d=h.inv()*q
    b=sp.zeros(n+1);b[0,0]=-1;b[n,n]=1
    r=sp.zeros(n+1);r[0,0]=r[n,n]=1
    if any(sp.expand(value)!=0 for value in h*d+d.T*h-b):
        raise AssertionError('Candidate SBP identity differs')
    return h,d,b,r,q


def error_minimizer(norms,left,theta):
    leading=[sp.expand(sum(left[i,j]*(j-i)**4 for j in range(9))/(24*norms[i])) for i in range(6)]
    objective=sp.factor(sum(norms[i]*leading[i]**2 for i in range(6)))
    minimizer,=sp.solve(sp.diff(objective,theta),theta)
    if not sp.diff(objective,theta,2).is_positive:
        raise AssertionError('Boundary leading-error objective is not strictly convex')
    return minimizer,dict(status='proposed_criterion_not_selected',
        objective='sum_i h_i*c_i(theta)^2 for local cubic-order leading h^3 derivative errors',
        leading_coefficients=list(map(str,leading)),minimizer=str(minimizer),
        minimizer_decimal=float(minimizer),objective_minimum=str(sp.factor(objective.subs(theta,minimizer))),
        limitation='minimizes this local error objective only, not CFD stability, dispersion or runtime')


def scalar_diffusion(h,d,b,r):
    n=h.rows-1
    x=sp.Matrix([sp.Rational(i,n) for i in range(n+1)])
    a=sp.diag(*(1+value for value in x));hi=h.inv()
    lift=d-hi*b
    operator=d*a*lift-hi*r*a*hi
    expected=-d.T*h*a*d+b*a*d+d.T*a*b-2*r*a*hi
    if any(sp.expand(value)!=0 for value in h*operator-expected):
        raise AssertionError('Gradient lift/flux penalty does not match the symmetric Nitsche form')
    value=sp.Matrix([1+z+z**3 for z in x]);gradient=d*value
    energy=sp.expand((value.T*h*operator*value)[0])
    squares=-sum(h[i,i]*a[i,i]*gradient[i]**2 for i in range(n+1))/2
    squares-=sum(h[i,i]*a[i,i]*gradient[i]**2 for i in range(1,n))/2
    squares-=sum(h[i,i]*a[i,i]*(gradient[i]-2*b[i,i]*value[i]/h[i,i])**2 for i in (0,n))/2
    if sp.expand(energy-squares)!=0:
        raise AssertionError('Scalar diffusion negative-square certificate differs')
    load=-hi*d.T*a*b+2*hi*r*a*hi
    if any(sp.expand(v)!=0 for v in operator*x+load*x-sp.ones(n+1,1)):
        raise AssertionError('Affine field with a=1+x does not reproduce (a*u_x)_x=1')
    return dict(sbp='H D + D.T H = B exactly for all theta',
        scalar_gradient='G=D*u+H^-1*B*(g-u)',
        scalar_rhs='D*A*G+H^-1*R*A*H^-1*(g-u)',
        homogeneous_matrix='L=D*A*(D-H^-1*B)-H^-1*R*A*H^-1',
        matrix_identity='H*L=-D.T*H*A*D+B*A*D+D.T*A*B-2*R*A*H^-1',
        negative_square_certificate='-1/2*sum_all(h*a*w^2)-1/2*sum_interior(h*a*w^2)-1/2*sum_endpoints(h*a*(w-2*n*u/h)^2), w=D*u',
        tested_coefficient='a_i=1+i/n > 0',
        affine_manufactured_rhs='exactly 1; nonhomogeneous endpoint data taken from u=x',
        interpretation='scalar/frozen-linear candidate, not a nonlinear Navier-Stokes wall proof')


def acoustic_certificate():
    rho,t,gas,gamma=sp.symbols('rho0 T0 R gamma',positive=True)
    cv=gas/(gamma-1)
    s=sp.diag(gas*t/rho,rho,rho,rho,rho*cv/t)
    a=sp.zeros(5);a[0,2]=rho;a[2,0]=gas*t/rho;a[2,4]=gas;a[4,2]=(gamma-1)*t
    c=sp.zeros(5);c[0,2]=rho;c[4,2]=(gamma-1)*t
    if sp.simplify(s*a-a.T*s)!=sp.zeros(5) or sp.simplify(s*c+c.T*s-s*a)!=sp.zeros(5):
        raise AssertionError('Frozen primitive acoustic wall cancellation differs')
    return dict(variables=['delta_rho','delta_u','delta_v_normal','delta_w','delta_T'],
        symmetrizer=list(map(str,s.diagonal())),assumptions='rho0,T0,R>0; gamma>1; stationary rest state',
        a=[list(map(str,a.row(i))) for i in range(5)],c=[list(map(str,c.row(i))) for i in range(5)],
        linear_rhs='-A*D*z+C*H^-1*B*z',
        identity='S*A=A.T*S; S*C+C.T*S=S*A exactly, hence the inviscid H-energy derivative is zero',
        limitation='no acoustic boundary damping claimed; no variable-shear or nonlinear stability implication')


def frozen_keep(q,b):
    n=q.rows-1;gamma=sp.Rational(7,5)
    x=[sp.Rational(i,n) for i in range(n+1)]
    rho=[1+v*v/20 for v in x];pressure=[1+v**3/30 for v in x]
    velocity=[sp.Matrix([sp.Rational(1,10)+v/20,sp.Rational(1,40)-v/30,v*v/50]) for v in x]
    internal=[pressure[i]/((gamma-1)*rho[i]) for i in range(n+1)]
    def wall_flux(i):
        return sp.Matrix([0,0,pressure[i],0,0])
    def physical_flux(i):
        u=velocity[i];mass=rho[i]*u[1]
        return sp.Matrix([mass,*(mass*u+sp.Matrix([0,pressure[i],0])),
                          mass*(internal[i]+u.dot(u)/2)+pressure[i]*u[1]])
    rates=[]
    for i in range(n+1):
        rate=sp.zeros(5,1)
        for j in range(n+1):
            if q[i,j]==0:
                continue
            u=(velocity[i]+velocity[j])/2
            mass=(rho[i]+rho[j])*u[1]/2
            flux=sp.Matrix([mass,*(mass*u+sp.Matrix([0,(pressure[i]+pressure[j])/2,0])),
                mass*((internal[i]+internal[j])/2+velocity[i].dot(velocity[j])/2)
                +(velocity[i][1]*pressure[j]+velocity[j][1]*pressure[i])/2])
            rate-=2*q[i,j]*flux
        if i in (0,n):
            rate-=b[i,i]*(wall_flux(i)-physical_flux(i))
        rates.append(rate)
    net=sum(rates,sp.zeros(5,1));expected=wall_flux(0)-wall_flux(n)
    if any(sp.expand(v)!=0 for v in net-expected):
        raise AssertionError('Weak inviscid wall does not conserve prescribed wall mass/energy flux')
    kinetic=sp.expand(sum(velocity[i].dot(rates[i][1:4,0])-velocity[i].dot(velocity[i])*rates[i][0]/2 for i in range(n+1)))
    pressure_work=-sum(velocity[i][1]*q[i,j]*pressure[j] for i in range(n+1) for j in range(n+1))
    if sp.expand(kinetic-pressure_work)!=0:
        raise AssertionError('Weak inviscid wall pressure-work exchange differs')
    return dict(state='positive rational rho/p; nonzero normal and tangential wall trace velocities',
        outward_wall_flux='(0,p*n_x,p*n_y,p*n_z,0)',
        rhs='-2*(D Hadamard F_keep)*1-H^-1*B*(F_wall-F_physical)',
        conservative_flux_residual=['0']*5,mass_rate='0',total_energy_rate='0',
        kinetic_pressure_work_residual='0',parameter='all real theta',
        limitation='frozen flux algebra, not positivity or an entropy-stable volume scheme')


def frozen_viscous(h,d,b,r):
    n=h.rows-1;hi=h.inv()
    x=[sp.Rational(i,n) for i in range(n+1)]
    velocity=sp.Matrix([[sp.Rational(1,10)+z/20,sp.Rational(1,40)-z/30,z*z/50] for z in x])
    temperature=sp.Matrix([(1+z/20)**2 for z in x]);wall=sp.ones(n+1,1)
    sutherland=sp.Rational(2206,5463)
    mu=[temperature[i]*(1+x[i]/20)*(1+sutherland)/(1000*(temperature[i]+sutherland)) for i in range(n+1)]
    kappa=[value/sp.Rational(18,25)/sp.Rational(9,250) for value in mu]
    gu=(d-hi*b)*velocity;gt=d*temperature+hi*b*(wall-temperature)
    stress=sp.zeros(n+1,3);penalty=sp.zeros(n+1,3)
    for i in range(n+1):
        for c,factor in enumerate((sp.S.One,sp.Rational(4,3),sp.S.One)):
            stress[i,c]=mu[i]*factor*gu[i,c]
            penalty[i,c]=mu[i]*factor*velocity[i,c]/h[i,i]
    rm=d*stress-hi*r*penalty
    bulk_energy=sp.Matrix([sum(velocity[i,c]*stress[i,c] for c in range(3))+kappa[i]*gt[i] for i in range(n+1)])
    numerical_energy=sp.Matrix([kappa[i]*gt[i]-b[i,i]*kappa[i]*(temperature[i]-1)/h[i,i] for i in range(n+1)])
    re=d*bulk_energy+hi*b*(numerical_energy-bulk_energy)
    kinetic=sp.expand(sum(h[i,i]*sum(velocity[i,c]*rm[i,c] for c in range(3)) for i in range(n+1)))
    dissipation=sum(h[i,i]*sum(gu[i,c]*stress[i,c] for c in range(3)) for i in range(n+1))
    boundary_loss=sum(sum(velocity[i,c]*penalty[i,c] for c in range(3)) for i in (0,n))
    total_energy=sp.expand(sum(h[i,i]*re[i] for i in range(n+1)))
    heat=sp.expand(numerical_energy[n]-numerical_energy[0])
    if sp.expand(kinetic+dissipation+boundary_loss)!=0 or sp.expand(total_energy-heat)!=0:
        raise AssertionError('Nonlinear primitive-lift wall work/heat identity failed')
    return dict(state='positive rational rho/T possible; T=(1+x/20)^2; nonzero wall trace velocity and thermal mismatch',
        material='Sutherland s=2206/5463, Re=1000, Pr=18/25, (gamma-1)*Ma^2=9/250',
        velocity_gradient='G_u=D*u+H^-1*B*(u_wall-u), u_wall=0',
        thermal_gradient='G_T=D*T+H^-1*B*(T_wall-T), T_wall=1',
        viscous_wall_momentum_flux='sigma(G_u)*n - mu/h_b*(I+n*n.T/3)*(u-u_wall)',
        viscous_wall_energy_flux='u_wall dot momentum_flux + kappa*(G_T dot n) - kappa/h_b*(T-T_wall)',
        kinetic_identity='dK/dt=-sum(H*sigma:G_u)-sum_wall(mu/h_b*(|u|^2+(u dot n)^2/3))',
        kinetic_identity_residual='0 for all theta',
        total_energy_identity='dE/dt=sum_wall(kappa*G_T dot n-kappa/h_b*(T-T_wall))',
        total_energy_identity_residual='0 for all theta',
        limitation='frozen conservative KE/total-energy identities, not a nonlinear entropy, positivity or RK3 stability proof')


def layout_preflight(norms,left,centered):
    checks=[]
    for n,ranks in ((16,1),(16,2),(20,4)):
        h,d,b,r,q=matrices(n,norms,left,centered)
        local=n//ranks;hm=5
        if n%ranks or (ranks>1 and local<hm):
            raise AssertionError('Proposed layout cannot pack the existing five-layer neighbor buffers')
        records=[]
        for rank in range(ranks):
            start=rank*local;end=start+local
            available=set(range(start,end+1))
            if rank:
                # Neighbor packs local jm-hm:jm-1, excluding its shared top node.
                available.update(range(start-hm,start))
            if rank<ranks-1:
                # Neighbor packs local 1:hm, excluding its shared bottom node.
                available.update(range(end+1,end+hm+1))
            for j in range(start,end+1):
                needed={k for k in range(n+1) if d[j,k]!=0}
                if not needed<=available:
                    raise AssertionError(f'Candidate row lacks a neighbor value: {n,ranks,rank,j,needed-available}')
            records.append(dict(rank=rank,owned_with_shared=[start,end],available=[min(available),max(available)],
                physical_sat_nodes=[j for j in (0,n) if start<=j<=end],
                shared_endpoint_global_weights={str(j):str(n*h[j,j]) for j in (start,end) if j not in (0,n)}))
        checks.append(dict(intervals=n,ranks_y=ranks,local_intervals=local,hm=hm,ranks=records))
    return dict(scope='index and pack-range analysis only, not native MPI execution',checks=checks,
        communication='current-state halo before gradients; modified stress/heat halo before divergence; global endpoint-row selection on all ranks')


def linear_preflight(norms,left,centered,minimizer,theta):
    gamma=1.4;rho=temperature=1.;mach=.3;reynolds=3000.;prandtl=.72;dt=1e-5
    gas=1/(gamma*mach**2);cv=gas/(gamma-1);mu=1/reynolds
    kappa=mu/prandtl/((gamma-1)*mach**2)
    component_norm=np.array([gas*temperature/rho,rho,rho,rho,rho*cv/temperature])
    reports=[]
    for n in (16,20):
        h,d,b,r,q=matrices(n,norms,left,centered)
        h=np.array(h,dtype=float)*np.pi
        dy=np.array(d.subs(theta,minimizer),dtype=float)/np.pi
        bb=np.array(b,dtype=float);rr=np.array(r,dtype=float);hi=np.diag(1/np.diag(h))
        eye=np.eye(n+1);mass=np.concatenate([value*np.diag(h) for value in component_norm])
        scale=np.sqrt(mass);worst=dict(rk3_energy_norm=-np.inf)
        max_abscissa=-np.inf;max_spectral_radius=0.;max_rk_eigen_amplification=0.
        for ix in range(n//2+1):
            for iz in range(n//2+1):
                symbol=lambda k:(1.5*np.sin(2*np.pi*k/n)-.3*np.sin(4*np.pi*k/n)+np.sin(6*np.pi*k/n)/30)/(2*np.pi/n)
                derivatives=[1j*symbol(ix)*eye,dy.astype(complex),1j*symbol(iz)*eye]
                lifted=[derivatives[0],derivatives[1]-hi@bb,derivatives[2]]
                laplace=sum(derivatives[a]@lifted[a] for a in range(3))
                matrix=np.zeros((5*(n+1),5*(n+1)),dtype=complex)
                def add(row,column,values):
                    matrix[row*(n+1):(row+1)*(n+1),column*(n+1):(column+1)*(n+1)]+=values
                for c in range(3):
                    add(0,c+1,-rho*derivatives[c]);add(4,c+1,-(gamma-1)*temperature*derivatives[c])
                    add(c+1,0,-gas*temperature/rho*derivatives[c]);add(c+1,4,-gas*derivatives[c])
                add(0,2,rho*hi@bb);add(4,2,(gamma-1)*temperature*hi@bb)
                for a in range(3):
                    for c in range(3):
                        viscous=derivatives[c]@lifted[a]-(2/3)*derivatives[a]@lifted[c]
                        if a==c:
                            viscous+=laplace-(4/3 if a==1 else 1)*hi@rr@hi
                        add(a+1,c+1,mu/rho*viscous)
                add(4,4,kappa/(rho*cv)*(laplace-hi@rr@hi))
                whitened=scale[:,None]*matrix/scale[None,:]
                eigenvalues=np.linalg.eigvals(whitened)
                z=dt*whitened;z2=z@z
                rk=np.eye(len(z))+z+z2/2+(z2@z)/6
                norm=float(np.linalg.svd(rk,compute_uv=False)[0])
                if norm>worst['rk3_energy_norm']:
                    worst=dict(modes=[ix,iz],rk3_energy_norm=norm)
                max_abscissa=max(max_abscissa,float(eigenvalues.real.max()))
                max_spectral_radius=max(max_spectral_radius,float(abs(eigenvalues).max()))
                zz=dt*eigenvalues
                max_rk_eigen_amplification=max(max_rk_eigen_amplification,float(abs(1+zz+zz**2/2+zz**3/6).max()))
        reports.append(dict(intervals=n,mode_pairs=(n//2+1)**2,max_real_eigenvalue=max_abscissa,
            spectral_radius=max_spectral_radius,max_abs_dt_lambda=dt*max_spectral_radius,
            max_rk3_eigen_amplification=max_rk_eigen_amplification,worst_energy_mode=worst))
    return dict(scope='complete frozen-rest normal/tangential Fourier blocks on the proposed uniform box; no CFD advancement',
        domain=[2*np.pi,np.pi,2*np.pi],dt=dt,reynolds=reynolds,mach=mach,prandtl=prandtl,theta=float(minimizer),
        assumptions='stationary uniform rho=T=1; constant material at that state; nonnegative Fourier quadrant by rest-state parity',
        interpretation='floating eigenvalues/norms reported without a new pass threshold; not a guarantee for nonlinear shear or evolving material',checks=reports)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    norms,left,centered,theta=derive()
    minimizer,objective=error_minimizer(norms,left,theta)
    if args.preflight_only:
        report=dict(status='proposed_candidate_prerequisites_reviewed_not_ported',
            layout=layout_preflight(norms,left,centered),
            frozen_linear=linear_preflight(norms,left,centered,minimizer,theta))
        with args.output.open('x') as stream:
            json.dump(report,stream,indent=2);stream.write('\n')
        print('ROUTE_A_PREFLIGHT_RECORDED; no native MPI or CFD execution',flush=True)
        return
    checks=[]
    for n in (16,32):
        h,d,b,r,q=matrices(n,norms,left,centered)
        record=dict(intervals=n,scalar_diffusion=scalar_diffusion(h,d,b,r))
        if n==16:
            record['frozen_keep_wall']=frozen_keep(q,b)
            record['frozen_nonlinear_viscous_wall']=frozen_viscous(h,d,b,r)
        # Floating spectra are observations, not the basis of the exact stability certificate.
        chosen=np.array(d.subs(theta,minimizer),dtype=float)
        hh=np.array(h,dtype=float);bb=np.array(b,dtype=float);rr=np.array(r,dtype=float)
        hi=np.diag(1/np.diag(hh));normal=chosen@(chosen-hi@bb)-hi@rr@hi
        eigenvalues=np.linalg.eigvalsh((np.sqrt(hh)@normal@np.sqrt(hi)+np.sqrt(hi)@normal.T@np.sqrt(hh))/2)
        record['proposed_theta_unit_diffusion_symmetric_extrema']=[float(eigenvalues[0]),float(eigenvalues[-1])]
        checks.append(record)
    support=max(abs(i-j) for i in range(6) for j in range(9) if left[i,j]!=0)
    report=dict(status='route_a_approved_candidate_formulas_not_approved_or_ported',
        scope='explicit SBP6/3, stationary flat two-wall reference; original arithmetic KEEP interior',
        maximum_boundary_row_offset=support,
        theta_criterion=objective,acoustic=acoustic_certificate(),checks=checks,
        unresolved=['theta objective and complete candidate formula approval','full 3D conservative viscous/thermal SAT coupling and native checks',
                    'stretched-grid metric/norm pairing','explicit RK3 parabolic/penalty timestep constraint',
                    'MPI physical-end data ownership','new candidate identity and native acceptance gates'])
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')
    print('ROUTE_A_EXACT_CANDIDATE_CHECKS_PASS; no native case or production formula changed',flush=True)


if __name__=='__main__':
    main()
