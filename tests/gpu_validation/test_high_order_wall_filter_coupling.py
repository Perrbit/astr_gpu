"""Actual CPU filter matrices and frozen RK/filter/wall composition, no CFD march."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import sympy as sp
from scipy.linalg import block_diag,eig

from test_high_order_boundary_geometry import procedure,ROOT
from test_high_order_explicit_wall_candidates import dynamic_wall_cns_matrix,state_projection
from test_high_order_adiabatic_wall_candidate import adiabatic_cns_matrix
from test_high_order_boundary_closures import derivative_matrix


def native_filter_matrices(directory,intervals=(32,64,128),wall_kind=0):
    directory=directory.resolve()
    directory.mkdir(parents=True,exist_ok=False)
    initialization=procedure(ROOT/'src/filter.F90','subroutine','filter_coefficient_explicit')
    filtering=procedure(ROOT/'src/filter.F90','function','spafilter10exp')
    source=directory/'filter_probe.F90'
    source.write_text('''module commvar
  integer,parameter :: hm=5
end module
module filter_probe
  use astr_precision
  implicit none
  real(real_kind),allocatable :: coef2e(:),coef4e(:),coef6e(:),coef8e(:),coef10e(:)
  real(real_kind),allocatable :: coef3be(:,:),coef4be(:,:),coef6be(:,:)
contains
'''+initialization+filtering+'''end module
program probe
  use filter_probe
  use commvar
  implicit none
  integer :: n,column,unit,wall_kind
  character(1200) :: arg,filename
  real(real_kind),allocatable :: f(:),matrix(:,:),a(:),b(:)
  call get_command_argument(1,arg);read(arg,*) n
  call get_command_argument(2,filename)
  call get_command_argument(3,arg);read(arg,*) wall_kind
  call filter_coefficient_explicit()
  allocate(f(-hm:n+hm),matrix(0:n,0:n))
  do column=0,n
    f=0.e0_real_kind;f(column)=1.e0_real_kind
    matrix(:,column)=spafilter10exp(f,4,n,wall_kind)
  enddo
  f=0.e0_real_kind
  do column=0,n
    f(column)=exp(4.e0_real_kind*real(column,real_kind)/n)
  enddo
  a=spafilter10exp(f,4,n,wall_kind)
  f(-hm:-1)=123.e0_real_kind;f(n+1:n+hm)=-456.e0_real_kind
  b=spafilter10exp(f,4,n,wall_kind)
  if(any(a/=b)) error stop 'physical filter consumes exterior ghosts'
  open(newunit=unit,file=trim(filename),status='new',access='stream',form='unformatted')
  write(unit) matrix
  close(unit)
end program
''')
    executable=directory/'filter_probe'
    result=subprocess.run(['gfortran','-cpp','-ffree-line-length-none',str(ROOT/'src/astr_precision.F90'),
        str(ROOT/'src/boundary_high_order.F90'),str(source),'-o',str(executable)],cwd=directory,text=True,capture_output=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    matrices={}
    for n in intervals:
        path=directory/f'filter_{n}.bin'
        result=subprocess.run([str(executable),str(n),str(path),str(wall_kind)],cwd=directory,text=True,capture_output=True,timeout=30)
        assert result.returncode==0,result.stdout+result.stderr
        matrices[n]=np.fromfile(path,'<f8').reshape(n+1,n+1,order='F')
    return matrices


def exact_filter_moments(matrix):
    n=len(matrix)-1
    for row in range(n+1):
        distance=min(row,n-row)
        if distance==0:
            expected=np.zeros(n+1);expected[row]=1
            assert np.array_equal(matrix[row],expected)
            continue
        order=6 if distance<=3 else 8 if distance==4 else 10
        delta=matrix[row].copy();delta[row]-=1
        coefficients=[(j-row,Fraction.from_float(float(delta[j]))) for j in np.flatnonzero(delta)]
        for power in range(order):
            assert sum(c*int(offset)**power for offset,c in coefficients)==0,(row,power)
        assert sum(c*int(offset)**order for offset,c in coefficients)!=0,(row,order)


def rk3_filtered_map(matrix,filtering,dt):
    # The first filtered state is qsave. Each subsequent stage is filtered before
    # its Euler RHS, then the constrained wall state is reconstructed.
    euler=np.eye(len(matrix))+dt*matrix
    stage1=euler@filtering
    stage2=.75*filtering+.25*euler@filtering@stage1
    return filtering/3+(2/3)*euler@filtering@stage2


def joint_report(matrices):
    rows=[]
    for n,f in matrices.items():
        exact_filter_moments(f)
        for kind in (41,42,421):
            velocity=state_projection(n,'dirichlet')
            thermal=state_projection(n,'dirichlet' if kind==41 else 'neumann')
            reduction=block_diag(f,f[1:-1]@velocity,f[1:-1]@thermal)
            tangent=state_projection(n,'neumann' if kind==421 else 'dirichlet')
            tangent_filter=f[1:-1]@tangent
            derivative=n*derivative_matrix(n,True)
            for nu in (0.,1/1600):
                a=dynamic_wall_cns_matrix(n,nu) if kind==41 else adiabatic_cns_matrix(n,nu,'neumann7-projection')
                tangent_a=nu*(derivative@derivative@tangent)[1:-1]
                row=dict(intervals=n,bctype=kind,viscosity=nu,samples=[])
                for cfl in (0.,.05,.1,.2):
                    dt=cfl/(n*np.sqrt(1.4))
                    coupled=rk3_filtered_map(a,reduction,dt)
                    tangent_step=rk3_filtered_map(tangent_a,tangent_filter,dt)
                    radius=max(float(np.max(abs(np.linalg.eigvals(coupled)))),
                        float(np.max(abs(np.linalg.eigvals(tangent_step)))))
                    tolerance=100*np.finfo(float).eps*max(1.,np.linalg.norm(coupled,np.inf),np.linalg.norm(tangent_step,np.inf))
                    row['samples'].append(dict(sound_cfl=cfl,spectral_radius=radius,
                        roundoff_reference=float(tolerance),passes=bool(radius<=1+tolerance)))
                rows.append(row)
    return dict(scope='actual 0-6-6-6-8-10 filter, frozen normal CNS and tangential diffusion; not nonlinear ASTR',
        state_phase='filter each stage; qsave after first filter and wall projection',
        samples=rows,all_pass=all(s['passes'] for r in rows for s in r['samples']))


def coupling_diagnosis(f):
    """Resolve the growing normal mode without changing the acceptance gate."""
    n=len(f)-1
    reduction=block_diag(f,f[1:-1]@state_projection(n,'dirichlet'),
        f[1:-1]@state_projection(n,'neumann'))
    samples=[]
    for nu in (0.,1e-6,1e-5,1e-4,1/1600):
        a=adiabatic_cns_matrix(n,nu,'neumann7-projection')
        for cfl in (.05,.1,.15,.2):
            dt=cfl/(n*np.sqrt(1.4))
            mapping=rk3_filtered_map(a,reduction,dt)
            values,left,right=eig(mapping,left=True,right=True)
            index=int(np.argmax(abs(values)))
            value=values[index];u=left[:,index];v=right[:,index]
            norm=float(np.linalg.norm(mapping,np.inf))
            condition=float(np.linalg.norm(u)*np.linalg.norm(v)/abs(np.vdot(u,v)))
            residual=float(np.linalg.norm(mapping@v-value*v,np.inf))
            unfiltered=rk3_filtered_map(a,np.eye(len(a)),dt)
            samples.append(dict(viscosity=nu,sound_cfl=cfl,
                eigenvalue=[float(value.real),float(value.imag)],spectral_radius=float(abs(value)),
                eigenpair_residual=residual,
                relative_eigenpair_residual=residual/(norm*float(np.linalg.norm(v,np.inf))),
                eigenvalue_condition=condition,
                conditioned_roundoff_scale=float(np.finfo(float).eps*norm*condition),
                unfiltered_rk_radius=float(np.max(abs(np.linalg.eigvals(unfiltered))))))
    return dict(intervals=n,bctype=[42,421],
        scope='same frozen normal block at rest; not nonlinear or production validation',
        filter_alone_radius=float(np.max(abs(np.linalg.eigvals(reduction)))),
        samples=samples,
        verdict='coupled growth exists at positive viscosity; filtered hermite7 admission remains closed',
        caveat='Near-unit raw eigenvalues are retained; sampled CFLs do not establish an admissible interval.')


def constraint_diagnosis(f):
    """Locate the existing growing mode and measure, not replace, its projection."""
    n=len(f)-1
    thermal=state_projection(n,'neumann')
    velocity=state_projection(n,'dirichlet')
    # This is the actual Neumann7 state constraint, not the distinct D6
    # diagnostic derivative used to evaluate the converging wall heat flux.
    weights=np.array(sp.finite_diff_weights(1,list(range(8)),0)[1][-1],float)
    constraint=np.zeros((2,n+1))
    constraint[0,:8]=weights;constraint[1,-8:]=weights[::-1]
    initial_residual=float(np.linalg.norm(constraint@thermal,np.inf))
    assert initial_residual<1e-12
    filtering=block_diag(f,f[1:-1]@velocity,f[1:-1]@thermal)
    a=adiabatic_cns_matrix(n,1e-5,'neumann7-projection')
    mapping=rk3_filtered_map(a,filtering,.15/(n*np.sqrt(1.4)))
    values,vectors=eig(mapping)
    index=int(np.argmax(abs(values)))
    mode=vectors[:,index];mode=mode/np.max(abs(mode))
    rho=mode[:n+1];v=velocity@mode[n+1:2*n];t=thermal@mode[2*n:]
    raw=f@t;projected=thermal@raw[1:-1]
    near=np.r_[0:8,n-7:n+1]
    fields={}
    for name,field in (('rho',rho),('normal_velocity',v),('temperature',t),('pressure',rho+t)):
        fields[name]=dict(maximum_node=int(np.argmax(abs(field))),
            boundary_strip_squared_l2_fraction=float(np.linalg.norm(field[near])**2/np.linalg.norm(field)**2))
    return dict(intervals=n,viscosity=1e-5,sound_cfl=.15,
        scope='frozen eigenmode with max reduced amplitude normalized to one, not a physical field',
        spectral_radius=float(abs(values[index])),fields=fields,
        boundary_strip_nodes=near.tolist(),
        constraint='h times the inward seventh-order one-sided derivative at each wall',
        initial_constraint_operator_residual=initial_residual,
        filter_constraint_operator_defect=float(np.linalg.norm(constraint@f@thermal,np.inf)),
        mode_constraint_before_filter=float(np.max(abs(constraint@t))),
        mode_constraint_after_filter=float(np.max(abs(constraint@raw))),
        mode_constraint_after_projection=float(np.max(abs(constraint@projected))),
        maximum_temperature_projection_change_relative_to_mode=float(np.max(abs(projected-raw))/np.max(abs(t))),
        caveat='Noncommuting filter/projection is a measured lead, not a proof of the sole instability mechanism.')


@pytest.fixture(scope='module')
def filter_data(tmp_path_factory):
    root=tmp_path_factory.mktemp('wall_filter_static')
    matrices=native_filter_matrices(root/'native')
    report=joint_report(matrices)
    (root/'joint_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return matrices,report


def test_existing_main_filter_exact_moments(filter_data):
    matrices,_=filter_data
    for matrix in matrices.values():exact_filter_moments(matrix)


def test_frozen_filter_rk_wall_spectrum(filter_data):
    _,report=filter_data
    assert report['all_pass'],[(r['intervals'],r['bctype'],r['viscosity'],s)
        for r in report['samples'] for s in r['samples'] if not s['passes']]


def test_constraint_diagnosis_uses_neumann7_not_d6(filter_data):
    matrices,_=filter_data
    report=constraint_diagnosis(matrices[32])
    assert report['mode_constraint_before_filter']<1e-12
    assert report['mode_constraint_after_projection']<1e-12


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reuse-native',type=Path,
        help='Read immutable native filter matrices instead of recompiling the probe')
    args=parser.parse_args()
    if args.reuse_native is None:
        matrices=native_filter_matrices(args.output.resolve()/'native')
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        matrices={n:np.fromfile(args.reuse_native/f'filter_{n}.bin','<f8').reshape(n+1,n+1,order='F')
            for n in (32,64,128)}
    report=joint_report(matrices)
    (args.output/'joint_report.json').write_text(json.dumps(report,indent=2)+'\n')
    diagnosis=coupling_diagnosis(matrices[32])
    (args.output/'coupling_diagnosis.json').write_text(json.dumps(diagnosis,indent=2)+'\n')
    constraints=constraint_diagnosis(matrices[32])
    (args.output/'constraint_diagnosis.json').write_text(json.dumps(constraints,indent=2)+'\n')
    print(json.dumps(dict(all_pass=report['all_pass'],failures=[dict(intervals=r['intervals'],bctype=r['bctype'],
        viscosity=r['viscosity'],sample=s) for r in report['samples'] for s in r['samples'] if not s['passes']]),indent=2))
