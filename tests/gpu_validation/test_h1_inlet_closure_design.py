"""Exact moments and unchanged-interior checks for the approved design family."""
import numpy as np
import sympy as sp
import os
import subprocess
import pytest

from analyze_h1_inlet_closure_design import CURRENT,PROPOSED,SINGLE_STEP_TRIAL,exact_flux_rows,scalar_generator,mp_weights_exact,rk3_action
from test_high_order_boundary_closures import stabilized_mp_matrix,stabilized_flux_rows


def test_eight_node_flux_family_and_final_rhs_moments():
    beta=sp.symbols('b0:4')
    rows=exact_flux_rows(beta)
    for r,row in enumerate(rows):
        for m in range(7):
            assert sp.simplify(sum(w*j**m for j,w in enumerate(row))-sp.bernoulli(m,r+1))==0
    for r in range(1,4):
        for m in range(7):
            assert sp.simplify(sum((rows[r][j]-rows[r-1][j])*j**m for j in range(8))-
                (0 if m==0 else m*r**(m-1)))==0
    next_row=mp_weights_exact()
    for m in range(7):
        assert sp.simplify(sum(next_row[j]*(j+1)**m-rows[3][j]*j**m for j in range(8))-
            (0 if m==0 else m*4**(m-1)))==0


def test_current_coefficients_reproduce_the_existing_scalar_operator():
    assert exact_flux_rows(CURRENT)[:3]==stabilized_flux_rows()
    assert exact_flux_rows(CURRENT)[3]==mp_weights_exact()
    for n in (32,64):np.testing.assert_array_equal(scalar_generator(n,CURRENT),-stabilized_mp_matrix(n,.1)[1:,1:])


def test_proposed_family_changes_only_the_inlet_transition_rows():
    before=scalar_generator(64,CURRENT);after=scalar_generator(64,PROPOSED)
    np.testing.assert_array_equal(before[4:],after[4:])
    assert np.any(before[:4]!=after[:4])
    np.testing.assert_array_equal(before[:4,8:],after[:4,8:])


def test_single_step_improvement_does_not_prove_multistep_robustness():
    old=rk3_action(scalar_generator(64,CURRENT),.1,64)
    trial=rk3_action(scalar_generator(64,SINGLE_STEP_TRIAL),.1,64)
    norm=lambda value:np.linalg.svd(value[:,:8],compute_uv=False)[0]
    assert norm(trial)<norm(old)
    assert norm(np.linalg.matrix_power(trial,640))>norm(np.linalg.matrix_power(old,640))


def test_proposed_scalar_spectrum_and_rk3_region():
    for n in (32,64,128):
        eigen=np.linalg.eigvals(scalar_generator(n,PROPOSED))
        assert eigen.real.max()<0
        for cfl in (.05,.1,.2):
            z=cfl*eigen
            assert abs(1+z+z*z/2+z*z*z/6).max()<1


@pytest.mark.skipif(not os.environ.get('ASTR_HIGH_ORDER_BOUNDARY_PROBE'),reason='Select the root native boundary probe')
def test_native_compatible_flux_coefficients():
    result=subprocess.run([os.environ['ASTR_HIGH_ORDER_BOUNDARY_PROBE']],capture_output=True,text=True,check=True)
    rows=[list(map(float,line.split()[2:])) for line in result.stdout.splitlines() if line.startswith('HB_INLET_COMPAT ')]
    assert len(rows)==4,result.stdout
    np.testing.assert_allclose(rows,np.array(exact_flux_rows(PROPOSED),dtype=float),rtol=2e-14,atol=2e-14)


@pytest.mark.skipif(not os.environ.get('ASTR_HIGH_ORDER_BOUNDARY_PROBE'),reason='Select the root native boundary probe')
def test_native_compression_sensor_formula():
    result=subprocess.run([os.environ['ASTR_HIGH_ORDER_BOUNDARY_PROBE']],capture_output=True,text=True,check=True)
    actual=[float(line.split()[2]) for line in result.stdout.splitlines() if line.startswith('HB_COMPRESSION_SENSOR ')]
    divergence=np.array([-2.,-1.,-1e-10,0.,1e-10,1.,2.,-1e-16,1e-16])
    vorticity=np.array([0.,3.,1.,.5,1.,3.,0.,0.,0.])
    curvature=np.array([1.,.25,.5,1.,.5,.25,1.,1.,1.])
    expected=np.minimum(divergence,0)**2/(divergence**2+vorticity+1e-30)*curvature
    np.testing.assert_allclose(actual,expected,rtol=2e-14,atol=2e-14)
    assert np.all(np.array(actual)[divergence>=0]==0)
