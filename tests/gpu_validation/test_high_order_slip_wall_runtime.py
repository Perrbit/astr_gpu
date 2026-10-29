"""HB1-D actual ASTR slip-wall constraints and full wall continuity."""
import h5py
import numpy as np
import pytest
import sympy as sp

from test_high_order_wall_runtime import launch,EXE,GRID,check_completed_wall_diagnostics
from test_high_order_adiabatic_wall_runtime import (
    test_adiabatic_same_stage_fields as check_fields,
    test_adiabatic_state_and_tangent_rhs as check_temperature,
    test_adiabatic_actual_rhs_spatial_convergence as check_convergence,
    test_adiabatic_exact_restart as check_restart,
    test_adiabatic_algorithm_change_rejected as check_identity)
from test_high_order_boundary_closures import derivative_rows
from run_native_precision_tgv_gate import physical_state,rhs_state
from run_output_restart_validation import compare_fields

pytestmark=pytest.mark.skipif(not EXE,reason='Set ASTR_HB1B_EXE to the root-CMake FP64 CUDA executable')


@pytest.mark.parametrize('topology',[(1,1,1),(2,1,1),(1,2,1),(1,1,2),(2,2,1),(1,4,1)])
def test_slip_same_stage_fields(launch,topology,record_property):
    check_fields(launch,topology,record_property,wall_type=421)


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_slip_temperature_and_tangent_rhs(launch,backend):
    check_temperature(launch,backend,wall_type=421)


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_slip_velocity_and_complete_wall_continuity(launch,backend,record_property):
    case,checkpoint=launch(backend,wall_type=421)
    suffix='step00000000.rk01.rank00000000.bin'
    q=physical_state(case/'diagnostics'/f'state.pre_rhs.{suffix}','fp64')
    rhs=rhs_state(case/'diagnostics'/f'state.full.{suffix}','fp64')
    after=physical_state(case/'diagnostics'/f'state.post_update.{suffix}','fp64')
    with h5py.File(checkpoint/'state.h5') as f:
        final=np.stack([f[f'q{m:04d}'][:].transpose(2,1,0) for m in range(1,6)],axis=-1)
    with h5py.File(case/'outdat/new/resources/geometry.h5') as f:
        jacobian=f['q0004'][:].transpose(2,1,0)
        metric=[f[f'q{m:04d}'][:].transpose(2,1,0) for m in (5,9,13)]
    a=np.array(sp.finite_diff_weights(1,list(range(8)),0)[1][-1],float)
    projection=-a[1:]/a[0]
    for state in (q,after,final):
        velocity=state[...,1:4]/state[...,0,None]
        for j,side in ((0,1),(GRID[1],-1)):
            assert np.max(abs(velocity[:,j,:,1]))==0
            assert np.max(abs(velocity[:,j,:,0]))>.05
            for c in (0,2):
                expected=np.einsum('s,isk->ik',projection,velocity[...,c][:,j+side*np.arange(1,8),:])
                assert np.max(abs(velocity[:,j,:,c]-expected))<2e-14
    physical_rhs=rhs/jacobian[...,None]
    velocity=q[...,1:4]/q[...,0,None]
    velocity_rhs=(physical_rhs[...,1:4]-velocity*physical_rhs[...,0,None])/q[...,0,None]
    normal_weights=np.array(derivative_rows(True)[0],float)
    worst=0
    for j,side in ((0,1),(GRID[1],-1)):
        for c in (0,2):
            expected=np.einsum('s,isk->ik',projection,velocity_rhs[...,c][:,j+side*np.arange(1,8),:])
            assert np.max(abs(velocity_rhs[:,j,:,c]-expected))<2e-12
        flux_y=jacobian[:,j+side*np.arange(8),:]*q[...,2][:,j+side*np.arange(8),:]*metric[1][:,j+side*np.arange(8),:]
        expected=-side*np.einsum('s,isk->ik',normal_weights,flux_y)[:-1,:-1]
        tangential=np.zeros_like(expected)
        for direction,component,axis in ((0,1,0),(2,3,1)):
            flux=(jacobian[:,j,:]*q[:,j,:,component]*metric[direction][:,j,:])[:-1,:-1]
            for offset,weight in ((1,.75),(2,-.15),(3,1/60)):
                tangential-=weight*(np.roll(flux,-offset,axis=axis)-np.roll(flux,offset,axis=axis))
        assert np.max(abs(tangential))>1e-7
        expected+=tangential
        error=float(np.max(abs(rhs[:-1,j,:-1,0]-expected)))
        worst=max(worst,error)
        assert error<5e-13,(backend,j,error)
    record_property('complete_wall_continuity_maxabs',worst)


def test_slip_actual_rhs_and_heat_convergence(launch,record_property):
    check_convergence(launch,record_property,wall_type=421)


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_slip_exact_restart(launch,backend):
    check_restart(launch,backend,wall_type=421)


def test_slip_algorithm_change_rejected(launch):
    check_identity(launch,wall_type=421)


def test_slip_memory_safety(launch):
    launch('gpu',(1,2,1),steps=2,wall_type=421,wall_samples=True,sanitizer=True)


@pytest.mark.parametrize('topology',[(1,1,1),(1,2,1)])
def test_slip_wall_diagnostics(launch,topology,record_property):
    check_completed_wall_diagnostics(launch,topology,'hermite7',record_property,wall_type=421)


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_slip_wall_diagnostic_isolation(launch,backend):
    _,plain=launch(backend,steps=2,wall_type=421)
    _,sampled=launch(backend,steps=2,wall_type=421,wall_samples=True)
    compare_fields(plain/'state.h5',sampled/'state.h5')
