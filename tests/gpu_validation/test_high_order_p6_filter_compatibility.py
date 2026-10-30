"""Model-selection and known-counterexample checks, not filter admission."""
import numpy as np
import pytest

import review_high_order_p6_filter_compatibility as review
from test_high_order_adiabatic_wall_candidate import adiabatic_cns_matrix
from test_high_order_adiabatic_pressure_candidate import pressure_gradient
from test_high_order_explicit_wall_candidates import dynamic_wall_cns_matrix
from test_high_order_wall_filter_coupling import native_filter_matrices
from test_high_order_coupled_filter_candidate import modal_filter


@pytest.mark.parametrize('n',[8,9,32])
def test_current_space_keeps_41_p7_and_changes_only_adiabatic_pressure_rows(n):
    original=adiabatic_cns_matrix(n,1/1600,'neumann7-projection')
    actual=review.current_normal(n,1/1600,42)
    np.testing.assert_array_equal(review.current_normal(n,1/1600,41),dynamic_wall_cns_matrix(n,1/1600))
    np.testing.assert_array_equal(actual,review.current_normal(n,1/1600,421))
    # Unchanged central rows use different coefficient evaluation orders.
    roundoff=100*np.finfo(float).eps*max(1.,np.linalg.norm(original,np.inf))
    changed=np.flatnonzero(np.max(abs(actual-original),axis=1)>roundoff)
    np.testing.assert_array_equal(changed,[n+1,n+2,2*n-2,2*n-1])
    np.testing.assert_array_equal(actual[n+1:2*n,:n+1],-pressure_gradient(n,6))


def test_corrected_native_baseline_reproduces_filtered_growth_without_false_space_failure(tmp_path):
    matrices=native_filter_matrices(tmp_path/'native',(8,))
    report=review.baseline_counterexample(matrices[8])
    assert abs(report['spatial_maximum_real'])<1e-12
    assert abs(report['filter_alone_radius']-1)<1e-12
    assert abs(report['unfiltered_rk_radius']-1)<1e-12
    assert abs(report['filtered_rk']['spectral_radius']-1.0055635020650118)<2e-13
    samples,failure=review.review(matrices,42,(6,6,6))
    assert failure is samples[-1] and not failure['passes']
    assert failure['diagnostic']['eigenpair_residual']<1e-12


def test_pre_fix_native_overlap_is_rejected_as_stale_evidence(tmp_path):
    matrix=np.eye(9);matrix[4,4]=2
    matrix.ravel(order='F').astype('<f8').tofile(tmp_path/'filter_8.bin')
    with pytest.raises(ValueError,match='pre-fix N8'):
        review.load_native(tmp_path)


@pytest.mark.parametrize('kind',[41,42,421])
def test_acoustic_cross_filter_has_the_original_scalar_limit_and_wall_parity(tmp_path,kind):
    f=native_filter_matrices(tmp_path/'native',(8,16))
    for n,original in f.items():
        baseline=review.acoustic_filter(original,kind,1.,1.,0.)
        np.testing.assert_array_equal(baseline,modal_filter(original,original,original,kind))
        matrix=review.acoustic_filter(original,kind,.5,.25,.125)
        reflection=np.diag(np.r_[np.ones(n+1),-np.ones(n-1),np.ones(n-1)])
        indices=np.r_[np.arange(n,-1,-1),np.arange(2*n-1,n,-1),np.arange(3*n-2,2*n-1,-1)]
        np.testing.assert_array_equal(matrix,reflection@matrix[np.ix_(indices,indices)]@reflection)
        np.testing.assert_array_equal(matrix[[0,n]],np.eye(len(matrix))[[0,n]])
        x=np.linspace(0,1,n+1)
        for degree in range(6):
            values=np.r_[x**degree,np.zeros(2*(n-1))]
            np.testing.assert_allclose(matrix@values,values,rtol=0,atol=2e-14)
        for degree in range(4):
            values=np.r_[np.zeros(n+1),(x*(1-x)*x**degree)[1:-1],np.zeros(n-1)]
            np.testing.assert_allclose(matrix@values,values,rtol=0,atol=2e-14)
        thermal_fields=([x*(1-x)*x**degree for degree in range(4)] if kind==41 else
            [np.ones(n+1)]+[x**(degree+2)/(degree+2)-x**(degree+3)/(degree+3) for degree in range(3)])
        for field in thermal_fields:
            values=np.r_[np.zeros(2*n),field[1:-1]]
            np.testing.assert_allclose(matrix@values,values,rtol=0,atol=2e-14)
        if n==16:
            expected=modal_filter(original,original,original,kind)
            rows=np.r_[np.arange(5,n-4),n+np.arange(5,n-4),2*n-1+np.arange(5,n-4)]
            np.testing.assert_array_equal(matrix[rows],expected[rows])


def test_inviscid_energy_reference_does_not_claim_viscous_filter_admission():
    filtering,metric,report=review.energy_reference(8,42)
    assert report['metric_minimum_eigenvalue']>0
    assert report['polynomial_residual']<2e-10 and report['endpoint_residual']<2e-10
    tolerance=100*np.finfo(float).eps*max(1.,np.linalg.norm(metric,np.inf))
    assert report['contraction_maximum_eigenvalue']<tolerance
    assert report['first_failure']['viscosity']==1e-5
    assert report['first_failure']['sound_cfl']==.001
    assert report['first_failure']['diagnostic']['spectral_radius']>1+1e-8
    assert report['status']=='rejected'


@pytest.mark.parametrize('n',[8,9])
@pytest.mark.parametrize('kind',[41,42])
@pytest.mark.parametrize('viscosity',[1e-5,1/1600])
def test_matched_viscous_reference_is_not_a_local_filter_admission(n,kind,viscosity):
    filtering,metric,report=review.energy_reference(n,kind,viscosity)
    assert report['metric_minimum_eigenvalue']>0
    assert report['viscosities_tested']==(viscosity,) and report['reference_viscosity']==viscosity
    assert report['endpoint_residual']<2e-10 and report['polynomial_residual']<2e-10
    tolerance=100*np.finfo(float).eps*max(1.,np.linalg.norm(metric,np.inf))
    assert report['contraction_maximum_eigenvalue']<tolerance
    assert report['matched_generator_energy_maximum_eigenvalue']<=report['generator_roundoff_reference']
    assert report['first_failure'] is None
    assert report['status']=='small dense reference only; not admitted'


def test_matched_metric_does_not_silently_override_fixed_deep_rows(tmp_path):
    original=native_filter_matrices(tmp_path/'native',(16,))[16]
    report=review.endpoint_dual_locality(original,42,1e-5,4)
    assert report['immutable_deep_defect']>100*report['numerical_uncertainty_estimate']
    assert not report['compatible_with_fixed_deep_rows']
