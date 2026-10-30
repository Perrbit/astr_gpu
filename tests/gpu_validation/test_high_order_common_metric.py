"""Check exact common-metric reasoning against the accepted frozen space blocks."""
import numpy as np
import pytest

from review_high_order_common_metric import exact_blocks,assemble,exact_common_space
from review_high_order_p6_filter_compatibility import current_normal


@pytest.mark.parametrize('n',[8,9])
@pytest.mark.parametrize('kind',[41,42,421])
def test_exact_blocks_match_the_current_floating_reference(n,kind):
    a,diffusion=assemble(exact_blocks(n,kind))
    actual=np.array(a,float);viscous=np.array(diffusion,float)
    reference=current_normal(n,0.,kind)
    tolerance=100*np.finfo(float).eps*max(1.,np.linalg.norm(reference,np.inf))
    np.testing.assert_allclose(actual,reference,rtol=0,atol=tolerance)
    reference=current_normal(n,1.,kind)-reference
    tolerance=100*np.finfo(float).eps*max(1.,np.linalg.norm(reference,np.inf))
    np.testing.assert_allclose(viscous,reference,rtol=0,atol=tolerance)


@pytest.mark.parametrize('n',[8,9])
def test_adiabatic_common_metric_has_an_exact_indefinite_restriction(n):
    metrics,report=exact_common_space(n,42)
    proof=report['parity_certificate']
    assert metrics and report['common_kernel_dimension']==0
    assert proof['restriction_span_dimension']==1
    assert proof['exact_scalar_multiple_check']
    assert set(proof['witnesses'])=={'positive','negative'}
    assert report['positive_definite_common_metric_impossible']


def test_isothermal_case_is_not_rejected_by_the_adiabatic_certificate():
    _,report=exact_common_space(8,41)
    assert not report['parity_certificate']['indefinite_one_dimensional_restriction']
    assert not report['positive_definite_common_metric_impossible']
