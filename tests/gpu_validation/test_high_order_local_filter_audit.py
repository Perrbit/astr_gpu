"""Accuracy diagnostic control on the actual unchanged native filter."""
from fractions import Fraction

import numpy as np
import pytest

from review_high_order_current_local_filter import (
    density_accuracy_errors, density_accuracy_training, project_density_accuracy)
from review_high_order_local_filter_audit import adiabatic_density_taylor, reference_state, single_mapping_accuracy
from test_high_order_wall_filter_coupling import native_filter_matrices
from test_high_order_wall_runtime import initial_fields


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    return native_filter_matrices(tmp_path_factory.mktemp('local_filter_audit')/'native', (32, 64, 128))


@pytest.mark.parametrize('kind', [41, 42, 421])
def test_native_single_mapping_control_does_not_hide_preasymptotic_failure(native, kind):
    parameters = [Fraction(x) for x in (1, 0, -1, 0, 1, 0, '-1/4', '-1/4')]
    report = single_mapping_accuracy(native, parameters, kind)
    # This nonlinear p/T field has a finite-grid cancellation with the native
    # 42/421 rows. Keep its failed 64->128 density order, not a false pass based
    # solely on exact sixth-order moments.
    assert report['passes_observed_order'] == (kind == 41), report
    if kind != 41:
        assert 5.21 < report['components']['rho']['orders'][1] < 5.22
    assert report['components']['rho_v']['zero_field']
    assert report['components']['rho_v']['orders'] is None
    assert all(not report['components'][name]['zero_field']
               for name in ('rho', 'rho_u', 'rho_w', 'rho_E'))
    assert [row['intervals'] for row in report['samples']] == [32, 64, 128]


def test_affine_density_constraint_matches_nonlinear_mapping_raw_error(native):
    parameters = [Fraction(x) for x in ('544/3743', '-1093/2657', '-152/2883',
                                      '822/1711', '1582/2067', '121/3956', '1083/580', '1083/580')]
    report = single_mapping_accuracy(native, parameters, 42)
    predicted = density_accuracy_errors(density_accuracy_training(native),
                                        np.array([float(x) for x in parameters[:7]]))
    actual = [row['near_strip_Linf'][0] for row in report['samples']]
    np.testing.assert_allclose(predicted, actual, rtol=2e-13, atol=0)
    assert not report['passes_observed_order']


def test_density_projection_enforces_fixed_field_order_without_changing_grid(native):
    start = np.array([.14533804782763482, -.4113662430646248, -.05272285373858461,
                      .4804207297787502, .7653603757322006, .03058643816886919, 1.8672412687475646])
    result, report = project_density_accuracy(density_accuracy_training(native), start)
    assert result.success, result.message
    assert min(report['observed_orders']) > 5.59
    assert report['maximum_scaled_constraint_violation'] < 1e-10
    assert np.linalg.norm(result.x-start) > 0


def test_low_observed_density_order_has_nonzero_sixth_order_term_and_opposite_eighth_term():
    parameters = [Fraction(x) for x in ('544/3743', '-1093/2657', '-152/2883',
                                      '822/1711', '1582/2067', '121/3956', '1083/580', '1083/580')]
    report = adiabatic_density_taylor(parameters)
    assert report['exact_leading_coefficient_in_angle_h'] != '0'
    middle = report['samples'][1]
    assert middle['signed_terms'][6] < 0 < middle['signed_terms'][8]
    assert abs(middle['signed_error']) < .1*abs(middle['signed_terms'][6])+.1*abs(middle['signed_terms'][8])
    assert abs(middle['remaining_series_tail']) < 5e-17


@pytest.mark.parametrize('kind', [41, 42, 421])
def test_approved_reference_reuses_existing_fields_and_energy_normalization(kind):
    import mpmath as mp
    _, fields = initial_fields((2, 32, 2), manufactured=True, wall_type=kind)
    with mp.workdps(80):
        for j in (0, 1, 4, 16, 31, 32):
            state, cv = reference_state(mp.mpf(j)/32, kind, 'wall-manufactured')
            actual = np.array([float(x) for x in state])
            rho = fields['ro'][0, j, 0]
            velocity = np.array([fields[f'u{k}'][0, j, 0] for k in (1, 2, 3)])
            energy = 2.5*rho*fields['t'][0, j, 0]/.126+.5*rho*np.dot(velocity, velocity)
            expected = np.r_[rho, rho*velocity, energy]
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
            assert abs(float(cv)-2.5/.126) < 4e-15


def test_approved_density_reference_has_no_original_sixth_eighth_cancellation():
    parameters = [Fraction(x) for x in ('544/3743', '-1093/2657', '-152/2883',
                                      '822/1711', '1582/2067', '121/3956', '1083/580', '1083/580')]
    report = adiabatic_density_taylor(parameters, 'wall-manufactured')
    assert report['angle_factor_pi'] == 1
    assert report['exact_leading_coefficient_in_angle_h'] != '0'
    middle = report['samples'][1]
    assert abs(middle['signed_terms'][8]/middle['signed_terms'][6]) < .1
    assert abs(middle['signed_error']/middle['signed_terms'][6]-1) < .1


@pytest.mark.parametrize('kind', [41, 42, 421])
def test_single_mapping_state_diagnostics_do_not_reset_wall_density(native, kind):
    parameters = [Fraction(x) for x in (1, 0, -1, 0, 1, 0, '-1/4', '-1/4')]
    report = single_mapping_accuracy(native, parameters, kind, 'wall-manufactured')
    for sample in report['samples']:
        assert sample['minimum_density'] > 0
        assert sample['minimum_temperature'] > 0
        assert sample['wall_density_change'] == 0
        assert sample['wall_normal_velocity_maximum'] == 0
        assert sample['thermal_wall_constraint_residual'] < 1e-70
