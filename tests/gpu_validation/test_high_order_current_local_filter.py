"""Research-tool consistency, not admission of any filter coefficients."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from review_high_order_current_local_filter import (
    affine_training, local_blocks, objective, parameters7, phase_comparison, review)
from review_high_order_p6_filter_compatibility import current_normal
from test_high_order_wall_filter_coupling import native_filter_matrices
from test_high_order_wall_filter_candidates import sixth_order_family


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    return native_filter_matrices(tmp_path_factory.mktemp('current_local')/'native', (8, 9, 16))


@pytest.mark.parametrize('kind', [41, 42, 421])
def test_affine_cache_reproduces_current_operators_and_filter(native, kind):
    values = np.array([.5, -.25, -1., .125, 1., -.5, .25])
    cached = affine_training({8: native[8]}, kind)
    direct = local_blocks(native[8], kind, values)
    for n, name, nu, _, base, increments, a in cached:
        block = next(block for block in direct if block[0] == name)
        np.testing.assert_allclose(base+np.tensordot(values, increments, axes=1), block[1],
                                   rtol=0, atol=3e-15)
        np.testing.assert_array_equal(a, block[2]+nu*block[3])
    np.testing.assert_array_equal(direct[0][2], current_normal(8, 0., kind))
    np.testing.assert_array_equal(sixth_order_family(native[8], parameters7(values)),
                                  sixth_order_family(native[8], parameters7(values))[::-1, ::-1])


def test_native_coefficients_still_fail_original_gate(native):
    baseline = np.array([1., 0., -1., 0., 1., 0., -.25])
    assert objective(affine_training({8: native[8]}, 42), baseline) > .1
    records, failure = review(native, 42, baseline)
    assert failure is records[-1] and failure['intervals'] == 8
    assert failure['diagnostic']['eigenpair_residual'] < 1e-12
    assert not failure['passes']


def test_phase_report_distinguishes_no_filter_single_filter_and_actual_rk(native):
    baseline = np.array([1., 0., -1., 0., 1., 0., -.25])
    maps = phase_comparison(native[8], 42, baseline, 0., .2)['maps']
    assert abs(maps['filter_alone']['spectral_radius']-1) < 1e-12
    assert abs(maps['unfiltered_rk']['spectral_radius']-1) < 1e-12
    assert abs(maps['actual_stage_rk']['spectral_radius']-1.0055635020650118) < 2e-13
    assert set(maps) == {'filter_alone', 'unfiltered_rk', 'one_filter_after_rk', 'actual_stage_rk'}
    assert all(len(record['eigenvalues']) == 23 for record in maps.values())


@pytest.mark.parametrize('n', [8, 9, 16])
def test_local_family_keeps_endpoints_overlap_and_deep_rows(native, n):
    original = native[n]
    values = np.array([-.5, -.25, -1., .125, 1., -.5, .25])
    f = sixth_order_family(original, parameters7(values))
    np.testing.assert_array_equal(f, f[::-1, ::-1])
    np.testing.assert_array_equal(f[[0, n]], original[[0, n]])
    np.testing.assert_array_equal(f[5:n-4], original[5:n-4])
    for degree in range(6):
        x = np.arange(n+1, dtype=float)/n
        np.testing.assert_allclose(f@x**degree, x**degree, rtol=0, atol=3e-15)


def test_review_only_cli_preserves_known_failure_and_does_not_fit(native, tmp_path):
    native[8].ravel(order='F').tofile(tmp_path/'filter_8.bin')
    source = tmp_path/'design.json'
    source.write_text(json.dumps(dict(floating_parameters=[1, 0, -1, 0, 1, 0, -.25, -.25],
                                     rational_parameters=['1', '0', '-1', '0', '1', '0', '-1/4', '-1/4'])))
    output = tmp_path/'review'
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('review_high_order_current_local_filter.py')),
                             '--native', str(tmp_path), '--output', str(output), '--start', str(source),
                             '--kind', '42', '--review-only', '--intervals', '8'],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr
    report = json.loads((output/'report.json').read_text())
    assert report['coefficient_search'] is False
    assert report['status'] == 'rejected'
    assert report['first_failure']['intervals'] == 8
