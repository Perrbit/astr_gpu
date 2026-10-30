"""Selected-eigenpair diagnostic checks; no physical or candidate admission."""
from fractions import Fraction

import mpmath as mp
import numpy as np
from scipy.linalg import eig

from review_high_order_common_metric import assemble, exact_blocks
from review_high_order_current_local_filter import local_blocks
from review_high_order_filter_eigenpair import (
    exact_filter, refine, sparse_rows, stage_action, transpose_action,
    tangential_unit_certificate, unit_certificates, unit_rank_certificate)
from test_high_order_wall_filter_coupling import native_filter_matrices, rk3_filtered_map


def test_bordered_refinement_resolves_small_positive_growth():
    with mp.workdps(80):
        a = 1+mp.mpf('1e-14')
        b = mp.mpf('0.000004')
        matrix = np.array([[float(a), -float(b)], [float(b), float(a)]])
        values, vectors = eig(matrix)
        index = int(np.argmax(values.imag))
        action = lambda v: [a*v[0]-b*v[1], b*v[0]+a*v[1]]
        lam, _, history = refine(action, matrix, values[index], vectors[:, index], 80)
        assert abs(lam-(a+mp.j*b)) < mp.mpf('1e-65')
        assert mp.mpf(history[-1]) < mp.mpf('1e-65')
        assert abs(lam)-1 > mp.mpf('1e-14')


def test_modular_rank_does_not_hide_a_unit_jordan_chain():
    prime = 1000003
    basis = np.array([[1], [0]], dtype=np.int64)
    good = np.array([[1, 0], [0, pow(2, -1, prime)]], dtype=np.int64)
    bad = np.array([[1, 1], [0, 1]], dtype=np.int64)
    assert unit_rank_certificate(good, basis, prime)['unit_semisimple_certified']
    rejected = unit_rank_certificate(bad, basis, prime)
    assert not rejected['unit_semisimple_certified']
    assert rejected['rank_delta_mod_prime'] == 1
    assert rejected['rank_delta_squared_mod_prime'] == 0


def test_exact_filter_and_adjoint_match_approved_stage_map(tmp_path):
    original = native_filter_matrices(tmp_path/'native', (8,))[8]
    parameters = [Fraction(x) for x in (1, 0, -1, 0, 1, 0, '-1/4', '-1/4')]
    f = exact_filter(original, parameters, 42)
    a, _ = assemble(exact_blocks(8, 42))
    blocks = local_blocks(original, 42, np.array([1., 0., -1., 0., 1., 0., -.25]))
    # Exact rational multiplication and rounded Neumann projection differ at
    # FP64 roundoff; the diagnostic does not assert bitwise evaluation order.
    np.testing.assert_allclose(np.array(f, float), blocks[0][1], rtol=0,
                               atol=np.finfo(float).eps*np.linalg.norm(blocks[0][1], np.inf))
    np.testing.assert_allclose(np.array(a, float), blocks[0][2], rtol=0, atol=1e-13)
    for cfl in ('0', '0.0001'):
        report = unit_certificates(original, parameters, 42, '0', cfl)
        assert report['exact_fixed_basis_verified']
        assert report['unit_semisimple_certified']
        assert report['explicit_fixed_dimension'] == (14 if cfl == '0' else 5)
    for kind in (41, 42, 421):
        for nu in ('0', '1/1600'):
            record = tangential_unit_certificate(original, parameters, kind, nu, '0.0001')
            assert record['unit_semisimple_certified']
            assert record['explicit_fixed_dimension'] == (4 if nu == '0' else int(kind == 421))
    with mp.workdps(80):
        ar, fr = sparse_rows(a), sparse_rows(f)
        at, ft = sparse_rows(a.T), sparse_rows(f.T)
        x = [mp.mpf(j)/23 for j in range(23)]
        y = [mp.mpf(23-j)/31 for j in range(23)]
        dt = mp.mpf('0.001')
        action = stage_action(ar, fr, dt, x)
        adjoint = transpose_action(at, ft, dt, y)
        assert abs(mp.fsum(u*v for u, v in zip(y, action))-
                   mp.fsum(u*v for u, v in zip(adjoint, x))) < mp.mpf('1e-70')
        machine = rk3_filtered_map(np.array(a, float), np.array(f, float), float(dt))
        np.testing.assert_allclose([float(z) for z in action], machine@np.array(x, float),
                                   rtol=0, atol=1e-14)
