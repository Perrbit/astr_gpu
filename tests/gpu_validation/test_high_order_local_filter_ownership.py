"""Exact stencil ownership and scalar/full algebra; no MPI or CFD execution."""
from fractions import Fraction
import json

import pytest
import sympy as sp

from review_high_order_filter_eigenpair import exact_filter
from review_high_order_wall_filter_design import exact_rows
from test_high_order_wall_filter_coupling import native_filter_matrices


COEFFICIENTS = {
    41: ('323/2898', '-72/1639', '701/2490', '1461/3958', '827/2383', '174/3949', '2413/3032', '2413/3032'),
    42: ('544/3743', '-1093/2657', '-152/2883', '822/1711', '1582/2067', '121/3956', '1083/580', '1083/580'),
}


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    return native_filter_matrices(tmp_path_factory.mktemp('local_filter_ownership')/'native', (8, 9, 14, 16, 32, 64))


@pytest.mark.parametrize('kind', [41, 42])
@pytest.mark.parametrize('cuts', [(0, 8), (0, 9), (0, 7, 14), (0, 8, 16),
                                  (0, 8, 16, 24, 32), (0, 16, 32, 48, 64)])
def test_private_face_rows_fit_existing_halo_and_shared_nodes_match_exactly(native, kind, cuts, tmp_path):
    n = cuts[-1]
    parameters = [Fraction(x) for x in COEFFICIENTS[kind]]
    scalar = exact_filter(native[n], parameters, kind)[:n+1, :n+1]
    near = {row['row']: [sp.Rational(x) for x in row['coefficients']]
            for row in exact_rows(parameters)}
    central = [sp.Rational(float(x)) for x in native[32][16, 11:22]]
    replicas = {}
    for start, end in zip(cuts[:-1], cuts[1:]):
        available = set(range(max(0, start-5), min(n, end+5)+1))
        for index in range(start, end+1):
            row = sp.zeros(1, n+1)
            if index in (0, n):
                row[index] = 1
            elif start == 0 and index in near:
                row[0, :9] = sp.Matrix([near[index]])
            elif end == n and n-index in near:
                row[0, n-8:] = sp.Matrix([near[n-index][::-1]])
            else:
                for offset, value in zip(range(-5, 6), central):
                    assert 0 <= index+offset <= n
                    row[index+offset] = value
            assert row == scalar[index, :]
            assert {j for j, value in enumerate(row) if value} <= available
            if index in replicas:
                assert row == replicas[index]
            replicas[index] = row
    # Each component reads only its own pre-filter line. Wall thermodynamics
    # must wait until all five components have completed this mapping.
    inputs = sp.Matrix([[sp.Rational((j+1)**(k+6), (n+1)**(k+6))
                         for k in range(5)] for j in range(n+1)])
    full = scalar*inputs
    componentwise = inputs.copy()
    for k in range(5):
        componentwise[:, k] = scalar*componentwise[:, k]
    assert full == componentwise
    (tmp_path/'ownership.json').write_text(json.dumps(dict(
        bctype=kind, global_intervals=n, partition_cuts=cuts, halo_layers=5,
        source='actual native central rows plus exact candidate physical rows',
        exact_shared_node_match=True, scalar_full_exact_algebra=True,
        scope='index/algebra proof only; not executed MPI or GPU equivalence'), indent=2))
