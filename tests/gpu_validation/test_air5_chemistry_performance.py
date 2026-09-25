from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

from check_air5_c5_reacting_tgv import ReactingTGVMetrics
from run_air5_chemistry_performance_gate import check_trace_metrics

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('shape,active', [
    ((63, 511, 7), ((1, 0, 0), (62, 511, 7))),
    ((24, 6, 6), ((0, 0, 0), (24, 6, 6))),
    ((1, 3, 3), ((0, 0, 0), (1, 3, 3))),
])
def test_reuse_certificate_excludes_all_faces_and_inactive_nodes(shape, active):
    nodes = np.indices(tuple(n+1 for n in shape))
    lo = np.maximum(active[0], 1)
    hi = np.minimum(active[1], np.array(shape)-1)
    reuse = np.ones(nodes.shape[1:], dtype=bool)
    for axis in range(3):
        reuse &= (nodes[axis] >= lo[axis]) & (nodes[axis] <= hi[axis])
    for axis in range(3):
        assert not np.any(reuse & ((nodes[axis] == 0) | (nodes[axis] == shape[axis])))
        assert not np.any(reuse & ((nodes[axis] < active[0][axis]) |
                                   (nodes[axis] > active[1][axis])))
    assert np.count_nonzero(reuse) == np.prod(np.maximum(hi-lo+1, 0))


def test_packed_extrema_match_separate_reductions():
    minima = np.array([[0., 1000., 300.], [1e-30, 1200., 200.],
                       [np.finfo(float).max]*3])
    maxima = np.array([[7000., 1000.], [6500., 2000.], [0., 0.]])
    packed = np.max(np.concatenate((-minima, maxima), axis=1), axis=0)
    np.testing.assert_array_equal(-packed[:3], minima.min(axis=0))
    np.testing.assert_array_equal(packed[3:], maxima.max(axis=0))
    status_counts = np.array([[0, 8, 4, 20, 6], [3, 1, 2, 3, 4]])
    assert status_counts.max(axis=0)[0] == 3


@pytest.mark.parametrize('shape,lo,hi', [
    ((6, 6, 6), (1, 1, 1), (5, 5, 5)),
    ((31, 511, 7), (1, 1, 1), (30, 510, 6)),
    ((8, 9, 10), (2, 3, 1), (5, 7, 8)),
])
def test_compact_shell_visits_exact_complement_once(shape, lo, hi):
    im, jm, km = shape
    il, jl, kl = lo
    nx, ny, nz = np.array(hi)-np.array(lo)+1
    ox, oy, oz = np.array(shape)+1-(nx, ny, nz)
    cx, cy = ox*(jm+1)*(km+1), nx*oy*(km+1)
    nodes = []
    for offset in range(cx+cy+nx*ny*oz):
        if offset < cx:
            i = offset % ox
            nodes.append((i if i < il else i+nx, (offset//ox) % (jm+1), offset//(ox*(jm+1))))
        elif offset < cx+cy:
            offset -= cx
            j = (offset//nx) % oy
            nodes.append((il+offset % nx, j if j < jl else j+ny, offset//(nx*oy)))
        else:
            offset -= cx+cy
            k = offset//(nx*ny)
            nodes.append((il+offset % nx, jl+(offset//nx) % ny, k if k < kl else k+nz))
    assert len(set(nodes)) == len(nodes)
    expected = {(i, j, k) for i in range(im+1) for j in range(jm+1) for k in range(km+1)
                if not all(lo[a] <= p <= hi[a] for a, p in enumerate((i, j, k)))}
    assert set(nodes) == expected


def test_reuse_is_single_call_and_failure_precedes_boundary():
    coupling = (ROOT/'src_gpu/chemistry_coupling_gpu.cuf').read_text()
    solver = (ROOT/'src_gpu/chemistry_solver_gpu.cuf').read_text()
    assert coupling.count('chemistry_recovered=reuse_primitives') == 2
    assert 'reuse=.false.' in solver
    assert 'lo=max([is,js,ks],1); hi=min([ie,je,ke],[im-1,jm-1,km-1])' in solver
    half = coupling.split('  subroutine air5_chemistry_half_step_gpu(')[1]
    assert half.index('if(global_status/=chemistry_status_ok)') < half.index('if(half_index==1)')
    assert 'any(global(1:2)/=-global(4:5))' in coupling


def test_prescribed_zero_is_allowed_but_negative_or_invalid_state_is_not():
    metrics = ReactingTGVMetrics(0., 1., 0., 0., 0., 6000., 6001., 1000., 1001., 1., 100., 100., 100.)
    assert check_trace_metrics(metrics)
    assert not check_trace_metrics(replace(metrics, minimum_species_density=-1e-300))
    assert not check_trace_metrics(replace(metrics, max_element_relative_change=3e-11))
    assert not check_trace_metrics(replace(metrics, minimum_temperature=4999.))
    assert not check_trace_metrics(replace(metrics, maximum_temperature=float('nan')))
