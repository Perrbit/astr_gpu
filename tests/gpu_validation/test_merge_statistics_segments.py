"""Independent weighted observations and bounded file-segment checks."""

import importlib.util
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/output'))
spec = importlib.util.spec_from_file_location('statistics_merge', ROOT / 'scripts/output/merge_statistics_segments.py')
merge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge)


def independent_moments(weights, density, velocity):
    result = np.zeros((1, 26), dtype=np.float64)
    for slot, mean, central, w in ((0, 2, 8, weights), (1, 5, 17, weights * density)):
        result[0, slot] = w.sum()
        if not w.sum():
            continue
        average = np.sum(w[:, None] * velocity, axis=0) / w.sum()
        delta = velocity - average
        moment = np.einsum('n,ni,nj->ij', w, delta, delta)
        result[0, mean:mean + 3] = average
        result[0, central:central + 9] = moment.ravel(order='F')
    return result


def test_weighted_central_merge_is_not_rms_average():
    w = np.array([1., 3., 2., 4.])
    rho = np.array([1., 2., 4., .5])
    u = np.array([[1., -2., 3.], [2., 1., -1.], [8., 3., 0.], [-2., 4., 5.]])
    expected = independent_moments(w, rho, u)
    a = independent_moments(w[:2], rho[:2], u[:2])
    b = independent_moments(w[2:], rho[2:], u[2:])
    merge.merge_moments(a, b)
    np.testing.assert_allclose(a, expected, rtol=2e-15, atol=2e-14)
    assert a[0, 0] == 10 and a[0, 1] == 17


def test_zero_coverage_does_not_contribute_unused_large_values():
    state = np.zeros((1, 26))
    state[0, :2] = 1
    state[0, 2:8] = -1e308
    original = state.copy()
    unused = np.full((1, 26), 1e308)
    unused[0, :2] = 0
    merge.merge_moments(state, unused)
    np.testing.assert_array_equal(state, original)


@pytest.mark.parametrize('defect', ['nan', 'weight', 'mass', 'variance', 'symmetry'])
def test_invalid_moments_rejected(defect):
    values = independent_moments(np.ones(2), np.ones(2), np.array([[0., 1., 2.], [2., 3., 4.]]))
    if defect == 'nan':
        values[0, 5] = np.nan
    elif defect == 'weight':
        values[0, 0] = -1
    elif defect == 'mass':
        values[0, 1] = 0
    elif defect == 'variance':
        values[0, 8] = -1
    else:
        values[0, 9] += 1
    with pytest.raises(ValueError):
        merge.merge_moments(np.zeros((1, 26)), values)


def word(value, width):
    return np.asarray(value, dtype=f'<f{width}').view(f'<i{width}').item()


def make_segment(root, name, width, start_step, end_step, start, end, moments, *, parent=None):
    run = root / name
    resources = run / 'resources'
    batch = run / 'checkpoints' / f'step{end_step:012d}'
    resources.mkdir(parents=True)
    batch.mkdir(parents=True)
    (resources / 'input.txt').write_text('tgv\n2,2,2\nt,t,t\nt,t,t,f,f,f,t,f,t\nf\ngrid.h5\n')
    (batch / 'COMPLETE').write_text('ASTR_COMPLETE_2\n')
    (batch / 'MANIFEST').write_text('ASTR_CHECKPOINT_BUNDLE 2\n1\nstatistics.h5\n')
    (batch / 'RESOURCES').write_text('ASTR_SHARED_RESOURCES 2\n1\ninput.txt\n')
    dtype = np.dtype(f'<f{width}')
    shape = (3, 3, 3)
    point = np.zeros(34, dtype=dtype)
    point[:4] = [0, 10, end, 1]
    point[7:33] = moments[0]
    point[33] = 1
    parent_width, parent_time = parent or (width, start)
    record = np.array([1, 1, start_step, word(start, width), word(0, width), word(10, width),
                       parent_width, word(parent_time, parent_width), 7, 0, 1, 1, 0, *range(1, 9)], dtype='<i8')
    with h5py.File(batch / 'statistics.h5', 'x') as h:
        h['identity'] = np.array([0x4153545251533031, 3, 1, 3, 3, 3, 34, 1, end_step,
            word(end, width), word(.25, width), word(.25, width), 3], dtype='<i8')
        h['numerical_kind_bytes'] = np.array([width], dtype='<i8')
        h['partitions'] = np.array([0, 0, 0, 2, 2, 2, 0, 0], dtype='<i8')
        h['rank_extras'] = np.empty(0, dtype=dtype)
        h['statistics_segment'] = record
        metadata = [1, 0, end_step, word(end, width), *point.view(f'<i{width}').astype('<i8')]
        h['metadata'] = np.array(metadata, dtype='<i8')
        for component in range(34):
            h[f'q{component + 1:04d}'] = np.full(shape, point[component], dtype=dtype)
    return batch / 'statistics.h5'


@pytest.fixture
def segments(tmp_path):
    a = independent_moments(np.array([1., 1.]), np.array([1., 2.]), np.array([[0., 2., 0.], [2., 0., 1.]]))
    b = independent_moments(np.array([1., 2.]), np.array([3., .5]), np.array([[4., 1., 0.], [-1., 3., 2.]]))
    x = make_segment(tmp_path, 'first', 4, 0, 2, 0, 2, a)
    y = make_segment(tmp_path, 'second', 8, 2, 5, 2, 5, b, parent=(4, 2))
    return x, y


def test_native_segments_outputs_and_source_preservation(segments, tmp_path):
    original = {p: (p.stat().st_size, p.stat().st_mtime_ns)
                for root in (tmp_path / 'first', tmp_path / 'second') for p in root.rglob('*') if p.is_file()}
    result = merge.merge_segments(segments, tmp_path / 'merged', buffer_bytes=2048)
    assert result['source_precisions'] == ['fp32', 'fp64']
    assert result['continuous_step_coverage']
    assert result['covered_point_count'] == result['point_count'] == 8
    assert result['reserved_controlled_array_bytes'] <= 2048
    with h5py.File(tmp_path / 'merged/statistics.h5') as f:
        assert f['point/duration'].shape == (2, 2, 2)
        assert np.all(f['point/covered'][:] == 1)
        assert f['point/mean_favre'].dtype == np.dtype('<f8')
        assert f['regional'].attrs['density_weight'] == 'dimensionless unity'
        assert np.all(f['point/duration'][:] == 5)
        stress = f['point/density_stress'][:]
        central = f['point/central_favre'][:]
        np.testing.assert_array_equal(stress, central / 5)
        cov = f['point/covariance_reynolds'][:]
        np.testing.assert_array_equal(f['point/rms_reynolds'][:],
                                      np.sqrt(np.diagonal(cov, axis1=-2, axis2=-1)))
    assert (tmp_path / 'merged/COMPLETE.json').is_file()
    assert all((p.stat().st_size, p.stat().st_mtime_ns) == value for p, value in original.items())
    with pytest.raises(FileExistsError):
        merge.merge_segments(segments, tmp_path / 'merged')


@pytest.mark.parametrize('defect', ['duplicate', 'overlap', 'parent', 'unknown', 'mesh', 'input', 'window'])
def test_segment_mismatch_rejected(segments, tmp_path, defect):
    a, b = segments
    if defect == 'duplicate':
        paths = [a, a]
    else:
        paths = [a, b]
        with h5py.File(b, 'r+') as h:
            if defect == 'overlap':
                h['statistics_segment'][2] = 1
            elif defect == 'parent':
                h['statistics_segment'][7] = word(3, 4)
            elif defect == 'unknown':
                h['statistics_segment'][1] = 0
            elif defect == 'mesh':
                h['statistics_segment'][13] = 99
            elif defect == 'window':
                h['statistics_segment'][5] = word(11, 8)
        if defect == 'input':
            p = tmp_path / 'second/resources/input.txt'
            p.write_text(p.read_text().replace('2,2,2', '4,4,4'))
    with pytest.raises(ValueError):
        merge.merge_segments(paths, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()


def test_gap_is_reported_not_filled(tmp_path):
    moments = independent_moments(np.ones(2), np.ones(2), np.array([[0., 1., 2.], [2., 3., 4.]]))
    a = make_segment(tmp_path, 'a', 8, 0, 2, 0, 2, moments)
    b = make_segment(tmp_path, 'b', 8, 4, 6, 4, 6, moments)
    report = merge.merge_segments([a, b], tmp_path / 'gap')
    assert not report['continuous_step_coverage'] and report['seams'][0]['gap']
    with h5py.File(tmp_path / 'gap/statistics.h5') as f:
        assert np.all(f['point/duration'][:] == 4)


def test_clock_quantization_is_an_explicit_map_not_a_gap(tmp_path):
    moments = independent_moments(np.ones(2), np.ones(2), np.zeros((2, 3)))
    a = make_segment(tmp_path, 'a', 8, 0, 2, 0, .1, moments)
    b = make_segment(tmp_path, 'b', 4, 2, 4, np.float32(.1), .2, moments, parent=(8, .1))
    report = merge.merge_segments([a, b], tmp_path / 'clock')
    assert report['continuous_step_coverage']
    assert report['seams'][0]['clock_delta'] != 0
    assert not report['raw_clock_coordinates_continuous']


def test_zero_coverage_file_has_explicit_mask(tmp_path):
    path = make_segment(tmp_path, 'zero', 4, 0, 0, 0, 0, np.zeros((1, 26)))
    report = merge.merge_segments([path], tmp_path / 'zero_result')
    assert report['sources'][0]['clipped_interval'] is None
    assert report['covered_point_count'] == 0
    with h5py.File(tmp_path / 'zero_result/statistics.h5') as h:
        assert np.all(h['point/covered'][:] == 0)
        assert np.all(h['point/rms_favre'][:] == 0)


@pytest.mark.parametrize('changed', [False, True])
def test_frozen_grid_resource_identity(segments, tmp_path, changed):
    for index, path in enumerate(segments):
        with h5py.File(path, 'r+') as h:
            h['statistics_segment'][12] = 1
        resources = path.parent.parent.parent / 'resources'
        (resources / 'grid.h5').write_bytes(b'different' if changed and index else b'synthetic frozen grid')
        (path.parent / 'RESOURCES').write_text('ASTR_SHARED_RESOURCES 2\n2\ninput.txt\ngrid.h5\n')
    if changed:
        with pytest.raises(ValueError, match='resources differ'):
            merge.merge_segments(segments, tmp_path / 'grid_result')
    else:
        report = merge.merge_segments(segments, tmp_path / 'grid_result')
        assert report['resource_compare_bytes'] > 0


def test_invalid_buffer_precedes_source_reads(tmp_path):
    with pytest.raises(ValueError, match='1024'):
        merge.merge_segments([tmp_path / 'absent.h5'], tmp_path / 'invalid_buffer', buffer_bytes=0)
    assert not (tmp_path / 'invalid_buffer').exists()


@pytest.mark.parametrize('flags', ['t,t,t,f,f,f,f,f', 't,t,t,f,f,f,f'])
def test_optional_cpu_device_selector_does_not_change_physics(segments, tmp_path, flags):
    source = tmp_path / 'second/resources/input.txt'
    source.write_text(source.read_text().replace('t,t,t,f,f,f,t,f,t', flags))
    result = merge.merge_segments(segments, tmp_path / 'eight_flags')
    assert result['continuous_step_coverage']
