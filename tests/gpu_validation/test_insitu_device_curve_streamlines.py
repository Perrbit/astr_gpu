"""X4 CURVE runtime gate; component inverse/affine oracles remain separate."""
import csv
import json
import re

import h5py
import numpy as np
from PIL import Image
import pytest

from run_output_restart_validation import run_case, compare_fields
from test_output_insitu_restart import configuration as host_configuration
from test_insitu_device_curve_surface import configuration as surface_configuration
from test_insitu_device_curve_wall_fields import current_arguments
from test_insitu_curve_derivatives import ROOT, FINAL
from test_insitu_device_products import cross_backend_statistics_difference
from test_insitu_is4 import compare_numerical

TOPOLOGIES = ((1, 'x'), (2, 'x'), (2, 'y'), (2, 'z'))
PRODUCTS = {'instantaneous_streamlines', 'crossing_streamlines'}
MEANS = {'mean_reynolds_streamlines', 'mean_favre_streamlines'}


def configuration(pipeline, mode, mean=True):
    text = surface_configuration(pipeline, mode).replace("products='q_surface'", "products='streamlines'")
    if mean:
        text = text.replace('statistics=.false.', 'statistics=.true.,mean_streamline_render=.true.')
    return text


def validate(case, ranks, pipeline, steps=(2, 4), mean=True, statistics=None, audit=True):
    if statistics is None:
        statistics = mean
    log = (case / 'run.log').read_text()
    frames = re.findall(r'ASTR_INSITU_CURVE_TRACE_FRAME rank=(\d+) step=(\d+) '
        r'physical_coordinates=1 physical_hexahedral_interpolation=1 '
        r'field_download_bytes=0 geometry_host_bytes=0', log)
    assert len(frames) == ranks * len(steps), log
    expected = PRODUCTS | (MEANS if mean else set())
    audits = re.findall(r'ASTR_INSITU_RESIDENT_AUDIT rank=(\d+) step=(\d+) product=(\S+) '
        r'field_maxabs=(\S+) display_maxabs=(\S+) points=(\d+)', log)
    active = [row for row in audits if row[2] in expected]
    if audit:
        assert len(active) == ranks * len(steps) * len(expected), log
        assert all(np.isfinite(float(row[3])) and float(row[3]) <= 2e-10 and float(row[4]) == 0.
                   for row in active), active
    else:
        assert not active
    endpoints = re.findall(r'ASTR_INSITU_RESIDENT_CROSSING rank=\d+ endpoint_maxabs=(\S+)', log)
    assert len(endpoints) == ranks * len(steps) and all(float(v) <= 2e-10 for v in endpoints), log
    owners = re.findall(r'owner_query_read_bytes=(\d+)', log)
    assert len(owners) == ranks*len(steps)*len(expected) and all(int(v) >= 256 and int(v) % 128 == 0 for v in owners), log
    for step in steps:
        cells = dict.fromkeys(expected, 0)
        for rank in range(ranks):
            record = json.loads((case / f'outdat/render/mesh_step{step:08d}_rank{rank}.json').read_text())
            assert record['geometry_host_bytes'] == 0 and record['rendering_pipeline'] == pipeline
            assert set(record['products']) == expected
            if mean:
                coverage = record['statistics']
                assert coverage['statistics_duration'] == pytest.approx(step*.001-.0005, abs=2e-15)
                assert coverage['statistics_window_start'] == pytest.approx(.0005, abs=2e-15)
                assert coverage['statistics_window_end'] == pytest.approx(step*.001, abs=2e-15)
            for name in expected:
                local = record['products'][name]['local_cells']
                assert local >= 0, (rank, name, record)
                cells[name] += local
        assert all(value > 0 for value in cells.values()), cells
        for name in expected:
            path = case / f'outdat/render/{name}.step{step:08d}.jpeg'
            assert path.with_suffix('.eps').is_file()
            with Image.open(path) as image:
                pixels = np.asarray(image.convert('RGB'))
                assert pixels.shape == (600, 800, 3)
                assert np.count_nonzero(np.any(pixels < 220, axis=-1)) > 500, path
    assert not list((case / 'outdat/render').glob('*.vtp'))
    assert not list((case / 'outdat/render').glob('sample.*.bin'))
    assert not list((case / 'outdat/render').glob('curve_trace_oracle.*'))
    for rank in range(ranks):
        with (case / f'outdat/render/resources.rank{rank:08d}.csv').open() as stream:
            stream.readline()
            resources = list(csv.DictReader(stream))
        stages = ['baseline', 'initialized'] + ['frame_before', 'frame_after'] * len(steps)
        stages += ['finalize_before', 'finalize_after']
        if statistics:
            stages.append('statistics_retained_device')
        assert [row['stage'] for row in resources] == stages + ['session_released']
        assert max(int(row['host_increment_bytes']) for row in resources) <= 4*1024**3
        assert max(int(row['device_increment_bytes']) for row in resources) <= 2*1024**3
        assert min(int(row['device_free_bytes']) for row in resources) >= 1024**3


def compare_statistics(actual, expected, mapping):
    error = cross_backend_statistics_difference(actual, expected)
    if mapping == 'y-wavy':
        # The nonperiodic upper wall is owned, unlike the periodic x/z endpoints.
        with h5py.File(actual) as a, h5py.File(expected) as b:
            for c in range(1, 35):
                name = f'q{c:04d}'
                first, second = a[name][:32, 32, :32], b[name][:32, 32, :32]
                assert np.isfinite(first).all() and np.isfinite(second).all()
                difference = float(np.max(abs(first-second)))
                assert difference <= 2e-10, (name, difference)
                error = max(error, difference)
    return error


@pytest.fixture(scope='module')
def references(tmp_path_factory):
    cases = {}
    def get(mapping, ranks, axis):
        key = mapping, ranks, axis
        if key not in cases:
            root = tmp_path_factory.mktemp('curve_trace_reference')
            config = host_configuration(render=False, interval=2)
            if mapping == 'y-wavy':
                config = config.replace('&insitu_run\n', "&insitu_run\n products='channel_walls',\n")
            cpu, _ = run_case(current_arguments(root, 'cpu', axis, samples=False), ROOT,
                'cpu', ranks, 'cpu', 4, grid='32,32,32', tgv_mapping=mapping,
                insitu_config=config, checkpoint_interval=1)
            plain, _ = run_case(current_arguments(root, axis=axis, samples=False), ROOT,
                'gpu', ranks, 'off', 4, grid='32,32,32', tgv_mapping=mapping, checkpoint_interval=1)
            cases[key] = cpu, plain
        return cases[key]
    return get


@pytest.mark.parametrize('mapping', ('periodic', 'y-wavy'))
@pytest.mark.parametrize('ranks,axis', TOPOLOGIES, ids=('single', 'x', 'y', 'z'))
@pytest.mark.parametrize('mode', ('pinned', 'device-aware'))
@pytest.mark.parametrize('pipeline', ('standard-device', 'direct-device'))
def test_runtime_images_fields_restart(tmp_path, mapping, ranks, axis, mode, pipeline, references, record_property):
    cpu, plain = references(mapping, ranks, axis)
    args = current_arguments(tmp_path, axis=axis, samples=False)
    args.runtime_timeout_seconds = 600
    config = configuration(pipeline, mode)
    kwargs = dict(grid='32,32,32', tgv_mapping=mapping, checkpoint_interval=1,
                  insitu_config=config, postprocess_transport=mode, resident_audit=True)
    actual, size = run_case(args, ROOT, 'gpu', ranks, 'trace', 4, **kwargs)
    validate(actual, ranks, pipeline)
    record_property('cpu_gpu_state_maxabs', compare_numerical(actual / FINAL / 'state.h5', cpu / FINAL / 'state.h5'))
    record_property('cpu_gpu_statistics_maxabs', compare_statistics(actual / FINAL / 'statistics.h5',
                                                                  cpu / FINAL / 'statistics.h5', mapping))
    compare_fields(actual / FINAL / 'state.h5', plain / FINAL / 'state.h5')
    source = actual / 'outdat/new/checkpoints/step000000000003'
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    resumed, _ = run_case(args, ROOT, 'gpu', ranks, 'resumed', 4, restore=source, **kwargs)
    validate(resumed, ranks, pipeline, steps=(4,))
    for name in ('state.h5', 'statistics.h5'):
        compare_fields(actual / FINAL / name, resumed / FINAL / name)
    for name in ('control.bin', 'insitu_control.bin'):
        assert (actual / FINAL / name).read_bytes() == (resumed / FINAL / name).read_bytes()
    for name in PRODUCTS | MEANS:
        for suffix in ('jpeg', 'eps'):
            path = f'outdat/render/{name}.step00000004.{suffix}'
            assert (actual / path).read_bytes() == (resumed / path).read_bytes()
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    record_property('directory_bytes', size)


@pytest.mark.parametrize('mapping', ('periodic', 'y-wavy'))
def test_mean_default_off(tmp_path, mapping):
    args = current_arguments(tmp_path, samples=False)
    case, _ = run_case(args, ROOT, 'gpu', 1, 'default', 2, grid='32,32,32', tgv_mapping=mapping,
        insitu_config=configuration('standard-device', 'pinned', mean=False), postprocess_transport='pinned',
        resident_audit=True, checkpoint_enabled=False, enabled=False)
    validate(case, 1, 'standard-device', steps=(2,), mean=False)
    assert not list((case / 'outdat/render').glob('mean_*'))


@pytest.mark.parametrize('mapping', ('periodic', 'y-wavy'))
def test_zero_coverage_no_instantaneous_substitution(tmp_path, mapping):
    args = current_arguments(tmp_path, axis='y', samples=False)
    config = configuration('standard-device', 'pinned').replace('statistics_window=0.0005,0.0115',
                                                                'statistics_window=1.d0,2.d0')
    assert 'statistics_window=1.d0,2.d0' in config
    case, _ = run_case(args, ROOT, 'gpu', 2, 'uncovered', 2, grid='32,32,32', tgv_mapping=mapping,
        insitu_config=config, postprocess_transport='pinned', resident_audit=True,
        checkpoint_enabled=False, enabled=False)
    validate(case, 2, 'standard-device', steps=(2,), mean=False, statistics=True)
    assert not list((case / 'outdat/render').glob('mean_*'))


def test_restart_mean_selection_identity(tmp_path):
    args = current_arguments(tmp_path, samples=False)
    config = configuration('standard-device', 'pinned')
    case, _ = run_case(args, ROOT, 'gpu', 1, 'identity', 4, grid='32,32,32', tgv_mapping='periodic',
        insitu_config=config, postprocess_transport='pinned', checkpoint_interval=1)
    source = case / 'outdat/new/checkpoints/step000000000003'
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    changed = config.replace('mean_streamline_render=.true.', 'mean_streamline_render=.false.')
    run_case(args, ROOT, 'gpu', 1, 'changed_mean', 4, grid='32,32,32', tgv_mapping='periodic',
        insitu_config=changed, postprocess_transport='pinned', restore=source, checkpoint_interval=1,
        reject='native render configuration differs; select explicit override')
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}


@pytest.mark.parametrize('mapping,axis,mode', [('periodic', 'x', 'pinned'), ('y-wavy', 'y', 'device-aware')])
def test_memcheck(tmp_path, mapping, axis, mode):
    args = current_arguments(tmp_path, axis=axis, samples=False)
    args.runtime_timeout_seconds = 600
    case, _ = run_case(args, ROOT, 'gpu', 2, 'safe', 2, grid='32,32,32', tgv_mapping=mapping,
        insitu_config=configuration('standard-device', mode), postprocess_transport=mode,
        resident_audit=True, checkpoint_enabled=False, enabled=False, memcheck=True)
    validate(case, 2, 'standard-device', steps=(2,))
    reports = list(case.glob('memcheck.*.log'))
    assert len(reports) == 2 and all('ERROR SUMMARY: 0 errors' in path.read_text() for path in reports)


def read_trace_oracle(path):
    data = path.read_bytes()
    points, cells = map(int, np.frombuffer(data, dtype='<u8', count=2))
    offset, arrays = 16, []
    for dtype, shape in (('<f8', (points, 3)), ('<f8', (points, 3)), ('<f8', (points, 5)),
                         ('<i8', (points,)), ('<i8', (cells, 2))):
        count = int(np.prod(shape))
        arrays.append(np.frombuffer(data, dtype=dtype, count=count, offset=offset).reshape(shape))
        offset += 8*count
    assert offset == len(data)
    return arrays


class IndependentHexReference:
    """Exhaustive physical AABB candidates and NumPy trilinear Newton solve.

    This bounded host diagnostic does not reuse the production locator/inverse.
    Every containing cell is considered, including shared faces and warped walls.
    """
    bits = np.array(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                     (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)))

    def __init__(self, positions):
        indices = np.indices((32, 32, 32)).reshape(3, -1).T[:, ::-1]
        self.nodes = indices[:, None, :] + self.bits[None, :, :]
        self.vertices = positions[tuple(self.nodes[..., d] for d in (2, 1, 0))]
        self.lower, self.upper = self.vertices.min(axis=1), self.vertices.max(axis=1)

    @classmethod
    def weights(cls, parameters):
        factors = np.where(cls.bits[None, :, :], parameters[:, None, :], 1-parameters[:, None, :])
        return np.prod(factors, axis=2), factors

    def locate(self, points):
        owners = np.empty(len(points), dtype=int)
        parameters = np.empty_like(points)
        for start in range(0, len(points), 64):
            xyz = points[start:start+64]
            inside = np.all((xyz[:, None, :] >= self.lower[None, :, :]-2e-13) &
                            (xyz[:, None, :] <= self.upper[None, :, :]+2e-13), axis=2)
            point_id, cell_id = np.nonzero(inside)
            assert np.all(inside.any(axis=1)), ('No physical AABB candidate', xyz[~inside.any(axis=1)])
            vertices, target = self.vertices[cell_id], xyz[point_id]
            p = np.full((len(cell_id), 3), .5)
            for _ in range(20):
                weights, factors = self.weights(p)
                residual = np.einsum('pn,pnv->pv', weights, vertices)-target
                derivative = np.empty((len(p), 8, 3))
                for d in range(3):
                    derivative[..., d] = (2*self.bits[:, d]-1)*np.prod(np.delete(factors, d, axis=2), axis=2)
                jacobian = np.einsum('pnv,pnd->pvd', vertices, derivative)
                assert np.isfinite(jacobian).all() and (np.linalg.det(jacobian) > 0).all()
                correction = np.linalg.solve(jacobian, residual[..., None])[..., 0]
                p -= correction
                if np.max(abs(correction), initial=0.) <= 2e-14:
                    break
            weights, _ = self.weights(p)
            residual = np.max(abs(np.einsum('pn,pnv->pv', weights, vertices)-target), axis=1)
            valid = (residual <= 2e-13) & np.all((p >= -2e-13) & (p <= 1+2e-13), axis=1)
            for i in range(len(xyz)):
                candidates = np.flatnonzero(valid & (point_id == i))
                assert len(candidates), ('No converged containing physical hex', xyz[i])
                candidate = candidates[np.argmin(cell_id[candidates])]
                owners[start+i], parameters[start+i] = cell_id[candidate], np.clip(p[candidate], 0., 1.)
        return owners, parameters

    def interpolate(self, values, owners, parameters):
        nodes = self.nodes[owners]
        corners = values[tuple(nodes[..., d] for d in (2, 1, 0))]
        weights, _ = self.weights(parameters)
        return np.einsum('pn,pnv->pv', weights, corners)


@pytest.mark.parametrize('mapping', ('periodic', 'y-wavy'))
def test_independent_reference_affine(mapping):
    from generate_curvilinear_tgv_grid import mapped_grid
    positions = np.stack([v.transpose(2, 1, 0) for v in mapped_grid(32, 32, 32, .15, mapping)[:3]], axis=-1)
    reference = IndependentHexReference(positions)
    cell = np.array((0, 33, 15855, 16912, 32767))
    p = np.array(((.13, .39, .78), (0., .5, 1.), (.31, .73, .22), (1., 0., .5), (.8, .9, .4)))
    weights, _ = reference.weights(p)
    xyz = np.einsum('pn,pnv->pv', weights, reference.vertices[cell])
    owners, recovered = reference.locate(xyz)
    matrix = np.array(((.1, -.2, .3), (.4, .7, -.5), (.3, .2, .9)))
    offset = np.array((.7, -.9, .3))
    actual = reference.interpolate(positions @ matrix.T+offset, owners, recovered)
    np.testing.assert_allclose(actual, xyz @ matrix.T+offset, rtol=0., atol=2e-13)


def source_vectors(case, mapping, name):
    checkpoint = case / FINAL
    if name in PRODUCTS:
        with h5py.File(checkpoint / 'state.h5') as state:
            rho = state['q0001'][...]
            values = np.stack([state[f'q{d:04d}'][...]/rho for d in (2, 3, 4)], axis=-1)
    else:
        first = 10 if name == 'mean_reynolds_streamlines' else 13
        with h5py.File(checkpoint / 'statistics.h5') as state:
            values = np.stack([state[f'q{d:04d}'][...] for d in range(first, first+3)], axis=-1)
    # Reconstruct only periodic endpoint copies; keep the physical y=upper wall.
    values[:, :, 32] = values[:, :, 0]
    values[32] = values[0]
    if mapping == 'periodic':
        values[:, 32] = values[:, 0]
    assert np.isfinite(values).all()
    return values


def check_trace_oracles(case, ranks, axis, mapping, reference_case):
    with h5py.File(case / 'outdat/new/resources/geometry.h5') as grid:
        positions = np.stack([grid[f'q{d:04d}'][...] for d in (1, 2, 3)], axis=-1)
    reference = IndependentHexReference(positions)
    worst = 0.
    for name in sorted(PRODUCTS | MEANS):
        source = source_vectors(reference_case, mapping, name)
        grouped = []
        for rank in range(ranks):
            path = case / f'outdat/render/curve_trace_oracle.{name}.step4.rank{rank}.bin'
            xyz, velocity, accepted, particle, cells = read_trace_oracle(path)
            assert np.isfinite(xyz).all() and np.isfinite(velocity).all() and np.isfinite(accepted).all()
            assert (particle >= 0).all() and (particle < (16 if name == 'crossing_streamlines' else 32)).all()
            assert (cells >= 0).all() and (cells < len(xyz)).all()
            np.testing.assert_array_equal(xyz, accepted[:, :3])
            owners, parameters = reference.locate(xyz)
            expected = reference.interpolate(source, owners, parameters)
            error = float(np.max(abs(velocity-expected), initial=0.))
            assert error <= 2e-10, (name, rank, error)
            worst = max(worst, error)
            assert np.array_equal(particle[cells[:, 0]], particle[cells[:, 1]])
            distance = accepted[cells[:, 1], 3]-accepted[cells[:, 0], 3]
            chord = np.linalg.norm(xyz[cells[:, 1]]-xyz[cells[:, 0]], axis=1)
            assert (distance > 0).all() and (chord > 0).all()
            assert (distance <= .5*(2*np.pi/32)+2e-10).all()
            assert (accepted[:, 3] >= 0).all() and (accepted[:, 3] <= np.pi+2e-10).all()
            grouped.append((xyz, accepted, particle, cells))
            if name == 'crossing_streamlines':
                direction = 'xyz'.index(axis) if ranks == 2 else 0
                seed = np.column_stack((np.full(len(xyz), np.pi/2),
                    np.pi/8+(particle % 16)*(3*np.pi/4)/15, np.full(len(xyz), np.pi/4)))
                seed = np.roll(seed, direction, axis=1)
                seed[:, direction] += accepted[:, 3]
                error = float(np.max(abs(xyz-seed), initial=0.))
                assert error <= 2e-10
                worst = max(worst, error)
        for pid in range(16 if name == 'crossing_streamlines' else 32):
            intervals, points = [], []
            for xyz, accepted, particle, cells in grouped:
                edge = cells[particle[cells[:, 0]] == pid]
                intervals.extend(accepted[edge, 3])
                mask = particle == pid
                points.extend(np.column_stack((accepted[mask, 3], xyz[mask])))
            intervals = np.asarray(sorted(intervals, key=lambda value: value[0]))
            points = np.asarray(sorted(points, key=lambda value: value[0]))
            assert len(intervals) and len(points), (name, pid)
            assert abs(intervals[0, 0]) <= 2e-10
            assert np.max(abs(intervals[1:, 0]-intervals[:-1, 1]), initial=0.) <= 2e-10
            repeated = abs(np.diff(points[:, 0])) <= 2e-13
            assert np.max(abs(np.diff(points[:, 1:], axis=0)[repeated]), initial=0.) <= 2e-10
            if name == 'crossing_streamlines':
                assert abs(intervals[-1, 1]-np.pi) <= 2e-10
    assert 'runtime_zero_readback_evidence=0' in (case / 'run.log').read_text()
    return worst


@pytest.mark.parametrize('mapping', ('periodic', 'y-wavy'))
@pytest.mark.parametrize('ranks,axis', TOPOLOGIES, ids=('single', 'x', 'y', 'z'))
@pytest.mark.parametrize('mode', ('pinned', 'device-aware'))
def test_independent_physical_trace_geometry(tmp_path, mapping, ranks, axis, mode, references, record_property):
    cpu, _ = references(mapping, ranks, axis)
    args = current_arguments(tmp_path, axis=axis, samples=False)
    args.runtime_timeout_seconds = 600
    kwargs = dict(grid='32,32,32', tgv_mapping=mapping, checkpoint_interval=1,
        insitu_config=configuration('standard-device', mode), postprocess_transport=mode,
        resident_audit=True, curve_trace_oracle=True)
    actual, size = run_case(args, ROOT, 'gpu', ranks, 'oracle', 4, **kwargs)
    record_property('same_state_independent_maxabs', check_trace_oracles(actual, ranks, axis, mapping, actual))
    record_property('cpu_gpu_product_maxabs', check_trace_oracles(actual, ranks, axis, mapping, cpu))
    resumed, _ = run_case(args, ROOT, 'gpu', ranks, 'resumed_oracle', 4,
        restore=actual / 'outdat/new/checkpoints/step000000000003', **kwargs)
    for path in (actual / 'outdat/render').glob('curve_trace_oracle.*.step4.rank*.bin'):
        assert path.read_bytes() == (resumed / 'outdat/render' / path.name).read_bytes()
    record_property('directory_bytes', size)
