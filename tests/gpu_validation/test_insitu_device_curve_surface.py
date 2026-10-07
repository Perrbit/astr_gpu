"""Bounded periodic CURVE Q surfaces in real physical coordinates."""
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import pytest

from run_output_restart_validation import run_case, compare_fields
from run_insitu_gpu_derivatives import check_resources
from test_insitu_device_products import resident_configuration
from test_insitu_device_curve_wall_fields import current_arguments
from test_insitu_curve_derivatives import ROOT, FINAL, reference
from test_insitu_is4 import compare_numerical


def configuration(pipeline, mode):
    text = resident_configuration(pipeline, mode, statistics=False, interval=2, profile='q_surface')
    backend = ROOT.parent / 'astr_dependencies/build/paraview-6.1.1-astr-device-gcc13/lib/catalyst'
    return re.sub(r"implementation_path='[^']+'", f"implementation_path='{backend}'", text)


def validate(case, ranks, steps, pipeline):
    log = (case / 'run.log').read_text()
    frames = re.findall(r'ASTR_INSITU_CURVE_SURFACE_FRAME rank=(\d+) step=(\d+) '
        r'physical_coordinates=1 field_download_bytes=0 geometry_host_bytes=0', log)
    assert len(frames) == ranks * len(steps), log
    audits = re.findall(r'ASTR_INSITU_RESIDENT_AUDIT rank=(\d+) step=(\d+) product=q_surface '
        r'field_maxabs=(\S+) display_maxabs=(\S+) points=(\d+)', log)
    assert len(audits) == ranks * len(steps), log
    assert all(np.isfinite(float(row[2])) and float(row[2]) <= 2e-10 and float(row[3]) == 0.
               for row in audits)
    for step in steps:
        image = case / f'outdat/render/q_surface.step{step:08d}.jpeg'
        assert image.with_suffix('.eps').is_file()
        with Image.open(image) as source:
            pixels = np.asarray(source.convert('RGB'))
            assert pixels.shape == (600, 800, 3)
            assert np.count_nonzero(np.any(pixels < 220, axis=2)) > 10000
        for rank in range(ranks):
            record = json.loads((case / f'outdat/render/mesh_step{step:08d}_rank{rank}.json').read_text())
            assert record['geometry_host_bytes'] == 0 and record['rendering_pipeline'] == pipeline
            assert set(record['products']) == {'q_surface'}
            assert record['products']['q_surface']['local_cells'] > 0
    assert not list((case / 'outdat/render').glob('*.vtp'))
    check_resources(case, ranks, steps=steps, capture=False)


@pytest.fixture(scope='module')
def cpu_references(tmp_path_factory):
    cases = {}
    for ranks, axis in ((1, 'x'), (2, 'x'), (2, 'y'), (2, 'z')):
        args = current_arguments(tmp_path_factory.mktemp(f'curve_surface_cpu_{ranks}_{axis}'),
                                 backend='cpu', axis=axis, samples=False)
        cases[ranks, axis], _ = run_case(args, ROOT, 'cpu', ranks, 'reference', 4,
            grid='32,32,32', tgv_mapping='periodic', checkpoint_interval=1)
    return cases


def read_oracle(path):
    data = path.read_bytes()
    points, cells = np.frombuffer(data, dtype='<u8', count=2).astype(int)
    offset = 16
    arrays = []
    for dtype, shape in (('<f8', (points, 3)), ('<f8', (points, 3)), ('<f8', (points, 3)),
                         ('<f8', (points,)), ('<i8', (cells, 3))):
        count = int(np.prod(shape))
        arrays.append(np.frombuffer(data, dtype=dtype, count=count, offset=offset).reshape(shape))
        offset += 8 * count
    assert offset == len(data)
    return arrays


def check_oracles(case, ranks, reference_case, field_reference=reference, iso=.25):
    with h5py.File(reference_case / FINAL / 'state.h5') as state:
        rho = state['q0001'][...]
        cells = rho.shape[0]-1
        assert rho.shape == (cells+1,)*3
        velocity = np.stack([state[f'q{d:04d}'][...] / rho for d in (2, 3, 4)], axis=-1)
    with h5py.File(case / 'outdat/new/resources/geometry.h5') as grid:
        positions = np.stack([grid[f'q{d:04d}'][...] for d in (1, 2, 3)], axis=-1)
    q = field_reference(reference_case, 4)[9]
    worst = 0.
    for rank in range(ranks):
        xyz, computational, vectors, contour, triangles = read_oracle(
            case / f'outdat/render/curve_q_oracle.step4.rank{rank}.bin')
        assert np.isfinite(xyz).all() and np.isfinite(vectors).all()
        sample = computational / (2*np.pi/cells)
        base = np.minimum(cells-1, np.maximum(0, np.floor(sample).astype(int)))
        fraction = sample - base
        expected_xyz = np.zeros_like(xyz)
        expected_v = np.zeros_like(vectors)
        expected_q = np.zeros_like(contour)
        for z in (0, 1):
            for y in (0, 1):
                for x in (0, 1):
                    bits = np.array([x, y, z])
                    weights = np.prod(np.where(bits, fraction, 1-fraction), axis=1)
                    indices = base + bits
                    index = tuple(indices[:, d] for d in (2, 1, 0))
                    expected_xyz += weights[:, None]*positions[index]
                    expected_v += weights[:, None]*velocity[index]
                    expected_q += weights*q[index]
        for actual, expected in ((xyz, expected_xyz), (vectors, expected_v),
                                 (expected_q, np.full_like(contour, iso)),
                                 (contour, np.full_like(contour, iso))):
            error = float(np.max(abs(actual-expected)))
            assert np.isfinite(error) and error <= 2e-10, (rank, error)
            worst = max(worst, error)
        assert len(triangles) and (triangles >= 0).all() and (triangles < len(xyz)).all()
        a, b, c = (xyz[triangles[:, d]] for d in range(3))
        area = np.linalg.norm(np.cross(b-a, c-a), axis=1)/2
        assert np.isfinite(area).all() and (area > 0).all()
    return worst


@pytest.mark.parametrize('axis', ['x', 'y', 'z'])
@pytest.mark.parametrize('pipeline,mode', [('standard-device', 'pinned'), ('direct-device', 'device-aware')])
def test_wall_curve_sixth_q(tmp_path, axis, pipeline, mode, record_property):
    from insitu_curve_boundary_reference import completed_wall_fields
    args = current_arguments(tmp_path, axis=axis, samples=False)
    kwargs = dict(grid='32,32,32', tgv_mapping='y-wavy', checkpoint_interval=1)
    actual, _ = run_case(args, ROOT, 'gpu', 2, 'wall_surface', 4,
        insitu_config=configuration(pipeline, mode), postprocess_transport=mode,
        resident_audit=True, curve_surface_oracle=True, **kwargs)
    validate(actual, 2, (2, 4), pipeline)
    record_property('same_state_sixth_q_maxabs', check_oracles(actual, 2, actual, completed_wall_fields))
    cpu, _ = run_case(current_arguments(tmp_path, 'cpu', axis, samples=False), ROOT,
                      'cpu', 2, 'reference', 4, **kwargs)
    record_property('cpu_gpu_state_maxabs', compare_numerical(cpu/FINAL/'state.h5', actual/FINAL/'state.h5'))
    record_property('cpu_gpu_sixth_q_maxabs', check_oracles(actual, 2, cpu, completed_wall_fields))
    off, _ = run_case(args, ROOT, 'gpu', 2, 'off', 4, **kwargs)
    compare_fields(actual/FINAL/'state.h5', off/FINAL/'state.h5')
    resumed, _ = run_case(args, ROOT, 'gpu', 2, 'resumed', 4,
        restore=actual/'outdat/new/checkpoints/step000000000003',
        insitu_config=configuration(pipeline, mode), postprocess_transport=mode,
        resident_audit=True, curve_surface_oracle=True, **kwargs)
    compare_fields(actual/FINAL/'state.h5', resumed/FINAL/'state.h5')
    for name in ('control.bin', 'insitu_control.bin'):
        assert (actual/FINAL/name).read_bytes() == (resumed/FINAL/name).read_bytes()
    for suffix in ('jpeg', 'eps'):
        name = f'outdat/render/q_surface.step00000004.{suffix}'
        assert (actual/name).read_bytes() == (resumed/name).read_bytes()
    for rank in range(2):
        name = f'outdat/render/curve_q_oracle.step4.rank{rank}.bin'
        assert (actual/name).read_bytes() == (resumed/name).read_bytes()


@pytest.mark.parametrize('ranks,axis', [(1, 'x'), (2, 'x'), (2, 'y'), (2, 'z')],
                         ids=['single', 'x', 'y', 'z'])
@pytest.mark.parametrize('mode', ['pinned', 'device-aware'])
@pytest.mark.parametrize('pipeline', ['standard-device', 'direct-device'])
def test_real_periodic_curve_surface(tmp_path, ranks, axis, mode, pipeline, record_property, cpu_references):
    args = current_arguments(tmp_path, axis=axis, samples=False)
    args.runtime_timeout_seconds = 300
    kwargs = dict(grid='32,32,32', tgv_mapping='periodic', checkpoint_interval=1)
    actual, size = run_case(args, ROOT, 'gpu', ranks, 'surface', 4,
        insitu_config=configuration(pipeline, mode), postprocess_transport=mode,
        resident_audit=True, **kwargs)
    validate(actual, ranks, (2, 4), pipeline)
    record_property('cpu_gpu_state_maxabs', compare_numerical(
        actual / FINAL / 'state.h5', cpu_references[ranks, axis] / FINAL / 'state.h5'))
    record_property('directory_bytes', size)
    off, _ = run_case(args, ROOT, 'gpu', ranks, 'off', 4, **kwargs)
    compare_fields(actual / FINAL / 'state.h5', off / FINAL / 'state.h5')
    resumed, _ = run_case(args, ROOT, 'gpu', ranks, 'restart', 4,
        restore=actual / 'outdat/new/checkpoints/step000000000003',
        insitu_config=configuration(pipeline, mode), postprocess_transport=mode,
        resident_audit=True, **kwargs)
    validate(resumed, ranks, (4,), pipeline)
    compare_fields(actual / FINAL / 'state.h5', resumed / FINAL / 'state.h5')
    for name in ('control.bin', 'insitu_control.bin'):
        assert (actual / FINAL / name).read_bytes() == (resumed / FINAL / name).read_bytes()
    for suffix in ('jpeg', 'eps'):
        name = f'outdat/render/q_surface.step00000004.{suffix}'
        assert (actual / name).read_bytes() == (resumed / name).read_bytes()


@pytest.mark.parametrize('ranks,axis', [(1, 'x'), (2, 'x'), (2, 'y'), (2, 'z')],
                         ids=['single', 'x', 'y', 'z'])
@pytest.mark.parametrize('mode', ['pinned', 'device-aware'])
def test_curve_surface_independent_geometry(tmp_path, ranks, axis, mode, cpu_references, record_property):
    args = current_arguments(tmp_path, axis=axis, samples=False)
    case, _ = run_case(args, ROOT, 'gpu', ranks, 'diagnostic', 4, grid='32,32,32', tgv_mapping='periodic',
        checkpoint_interval=1, insitu_config=configuration('standard-device', mode), postprocess_transport=mode,
        resident_audit=True, curve_surface_oracle=True)
    record_property('same_state_independent_maxabs', check_oracles(case, ranks, case))
    record_property('cpu_gpu_product_maxabs', check_oracles(case, ranks, cpu_references[ranks, axis]))
    resumed, _ = run_case(args, ROOT, 'gpu', ranks, 'resumed', 4, grid='32,32,32', tgv_mapping='periodic',
        checkpoint_interval=1, restore=case / 'outdat/new/checkpoints/step000000000003',
        insitu_config=configuration('standard-device', mode), postprocess_transport=mode,
        resident_audit=True, curve_surface_oracle=True)
    for rank in range(ranks):
        name = f'outdat/render/curve_q_oracle.step4.rank{rank}.bin'
        assert (case / name).read_bytes() == (resumed / name).read_bytes()
    assert 'runtime_zero_readback_evidence=0' in (case / 'run.log').read_text()


@pytest.mark.parametrize('pipeline,mode,axis', [('standard-device', 'device-aware', 'z'),
                                              ('direct-device', 'pinned', 'x')])
@pytest.mark.parametrize('mapping', ['periodic', 'y-wavy'])
def test_curve_surface_memcheck(tmp_path, pipeline, mode, axis, mapping):
    args = current_arguments(tmp_path, axis=axis, samples=False)
    args.runtime_timeout_seconds = 300
    case, _ = run_case(args, ROOT, 'gpu', 2, 'memcheck', 2, grid='32,32,32', tgv_mapping=mapping,
        insitu_config=configuration(pipeline, mode), postprocess_transport=mode, resident_audit=True,
        memcheck=True, enabled=False, checkpoint_enabled=False)
    validate(case, 2, (2,), pipeline)
    logs = list(case.glob('memcheck.*.log'))
    assert len(logs) == 2 and all('ERROR SUMMARY: 0 errors' in p.read_text() for p in logs)
    assert not list((case / 'outdat').rglob('*.h5'))
    assert not list((case / 'outdat/render').glob('curve_q_oracle.*'))


@pytest.mark.parametrize('change,expected', [
    ('statistics', 'CURVE device Q requires strict rendering without statistics'),
    ('compatible', 'CURVE device Q requires strict rendering without statistics'),
    ('q_streamlines', 'device products require 32^3 TGV'),
])
def test_unadmitted_curve_combinations_rejected(tmp_path, change, expected):
    args = current_arguments(tmp_path, samples=False)
    config = configuration('standard-device', 'pinned')
    if change == 'statistics':
        config = config.replace('statistics=.false.', 'statistics=.true.')
    elif change == 'compatible':
        config = config.replace("rendering_pipeline='standard-device'", "rendering_pipeline='compatible'")
    else:
        config = config.replace("products='q_surface'", "products='q_streamlines'")
    run_case(args, ROOT, 'gpu', 1, 'reject_'+change, 2, grid='32,32,32', tgv_mapping='periodic',
        insitu_config=config, postprocess_transport='pinned', reject=expected)


def test_curve_contour_controlled_budget_rejected(tmp_path):
    args = current_arguments(tmp_path, samples=False)
    config = configuration('standard-device', 'pinned').replace('device_budget_bytes=2147483648',
                                                               'device_budget_bytes=268435456')
    run_case(args, ROOT, 'gpu', 1, 'reject_budget', 2, grid='32,32,32', tgv_mapping='periodic',
        insitu_config=config, postprocess_transport='pinned', reject='CURVE Q controlled allocation budget/reserve',
        failure_after_start=True)


def edge_key(point):
    sample = point/(2*np.pi/32)
    near = abs(sample-np.rint(sample)) <= 16*np.finfo(float).eps*32
    assert near.sum() == 2, sample  # This actual fixture has no nodal Q=0.25 intersections.
    axis = int(np.flatnonzero(~near)[0])
    low = np.rint(sample).astype(int)
    low[axis] = int(np.floor(sample[axis]))
    return (axis, *low)


def keyed_geometry(case, ranks):
    points, triangles, edges = {}, set(), {}
    for rank in range(ranks):
        xyz, computational, velocity, q, cells = read_oracle(case / f'outdat/render/curve_q_oracle.step4.rank{rank}.bin')
        keys = [edge_key(p) for p in computational]
        assert len(keys) == len(set(keys))
        for key, position, value in zip(keys, xyz, velocity):
            data = np.concatenate((position, value))
            if key in points:
                np.testing.assert_allclose(points[key], data, atol=2e-10, rtol=0)
            points[key] = data
        for ids in cells:
            tri = tuple(keys[i] for i in ids)
            start = tri.index(min(tri))
            tri = tri[start:]+tri[:start]
            assert len(set(tri)) == 3 and tri not in triangles
            triangles.add(tri)
            for a, b in zip(tri, tri[1:]+tri[:1]):
                edges.setdefault(tuple(sorted((a, b))), []).append((a, b))
    for edge, oriented in edges.items():
        assert len(oriented) <= 2, edge
        if len(oriented) == 2:
            assert oriented[0] == oriented[1][::-1], (edge, oriented)
        else:
            # An open edge must be on the physical box's computational boundary,
            # not on an internal MPI seam. Periodic geometric endpoints stay distinct.
            a, b = edge
            assert any(a[0] != d and b[0] != d and a[d+1] == b[d+1] and a[d+1] in (0, 32)
                       for d in range(3)), edge
    return points, triangles


def test_completed_curve_surface_partition_geometry(record_property):
    root = os.environ.get('ASTR_INSITU_CURVE_SURFACE_MATRIX_ROOT')
    if not root:
        pytest.skip('Select the immutable completed CURVE surface matrix')
    cases = sorted(p for p in Path(root).glob('test_curve_surface_independent*/gpu_np*_diagnostic')
                   if not p.parent.is_symlink())
    assert len(cases) == 8
    reference_points = reference_triangles = None
    worst = 0.
    for case in cases:
        ranks = int(re.search(r'gpu_np(\d+)_', case.name)[1])
        points, triangles = keyed_geometry(case, ranks)
        if reference_points is None:
            reference_points, reference_triangles = points, triangles
        assert set(points) == set(reference_points) and triangles == reference_triangles
        for key in points:
            worst = max(worst, float(np.max(abs(points[key]-reference_points[key]))))
    assert worst <= 2e-10
    record_property('cross_partition_coordinates_velocity_maxabs', worst)
    record_property('global_triangles', len(reference_triangles))
    record_property('global_points', len(reference_points))
