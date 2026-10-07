"""Actual static-plane products; diagnostic downloads and strict runs are separate."""
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np
import pytest
from PIL import Image

from run_output_restart_validation import run_case, compare_fields
from test_insitu_device_products import resident_configuration
from test_insitu_device_curve_wall_fields import current_arguments
from test_insitu_curve_derivatives import ROOT, FINAL
from test_insitu_is4 import compare_numerical


def configuration(pipeline, mode, kind='oblique'):
    text = resident_configuration(pipeline, mode, statistics=False, interval=2, profile='velocity_slice')
    origin = (3., 0., 0.) if kind == 'oblique' else ((np.pi, 0., 0.) if kind == 'coincident' else (20., 0., 0.))
    normal = (1., .25, -.125) if kind == 'oblique' else (1., 0., 0.)
    return text.replace("products='velocity_slice',", "products='velocity_slice',slice_definition='plane',"
        + 'slice_origin=' + ','.join(f'{value:.17e}' for value in origin) + ','
        + 'slice_normal=' + ','.join(f'{value:.17e}' for value in normal) + ',')


def oracle(path):
    data = path.read_bytes()
    points, cells = np.frombuffer(data, dtype='<u8', count=2).astype(int)
    offset = 16
    result = []
    for dtype, shape in (('<f8', (points, 3)), ('<f8', (points, 3)),
                         ('<i8', (points, 2)), ('<i8', (cells, 3))):
        count = int(np.prod(shape))
        result.append(np.frombuffer(data, dtype=dtype, offset=offset, count=count).reshape(shape))
        offset += 8 * count
    assert offset == len(data)
    return result


def check_oracles(case, ranks, step, kind='oblique', reference_case=None):
    reference_case = case if reference_case is None else reference_case
    source = f'outdat/new/checkpoints/step{step:012d}/state.h5'
    with h5py.File(reference_case / source) as state:
        rho = state['q0001'][...]
        velocity = np.stack([state[f'q{component:04d}'][...] / rho for component in (2, 3, 4)], axis=-1)
    with h5py.File(case / 'outdat/new/resources/geometry.h5') as grid:
        positions = np.stack([grid[f'q{d:04d}'][...] for d in (1, 2, 3)], axis=-1)
    points, triangles, edges = {}, set(), {}
    worst = area = 0.
    plane = None
    for rank in range(ranks):
        record = json.loads((case / f'outdat/render/mesh_step{step:08d}_rank{rank}.json').read_text())
        identity = record['products']['velocity_slice']['plane']
        if plane is not None:
            assert plane == identity
        plane = identity
        origin, normal = np.asarray(identity['origin']), np.asarray(identity['normal'])
        xyz, value, keys, indices = oracle(case / f'outdat/render/plane_oracle.step{step}.rank{rank}.bin')
        assert xyz.shape[0] == record['local_points'] and indices.shape[0] == record['local_cells']
        assert np.isfinite(xyz).all() and np.isfinite(value).all()
        assert all(tuple(a) < tuple(b) for a, b in zip(keys, keys[1:]))
        for p, v, key in zip(xyz, value, keys):
            key = tuple(key)
            assert 0 <= key[0] <= key[1] < 33**3
            endpoints = [np.array([node % 33, node // 33 % 33, node // 33**2]) for node in key]
            a, b = [positions[tuple(index[::-1])] for index in endpoints]
            va, vb = [velocity[tuple((index % 32)[::-1])] for index in endpoints]
            if key[0] == key[1]:
                fraction = 0.
            else:
                direction = int(np.argmax(abs(b-a)))
                fraction = (p[direction]-a[direction])/(b[direction]-a[direction])
            assert 0. <= fraction <= 1.
            worst = max(worst, float(np.max(abs(p-(a+fraction*(b-a))))),
                        float(np.max(abs(v-(va+fraction*(vb-va))))),
                        float(abs(np.dot((p-origin).astype(np.longdouble), normal.astype(np.longdouble)))))
            assert worst <= 2e-10, (key, worst)
            actual = np.concatenate((p, v))
            if key in points:
                np.testing.assert_array_equal(points[key], actual)
            points[key] = actual
        for ids in indices:
            assert np.all(ids >= 0) and np.all(ids < len(keys))
            tri = tuple(tuple(keys[p]) for p in ids)
            unordered = tuple(sorted(tri))
            assert unordered not in triangles
            triangles.add(unordered)
            a, b, c = xyz[ids].astype(np.longdouble)
            cross = np.cross(b-a, c-a)
            assert np.dot(cross, normal) > 0. and np.dot(cross, cross) > 0.
            area += float(.5*np.sqrt(np.dot(cross, cross)))
            for first, second in zip(tri, tri[1:] + tri[:1]):
                pair = tuple(sorted((first, second)))
                count, winding = edges.get(pair, (0, 0))
                edges[pair] = (count+1, winding+(1 if first < second else -1))
    if kind == 'empty':
        assert not points and not triangles and area == 0.
        return worst, area, 0
    graph, boundary = {}, {}
    for (first, second), (count, winding) in edges.items():
        graph.setdefault(first, set()).add(second)
        graph.setdefault(second, set()).add(first)
        if count == 1:
            ids = np.array([first[0], first[1], second[0], second[1]])
            assert any(np.all(component == side) for component in (ids % 33, ids // 33 % 33, ids // 33**2)
                       for side in (0, 32)), (first, second)
            boundary.setdefault(first, set()).add(second)
            boundary.setdefault(second, set()).add(first)
        else:
            assert count == 2 and winding == 0
    assert len(points)-len(edges)+len(triangles) == 1
    assert boundary and all(len(neighbors) == 2 for neighbors in boundary.values())
    for links in (graph, boundary):
        pending, seen = [next(iter(links))], set()
        while pending:
            node = pending.pop()
            if node not in seen:
                seen.add(node)
                pending.extend(links[node]-seen)
        assert len(seen) == len(links)
    expected = 4*np.pi**2 * (np.sqrt(1.+.25**2+.125**2) if kind == 'oblique' else 1.)
    assert abs(area-expected) <= 2e-10, (area, expected)
    return worst, area, len(triangles)


@pytest.fixture(scope='module')
def solver_references(tmp_path_factory):
    cases = {}
    for mapping in (None, 'periodic'):
        for ranks, axis in ((1, 'x'), (2, 'x'), (2, 'y'), (2, 'z')):
            directory = tmp_path_factory.mktemp(f'plane_ref_{mapping}_{ranks}_{axis}')
            for backend in ('cpu', 'gpu'):
                args = current_arguments(directory, backend, axis, samples=False)
                cases[mapping, ranks, axis, backend], _ = run_case(args, ROOT, backend, ranks, 'reference', 4,
                    grid='32,32,32', tgv_mapping=mapping, checkpoint_interval=1)
            compare_numerical(cases[mapping, ranks, axis, 'cpu'] / FINAL / 'state.h5',
                              cases[mapping, ranks, axis, 'gpu'] / FINAL / 'state.h5')
    return cases


@pytest.mark.parametrize('mapping', [None, 'periodic'], ids=['cartesian', 'curve'])
@pytest.mark.parametrize('ranks,axis', [(1, 'x'), (2, 'x'), (2, 'y'), (2, 'z')], ids=['single', 'x', 'y', 'z'])
@pytest.mark.parametrize('mode', ['pinned', 'device-aware'])
@pytest.mark.parametrize('pipeline', ['standard-device', 'direct-device'])
def test_completed_plane_products(tmp_path, mapping, ranks, axis, mode, pipeline, solver_references, record_property):
    args = current_arguments(tmp_path, axis=axis, samples=False)
    args.runtime_timeout_seconds = 300
    config = configuration(pipeline, mode)
    kwargs = dict(grid='32,32,32', tgv_mapping=mapping, checkpoint_interval=1,
                  insitu_config=config, postprocess_transport=mode)
    diagnostic, _ = run_case(args, ROOT, 'gpu', ranks, 'diagnostic', 4, plane_oracle=True, **kwargs)
    reference = solver_references[mapping, ranks, axis, 'gpu']
    compare_fields(diagnostic / FINAL / 'state.h5', reference / FINAL / 'state.h5')
    error, area, triangles = check_oracles(diagnostic, ranks, 4)
    cpu_error, _, _ = check_oracles(diagnostic, ranks, 4,
                                   reference_case=solver_references[mapping, ranks, axis, 'cpu'])
    strict, _ = run_case(args, ROOT, 'gpu', ranks, 'strict', 4, **kwargs)
    compare_fields(strict / FINAL / 'state.h5', reference / FINAL / 'state.h5')
    assert not list((strict / 'outdat/render').glob('plane_oracle.*'))
    assert 'ASTR_INSITU_PLANE_TEST_ORACLE' not in (strict / 'run.log').read_text()
    assert 'device_gradient_q_inclusive' not in (strict / 'run.log').read_text()
    source = diagnostic / 'outdat/new/checkpoints/step000000000003'
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    resumed, _ = run_case(args, ROOT, 'gpu', ranks, 'resumed', 4, restore=source, plane_oracle=True, **kwargs)
    compare_fields(diagnostic / FINAL / 'state.h5', resumed / FINAL / 'state.h5')
    for name in ('control.bin', 'insitu_control.bin'):
        assert (diagnostic / FINAL / name).read_bytes() == (resumed / FINAL / name).read_bytes()
    assert (source / 'insitu_control.bin').read_bytes()[:8] == b'ASTRIR05'
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    for rank in range(ranks):
        name = f'plane_oracle.step4.rank{rank}.bin'
        assert (diagnostic / 'outdat/render' / name).read_bytes() == (resumed / 'outdat/render' / name).read_bytes()
    for case in (diagnostic, strict, resumed):
        assert not list((case / 'outdat/render').rglob('*.vtp'))
        log = (case / 'run.log').read_text()
        audits = re.findall(r'ASTR_INSITU_DEVICE_PLANE rank=\d+ step=\d+ points=\d+ triangles=\d+ '
            r'area=(\S+) plane_maxabs=(\S+) invalid_triangles=(\S+)', log)
        assert audits and all(float(distance) <= 2e-10 and float(bad) == 0. for _, distance, bad in audits)
        for extension in ('jpeg', 'eps'):
            name = f'velocity_slice.step00000004.{extension}'
            assert (case / 'outdat/render' / name).read_bytes() == (diagnostic / 'outdat/render' / name).read_bytes()
        with Image.open(case / 'outdat/render/velocity_slice.step00000004.jpeg') as image:
            pixels = np.asarray(image.convert('RGB')).astype(int)[50:550,50:620]
        assert np.count_nonzero(pixels.max(2)-pixels.min(2) > 40) > 100
    record_property('private_interpolation_maxabs', error)
    record_property('cpu_gpu_interpolated_maxabs', cpu_error)
    record_property('global_area', area)
    record_property('global_triangles', triangles)


@pytest.mark.parametrize('mapping', [None, 'periodic'], ids=['cartesian', 'curve'])
@pytest.mark.parametrize('kind', ['coincident', 'empty'])
@pytest.mark.parametrize('pipeline,mode', [('standard-device', 'pinned'), ('direct-device', 'device-aware')])
def test_empty_or_coincident_partition(tmp_path, mapping, kind, pipeline, mode):
    args = current_arguments(tmp_path, axis='x', samples=False)
    case, _ = run_case(args, ROOT, 'gpu', 2, kind, 4, grid='32,32,32', tgv_mapping=mapping,
        insitu_config=configuration(pipeline, mode, kind), postprocess_transport=mode,
        checkpoint_interval=1, plane_oracle=True)
    check_oracles(case, 2, 4, kind)
    _, _, _, high = oracle(case / 'outdat/render/plane_oracle.step4.rank1.bin')
    assert high.size == 0


def test_plane_restart_identity_mismatch(tmp_path):
    args = current_arguments(tmp_path, axis='x', samples=False)
    config = configuration('standard-device', 'pinned')
    case, _ = run_case(args, ROOT, 'gpu', 2, 'seed', 3, grid='32,32,32', tgv_mapping='periodic',
        insitu_config=config, postprocess_transport='pinned', checkpoint_interval=1)
    source = case / 'outdat/new/checkpoints/step000000000003'
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    for name, changed in [('origin', config.replace(f'{3.:.17e}', f'{3.25:.17e}')),
                          ('normal', config.replace(f'{.25:.17e}', f'{.3:.17e}'))]:
        run_case(args, ROOT, 'gpu', 2, 'reject_'+name, 4, grid='32,32,32', tgv_mapping='periodic',
            insitu_config=changed, postprocess_transport='pinned', checkpoint_interval=1, restore=source,
            reject='native render configuration differs', failure_after_start=True)
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}


def test_unrepresentable_intersections_are_rejected(tmp_path):
    args = current_arguments(tmp_path, samples=False)
    config = configuration('standard-device', 'pinned').replace(f'{3.:.17e}', f'{np.pi:.17e}')
    case, _ = run_case(args, ROOT, 'gpu', 1, 'range_rejection', 2, grid='32,32,32', insitu_config=config,
        postprocess_transport='pinned', reject='Invalid structured plane geometry/range', failure_after_start=True)
    assert not list((case / 'outdat/render').glob('*.jpeg'))


@pytest.mark.parametrize('pipeline,mode,axis', [('standard-device', 'device-aware', 'z'),
                                              ('direct-device', 'pinned', 'x')])
def test_physical_plane_memory_safety(tmp_path, pipeline, mode, axis):
    args = current_arguments(tmp_path, axis=axis, samples=False)
    args.runtime_timeout_seconds = 300
    case, _ = run_case(args, ROOT, 'gpu', 2, 'memcheck', 2, grid='32,32,32', tgv_mapping='periodic',
        insitu_config=configuration(pipeline, mode), postprocess_transport=mode, memcheck=True, no_field_io=True)
    reports = list(case.glob('memcheck.*.log'))
    assert len(reports) == 2 and all('ERROR SUMMARY: 0 errors' in p.read_text() for p in reports)
    assert not list((case / 'outdat/new').rglob('*.h5'))
    assert not list((case / 'outdat/render').glob('plane_oracle.*'))


def keyed_product(case, step, ranks):
    points, triangles = {}, set()
    for rank in range(ranks):
        xyz, velocity, keys, indices = oracle(case / f'outdat/render/plane_oracle.step{step}.rank{rank}.bin')
        for key, position, value in zip(keys, xyz, velocity):
            key, data = tuple(key), np.concatenate((position, value))
            if key in points:
                np.testing.assert_array_equal(points[key], data)
            points[key] = data
        for ids in indices:
            tri = tuple(tuple(keys[i]) for i in ids)
            start = tri.index(min(tri))
            oriented = tri[start:] + tri[:start]
            assert oriented not in triangles
            triangles.add(oriented)
    return points, triangles


def test_completed_plane_partition_geometry_and_images(record_property):
    root = os.environ.get('ASTR_INSITU_PLANE_MATRIX_ROOT')
    if not root:
        pytest.skip('Select the immutable completed physical-plane matrix')
    from scipy.ndimage import distance_transform_edt
    cases = sorted(p for p in Path(root).glob('test_completed_plane_products_*/gpu_np*_diagnostic')
                   if not p.parent.is_symlink())
    assert len(cases) == 32
    references, pairs, counts = {}, {}, {}
    geometry_error = field_error = silhouette_error = 0.
    entry_error = 0
    for case in cases:
        log = (case / 'run.log').read_text()
        topology = tuple(map(int, re.search(r'forced mpi topology=\s*(\d+)\s+(\d+)\s+(\d+)',
                                           log).groups()))
        transport = re.search(r'ASTR_INSITU_DEVICE_PLANE_FRAME .*?transport=(\S+)', log).group(1)
        ranks = int(np.prod(topology))
        mapping = 'curve' if (case / 'datin/grid.tgv.h5').exists() else 'cartesian'
        for step in (2, 4):
            record = json.loads((case / f'outdat/render/mesh_step{step:08d}_rank0.json').read_text())
            points, triangles = keyed_product(case, step, ranks)
            with Image.open(case / f'outdat/render/velocity_slice.step{step:08d}.jpeg') as image:
                pixels = np.asarray(image.convert('RGB')).astype(int)[50:550, 50:620]
            mask = pixels.max(2) - pixels.min(2) > 40
            assert mask.sum() > 10000
            identity = mapping, step
            counts[identity] = counts.get(identity, 0) + 1
            if identity not in references:
                references[identity] = points, triangles, mask, record['products']['velocity_slice']['plane']
            reference, reference_triangles, reference_mask, plane = references[identity]
            assert points.keys() == reference.keys() and triangles == reference_triangles
            assert record['products']['velocity_slice']['plane'] == plane
            difference = np.array([points[key]-reference[key] for key in sorted(points)])
            geometry_error = max(geometry_error, float(abs(difference[:, :3]).max()))
            field_error = max(field_error, float(abs(difference[:, 3:]).max()))
            error = max(float(distance_transform_edt(~reference_mask)[mask].max()),
                        float(distance_transform_edt(~mask)[reference_mask].max()))
            assert error <= 1., (case, step, error)
            silhouette_error = max(silhouette_error, error)
            pair = mapping, step, topology, transport
            if pair in pairs:
                entry_error = max(entry_error, int(abs(pixels-pairs.pop(pair)).max()))
            else:
                pairs[pair] = pixels
    assert not pairs and len(counts) == 4 and set(counts.values()) == {16}
    assert geometry_error == 0. and field_error <= 2e-10 and entry_error == 0
    record_property('cross_topology_geometry_maxabs', geometry_error)
    record_property('cross_topology_velocity_maxabs', field_error)
    record_property('partition_seam_max_pixels', silhouette_error)
    record_property('cross_entry_pixel_maxabs', entry_error)
