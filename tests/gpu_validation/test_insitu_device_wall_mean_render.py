"""X4 resident wall means: field oracle is explicit test I/O, not a render gate."""
import json
import os
import re
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from test_insitu_device_wall_statistics import run, read_statistics, copies, compare_fields, FINAL
from test_insitu_air5_walls import FIELD_NAMES, FIELD_SCALES

PROFILES = ('channel', 'curve', 'air5')
TOPOLOGIES = ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2))
CHANNEL_PRODUCTS = ('wall_pressure', 'wall_shear_x', 'wall_heat_into_gas')
AIR5_PRODUCTS = ('pressure', 'temperature', 'vibrational_temperature', 'Y_N2', 'Y_O2',
                 'Y_N', 'Y_O', 'Y_NO', 'wall_shear_x', 'wall_heat_into_gas')


def key(header, index, profile):
    i, k, wall = index
    cells = 32 if profile == 'curve' else 16
    x = i + int(header[6]); z = (k + int(header[8])) % cells
    if profile != 'air5':
        x %= cells
    if header[5] == 1 and header[7] > 0:
        wall = 1
    return x, z, wall


def owned_values(case, ranks, profile):
    expected = {}
    nf = 18 if profile == 'air5' else 3
    for rank in range(ranks):
        header, clock, _, _, values, owned = read_statistics(
            case / f'outdat/render/sample.wall_statistics.step00000004.rank{rank:08d}.bin')
        for index in np.ndindex(owned.shape):
            if owned[index]:
                identity = key(header, index, profile)
                assert identity not in expected
                expected[identity] = values[index][:nf]
    return expected


def read_oracle(path):
    with path.open('rb') as stream:
        assert stream.read(8) == b'ASTRWM01'
        header = np.fromfile(stream, '<i4', 6)
        clock = np.fromfile(stream, '<f8', 4)
        shape = tuple(header[2:])
        values = np.fromfile(stream, '<f8', int(np.prod(shape))).reshape(shape, order='F')
        assert not stream.read(1)
    assert np.isfinite(values).all() and np.isfinite(clock).all()
    return header, clock, values


def verify(case, ranks, profile, pipeline, steps=(2, 4), covered=True):
    products = AIR5_PRODUCTS if profile == 'air5' else CHANNEL_PRODUCTS
    log = (case / 'run.log').read_text()
    rows = re.findall(r'ASTR_INSITU_DEVICE_WALL_GEOMETRY rank=(\d+) step=(\d+) product=(\S+) '
        r'points=(\d+) triangles=(\d+) local_area=(\S+) global_area=(\S+) '
        r'invalid_triangles=0 geometry_host_bytes=0', log)
    expected_rows = 0
    for step in steps:
        for rank in range(ranks):
            receipt = json.loads((case / f'outdat/render/mesh_step{step:08d}_rank{rank}.json').read_text())
            assert receipt['processing_backend'] == 'device'
            assert receipt['rendering_pipeline'] == pipeline and receipt['geometry_host_bytes'] == 0
            step_covered = receipt['statistics']['statistics_duration'] > 0.
            if step == steps[-1]:
                assert step_covered == covered
            expected = set(products) | ({'mean_'+name for name in products} if step_covered else set())
            expected_rows += len(expected)
            assert set(receipt['products']) == expected
            for name in products:
                instant = receipt['products'][name]
                if step_covered:
                    mean = receipt['products']['mean_'+name]
                    assert mean['color_field'] == 'mean_'+name
                    for field in ('color_range', 'local_points', 'local_cells'):
                        assert instant[field] == mean[field]
                    if profile == 'air5':
                        assert instant['field_unit'] == mean['field_unit']
                    a = [r[3:] for r in rows if int(r[0]) == rank and int(r[1]) == step and r[2] == name]
                    b = [r[3:] for r in rows if int(r[0]) == rank and int(r[1]) == step and r[2] == 'mean_'+name]
                    assert a == b
        for name in expected:
            path = case / f'outdat/render/{name}.step{step:08d}.jpeg'
            assert path.with_suffix('.eps').is_file()
            with Image.open(path) as image:
                pixels = np.asarray(image.convert('RGB'))
                assert pixels.shape == (600, 800, 3)
                assert np.count_nonzero(np.any(pixels < 220, axis=-1)) > 500
    assert not list((case / 'outdat/render').glob('*.vtp'))
    assert not list((case / 'outdat/render').glob('*.pvtp'))
    assert len(rows) == expected_rows, log


@pytest.fixture(scope='module')
def reference_cases(tmp_path_factory):
    cases = {}
    def get(profile, topology):
        identity = profile, topology
        if identity not in cases:
            root = tmp_path_factory.mktemp('wall_mean_reference')
            cpu, _ = run(root, profile, topology, 'pinned', 'standard-device', 'cpu', device=False, backend='cpu')
            plain, _ = run(root, profile, topology, 'pinned', 'standard-device', 'plain')
            cases[identity] = cpu, plain
        return cases[identity]
    return get


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('topology', TOPOLOGIES, ids=('single', 'x', 'y', 'z'))
@pytest.mark.parametrize('mode', ('pinned', 'device-aware'))
@pytest.mark.parametrize('pipeline', ('standard-device', 'direct-device'))
def test_mean_fields_images_and_exact_restart(tmp_path, profile, topology, mode, pipeline, reference_cases, record_property):
    ranks = int(np.prod(topology))
    cpu, plain = reference_cases(profile, topology)
    actual, size = run(tmp_path, profile, topology, mode, pipeline, 'means', wall_mean=True, mean_oracle=True)
    verify(actual, ranks, profile, pipeline)
    expected = owned_values(cpu, ranks, profile)
    device_expected = owned_values(actual, ranks, profile)
    scales = FIELD_SCALES if profile == 'air5' else np.ones(3)
    raw = np.zeros(len(scales))
    for rank in range(ranks):
        header, clock, _, _, _, _ = read_statistics(
            actual / f'outdat/render/sample.wall_statistics.step00000004.rank{rank:08d}.bin')
        oh, oc, means = read_oracle(actual / f'outdat/render/wall_mean_oracle.step00000004.rank{rank:08d}.bin')
        np.testing.assert_array_equal(oh[:2], [4, rank])
        np.testing.assert_array_equal(oh[2:5], header[3:6])
        np.testing.assert_array_equal(oc, clock[[0, 3, 1, 2]])
        for index in np.ndindex(means.shape[:3]):
            identity = key(header, index, profile)
            np.testing.assert_array_equal(means[index], device_expected[identity])
            raw = np.maximum(raw, abs(means[index] - expected[identity]))
    assert np.max(raw/scales) <= 2e-10
    for name, error in zip(FIELD_NAMES if profile == 'air5' else CHANNEL_PRODUCTS, raw):
        record_property(name+'_mean_raw_maxabs', float(error))
    record_property('scaled_mean_maxabs', float(np.max(raw/scales)))
    source = actual / 'outdat/new/checkpoints/step000000000003'
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    resumed, _ = run(tmp_path, profile, topology, mode, pipeline, 'resumed', restore=source,
        wall_mean=True, mean_oracle=True)
    verify(resumed, ranks, profile, pipeline, steps=(4,))
    for name in ('state.h5', 'statistics.h5'):
        compare_fields(actual / FINAL / name, resumed / FINAL / name)
        compare_fields(actual / FINAL / name, plain / FINAL / name)
    for name in ('control.bin', 'insitu_control.bin'):
        assert (actual / FINAL / name).read_bytes() == (resumed / FINAL / name).read_bytes()
    for pattern in ('*step00000004*.jpeg', '*step00000004*.eps', '*step00000004*.bin'):
        files = list((actual / 'outdat/render').glob(pattern))
        assert files
        for path in files:
            assert path.read_bytes() == (resumed / 'outdat/render' / path.name).read_bytes()
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    assert size <= 256*1024**2
    record_property('directory_bytes', size)


@pytest.mark.parametrize('profile', PROFILES)
def test_zero_coverage_has_no_fake_mean(tmp_path, profile):
    window = '1.d-8,2.d-8' if profile == 'air5' else '1.d0,2.d0'
    case, _ = run(tmp_path, profile, (1,2,1), 'pinned', 'standard-device', 'uncovered',
        wall_mean=True, window=window)
    verify(case, 2, profile, 'standard-device', covered=False)
    assert not list((case / 'outdat/render').glob('mean_*'))


@pytest.mark.parametrize('profile,topology,mode', (
    ('air5', (1,2,1), 'device-aware'), ('channel', (2,1,1), 'pinned'), ('curve', (1,1,2), 'device-aware')))
def test_mean_memory_safety(tmp_path, profile, topology, mode):
    case, _ = run(tmp_path, profile, topology, mode, 'standard-device', 'memcheck', wall_mean=True, memcheck=True)
    assert not list((case / 'outdat/render').glob('wall_mean_oracle*'))
    reports = list(case.glob('memcheck.*.log'))
    assert len(reports) == 2 and all('ERROR SUMMARY: 0 errors' in p.read_text() for p in reports)
    verify(case, 2, profile, 'standard-device')


@pytest.mark.parametrize('profile,topology,mode', (
    ('air5', (1,2,1), 'device-aware'), ('channel', (2,1,1), 'pinned'), ('curve', (1,1,2), 'device-aware')))
@pytest.mark.parametrize('pipeline', ('standard-device', 'direct-device'))
def test_mean_transfer_trace(tmp_path, profile, topology, mode, pipeline, record_property):
    case, _ = run(tmp_path, profile, topology, mode, pipeline, 'trace', wall_mean=True, nsys=True)
    trace_transfers(case, profile, topology, mode, record_property)
    verify(case, 2, profile, pipeline)


def trace_transfers(case, profile, topology, mode, record_property):
    records = []
    cells = 32 if profile == 'curve' else 16
    im, km = cells//topology[0], cells//topology[2]
    for rank in range(2):
        with sqlite3.connect((case / f'trace.rank{rank}.sqlite').resolve().as_uri()+'?mode=ro', uri=True) as db:
            counts = dict(db.execute('select text,count(*) from NVTX_EVENTS group by text'))
            assert counts['ASTR_X4_DEVICE_WALL_MEAN_SUPPLY'] == 2
            for kind, size in copies(db, 'ASTR_X4_DEVICE_WALL_MEAN_SUPPLY'):
                assert kind != 'CUDA_MEMCPY_KIND_UVM_DTOH'
                nw = (1 if rank == 0 else 0) if profile == 'air5' else 2
                face_sizes = {8*(im+1)*nw*(18 if profile == 'air5' else 4),
                              8*(km+1)*nw*(18 if profile == 'air5' else 4)}
                if kind == 'CUDA_MEMCPY_KIND_DTOH':
                    assert size in ({4, 8} | face_sizes)
                elif kind == 'CUDA_MEMCPY_KIND_HTOD':
                    assert size in face_sizes
                records.append((rank, kind, size))
            if mode == 'device-aware':
                tables = [name for name in ('MPI_P2P_EVENTS', 'MPI_START_WAIT_EVENTS',
                    'MPI_COLLECTIVES_EVENTS', 'MPI_OTHER_EVENTS') if db.execute(
                        "select 1 from sqlite_master where type='table' and name=?", (name,)).fetchone()]
                calls = ' union all '.join(f'select start,end,globalTid from {table}' for table in tables)
                assert calls
                rows = db.execute('with mpi_calls as ('+calls+') '
                    'select e.name,m.bytes,exists(select 1 from CUPTI_ACTIVITY_KIND_RUNTIME r '
                    'where r.correlationId=m.correlationId and exists(select 1 from mpi_calls p '
                    'where p.globalTid=r.globalTid and r.start>=p.start and r.end<=p.end)) '
                    'from CUPTI_ACTIVITY_KIND_MEMCPY m join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind '
                    "where e.name in ('CUDA_MEMCPY_KIND_DTOH','CUDA_MEMCPY_KIND_HTOD') and m.bytes>8 "
                    'and exists(select 1 from NVTX_EVENTS n where n.text=? and m.start>=n.start and m.end<=n.end)',
                    ('ASTR_X4_DEVICE_WALL_MEAN_SUPPLY',)).fetchall()
                assert all(in_mpi and size in face_sizes for _, size, in_mpi in rows), rows
                record_property(f'rank{rank}_mpi_managed_mean_faces', json.dumps(rows))
            consumed = copies(db, 'ASTR_X4_DEVICE_WALL_CONSUMER')
            assert all(size <= 256 for kind, size in consumed if kind == 'CUDA_MEMCPY_KIND_DTOH')
            assert not any(kind == 'CUDA_MEMCPY_KIND_UVM_DTOH' for kind, _ in consumed)
            rendered = copies(db, 'ASTR_IS8_RESIDENT_RENDER')
            assert all(kind == 'CUDA_MEMCPY_KIND_DTOD' for kind, _ in rendered)
            record_property(f'rank{rank}_consumer_copies', json.dumps(consumed))
            record_property(f'rank{rank}_display_device_copies', json.dumps(rendered))
    assert 'ASTR_INSITU_TEST_WALL_MEAN_IO' not in (case / 'run.log').read_text()
    record_property('mean_supply_copies', json.dumps(records))


def test_mean_immutable_trace_attribution(record_property):
    root = os.environ.get('ASTR_INSITU_WALL_MEAN_TRACE_ROOT')
    if not root:
        pytest.skip('Select the unchanged six-case completed mean transfer trace')
    cases = [p for p in Path(root).glob('test_mean_transfer_trace*/gpu_np2_trace')
             if not p.parent.is_symlink()]
    assert len(cases) == 6
    for case in cases:
        receipt = json.loads((case / 'outdat/render/mesh_step00000004_rank0.json').read_text())
        profile = {'air5_walls':'air5', 'channel_walls':'channel', 'curve_walls':'curve'}[receipt['profile']]
        log = (case / 'run.log').read_text()
        topology = tuple(map(int, re.search(r'forced mpi topology=\s*(\d+)\s+(\d+)\s+(\d+)', log).groups()))
        mode = re.search(r'ASTR_INSITU_DEVICE_WALL_FRAME .*?transport=(\S+)', log).group(1)
        trace_transfers(case, profile, topology, mode, record_property)


@pytest.mark.parametrize('profile,topology,mode', (
    ('air5', (1,2,1), 'device-aware'), ('channel', (2,1,1), 'pinned'), ('curve', (1,1,2), 'device-aware')))
@pytest.mark.parametrize('pipeline', ('standard-device', 'direct-device'))
def test_mean_resources_and_short_cost(tmp_path, profile, topology, mode, pipeline, record_property):
    off, _ = run(tmp_path, profile, topology, mode, pipeline, 'off', enabled=False,
        monitor=True, timing=True)
    baseline = json.loads((off / 'resources.sampled.json').read_text())
    on, size = run(tmp_path, profile, topology, mode, pipeline, 'on', wall_mean=True,
        monitor=True, baseline=baseline, timing=True)
    report = json.loads((on / 'resources.sampled.json').read_text())
    assert report['samples'] > 0 and report['sampling_period_seconds'] == .02
    assert report['additional_host_peak_difference_bytes'] <= 4*1024**3
    assert all(value <= 2*1024**3 for value in report['additional_device_peak_difference_bytes'].values())
    assert all(value['min_free_bytes'] >= 1024**3 for value in report['devices'].values())
    assert size <= 256*1024**2
    log = (on / 'run.log').read_text()
    assert 'ASTR_INSITU_TEST_WALL_MEAN_IO' not in log
    rows = re.findall(r'ASTR_INSITU_STAGE_TIMING (\w+) (-?\d+) (\d+)\s+(\S+)', log)
    assert any(name == 'device_wall_mean_supply_inclusive' for name, _, _, _ in rows)
    assert all(np.isfinite(float(value)) and float(value) >= 0. for _, _, _, value in rows)
    record_property('external_20ms_resources', json.dumps(report))
    record_property('short_inclusive_stage_seconds', json.dumps(rows))
    record_property('directory_bytes', size)
    verify(on, 2, profile, pipeline)


def test_mean_partition_and_pipeline_images(record_property):
    root = os.environ.get('ASTR_INSITU_WALL_MEAN_MATRIX_ROOT')
    if not root:
        pytest.skip('Select the immutable completed forty-eight-case mean matrix')
    from scipy.ndimage import distance_transform_edt
    cases = [p for p in Path(root).glob('test_mean_fields_images_and_ex*/gpu_np*_means')
             if not p.parent.is_symlink()]
    assert len(cases) == 48
    records = {}
    for case in cases:
        receipt = json.loads((case / 'outdat/render/mesh_step00000004_rank0.json').read_text())
        log = (case / 'run.log').read_text()
        topology = re.search(r'forced mpi topology=\s*(\d+)\s+(\d+)\s+(\d+)', log).groups()
        transport = re.search(r'ASTR_INSITU_DEVICE_WALL_FRAME .*?transport=(\S+)', log).group(1)
        records.setdefault(receipt['profile'], []).append((case, topology, transport, receipt['rendering_pipeline']))
    for profile, group in records.items():
        assert len(group) == len({(t, m, p) for _, t, m, p in group}) == 16
        names = AIR5_PRODUCTS if profile == 'air5_walls' else CHANNEL_PRODUCTS
        maxima = {}
        for field in names:
            reference, pairs = None, {}
            worst = 0.
            for case, topology, mode, _ in group:
                with Image.open(case / f'outdat/render/mean_{field}.step00000004.jpeg') as image:
                    pixels = np.asarray(image.convert('RGB')).astype(int)[50:550, 50:620]
                # Near-zero signed shear is neutral-colored, not an empty surface.
                mask = pixels.min(2) < 250
                assert mask.sum() > 500
                if reference is None:
                    reference = mask
                error = max(float(distance_transform_edt(~reference)[mask].max()),
                    float(distance_transform_edt(~mask)[reference].max()))
                assert error <= 1., (case, field, error)
                worst = max(worst, error)
                identity = topology, mode
                if identity in pairs:
                    np.testing.assert_array_equal(pixels, pairs.pop(identity))
                else:
                    pairs[identity] = pixels
            assert not pairs
            maxima[field] = worst
        record_property(profile+'_partition_seam_max_pixels', json.dumps(maxima))
        record_property(profile+'_cross_entry_pixel_maxabs', 0)
