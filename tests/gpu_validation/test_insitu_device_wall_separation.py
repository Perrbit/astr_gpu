"""Resident physical-span reduction; the AIR5 fixture is not a separation bubble."""
import csv
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess

import h5py
import numpy as np
import pytest

from test_insitu_device_wall_statistics import run, copies, compare_fields, FINAL
from test_insitu_device_air5_render import validate
from test_insitu_air5_walls import FIELD_SCALES
from test_output_insitu_restart import ROOT, MPIEXEC

TOPOLOGIES = ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2))
PROBE = Path(os.environ.get('ASTR_INSITU_FIELDS_PROBE',
    ROOT / 'build_insitu_air5_device_render/bin/insitu_device_fields_probe'))


def launch_probe(ranks, axis, mode, scenario=''):
    prefix = Path(os.environ.get('ASTR_INSITU_DEVICE_MPI_PREFIX', MPIEXEC.parent.parent))
    env = dict(os.environ, OMPI_MCA_pml='ucx', OMPI_MCA_coll='^hcoll,ucc,cuda',
        OMPI_MCA_coll_hcoll_enable='0', UCX_MEMTYPE_CACHE='n',
        UCX_CUDA_COPY_ENABLE_FABRIC='no', UCX_CUDA_COPY_DMABUF='no',
        UCX_CUDA_IPC_ENABLE_MNNVL='no', UCX_TLS='self,sm,cuda_copy,cuda_ipc')
    return subprocess.run([str(prefix / 'bin/mpirun'), '--prefix', str(prefix), '-np', str(ranks),
        str(PROBE), str(axis), mode, scenario], env=env, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize('ranks,axis', ((1, 1), (2, 1), (2, 2), (2, 3)))
@pytest.mark.parametrize('mode', ('pinned', 'device-aware'))
def test_synthetic_device_span_profile(ranks, axis, mode, record_property):
    result = launch_probe(ranks, axis, mode)
    text = result.stdout + result.stderr
    assert result.returncode == 0, text
    error = float(re.search(r'PASS device physical-span shear profile max_error=\s*(\S+)', text).group(1))
    assert np.isfinite(error) and error <= 2e-10
    record_property('synthetic_physical_span_integral_maxabs', error)


@pytest.mark.parametrize('scenario,message', (
    ('wall-weight', 'wall shear reduction weights/budget'),
    ('wall-nonfinite', 'nonfinite wall shear integral'),
    ('wall-budget', 'wall shear reduction allocation budget')))
def test_reduction_rejection(scenario, message):
    result = launch_probe(2, 2, 'pinned', scenario)
    assert result.returncode != 0, result.stdout + result.stderr
    assert message in result.stdout + result.stderr


def profiles(case):
    paths = sorted((case / 'outdat/render').glob('sample.wall_separation.step*.csv'))
    assert paths
    records = {}
    for path in paths:
        lines = path.read_text().splitlines()
        assert lines[0] == '# Cartesian bottom wall; tangent=+x; z average=physical length; no periodic x wrap'
        metadata = lines[1].split()
        assert metadata[-1] == '0' and float(metadata[-3]) == 0.
        assert abs(float(metadata[-2]) - .002) <= 2e-17
        rows = list(csv.DictReader(line for line in lines if not line.startswith('#')))
        assert len(rows) == 17 and all(row['record'] == 'profile' for row in rows)
        assert all(row['classification'] == 'profile' and
            row['status'] == 'not_applicable_no_positive_inflow' for row in rows)
        values = np.array([[float(row[key]) for key in ('time', 'x_lower', 'x_upper', 'value')] for row in rows])
        assert np.isfinite(values).all()
        np.testing.assert_array_equal(values[:, 1], values[:, 2])
        assert np.all(np.diff(values[:, 1]) > 0.)
        step = int(rows[0]['step'])
        assert all(int(row['step']) == step for row in rows)
        records[step] = values
    return records


def compare_statistics_isolation(left, right):
    with h5py.File(left) as a, h5py.File(right) as b:
        assert set(a) == set(b)
        for name in a:
            aa, bb = a[name][:], b[name][:]
            assert aa.dtype == bb.dtype and aa.shape == bb.shape
            if name == 'metadata':
                assert aa.shape == (12,) and aa[11] == 1 and bb[11] == 0
                assert aa[:11].tobytes() == bb[:11].tobytes()
            else:
                assert aa.tobytes() == bb.tobytes(), name


@pytest.fixture(scope='module')
def references(tmp_path_factory):
    saved = {}
    def get(topology):
        if topology not in saved:
            root = tmp_path_factory.mktemp('separation_reference')
            cpu, _ = run(root, 'air5', topology, 'pinned', 'standard-device', 'cpu',
                device=False, backend='cpu', separation=True)
            host_gpu, _ = run(root, 'air5', topology, 'pinned', 'standard-device', 'host_gpu',
                device=False, separation=True)
            off, _ = run(root, 'air5', topology, 'pinned', 'standard-device', 'off')
            saved[topology] = cpu, host_gpu, off
        return saved[topology]
    return get


@pytest.mark.parametrize('topology', TOPOLOGIES, ids=('single', 'x', 'y', 'z'))
@pytest.mark.parametrize('mode', ('pinned', 'device-aware'))
@pytest.mark.parametrize('pipeline', ('standard-device', 'direct-device'))
def test_actual_profile_and_exact_restart(tmp_path, topology, mode, pipeline, references, record_property):
    cpu, host_gpu, off = references(topology)
    actual, size = run(tmp_path, 'air5', topology, mode, pipeline, 'separation', separation=True)
    a, c, h = profiles(actual), profiles(cpu), profiles(host_gpu)
    assert set(a) == set(c) == set(h) == set(range(5))
    raw = same_state = coordinate_error = 0.
    for step in a:
        np.testing.assert_allclose(a[step][:, :3], c[step][:, :3], rtol=0, atol=2e-10)
        np.testing.assert_array_equal(a[step][:, 0], h[step][:, 0])
        np.testing.assert_allclose(a[step][:, 1:3], h[step][:, 1:3], rtol=0, atol=2e-10)
        coordinate_error = max(coordinate_error, float(np.max(abs(a[step][:, 1:3] - h[step][:, 1:3]))))
        raw = max(raw, float(np.max(abs(a[step][:, 3] - c[step][:, 3]))))
        same_state = max(same_state, float(np.max(abs(a[step][:, 3] - h[step][:, 3]))))
    assert raw / FIELD_SCALES[12] <= 2e-10 and same_state / FIELD_SCALES[12] <= 2e-10
    record_property('profile_shear_cpu_gpu_raw_SI_maxabs', raw)
    record_property('profile_shear_cpu_gpu_scaled_maxabs', raw / FIELD_SCALES[12])
    record_property('profile_shear_same_gpu_state_host_reduce_raw_maxabs', same_state)
    record_property('profile_coordinate_cross_implementation_maxabs', coordinate_error)
    source = actual / 'outdat/new/checkpoints/step000000000003'
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    resumed, _ = run(tmp_path, 'air5', topology, mode, pipeline, 'resumed', restore=source, separation=True)
    assert set(profiles(resumed)) == {4}
    for name in ('state.h5', 'statistics.h5'):
        compare_fields(actual / FINAL / name, resumed / FINAL / name)
    compare_fields(actual / FINAL / 'state.h5', off / FINAL / 'state.h5')
    compare_statistics_isolation(actual / FINAL / 'statistics.h5', off / FINAL / 'statistics.h5')
    for name in ('control.bin', 'insitu_control.bin'):
        assert (actual / FINAL / name).read_bytes() == (resumed / FINAL / name).read_bytes()
    name = 'outdat/render/sample.wall_separation.step00000004.csv'
    assert (actual / name).read_bytes() == (resumed / name).read_bytes()
    for suffix in ('jpeg', 'eps'):
        for path in (actual / 'outdat/render').glob('*step00000004*.' + suffix):
            assert path.read_bytes() == (resumed / 'outdat/render' / path.name).read_bytes()
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    validate(actual, topology, (2, 4), pipeline, mode)
    assert size <= 256 * 1024**2
    record_property('directory_bytes', size)


@pytest.mark.parametrize('mode,pipeline', (('pinned', 'standard-device'), ('device-aware', 'direct-device')))
def test_separation_and_mean_together(tmp_path, mode, pipeline):
    from test_insitu_device_wall_mean_render import verify
    case, _ = run(tmp_path, 'air5', (1, 2, 1), mode, pipeline, 'combined', separation=True, wall_mean=True)
    assert set(profiles(case)) == set(range(5))
    verify(case, 2, 'air5', pipeline)


@pytest.mark.parametrize('topology,mode', (((1, 2, 1), 'device-aware'), ((1, 1, 2), 'pinned')))
def test_actual_profile_memcheck(tmp_path, topology, mode):
    case, _ = run(tmp_path, 'air5', topology, mode, 'standard-device', 'memcheck', separation=True, memcheck=True)
    assert set(profiles(case)) == set(range(5))
    logs = list(case.glob('memcheck.*.log'))
    assert len(logs) == 2 and all('ERROR SUMMARY: 0 errors' in path.read_text() for path in logs)


@pytest.mark.parametrize('pipeline', ('standard-device', 'direct-device'))
@pytest.mark.parametrize('topology,mode', (((1, 2, 1), 'device-aware'), ((1, 1, 2), 'pinned')))
def test_actual_profile_transfer(tmp_path, topology, mode, pipeline, record_property):
    case, _ = run(tmp_path, 'air5', topology, mode, pipeline, 'trace', separation=True, nsys=True)
    assert set(profiles(case)) == set(range(5))
    for rank in range(2):
        with sqlite3.connect((case / f'trace.rank{rank}.sqlite').resolve().as_uri() + '?mode=ro', uri=True) as db:
            counts = dict(db.execute('select text,count(*) from NVTX_EVENTS group by text'))
            assert counts['ASTR_X4_DEVICE_WALL_SEPARATION_REDUCE'] == 5
            assert 'ASTR_IS8_COMPACT_GEOMETRY_READ' not in counts
            assert 'ASTR_IS8_COMPACT_TRACE_READ' not in counts
            rows = copies(db, 'ASTR_X4_DEVICE_WALL_SEPARATION_REDUCE')
            empty = topology == (1, 2, 1) and rank == 1
            nx = 16 // topology[0] + 1; km = 16 // topology[2]
            if empty:
                assert not rows
            else:
                assert all(size in (4, 24 * nx) for kind, size in rows if kind == 'CUDA_MEMCPY_KIND_DTOH')
                assert all(size in (4, 8 * km) for kind, size in rows if kind == 'CUDA_MEMCPY_KIND_HTOD')
                assert [size for kind, size in rows if kind == 'CUDA_MEMCPY_KIND_DTOH'].count(24 * nx) == 5
            assert not any(kind == 'CUDA_MEMCPY_KIND_UVM_DTOH' for kind, _ in rows)
            record_property(f'rank{rank}_one_dimensional_profile_copies', json.dumps(rows))
            assert not db.execute('select m.bytes from CUPTI_ACTIVITY_KIND_MEMCPY m '
                'join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind '
                "where e.name='CUDA_MEMCPY_KIND_UVM_DTOH'").fetchall()
            consumer = copies(db, 'ASTR_X4_DEVICE_WALL_CONSUMER')
            assert all(size <= 256 for kind, size in consumer if kind == 'CUDA_MEMCPY_KIND_DTOH')
            display = copies(db, 'ASTR_IS8_RESIDENT_RENDER')
            assert all(kind == 'CUDA_MEMCPY_KIND_DTOD' for kind, _ in display)
            assert bool(display) == (not empty)
            record_property(f'rank{rank}_consumer_and_display_copies', json.dumps(dict(
                consumer=consumer, display=display)))


@pytest.mark.parametrize('pipeline', ('standard-device', 'direct-device'))
@pytest.mark.parametrize('topology,mode', (((1, 2, 1), 'device-aware'), ((1, 1, 2), 'pinned')))
def test_actual_profile_resources(tmp_path, topology, mode, pipeline, record_property):
    off, _ = run(tmp_path, 'air5', topology, mode, pipeline, 'off', enabled=False, monitor=True, timing=True)
    baseline = json.loads((off / 'resources.sampled.json').read_text())
    case, size = run(tmp_path, 'air5', topology, mode, pipeline, 'on', separation=True,
        monitor=True, baseline=baseline, timing=True)
    report = json.loads((case / 'resources.sampled.json').read_text())
    assert report['samples'] > 0 and report['sampling_period_seconds'] == .02
    assert report['additional_host_peak_difference_bytes'] <= 4 * 1024**3
    assert all(value <= 2 * 1024**3 for value in report['additional_device_peak_difference_bytes'].values())
    assert all(value['min_free_bytes'] >= 1024**3 for value in report['devices'].values())
    assert size <= 256 * 1024**2
    record_property('external_20ms_resources', json.dumps(report))
    record_property('directory_bytes', size)
    assert set(profiles(case)) == set(range(5))
