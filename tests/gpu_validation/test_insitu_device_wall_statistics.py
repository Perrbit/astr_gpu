"""X4 wall accumulation from resident fields, with archive I/O kept explicit."""
import re
import json
import sqlite3
import os
import hashlib
from pathlib import Path

import numpy as np
import h5py
import pytest

from run_output_air5_restart_validation import launch, DT
from run_output_restart_validation import run_case, compare_fields
from test_insitu_device_air5_fields import current_arguments as air5_arguments
from test_insitu_device_channel_fields import current_arguments as channel_arguments
from test_insitu_device_curve_wall_fields import current_arguments as curve_arguments
from test_insitu_device_air5_render import configuration as air5_render_config
from test_insitu_device_channel_render import configuration as channel_render_config
from test_insitu_wall_scalar_statistics import read_statistics, config as air5_host_config
from test_insitu_channel_statistics import config as channel_host_config
from test_insitu_air5_walls import FIELD_NAMES, FIELD_SCALES, FINAL
from test_output_insitu_restart import ROOT
from test_insitu_device_native_trace import copies


def settings(profile, pipeline, mode, device=True):
    if profile == 'air5':
        text = air5_render_config(pipeline, mode) if device else air5_host_config()
        window = 'statistics_window=5.d-11,3.5d-10'
    else:
        text = channel_render_config(pipeline, mode) if device else channel_host_config()
        window = 'statistics_window=0.003,0.0115'
    text = text.replace('statistics=.false.', 'statistics=.true.')
    for old in ('statistics_window=0.0005,0.0115', 'statistics_window=0.003,0.0115',
                'statistics_window=5.d-11,3.5d-10'):
        text = text.replace(old, window)
    return text


def run(output, profile, topology, mode, pipeline, label, device=True, backend='gpu', restore=None,
        memcheck=False, enabled=True, nsys=False, monitor=False, baseline=None,
        wall_mean=False, mean_oracle=False, window=None, timing=False, separation=False):
    ranks = int(np.prod(topology))
    config = settings(profile, pipeline, mode, device) if enabled else '&insitu_run enabled=f /\n'
    if wall_mean:
        config = config.replace('statistics=.true.', 'statistics=.true.,wall_mean_render=t')
    if separation:
        config = config.replace('statistics=.true.', 'statistics=.true.,wall_separation=t')
    if window is not None:
        config = re.sub(r'statistics_window=[^,\s]+,\s*[^,\s]+', 'statistics_window='+window, config)
    if profile == 'air5':
        args = air5_arguments(output, topology)
        args.initial_restart = True
        if memcheck:
            args.runtime_timeout_seconds = 600
        return launch(args, backend, ranks, label, 4, interval=1, restore=restore,
            insitu_config=config, postprocess_transport=mode if device else None,
            memcheck=memcheck, directory_budget_bytes=256 * 1024**2,
            nsys_trace=nsys, monitor_resources=monitor, resource_baseline=baseline, wall_mean_oracle=mean_oracle,
            insitu_timing=timing)
    if profile == 'curve':
        axis = 'xyz'[int(np.argmax(topology))]
        args = curve_arguments(output, backend, axis, samples=False)
        grid = dict(grid='32,32,32', tgv_mapping='y-wavy')
    else:
        args = channel_arguments(output, samples=False)
        grid = {}
    args.executable = ROOT / 'build_insitu_air5_device_render/bin/astr'
    if memcheck:
        args.runtime_timeout_seconds = 600
    args.statistics = enabled
    return run_case(args, ROOT, backend, ranks, label, 4, topology=topology, checkpoint_interval=1,
        restore=restore, insitu_config=config, postprocess_transport=mode if device else None,
        memcheck=memcheck, nsys_trace=nsys, monitor_resources=monitor, resource_baseline=baseline,
        wall_mean_oracle=mean_oracle, insitu_timing=timing, **grid)


@pytest.mark.parametrize('profile', ('air5', 'channel', 'curve'))
@pytest.mark.parametrize('topology', ((1,1,1), (2,1,1), (1,2,1), (1,1,2)), ids=('single','x','y','z'))
@pytest.mark.parametrize('pipeline,mode', (('standard-device','pinned'), ('direct-device','device-aware')))
def test_resident_wall_statistics_and_restart(tmp_path, profile, topology, pipeline, mode, record_property):
    ranks = int(np.prod(topology))
    actual, size = run(tmp_path, profile, topology, mode, pipeline, 'device')
    cpu, _ = run(tmp_path, profile, topology, mode, pipeline, 'cpu', device=False, backend='cpu')
    scales = FIELD_SCALES if profile == 'air5' else np.ones(3)
    worst = 0.
    for rank in range(ranks):
        name = f'outdat/render/sample.wall_statistics.step00000004.rank{rank:08d}.bin'
        a, b = read_statistics(actual / name), read_statistics(cpu / name)
        for index in (0, 1, 3, 5):
            np.testing.assert_array_equal(a[index], b[index])
        assert a[2] == b[2] == 5
        nf = len(scales)
        if a[4].size:
            mean = float(np.max(abs(a[4][...,:nf] - b[4][...,:nf]) / scales))
            variance = float(np.max(abs(a[4][...,nf:2*nf] - b[4][...,nf:2*nf]) / scales**2))
            worst = max(worst, mean, variance)
            assert worst <= 2e-10, (profile, rank, mean, variance)
            record_property(f'rank{rank}_rms_raw_maxabs', float(np.max(abs(a[4][...,2*nf:] - b[4][...,2*nf:]))))
    log = (actual / 'run.log').read_text()
    samples = re.findall(r'ASTR_INSITU_WALL_STATS rank=(\d+) step=(\d+) samples=(\d+) field_download_bytes=(\d+)', log)
    assert len(samples) == 5 * ranks and all(row[3] == '0' for row in samples), log
    assert 'ASTR_INSITU_DEVICE_WALL_STATS' in log and 'ASTR_INSITU_DEVICE_WALL_CHECK' not in log
    source = actual / 'outdat/new/checkpoints/step000000000003'
    before = {p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
    resumed, _ = run(tmp_path, profile, topology, mode, pipeline, 'restart', restore=source)
    for name in ('state.h5', 'statistics.h5'):
        compare_fields(actual / FINAL / name, resumed / FINAL / name)
    for name in ('control.bin', 'insitu_control.bin'):
        assert (actual / FINAL / name).read_bytes() == (resumed / FINAL / name).read_bytes()
    if profile == 'air5':
        for name in ('air5_config.bin', 'air5_conservation.bin'):
            assert (actual / FINAL / name).read_bytes() == (resumed / FINAL / name).read_bytes()
    for rank in range(ranks):
        name = f'outdat/render/sample.wall_statistics.step00000004.rank{rank:08d}.bin'
        assert (actual / name).read_bytes() == (resumed / name).read_bytes()
    assert before == {p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
    plain, _ = run(tmp_path, profile, topology, mode, pipeline, 'off', enabled=False)
    compare_fields(actual / FINAL / 'state.h5', plain / FINAL / 'state.h5')
    record_property('reference_scaled_maxabs', worst)
    record_property('directory_bytes', size)
    record_property('dt', DT if profile == 'air5' else .001)


@pytest.mark.parametrize('profile,topology,mode', (
    ('air5',(1,2,1),'device-aware'), ('channel',(1,2,1),'pinned'), ('curve',(2,1,1),'device-aware')))
def test_resident_wall_statistics_memory_safety(tmp_path, profile, topology, mode):
    case, _ = run(tmp_path, profile, topology, mode, 'standard-device', 'memcheck', memcheck=True)
    reports = list(case.glob('memcheck.*.log'))
    assert len(reports) == 2
    assert all('ERROR SUMMARY: 0 errors' in p.read_text() for p in reports)


@pytest.mark.parametrize('profile,topology,mode', (
    ('air5',(1,2,1),'device-aware'), ('channel',(2,1,1),'pinned'), ('curve',(1,1,2),'device-aware')))
def test_resident_wall_statistics_transfer_trace(tmp_path, profile, topology, mode, record_property):
    case, _ = run(tmp_path, profile, topology, mode, 'standard-device', 'trace', nsys=True)
    records = []
    for rank in range(2):
        with sqlite3.connect((case / f'trace.rank{rank}.sqlite').resolve().as_uri() + '?mode=ro', uri=True) as db:
            counts = dict(db.execute('select text,count(*) from NVTX_EVENTS group by text'))
            assert counts['ASTR_X4_DEVICE_WALL_STATISTICS'] == 5
            transfers = copies(db, 'ASTR_X4_DEVICE_WALL_STATISTICS')
            assert not any(kind == 'CUDA_MEMCPY_KIND_UVM_DTOH' for kind, _ in transfers)
            face_sizes = set()
            im, km = (16, 16) if profile != 'curve' else (32, 32)
            im //= topology[0]; km //= topology[2]
            nw = 1 if profile == 'air5' or topology[1] == 2 else 2
            components = 33 if profile == 'air5' else 15
            face_sizes.update(((im + 1) * nw * components * 8, (km + 1) * nw * components * 8))
            if profile == 'curve':
                # Tangential primitive halo faces are part of the approved CURVE provider.
                face_sizes.update(((im + 1) * nw * 4 * 3 * 8, (km + 1) * nw * 4 * 3 * 8))
            controls = db.execute('select e.name,m.bytes,s.value from CUPTI_ACTIVITY_KIND_MEMCPY m '
                'join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind '
                'join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId=m.correlationId '
                'join StringIds s on s.id=r.nameId '
                'where exists(select 1 from NVTX_EVENTS n where n.text=? '
                'and m.start>=n.start and m.end<=n.end) '
                "and e.name='CUDA_MEMCPY_KIND_HTOD' and m.bytes>144",
                ('ASTR_X4_DEVICE_WALL_STATISTICS',)).fetchall()
            # NVHPC binds the saved module arrays once. This symbol upload is
            # separate from scalar status D2H and the declared pinned faces.
            bindings = [(kind,n,api) for kind,n,api in controls
                if n == 288 and api == 'cudaMemcpyToSymbol_v3020']
            assert len(bindings) <= 1, (profile,rank,bindings)
            assert all(n in face_sizes or (kind,n,api) in bindings
                for kind,n,api in controls), (profile,rank,controls,face_sizes)
            assert all(n in (4,8) or n in face_sizes
                for kind,n in transfers if kind == 'CUDA_MEMCPY_KIND_DTOH'), (profile,rank,transfers)
            if profile != 'air5':
                assert counts['ASTR_X4_DEVICE_VOLUME_STATISTICS'] == 5
                volume = copies(db, 'ASTR_X4_DEVICE_VOLUME_STATISTICS')
                assert [copy for copy in volume if copy[0] == 'CUDA_MEMCPY_KIND_DTOH'] == [
                    ('CUDA_MEMCPY_KIND_DTOH',4)] * 5, (rank,volume)
                assert all(kind == 'CUDA_MEMCPY_KIND_DTOH' or
                    (kind == 'CUDA_MEMCPY_KIND_HTOD' and n <= 144) for kind,n in volume), (rank,volume)
            else:
                volume = []
            records.append(dict(rank=rank, wall_sample_transfers=transfers,
                one_time_symbol_bindings=bindings, volume_status_transfers=volume))
    record_property('per_sample_transfer_ledger', json.dumps(records))
    # Statistics/checkpoint export is deliberate I/O outside the audited sample spans.
    assert (case / FINAL / 'statistics.h5').is_file()


@pytest.mark.parametrize('profile,topology,mode', (
    ('air5',(1,2,1),'device-aware'), ('channel',(2,1,1),'pinned'), ('curve',(1,1,2),'device-aware')))
def test_resident_wall_statistics_resources(tmp_path, profile, topology, mode, record_property):
    off, _ = run(tmp_path, profile, topology, mode, 'standard-device', 'resource_off', enabled=False, monitor=True)
    baseline = json.loads((off / 'resources.sampled.json').read_text())
    on, size = run(tmp_path, profile, topology, mode, 'standard-device', 'resource_on', monitor=True, baseline=baseline)
    report = json.loads((on / 'resources.sampled.json').read_text())
    assert report['samples'] > 0 and report['sampling_period_seconds'] == .02
    assert report['additional_host_peak_difference_bytes'] <= 4 * 1024**3
    assert all(n <= 2 * 1024**3 for n in report['additional_device_peak_difference_bytes'].values())
    assert all(d['min_free_bytes'] >= 1024**3 for d in report['devices'].values())
    assert size <= 256 * 1024**2
    record_property('external_20ms_resources', json.dumps(report))
    record_property('directory_bytes', size)


def test_existing_volume_statistics_unchanged(record_property):
    root = os.environ.get('ASTR_INSITU_WALL_STATS_MATRIX_ROOT')
    if not root:
        pytest.skip('Select the completed resident wall-statistics matrix for read-only comparison')
    cases = [p for p in Path(root).glob('test_resident_wall_statistics_*') if not p.is_symlink()]
    worst = 0.; checked = 0
    for folder in cases:
        for actual in folder.glob('gpu_np*_device'):
            ranks = int(actual.name.split('_')[1][2:])
            reference = folder / f'cpu_np{ranks}_cpu'
            resumed = folder / f'gpu_np{ranks}_restart'
            with h5py.File(actual / FINAL / 'statistics.h5') as state:
                volume_enabled = 'wall_statistics' in state
            if not volume_enabled:
                continue  # AIR5 wall scalars do not implicitly enable volume statistics.
            # Strict rendering deliberately does not export full-volume statistics.
            # Compare the explicit checkpoint's owned (z,y,x) nodes instead, with
            # periodic x/z endpoints excluded and both physical y walls retained.
            with h5py.File(actual / FINAL / 'statistics.h5') as gpu, \
                    h5py.File(reference / FINAL / 'statistics.h5') as cpu:
                for component in range(1,35):
                    name = f'q{component:04d}'
                    error = float(np.max(abs(gpu[name][:-1,:,:-1] - cpu[name][:-1,:,:-1])))
                    assert error <= 2e-10, (actual, component, error)
                    worst = max(worst, error)
            compare_fields(actual / FINAL / 'statistics.h5', resumed / FINAL / 'statistics.h5')
            checked += 1
    assert checked == 16, checked
    record_property('unchanged_velocity_statistics_cases', checked)
    record_property('unchanged_velocity_statistics_maxabs', worst)


def test_existing_wall_statistics_raw_and_scaled_errors(record_property):
    root = os.environ.get('ASTR_INSITU_WALL_STATS_MATRIX_ROOT')
    if not root:
        pytest.skip('Select the completed resident wall-statistics matrix for read-only comparison')
    reports = {}; checked = 0
    for folder in Path(root).glob('test_resident_wall_statistics_*'):
        if folder.is_symlink():
            continue
        for actual in folder.glob('gpu_np*_device'):
            ranks = int(actual.name.split('_')[1][2:])
            reference = folder / f'cpu_np{ranks}_cpu'
            for rank in range(ranks):
                name = f'outdat/render/sample.wall_statistics.step00000004.rank{rank:08d}.bin'
                a, b = read_statistics(actual / name), read_statistics(reference / name)
                nf = int(a[0][9])
                assert nf in (3,18)
                profile = 'air5' if nf == 18 else 'channel-or-curve'
                scales = FIELD_SCALES if nf == 18 else np.ones(3)
                names = FIELD_NAMES if nf == 18 else ('pressure','shear','heat')
                report = reports.setdefault(profile, dict(names=list(names), scales=scales.tolist(),
                    mean_raw_maxabs=np.zeros(nf), variance_raw_maxabs=np.zeros(nf), rms_raw_maxabs=np.zeros(nf)))
                for index in (0,1,3,5):
                    np.testing.assert_array_equal(a[index], b[index])
                assert a[2] == b[2] == 5
                if a[4].size:
                    for offset,key in enumerate(('mean_raw_maxabs','variance_raw_maxabs','rms_raw_maxabs')):
                        error = np.max(abs(a[4][...,offset*nf:(offset+1)*nf] -
                            b[4][...,offset*nf:(offset+1)*nf]),axis=(0,1,2))
                        report[key] = np.maximum(report[key], error)
            checked += 1
    assert checked == 24, checked
    for profile,report in reports.items():
        scales = np.array(report['scales'])
        report['mean_scaled_maxabs'] = float(np.max(report['mean_raw_maxabs']/scales))
        report['variance_scaled_maxabs'] = float(np.max(report['variance_raw_maxabs']/scales**2))
        assert report['mean_scaled_maxabs'] <= 2e-10
        assert report['variance_scaled_maxabs'] <= 2e-10
        for key in ('mean_raw_maxabs','variance_raw_maxabs','rms_raw_maxabs'):
            report[key] = report[key].tolist()
        record_property(profile+'_per_field_statistic_errors',json.dumps(report))
    record_property('wall_statistics_cases', checked)


@pytest.mark.parametrize('profile', ('air5','channel','curve'))
@pytest.mark.parametrize('topology', ((1,1,1),(2,1,1),(1,2,1),(1,1,2)), ids=('single','x','y','z'))
@pytest.mark.parametrize('pipeline,mode', (('standard-device','pinned'),('direct-device','device-aware')))
def test_statistics_kernel_shape_regression(tmp_path,profile,topology,pipeline,mode,record_property):
    root = os.environ.get('ASTR_INSITU_WALL_STATS_MATRIX_ROOT')
    if not root:
        pytest.skip('Select the immutable pre-shape-change matrix for exact regression')
    index = (12 if mode == 'device-aware' else 0) + (
        ((1,1,1),(2,1,1),(1,2,1),(1,1,2)).index(topology)*3) + ('air5','channel','curve').index(profile)
    ranks = int(np.prod(topology))
    reference = Path(root) / f'test_resident_wall_statistics_{index}' / f'gpu_np{ranks}_device'
    assert reference.is_dir(), reference
    actual, _ = run(tmp_path,profile,topology,mode,pipeline,'shape_regression')
    before = {p.relative_to(reference/'datin'):p for p in (reference/'datin').rglob('*') if p.is_file()}
    after = {p.relative_to(actual/'datin'):p for p in (actual/'datin').rglob('*') if p.is_file()}
    assert before.keys() == after.keys(), (reference,actual)
    for name in before:
        if name == Path('grid.h5'):
            compare_fields(before[name],after[name])
        else:
            assert before[name].read_bytes() == after[name].read_bytes(), name
    for name in ('state.h5','statistics.h5'):
        compare_fields(actual/FINAL/name,reference/FINAL/name)
    old_control = (reference/FINAL/'control.bin').read_bytes()
    new_control = (actual/FINAL/'control.bin').read_bytes()
    # Cross-binary regression excludes only the executable size/CRC in
    # ASTROC04 contract(1:2); same-binary restart still compares every byte.
    assert old_control[:8] == new_control[:8] == b'ASTROC04'
    assert old_control[24:] == new_control[24:]
    assert (actual/FINAL/'insitu_control.bin').read_bytes() == (reference/FINAL/'insitu_control.bin').read_bytes()
    for rank in range(ranks):
        name = f'outdat/render/sample.wall_statistics.step00000004.rank{rank:08d}.bin'
        assert (actual/name).read_bytes() == (reference/name).read_bytes()
    record_property('immutable_reference',str(reference))
    record_property('reference_executable_sha256','cbd3ec8e9dbf5948477ea429e79b96141315c85c3b447ac4241987d91775003c')
    record_property('executable_sha256',hashlib.sha256((ROOT/'build_insitu_air5_device_render/bin/astr').read_bytes()).hexdigest())
