"""X4 strict images: external peaks, matched short cost, and attributed transfers."""
import json
import os
import re
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from run_output_restart_validation import run_case
from test_insitu_curve_derivatives import ROOT
from test_insitu_device_channel_fields import current_arguments as channel_arguments
from test_insitu_device_channel_render import configuration as channel_configuration
from test_insitu_device_curve_wall_fields import current_arguments as curve_arguments
from test_insitu_device_curve_wall_render import configuration as curve_configuration
from test_insitu_device_physical_plane import configuration as plane_configuration
from test_insitu_device_curve_surface import configuration as surface_configuration
from test_insitu_device_native_trace import copies

pytestmark = pytest.mark.skipif(os.environ.get('ASTR_INSITU_X4_OBSERVE') != '1',
    reason='Select the approved, numerically validated X4 executable for observation')


def settings(path, family, pipeline, mode, topology=(2, 1, 1)):
    if family == 'cartesian-wall':
        args = channel_arguments(path, samples=False)
        config = channel_configuration(pipeline, mode)
        kwargs = {}
    else:
        axis = 'xyz'[topology.index(2)] if 2 in topology else 'x'
        args = curve_arguments(path, axis=axis, samples=False)
        provider = (curve_configuration if family == 'curve-wall' else
                    surface_configuration if family == 'curve-surface' else plane_configuration)
        config = provider(pipeline, mode)
        mapping = None if family == 'cartesian-plane' else ('y-wavy' if family == 'curve-wall' else 'periodic')
        kwargs = dict(grid='32,32,32', tgv_mapping=mapping)
    args.statistics = False
    args.runtime_timeout_seconds = 300
    args.nsys_trace_domains = 'cuda,nvtx,mpi'
    kwargs.update(enabled=False, checkpoint_enabled=False, insitu_timing=True, topology=topology)
    return args, config, kwargs


def no_large_io(case):
    output = case / 'outdat'
    prohibited = [p for p in output.rglob('*') if p.is_file() and
                  (p.suffix in ('.h5', '.vtp', '.pvtp', '.vtu', '.pvtu', '.bin') or p.name == 'COMPLETE')]
    assert not prohibited, prohibited
    log = (case / 'run.log').read_text()
    for marker in ('ASTR_INSITU_PLANE_TEST_ORACLE', 'ASTR_INSITU_DEVICE_WALL_CHECK',
                   'ASTR_INSITU_CURVE_Q_TEST_ORACLE'):
        assert marker not in log


def timing(case, ranks=2):
    stages = {}
    for name, step, rank, value in re.findall(
            r'ASTR_INSITU_STAGE_TIMING (\w+) (-?\d+) (\d+)\s+(\S+)', (case / 'run.log').read_text()):
        value = float(value)
        assert np.isfinite(value) and value >= 0.
        stages.setdefault(name, {}).setdefault(rank, []).append(value)
    assert 'completed_window' in stages and set(stages['completed_window']) == {str(r) for r in range(ranks)}
    assert all(len(v) == 1 for v in stages['completed_window'].values())
    return {name: dict(seconds=max(sum(v) for v in ranks.values()), per_rank_seconds=ranks)
            for name, ranks in stages.items()}


def ledger(case, family, mode, ranks=2):
    log = (case / 'run.log').read_text()
    pixels = [dict(re.findall(r'(\w+)=([^ ]+)', row)) for row in
              re.findall(r'ASTR_INSITU_PIXEL_READ ([^\n]+)', log)]
    assert pixels
    kind = 'VOLUME' if family == 'curve-surface' else ('PLANE' if family.endswith('-plane') else 'WALL')
    sample = 'ASTR_IS8_DEVICE_SAMPLE' if kind == 'VOLUME' else f'ASTR_X4_DEVICE_{kind}_SAMPLE'
    consumer = f'ASTR_X4_DEVICE_{kind}_CONSUMER'
    result = []
    for rank in range(ranks):
        receipt = json.loads((case / f'outdat/render/mesh_step00000002_rank{rank}.json').read_text())
        marker = 'ASTR_INSITU_DEVICE_FRAME' if kind == 'VOLUME' else f'ASTR_INSITU_DEVICE_{kind}_FRAME'
        face = re.search(rf'{marker} rank={rank} step=2 transport={mode} '
            r'face_d2h_bytes=(\d+) face_h2d_bytes=(\d+) face_exchanges=(\d+)', log)
        assert face and receipt['geometry_host_bytes'] == 0
        path = case / f'trace.rank{rank}.sqlite'
        assert path.is_file()
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True) as db:
            available = {name for name, in db.execute("select name from sqlite_master where type='table'")}
            if ranks > 1:
                assert 'MPI_P2P_EVENTS' in available
            counts = dict(db.execute('select text,count(*) from NVTX_EVENTS group by text'))
            assert counts[sample] == counts[consumer] == counts['ASTR_IS8_RESIDENT_RENDER'] == 1
            assert 'ASTR_IS8_COMPACT_GEOMETRY_READ' not in counts
            assert 'ASTR_IS8_COMPACT_TRACE_READ' not in counts
            assert not db.execute('select m.bytes from CUPTI_ACTIVITY_KIND_MEMCPY m '
                'join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind '
                "where e.name='CUDA_MEMCPY_KIND_UVM_DTOH'").fetchall()
            transfers = copies(db, sample)
            staged = {}
            mpi_staged = []
            for direction, column in (('DTOH', 1), ('HTOD', 2)):
                payloads = [n for k, n in transfers if k == 'CUDA_MEMCPY_KIND_'+direction and n > 144]
                staged[direction] = sum(payloads)
                if mode == 'pinned':
                    assert staged[direction] == int(face[column]), (rank, mode, transfers, face.groups())
            if mode == 'pinned':
                assert not any(k == 'CUDA_MEMCPY_KIND_PTOP' for k, _ in transfers)
            else:
                assert int(face[1]) == int(face[2]) == 0
                # A device-buffer MPI call may stage its face payload internally.
                # Attribute by CUDA API correlation and the same-thread MPI call,
                # not by treating the application's zero counters as zero traffic.
                tables = [name for name in ('MPI_P2P_EVENTS', 'MPI_START_WAIT_EVENTS',
                    'MPI_COLLECTIVES_EVENTS', 'MPI_OTHER_EVENTS') if name in available]
                mpi_calls = ' union all '.join(f'select start,end,globalTid from {table}' for table in tables)
                if not mpi_calls:
                    mpi_calls = 'select 0 as start,0 as end,0 as globalTid where 0'
                rows = db.execute('with mpi_calls as ('+mpi_calls+') '
                    'select e.name,m.bytes,exists(select 1 from CUPTI_ACTIVITY_KIND_RUNTIME r '
                    'where r.correlationId=m.correlationId and exists(select 1 from mpi_calls p '
                    'where p.globalTid=r.globalTid and r.start>=p.start and r.end<=p.end)) '
                    'from CUPTI_ACTIVITY_KIND_MEMCPY m join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind '
                    "where e.name in ('CUDA_MEMCPY_KIND_DTOH','CUDA_MEMCPY_KIND_HTOD') and m.bytes>144 "
                    'and exists(select 1 from NVTX_EVENTS n where n.text=? and m.start>=n.start and m.end<=n.end)',
                    (sample,)).fetchall()
                sizes = {n for n, in db.execute('select m.size from MPI_P2P_EVENTS m '
                    'where exists(select 1 from NVTX_EVENTS n where n.text=? and m.start>=n.start and m.end<=n.end)',
                    (sample,))} if 'MPI_P2P_EVENTS' in available else set()
                assert all(in_mpi and size in sizes for _, size, in_mpi in rows), rows
                mpi_staged = [(kind, size) for kind, size, _ in rows]
            consumed = copies(db, consumer)
            controls = [n for k, n in consumed if k == 'CUDA_MEMCPY_KIND_DTOH']
            expected = [size for product in receipt['products'].values() for size in (
                product['local_points']*12, product['local_points']*4, product['local_cells']*12) if size]
            if expected:
                assert controls and all(n <= 256 for n in controls), controls
            else:
                assert not consumed, consumed
            rendered = copies(db, 'ASTR_IS8_RESIDENT_RENDER')
            assert bool(rendered) == bool(expected)
            assert all(k == 'CUDA_MEMCPY_KIND_DTOD' for k, _ in rendered)
            assert sorted(n for _, n in rendered) == sorted(expected)
            kernels = [value for value, in db.execute('select s.value from CUPTI_ACTIVITY_KIND_KERNEL k '
                'join StringIds s on s.id=k.demangledName where exists(select 1 from NVTX_EVENTS n '
                'where n.text=? and k.start>=n.start and k.end<=n.end)', (consumer,))]
            required = (('structured_geometry::CountPlane', 'structured_geometry::WritePlane') if kind == 'PLANE' else
                        ('astr_insitu::CheckSurfaceArea',) if kind == 'VOLUME' else
                        ('astr_insitu::PackWallField', 'astr_insitu::CheckWallTriangles'))
            if expected:
                for operator in (*required, 'astr_insitu::MapDisplayColor'):
                    assert any(operator in kernel for kernel in kernels), operator
            mpi = db.execute('select s.value,count(*),sum(m.size) from MPI_P2P_EVENTS m '
                'join StringIds s on s.id=m.textId where exists(select 1 from NVTX_EVENTS n '
                "where n.text='ASTR_IS8_RESIDENT_RENDER' and m.start>=n.start and m.end<=n.end) "
                    'group by s.value').fetchall() if 'MPI_P2P_EVENTS' in available else []
            assert bool(mpi) == (ranks > 1)
        reads = [p for p in pixels if int(p['rank']) == rank]
        for name in receipt['products']:
            group = [p for p in reads if p['product'] == name]
            formats = Counter((int(p['format']), int(p['type'])) for p in group)
            assert formats[(6407, 5121)] == 1
            assert formats[(6408, 5121)] == formats[(6402, 5126)]
            assert formats[(6408, 5121)] in (0, 2)
            assert set(formats) <= {(6407, 5121), (6408, 5121), (6402, 5126)}
        assert all(int(p['step']) == 2 and int(p['width']) == 800 and int(p['height']) == 600
                   and int(p['pack_buffer']) == 0 and int(p['row_length']) == 0
                   and int(p['payload_bytes']) == 800*600*(3 if int(p['format']) == 6407 else 4)
                   for p in reads)
        result.append(dict(rank=rank, face_transfers=transfers, bounded_consumer_control_reads=controls,
            application_face_d2h_bytes=int(face[1]), application_face_h2d_bytes=int(face[2]),
            mpi_managed_face_host_copies=mpi_staged,
            display_dtod_bytes=sum(expected), geometry_render_d2h_bytes=0,
            pixel_payload_bytes=sum(int(p['payload_bytes']) for p in reads),
            pixel_read_inclusive_seconds=sum(float(p['seconds']) for p in reads), render_mpi=mpi))
    return result


@pytest.mark.parametrize('family', ['cartesian-wall', 'curve-wall', 'cartesian-plane', 'curve-plane', 'curve-surface'])
@pytest.mark.parametrize('mode', ['pinned', 'device-aware'])
@pytest.mark.parametrize('pipeline', ['standard-device', 'direct-device'])
@pytest.mark.parametrize('topology', [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)],
                         ids=['single', 'x', 'y', 'z'])
def test_actual_x4_observation(tmp_path, family, mode, pipeline, topology, record_property):
    ranks = int(np.prod(topology))
    args, config, kwargs = settings(tmp_path, family, pipeline, mode, topology)
    disabled = '&insitu_run enabled=f /\n'
    off, _ = run_case(args, ROOT, 'gpu', ranks, 'off', 4,
                     insitu_config=disabled, postprocess_transport=mode, **kwargs)
    on, _ = run_case(args, ROOT, 'gpu', ranks, 'on', 4,
                    insitu_config=config, postprocess_transport=mode, **kwargs)
    for case in (off, on):
        no_large_io(case)
    off_timing, on_timing = timing(off, ranks), timing(on, ranks)
    record_property('matched_cost', json.dumps(dict(off=off_timing, on=on_timing,
        extra_completed_window_seconds=on_timing['completed_window']['seconds']-off_timing['completed_window']['seconds'],
        scope='one unprofiled four-step pair; initialization excluded; first-frame setup included; nested phases not summed')))
    baseline_case, _ = run_case(args, ROOT, 'gpu', ranks, 'resource_off', 2,
        insitu_config=disabled, postprocess_transport=mode, monitor_resources=True, **kwargs)
    baseline = json.loads((baseline_case / 'resources.sampled.json').read_text())
    observed, size = run_case(args, ROOT, 'gpu', ranks, 'resource_on', 2,
        insitu_config=config, postprocess_transport=mode, monitor_resources=True, resource_baseline=baseline, **kwargs)
    report = json.loads((observed / 'resources.sampled.json').read_text())
    assert report['samples'] > 0 and report['sampling_period_seconds'] == .02
    assert report['additional_host_peak_difference_bytes'] <= 4*1024**3
    assert all(n <= 2*1024**3 for n in report['additional_device_peak_difference_bytes'].values())
    assert all(d['min_free_bytes'] >= 1024**3 for d in report['devices'].values())
    for case in (baseline_case, observed):
        no_large_io(case)
    trace, _ = run_case(args, ROOT, 'gpu', ranks, 'trace', 2, insitu_config=config,
        postprocess_transport=mode, nsys_trace=True, pixel_audit=True, **kwargs)
    no_large_io(trace)
    record_property('actual_transfer_ledger', json.dumps(ledger(trace, family, mode, ranks)))
    record_property('external_20ms_resources', json.dumps(report))
    record_property('observed_directory_bytes', size)
