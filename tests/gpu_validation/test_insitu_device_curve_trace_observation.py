"""Strict CURVE traces: no reference I/O, attributed transfers and local cost."""
import json
import os
import re
import sqlite3
from collections import Counter

import numpy as np
import pytest

from run_output_restart_validation import run_case
from test_insitu_curve_derivatives import ROOT
from test_insitu_device_curve_wall_fields import current_arguments
from test_insitu_device_curve_streamlines import configuration, validate, TOPOLOGIES
from test_insitu_x4_observation import timing, no_large_io
from test_insitu_device_native_trace import copies

pytestmark = pytest.mark.skipif(os.environ.get('ASTR_INSITU_X4_OBSERVE') != '1',
    reason='Select the numerically validated CURVE streamline executable for observation')


def transfer_ledger(case, ranks, image_size=(800, 600)):
    log = (case / 'run.log').read_text()
    assert 'ASTR_INSITU_CURVE_TRACE_TEST_ORACLE' not in log
    pixels = [dict(re.findall(r'(\w+)=([^ ]+)', row)) for row in
              re.findall(r'ASTR_INSITU_PIXEL_READ ([^\n]+)', log)]
    assert pixels
    result = []
    for rank in range(ranks):
        receipt = json.loads((case / f'outdat/render/mesh_step00000002_rank{rank}.json').read_text())
        assert receipt['geometry_host_bytes'] == 0
        with sqlite3.connect((case / f'trace.rank{rank}.sqlite').resolve().as_uri()+'?mode=ro', uri=True) as db:
            counts = dict(db.execute('select text,count(*) from NVTX_EVENTS group by text'))
            for name in ('ASTR_IS8_DEVICE_SAMPLE', 'ASTR_IS8_DEVICE_MEAN_SUPPLY',
                         'ASTR_X4_DEVICE_VOLUME_CONSUMER', 'ASTR_IS8_RESIDENT_RENDER'):
                assert counts[name] == 1, counts
            assert not any(name in counts for name in ('ASTR_X4_CURVE_TRACE_TEST_ORACLE',
                'ASTR_IS8_COMPACT_GEOMETRY_READ', 'ASTR_IS8_COMPACT_TRACE_READ'))
            assert not db.execute('select m.bytes from CUPTI_ACTIVITY_KIND_MEMCPY m '
                'join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind '
                "where e.name='CUDA_MEMCPY_KIND_UVM_DTOH'").fetchall()
            controls = copies(db, 'ASTR_IS8_TRACE_CONTROL_READ')
            owners = copies(db, 'ASTR_X4_CURVE_TRACE_OWNER_QUERY')
            assert controls and owners
            assert all(n <= 2048 for k, n in controls if k == 'CUDA_MEMCPY_KIND_DTOH')
            assert all(n <= 256 for k, n in owners if k == 'CUDA_MEMCPY_KIND_DTOH')
            remainder = db.execute('select e.name,m.bytes from CUPTI_ACTIVITY_KIND_MEMCPY m '
                'join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind '
                'where exists(select 1 from NVTX_EVENTS n where n.text=? and m.start>=n.start and m.end<=n.end) '
                'and not exists(select 1 from NVTX_EVENTS n where n.text in (?,?) '
                'and m.start>=n.start and m.end<=n.end)',
                ('ASTR_X4_DEVICE_VOLUME_CONSUMER', 'ASTR_IS8_TRACE_CONTROL_READ',
                 'ASTR_X4_CURVE_TRACE_OWNER_QUERY')).fetchall()
            assert all(n <= 256 for k, n in remainder if k == 'CUDA_MEMCPY_KIND_DTOH'), remainder
            kernels = [name for name, in db.execute('select s.value from CUPTI_ACTIVITY_KIND_KERNEL k '
                'join StringIds s on s.id=k.demangledName where exists(select 1 from NVTX_EVENTS n '
                'where n.text=? and k.start>=n.start and k.end<=n.end)', ('ASTR_X4_DEVICE_VOLUME_CONSUMER',))]
            required = ['LocatePhysicalTraceOwner', 'RK45PhysicalTraceWorklet']
            if any(p['local_points'] for name, p in receipt['products'].items() if 'streamlines' in name):
                required.append('PackResidentTrace')
            if any(p['local_points'] for p in receipt['products'].values()):
                required.append('MapDisplayColor')
            for name in required:
                assert any(name in kernel for kernel in kernels), name
            display = copies(db, 'ASTR_IS8_RESIDENT_RENDER')
            expected = [size for name, p in receipt['products'].items() for size in
                        (p['local_points']*12, p['local_points']*4,
                         p['local_cells']*(12 if name == 'q_surface' else 8)) if size]
            assert sorted(display) == sorted(('CUDA_MEMCPY_KIND_DTOD', n) for n in expected), display
            faces = {name: copies(db, name) for name in ('ASTR_IS8_DEVICE_SAMPLE', 'ASTR_IS8_DEVICE_MEAN_SUPPLY')}
            face = re.search(rf'ASTR_INSITU_DEVICE_FRAME rank={rank} step=2 transport=(\S+) '
                r'face_d2h_bytes=(\d+) face_h2d_bytes=(\d+) face_exchanges=(\d+)', log)
            assert face, log
            managed = []
            if face[1] == 'pinned':
                for kind, counter in (('DTOH', 2), ('HTOD', 3)):
                    actual = sum(n for rows in faces.values() for k, n in rows
                                 if k == 'CUDA_MEMCPY_KIND_'+kind and n > 144)
                    assert actual == int(face[counter]), (rank, kind, actual, face.groups())
            else:
                assert int(face[2]) == int(face[3]) == 0
                tables = {name for name, in db.execute("select name from sqlite_master where type='table'")}
                mpi = ' union all '.join('select start,end,globalTid from '+table for table in
                    ('MPI_P2P_EVENTS', 'MPI_COLLECTIVES_EVENTS', 'MPI_START_WAIT_EVENTS', 'MPI_OTHER_EVENTS')
                    if table in tables)
                if mpi:
                    managed = db.execute('with mpi_calls as ('+mpi+') '
                        'select e.name,m.bytes,exists(select 1 from CUPTI_ACTIVITY_KIND_RUNTIME r '
                        'where r.correlationId=m.correlationId and exists(select 1 from mpi_calls p '
                        'where p.globalTid=r.globalTid and r.start>=p.start and r.end<=p.end)) '
                        'from CUPTI_ACTIVITY_KIND_MEMCPY m join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind '
                        "where e.name in ('CUDA_MEMCPY_KIND_DTOH','CUDA_MEMCPY_KIND_HTOD') and m.bytes>144 "
                        'and exists(select 1 from NVTX_EVENTS n where n.text in (?,?) '
                        'and m.start>=n.start and m.end<=n.end)', tuple(faces)).fetchall()
                    assert all(in_mpi for _, _, in_mpi in managed), managed
            reads = [p for p in pixels if int(p['rank']) == rank]
            for name in receipt['products']:
                formats = Counter((int(p['format']), int(p['type'])) for p in reads if p['product'] == name)
                assert formats[(6407, 5121)] == 1
                assert formats[(6408, 5121)] == formats[(6402, 5126)]
                assert formats[(6408, 5121)] in (0, 2)
                assert set(formats) <= {(6407, 5121), (6408, 5121), (6402, 5126)}
            width, height = image_size
            assert all(int(p['step']) == 2 and int(p['width']) == width and int(p['height']) == height
                and int(p['pack_buffer']) == 0 and int(p['row_length']) == 0
                and int(p['payload_bytes']) == width*height*(3 if int(p['format']) == 6407 else 4) for p in reads)
            result.append(dict(rank=rank, continuation_copies=controls, owner_query_copies=owners,
                other_consumer_copies=remainder, face_copies=faces, application_face_d2h_bytes=int(face[2]),
                application_face_h2d_bytes=int(face[3]), mpi_managed_host_copies=managed,
                display_dtod_bytes=sum(expected), field_geometry_d2h_bytes=0,
                pixel_payload_bytes=sum(int(p['payload_bytes']) for p in reads),
                pixel_read_inclusive_seconds=sum(float(p['seconds']) for p in reads)))
    return result


@pytest.mark.parametrize('mapping', ('periodic', 'y-wavy'))
@pytest.mark.parametrize('ranks,axis', TOPOLOGIES, ids=('single', 'x', 'y', 'z'))
@pytest.mark.parametrize('mode', ('pinned', 'device-aware'))
@pytest.mark.parametrize('pipeline', ('standard-device', 'direct-device'))
def test_curve_trace_observation(tmp_path, mapping, ranks, axis, mode, pipeline, record_property):
    args = current_arguments(tmp_path, axis=axis, samples=False)
    args.runtime_timeout_seconds = 600
    args.nsys_trace_domains = 'cuda,nvtx,mpi'
    kwargs = dict(grid='32,32,32', tgv_mapping=mapping, enabled=False, checkpoint_enabled=False,
                  postprocess_transport=mode, insitu_timing=True)
    config = configuration(pipeline, mode)
    disabled = '&insitu_run enabled=f /\n'
    off, _ = run_case(args, ROOT, 'gpu', ranks, 'off', 4, insitu_config=disabled, **kwargs)
    on, _ = run_case(args, ROOT, 'gpu', ranks, 'on', 4, insitu_config=config, **kwargs)
    validate(on, ranks, pipeline, audit=False)
    a, b = timing(off, ranks), timing(on, ranks)
    record_property('matched_cost', json.dumps(dict(off=a, on=b,
        extra_completed_window_seconds=b['completed_window']['seconds']-a['completed_window']['seconds'],
        scope='one unprofiled four-step pair; initialization excluded; nested phases not summed')))
    baseline, _ = run_case(args, ROOT, 'gpu', ranks, 'resource_off', 2, insitu_config=disabled,
        monitor_resources=True, **kwargs)
    resource, size = run_case(args, ROOT, 'gpu', ranks, 'resource_on', 2, insitu_config=config,
        monitor_resources=True, resource_baseline=json.loads((baseline / 'resources.sampled.json').read_text()), **kwargs)
    report = json.loads((resource / 'resources.sampled.json').read_text())
    assert report['samples'] > 0 and report['sampling_period_seconds'] == .02
    assert report['additional_host_peak_difference_bytes'] <= 4*1024**3
    assert all(n <= 2*1024**3 for n in report['additional_device_peak_difference_bytes'].values())
    assert all(d['min_free_bytes'] >= 1024**3 for d in report['devices'].values())
    trace, _ = run_case(args, ROOT, 'gpu', ranks, 'trace', 2, insitu_config=config,
        nsys_trace=True, pixel_audit=True, **kwargs)
    validate(trace, ranks, pipeline, steps=(2,), audit=False)
    for case in (off, on, baseline, resource, trace):
        no_large_io(case)
    record_property('actual_transfer_ledger', json.dumps(transfer_ledger(trace, ranks)))
    record_property('external_20ms_resources', json.dumps(report))
    record_property('observed_directory_bytes', size)
