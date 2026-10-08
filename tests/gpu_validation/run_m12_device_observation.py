#!/usr/bin/env python3
"""Separate bounded geometry diagnostics from clean M12 transfer captures."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sqlite3

import h5py
import numpy as np

from run_m12_insitu_acceptance import configuration, check_images, PRODUCTS, STEPS
from run_m12_local_compare import run, TOPOLOGIES
from test_insitu_device_native_trace import copies
from test_insitu_device_physical_plane import oracle as read_plane
from test_insitu_device_curve_surface import read_oracle as read_surface
from test_insitu_device_curve_streamlines import read_trace_oracle
from test_insitu_x4_observation import timing
from compare_flowstate import read_flowstate


def check_geometry(case, ranks):
    with h5py.File(case/'datin/grid.h5') as grid:
        bounds = [(float(grid[name][...].min()), float(grid[name][...].max())) for name in ('x', 'y', 'z')]
    expected_area = ((np.longdouble(bounds[0][1])-bounds[0][0]) *
                     (np.longdouble(bounds[1][1])-bounds[1][0]))
    result = []
    for step in STEPS:
        plane_points, plane_triangles, plane_edges = {}, set(), {}
        q_points, q_triangles, q_edges = {}, set(), {}
        trace_points, trace_intervals = {}, {}
        seam_error = 0.
        area = np.longdouble(0.)
        for rank in range(ranks):
            record = json.loads((case/f'outdat/render/mesh_step{step:08d}_rank{rank}.json').read_text())
            xyz, velocity, keys, triangles = read_plane(case/f'diagnostics/plane.step{step}.rank{rank}.bin')
            if len(xyz) != record['products']['velocity_slice']['local_points'] or \
                    len(triangles) != record['products']['velocity_slice']['local_cells']:
                raise ValueError('plane geometry count differs')
            if not np.isfinite(xyz).all() or not np.isfinite(velocity).all() or np.max(abs(xyz[:, 2]-45.), initial=0.) > 2e-10:
                raise ValueError('invalid plane position/field')
            for p, v, key in zip(xyz, velocity, keys):
                key = tuple(key)
                values = np.r_[p, v]
                if key in plane_points:
                    seam_error = max(seam_error, float(np.max(abs(values-plane_points[key]))))
                plane_points[key] = values
            for ids in triangles:
                if np.any(ids < 0) or np.any(ids >= len(xyz)):
                    raise ValueError('plane connectivity out of bounds')
                tri = tuple(tuple(keys[i]) for i in ids)
                unordered = tuple(sorted(tri))
                if unordered in plane_triangles:
                    raise ValueError('duplicate plane triangle across MPI ranks')
                plane_triangles.add(unordered)
                a, b, c = xyz[ids].astype(np.longdouble)
                cross = np.cross(b-a, c-a)
                if cross[2] <= 0.:
                    raise ValueError('plane triangle has wrong winding or zero area')
                area += .5*np.linalg.norm(cross)
                add_edges(plane_edges, tri)
            xyz, computational, velocity, q, triangles = read_surface(case/f'diagnostics/surface.step{step}.rank{rank}.bin')
            if len(xyz) != record['products']['q_surface']['local_points'] or \
                    len(triangles) != record['products']['q_surface']['local_cells']:
                raise ValueError('Q geometry count differs')
            if not all(np.isfinite(v).all() for v in (xyz, computational, velocity, q)) or \
                    np.max(abs(q-.001), initial=0.) > 2e-10:
                raise ValueError('invalid Q contour field')
            # This is a bounded fixture seam identity, not a general contour tolerance.
            q_keys = [tuple(p) for p in np.round(computational, 12)]
            for p, v, key in zip(xyz, velocity, q_keys):
                values = np.r_[p, v]
                if key in q_points:
                    seam_error = max(seam_error, float(np.max(abs(values-q_points[key]))))
                q_points[key] = values
            for ids in triangles:
                if np.any(ids < 0) or np.any(ids >= len(xyz)):
                    raise ValueError('Q connectivity out of bounds')
                tri = tuple(q_keys[i] for i in ids)
                unordered = tuple(sorted(tri))
                if unordered in q_triangles:
                    raise ValueError('duplicate Q triangle across MPI ranks')
                q_triangles.add(unordered)
                a, b, c = xyz[ids].astype(np.longdouble)
                if np.dot(np.cross(b-a, c-a), np.cross(b-a, c-a)) <= 0.:
                    raise ValueError('zero-area Q triangle')
                add_edges(q_edges, tri)
            xyz, velocity, accepted, particle, cells = read_trace_oracle(
                case/f'diagnostics/trace.instantaneous_streamlines.step{step}.rank{rank}.bin')
            if not all(np.isfinite(v).all() for v in (xyz, velocity, accepted)) or \
                    np.any(particle < 0) or np.any(particle >= 128) or \
                    np.any(cells < 0) or np.any(cells >= len(xyz)) or not np.array_equal(xyz, accepted[:, :3]):
                raise ValueError('invalid physical streamline geometry/state')
            for p, v, state, pid in zip(xyz, velocity, accepted, particle):
                trace_points.setdefault(int(pid), []).append((state[3], np.r_[p, v]))
            for first, second in cells:
                if particle[first] != particle[second] or accepted[second, 3] <= accepted[first, 3] or \
                        np.linalg.norm(xyz[second]-xyz[first]) <= 0.:
                    raise ValueError('streamline particle identity/forward arc differs')
                trace_intervals.setdefault(int(particle[first]), []).append(
                    (accepted[first, 3], accepted[second, 3], rank))
        if seam_error > 2e-10:
            raise ValueError(f'physical product seam mismatch: {seam_error}')
        if len(plane_points)-len(plane_edges)+len(plane_triangles) != 1 or abs(area-expected_area) > 2e-10:
            raise ValueError(f'plane topology/area differs: {area}')
        for edges, points, surface in ((plane_edges, plane_points, 'plane'), (q_edges, q_points, 'Q')):
            for (first, second), (count, winding) in edges.items():
                if count == 2 and winding == 0:
                    continue
                a, b = points[first][:3], points[second][:3]
                exterior = any(abs(a[d]-side) <= 2e-10 and abs(b[d]-side) <= 2e-10
                               for d, sides in enumerate(bounds) for side in sides)
                if count != 1 or not exterior:
                    raise ValueError(f'{surface} interior open edge or inconsistent winding')
        crossings = 0
        for pid in range(128):
            intervals = sorted(trace_intervals.get(pid, []))
            points = sorted(trace_points.get(pid, []), key=lambda v: v[0])
            if not intervals or not points or abs(intervals[0][0]) > 2e-10:
                raise ValueError(f'missing streamline seed/arc: {pid}')
            for previous, current in zip(intervals, intervals[1:]):
                if abs(current[0]-previous[1]) > 2e-10:
                    raise ValueError('MPI streamline gap/overlap in accepted arc')
                crossings += previous[2] != current[2]
            for previous, current in zip(points, points[1:]):
                if abs(current[0]-previous[0]) <= 2e-13:
                    seam_error = max(seam_error, float(np.max(abs(current[1]-previous[1]))))
            if seam_error > 2e-10:
                raise ValueError('MPI continuation point/velocity mismatch')
        result.append(dict(step=step, seam_max_abs=seam_error, plane_area=float(area),
                           source_rectangle_area=float(expected_area),
                           plane_triangles=len(plane_triangles), q_triangles=len(q_triangles),
                           actual_streamline_handoffs=crossings,
                           scope='bounded diagnostic downloads; no zero-readback claim'))
    return result


def add_edges(edges, triangle):
    for a, b in zip(triangle, triangle[1:]+triangle[:1]):
        key = tuple(sorted((a, b)))
        count, winding = edges.get(key, (0, 0))
        edges[key] = count+1, winding+(1 if a < b else -1)


def check_transfers(case, ranks):
    log = (case/'run.log').read_text()
    forbidden = ('ASTR_X4_CURVE_TRACE_TEST_ORACLE', 'ASTR_IS8_COMPACT_GEOMETRY_READ',
                 'ASTR_IS8_COMPACT_TRACE_READ', 'ASTR_M12_RENDER_ISOLATION',
                 'ASTR_INSITU_RESIDENT_AUDIT')
    if any(name in log for name in forbidden) or list((case/'diagnostics').iterdir()) or \
            list((case/'outdat').rglob('*.h5')) or list((case/'outdat').rglob('*.vtp')):
        raise ValueError('clean capture contains diagnostics or large output')
    pixels = [dict(re.findall(r'(\w+)=([^ ]+)', row)) for row in
              re.findall(r'ASTR_INSITU_PIXEL_READ ([^\n]+)', log)]
    result = []
    for rank in range(ranks):
        records = [json.loads((case/f'outdat/render/mesh_step{step:08d}_rank{rank}.json').read_text()) for step in STEPS]
        with sqlite3.connect((case/f'trace.rank{rank}.sqlite').resolve().as_uri()+'?mode=ro', uri=True) as db:
            counts = dict(db.execute('select text,count(*) from NVTX_EVENTS group by text'))
            for name in ('ASTR_IS8_DEVICE_SAMPLE', 'ASTR_IS8_DEVICE_MEAN_SUPPLY',
                         'ASTR_X4_DEVICE_VOLUME_CONSUMER', 'ASTR_IS8_RESIDENT_RENDER'):
                if counts.get(name) != len(STEPS):
                    raise ValueError(f'incomplete NVTX frame capture: {rank}, {name}')
            if any(name in counts for name in forbidden) or db.execute(
                    'select m.bytes from CUPTI_ACTIVITY_KIND_MEMCPY m join ENUM_CUDA_MEMCPY_OPER e '
                    "on e.id=m.copyKind where e.name='CUDA_MEMCPY_KIND_UVM_DTOH'").fetchall():
                raise ValueError('diagnostic/managed host readback in strict capture')
            controls = copies(db, 'ASTR_IS8_TRACE_CONTROL_READ')
            owners = copies(db, 'ASTR_X4_CURVE_TRACE_OWNER_QUERY')
            if not controls or not owners or any(n > 8192 for k, n in controls if k.endswith('_DTOH')) or \
                    any(n > 1024 for k, n in owners if k.endswith('_DTOH')):
                raise ValueError('missing/unbounded particle control readback')
            remainder = db.execute('select e.name,m.bytes from CUPTI_ACTIVITY_KIND_MEMCPY m '
                'join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind '
                'where exists(select 1 from NVTX_EVENTS n where n.text=? and m.start>=n.start and m.end<=n.end) '
                'and not exists(select 1 from NVTX_EVENTS n where n.text in (?,?) and m.start>=n.start and m.end<=n.end)',
                ('ASTR_X4_DEVICE_VOLUME_CONSUMER', 'ASTR_IS8_TRACE_CONTROL_READ',
                 'ASTR_X4_CURVE_TRACE_OWNER_QUERY')).fetchall()
            if any(n > 256 for k, n in remainder if k.endswith('_DTOH')):
                raise ValueError('unattributed large consumer D2H')
            expected = [size for record in records for name, p in record['products'].items()
                        for size in (p['local_points']*12, p['local_points']*4,
                                     p['local_cells']*(8 if 'streamlines' in name else 12)) if size]
            display = copies(db, 'ASTR_IS8_RESIDENT_RENDER')
            if sorted(display) != sorted(('CUDA_MEMCPY_KIND_DTOD', n) for n in expected):
                raise ValueError('resident display payload is not exact device-to-device geometry')
            faces = {name: copies(db, name) for name in ('ASTR_IS8_DEVICE_SAMPLE', 'ASTR_IS8_DEVICE_MEAN_SUPPLY')}
            face = re.findall(rf'ASTR_INSITU_DEVICE_FRAME rank={rank} step=\d+ transport=pinned '
                              r'face_d2h_bytes=(\d+) face_h2d_bytes=(\d+) face_exchanges=(\d+)', log)
            if len(face) != len(STEPS):
                raise ValueError('missing pinned face accounting')
            for kind, counter in (('DTOH', 0), ('HTOD', 1)):
                actual = sum(n for rows in faces.values() for k, n in rows
                             if k == 'CUDA_MEMCPY_KIND_'+kind and n > 144)
                if actual != int(face[-1][counter]):
                    raise ValueError(f'pinned face bytes differ: {rank}, {kind}, {actual}, {face[-1]}')
            reads = [p for p in pixels if int(p['rank']) == rank]
            for step in STEPS:
                for name in PRODUCTS:
                    formats = Counter((int(p['format']), int(p['type'])) for p in reads
                                      if p['product'] == name and int(p['step']) == step)
                    if formats[(6407, 5121)] != 1 or formats[(6408, 5121)] != formats[(6402, 5126)] or \
                            formats[(6408, 5121)] not in (0, 2) or \
                            set(formats)-{(6407, 5121), (6408, 5121), (6402, 5126)}:
                        raise ValueError('unexpected RGB/color/depth pixel transfer')
            if not reads or any(int(p['width']) != 1536 or int(p['height']) != 384 or
                               int(p['pack_buffer']) != 0 or int(p['row_length']) != 0 or
                               int(p['payload_bytes']) != 1536*384*(3 if int(p['format']) == 6407 else 4)
                               for p in reads):
                raise ValueError('pixel payload dimensions/packing differ')
            result.append(dict(rank=rank, field_geometry_d2h_bytes=0,
                pixel_payload_bytes=sum(int(p['payload_bytes']) for p in reads),
                display_dtod_bytes=sum(expected), application_face_d2h_bytes=int(face[-1][0]),
                application_face_h2d_bytes=int(face[-1][1]), control_copies=controls,
                owner_copies=owners, other_consumer_copies=remainder,
                scope='complete-frame NVTX/CUPTI attribution; pinned halos and image/control reads allowed'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output', 'executable', 'mpiexec', 'library'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--mode', choices=('geometry', 'trace', 'timing'), required=True)
    parser.add_argument('--nsys', type=Path)
    parser.add_argument('--np4-profiler-host-budget-8gib', action='store_true',
                        help='approved external process-tree allowance for NP=4 trace only; native budget stays 4 GiB')
    parser.add_argument('--topologies', nargs='+', choices=TOPOLOGIES, default=TOPOLOGIES)
    args = parser.parse_args()
    if args.mode == 'trace' and args.nsys is None:
        parser.error('--trace mode requires --nsys')
    if args.np4_profiler_host_budget_8gib and args.mode != 'trace':
        parser.error('--np4-profiler-host-budget-8gib is only allowed with --mode trace')
    if args.nsys:
        args.nsys = args.nsys.resolve(strict=True)
    args.output = args.output.resolve()
    args.executable = args.executable.resolve(strict=True)
    args.library = args.library.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    args.memcheck = args.diagnostics = False
    for topology in args.topologies:
        ranks = int(np.prod([int(n) for n in topology.split(',')]))
        if args.mode == 'timing':
            off = run(args, 'gpu', topology, insitu_config='&insitu_run enabled=f /\n',
                      case_label='off', observation='images')
            baseline = json.loads((off/'resources.sampled.json').read_text())
            on = run(args, 'gpu', topology, insitu_config=configuration(args.library),
                     case_label='on', observation='images', resource_baseline=baseline)
            a, b = read_flowstate(off), read_flowstate(on)
            if a[0] != b[0] or not np.array_equal(a[1], b[1]):
                raise ValueError('clean timing pair changes existing statistics')
            t_off, t_on = timing(off, ranks), timing(on, ranks)
            receipt = dict(off=t_off, on=t_on, images=check_images(on, ranks, audit=False),
                extra_completed_window_seconds=t_on['completed_window']['seconds']-t_off['completed_window']['seconds'],
                scope='single unprofiled matched ten-step pair; initialization excluded; nested stages not summed; not a scaling benchmark')
            (args.output/f'timing_{topology.replace(",", "x")}.json').write_text(json.dumps(receipt, indent=2)+'\n')
            print('PASS M12 clean timing topology='+topology, flush=True)
            continue
        off = run(args, 'gpu', topology, insitu_config='&insitu_run enabled=f /\n',
                  case_label='baseline', observation='images')
        baseline = json.loads((off/'resources.sampled.json').read_text())
        case = run(args, 'gpu', topology, insitu_config=configuration(args.library),
                   case_label=args.mode, observation=args.mode, resource_baseline=baseline)
        check_images(case, ranks, audit=False)
        receipt = check_geometry(case, ranks) if args.mode == 'geometry' else check_transfers(case, ranks)
        (args.output/f'{args.mode}_{topology.replace(",", "x")}.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print('PASS M12 '+args.mode+' topology='+topology, flush=True)


if __name__ == '__main__':
    main()
