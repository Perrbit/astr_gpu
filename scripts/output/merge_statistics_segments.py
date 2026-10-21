"""Explicit offline FP64 merging of ASTR velocity-statistics segments."""

import argparse
import json
from pathlib import Path
import shlex

import h5py
import numpy as np

from export_checkpoint import (bundle_members, identity, native_dataset, plain_file,
                               small_text, state_clock, state_dtype)


MOMENT_NAMES = ('reynolds', 'favre')
ARRAY_BYTES_PER_POINT = 1024


def decode_words(words, width):
    words = np.asarray(words, dtype='<i8')
    if width == 4:
        return words.astype('<i4').view('<f4').astype(np.float64)
    return words.view('<f8')


def physics_records(path):
    records = []
    plain_file(path)
    with path.open(encoding='ascii') as source:
        for line in source:
            lexer = shlex.shlex(line, posix=True)
            lexer.whitespace += ','
            lexer.whitespace_split = True
            lexer.commenters = '#!'
            tokens = tuple(token.lower() for token in lexer)
            if tokens:
                records.append(tokens)
    if len(records) < 6 or len(records[3]) not in (7, 8, 9):
        raise ValueError('unsupported frozen primary-input layout')
    # Execution device, timing reports, restart selection and path spelling are
    # not physical input. The actual frozen grid resource is checked separately.
    flags = records[3] + (('f',) if len(records[3]) == 7 else ())
    records[3] = tuple(v for i, v in enumerate(flags) if i not in (6, 8))
    records[4] = ('completed-step-restore',)
    records[-1] = ('frozen-grid-resource',)
    return records


def read_segment(path):
    path = Path(path).resolve(strict=True)
    plain_file(path)
    batch = path.parent
    marker = small_text(batch / 'COMPLETE')
    if not (marker == 'ASTR_COMPLETE_2\n' or marker.startswith('ASTR_COMPLETE_1 ')):
        raise ValueError('statistics source is not a completed checkpoint')
    members = bundle_members(batch / 'MANIFEST', 'ASTR_CHECKPOINT_BUNDLE')
    if path.name not in members:
        raise ValueError('statistics source is not declared by its checkpoint')
    resources = bundle_members(batch / 'RESOURCES', 'ASTR_SHARED_RESOURCES')
    root = batch.parent.parent / 'resources'
    if 'input.txt' not in resources:
        raise ValueError('statistics source lacks its frozen physical input')
    with h5py.File(path, 'r') as source:
        role = int(native_dataset(source, 'identity', (13,), '<i8')[12])
        if role not in (3, 4):
            raise ValueError('source is not point velocity sufficient statistics')
        header, shape = identity(source, 34, role)
        dtype = state_dtype(source, header)
        record = native_dataset(source, 'statistics_segment', (21,), '<i8')[:]
        metadata = native_dataset(source, 'metadata', (38,), '<i8')[:]
        if record[0] != 1 or record[1] != 1:
            raise ValueError('statistics segment start is unknown; do not infer it from duration')
        if (record[2] < 0 or record[2] > header[8] or record[6] not in (4, 8) or
                record[8] < 0 or record[8] > 7 or record[9] not in (0, 1) or
                tuple(record[10:12]) != (1, 1) or record[12] not in (0, 1, 2) or
                np.any(record[13:] < 0) or np.any(record[13:] > 0xffffffff)):
            raise ValueError('unsupported statistics definition, layout or segment identity')
        clock = state_clock(header, dtype).astype(np.float64)
        start, *window = decode_words(record[3:6], dtype.itemsize)
        parent_time = float(decode_words(record[7:8], int(record[6]))[0])
        if (not np.isfinite([start, *window, parent_time]).all() or
                start < 0 or start > clock[0] or window[1] <= window[0]):
            raise ValueError('invalid statistics segment/window clock')
        projected = np.asarray(parent_time, dtype=dtype).view(f'<i{dtype.itemsize}').item()
        if projected != int(record[3]):
            raise ValueError('segment start does not match its declared source-clock conversion')
        if (metadata[0] != 1 or metadata[1] != int(role == 4) or
                metadata[2] != header[8] or metadata[3] != header[9]):
            raise ValueError('statistics regional/sample identity differs from the completed step')
        regional = decode_words(metadata[4:], dtype.itemsize)
        if (not np.isfinite(regional).all() or regional[33] != 1 or
                not np.array_equal(regional[:2], window) or regional[2] != clock[0]):
            raise ValueError('invalid regional sufficient statistics/window/sample identity')
    periodic = [(int(record[8]) & (1 << i)) != 0 for i in range(3)]
    unique_shape = tuple(int(n) - int(p) for n, p in zip(header[3:6], periodic))[::-1]
    physical = physics_records(root / 'input.txt')
    static = {name: root / name for name in resources if name not in ('input.txt', 'geometry.h5')}
    expected_grid = {1: 'grid.h5', 2: 'grid.2d'}.get(int(record[12]))
    if expected_grid is not None and expected_grid not in static:
        raise ValueError('statistics source lacks its frozen grid resource')
    return dict(path=path, header=header, dtype=dtype, record=record,
                shape=shape, unique_shape=unique_shape, periodic=periodic,
                start_step=int(record[2]), end_step=int(header[8]), start=float(start),
                end=float(clock[0]), window=np.asarray(window), parent_time=parent_time,
                regional=regional, physical=physical, static=static)


def same_file(a, b):
    a, b = Path(a), Path(b)
    if a == b:
        return 0
    if plain_file(a).st_size != plain_file(b).st_size:
        raise ValueError('frozen physical resources differ in size')
    compared = 0
    with a.open('rb') as left, b.open('rb') as right:
        while True:
            x, y = left.read(65536), right.read(65536)
            compared += len(x) + len(y)
            if x != y:
                raise ValueError('frozen physical/grid resources differ')
            if not x:
                return compared


def check_segments(segments):
    first = segments[0]
    seen = set()
    seams = []
    compared = 0
    for index, segment in enumerate(segments):
        if segment['path'] in seen:
            raise ValueError('duplicate statistics source')
        seen.add(segment['path'])
        if (segment['unique_shape'] != first['unique_shape'] or
                not np.array_equal(segment['record'][8:], first['record'][8:]) or
                segment['physical'] != first['physical'] or set(segment['static']) != set(first['static'])):
            raise ValueError('statistics definitions, mesh producers or frozen physical inputs differ')
        coarse = np.float32 if min(segment['dtype'].itemsize, first['dtype'].itemsize) == 4 else np.float64
        if not np.array_equal(segment['window'].astype(coarse), first['window'].astype(coarse)):
            raise ValueError('requested statistics windows differ')
        for name in first['static']:
            compared += same_file(first['static'][name], segment['static'][name])
        if not index:
            continue
        previous = segments[index - 1]
        if segment['start_step'] < previous['end_step']:
            raise ValueError('overlapping accumulated sampling histories')
        gap = segment['start_step'] > previous['end_step']
        if not gap and (segment['record'][6] != previous['dtype'].itemsize or
                        segment['record'][7] != previous['header'][9]):
            raise ValueError('adjacent step identities lack the matching parent clock')
        if gap and segment['start'] < previous['end']:
            raise ValueError('disjoint step intervals overlap in physical time')
        seams.append(dict(gap=gap, previous_end_step=previous['end_step'],
                          next_start_step=segment['start_step'],
                          previous_end_time=previous['end'], next_start_time=segment['start'],
                          clock_delta=segment['start'] - previous['end']))
    return seams, compared


def merge_moments(accumulated, incoming):
    """Merge unnormalised central moments; no compensation or RMS averaging."""
    if accumulated.shape != incoming.shape or accumulated.shape[-1] != 26:
        raise ValueError('invalid moment component layout')
    if not np.isfinite(incoming).all() or not np.isfinite(accumulated).all():
        raise ValueError('nonfinite sufficient statistics')
    if np.any(incoming[:, :2] < 0) or np.any((incoming[:, 0] == 0) != (incoming[:, 1] == 0)):
        raise ValueError('inconsistent time and density-time weights')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        for weight_slot, mean_start, central_start in ((0, 2, 8), (1, 5, 17)):
            total = accumulated[:, weight_slot]
            weight = incoming[:, weight_slot]
            active = weight > 0
            new_total = total + weight
            ratio = np.divide(weight, new_total, out=np.zeros_like(total), where=new_total > 0)
            factor = np.divide(total, new_total, out=np.zeros_like(total), where=new_total > 0) * weight
            delta = np.zeros_like(incoming[:, mean_start:mean_start + 3])
            np.subtract(incoming[:, mean_start:mean_start + 3], accumulated[:, mean_start:mean_start + 3],
                        out=delta, where=active[:, None])
            for j in range(3):
                for i in range(j + 1):
                    upper, lower = central_start + i + 3*j, central_start + j + 3*i
                    if not np.array_equal(incoming[active, upper], incoming[active, lower]):
                        raise ValueError('asymmetric source central moments')
                    if i == j and np.any(incoming[active, upper] < 0):
                        raise ValueError('negative source central variance')
                    entry = (accumulated[:, upper] + np.where(active, incoming[:, upper], 0.0) +
                             (factor * delta[:, i]) * delta[:, j])
                    accumulated[:, upper] = entry
                    accumulated[:, lower] = entry
            accumulated[:, mean_start:mean_start + 3] += ratio[:, None] * delta
            total[:] = new_total
    if not np.isfinite(accumulated).all():
        raise ValueError('nonfinite merged sufficient statistics')


def tiles(shape, buffer_bytes):
    capacity = buffer_bytes // ARRAY_BYTES_PER_POINT
    block = [1, 1, 1]
    for axis in (2, 1, 0):
        block[axis] = min(shape[axis], capacity)
        capacity //= block[axis]
    for origin in np.ndindex(*(int((n + b - 1) // b) for n, b in zip(shape, block))):
        yield tuple(slice(o*b, min((o + 1)*b, n)) for o, b, n in zip(origin, block, shape))


def create_results(group, shape):
    for name in ('duration', 'mass_duration', 'mean_density'):
        group.create_dataset(name, shape=shape, dtype='<f8')
    group.create_dataset('covered', shape=shape, dtype='u1')
    for kind in MOMENT_NAMES:
        for name in ('mean', 'rms'):
            group.create_dataset(f'{name}_{kind}', shape=shape + (3,), dtype='<f8')
        for name in ('central', 'covariance'):
            group.create_dataset(f'{name}_{kind}', shape=shape + (3, 3), dtype='<f8')
    group.create_dataset('density_stress', shape=shape + (3, 3), dtype='<f8')


def write_results(group, selection, values, block_shape):
    time, mass = values[:, 0], values[:, 1]
    covered = time > 0
    group['duration'][selection] = time.reshape(block_shape)
    group['mass_duration'][selection] = mass.reshape(block_shape)
    group['covered'][selection] = covered.reshape(block_shape).astype(np.uint8)
    group['mean_density'][selection] = np.divide(mass, time, out=np.zeros_like(time), where=covered).reshape(block_shape)
    for name, mean_start, central_start, weight in (('reynolds', 2, 8, time), ('favre', 5, 17, mass)):
        group[f'mean_{name}'][selection] = values[:, mean_start:mean_start + 3].reshape(block_shape + (3,))
        for j in range(3):
            for i in range(3):
                central = values[:, central_start + i + 3*j]
                covariance = np.divide(central, weight, out=np.zeros_like(weight), where=weight > 0)
                group[f'central_{name}'][selection + (i, j)] = central.reshape(block_shape)
                group[f'covariance_{name}'][selection + (i, j)] = covariance.reshape(block_shape)
                if i == j:
                    group[f'rms_{name}'][selection + (i,)] = np.sqrt(covariance).reshape(block_shape)
                if name == 'favre':
                    stress = np.divide(central, time, out=np.zeros_like(time), where=covered)
                    group['density_stress'][selection + (i, j)] = stress.reshape(block_shape)


def merge_segments(paths, output, *, buffer_bytes=1048576):
    if not paths:
        raise ValueError('select at least one statistics segment')
    if buffer_bytes < ARRAY_BYTES_PER_POINT:
        raise ValueError('buffer must reserve at least 1024 bytes for one point')
    segments = [read_segment(path) for path in paths]
    seams, compared = check_segments(segments)
    shape = segments[0]['unique_shape']
    # Validate the tile capacity before creating any destination.
    first_tile = next(tiles(shape, buffer_bytes))
    reserved = int(np.prod([s.stop - s.start for s in first_tile])) * ARRAY_BYTES_PER_POINT
    output = Path(output).absolute()
    for segment in segments:
        if output.resolve().is_relative_to(segment['path'].parent.resolve()):
            raise ValueError('merge output must not be inside a source checkpoint')
    output.mkdir(parents=True, exist_ok=False)
    provenance = []
    for s in segments:
        lo, hi = max(s['start'], s['window'][0]), min(s['end'], s['window'][1])
        provenance.append(dict(path=str(s['path']), precision=f"fp{s['dtype'].itemsize * 8}",
            start_step=s['start_step'], end_step=s['end_step'], start_time=s['start'], end_time=s['end'],
            requested_window=s['window'].tolist(), clipped_interval=[lo, hi] if hi > lo else None))
    intervals = [p['clipped_interval'] for p in provenance if p['clipped_interval'] is not None]
    leading = bool(not intervals or intervals[0][0] > segments[0]['window'][0])
    trailing = bool(not intervals or intervals[-1][1] < segments[-1]['window'][1])
    report = dict(format='ASTR_MERGED_VELOCITY_STATISTICS_1', numerical_precision='fp64',
                  source_precisions=sorted({p['precision'] for p in provenance}), sources=provenance,
                  shape_kji=shape, unique_periodic_nodes=True, seams=seams,
                  continuous_step_coverage=not any(s['gap'] for s in seams),
                  leading_window_coverage_missing=leading, trailing_window_coverage_missing=trailing,
                  raw_clock_coordinates_continuous=all(not s['gap'] and s['clock_delta'] == 0 for s in seams),
                  geometry_code=''.join(f'{int(n):08x}' for n in segments[0]['record'][13:]),
                  source_trajectory_exact=False, buffer_bytes=buffer_bytes,
                  regional_definition='temporal statistics of geometric spatial-mean velocity; unit density weight',
                  reserved_controlled_array_bytes=reserved, resource_compare_bytes=compared,
                  third_party_memory_included=False, restartable=False,
                  zero_coverage='covered=0; zero placeholders are not instantaneous statistics')
    regional = np.zeros((1, 26), dtype=np.float64)
    for segment in segments:
        merge_moments(regional, segment['regional'][7:33][None, :])
    covered_points = 0
    duration_min, duration_max = float('inf'), -float('inf')
    with h5py.File(output / 'statistics.h5', 'x') as destination, np.errstate(over='raise', invalid='raise', divide='raise'):
        destination.attrs.update(format=report['format'], numerical_precision='fp64',
                                 units='dimensionless' if segments[0]['record'][9] == 0 else 'si',
                                 definition='clipped endpoint trapezoid; unnormalised central moments',
                                 source_trajectory_exact=0)
        point = destination.create_group('point')
        region = destination.create_group('regional')
        point.attrs.update(quantity='nodal fluid velocity', density_weight='physical fluid density')
        region.attrs.update(quantity='geometric spatial-mean velocity signal', density_weight='dimensionless unity')
        create_results(point, shape)
        create_results(region, (1,))
        region['mean_density'].attrs['definition'] = 'constant unit statistical weight, not fluid density'
        region['density_stress'].attrs['definition'] = 'unit-weight signal covariance, not fluid density-weighted stress'
        region['mean_favre'].attrs['definition'] = 'unit-weight signal mean; equals Reynolds signal mean'
        write_results(region, (slice(0, 1),), regional, (1,))
        for selection in tiles(shape, buffer_bytes):
            block_shape = tuple(s.stop - s.start for s in selection)
            count = int(np.prod(block_shape))
            values = np.zeros((count, 26), dtype=np.float64)
            for segment in segments:
                with h5py.File(segment['path'], 'r') as source:
                    for component, expected in ((1, segment['window'][0]), (2, segment['window'][1]),
                                                (3, segment['end']), (34, 1)):
                        if not np.all(source[f'q{component:04d}'][selection] == expected):
                            raise ValueError('point window/sample identity differs from the segment')
                    incoming = np.stack([source[f'q{component:04d}'][selection].astype(np.float64).reshape(-1)
                                         for component in range(8, 34)], axis=-1)
                merge_moments(values, incoming)
                del incoming
            covered_points += int(np.count_nonzero(values[:, 0] > 0))
            duration_min = min(duration_min, float(values[:, 0].min()))
            duration_max = max(duration_max, float(values[:, 0].max()))
            write_results(point, selection, values, block_shape)
        report.update(covered_point_count=covered_points, point_count=int(np.prod(shape)),
                      duration_range=[duration_min, duration_max])
        destination.attrs['provenance_json'] = json.dumps(report, allow_nan=False)
    with (output / 'COMPLETE.json.tmp').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    (output / 'COMPLETE.json.tmp').rename(output / 'COMPLETE.json')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('statistics', type=Path, nargs='+', help='completed checkpoint statistics.h5, chronological order')
    parser.add_argument('--output', type=Path, required=True, help='new result directory')
    parser.add_argument('--buffer-bytes', type=int, default=1048576)
    args = parser.parse_args()
    print(json.dumps(merge_segments(args.statistics, args.output, buffer_bytes=args.buffer_bytes), indent=2))


if __name__ == '__main__':
    main()
