#!/usr/bin/env python3
"""Replay captured M12 sensor inputs and apply the approved D6 gate."""

import argparse
import json
from pathlib import Path

import numpy as np


def read_detail(path):
    with path.open('rb') as stream:
        header = np.fromfile(stream, np.int32, 4)
        if header.size != 4 or np.any(header[:3] < 1) or np.any(header[:3] > [64, 32, 24]) or header[3] < 1:
            raise ValueError(f'invalid bounded M12 sensor grid: {path}')
        shape = tuple(int(v + 1) for v in header[:3])
        halo_shape = tuple(int(v + 2 * header[3] + 1) for v in header[:3])
        old_size = 16 + 8 * (int(np.prod(shape)) * 9 + int(np.prod(halo_shape)))
        if path.stat().st_size == old_size and tuple(header[:3]) == (64, 32, 24):
            # Immutable NP=1 diagnosis files predate explicit partition metadata.
            header = np.concatenate((header, np.array([4, 4, 0], dtype=np.int32)))
        elif path.stat().st_size == old_size + 12:
            header = np.concatenate((header, np.fromfile(stream, np.int32, 3)))
        else:
            raise ValueError(f'invalid sensor-detail header/payload size: {path}')
        if np.any(header[4:] < 0) or np.any(header[4:] > 4):
            raise ValueError(f'invalid partition boundary metadata: {path}')
        gradient = np.fromfile(stream, np.float64, int(np.prod(shape)) * 9)
        pressure = np.fromfile(stream, np.float64, int(np.prod(halo_shape)))
        if gradient.size != np.prod(shape) * 9 or pressure.size != np.prod(halo_shape) or stream.read(1):
            raise ValueError(f'invalid sensor-detail payload: {path}')
    gradient = gradient.reshape((*shape, 3, 3), order='F')
    if not np.isfinite(gradient).all():
        raise ValueError(f'nonfinite physical gradient: {path}')
    return header, gradient, pressure.reshape(halo_shape, order='F')


def read_sensor(path, shape):
    count = int(np.prod(shape))
    with path.open('rb') as stream:
        header = np.fromfile(stream, np.int32, 3)
        sensor = np.fromfile(stream, np.float64, count)
        mask = np.fromfile(stream, np.int8, count)
        if tuple(header + 1) != shape or sensor.size != count or mask.size != count or stream.read(1):
            raise ValueError(f'invalid raw-sensor payload: {path}')
    if not np.isfinite(sensor).all():
        raise ValueError(f'nonfinite raw sensor: {path}')
    return sensor.reshape(shape, order='F'), mask


def replay(header, gradient, pressure):
    intervals = tuple(int(v) for v in header[:3])
    hm = int(header[3])
    indices = [np.arange(v + 1) + hm for v in intervals]
    center = pressure[np.ix_(*indices)]
    used_pressure = [center]
    curvature = []
    boundary = header[4:] if len(header) == 7 else (4, 4, 0)
    # Clamp only physical faces; MPI and periodic faces consume captured halos.
    for axis, count in enumerate(intervals):
        lower = indices[axis] - 1
        upper = indices[axis] + 1
        if boundary[axis] in (1, 4):
            lower = np.maximum(lower, hm)
        if boundary[axis] in (2, 4):
            upper = np.minimum(upper, hm + count)
        neighbors = indices.copy()
        neighbors[axis] = lower
        pm = pressure[np.ix_(*neighbors)]
        neighbors[axis] = upper
        pp = pressure[np.ix_(*neighbors)]
        used_pressure.extend((pm, pp))
        if not all(np.isfinite(p).all() and np.all(p > 0) for p in (center, pm, pp)):
            raise ValueError('nonfinite/nonpositive pressure used by the sensor stencil')
        curvature.append(np.abs(pp - 2 * center + pm) / (pp + 2 * center + pm))
    div = gradient[..., 0, 0] + gradient[..., 1, 1] + gradient[..., 2, 2]
    curl = np.stack((gradient[..., 2, 1] - gradient[..., 1, 2],
                     gradient[..., 0, 2] - gradient[..., 2, 0],
                     gradient[..., 1, 0] - gradient[..., 0, 1]), axis=-1)
    div2 = div**2
    vort2 = np.sum(curl**2, axis=-1)
    factor = div2 / (div2 + vort2 + 1e-30)
    curvature_max = np.maximum.reduce(curvature)
    if not all(np.isfinite(value).all() for value in (div, curl, div2, vort2, factor, curvature_max)):
        raise ValueError('nonfinite FP64 sensor replay arithmetic')
    return {'div': div, 'curl': curl, 'div2': div2, 'vort2': vort2,
            'factor': factor, 'curvature': curvature_max,
            'sensor': factor * curvature_max, 'used_pressure': used_pressure}


def validate_snapshot(gradient_difference, pressure_difference, residuals, masks):
    if not np.isfinite([gradient_difference, pressure_difference, *residuals]).all():
        raise ValueError('nonfinite sensor gate diagnostics')
    if max(gradient_difference, pressure_difference) > 2e-10:
        raise ValueError('sensor captured gradient/used-pressure gate failed')
    if max(residuals) > 2e-10:
        raise ValueError('captured inputs do not explain sensor output')
    if not np.array_equal(*masks):
        raise ValueError('CPU/GPU characteristic activation masks differ')


def analyze_pair(cpu, gpu, expected_snapshots=None):
    details = sorted((cpu / 'diagnostics').glob('state.sensor_detail.*.bin'))
    candidates = sorted((gpu / 'diagnostics').glob('state.sensor_detail.*.bin'))
    if not details or [p.name for p in details] != [p.name for p in candidates]:
        raise ValueError('missing paired production sensor inputs')
    if expected_snapshots is not None and len(details) != expected_snapshots:
        raise ValueError('missing three-stage/ten-step sensor inputs')
    results = []
    for reference, candidate in zip(details, candidates):
        hc, gc, pc = read_detail(reference)
        hg, gg, pg = read_detail(candidate)
        if not np.array_equal(hc, hg):
            raise ValueError('sensor input headers differ')
        shape = gc.shape[:3]
        sensor_name = reference.name.replace('.sensor_detail.', '.sensor.')
        sc, mc = read_sensor(reference.with_name(sensor_name), shape)
        sg, mg = read_sensor(candidate.with_name(sensor_name), shape)
        rc, rg = replay(hc, gc, pc), replay(hg, gg, pg)
        residuals = [float(np.max(np.abs(r['sensor'] - s))) for r, s in ((rc, sc), (rg, sg))]
        gradient_difference = float(np.max(np.abs(gg - gc)))
        pressure_difference = max(float(np.max(np.abs(a - b)))
                                 for a, b in zip(rc['used_pressure'], rg['used_pressure']))
        validate_snapshot(gradient_difference, pressure_difference, residuals, (mc, mg))
        difference = np.abs(sg - sc)
        point = np.unravel_index(difference.argmax(), shape)
        gradient_effect = (rg['factor'] - rc['factor']) * rc['curvature']
        pressure_effect = rg['factor'] * (rg['curvature'] - rc['curvature'])
        decomposition_residual = float(np.max(np.abs((sg - sc) - gradient_effect - pressure_effect)))
        if decomposition_residual > 2e-10:
            raise ValueError('sensor difference decomposition does not close')
        inputs = {}
        for backend, gradient, r, sensor in (('cpu', gc, rc, sc), ('gpu', gg, rg, sg)):
            inputs[backend] = {key: float(r[key][point]) for key in
                               ('div', 'div2', 'vort2', 'factor', 'curvature')}
            inputs[backend].update(gradient=gradient[point].tolist(), curl=r['curl'][point].tolist(),
                                   sensor=float(sensor[point]))
        results.append({'snapshot': sensor_name, 'worst_physical_node': [int(v) for v in point],
                        'raw_sensor_max_abs': float(difference.max()),
                        'gradient_max_abs': gradient_difference,
                        'sensor_used_pressure_max_abs': pressure_difference,
                        'mask_mismatch_nodes': int(np.count_nonzero(mc != mg)),
                        'own_input_replay_max_abs_cpu_gpu': residuals,
                        'decomposition_max_abs_residual': decomposition_residual,
                        'gradient_factor_effect_at_worst_node': float(gradient_effect[point]),
                        'pressure_curvature_effect_at_worst_node': float(pressure_effect[point]),
                        'worst_node_inputs': inputs})
    return {'purpose': 'D6 captured-input replay and exact activation-mask gate',
            'passed': True, 'strict_absolute_tolerance': 2e-10,
            'raw_sensor_between_trajectories': 'reported, not a universal absolute gate',
            'expected_snapshots': expected_snapshots,
            'cpu': str(cpu.resolve()), 'gpu': str(gpu.resolve()), 'snapshots': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', type=Path, required=True)
    parser.add_argument('--gpu', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = analyze_pair(args.cpu, args.gpu)
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f"D6 passed for {len(receipt['snapshots'])} captured-input pairs; raw differences reported")
