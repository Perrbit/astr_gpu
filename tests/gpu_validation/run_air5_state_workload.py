#!/usr/bin/env python3
"""Replay the U1 bounded fixture's exact per-rank chemistry inputs, not CFD."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np

from run_air5_performance_diagnosis import batch_identity, digest

ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = list(range(1, 12)) + list(range(17, 23))
TOLERANCES = {'loose': (1e-8, 1e-12), 'baseline': (1e-9, 1e-13), 'tight': (1e-10, 1e-14)}


def rank_inputs(state, rank, include_history=False):
    """Restore physical nodes, including non-owner shared copies from extras."""
    header = state['identity'][:]
    components = int(header[6])
    if header[1:3].tolist() != [2, 1] or components not in (34, 35) or header[12] != 7:
        raise ValueError('require completed-phase schema-2 AIR5 exact state')
    if include_history and components != 35:
        raise ValueError('saved history is missing from checkpoint')
    if state['metadata'][:].tolist() != [1, 1]:
        raise ValueError('require compensated AIR5 checkpoint')
    partitions = state['partitions'][:].reshape(-1, 8)
    origin, cells = partitions[rank, :3], partitions[rank, 3:6]
    halo, extra_count = map(int, partitions[rank, 6:])
    physical_shape = cells + 1
    shape = physical_shape + 2 * halo
    # C-order flattening of z,y,x is the checkpoint's Fortran x,y,z order.
    z, y, x = np.indices(tuple(shape[::-1]))
    indices = np.stack((x.ravel(), y.ravel(), z.ravel()))
    owned = cells + (origin + cells == header[3:6] - 1)
    owner = np.all((indices >= halo) & (indices < (halo + owned)[:, None]), axis=0)
    extra_indices = indices[:, ~owner] - halo
    physical = np.all((extra_indices >= 0) & (extra_indices <= cells[:, None]), axis=0)
    if extra_count != components * len(extra_indices[0]):
        raise ValueError('rank extras layout mismatch')
    positions = np.ravel_multi_index(extra_indices[:, physical], tuple(physical_shape), order='F')
    offset = int(partitions[:rank, 7].sum())
    extra = state['rank_extras'][offset:offset + extra_count].reshape(components, -1)
    selection = tuple(slice(int(o), int(o + n)) for o, n in zip(origin[::-1], physical_shape[::-1]))
    selected = COMPONENTS + ([35] if include_history else [])
    values = np.empty((int(np.prod(physical_shape)), len(selected)))
    for column, component in enumerate(selected):
        values[:, column] = state[f'q{component:04d}'][selection].ravel()
        values[positions, column] = extra[component - 1, physical]
    if include_history and (not np.isfinite(values[:, -1]).all() or np.any(values[:, -1] < 0)):
        raise ValueError('invalid saved step history')
    z, y, x = np.indices(tuple(physical_shape[::-1]))
    coordinates = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1) + origin
    return values, coordinates, header[3:6] - 1


def chemistry_regions(coordinates, global_cells, dynamic_top):
    """U1 fixture: open x/y, periodic z; shared nodes are real rank work."""
    interior = ((coordinates[:, 0] > 0) & (coordinates[:, 0] < global_cells[0]) &
                (coordinates[:, 1] > 0) & (coordinates[:, 1] < global_cells[1]))
    top = (coordinates[:, 0] > 0) & (coordinates[:, 1] == global_cells[1])
    regions = {'interior': interior}
    if dynamic_top and np.any(top):
        regions['dynamic_top'] = top
    return regions


def distribution(values):
    values = np.asarray(values)
    unique, counts = np.unique(values, return_counts=True)
    return dict(min=int(values.min()), median=float(np.median(values)),
                p95=float(np.quantile(values, .95)), p99=float(np.quantile(values, .99)),
                max=int(values.max()), total=int(values.sum()),
                histogram={str(int(k)): int(n) for k, n in zip(unique, counts)})


def parse_results(path, count):
    rows = np.loadtxt(path, ndmin=2)
    if rows.shape != (count, 19) or not np.isfinite(rows).all():
        raise ValueError('nonfinite/incomplete state batch')
    if not np.array_equal(rows[:, 0], np.arange(1, count + 1)) or np.any(rows[:, 1] != 0):
        raise ValueError('failed or reordered state batch')
    if np.any(rows[:, 2:6] < 0) or np.any(rows[:, 2:6] != np.floor(rows[:, 2:6])):
        raise ValueError('invalid logical work counts')
    if np.any(rows[:, 6:11] < 0) or np.any(rows[:, 18] <= 0):
        raise ValueError('negative species or nonpositive suggested step')
    return {name: distribution(rows[:, column]) for column, name in enumerate(
        ('accepted', 'rejected', 'logical_rhs', 'logical_jacobian'), 2)}


def compare_history_work(reset_input, history_input, reset_output, history_output, duration):
    """Compare work on identical states; these counts are not an accuracy gate."""
    count = len(reset_input)
    if reset_input.shape != (count, 17) or history_input.shape != (count, 18):
        raise ValueError('require paired reset/saved-history input layouts')
    if reset_input.tobytes() != history_input[:, :17].tobytes():
        raise ValueError('work comparison requires bitwise identical physical state and carry')
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('invalid work comparison duration')
    history = history_input[:, 17]
    if not np.isfinite(history).all() or np.any(history < 0):
        raise ValueError('invalid saved history')
    for rows in (reset_output, history_output):
        if rows.shape != (count, 19) or not np.isfinite(rows).all():
            raise ValueError('incomplete or nonfinite batch results')
        if not np.array_equal(rows[:, 0], np.arange(1, count + 1)) or np.any(rows[:, 1] != 0):
            raise ValueError('work comparison requires successful, identically ordered tasks')
        if np.any(rows[:, 2:6] < 0) or np.any(rows[:, 2:6] != np.floor(rows[:, 2:6])):
            raise ValueError('invalid work counts')
    old, new = reset_output[:, 2:6].astype(np.int64), history_output[:, 2:6].astype(np.int64)
    delta = new - old
    changed = np.any(delta != 0, axis=1)
    reduced = (history > 0) & (history < duration)
    simt_proxy = {}
    # The fixture's rectangular kernel order matches each region's row order.
    # This models loop bounds only, not occupancy or instruction/clock cost.
    for name, counts in (('reset', old), ('history', new)):
        attempts = counts[:, :2].sum(axis=1)
        warp_bounds = [int(attempts[start:start + 32].max()) for start in range(0, count, 32)]
        slots = 32 * sum(warp_bounds)
        simt_proxy[name] = dict(warps=len(warp_bounds), warp_loop_bound_sum=sum(warp_bounds),
            lane_attempt_slots=slots, useful_attempts=int(attempts.sum()))
    return dict(tasks=count, changed_tasks=int(changed.sum()), reduced_history_tasks=int(reduced.sum()),
        changed_full_duration_tasks=int((changed & ~reduced).sum()),
        more_attempt_tasks=int(np.count_nonzero(delta[:, :2].sum(axis=1) > 0)),
        fewer_attempt_tasks=int(np.count_nonzero(delta[:, :2].sum(axis=1) < 0)),
        reset_totals=old.sum(axis=0).tolist(), history_totals=new.sum(axis=0).tolist(),
        delta_totals=delta.sum(axis=0).tolist(),
        reset_max_attempts=int(old[:, :2].sum(axis=1).max()),
        history_max_attempts=int(new[:, :2].sum(axis=1).max()),
        attempt_simt_proxy=simt_proxy,
        changed_rows=[dict(row=int(index + 1), history_ratio=float(history[index] / duration),
            reset_counts=old[index].tolist(), history_counts=new[index].tolist(),
            delta=delta[index].tolist()) for index in np.flatnonzero(changed)])


def run(args):
    if args.checkpoint is not None:
        batch = args.checkpoint.resolve(strict=True)
        with h5py.File(batch / 'state.h5') as state:
            grid = (state['identity'][3:6] - 1).tolist()
        config = (batch / 'air5_config.bin').read_bytes()
        if config[:8] not in (b'ASTRA501', b'ASTRA502', b'ASTRA503', b'ASTRA504') or \
                config[8:136].decode('ascii').strip() != 'coupled':
            raise ValueError('require coupled AIR5 HBL/SBLI fixture configuration')
        top_mode = config[8 + 4 * 128:8 + 5 * 128].decode('ascii').strip()
        if top_mode not in ('prescribed', 'characteristic'):
            raise ValueError('invalid fixture top mode')
        baseline = dict(status='checkpoint-read-only', grid_intervals=grid,
            case='sbli' if config[:8] in (b'ASTRA502', b'ASTRA504') else 'hbl',
            top_mode=top_mode, cases={'gpu_checkpoint': dict(seed=str(batch), seed_files_sha256=batch_identity(batch))})
    else:
        baseline = json.loads((args.baseline / 'timing_summary.json').read_text())
    if baseline['status'] not in ('bounded-timing-collected', 'checkpoint-read-only') or baseline['grid_intervals'] != [16, 16, 16]:
        raise ValueError('require successful U1 16-cubed baseline')
    if baseline['case'] not in ('hbl', 'sbli'):
        raise ValueError('unsupported fixture boundary layout')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', scope='isolated next-first-half replay from completed-step states; '
                  'not the work distribution of the subsequent transport or second chemistry half',
                  baseline=str((args.checkpoint or args.baseline).resolve()), probe_sha256=digest(args.probe),
                  chemistry_inputs='q1:11 and carry q17:22' + (' and q35 history' if args.step_history == 'previous' else '') +
                      '; exact non-owner physical copies restored',
                  count_semantics='logical ROS-2 counters; actual tangent operations require compiled attribution',
                  gpu=int(args.gpu), source_evaluation=args.source_evaluation,
                  tolerance=args.tolerance, step_history=args.step_history, cases={})
    try:
        for label, case in baseline['cases'].items():
            if not label.startswith('gpu_'):
                continue
            batch = Path(case['seed'])
            before = batch_identity(batch)
            if before != case['seed_files_sha256']:
                raise ValueError('baseline checkpoint changed')
            entry = dict(checkpoint=str(batch), checkpoint_sha256=before, ranks={})
            report['cases'][label] = entry
            with h5py.File(batch / 'state.h5') as state:
                header = state['identity'][:]
                duration = np.asarray(header[11], dtype=np.int64).view(np.float64).item() / 2
                entry.update(step=int(header[8]), duration=duration)
                for rank in range(int(header[7])):
                    values, coordinates, global_cells = rank_inputs(state, rank, args.step_history == 'previous')
                    for name, mask in chemistry_regions(coordinates, global_cells,
                            baseline['top_mode'] == 'characteristic').items():
                        inputs = values[mask]
                        if not 0 < len(inputs) <= 10000 or inputs.nbytes * 4 > 64 * 1024**2:
                            raise ValueError('batch outside existing bounded probe budget')
                        stem = args.output / f'{label}_rank{rank}_{name}'
                        input_path, output_path = stem.with_suffix('.in'), stem.with_suffix('.out')
                        with input_path.open('w') as stream:
                            stream.write(f'{len(inputs)} {duration:.17e} 0\n')
                            np.savetxt(stream, inputs, fmt='%.17e')
                        rtol, atol_factor = TOLERANCES[args.tolerance]
                        command = [str(args.probe.resolve()), 'batch', str(input_path.resolve()), str(output_path.resolve()),
                                   f'{rtol:.17e}', f'{atol_factor:.17e}', '1',
                                   'history-from-input' if args.step_history == 'previous' else 'reset']
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
                                   ASTR_AIR5_SOURCE_EVALUATION=args.source_evaluation)
                        with stem.with_suffix('.log').open('w') as log:
                            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
                        entry['ranks'].setdefault(str(rank), {})[name] = dict(
                            cells=len(inputs), nonzero_input_carry=int(np.count_nonzero(inputs[:, 11:17])),
                            nonzero_input_history=int(np.count_nonzero(inputs[:, 17:])),
                            input_sha256=digest(input_path), output_sha256=digest(output_path),
                            command=command, work=parse_results(output_path, len(inputs)))
            if batch_identity(batch) != before:
                raise ValueError('read-only replay changed source checkpoint')
        report['status'] = 'bounded-state-workload-collected'
    except BaseException as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        (args.output / 'workload.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(report['status'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--baseline', type=Path)
    source.add_argument('--checkpoint', type=Path, help='Read-only 16-cubed AIR5 fixture checkpoint')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe', type=Path, default=ROOT / 'build_gpu_probe/bin/chemistry_ros2_compensation_gpu_probe')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--source-evaluation', choices=('joint', 'source-only'), default='joint')
    parser.add_argument('--tolerance', choices=tuple(TOLERANCES), default='baseline')
    parser.add_argument('--step-history', choices=('off', 'previous'), default='off')
    run(parser.parse_args())
