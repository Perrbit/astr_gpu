#!/usr/bin/env python3
"""Current-output AIR5 complete-step baseline; historical legacy runs remain archived."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess

from run_output_air5_restart_validation import DT, ROOT, launch, state_sanity
from run_output_restart_validation import compare_fields


def timing_records(text, ranks, start, updates):
    if ranks < 1 or updates < 1:
        raise ValueError('positive ranks and updates required')
    records = {}
    for line in text.splitlines():
        if not line.startswith('ASTR_COMPLETE_STEP_TIMING '):
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError('malformed complete-step timing record')
        step, rank = map(int, parts[1:3])
        duration = float(parts[3])
        if not start <= step < start+updates or not 0 <= rank < ranks:
            raise ValueError('unexpected timing step/rank')
        if not math.isfinite(duration) or duration <= 0 or (step, rank) in records:
            raise ValueError('invalid or duplicate timing')
        records[step, rank] = duration
    if len(records) != ranks*updates:
        raise ValueError('incomplete timing records')
    return records


def step_samples(text, ranks, start, updates):
    """Historical supplemental metric: maximum rank separately at every step."""
    records = timing_records(text, ranks, start, updates)
    return [max(records[step, rank] for rank in range(ranks)) for step in range(start, start+updates)]


def window_summary(text, ranks, start, updates, warmup):
    if not 0 <= warmup < updates:
        raise ValueError('warmup must leave measured steps')
    records = timing_records(text, ranks, start, updates)
    measured = range(start+warmup, start+updates)
    totals = [math.fsum(records[step, rank] for step in measured) for rank in range(ranks)]
    return dict(rank_window_seconds=totals, critical_rank=totals.index(max(totals)),
                seconds_per_step=max(totals)/(updates-warmup),
                supplemental_step_max_seconds=[max(records[step, rank] for rank in range(ranks))
                    for step in measured])


def phase_summary(text, ranks, start, updates, warmup):
    chemistry = ('setup', 'kernels', 'failure_reduce', 'report_reduce', 'boundary',
                 'interior_primitives', 'solution_halo', 'face_primitives')
    transport = ('filter', 'spatial_state', 'sensor', 'convection_raw', 'convection_limiter_top',
                 'viscous_flux', 'viscous_halo', 'viscous_limiter_rhs', 'rk_update', 'boundary')
    phases = [(f'chem_{name}', half) for name in chemistry for half in (1, 2)]
    phases += [(f'transport_{name}', rk) for name in transport for rk in (1, 2, 3)]
    expected = {(rank, step, stage, label) for rank in range(ranks)
                for step in range(start, start+updates) for label, stage in phases}
    records = {}
    for line in text.splitlines():
        if not line.startswith('ASTR_AIR5_PHASE_TIMING '):
            continue
        parts = line.split()
        if len(parts) != 6:
            raise ValueError('malformed AIR5 phase timing')
        key = (*map(int, parts[1:4]), parts[4])
        value = float(parts[5])
        if key not in expected or key in records or not math.isfinite(value) or value < 0:
            raise ValueError('unexpected/duplicate/nonfinite AIR5 phase timing')
        records[key] = value
    if set(records) != expected:
        raise ValueError('incomplete AIR5 phase timings')
    labels = sorted({label for label, _ in phases})
    return {str(rank): {label: math.fsum(value for (r, step, stage, name), value in records.items()
                        if r == rank and step >= start+warmup and name == label)/(updates-warmup)
                       for label in labels} for rank in range(ranks)}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def batch_identity(batch):
    if not (batch/'COMPLETE').is_file() or not (batch/'state.h5').is_file():
        raise ValueError('require a completed new checkpoint batch, not a legacy flowfield.h5')
    return {str(path.relative_to(batch)): digest(path) for path in sorted(batch.rglob('*')) if path.is_file()}


def check_no_field_output(case):
    large = [str(p.relative_to(case)) for pattern in ('*.h5', '*.bin', '*.vtk', '*.vtu', '*.jpeg', '*.eps')
             for p in (case/'outdat').rglob(pattern)]
    validation = list((case/'validation').glob('*.bin'))
    if large or validation or list((case/'outdat/new').rglob('COMPLETE')):
        raise ValueError(f'field/validation output during timing: {large or validation}')


def run(args):
    diagnostic = args.nsys or args.phase_timing
    if args.repeats < (1 if diagnostic else 3) or args.warmup < 1 or args.steps < 1 or args.report_interval < 1:
        raise ValueError('require >=3 unprofiled repeats (>=1 profiled), >=1 warmup/measured step/report interval')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    args.executable = args.executable.resolve(strict=True)
    args.mpiexec = args.mpiexec.resolve(strict=True)
    args.reconstruction = 3 if args.case == 'sbli' else 5
    args.mean_statistics, args.sample_interval = False, 2
    args.mode, args.initial_restart = 'steps', False
    binary_hash = digest(args.executable)
    source_diff = subprocess.check_output(['git', 'diff', '--', 'src', 'src_gpu',
        'tests/gpu_validation/run_air5_performance_diagnosis.py',
        'tests/gpu_validation/run_output_air5_restart_validation.py'], cwd=ROOT)
    (args.output/'source_changes.patch').write_bytes(source_diff)
    updates = args.warmup+args.steps
    summary = dict(status='running', cases={}, warmup=args.warmup, measured_steps=args.steps,
        repeats=args.repeats, dt=DT, grid_intervals=[16, 16, 16], case=args.case,
        executable=str(args.executable), executable_sha256=binary_hash,
        source_diff_sha256=hashlib.sha256(source_diff).hexdigest(),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        scope='bounded AIR5 fixture, not physical/production acceptance',
        timing_scope='crashcheck + chemistry/transport/chemistry + completed-step scalar monitor; startup and field I/O excluded',
        primary_metric='max over ranks of measured-window cumulative seconds / measured steps',
        report_interval=args.report_interval, profiled=args.nsys, phase_timing=args.phase_timing,
        chemistry_diagnostics=args.chemistry_diagnostics, source_evaluation=args.source_evaluation,
        chemistry_block_size=args.chemistry_block_size, axis=args.axis,
        topology=list(args.topology) if args.topology else None,
        top_mode=args.top_mode, filter_workspace=args.filter_workspace)
    try:
        baseline = None
        if args.compare_to is not None:
            baseline = json.loads((args.compare_to/'timing_summary.json').read_text())
            if baseline['status'] != 'bounded-timing-collected':
                raise ValueError('comparison requires a successful unprofiled baseline')
            for key in ('executable_sha256', 'dt', 'grid_intervals', 'case', 'report_interval',
                        'warmup', 'measured_steps', 'repeats', 'primary_metric',
                        'axis', 'topology', 'top_mode', 'filter_workspace'):
                if baseline[key] != summary[key]:
                    raise ValueError(f'baseline protocol mismatch: {key}')
        for label in args.cases:
            backend, n = label.split('_np')
            ranks = int(n)
            seed_case, _ = launch(args, backend, ranks, 'seed', 2, interval=2,
                                 chemistry_diagnostics=args.chemistry_diagnostics)
            seed = seed_case/'outdat/new/checkpoints/step000000000002'
            sanity = state_sanity(seed/'state.h5')
            identity = batch_identity(seed)
            timed_check, _ = launch(args, backend, ranks, 'timing_check', 2, interval=2,
                                    complete_step_timing=True, air5_phase_timing=args.phase_timing,
                                    chemistry_diagnostics=args.chemistry_diagnostics)
            compared = compare_fields(seed/'state.h5',
                timed_check/'outdat/new/checkpoints/step000000000002/state.h5')
            verification, _ = launch(args, backend, ranks, 'verification', 2+updates, restore=seed,
                interval=2+updates, override=True, chemistry_diagnostics=args.chemistry_diagnostics)
            final = verification/f'outdat/new/checkpoints/step{2+updates:012d}/state.h5'
            final_sanity = state_sanity(final)
            rounds = []
            summary['cases'][label] = dict(rounds=rounds, seed=str(seed), seed_files_sha256=identity,
                seed_sanity=sanity, timing_activation_equal_datasets=compared,
                verification=str(verification), final_sanity=final_sanity)
            if baseline is not None:
                reference = baseline['cases'][label]
                if batch_identity(Path(reference['seed'])) != reference['seed_files_sha256']:
                    raise ValueError('baseline checkpoint batch changed')
                compare_fields(seed/'state.h5', Path(reference['seed'])/'state.h5')
                relative = final.relative_to(verification)
                equal = compare_fields(final, Path(reference['verification'])/relative)
                for actual, expected in ((seed_case, Path(reference['seed']).parents[3]),
                                         (verification, Path(reference['verification']))):
                    reports = lambda case: [line for line in (case/'run.log').read_text().splitlines()
                                           if line.startswith('AIR5_CHEMISTRY half=')]
                    if reports(actual) != reports(expected):
                        raise ValueError('chemistry report identity/values differ')
                    if (actual/'flowstate.dat').read_bytes() != (expected/'flowstate.dat').read_bytes():
                        raise ValueError('scalar statistics differ')
                summary['cases'][label]['baseline_equal_datasets'] = equal
            for repeat in range(args.repeats):
                if digest(args.executable) != binary_hash:
                    raise ValueError('executable changed within the timing matrix')
                case, disk = launch(args, backend, ranks, f'r{repeat+1}', 2+updates,
                    restore=seed, checkpoint_enabled=False, override=True,
                    complete_step_timing=True, nsys_trace=args.nsys, air5_phase_timing=args.phase_timing,
                    chemistry_diagnostics=args.chemistry_diagnostics)
                check_no_field_output(case)
                log = (case/'run.log').read_text()
                record = window_summary(log, ranks, 2, updates, args.warmup)
                if args.phase_timing:
                    record['phase_seconds_per_step'] = phase_summary(log, ranks, 2, updates, args.warmup)
                record.update(directory=str(case), directory_bytes=disk)
                rounds.append(record)
                print(label, repeat+1, record['seconds_per_step'], flush=True)
            if batch_identity(seed) != identity:
                raise ValueError('source checkpoint batch changed')
            values = [r['seconds_per_step'] for r in rounds]
            summary['cases'][label].update(median_seconds=statistics.median(values),
                min_seconds=min(values), max_seconds=max(values),
                stdev_seconds=statistics.stdev(values) if len(values) > 1 else None)
            if baseline is not None and not diagnostic:
                summary['cases'][label]['speedup_over_baseline'] = (
                    baseline['cases'][label]['median_seconds']/statistics.median(values))
        if not diagnostic and all(k in summary['cases'] for k in ('gpu_np1', 'gpu_np2')):
            one, two = [summary['cases'][k]['median_seconds'] for k in ('gpu_np1', 'gpu_np2')]
            summary.update(two_gpu_speedup=one/two, two_gpu_efficiency=one/(2*two))
        summary['status'] = 'instrumented-attribution-not-performance' if diagnostic else 'bounded-timing-collected'
    except BaseException as exc:
        summary.update(status='failed', error=str(exc))
        raise
    finally:
        (args.output/'timing_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare-to', type=Path, help='Same-binary baseline matrix; require exact state/report/statistics equality')
    parser.add_argument('--executable', type=Path, default=ROOT/'build_gpu_probe/bin/astr')
    parser.add_argument('--mpiexec', type=Path, default=shutil.which('mpiexec'))
    parser.add_argument('--cases', nargs='+', choices=tuple(f'{b}_np{n}' for b in ('cpu', 'gpu') for n in (1, 2, 4)),
                        default=['gpu_np1', 'gpu_np2'])
    parser.add_argument('--case', choices=('hbl', 'sbli'), default='sbli')
    parser.add_argument('--axis', choices=('x', 'y', 'z'), default='x')
    parser.add_argument('--topology', type=lambda value: tuple(map(int, value.split(','))))
    parser.add_argument('--top-mode', choices=('characteristic', 'prescribed'), default='characteristic')
    parser.add_argument('--filter-workspace', choices=('scalar', 'full'), default='scalar')
    parser.add_argument('--report-interval', type=int, default=4)
    parser.add_argument('--chemistry-diagnostics', choices=('every-step', 'report-steps'), default='every-step')
    parser.add_argument('--source-evaluation', choices=('joint', 'source-only'), default='joint')
    parser.add_argument('--chemistry-block-size', type=int, choices=(32, 128), default=128)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--nsys', action='store_true', help='Separate attribution runs; never a speedup denominator')
    parser.add_argument('--phase-timing', action='store_true', help='Diagnostic host phases; adds no GPU fences')
    parser.add_argument('--runtime-timeout-seconds', type=float, default=180)
    run(parser.parse_args())
