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

from air5_checkpoint_contract import FILTER_IDENTITY, read_air5_contract
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


def compare_scalar_statistics(actual, expected, single_initial_update=False):
    actual_file, expected_file = actual/'flowstate.dat', expected/'flowstate.dat'
    # Native rkfirst intentionally skips scalar output on its first call.
    if single_initial_update and not actual_file.exists() and not expected_file.exists():
        return 'not-emitted-in-first-rk-call'
    if actual_file.read_bytes() != expected_file.read_bytes():
        raise ValueError('scalar statistics differ')
    return 'equal'


def qualified_flow_case(args, label, binary_hash):
    report = json.loads(args.flow_control.read_text())
    expected = dict(status='local-flow-control-error-passed-not-performance',
        executable_sha256=binary_hash, grid_intervals=list(args.grid), dt=DT,
        complete_steps=args.warmup+args.steps, air5_filter_mode=args.air5_filter_mode,
        mechanism_sha256=digest(ROOT/'chemMech/air5_kimjo12.json'),
        checker_sha256=digest(Path(__file__).with_name('check_air5_flow_actual_error.py')))
    for key, value in expected.items():
        if report[key] != value:
            raise ValueError(f'qualified flow protocol mismatch: {key}')
    if (not args.start_from_initial or args.case != 'sbli' or args.axis != 'x' or
            args.top_mode != 'characteristic' or args.filter_workspace != 'scalar' or
            args.report_interval != 4):
        raise ValueError('qualified flow reuse requires the original G2 phase and fixture')
    backend, ranks = label.split('_np')
    ranks = int(ranks)
    if args.topology is not None and tuple(args.topology) != (ranks, 1, 1):
        raise ValueError('qualified flow topology mismatch')
    rows = [row for row in report['cases'] if (row['backend'], row['ranks'],
        row['tolerance'], row['history']) == (backend, ranks, args.chemistry_tolerance, args.step_history)]
    if len(rows) != 1:
        raise ValueError('require one qualified flow case for the selected control/backend')
    row = rows[0]
    if row['executable_sha256'] != binary_hash:
        raise ValueError('qualified flow case executable changed')
    if not row['error']['within_reference_margin'] or not row['error']['within_candidate_error']:
        raise ValueError('qualified flow numerical gate failed')
    reference = report['reference_summary']
    if batch_identity(Path(reference['initial']).parent) != reference['initial_batch_sha256']:
        raise ValueError('qualified reference initial batch changed')
    for item in (*reference['references'].values(), row):
        if batch_identity(Path(item['state']).parent) != item['batch_files_sha256']:
            raise ValueError('qualified flow/reference checkpoint batch changed')
    contract = read_air5_contract((Path(row['state']).parent/'air5_config.bin').read_bytes())
    control = (('baseline', 'loose', 'tight').index(args.chemistry_tolerance),
               int(args.step_history == 'previous'))
    filter_control = FILTER_IDENTITY if args.air5_filter_mode == 'conservative-flux' else (0,)*5
    if (contract['source_evaluation'] != args.source_evaluation or
            contract['integration_control'] != control or contract['filter_identity'] != filter_control):
        raise ValueError('qualified flow numerical configuration mismatch')
    return row


def campaign_bytes(root):
    return sum(path.stat().st_size for path in root.rglob('*') if path.is_file())


def budgeted_launch(args, *positional, **kwargs):
    root = getattr(args, 'campaign_root', None)
    if root is not None and campaign_bytes(root)+args.directory_budget_bytes > args.campaign_budget_bytes:
        raise RuntimeError('timing campaign lacks a per-run disk reservation')
    result = launch(args, *positional, **kwargs)
    if root is not None and campaign_bytes(root) > args.campaign_budget_bytes:
        raise RuntimeError('timing campaign disk budget exceeded')
    return result


def run(args):
    diagnostic = args.nsys or args.phase_timing
    if args.repeats < (1 if diagnostic else 3) or args.warmup < 1 or args.steps < 1 or args.report_interval < 1:
        raise ValueError('require >=3 unprofiled repeats (>=1 profiled), >=1 warmup/measured step/report interval')
    args.output = args.output.resolve()
    args.air5_filter_mode = getattr(args, 'air5_filter_mode', 'legacy')
    args.flow_control = getattr(args, 'flow_control', None)
    args.campaign_root = getattr(args, 'campaign_root', None)
    if args.campaign_root is not None:
        args.campaign_root = args.campaign_root.resolve(strict=True)
        args.output.relative_to(args.campaign_root)
        if args.campaign_budget_bytes is None:
            raise ValueError('an explicit timing campaign requires its byte budget')
    args.output.mkdir(parents=True, exist_ok=False)
    args.executable = args.executable.resolve(strict=True)
    args.mpiexec = args.mpiexec.resolve(strict=True)
    args.reconstruction = 3 if args.case == 'sbli' else 5
    args.mean_statistics, args.sample_interval = False, 2
    args.mode, args.initial_restart = 'steps', args.start_from_initial
    binary_hash = digest(args.executable)
    source_diff = subprocess.check_output(['git', 'diff', '--', 'src', 'src_gpu',
        'tests/gpu_validation/run_air5_performance_diagnosis.py',
        'tests/gpu_validation/run_output_air5_restart_validation.py'], cwd=ROOT)
    (args.output/'source_changes.patch').write_bytes(source_diff)
    updates = args.warmup+args.steps
    start = 0 if args.start_from_initial else 2
    seed_updates = max(start, 1)
    output_budgets = dict(host_budget_bytes=args.host_budget_bytes,
                         device_budget_bytes=args.device_budget_bytes,
                         directory_budget_bytes=args.directory_budget_bytes)
    summary = dict(status='running', cases={}, warmup=args.warmup, measured_steps=args.steps,
        repeats=args.repeats, dt=DT, grid_intervals=list(args.grid), case=args.case, timing_start_step=start,
        executable=str(args.executable), executable_sha256=binary_hash,
        source_diff_sha256=hashlib.sha256(source_diff).hexdigest(),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        scope='bounded AIR5 fixture, not physical/production acceptance',
        timing_scope='crashcheck + chemistry/transport/chemistry + completed-step scalar monitor; startup and field I/O excluded',
        primary_metric='max over ranks of measured-window cumulative seconds / measured steps',
        report_interval=args.report_interval, profiled=args.nsys, phase_timing=args.phase_timing,
        chemistry_diagnostics=args.chemistry_diagnostics, source_evaluation=args.source_evaluation,
        chemistry_tolerance=args.chemistry_tolerance, step_history=args.step_history,
        chemistry_block_size=args.chemistry_block_size, axis=args.axis,
        topology=list(args.topology) if args.topology else None,
        top_mode=args.top_mode, filter_workspace=args.filter_workspace,
        air5_filter_mode=args.air5_filter_mode,
        qualified_flow_control=str(args.flow_control) if args.flow_control is not None else None,
        campaign_root=str(args.campaign_root) if args.campaign_root is not None else None,
        campaign_budget_bytes=getattr(args, 'campaign_budget_bytes', None), **output_budgets)
    try:
        baseline = None
        if args.compare_to is not None:
            baseline = json.loads((args.compare_to/'timing_summary.json').read_text())
            if baseline['status'] != 'bounded-timing-collected':
                raise ValueError('comparison requires a successful unprofiled baseline')
            for key in ('executable_sha256', 'dt', 'grid_intervals', 'case', 'report_interval',
                        'warmup', 'measured_steps', 'repeats', 'primary_metric',
                        'axis', 'topology', 'top_mode', 'filter_workspace', 'chemistry_tolerance', 'step_history',
                        'timing_start_step', 'air5_filter_mode'):
                if baseline[key] != summary[key]:
                    raise ValueError(f'baseline protocol mismatch: {key}')
        for label in args.cases:
            backend, n = label.split('_np')
            ranks = int(n)
            reused = qualified_flow_case(args, label, binary_hash) if args.flow_control is not None else None
            if reused is None:
                seed_case, _ = budgeted_launch(args, backend, ranks, 'seed', seed_updates, interval=seed_updates,
                    chemistry_diagnostics=args.chemistry_diagnostics, **output_budgets)
            else:
                seed_case = Path(reused['case'])
            seed = seed_case/f'outdat/new/checkpoints/step{start:012d}'
            sanity = state_sanity(seed/'state.h5')
            identity = batch_identity(seed)
            if reused is None:
                timed_check, _ = budgeted_launch(args, backend, ranks, 'timing_check', seed_updates,
                    interval=seed_updates, complete_step_timing=True, air5_phase_timing=args.phase_timing,
                    chemistry_diagnostics=args.chemistry_diagnostics, **output_budgets)
                activation_state = f'outdat/new/checkpoints/step{seed_updates:012d}/state.h5'
                compared = compare_fields(seed_case/activation_state, timed_check/activation_state)
            verification, _ = budgeted_launch(args, backend, ranks, 'verification', start+updates, restore=seed,
                interval=start+updates, override=True, chemistry_diagnostics=args.chemistry_diagnostics,
                complete_step_timing=reused is not None, air5_phase_timing=args.phase_timing, **output_budgets)
            final = verification/f'outdat/new/checkpoints/step{start+updates:012d}/state.h5'
            final_sanity = state_sanity(final)
            if reused is not None:
                compared = compare_fields(final, Path(reused['state']))
                compare_scalar_statistics(verification, seed_case)
                window_summary((verification/'run.log').read_text(), ranks, start, updates, args.warmup)
            rounds = []
            summary['cases'][label] = dict(rounds=rounds, seed=str(seed), seed_files_sha256=identity,
                seed_sanity=sanity, timing_activation_equal_datasets=compared,
                verification=str(verification), final_sanity=final_sanity)
            if reused is not None:
                summary['cases'][label]['qualified_flow_reuse'] = dict(
                    report_sha256=digest(args.flow_control), source_case=str(seed_case),
                    source_final_batch_sha256=reused['batch_files_sha256'],
                    timer_and_restart_equal_datasets=compared,
                    verification_is_performance_evidence=False)
            if baseline is not None:
                reference = baseline['cases'][label]
                if batch_identity(Path(reference['seed'])) != reference['seed_files_sha256']:
                    raise ValueError('baseline checkpoint batch changed')
                initial_equal = compare_fields(seed/'state.h5', Path(reference['seed'])/'state.h5')
                relative = final.relative_to(verification)
                equal = compare_fields(final, Path(reference['verification'])/relative)
                scalar_checks = {}
                phases = [('verification', verification, Path(reference['verification']))]
                # Step-0 batches may belong to seed generators or full G2 trajectories.
                if start > 0:
                    phases.insert(0, ('seed', seed_case, Path(reference['seed']).parents[3]))
                for phase, actual, expected in phases:
                    reports = lambda case: [line for line in (case/'run.log').read_text().splitlines()
                                           if line.startswith('AIR5_CHEMISTRY half=')]
                    if not reports(expected) or reports(actual) != reports(expected):
                        raise ValueError('chemistry report identity/values differ')
                    scalar_checks[phase] = compare_scalar_statistics(actual, expected,
                        single_initial_update=phase=='seed' and seed_updates==1)
                summary['cases'][label]['baseline_equal_datasets'] = equal
                summary['cases'][label]['baseline_initial_equal_datasets'] = initial_equal
                summary['cases'][label]['baseline_scalar_statistics'] = scalar_checks
            for repeat in range(args.repeats):
                if digest(args.executable) != binary_hash:
                    raise ValueError('executable changed within the timing matrix')
                case, disk = budgeted_launch(args, backend, ranks, f'r{repeat+1}', start+updates,
                    restore=seed, checkpoint_enabled=False, override=True,
                    complete_step_timing=True, nsys_trace=args.nsys, air5_phase_timing=args.phase_timing,
                    chemistry_diagnostics=args.chemistry_diagnostics, **output_budgets)
                check_no_field_output(case)
                log = (case/'run.log').read_text()
                record = window_summary(log, ranks, start, updates, args.warmup)
                if args.phase_timing:
                    record['phase_seconds_per_step'] = phase_summary(log, ranks, start, updates, args.warmup)
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
        if args.campaign_root is not None:
            summary['campaign_directory_bytes'] = campaign_bytes(args.campaign_root)
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
    parser.add_argument('--grid', nargs=3, type=int, default=(16, 16, 16), metavar=('NX', 'NY', 'NZ'),
                        help='Global interval counts; physical domain and dt remain fixed')
    parser.add_argument('--host-budget-bytes', type=int, default=64*1024**2)
    parser.add_argument('--device-budget-bytes', type=int, default=64*1024**2)
    parser.add_argument('--directory-budget-bytes', type=int, default=64*1024**2)
    parser.add_argument('--axis', choices=('x', 'y', 'z'), default='x')
    parser.add_argument('--topology', type=lambda value: tuple(map(int, value.split(','))))
    parser.add_argument('--top-mode', choices=('characteristic', 'prescribed'), default='characteristic')
    parser.add_argument('--filter-workspace', choices=('scalar', 'full'), default='scalar')
    parser.add_argument('--air5-filter-mode', choices=('legacy', 'conservative-flux'), default='legacy')
    parser.add_argument('--flow-control', type=Path,
                        help='Reuse qualified G2 step-0 batches and verify timed restore against their final states')
    parser.add_argument('--campaign-root', type=Path)
    parser.add_argument('--campaign-budget-bytes', type=int)
    parser.add_argument('--report-interval', type=int, default=4)
    parser.add_argument('--chemistry-diagnostics', choices=('every-step', 'report-steps'), default='every-step')
    parser.add_argument('--source-evaluation', choices=('joint', 'source-only'), default='joint')
    parser.add_argument('--chemistry-block-size', type=int, choices=(32, 128), default=128)
    parser.add_argument('--chemistry-tolerance', choices=('baseline', 'loose', 'tight'), default='baseline')
    parser.add_argument('--step-history', choices=('off', 'previous'), default='off')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--start-from-initial', action='store_true',
                        help='Start at exact step-0 checkpoint to match the full-flow error window')
    parser.add_argument('--nsys', action='store_true', help='Separate attribution runs; never a speedup denominator')
    parser.add_argument('--phase-timing', action='store_true', help='Diagnostic host phases; adds no GPU fences')
    parser.add_argument('--runtime-timeout-seconds', type=float, default=180)
    run(parser.parse_args())
