#!/usr/bin/env python3
"""Approved G2 local 64-cubed error matrix; stop before timing on any failure."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from air5_radau_reference import Air5RadauReference
from air5_checkpoint_contract import FILTER_IDENTITY, read_air5_contract
from check_air5_flow_actual_error import compare_checkpoints, require_flow_error
from run_air5_performance_diagnosis import batch_identity, digest
from run_output_air5_restart_validation import DT, ROOT, launch, state_sanity
from run_output_restart_validation import controlled_checkpoint_buffers, compare_fields

FINAL = 'outdat/new/checkpoints/step000000000024'
INITIAL = 'outdat/new/checkpoints/step000000000000'
CONTROLS = (('baseline', 'off'), ('loose', 'off'), ('tight', 'off'),
            ('baseline', 'previous'), ('loose', 'previous'))
CPU_TIMEOUT_SECONDS = 4*3600
CASE_LABELS = ('cpu_np1', 'gpu_np1', 'gpu_np2', 'gpu_np4')


def arguments(args, executable):
    return SimpleNamespace(output=args.output, executable=executable, mpiexec=args.mpiexec,
        grid=(64, 64, 64), case='sbli', reconstruction=3, mean_statistics=False,
        sample_interval=2, mode='steps', axis='x', initial_restart=True,
        top_mode='characteristic', filter_workspace='scalar', report_interval=4,
        chemistry_tolerance='baseline', step_history='off', source_evaluation='joint',
        chemistry_block_size=128, runtime_timeout_seconds=1800,
        air5_filter_mode=getattr(args, 'air5_filter_mode', 'legacy'),
        campaign_root=getattr(args, 'campaign_root', None) or args.output.parent,
        campaign_budget_bytes=getattr(args, 'campaign_budget_bytes', 16*1024**3))


def storage_bytes(root):
    return sum(path.stat().st_size for path in root.rglob('*') if path.is_file())


def run_case(args, backend, ranks, label, chemistry_diagnostics=None, restore=None):
    if storage_bytes(args.campaign_root)+1024**3 > args.campaign_budget_bytes:
        raise RuntimeError('G2 campaign lacks a per-run disk reservation')
    case, disk = launch(args, backend, ranks, label, 24, interval=24,
        host_budget_bytes=256*1024**2, device_budget_bytes=256*1024**2,
        directory_budget_bytes=1024**3, chemistry_diagnostics=chemistry_diagnostics,
        restore=restore, override=restore is not None)
    state = case / FINAL / 'state.h5'
    sanity = state_sanity(state)
    bounds = controlled_checkpoint_buffers(state, 0, backend, budget_bytes=256*1024**2)
    if storage_bytes(args.campaign_root) > args.campaign_budget_bytes:
        raise RuntimeError('G2 campaign disk budget exceeded')
    return dict(case=str(case), state=str(state), sanity=sanity, directory_bytes=disk,
                controlled_host_buffer_bounds=bounds, executable_sha256=digest(args.executable),
                batch_files_sha256=batch_identity(case / FINAL))


def same_initial(actual, reference):
    with h5py.File(actual) as file, h5py.File(reference) as expected:
        for m in range(1, 23):
            name = f'q{m:04}'
            if file[name][:].tobytes() != expected[name][:].tobytes():
                raise ValueError(f'G2 initial q/carry differ: {name}')


def frozen_freestream(model, case):
    row = np.loadtxt(case / 'outdat/new/resources/air5_hbl_profile.dat')[-1]
    species = row[1]*row[8:13]
    ev = model.ev_from_tv(species, row[7])
    return dict(rho=float(row[1]), u=float(row[2]), p=float(row[5]),
                q5=model.q5_from_state(row[1], row[1]*row[2:5], species, ev, row[6]))


def collect_references(args, model):
    report = dict(status='running', scope='fixed outer dt/RK/filter reference pair; not independent CFD',
                  mechanism_sha256=digest(ROOT / 'chemMech/air5_kimjo12.json'),
                  checker_sha256=digest(Path(__file__).with_name('check_air5_flow_actual_error.py')),
                  references={})
    try:
        for level in (11, 12):
            options = arguments(args, getattr(args, f'reference{level}_executable'))
            options.runtime_timeout_seconds = 4*3600
            restore = args.reference11_restore if level == 11 else None
            print(f'CPU reference {level}: 64^3, 24 complete steps', flush=True)
            row = run_case(options, 'cpu', 1, f'reference{level}', restore=restore)
            initial_case = Path(row['case']) if restore is None else restore.parents[3]
            row['initial_state'] = str(initial_case / INITIAL / 'state.h5')
            if restore is not None:
                row.update(continuation_source=str(restore), continuation_source_sha256=batch_identity(restore))
            config = (Path(row['case']) / FINAL / 'air5_config.bin').read_bytes()
            if read_air5_contract(config)['integration_control'] != (level, 0):
                raise ValueError('test reference tolerance identity is missing')
            row.update(level=level, rtol=10.**(-level), atol_factor=10.**(-level-4))
            report['references'][str(level)] = row
        previous, target = (report['references'][str(level)] for level in (11, 12))
        initial = Path(target['initial_state'])
        same_initial(Path(previous['initial_state']), initial)
        freestream = frozen_freestream(model, Path(target['case']))
        margin = compare_checkpoints(model, initial, target['state'], previous['state'], target['state'], freestream)
        report.update(initial=str(initial), initial_batch_sha256=batch_identity(initial.parent),
                      freestream=freestream, margin=margin)
        require_flow_error(margin)
        report['status'] = 'reference-pair-within-approved-margin'
        return report
    except BaseException as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        (args.output / 'reference_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


def load_references(path, filter_mode='legacy'):
    report = json.loads(path.read_text())
    if report['status'] != 'reference-pair-within-approved-margin':
        raise ValueError('cannot reuse an unresolved reference pair')
    if (report['mechanism_sha256'] != digest(ROOT / 'chemMech/air5_kimjo12.json') or
            report['checker_sha256'] != digest(Path(__file__).with_name('check_air5_flow_actual_error.py')) or
            report['initial_batch_sha256'] != batch_identity(Path(report['initial']).parent)):
        raise ValueError('reference mechanism/checker/initial state changed')
    for row in report['references'].values():
        if batch_identity(Path(row['case']) / FINAL) != row['batch_files_sha256']:
            raise ValueError('reference checkpoint batch changed')
        contract = read_air5_contract((Path(row['case']) / FINAL / 'air5_config.bin').read_bytes())
        expected = FILTER_IDENTITY if filter_mode == 'conservative-flux' else (0,)*5
        if contract['filter_identity'] != expected:
            raise ValueError('reference filter numerical identity differs from candidate')
    return report


def engineering_gate(args, report):
    for ranks in (1, 2):
        baseline = run_case(arguments(args, args.executable), 'gpu', ranks, 'engineering_baseline')
        options = arguments(args, args.executable)
        options.source_evaluation, options.chemistry_block_size = 'source-only', 32
        combined = run_case(options, 'gpu', ranks, 'engineering_combined', 'report-steps')
        equal = compare_fields(Path(baseline['state']), Path(combined['state']))
        base_case, combined_case = Path(baseline['case']), Path(combined['case'])
        if (base_case / 'flowstate.dat').read_bytes() != (combined_case / 'flowstate.dat').read_bytes():
            raise ValueError('engineering scalar statistics differ')
        reports = lambda case: [line for line in (case / 'run.log').read_text().splitlines()
                               if line.startswith('AIR5_CHEMISTRY half=')]
        if reports(base_case) != reports(combined_case):
            raise ValueError('engineering report-step values differ')
        report['cases'].append(dict(ranks=ranks, baseline=baseline, combined=combined, exact_datasets=equal))
    report['status'] = 'engineering-exact-field-passed-not-performance'


def run(args):
    args.output = args.output.resolve()
    args.campaign_root = (getattr(args, 'campaign_root', None) or args.output.parent).resolve(strict=True)
    args.output.relative_to(args.campaign_root)
    args.campaign_budget_bytes = getattr(args, 'campaign_budget_bytes', 16*1024**3)
    args.output.mkdir(parents=True, exist_ok=False)
    args.executable = args.executable.resolve(strict=True)
    args.mpiexec = args.mpiexec.resolve(strict=True)
    mechanism = ROOT / 'chemMech/air5_kimjo12.json'
    model = Air5RadauReference(mechanism)
    selected_cases = getattr(args, 'cases', ('cpu_np1', 'gpu_np1', 'gpu_np2'))
    cpu_timeout = getattr(args, 'cpu_timeout_seconds', CPU_TIMEOUT_SECONDS)
    report = dict(status='running', grid_intervals=[64]*3, dt=DT, complete_steps=24,
        coupled_duration=24*DT,
        backend_topologies={label: [int(label.split('_np')[1]), 1, 1] for label in selected_cases}, cases=[],
        mechanism_sha256=digest(mechanism), checker_sha256=digest(Path(__file__).with_name('check_air5_flow_actual_error.py')),
        executable_sha256=digest(args.executable), host_budget_bytes=256*1024**2,
        device_budget_bytes=256*1024**2, directory_budget_bytes=1024**3,
        campaign_budget_bytes=args.campaign_budget_bytes, campaign_root=str(args.campaign_root),
        air5_filter_mode=getattr(args, 'air5_filter_mode', 'legacy'),runtime_timeout_seconds=1800)
    report.update(selected_cases=list(selected_cases), cpu_runtime_timeout_seconds=cpu_timeout)
    report['reference_runtime_timeout_seconds'] = 4*3600
    try:
        if args.engineering_only:
            engineering_gate(args, report)
            return
        references = (load_references(args.references, getattr(args, 'air5_filter_mode', 'legacy'))
                      if args.references else collect_references(args, model))
        report['reference_summary'] = references
        if args.reference_only:
            report['status'] = 'reference-only-passed-not-control-acceptance'
            return
        previous, target = (references['references'][str(level)] for level in (11, 12))
        for tolerance, history in CONTROLS:
            for case_label in selected_cases:
                backend, rank_text = case_label.split('_np')
                ranks = int(rank_text)
                options = arguments(args, args.executable)
                if backend == 'cpu':
                    options.runtime_timeout_seconds = cpu_timeout
                options.chemistry_tolerance, options.step_history = tolerance, history
                label = f'{tolerance}_{history}'
                print(f'{backend} NP{ranks} {label}: 64^3, 24 complete steps', flush=True)
                row = run_case(options, backend, ranks, label)
                same_initial(Path(row['case']) / INITIAL / 'state.h5', references['initial'])
                result = compare_checkpoints(model, references['initial'], target['state'],
                    previous['state'], row['state'], references['freestream'])
                row.update(tolerance=tolerance, history=history, backend=backend, ranks=ranks, error=result)
                report['cases'].append(row)
                require_flow_error(result)
                (Path(row['case']) / 'actual_error.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        complete_matrix = set(selected_cases) == {'cpu_np1', 'gpu_np1', 'gpu_np2'}
        report['status'] = ('local-flow-control-error-passed-not-performance' if complete_matrix else
                            'local-selected-flow-control-error-passed-not-performance')
    except BaseException as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        report['campaign_directory_bytes'] = storage_bytes(args.campaign_root)
        (args.output / 'flow_control.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'executable', 'mpiexec'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--reference11-executable', type=Path)
    parser.add_argument('--reference12-executable', type=Path)
    parser.add_argument('--reference11-restore', type=Path, help='Exact same-configuration completed-step reference prefix')
    parser.add_argument('--references', type=Path, help='Reuse an unchanged, qualified reference_summary.json')
    parser.add_argument('--reference-only', action='store_true')
    parser.add_argument('--air5-filter-mode', choices=('legacy','conservative-flux'), default='legacy')
    parser.add_argument('--campaign-root', type=Path)
    parser.add_argument('--campaign-budget-bytes', type=int, default=16*1024**3)
    parser.add_argument('--engineering-only', action='store_true',
                        help='Independent same-algorithm exact field gate; no new reference or timing')
    parser.add_argument('--cases', nargs='+', choices=CASE_LABELS,
                        default=['cpu_np1', 'gpu_np1', 'gpu_np2'],
                        help='Backend subset for a staged control error matrix; omitted cases remain unvalidated')
    parser.add_argument('--cpu-timeout-seconds', type=float, default=CPU_TIMEOUT_SECONDS,
                        help='Ordinary CPU control runtime budget (approved default: 4 hours); GPU remains 1800 seconds')
    args = parser.parse_args()
    if args.reference11_restore is not None:
        args.reference11_restore = args.reference11_restore.resolve(strict=True)
    if not args.references and not args.engineering_only:
        if args.reference11_executable is None or args.reference12_executable is None:
            parser.error('provide both test reference executables, or --references')
        args.reference11_executable = args.reference11_executable.resolve(strict=True)
        args.reference12_executable = args.reference12_executable.resolve(strict=True)
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    run(args)
