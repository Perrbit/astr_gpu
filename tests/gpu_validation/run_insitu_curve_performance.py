#!/usr/bin/env python3
"""Matched 32^3 CURVE timing after the separate numerical and memory gates.

Each independent process advances four steps and renders at steps 2/4.
The complete window includes first-frame lazy setup. The later output window
is reported separately; nested timings must not be added to form a total.
"""
import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

from run_output_restart_validation import run_case
from test_insitu_curve_derivatives import ROOT
from test_insitu_device_curve_wall_fields import current_arguments
from test_insitu_device_curve_streamlines import configuration, validate
from test_insitu_x4_observation import no_large_io, timing


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def later_output(case, ranks):
    rows = re.findall(r'ASTR_INSITU_STAGE_TIMING insitu_sample_inclusive 4 (\d+)\s+(\S+)',
                      (case/'run.log').read_text())
    values = {int(rank): float(value) for rank, value in rows}
    if len(rows) != ranks or set(values) != set(range(ranks)):
        raise ValueError('Missing or repeated step-4 in-situ timing')
    return max(values.values())


def summarize(rows):
    result = {}
    for label in ('baseline', 'candidate'):
        selected = [row for row in rows if row['binary'] == label and row['round'] >= 0]
        result[label] = {}
        for metric in ('completed_window', 'resident_streamlines_inclusive', 'advance_inclusive'):
            values = [row['timing'][metric]['seconds'] for row in selected]
            result[label][metric] = dict(median=statistics.median(values), minimum=min(values),
                                          maximum=max(values), values=values)
        values = [row['step4_output_seconds'] for row in selected]
        result[label]['step4_output'] = dict(median=statistics.median(values), minimum=min(values),
                                             maximum=max(values), values=values)
    result['median_speedup'] = {metric: result['baseline'][metric]['median']/result['candidate'][metric]['median']
                               for metric in result['baseline']}
    return result


def refresh_report(path):
    """Reparse immutable run logs after a reporting-only correction."""
    report = json.loads(path.read_text())
    for row in report['records']:
        row['step4_output_seconds'] = later_output(Path(row['case']), report['ranks'])
    report['summary'] = summarize(report['records'])
    report['later_output_timer'] = 'insitu_sample_inclusive'
    path.write_text(json.dumps(report, indent=2)+'\n')
    return report['summary']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--mapping', choices=('periodic', 'y-wavy'), required=True)
    parser.add_argument('--ranks', choices=(1, 2), type=int, default=2)
    parser.add_argument('--axis', choices=('x', 'y', 'z'), default='x')
    parser.add_argument('--pipeline', choices=('standard-device', 'direct-device'), default='direct-device')
    parser.add_argument('--transport', choices=('pinned', 'device-aware'), default='device-aware')
    parser.add_argument('--rounds', type=int, default=5)
    args = parser.parse_args()
    if args.rounds < 5:
        parser.error('At least five matched rounds are required')
    binaries = {name: getattr(args, name).resolve(strict=True) for name in ('baseline', 'candidate')}
    # HPC-X launchers use their invoked path to locate the underlying binary.
    args.mpiexec = args.mpiexec.absolute()
    if not args.mpiexec.is_file():
        parser.error(f'MPI launcher not found: {args.mpiexec}')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    report = dict(scope='32^3 four-step windows; statistics and four trace products at steps 2/4; no checkpoint/field I/O',
                  mapping=args.mapping, ranks=args.ranks, axis=args.axis, pipeline=args.pipeline,
                  transport=args.transport, rounds=args.rounds,
                  binaries={label: dict(path=str(path), sha256=digest(path)) for label, path in binaries.items()},
                  pipeline_sha256=digest(ROOT/'scripts/insitu/device_render_pipeline.py'), records=records)
    for round_id in range(-1, args.rounds):
        # The first pair is an excluded system warm-up, not an in-process warm-up.
        order = ('baseline', 'candidate') if round_id % 2 == 0 else ('candidate', 'baseline')
        for label in order:
            run_args = current_arguments(args.output, axis=args.axis, samples=False)
            run_args.executable = binaries[label]
            run_args.mpiexec = args.mpiexec
            run_args.runtime_timeout_seconds = 600
            case, size = run_case(run_args, ROOT, 'gpu', args.ranks, f'{label}_{round_id+1}', 4,
                grid='32,32,32', tgv_mapping=args.mapping, enabled=False, checkpoint_enabled=False,
                insitu_config=configuration(args.pipeline, args.transport),
                postprocess_transport=args.transport, insitu_timing=True)
            validate(case, args.ranks, args.pipeline, audit=False)
            no_large_io(case)
            records.append(dict(binary=label, round=round_id, case=str(case), directory_bytes=size,
                                timing=timing(case, args.ranks), step4_output_seconds=later_output(case, args.ranks)))
            (args.output/'timing.json').write_text(json.dumps(report, indent=2)+'\n')
            print(label, round_id, records[-1]['timing']['completed_window']['seconds'], flush=True)
    if any(digest(path) != report['binaries'][label]['sha256'] for label, path in binaries.items()):
        raise RuntimeError('A measured binary changed during the experiment')
    report['summary'] = summarize(records)
    (args.output/'timing.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
