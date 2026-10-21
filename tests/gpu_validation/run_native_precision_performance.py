"""FP8 local interleaved native FP32/FP64 GPU matrix using the existing advancer."""

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import subprocess
import sys
import sqlite3

from run_compact_performance import CASES
from summarize_compact_profile import summarize as summarize_profile


ROOT = Path(__file__).resolve().parents[2]


def configurations(cases):
    return list(itertools.product(cases, ('643e', '643c'), ('f', 't'), (1, 2)))


def ordered_runs(cases, repeat):
    items = configurations(cases)
    items = items[repeat:] + items[:repeat]
    if repeat % 2:
        items.reverse()
    kinds = ('fp64', 'fp32') if repeat % 2 == 0 else ('fp32', 'fp64')
    return [(*item, kind) for item in items for kind in kinds]


def label(case, scheme, filtering, ranks, precision):
    return f'{case}_{scheme}_filter{filtering}_np{ranks}_{precision}'


def summarize(root, cases):
    rows = []
    for case, scheme, filtering, ranks in configurations(cases):
        pair = {}
        for precision in ('fp64', 'fp32'):
            name = label(case, scheme, filtering, ranks, precision)
            records = [json.loads((root / 'timing' / f'{name}_r{repeat+1}' / 'summary.json').read_text())
                       for repeat in range(3)]
            if any(record['numerical_precision'] != precision or record['profiled'] or record['phase_timing'] or
                   record['preflight'] or record['steps'] != 40 or record['warmup'] != 10 or
                   record['statistics_interval'] != 0 or not record['buffered_output'] for record in records):
                raise ValueError('timing matrix does not use the frozen FP8 measurement window')
            if any((r['case'], r['scheme'], r['filter'], r['topology'], tuple(r['grid_intervals'])) !=
                   (case, scheme, filtering, f'{ranks},1,1', tuple(map(int, CASES[case][0].split(','))))
                   for r in records):
                raise ValueError('timing record belongs to a different case/configuration')
            identities = {(r['executable_sha256'], r['workspace'], r['compact_sync'], r['transport'],
                           r['batch_rhs'], r['deltat'], r['crashfix']) for r in records}
            if len(identities) != 1:
                raise ValueError('precision repetitions differ in executable or runtime settings')
            pair[precision] = records
        for key in ('workspace', 'compact_sync', 'transport', 'batch_rhs', 'deltat', 'crashfix'):
            if pair['fp32'][0][key] != pair['fp64'][0][key]:
                raise ValueError('precision pair differs in numerical/runtime configuration')
        samples = {kind: [r['rk_seconds_per_step'] for r in records] for kind, records in pair.items()}
        row = dict(case=case, scheme=scheme, filter=filtering, np=ranks, crashfix=pair['fp64'][0]['crashfix'],
                   repair_scope='full 40 steps including warmup; pointwise maxima, not integrated conservation defects')
        for kind, values in samples.items():
            row.update({kind + '_median_ms': statistics.median(values)*1000,
                        kind + '_min_ms': min(values)*1000, kind + '_max_ms': max(values)*1000,
                        kind + '_samples_ms': [value*1000 for value in values]})
            for source, target in (('crashfix_repaired_node_visits', 'repair_visits'),
                                   ('crashfix_nonzero_calls', 'repair_calls')):
                counts = [record[source] for record in pair[kind]]
                row[kind+'_'+target+'_samples'] = counts
                row[kind+'_'+target+'_min'] = min(counts)
                row[kind+'_'+target+'_max'] = max(counts)
            row[kind+'_repair_max_pointwise_change'] = [max(values) for values in zip(*(
                record['crashfix_max_pointwise_conservative_change'] for record in pair[kind]))]
            preflight = root/'preflight'/f'{label(case, scheme, filtering, ranks, kind)}_r1/summary.json'
            if preflight.is_file():
                observation = json.loads(preflight.read_text())
                row[kind+'_sampled_resources'] = observation['sampled_preflight_resources']
                row[kind+'_packed_device_bytes'] = observation['packed_device_bytes']
                resources = observation['sampled_preflight_resources']
                row[kind+'_largest_sampled_device_peak_gib'] = max(
                    device['peak_bytes'] for device in resources['devices'].values())/1024**3
                row[kind+'_sampled_process_tree_rss_peak_gib'] = resources['host_rss_peak_bytes']/1024**3
        row['fp64_over_fp32'] = row['fp64_median_ms'] / row['fp32_median_ms']
        rows.append(row)
    by_key = {(row['case'], row['scheme'], row['filter'], row['np']): row for row in rows}
    for row in rows:
        for kind in ('fp64', 'fp32'):
            single = by_key[row['case'], row['scheme'], row['filter'], 1][kind + '_median_ms']
            row[kind + '_speedup_np'] = single / row[kind + '_median_ms']
            row[kind + '_efficiency_np'] = row[kind + '_speedup_np'] / row['np']
            explicit = by_key[row['case'], '643e', row['filter'], row['np']][kind+'_median_ms']
            row[kind+'_compact_over_explicit'] = row[kind+'_median_ms']/explicit
    (root / 'performance.json').write_text(json.dumps(rows, indent=2) + '\n')
    with (root / 'performance.csv').open('w') as stream:
        fields = [name for name in rows[0] if not isinstance(rows[0][name], (list, dict))]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({name: row[name] for name in fields} for row in rows)
    return rows


def communication_report(directory):
    ranks = []
    for path in sorted(directory.glob('*.sqlite')):
        record = summarize_profile(path)
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True) as db:
            record['mpi_wait_start_table_present'] = bool(db.execute(
                "select 1 from sqlite_master where name='MPI_START_WAIT_EVENTS'").fetchone())
            record['mpi_wait_start'] = [dict(api=api, calls=calls, seconds=seconds)
                for api, calls, seconds in db.execute(
                    'select s.value,count(*),sum(a.end-a.start)*1e-9 from MPI_START_WAIT_EVENTS a '
                    'join StringIds s on a.textId=s.id group by a.textId')] if record['mpi_wait_start_table_present'] else []
            record['point_to_point_by_tag'] = [dict(api=api, tag=tag, calls=calls,
                recorded_message_bytes=volume) for api, tag, calls, volume in db.execute(
                    'select s.value,a.tag,count(*),sum(a.size) from MPI_P2P_EVENTS a '
                    'join StringIds s on a.textId=s.id group by a.textId,a.tag order by a.tag,s.value')]
        ranks.append(record)
    if len(ranks) != 2 or any(not record['kernels'] for record in ranks):
        raise ValueError('Independent NP2 communication observation is incomplete')
    result = dict(scope='whole profiled process, including initialization; not formal timing',
        interpretation='MPI size is the recorded API message/buffer size, not measured wire traffic; '
                       'send and receive events are not added as unique network bytes; '
                       'API, CUDA and MPI times overlap and must not be summed as wall time',
        ranks=ranks)
    (directory/'communication.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for case in CASES:
        for precision in ('fp32', 'fp64'):
            parser.add_argument(f'--{case}-{precision}', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--mode', choices=('preflight', 'timing', 'profile', 'summarize'), required=True)
    parser.add_argument('--resume', action='store_true', help='reuse completed matching timing records')
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.mode == 'summarize':
        summarize(root, args.cases)
        return
    for repeat in range(3 if args.mode == 'timing' else 1):
        for case, scheme, filtering, ranks, precision in ordered_runs(args.cases, repeat):
            if args.mode == 'profile' and (filtering != 't' or ranks != 2):
                continue
            name = label(case, scheme, filtering, ranks, precision)
            executable = getattr(args, case + '_' + precision).resolve(strict=True)
            if args.mode in ('timing', 'profile'):
                prior = json.loads((root / 'preflight' / f'{name}_r1/summary.json').read_text())
                with executable.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                if prior['status'] != 'pass' or prior['executable_sha256'] != digest or prior['numerical_precision'] != precision:
                    raise ValueError('FP8 timing requires the matching executable preflight')
            destination = root / args.mode / f'{name}_r{repeat+1}'
            if (args.mode == 'profile' or (args.mode == 'timing' and args.resume)) and (destination/'summary.json').is_file():
                recorded = json.loads((destination/'summary.json').read_text())
                expected_steps = 40 if args.mode == 'timing' else 3
                if (recorded['status'] != 'pass' or recorded['profiled'] != (args.mode == 'profile') or
                    recorded['executable_sha256'] != digest or recorded['numerical_precision'] != precision or
                    (recorded['case'], recorded['scheme'], recorded['filter'], recorded['topology'], recorded['steps']) !=
                    (case, scheme, filtering, f'{ranks},1,1', expected_steps) or
                    recorded['warmup'] != (10 if args.mode == 'timing' else 0) or
                    recorded['statistics_interval'] != 0 or not recorded['buffered_output'] or
                    any(recorded[key] != prior[key] for key in ('workspace', 'compact_sync', 'transport',
                        'batch_rhs', 'deltat', 'crashfix', 'grid_intervals'))):
                    raise ValueError('Existing observation differs from the frozen configuration')
                if args.mode == 'profile':
                    communication_report(destination)
                print('REUSE recorded '+args.mode+' '+name, flush=True)
                continue
            grid, dt = CASES[case]
            command = [sys.executable, str(ROOT / 'tests/gpu_validation/run_compact_execution.py'),
                '--exe', str(executable), '--out', str(destination),
                '--case', case, '--grid', grid, '--deltat', str(dt), '--scheme', scheme, '--filter', filtering,
                '--native-precision', precision, '--topology', f'{ranks},1,1',
                '--steps', '40' if args.mode == 'timing' else '3',
                '--warmup', '10' if args.mode == 'timing' else '0',
                '--statistics-interval', '0', '--buffer-output',
                '--compact-sync', 'explicit', '--transport', 'pinned', '--workspace', 'scalar']
            if case == 'plate':
                command.append('--crashfix')
            if args.mode == 'preflight':
                command.append('--preflight')
            if args.mode == 'profile':
                command.append('--nsys')
            print('START ' + name + f' round {repeat+1}', flush=True)
            subprocess.run(command, check=True)
            if args.mode == 'profile':
                communication_report(destination)
    if args.mode == 'timing':
        summarize(root, args.cases)


if __name__ == '__main__':
    main()
