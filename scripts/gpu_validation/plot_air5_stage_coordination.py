#!/usr/bin/env python3
"""Read-only plots of the frozen local AIR5 error/cost and ablation reports."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / 'tests/gpu_validation/out/air5_g2_64_20261013'
OUTPUT_DIR = CAMPAIGN / 'paper_evidence_conservative_20261016_r2'
CONTROLS = ('baseline_off', 'loose_off', 'tight_off', 'baseline_previous', 'loose_previous')
ENGINEERING = ('block32', 'report_only', 'source_only', 'source_report', 'combined', 'combined_previous')
PROTOCOL = ('executable_sha256', 'dt', 'grid_intervals', 'case', 'report_interval',
            'warmup', 'measured_steps', 'repeats', 'primary_metric', 'axis', 'topology',
            'top_mode', 'filter_workspace', 'timing_start_step', 'air5_filter_mode')
COLORS = ('#b2182b', '#2166ac')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_protocol(report, baseline, flow):
    if report['status'] != 'bounded-timing-collected' or report['profiled'] or report['phase_timing']:
        raise ValueError('require successful unprofiled timing, not attribution')
    for key in PROTOCOL:
        if report[key] != baseline[key]:
            raise ValueError(f'timing protocol mismatch: {key}')
    for key in ('executable_sha256', 'dt', 'grid_intervals', 'air5_filter_mode'):
        if report[key] != flow[key]:
            raise ValueError(f'error/cost identity mismatch: {key}')
    if (flow['status'] != 'local-flow-control-error-passed-not-performance' or
            flow['complete_steps'] != report['warmup'] + report['measured_steps']):
        raise ValueError('require the same qualified full-flow error window')


def timing_stats(case, repeats):
    values = [row['seconds_per_step'] for row in case['rounds']]
    if len(values) != repeats or not all(math.isfinite(x) and x > 0 for x in values):
        raise ValueError('invalid timing samples')
    result = dict(median_seconds=statistics.median(values), min_seconds=min(values),
                  max_seconds=max(values), stdev_seconds=statistics.stdev(values))
    if any(case[key] != value for key, value in result.items()):
        raise ValueError('stored timing statistics do not match all samples')
    return result


def max_error(error):
    return max(value for representation in error['candidate'].values()
               for value in representation['max_normalized_error'])


def collect(campaign):
    paths = {name: campaign / 'control_timing_conservative_20261016_r1' / name / 'timing_summary.json'
             for name in CONTROLS}
    paths.update({name: campaign / 'engineering_timing_conservative_20261016_r1' / name / 'timing_summary.json'
                  for name in ENGINEERING})
    flow_path = campaign / 'controls_conservative_20261016_r1/flow_control.json'
    flow = json.loads(flow_path.read_text())
    reports = {name: json.loads(path.read_text()) for name, path in paths.items()}
    baseline = reports['baseline_off']
    errors = {(row['backend'], row['ranks'], row['tolerance'], row['history']): row for row in flow['cases']}
    costs = []
    for name, report in reports.items():
        validate_protocol(report, baseline, flow)
        for ranks in (1, 2):
            label = f'gpu_np{ranks}'
            case = report['cases'][label]
            control = errors['gpu', ranks, report['chemistry_tolerance'], report['step_history']]
            error = control['error']
            if not error['within_reference_margin'] or not error['within_candidate_error']:
                raise ValueError('failed actual-error gate')
            if name in ENGINEERING:
                expected = 39 if report['step_history'] == 'previous' else 38
                if (len(case['baseline_equal_datasets']) != expected or
                        len(case['baseline_initial_equal_datasets']) != expected or
                        case['baseline_scalar_statistics'] != {'verification': 'equal'}):
                    raise ValueError('engineering case lacks baseline equivalence evidence')
            stats = timing_stats(case, report['repeats'])
            base_seconds = baseline['cases'][label]['median_seconds']
            costs.append(dict(configuration=name, ranks=ranks, **stats,
                seconds_relative_to_baseline_off=stats['median_seconds']/base_seconds,
                time_reduction_vs_baseline_off_percent=100*(1-stats['median_seconds']/base_seconds),
                max_budget_normalized_error=max_error(error), error_source_case=control['case'],
                error_basis='baseline-equivalent' if name in ENGINEERING else 'direct-G2-report',
                timing_report=str(paths[name])))
    components = []
    for row in flow['cases']:
        for representation, data in row['error']['candidate'].items():
            for i, name in enumerate(row['error']['components']):
                components.append(dict(backend=row['backend'], ranks=row['ranks'],
                    tolerance=row['tolerance'], history=row['history'], representation=representation,
                    component=name, **{key: data[key][i] for key in ('max_absolute_error',
                        'max_normalized_error', 'max_relative_error', 'undefined_relative_nodes')}))
    sources = {str(path): digest(path) for path in (*paths.values(), flow_path)}
    return reports, flow, costs, components, sources


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_ranges(plt, reports, names, labels, output, normalized=False):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for ranks, offset, color in zip((1, 2), (-0.12, 0.12), COLORS):
        label = f'gpu_np{ranks}'
        denominator = reports['baseline_off']['cases'][label]['median_seconds'] if normalized else 1.
        for i, name in enumerate(names):
            case = reports[name]['cases'][label]
            mid, lo, hi = [case[key]/denominator for key in ('median_seconds', 'min_seconds', 'max_seconds')]
            ax.errorbar(mid, i+offset, xerr=[[mid-lo], [hi-mid]], fmt='o' if ranks == 1 else 's',
                        color=color, capsize=4, label=f'{ranks} GPU' + ('s' if ranks > 1 else '') if i == 0 else None)
            ax.plot([r['seconds_per_step']/denominator for r in case['rounds']],
                    [i+offset]*len(case['rounds']), '.', color=color, markersize=3)
    if normalized:
        ax.axvline(1., color='0.4', linestyle='--', linewidth=1)
    ax.set_yticks(range(len(names)), labels)
    ax.invert_yaxis()
    ax.set_xlabel('Complete-step time / baseline median' if normalized else 'Complete-step time (s)')
    ax.legend(frameon=False, loc='center right' if normalized else 'lower right')
    fig.tight_layout()
    save_figure(plt, fig, output)


def save_figure(plt, fig, stem):
    for extension in ('eps', 'jpeg'):
        fig.savefig(OUTPUT_DIR / f'{stem}.{extension}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_errors(plt, flow):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    selected = [row for row in flow['cases'] if row['backend'] == 'gpu' and row['tolerance'] == 'baseline']
    for row in selected:
        data = row['error']['candidate']
        values = [max(a, b) for a, b in zip(data['q']['max_normalized_error'],
                    data['q-minus-carry']['max_normalized_error'], strict=True)]
        ax.semilogy(range(len(values)), values, color=COLORS[row['ranks']-1],
            linestyle='--' if row['history'] == 'previous' else '-',
            marker='s' if row['history'] == 'previous' else 'o',
            label=f'{row["ranks"]} GPU' + ('s' if row['ranks'] > 1 else '') +
                  (', history' if row['history'] == 'previous' else ', no history'))
    labels = (r'$Y_{N_2}$', r'$Y_{O_2}$', r'$Y_N$', r'$Y_O$', r'$Y_{NO}$', r'$e_v$',
              r'$T$', r'$T_v$', r'$\rho$', r'$u$', r'$v$', r'$w$', r'$p$', r'$\rho E$')
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel('Maximum budget-normalized error')
    ax.legend(frameon=False, ncol=2, loc='upper right')
    fig.tight_layout()
    save_figure(plt, fig, 'flow_error_components')


def main():
    reports, flow, costs, components, sources = collect(CAMPAIGN)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401
    plt.style.use(['science', 'ieee', 'std-colors'])
    plt.rcParams['axes.grid'] = False
    plt.rcParams['grid.alpha'] = 0.0
    plt.rcParams.update({'axes.labelsize': 16, 'xtick.labelsize': 14,
                        'ytick.labelsize': 14, 'legend.fontsize': 14})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    write_csv(OUTPUT_DIR / 'complete_step_cost.csv', costs)
    write_csv(OUTPUT_DIR / 'flow_error_components.csv', components)
    plot_ranges(plt, reports,
        ('baseline_off', 'report_only', 'source_only', 'block32', 'source_report', 'combined', 'combined_previous'),
        ('Baseline', 'Report diagnostics', 'Source-only evaluation', '32 threads',
         'Source + report', 'Combined', 'Combined + history'), 'engineering_cost')
    plot_ranges(plt, reports, CONTROLS,
        ('Baseline, no history', 'Loose, no history', 'Tight, no history',
         'Baseline, history', 'Loose, history'), 'control_cost', normalized=True)
    plot_errors(plt, flow)
    manifest = dict(scope='local G2 summaries only; not A800 or physical SBLI acceptance',
        timing_metric=reports['baseline_off']['primary_metric'],
        numerical_error='maximum over nodes and q/q-minus-carry of abs(error)/(A_j+R_j*abs(reference)); threshold 1',
        timing_range='all five samples, not confidence intervals; no samples removed',
        relative_time_denominator='baseline_off median at the same GPU count',
        sources_sha256=sources, script_sha256=digest(Path(__file__)),
        protocol={key: reports['baseline_off'][key] for key in PROTOCOL},
        output_files_sha256={path.name: digest(path) for path in sorted(OUTPUT_DIR.iterdir())})
    (OUTPUT_DIR / 'sources.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    print(OUTPUT_DIR)


if __name__ == '__main__':
    main()
