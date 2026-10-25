"""Offline FP8 timing, capacity and short-profile plots; edit paths below."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_native_precision_tgv import COLORS
from plot_tgv_diagnostics import save_figure


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/'tests/gpu_validation/out'
PERFORMANCE = EVIDENCE/'native_fp32_fp8_medium_20261021'
LONG_TGV = EVIDENCE/'native_fp32_tgv_long_20261021'
OUTPUT = EVIDENCE/'native_fp32_postprocessing_20261022'
GROUPS = [('643e', 1), ('643e', 2), ('643c', 1), ('643c', 2)]
GROUP_LABELS = ['Explicit\n1 GPU', 'Explicit\n2 GPU', 'Compact\n1 GPU', 'Compact\n2 GPU']
PROFILE_Y_MAX = 15.


def grouped_plot(rows, key, ylabel, output, stem, *, samples=False):
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    positions = np.arange(len(rows))
    upper = 0.
    for kind, offset in (('fp64', -.19), ('fp32', .19)):
        values = np.asarray([row[kind+'_'+key] for row in rows])
        ax.bar(positions+offset, values, width=.36, color=COLORS[kind], label=kind.upper())
        upper = max(upper, float(values.max()))
        if samples:
            low = np.asarray([row[kind+'_min_ms'] for row in rows])
            high = np.asarray([row[kind+'_max_ms'] for row in rows])
            ax.errorbar(positions+offset, values, yerr=[values-low, high-values],
                        fmt='none', color='0.15', capsize=3, linewidth=1)
            for index, row in enumerate(rows):
                ax.plot(positions[index]+offset+np.array([-.035, 0., .035]),
                        row[kind+'_samples_ms'], '.', color='0.15', markersize=4)
            upper = max(upper, float(high.max()))
    ax.set_xticks(positions, GROUP_LABELS)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, upper*1.27)
    ax.legend(frameon=False, ncol=2, loc='upper left')
    return save_figure(fig, output, stem)


def scaling_plot(rows, output, case):
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for kind in COLORS:
        for scheme, style, marker, label in (('643e', '--', 'o', 'explicit'),
                                             ('643c', '-', 's', 'compact')):
            values = [next(row[kind+'_speedup_np'] for row in rows
                           if row['scheme']==scheme and row['np']==ranks and row['filter']=='t')
                      for ranks in (1, 2)]
            ax.plot([1, 2], values, color=COLORS[kind], linestyle=style, marker=marker,
                    linewidth=1.5, label=kind.upper()+' '+label)
    ax.plot([1, 2], [1, 2], ':', color='0.45', label='Ideal')
    ax.set_xticks([1, 2])
    ax.set_xlim(.9, 2.1)
    ax.set_ylim(.75, 2.65)
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel(r'$T_{1\,\mathrm{GPU}}/T_{N\,\mathrm{GPU}}$')
    ax.legend(frameon=False, ncol=2, loc='upper left')
    return save_figure(fig, output, f'{case}_filter_on_scaling')


def profile_plots(directory, output, scheme):
    report = json.loads((directory/'report.json').read_text())
    tag = '2x1x1_filter_1'
    paths = []
    for field in ('u', 'T'):
        fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.3), sharey=True)
        for ax, profile in zip(axes, report['cases'][tag]['profiles']):
            data = np.genfromtxt(directory/tag/profile['file'], delimiter=',', names=True)
            for kind in COLORS:
                ax.plot(data[field+'_'+kind], data['y_'+kind], color=COLORS[kind],
                        linestyle='-' if kind=='fp64' else '--', label=kind.upper(), linewidth=1.5)
            ax.set_xlabel('$'+field+'$')
            ax.set_ylim(0, PROFILE_Y_MAX)
            ax.text(.08, .86, r'$x='+format(profile['x_fp64'], '.6g')+'$', transform=ax.transAxes)
        axes[0].set_ylabel('$y$')
        axes[-1].legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, 1.2), ncol=2)
        fig.subplots_adjust(wspace=.18)
        paths.extend(save_figure(fig, output, f'm12_{scheme}_{field}_short_profiles'))
    return paths


def precision_summary(source):
    report = json.loads((source/'accuracy.json').read_text())
    result = dict(source=str(source/'accuracy.json'), comparison='same step / nominal time',
                  acceptance='report only, no cross-precision error ceiling', pairs={})
    for scheme in ('643e', '643c'):
        curves = {kind: np.genfromtxt(source/f'{scheme}_{kind}_curves.csv', delimiter=',', names=True)
                  for kind in COLORS}
        reference, candidate = curves['fp64'], curves['fp32']
        if not np.array_equal(reference['step'], candidate['step']):
            raise ValueError('Cannot compare curves with different sample identities')
        pair = {}
        for name in ('kenergy', 'enstrophy', 'dissipation'):
            a, b = reference[name], candidate[name]
            # All three TGV diagnostics are strictly positive in this observation window.
            if np.any(a<=0):
                raise ValueError('Pointwise relative differences require positive reference diagnostics')
            relative = np.abs((b-a)/a)
            index = int(relative.argmax())
            pair[name] = dict(report['pairs'][scheme][name],
                max_pointwise_relative_percent=float(100*relative[index]),
                max_pointwise_relative_step=int(reference['step'][index]),
                peak_relative_percent=float(100*(b.max()-a.max())/a.max()),
                final_fp64=float(a[-1]), final_fp32=float(b[-1]))
        result['pairs'][scheme] = pair
    return result


def generate(performance, long_tgv, output):
    rows = json.loads((performance/'performance.json').read_text())
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for case in ('tgv', 'plate'):
        case_rows = [row for row in rows if row['case']==case]
        for filtering in ('f', 't'):
            selected = [next(row for row in case_rows if (row['scheme'], row['np'])==group
                             and row['filter']==filtering) for group in GROUPS]
            paths.extend(grouped_plot(selected, 'median_ms', r'$T_{\mathrm{RK}}$ [ms/step]', output,
                                      f'{case}_filter_{filtering}_rk_time', samples=True))
            if filtering=='t':
                paths.extend(grouped_plot(selected, 'largest_sampled_device_peak_gib',
                    'Sampled GPU memory [GiB]', output, f'{case}_filter_on_gpu_memory'))
        paths.extend(scaling_plot(case_rows, output, case))
    for scheme, label in (('643e', 'explicit'), ('643c', 'compact')):
        paths.extend(profile_plots(EVIDENCE/f'native_fp32_fp8_fields_m12_{label}_20261021', output, scheme))
    manifest = dict(performance_source=str(performance/'performance.json'),
        performance_definition='three-run median of max-rank cumulative RK window / 30 measured steps; '
                               '10 warmup steps excluded; no profiling/output/statistics in timing',
        error_bars='min/max over all three raw measurements; dots retain all measurements',
        capacity_definition='independent 20 ms observations; largest per-device process-memory peak, not sum',
        short_profiles='step9/RK3 pre_rhs; instantaneous spanwise means, not time averages; '
                       'source nondimensionalization and each precision geometry retained; '
                       f'plots show 0 <= y <= {PROFILE_Y_MAX}, CSV profiles retain the full domain',
        renderer='Science/IEEE style with MathText; no external TeX font dependency',
        figures=[str(path) for path in paths], long_tgv=precision_summary(long_tgv))
    (output/'postprocessing_summary.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return paths


if __name__=='__main__':
    generate(PERFORMANCE, LONG_TGV, OUTPUT)
