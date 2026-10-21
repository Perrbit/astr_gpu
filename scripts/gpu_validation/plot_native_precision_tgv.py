"""Matched-step native TGV curves. Edit the two paths below for another campaign."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_tgv_diagnostics import save_figure  # Also applies the shared scientific plotting style.


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT/'tests/gpu_validation/out/native_fp32_tgv_long_20261021'
OUTPUT_DIR = INPUT_DIR/'figures'
COLORS = {'fp64': '#c43c39', 'fp32': '#2864b7'}
LABELS = dict(kenergy=r'$E_k^*$', enstrophy=r'$\Omega_\rho^*$', dissipation=r'$\varepsilon_\mu^*$')


def generate(source, output, schemes=('643e','643c'), report_file=None):
    import json
    report_file = source/'accuracy.json' if report_file is None else report_file
    report = json.loads(report_file.read_text())
    dlr = np.loadtxt(ROOT/'documents/reference_data/tgv/spectral_Re1600_512.gdiag')
    paths = []
    for scheme in schemes:
        curves = {kind: np.genfromtxt(source/f'{scheme}_{kind}_curves.csv', delimiter=',', names=True)
                  for kind in COLORS}
        end_time = max(float(data['nominal_time'][-1]) for data in curves.values())
        reference = dlr[dlr[:, 0] <= end_time]
        for name, ylabel in LABELS.items():
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            for kind, data in curves.items():
                ax.plot(data['nominal_time'], data[name], color=COLORS[kind],
                        linestyle='-' if kind == 'fp64' else '--', label=kind.upper(), linewidth=1.5)
            column = dict(kenergy=1, enstrophy=3, dissipation=2)[name]
            reference_label = dict(dissipation=r'DLR $-dE_k^*/dt^*$',
                                   enstrophy='DLR (unweighted)', kenergy='DLR spectral')[name]
            ax.plot(reference[:, 0], reference[:, column], ':', color='0.35', label=reference_label)
            ax.set_xlabel(r'$t^*=n\Delta t_{\mathrm{input}}$')
            ax.set_ylabel(ylabel)
            ax.legend(frameon=False)
            paths.extend(save_figure(fig, output, f'tgv_{scheme}_{name}'))
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            a,b=curves['fp64'],curves['fp32']
            ax.plot(a['nominal_time'],b[name]-a[name],color=COLORS['fp32'],linewidth=1.5)
            ax.set_xlabel(r'$t^*=n\Delta t_{\mathrm{input}}$')
            ax.set_ylabel(r'$\Delta_{32-64}$ '+ylabel)
            ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
            paths.extend(save_figure(fig, output, f'tgv_{scheme}_{name}_difference'))
        for column, name, ylabel in ((2, 'density', r'$\Delta\langle\rho\rangle/\langle\rho\rangle_0$'),
                                    (3, 'energy', r'$\Delta\langle\rho E\rangle/\langle\rho E\rangle_0$')):
            fig, ax = plt.subplots(figsize=(6.4, 4.2))
            for kind in COLORS:
                values = np.loadtxt(source/f'{scheme}_{kind}/tgv_conservation.dat', skiprows=1, ndmin=2)
                drift = (values[:, column]-values[0, column])/values[0, column]
                ax.plot(values[:, 0]*report['nominal_dt'], drift, color=COLORS[kind],
                        linestyle='-' if kind == 'fp64' else '--', label=kind.upper(), linewidth=1.5)
            ax.set_xlabel(r'$t^*=n\Delta t_{\mathrm{input}}$')
            ax.set_ylabel(ylabel)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax.legend(frameon=False)
            paths.extend(save_figure(fig, output, f'tgv_{scheme}_{name}_drift'))
        fig, ax = plt.subplots(figsize=(6.4,4.2))
        for kind,data in curves.items():
            ax.plot(data['nominal_time'],data['native_time']-data['nominal_time'],color=COLORS[kind],
                    linestyle='-' if kind=='fp64' else '--',label=kind.upper(),linewidth=1.5)
        ax.set_xlabel(r'$t^*=n\Delta t_{\mathrm{input}}$')
        ax.set_ylabel(r'$t^*_{\mathrm{clock}}-n\Delta t_{\mathrm{input}}$')
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        ax.legend(frameon=False)
        paths.extend(save_figure(fig,output,f'tgv_{scheme}_clock_drift'))
    (output/'figure_sources.json').write_text(json.dumps(dict(
        source=str(source), report=str(report_file), schemes=list(schemes),
        coordinate='matched nominal sample time; native clock drift is reported separately',
        dlr='background only, not an identical thermodynamic/density-weighted definition',
        figures=[str(path) for path in paths]), indent=2)+'\n')
    return paths


if __name__ == '__main__':
    generate(INPUT_DIR, OUTPUT_DIR)
