"""Plot stored checkpoint spectra; paths are explicitly configured below."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_tgv_diagnostics import save_figure

ROOT=Path(__file__).resolve().parents[2]
INPUT_DIR=ROOT/'tests/gpu_validation/out/native_fp32_tgv_spectra_20261028'
OUTPUT_DIR=INPUT_DIR/'figures'
CASES=('643e_fp64','643e_fp32','643c_fp64','643c_fp32')
STEPS=(35000,40000)
COLORS={'fp64':'#B8433B','fp32':'#135F96'}
LABELS={case:('Explicit' if case.startswith('643e') else 'Compact')+' '+case[-4:].upper() for case in CASES}


def plot_all():
    paths=[]
    for step in STEPS:
        curves={case:dict(np.load(INPUT_DIR/f'{case}_step{step}.npz')) for case in CASES}
        for field,ylabel,name in [('energy',r'$E_u(k)$','velocity_energy'),
                                  ('weighted_energy',r'$E_\rho(k)$','density_weighted_energy')]:
            fig,ax=plt.subplots(figsize=(7.4,4.6))
            for case in CASES:
                k=curves[case]['k'];energy=curves[case][field]
                selected=(k>=2)&(k<=128)&(energy>0)
                ax.loglog(k[selected],energy[selected],color=COLORS[case[-4:]],
                          ls='-' if case.startswith('643e') else '--',lw=1.5,label=LABELS[case])
            ax.set(xlabel=r'$kL_{\mathrm{ref}}$',ylabel=ylabel,xlim=(2,128))
            ax.legend(frameon=False)
            paths.extend(save_figure(fig,OUTPUT_DIR,f'{name}_step{step}'))
        for axis,axis_name in enumerate('xyz'):
            fig,axes=plt.subplots(1,3,figsize=(13.5,3.6),sharex=True,sharey=True)
            for component,ax in enumerate(axes):
                for case in CASES:
                    r=np.arange(129)/256
                    values=curves[case]['normalized_correlation'][component,axis,:129]
                    ax.plot(r,values,color=COLORS[case[-4:]],
                            ls='-' if case.startswith('643e') else '--',lw=1.5,label=LABELS[case])
                ax.axhline(0,color='.6',lw=.6)
                ax.set(xlabel=rf'$r_{axis_name}/L_{{\mathrm{{box}}}}$',
                       ylabel=rf'$C_{{{"uvw"[component]}{"uvw"[component]}}}^{{({axis_name})}}$',xlim=(0,.5),ylim=(-1.02,1.02))
            axes[1].legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.31),ncol=4)
            fig.subplots_adjust(wspace=.35)
            paths.extend(save_figure(fig,OUTPUT_DIR,f'correlation_axis_{axis_name}_step{step}'))
        fig,axes=plt.subplots(1,2,figsize=(12,4.4))
        for scheme in ('643e','643c'):
            values=np.load(INPUT_DIR/f'{scheme}_fp32_minus_{scheme}_fp64_step{step}.npz')
            k=np.arange(len(values['error_energy']))
            selection=(k>=1)&(k<=128)
            label='Explicit FP32-FP64' if scheme=='643e' else 'Compact FP32-FP64'
            color=COLORS['fp32' if scheme=='643e' else 'fp64']
            axes[0].loglog(k[selection],values['error_energy'][selection]/values['reference_energy'].sum(),color=color,label=label,lw=1.5)
            resolved=(k>=2)&(k<=128)
            axes[1].semilogx(k[resolved],values['spectral_alignment'][resolved],color=color,label=label,lw=1.5)
        axes[0].set(xlabel=r'$kL_{\mathrm{ref}}$',ylabel=r'$E_{\Delta u}(k)/K_{u,\mathrm{ref}}$',xlim=(1,128))
        axes[1].set(xlabel=r'$kL_{\mathrm{ref}}$',ylabel=r'$\gamma_u(k)$',xlim=(2,128),ylim=(-1.05,1.05))
        axes[0].legend(frameon=False,fontsize=11)
        fig.subplots_adjust(wspace=.32)
        paths.extend(save_figure(fig,OUTPUT_DIR,f'precision_error_and_alignment_step{step}'))
    (OUTPUT_DIR/'figure_sources.json').write_text(json.dumps(dict(
        source=str(INPUT_DIR),steps=STEPS,cases=CASES,analysis_report=str(INPUT_DIR/'report.json'),
        spectrum_display='energy/alignment figures show integer shell centers k=2..128; near-zero k=1 and incomplete corner shells k>128 retained in CSV/report; error-energy figure also includes k=1',
        coordinate='same completed step, not time-interpolated fields; nominal t*=8.75 and 10',
        correlations='instantaneous volume average, all component/separation-axis combinations retained',
        figures=[str(p) for p in paths]),indent=2)+'\n')
    print(f'{len(paths)//2} figures, each JPEG + EPS: {OUTPUT_DIR}')


if __name__=='__main__':
    plot_all()
