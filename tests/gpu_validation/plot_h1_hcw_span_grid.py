"""Plot raw physical-time histories and final spanwise modal RMS, not convergence."""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from compare_flowstate import read_flowstate

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'tests/gpu_validation/out/h1_span_grid_20261105/window_after_archive_fix'


def main():
    plt.style.use(['science','ieee','std-colors'])
    plt.rcParams['axes.grid']=False
    plt.rcParams['grid.alpha']=0.
    plt.rcParams.update({'axes.labelsize':16,'xtick.labelsize':14,
                         'ytick.labelsize':14,'legend.fontsize':14})
    report=json.loads((OUTPUT/'span_grid.json').read_text())
    assert len(report['cases'])==2
    cases=[report['baseline'],*report['cases']]
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    spectrum,modal=plt.subplots(figsize=(8,5),layout='constrained')
    rows=[]
    for case,color,style in zip(cases,('#222222','#b72d35','#2464a0'),('-','--','-.')):
        names,data=read_flowstate(Path(case['case']))
        t=data[:,names.index('time')];w=data[:,names.index('wrms')]
        label=rf"$L_z={case['span_m']*1000:g}\,\mathrm{{mm}},\ N_z={case['z_intervals']}$"
        positive=w>0
        ax.semilogy(t[positive],w[positive],color=color,linestyle=style,label=label)
        modes=np.asarray(case['final']['w_fourier_mode_rms'])[1:]
        wavelength=1000*case['span_m']/np.arange(1,len(modes)+1)
        modal.loglog(wavelength,modes,color=color,linestyle=style,marker='o',markersize=4,label=label)
        rows.append(dict(span_mm=case['span_m']*1000,z_intervals=case['z_intervals'],
            dz_mm=case['dz_m']*1000,steps=case['steps'],final_time=case['final_time'],
            rho_min=case['rho_min'],temperature_min=case['temperature_min'],
            statistics_time=float(t[-1]),native_w_rms=float(w[-1]),
            density_spanwise_deviation_rms=case['final']['spanwise_deviation_rms']['density'],
            w_spanwise_deviation_rms=case['final']['spanwise_deviation_rms']['w']))
    ax.set_xlabel(r'$tU_\infty/L_{\mathrm{ref}}$')
    ax.set_ylabel(r'$w_{\mathrm{rms}}/U_\infty$')
    ax.legend(frameon=False)
    modal.set_xlabel(r'$\lambda_z\ (\mathrm{mm})$')
    modal.set_ylabel(r'$A_{w,m}/U_\infty$')
    modal.legend(frameon=False)
    for figure,name in ((fig,'spanwise_velocity_history'),(spectrum,'final_w_modal_rms')):
        for extension in ('jpeg','eps'):
            figure.savefig(OUTPUT/f'{name}.{extension}',dpi=300,bbox_inches='tight')
        plt.close(figure)
    with (OUTPUT/'span_grid_summary.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=='__main__':main()
