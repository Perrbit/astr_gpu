"""Raw w RMS histories for grid/dt attribution; no modal-growth fit."""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import scienceplots

from compare_flowstate import read_flowstate

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'tests/gpu_validation/out/h1_span_grid_20261105/common_fine_dt'


def main():
    plt.style.use(['science','ieee','std-colors'])
    plt.rcParams['axes.grid']=False
    plt.rcParams['grid.alpha']=0.
    plt.rcParams.update({'axes.labelsize':16,'xtick.labelsize':14,
                         'ytick.labelsize':14,'legend.fontsize':14})
    report=json.loads((OUTPUT/'span_timestep.json').read_text())
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    rows=[]
    options=(('coarse_auto',r'$N_z=5,\ \Delta t_5(t)$','#222222','-'),
             ('fine_auto',r'$N_z=16,\ \Delta t_{16}(t)$','#b72d35','--'),
             ('coarse_replay',r'$N_z=5,\ \Delta t_{16}(t)$','#2464a0','-.'))
    for key,label,color,style in options:
        case=report[key];names,data=read_flowstate(Path(case['case']))
        time=data[:,names.index('time')];w=data[:,names.index('wrms')]
        positive=w>0
        ax.semilogy(time[positive],w[positive],label=label,color=color,linestyle=style)
        rows.append(dict(configuration=key,z_intervals=case['z_intervals'],
            steps=case['steps'],final_time=case['final_time'],
            cfl_upper_min=case['observed_cfl_upper_range'][0],
            cfl_upper_max=case['observed_cfl_upper_range'][1],
            weighted_w_spanwise_deviation_rms=case['final']['spanwise_deviation_rms']['w'],
            weighted_density_spanwise_deviation_rms=case['final']['spanwise_deviation_rms']['density']))
    ax.set_xlabel(r'$tU_\infty/L_{\mathrm{ref}}$')
    ax.set_ylabel(r'$w_{\mathrm{rms}}/U_\infty$')
    ax.legend(frameon=False)
    for extension in ('jpeg','eps'):
        fig.savefig(OUTPUT/f'span_grid_timestep_history.{extension}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    with (OUTPUT/'span_timestep_summary.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=='__main__':main()
