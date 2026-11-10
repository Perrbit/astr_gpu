"""GPU-only controlled viscous TGV window; not a performance or DNS benchmark."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from run_output_restart_validation import run_case

ROOT=Path(__file__).resolve().parents[2]


def final_diagnostics(path):
    with h5py.File(path) as state:
        fields=[state[f'q{c:04d}'][:-1,:-1,:-1].astype(np.float64) for c in range(1,6)]
    if not all(np.isfinite(x).all() for x in fields):
        raise AssertionError('nonfinite physical conservative state')
    rho=fields[0]
    kinetic=sum(x*x for x in fields[1:4])/(2*rho)
    pressure=0.4*(fields[4]-kinetic)
    if not (rho>0).all() or not (pressure>0).all():
        raise AssertionError('nonpositive physical density/pressure')
    # Uniform periodic trapezoidal measure on unique nodes; s/R up to a constant.
    entropy=rho*(np.log(pressure)-1.4*np.log(rho))/0.4
    return dict(density_mean=float(rho.mean()),total_energy_mean=float(fields[4].mean()),
        kinetic_energy_mean=float(kinetic.mean()),entropy_density_mean=float(entropy.mean()),
        density_min=float(rho.min()),pressure_min=float(pressure.min()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=Path,required=True)
    parser.add_argument('--mpiexec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scheme',choices=['643e','643c'],required=True)
    opts=parser.parse_args()
    opts.output=opts.output.resolve()
    opts.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    steps=10000
    args=SimpleNamespace(output=opts.output,executable=opts.gpu.resolve(),mpiexec=opts.mpiexec.resolve(),
        case='tgv',mode='steps',restart_step=steps,initial_dimension=0,initial_restart=True,
        legacy_statistics=False,statistics=False,filter_workspace='scalar',force='feedback',axis='x',
        no_samples=True,wall_samples=False,unrestricted=True,maximum_cfl=0.5)
    runtime=f'''&run_control max_steps={steps}, reload_steps=1000 /
&timestep mode='fixed', fixed_dt=1.d-3 /
&logging list_steps=1 /
&repair crashfix=f /
&statistics enabled=f /
'''
    case,_=run_case(args,ROOT,'gpu',1,'keep_long',steps,grid='32,32,32',scheme=opts.scheme,
        lfilter=False,numerics={'convection_form':'keep'},checkpoint_interval=steps,checkpoint_keep=2,
        run_config_text=runtime,gpu_statistics_interval=100,tgv_conservation=True)
    initial=final_diagnostics(case/'outdat/new/checkpoints/step000000000000/state.h5')
    final=final_diagnostics(case/f'outdat/new/checkpoints/step{steps:012d}/state.h5')
    conservation=np.loadtxt(case/'tgv_conservation.dat',skiprows=1,ndmin=2)
    statistics=np.loadtxt(case/'flowstate.dat',skiprows=1,ndmin=2)
    if not np.isfinite(conservation).all() or not np.isfinite(statistics).all():
        raise AssertionError('nonfinite native TGV statistics')
    expected=np.arange(0,steps,100)
    if not np.array_equal(conservation[:,0],expected):
        raise AssertionError('native conservation sampling sequence differs')
    if not np.array_equal(statistics[:,0],expected[1:]):
        raise AssertionError('native flow statistics sampling sequence differs')
    report=dict(status='passed',scheme=opts.scheme,grid_intervals=[32]*3,steps=steps,dt=0.001,
        nominal_end_time=10.0,scope='same-path robustness; convection and viscosity vary together',
        initial=initial,final=final,cfl=json.loads((case/'cfl_gate.json').read_text()),
        flow_statistics_rows=len(statistics),conservation_rows=len(conservation),
        density_relative_drift=abs(final['density_mean']-initial['density_mean'])/abs(initial['density_mean']),
        total_energy_relative_drift=abs(final['total_energy_mean']-initial['total_energy_mean'])/abs(initial['total_energy_mean']),
        entropy_density_change=final['entropy_density_mean']-initial['entropy_density_mean'],
        maximum_sampled_density_relative_drift=float(np.max(abs(conservation[:,2]-conservation[0,2]))/abs(conservation[0,2])),
        maximum_sampled_energy_relative_drift=float(np.max(abs(conservation[:,3]-conservation[0,3]))/abs(conservation[0,3])))
    (opts.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
