"""Approved unforced span-grid sensitivity; ASTR performs all flow advancement."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from compare_flowstate import read_flowstate
from run_adaptive_tgv_gate import STEP
from test_high_order_boundary_runtime import EXE,H1_VARIANT,launch_hcw
from test_h1_hcw_inlet_compatible import state

BASELINE=Path(__file__).parent/'out/h1_inlet_compatible_long_20261105/adaptive_early/test_hcw_early_trajectory_gate0/gpu'
CASES=(('span2mm_n16',16,.002),('span4mm_n32',32,.004))


def clock(case):
    return np.array([[int(m[1]),*map(float,m.groups()[1:])]
        for m in STEP.finditer((case/'run.log').read_text())])


def deviation(value):
    difference=value-value[:1]
    return difference-difference.mean(axis=0,keepdims=True)


def measure(case,path,time,step,checkpoint=False):
    with h5py.File(case/'outdat/new/resources/geometry.h5') as file:
        weight=file['q0004'][:-1,1:-1,1:-1]
    with h5py.File(path) as file:
        if checkpoint:
            rho=file['q0006'][...]
            velocity=np.stack([file[f'q{c:04d}'][...] for c in (7,8,9)],axis=-1)
            pressure=file['q0010'][...]
        else:
            rho=file['density'][...];velocity=file['velocity'][...];pressure=file['pressure'][...]
    plane=lambda x:x[:-1,1:-1,1:-1]
    variables=dict(density=plane(rho),u=plane(velocity[...,0]),v=plane(velocity[...,1]),
                   w=plane(velocity[...,2]),pressure=plane(pressure))
    rms={}
    for name,value in variables.items():
        perturbation=deviation(value)
        rms[name]=float(np.sqrt(np.sum(weight*perturbation**2)/weight.sum()))
    w=variables['w'];nz=w.shape[0]
    transformed=np.fft.rfft(deviation(w),axis=0)/nz
    power=abs(transformed)**2
    if nz%2==0:power[1:-1]*=2
    else:power[1:]*=2
    modes=np.sqrt(np.sum(power*weight[0],axis=(1,2))/weight[0].sum())
    return dict(step=step,time=time,spanwise_deviation_rms=rms,
                w_fourier_mode_rms=modes.tolist())


def analyze_case(case):
    records=clock(case);steps=int(records[-1,0])
    values=state(case,steps)
    names,stats=read_flowstate(case)
    manifest=json.loads((case/'case_manifest.json').read_text())
    samples=[]
    for path in sorted((case/'outdat/new/fields').glob('segment*/step*/data.h5')):
        with h5py.File(path) as file:
            metadata=file['metadata'][...]
        time=float(metadata[2:3].view(np.float64)[0])
        samples.append(measure(case,path,time,int(metadata[1])))
    final=measure(case,case/f'outdat/new/checkpoints/step{steps:012d}/state.h5',
                  float(records[-1,2]),steps,checkpoint=True)
    internal=values[...,4]-.5*np.sum(values[...,1:4]*(values[...,1:4]/values[...,:1]),axis=-1)
    return dict(case=str(case.resolve()),span_m=manifest['geometry']['span_m'],
        z_intervals=manifest['grid']['astr_intervals'][2],dz_m=manifest['grid']['dz_m'],
        steps=steps,final_time=float(records[-1,2]),last_time_start=float(records[-1,1]),
        dt_min=float(records[:,3].min()),dt_max=float(records[:,3].max()),
        rho_min=float(values[...,0].min()),pressure_min=float(values[...,9].min()),
        temperature_min=float(values[...,10].min()),internal_energy_min=float(internal.min()),
        final_statistics=dict(zip(names,stats[-1].tolist())),samples=samples,final=final)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,default=BASELINE)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--analyze-only',action='store_true')
    args=parser.parse_args()
    assert EXE and H1_VARIANT=='inlet-compatible','Select the approved root executable and H1 candidate'
    args.output.mkdir(parents=True,exist_ok=True)
    baseline=analyze_case(args.baseline)
    target=baseline['final_time']
    report=dict(scope='Unforced 2D extrusion sensitivity, not CPU/GPU equivalence or 3D physical convergence',
        target_time=target,cfl_target=.5,growth_factor=1.1,
        norm='Jacobian-weighted nodal RMS of f minus its unique-z mean, interior x/y only; not time RMS',
        seed='No imposed perturbation; roundoff seeds differ across grids',baseline=baseline,cases=[])
    for label,nz,span in CASES:
        case=args.output/label
        if not args.analyze_only:
            launch_hcw(case,'gpu',steps=2 if args.smoke else 420,z_intervals=nz,span_m=span,
                automatic_cfl=.5,stop_time=None if args.smoke else target,
                checkpoint_interval=1000000,volume_time_interval=None if args.smoke else target/10)
        result=analyze_case(case)
        if not args.smoke:
            assert result['last_time_start']<target<=result['final_time']
        report['cases'].append(result)
        (args.output/'span_grid.json').write_text(json.dumps(report,indent=2)+'\n')
        print(label,'completed',result['steps'],'time',result['final_time'],
              'spanwise RMS',result['final']['spanwise_deviation_rms'],flush=True)


if __name__=='__main__':main()
