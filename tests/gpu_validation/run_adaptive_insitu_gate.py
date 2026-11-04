"""Variable-dt numerical planes and independent time-statistics quadrature."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from run_output_restart_validation import run_case
from run_adaptive_tgv_gate import compare_state,STEP
from run_adaptive_mean_statistics_gate import exact_file
from insitu_statistics_reference import endpoint_weights,reference

ROOT=Path(__file__).resolve().parents[2]
FINAL=Path('outdat/new/checkpoints/step000000000012')
WINDOW=(0.0005,0.020)
UNLIMITED_BUDGET=np.iinfo(np.int64).max//4096


def product(name,source,clock,interval):
    return dict(id=name,kind='slice',enable_request=1,source=source,
        fields=(['density'] if source=='instantaneous' else [])+['velocity_x','velocity_y','velocity_z'],
        sampling=dict(mode='grid_index',axis='k',index=4,index_ranges=dict(i=[0,32],j=[0,32])),
        output=dict(channel='data',format='hdf5-xdmf'),schedule=dict(mode=clock,interval=interval))


def frames(case,name):
    segment=list((case/'outdat/new/slice_products'/name).glob('*/segment*'))
    if len(segment)!=1:
        raise AssertionError(f'expected one active segment for {name}: {segment}')
    return {int(path.stem[4:]):path for path in sorted(segment[0].glob('step*.h5'))}


def launch(args,backend,label,restore=None,products=True):
    setup=SimpleNamespace(output=args.output,executable=getattr(args,backend),mpiexec=args.mpiexec,
        case='tgv',mode='steps',restart_step=5,initial_dimension=0,legacy_statistics=False,
        statistics=True,initial_restart=False,filter_workspace='scalar',force='feedback',axis='x',
        no_samples=True,unrestricted=True)
    definitions=[product('instant','instantaneous','steps',1),
        product('reynolds','reynolds','time',0.004),product('favre','favre','time',0.004)]
    config=('&insitu_run\n enabled=t,statistics=t,render=f,statistics_window=0.0005,0.020,\n'
        f'output_directory="outdat",host_budget_bytes={UNLIMITED_BUDGET},'
        f'device_budget_bytes={UNLIMITED_BUDGET},device_reserve_bytes=0\n/\n')
    runtime=('&run_control max_steps=12,reload_steps=100 /\n&timestep mode="auto",initial_dt=0.001 /\n'
        '&logging list_steps=0 /\n')
    case,_=run_case(setup,ROOT,backend,2,label,12,restore=restore,
        checkpoint_interval=5 if products else 12,insitu_config=config,
        product_catalog_records=definitions if products else [],grid='32,32,32',
        run_config_text=runtime,insitu_timing=True,
        capture_checkpoint_step=5 if products and restore is None else None)
    return case


def clocks(case):
    data={0:0.0}
    data.update({int(m[1]):float(m[3]) for m in STEP.finditer((case/'run.log').read_text())})
    if sorted(data)!=list(range(13)):
        raise AssertionError('instantaneous reference must cover all complete steps')
    return data


def point_result(path):
    with h5py.File(path) as source:
        state=np.stack([source[f'q{i:04d}'][4,:32,:32] for i in range(1,35)],axis=-1).astype(float)
    duration=state[...,7];mass=state[...,8]
    if np.any(duration<=0) or np.any(mass<=0):
        raise AssertionError('formal statistics lack positive coverage or mass')
    density=mass/duration
    covariance_r=state[...,15:24]/duration[...,None]
    covariance_f=state[...,24:33]/mass[...,None]
    rms_r=np.sqrt(covariance_r[...,[0,4,8]])
    rms_f=np.sqrt(covariance_f[...,[0,4,8]])
    result=np.concatenate([duration[...,None],density[...,None],state[...,9:15],rms_r,rms_f,
        covariance_r,covariance_f,density[...,None]*covariance_f],axis=-1)
    return result,tuple(state[0,0,:2])


def difference(actual,expected,native_fp32):
    a,b=np.asarray(actual,float).copy(),np.asarray(expected,float).copy()
    a[...,8:14]**=2;b[...,8:14]**=2
    error=float(np.max(np.abs(a-b)))
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError('nonfinite mean/covariance/RMS')
    if not native_fp32 and error>2e-10:
        raise AssertionError(f'time quadrature differs from independent reference: {error}')
    return dict(max_abs_with_rms_squared=error,
        rms_max_abs=float(np.max(np.abs(actual[...,8:14]-expected[...,8:14]))))


def check_products(case,native_fp32):
    time=clocks(case)
    instant=frames(case,'instant')
    if sorted(instant)!=list(range(13)):
        raise AssertionError('instantaneous plane must retain every actual complete state')
    dtype=np.dtype('float32' if native_fp32 else 'float64')
    samples=np.zeros((13,32,32,11),dtype=np.longdouble)
    for step,path in instant.items():
        with h5py.File(path) as data:
            if data['density'].dtype!=dtype or float(data['clock'][0])!=time[step]:
                raise AssertionError('plane storage kind or physical time differs')
            samples[step,...,5]=data['density'][:32,:32]
            for component,name in enumerate(('velocity_x','velocity_y','velocity_z'),start=6):
                samples[step,...,component]=data[name][:32,:32]
    actual,window=point_result(case/FINAL/'statistics.h5')
    expected=reference(samples,endpoint_weights(list(time.values()),window))
    report=difference(actual,expected,native_fp32)
    due=[];last_time=None
    for step,t in time.items():
        if t<=window[0]:
            continue
        if last_time is None or t-last_time>=0.004:
            due.append(step);last_time=t
    for name,columns in (('reynolds',slice(2,5)),('favre',slice(5,8))):
        observed=frames(case,name)
        if sorted(observed)!=due:
            raise AssertionError(f'{name} time output identity differs: {sorted(observed)} versus {due}')
        errors=[]
        for step,path in observed.items():
            expected=reference(samples[:step+1],endpoint_weights(list(time.values())[:step+1],window))[...,columns]
            with h5py.File(path) as data:
                value=np.stack([data[field][:32,:32] for field in ('velocity_x','velocity_y','velocity_z')],axis=-1)
                if float(data['clock'][0])!=time[step] or not np.isfinite(value).all():
                    raise AssertionError('mean plane time or finite-value contract differs')
                error=float(np.max(np.abs(value-expected)))
                if not native_fp32 and error>2e-10:
                    raise AssertionError('mean plane differs from independent actual-time quadrature')
                errors.append(error)
        report[name]=dict(steps=due,max_abs=max(errors))
    return report,actual


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cpu','gpu','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--native-fp32',action='store_true')
    parser.add_argument('--analyze-only',action='store_true')
    args=parser.parse_args()
    for name in ('cpu','gpu','mpiexec','output'):
        setattr(args,name,getattr(args,name).resolve())
    if not args.analyze_only:
        args.output.mkdir(parents=True,exist_ok=False)
    report={};results={}
    for backend in ('cpu','gpu'):
        case=args.output/f'{backend}_np2_continuous'
        if not args.analyze_only:
            case=launch(args,backend,'continuous')
            resumed=launch(args,backend,'resumed',case/'captured/checkpoints/step000000000005')
            disabled=launch(args,backend,'without_products',products=False)
        else:
            resumed=args.output/f'{backend}_np2_resumed'
            disabled=args.output/f'{backend}_np2_without_products'
        for name in ('state.h5','statistics.h5'):
            exact_file(case/FINAL/name,resumed/FINAL/name)
            exact_file(case/FINAL/name,disabled/FINAL/name)
        for name in ('instant','reynolds','favre'):
            for step,path in frames(resumed,name).items():
                exact_file(path,frames(case,name)[step])
        report[backend],results[backend]=check_products(case,args.native_fp32)
        report[backend].update(exact_restart=True,products_do_not_change_state_or_statistics=True)
    report['cpu_gpu']=difference(results['cpu'],results['gpu'],args.native_fp32)
    report['status']='passed'
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
