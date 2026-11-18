"""Ordinary ASTR sparse selected-time fields and exact lifecycle checks."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from test_index_slice_runtime import arguments,product,segments,frames
from test_insitu_station_profiles import station,read_profile,check_profiles
from run_output_restart_validation import run_case
from run_cpp_insitu_gate import compare_hdf
from run_cpp_tgv_gate import logged_steps

FIELDS=['density','velocity_x','velocity_y','velocity_z','pressure','temperature']


def configuration(enabled=True):
    unlimited=np.iinfo(np.int64).max//4096
    return f"""&insitu_run
 enabled={'.true.' if enabled else '.false.'},statistics={'.true.' if enabled else '.false.'},render=f,
 products='{'catalog_stats' if enabled else 'all'}',statistics_window=.0005,.0115,output_directory='outdat',
 host_budget_bytes={unlimited},device_budget_bytes={unlimited},device_reserve_bytes=0
/
"""


def definitions(expanded,thermal_gradients=True):
    raw=product('raw',FIELDS)
    raw['schedule']['interval']=1
    if expanded:
        raw['sampling']=dict(mode='grid_index',axis='i',index=8,index_ranges=dict(j=[0,16],k=[0,16]))
    records=[raw]
    derivative=['velocity_gradient_xx','velocity_gradient_yx','velocity_gradient_zx',
                'velocity_gradient_xy','velocity_gradient_yy','velocity_gradient_zy',
                'velocity_gradient_xz','velocity_gradient_yz','velocity_gradient_zz',
                'Q_rs','velocity_divergence','vorticity_x','vorticity_y','vorticity_z']
    if thermal_gradients:
        derivative+=['temperature_gradient_x','temperature_gradient_y','temperature_gradient_z']
    for source in ('reynolds','favre'):
        mean=deepcopy(raw);mean.update(id=source,source=source,statistics_source='selected-time')
        mean['fields']=FIELDS.copy() if source=='reynolds' else FIELDS[1:]
        if expanded:
            mean['fields']+=derivative+(['density_gradient_magnitude'] if source=='reynolds' else [])
        records.append(mean)
    if expanded:
        records.extend([station('profile-r','reynolds'),station('profile-f','favre')])
        r=product('raw-j',FIELDS);r['schedule']['interval']=1
        r['sampling']=dict(mode='grid_index',axis='j',index=8,index_ranges=dict(i=[0,16],k=[0,16]))
        records.append(r)
        for source in ('reynolds','favre'):
            mean=deepcopy(r);mean.update(id='j-'+source,source=source,statistics_source='selected-time')
            mean['fields']=['velocity_x','pressure','temperature'] if source=='reynolds' else ['velocity_x','velocity_z']
            records.append(mean)
            fluct=deepcopy(mean);fluct.update(id='fluct-'+source,fluctuation=True);records.append(fluct)
    return records


def slice_map(case):
    return {p.parent.parent.name:p for p in segments(case)}


def compare_slices(actual,reference,exact=False,limit=2e-10):
    a,b=slice_map(actual),slice_map(reference)
    if set(a)!=set(b):
        raise AssertionError(('selected product identities',set(a),set(b)))
    errors={}
    for name in a:
        current=frames(a[name])
        expected=[f'step{s:012d}.h5' for s in range(6,13)] if exact else [p.name for p in frames(b[name])]
        if [p.name for p in current]!=expected:
            raise AssertionError(('selected schedule',name,[p.name for p in current],expected))
        error=0.
        for p in current:
            error=max(error,compare_hdf(p,b[name]/p.name,exact=exact,limit=limit))
        errors[name]=error
    return errors


def independent_means(case,window=(.0005,.0115),limit=2e-10):
    paths=slice_map(case)
    samples=[]
    for path in frames(paths['raw']):
        with h5py.File(path) as data:
            samples.append({field:data[field][...] for field in FIELDS})
    if len(samples)!=13:
        raise AssertionError('thirteen complete-state endpoint samples required')
    w=np.zeros(13)
    for s,row in enumerate(logged_steps(case,12),start=1):
        t0,t1=row[1:3];dt=t1-t0
        a,b=max(t0,window[0]),min(t1,window[1])
        if b<=a:
            continue
        right=.5*((a-t0)/dt)+.5*((b-t0)/dt)
        w[s-1]+=(b-a)*(1-right);w[s]+=(b-a)*right
    rho=np.stack([s['density'] for s in samples]).astype(np.float64);mass=np.tensordot(w,rho,axes=1)
    error=0.
    for source in ('reynolds','favre'):
        with h5py.File(frames(paths[source])[-1]) as data:
            for field in FIELDS if source=='reynolds' else FIELDS[1:]:
                x=np.stack([s[field] for s in samples]).astype(np.float64)
                expected=(np.tensordot(w,x,axes=1)/w.sum() if source=='reynolds'
                          else np.tensordot(w,rho*x,axes=1)/mass)
                y=data[field][...]
                if not np.isfinite(y).all() or not np.isfinite(expected).all():
                    raise AssertionError('nonfinite endpoint integral')
                error=max(error,float(np.max(abs(y-expected))/max(1.,float(np.max(abs(expected))))))
    if limit is not None and error>limit:
        raise AssertionError(('independent endpoint integral',error))
    return error


def profiles(case):
    return {p.parent.parent.name:p for p in (case/'outdat/new/profile_products').glob('*/*/segment*')}


def compare_profiles(actual,reference,exact=False,limit=2e-10):
    a,b=profiles(actual),profiles(reference)
    if set(a)!=set(b):
        raise AssertionError('profile identities differ')
    error=raw_rms=0.
    for name in a:
        files=sorted(a[name].glob('frame.step*.csv'))
        expected=[f'frame.step{s:012d}.csv' for s in range(6,13)] if exact else [p.name for p in sorted(b[name].glob('frame.step*.csv'))]
        if [p.name for p in files]!=expected:
            raise AssertionError(('profile sampling',name))
        for p in files:
            x,metadata=read_profile(p);y,old=read_profile(b[name]/p.name)
            if x.dtype.names!=y.dtype.names or not all(np.isfinite(x[field]).all() for field in x.dtype.names):
                raise AssertionError('profile field layout/nonfinite')
            if exact:
                if p.read_bytes()!=(b[name]/p.name).read_bytes():
                    raise AssertionError('same-backend profile bytes differ')
                continue
            for field in x.dtype.names:
                first,second=x[field],y[field]
                if field.startswith('rms_'):
                    raw_rms=max(raw_rms,float(np.max(abs(first-second))));first=first**2;second=second**2
                error=max(error,float(np.max(abs(first-second))/max(1.,float(np.max(abs(second))))))
    if limit is not None and error>limit:
        raise AssertionError(('profile comparison',error))
    return dict(scaled=error,raw_rms_maxabs=raw_rms)


def check_fluctuations(case,limit=2e-10):
    p=slice_map(case);error=0.
    for source in ('reynolds','favre'):
        for path in frames(p['fluct-'+source]):
            with h5py.File(path) as f,h5py.File(p['j-'+source]/path.name) as m,h5py.File(p['raw-j']/path.name) as raw:
                fields=['velocity_x','pressure','temperature'] if source=='reynolds' else ['velocity_x','velocity_z']
                for field in fields:
                    x=f[field][...]+m[field][...];y=raw[field][...]
                    if not np.isfinite(x).all() or not np.isfinite(y).all():
                        raise AssertionError('nonfinite fluctuation identity')
                    error=max(error,float(np.max(abs(x-y))/max(1.,float(np.max(abs(y))))))
    if limit is not None and error>limit:
        raise AssertionError(('instantaneous - selected mean',error))
    return error


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--cpu-reference',type=Path,required=True)
    parser.add_argument('--gpu-reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ranks',type=int,default=1)
    parser.add_argument('--axis',choices=['x','y','z'],default='x')
    parser.add_argument('--expanded',action='store_true')
    parser.add_argument('--memcheck',action='store_true')
    parser.add_argument('--precision',choices=('fp64','fp32'),default='fp64')
    parser.add_argument('--without-thermal-gradients',action='store_true',
        help='Explicit fluid-derived fixture; the shared native32 admission excludes the new thermal diagnostics')
    opt=parser.parse_args()
    output=opt.output.resolve();output.mkdir(parents=True,exist_ok=False)
    args=arguments(output);args.executable=opt.binary.resolve();args.axis=opt.axis
    args.native_precision=opt.precision
    dtype=np.float32 if opt.precision=='fp32' else np.float64
    limit=None if opt.precision=='fp32' else 2e-10
    window=tuple(map(float,np.array([.0005,.0115],dtype=dtype)))
    args.unrestricted=True;args.insitu_write_eps='0'
    common=dict(grid='16,16,16',checkpoint_interval=5,numerics=dict(filter_scheme='explicit10'))
    records=definitions(opt.expanded,not opt.without_thermal_gradients)
    quiet,_=run_case(args,ROOT,'gpu',opt.ranks,'cpp-off',12,insitu_config=configuration(False),
        product_catalog_records=[],**common)
    current,_=run_case(args,ROOT,'gpu',opt.ranks,'cpp-selected',12,insitu_config=configuration(),
        product_catalog_records=records,memcheck=opt.memcheck,**common)
    final='outdat/new/checkpoints/step000000000012'
    with h5py.File(current/final/'state.h5') as state:
        if any(state[f'q{c:04d}'].dtype!=np.dtype(dtype) for c in range(1,6)):
            raise AssertionError('actual selected-statistics solver precision differs')
    compare_hdf(current/final/'state.h5',quiet/final/'state.h5',exact=True)
    if not np.array_equal(logged_steps(current,12),logged_steps(quiet,12)):
        raise AssertionError('selected statistics changed step control')
    status=json.loads((current/'outdat/new/products.status.json').read_text())
    if any(p['state']!='active' or p['last_output_step']!=12 for p in status['products']):
        raise AssertionError(status)
    report=dict(ranks=opt.ranks,axis=opt.axis,expanded=opt.expanded,memcheck=opt.memcheck,precision=opt.precision,
        thermal_gradients=not opt.without_thermal_gradients,
        on_off_state_bits=True,endpoint_integral_error=independent_means(current,window,limit),references={})
    if opt.expanded:
        report['fluctuation_identity_error']=check_fluctuations(current,limit)
        independent={}
        check_profiles(current,[records[0],*[p for p in records if p['kind']=='station_profile']],
            lambda name,value: independent.__setitem__(name,value),window=window,limit=limit,
            sample_times=np.r_[0.,logged_steps(current,12)[:,2]])
        report['independent_profiles']=independent
    for name,binary,backend in [('cpu',opt.cpu_reference,'cpu'),('old_gpu',opt.gpu_reference,'gpu')]:
        args.executable=binary.resolve()
        reference,_=run_case(args,ROOT,backend,opt.ranks,name,12,insitu_config=configuration(),
            product_catalog_records=records,**common)
        if not np.array_equal(logged_steps(current,12),logged_steps(reference,12)):
            raise AssertionError('selected reference timestep sequence differs')
        entry=dict(state_scaled=compare_hdf(current/final/'state.h5',reference/final/'state.h5',limit=limit),
                   fields=compare_slices(current,reference,limit=limit))
        if opt.expanded:
            entry['profiles']=compare_profiles(current,reference,limit=limit)
        report['references'][name]=entry
    args.executable=opt.binary.resolve()
    prefix,_=run_case(args,ROOT,'gpu',opt.ranks,'cpp-prefix',5,insitu_config=configuration(),product_catalog_records=records,**common)
    resumed,_=run_case(args,ROOT,'gpu',opt.ranks,'cpp-resumed',12,insitu_config=configuration(),
        product_catalog_records=[],restore=prefix/'outdat/new/checkpoints/step000000000005',**common)
    for name in ('state.h5','statistics.h5'):
        compare_hdf(current/final/name,resumed/final/name,exact=True)
    if not np.array_equal(logged_steps(resumed,7,first_step=6),logged_steps(current,12)[5:]):
        raise AssertionError('selected restart changed timestep history')
    report['same_backend_restart']=dict(state_bits=True,statistics_bits=True,slices=compare_slices(resumed,current,exact=True))
    if opt.expanded:
        report['same_backend_restart']['profiles']=compare_profiles(resumed,current,exact=True)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
