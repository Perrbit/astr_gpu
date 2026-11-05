"""Native observed-error attribution; no imposed mode or Python flow integrator."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from analyze_h1_hcw_sensitivity import native_q,rates,endpoint_decomposition
from run_h1_hcw_span_grid import clock
from test_high_order_boundary_runtime import EXE,H1_VARIANT,launch_hcw

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'tests/gpu_validation/out/h1_cfl03_20261105/window/test_independent_cfl03_physica0'


def sensor_inputs(root,step):
    from analyze_m12_sensor_diagnostics import replay
    from check_air5_numq11_shock_tube import read_sensor_mask
    from run_native_precision_curve_gate import read_gradients
    from analyze_h1_hcw_sensitivity import rank_rhs
    def neighbors(face,j,k):
        points=set()
        for i in (face,face+1):
            for axis in range(3):
                for offset in range(-4,6):
                    point=[i,j,k];point[axis]+=offset
                    if axis<2:point[axis]=max(0,min((32,48)[axis],point[axis]))
                    elif point[axis]<0 or point[axis]>5:point[axis]%=5
                    points.add(tuple(point))
        return points
    values={};residuals={}
    for source in ('cpu','gpu'):
        prefix=root/f'endpoints_{source}{step}'/'diagnostics'
        gradient,_=read_gradients(prefix/'state.gradients.step00000000.rk01.rank00000000.bin','fp64')
        with (prefix/'state.pre_rhs_primitives.step00000000.rk01.rank00000000.bin').open('rb') as file:
            header=np.fromfile(file,np.int32,4)
            storage=np.fromfile(file,np.float64).reshape((*tuple(header[:3]+2*header[3]+1),6),order='F')
        velocity_gradient=gradient[...,:9].reshape((*gradient.shape[:3],3,3),order='F')
        # This observed pair is rank0 of the fixed NP2x fixture, not an M12 detail-file parser.
        parts=replay(np.r_[header,1,4,0],velocity_gradient,storage[...,4])
        raw,_=read_sensor_mask(prefix/'state.sensor.step00000000.rk01.rank00000000.bin')
        residuals[source]=float(abs(parts['sensor']-raw).max())
        assert residuals[source]<=2e-10,residuals
        values[source]=dict(raw=raw,parts=parts,endpoint=rank_rhs(root/f'endpoints_{source}{step}','h1_inlet_flux_endpoints'))
    traces=[]
    for j in (13,15,25):
        for face in range(4):
            candidates=neighbors(face,j,2)
            trace=dict(interface=face,j=j,k=2,inputs={})
            for source,data in values.items():
                high=max(float(data['raw'][point]) for point in candidates)
                origins=sorted(point for point in candidates if data['raw'][point]==high)
                origin=origins[0];parts=data['parts'];t=min(1,max(0,high/.001))
                theta=(t*t)*(t*t)*t*(6-5*t)
                native=float(data['endpoint'][face,j,2,10])
                assert abs(theta-native)<=2e-14,(source,face,j,theta,native)
                trace['inputs'][source]=dict(origins=origins,sensor=high,theta=native,
                    factor=float(parts['factor'][origin]),curvature=float(parts['curvature'][origin]),
                    div=float(parts['div'][origin]),div2=float(parts['div2'][origin]),vort2=float(parts['vort2'][origin]))
            a,b=trace['inputs']['cpu'],trace['inputs']['gpu']
            trace['same_argmax_origins']=a['origins']==b['origins']
            if trace['same_argmax_origins']:
                trace['sensor_gradient_factor_change']=(b['factor']-a['factor'])*a['curvature']
                trace['sensor_pressure_curvature_change']=b['factor']*(b['curvature']-a['curvature'])
            traces.append(trace)
    theta={}
    for source,data in values.items():
        signed=np.where(data['parts']['div']<0,data['raw'],0.)
        weights=np.empty((4,47,5,1))
        for face in range(4):
            for j in range(1,48):
                for k in range(5):
                    s=max(float(signed[point]) for point in neighbors(face,j,k))
                    t=min(1,max(0,s/.001));weights[face,j-1,k,0]=(t*t)*(t*t)*t*(6-5*t)
        theta[source]=weights
    packets={source:data['endpoint'][:4,1:-1,:-1] for source,data in values.items()}
    high={source:p[...,:5] for source,p in packets.items()}
    low={source:p[...,5:10] for source,p in packets.items()}
    flux={source:high[source]+theta[source]*(low[source]-high[source]) for source in values}
    change=flux['gpu']-flux['cpu']
    frozen=(1-theta['cpu'])*(high['gpu']-high['cpu'])+theta['cpu']*(low['gpu']-low['cpu'])
    feedback=(theta['gpu']-theta['cpu'])*(low['gpu']-high['gpu'])
    with h5py.File(root/f'endpoints_cpu{step}'/'outdat/new/resources/geometry.h5') as file:
        jac=file['q0004'][...].transpose(2,1,0)[1:4,1:-1,:-1,None]
    delta=(native_q(root/f'endpoints_gpu{step}')-native_q(root/f'endpoints_cpu{step}'))[1:4,1:-1,:-1]
    norm=float(np.sum(jac*delta**2))
    project=lambda value:float(np.sum(delta*(value[:-1]-value[1:]))/norm)
    proposal=dict(scope='Offline compression-only raw-sensor assessment on native H/L packets; NOT a native candidate RK or positivity test',
        raw_rule='raw_original if div<0 else 0; same epsilon, pressure curvature, dilation and phi',
        x_all5_projection=project(change),frozen_weight_projection=project(frozen),weight_feedback_projection=project(feedback))
    return dict(scope='Native gradients and cached pressure replayed offline; raw sensor and current weight unchanged',
        own_input_sensor_residuals=residuals,face_traces=traces,compression_only_proposal=proposal)


def analyze(root,dt,step):
    cases={name:root/f'endpoints_{name}{step}' for name in ('cpu','gpu')}
    with h5py.File(cases['cpu']/'outdat/new/resources/geometry.h5') as file:
        jac=file['q0004'][...].transpose(2,1,0)[...,:-1,None]
    initial={name:native_q(case)[...,:-1,:] for name,case in cases.items()}
    rhs={name:rates(case,jac,-1) for name,case in cases.items()}
    delta=initial['gpu']-initial['cpu']
    changes={name:rhs['gpu'][name]-rhs['cpu'][name] for name in rhs['cpu']}
    ending={}
    for name,case in cases.items():
        with h5py.File(case/'outdat/new/checkpoints/step000000000001/state.h5') as file:
            ending[name]=np.stack([file[f'q{q:04d}'][...].transpose(2,1,0) for q in range(1,6)],axis=-1)[...,:-1,:]
    evolved=ending['gpu']-ending['cpu']
    mean=delta.mean(axis=2,keepdims=True);nonzero=delta-mean
    regions={}
    for label,start,end in (('inlet_columns1to3',1,4),('remaining_interior',4,64),('all_interior',1,64)):
        mask=np.zeros(delta.shape[:-1],bool);mask[start:end,1:-1,:]=True
        norm=float(np.sum(jac[mask]*delta[mask]**2));assert norm>0
        project=lambda value:float(np.sum(jac[mask]*delta[mask]*value[mask])/norm)
        # The actual J may vary slightly in z through floating-point metric evaluation.
        broad_mean=np.broadcast_to(mean,delta.shape)
        fractions=dict(mean=float(np.sum(jac[mask]*broad_mean[mask]**2)/norm),
                       nonzero=float(np.sum(jac[mask]*nonzero[mask]**2)/norm),
                       cross=float(2*np.sum(jac[mask]*broad_mean[mask]*nonzero[mask])/norm))
        split={}
        for name,change in changes.items():
            rmean=np.broadcast_to(change.mean(axis=2,keepdims=True),change.shape)
            rnonzero=change-rmean
            split[name]=dict(mean=float(np.sum(jac[mask]*broad_mean[mask]*rmean[mask])/norm),
                nonzero=float(np.sum(jac[mask]*nonzero[mask]*rnonzero[mask])/norm),
                cross=float(np.sum(jac[mask]*(broad_mean[mask]*rnonzero[mask]+nonzero[mask]*rmean[mask]))/norm))
        regions[label]=dict(norm_fractions=fractions,instantaneous_log_norm_rate={name:project(value) for name,value in changes.items()},
            mean_nonzero_rate_split=split,
            one_native_step_error_norm_factor=float(np.sqrt(np.sum(jac[mask]*evolved[mask]**2)/norm)),
            reconstruction_residual=float(abs((changes['x']+changes['y']+changes['z']+
                changes['diffusion']+changes['boundary']-changes['complete'])[mask]).max()))
    report=dict(scope='Observed same-phase state pair on the same CPU and common dt; not a new physical mode or stability proof',
        norm='Jacobian-weighted nodal all-five Euclidean diagnostic norm; unique z, no x/y physical faces; not physical energy',
        dt=dt,step=step,regions=regions,endpoint=endpoint_decomposition(root,step))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--step',type=int,default=415)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--replay-library',type=Path,required=True)
    parser.add_argument('--analyze-only',action='store_true')
    parser.add_argument('--sensor-inputs',action='store_true')
    args=parser.parse_args()
    assert EXE and H1_VARIANT=='inlet-compatible'
    args.output.mkdir(parents=True,exist_ok=True)
    dt=float(clock(args.source/'cpu')[args.step-1,4])
    sequence=np.array([[1,0.,dt,dt,dt]])
    if not args.analyze_only:
        for source in ('cpu','gpu'):
            field=next((args.source/source/'outdat/new/fields').glob(f'segment*/step{args.step:012d}/data.h5'))
            launch_hcw(args.output/f'endpoints_{source}{args.step}','cpu',steps=1,field_source=field,
                diagnostics=True,inlet_flux=True,replay_clocks=sequence,replay_library=args.replay_library,
                validation_gradients=args.sensor_inputs)
        without=args.output/'diagnostic_off'
        field=next((args.source/'cpu/outdat/new/fields').glob(f'segment*/step{args.step:012d}/data.h5'))
        launch_hcw(without,'cpu',steps=1,field_source=field,diagnostics=True,
                   replay_clocks=sequence,replay_library=args.replay_library,validation_gradients=args.sensor_inputs)
        end='outdat/new/checkpoints/step000000000001/state.h5'
        with h5py.File(args.output/f'endpoints_cpu{args.step}'/end) as a,h5py.File(without/end) as b:
            for n in range(1,12):np.testing.assert_array_equal(a[f'q{n:04d}'][...],b[f'q{n:04d}'][...])
    report=analyze(args.output,dt,args.step)
    if args.sensor_inputs:report['sensor_inputs']=sensor_inputs(args.output,args.step)
    (args.output/'observed_pair.json').write_text(json.dumps(report,indent=2)+'\n')
    assert report['endpoint']['reconstruction_residual_absolute']<=2e-10
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
