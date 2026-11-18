"""Isolate MP1 trajectory perturbations with the same public ASTR device operator."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np

from run_cpp_multi_tgv_gate import raw_field


def read_rhs(path):
    with path.open('rb') as stream:
        header=np.fromfile(stream,np.int32,4)
        return np.fromfile(stream,np.float64).reshape((*header[:3]+1,header[3]),order='F')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','reference','current','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--step',type=int,default=3)
    parser.add_argument('--stage',type=int,default=2)
    parser.add_argument('--point',default='3,5,10')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    point=tuple(map(int,args.point.split(',')))
    if len(point)!=3:parser.error('--point requires three indices')
    suffix=f'.step{args.step:08d}.rk{args.stage:02d}.rank00000000.bin'
    with h5py.File(args.reference/'outdat/geometry.h5') as source:
        jac=source['jacob'][...].transpose(2,1,0)
        metric=np.empty((*jac.shape,3,3))
        for a in range(3):
            for b in range(3):metric[...,a,b]=source[f'dxi{a+1}{b+1}'][...].transpose(2,1,0)
    shutil.copy2(args.binary.resolve(),out/'tested_probe');shutil.copyfile(__file__,out/'runner.py')
    report=dict(step=args.step,stage=args.stage,point=point,
        geometry=str((args.reference/'outdat/geometry.h5').resolve()),
        binary_sha256=hashlib.sha256((out/'tested_probe').read_bytes()).hexdigest(),
        scope='diagnostic only; frozen q/cache replay, no CFD advance, no changed acceptance gate; LP2 pending',sources={})
    results={}
    for name,case in (('reference',args.reference),('current',args.current)):
        h,q=raw_field(case/('outdat/rhs.pre_rhs'+suffix))
        ph,p=raw_field(case/('outdat/rhs.pre_rhs_primitives'+suffix),primitive=True)
        upper=np.asarray(h[:3]);halo=h[3]
        if list(h[:4])!=list(ph) or tuple(upper+1)!=jac.shape:raise AssertionError('replay geometry/state layout')
        grid=np.indices(q.shape[:3])-halo;wrapped=tuple(grid[a]%upper[a] for a in range(3))
        physical=tuple(slice(halo,halo+n+1) for n in upper)
        metrics=metric[wrapped].copy();metrics[physical]=metric
        jacobian=jac[wrapped].copy();jacobian[physical]=jac
        packet=out/(name+'.bin')
        with packet.open('xb') as stream:
            np.array([*upper,halo],np.int32).tofile(stream)
            for value in (q,p,metrics,jacobian):value.ravel(order='F').tofile(stream)
        with (out/(name+'.log')).open('w') as log:
            subprocess.run([str((out/'tested_probe').resolve()),str(packet),str(out/name)],
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,check=True)
        actual=read_rhs(case/('outdat/rhs.conv'+suffix))
        rhs=np.fromfile(out/(name+'.rhs.f64'),np.float64).reshape((*upper+1,5),order='F')
        reconstructed=np.zeros_like(rhs)
        ledger=[]
        for axis in range(3):
            work=np.fromfile(out/(name+f'.axis{axis}.f32'),np.float32).reshape((*upper+3,5),order='F')
            lower=[slice(1,n+2) for n in upper];higher=lower.copy();lower[axis]=slice(0,upper[axis]+1)
            reconstructed+=work[tuple(lower)].astype(np.float64)-work[tuple(higher)].astype(np.float64)
            right=list(point);left=list(point);left[axis]-=1
            def stored(location):
                value=work[tuple(n+1 for n in location)+(4,)]
                return dict(value=float(value),bits=f'{value.view(np.uint32):08x}',ulp=float(abs(np.spacing(value))))
            l,r=stored(left),stored(right)
            ledger.append(dict(axis=axis,left=l,right=r,contribution=l['value']-r['value']))
        if not np.isfinite(rhs).all() or not np.array_equal(rhs,reconstructed):
            raise AssertionError('stored-interface ledger does not reconstruct the device RHS exactly')
        observed=dict(source=str(case.resolve()),replay_vs_actual_max_abs=np.max(abs(rhs-actual),axis=(0,1,2)).tolist(),
            replay_target=float(rhs[point+(4,)]),actual_target=float(actual[point+(4,)]),energy_interface_ledger=ledger)
        report['sources'][name]=observed;results[name]=dict(q=q,p=p,rhs=rhs,actual=actual)
    old,new=results['reference'],results['current']
    reference_replay=float(old['rhs'][point+(4,)]);current_replay=float(new['rhs'][point+(4,)])
    reference_actual=float(old['actual'][point+(4,)]);current_actual=float(new['actual'][point+(4,)])
    contributions=dict(same_operator_input_change=current_replay-reference_replay,
        old_state_implementation_difference=reference_replay-reference_actual,
        current_state_replay_difference=current_actual-current_replay)
    total=current_actual-reference_actual
    if sum(contributions.values())!=total:raise AssertionError('pointwise error decomposition does not close exactly')
    stencil=np.count_nonzero((grid<0)|(grid>upper[:,None,None,None]),axis=0)<=1
    report.update(input_q_max_abs=float(np.max(abs(new['q'][stencil]-old['q'][stencil]))),
        input_primitive_max_abs=float(np.max(abs(new['p'][stencil]-old['p'][stencil]))),
        input_scope='physical nodes plus single-axis face halos; unused multi-axis corners excluded',
        target_actual_difference=total,decomposition=contributions,decomposition_exact=True,
        replay_rhs_input_change_max_abs=np.max(abs(new['rhs']-old['rhs']),axis=(0,1,2)).tolist())
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
