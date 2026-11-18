"""Repack immutable corrected ASTR outputs; no numerical advance in Python."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from run_cpp_multi_tgv_gate import raw_field


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    for backend in ('cpu','gpu'):
        case=args.reference.resolve()/f'{backend}_np1'
        for stage in (1,2,3):
            suffix=f'.step00000000.rk{stage:02d}.rank00000000.bin'
            header,q=raw_field(case/('outdat/rhs.pre_rhs'+suffix))
            ph,p=raw_field(case/('outdat/rhs.pre_rhs_primitives'+suffix),primitive=True)
            if header[:4]!=ph:raise AssertionError('q/primitive phase layout')
            upper=np.asarray(header[:3]);hm=header[3];nodes=int(np.prod(upper+1))
            with (case/('outdat/rhs.gradients'+suffix)).open('rb') as stream:
                gh=np.fromfile(stream,np.int32,5);gradient=np.fromfile(stream,np.float64,nodes*9)
            with (case/('outdat/rhs.sensor'+suffix)).open('rb') as stream:
                sh=np.fromfile(stream,np.int32,3);sensor=np.fromfile(stream,np.float64,nodes)
                mask=np.fromfile(stream,np.int8,nodes)
            with (case/('outdat/rhs.conv'+suffix)).open('rb') as stream:
                rh=np.fromfile(stream,np.int32,4);rhs=np.fromfile(stream,np.float64)
            if tuple(gh[:3])!=tuple(upper) or tuple(sh)!=tuple(upper) or tuple(rh)!=(*upper,5):
                raise AssertionError('corrected reference layout')
            points=np.indices(q.shape[:3])-hm
            indices=tuple(points[a]%upper[a] if a==2 else np.clip(points[a],0,upper[a]) for a in range(3))
            metric=np.zeros((*q.shape[:3],3,3),order='F');physical=tuple(slice(hm,hm+n+1) for n in upper)
            with h5py.File(case/'outdat/geometry.h5') as geometry:
                jacobian=geometry['jacob'][...].transpose(2,1,0);jac=jacobian[indices];jac[physical]=jacobian
                for a in range(3):
                    for b in range(3):
                        values=geometry[f'dxi{a+1}{b+1}'][...].transpose(2,1,0)
                        metric[...,a,b]=values[indices];metric[physical+(a,b)]=values
            with (root/f'mp7ld_flat_{backend}.rk{stage:02d}.bin').open('xb') as target:
                np.asarray([*upper,hm,5,1,0,0,1],np.int32).tofile(target)
                np.asarray([1.4,2.,.1,.001],np.float64).tofile(target)
                for values in (q,p,metric,jac,gradient,sensor,mask,rhs):values.ravel(order='F').tofile(target)
    (root/'reference.json').write_text(json.dumps(dict(source=str(args.reference.resolve()),
        scope='Frozen-gradient convection and byte-mask consumer only; compact gradients/viscosity are not replaced by explicit derivatives'),indent=2)+'\n')
    for path in root.rglob('*'):path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


if __name__=='__main__':main()
