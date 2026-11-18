"""Copy frozen CPU/old mixed MP7 phases into the existing upwind probe layout."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from run_cpp_multi_tgv_gate import raw_field


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--step',type=int,default=0)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    (root/'steps').mkdir()
    for backend,source_name in (('cpu','cpu'),('gpu','old_gpu_mixed')):
        source=args.reference/source_name/'outdat'
        h,base=raw_field(source/f'rhs.pre_rhs.step{args.step:08d}.rk01.rank00000000.bin')
        upper=np.array(h[:3]);halo=h[3];physical=tuple(slice(halo,halo+n+1) for n in upper)
        indices=np.indices(base.shape[:3])-halo;wrapped=tuple(indices[a]%upper[a] for a in range(3))
        with h5py.File(source/'geometry.h5') as geometry:
            jac=geometry['jacob'][...].transpose(2,1,0)
            metric=np.stack([geometry[f'dxi{a+1}{b+1}'][...].transpose(2,1,0)
                for b in range(3) for a in range(3)],axis=-1).reshape((*jac.shape,3,3),order='F')
        metric_halo=metric[wrapped].copy();metric_halo[physical]=metric
        jac_halo=jac[wrapped].copy();jac_halo[physical]=jac
        nodes=int(np.prod(upper+1))
        for stage in (1,2,3):
            suffix=f'.step{args.step:08d}.rk{stage:02d}.rank00000000.bin'
            _,q=raw_field(source/('rhs.pre_rhs'+suffix))
            _,p=raw_field(source/('rhs.pre_rhs_primitives'+suffix),primitive=True)
            _,post=raw_field(source/('rhs.post_update'+suffix))
            with (source/('rhs.gradients'+suffix)).open('rb') as stream:
                gh=np.fromfile(stream,np.int32,5)
                gradient=np.fromfile(stream,np.float64,nodes*9)
            if not np.array_equal(gh[:3],upper):raise AssertionError('old MP1 gradient layout')
            def read_rhs(name):
                with (source/('rhs.'+name+suffix)).open('rb') as stream:
                    header=np.fromfile(stream,np.int32,4)
                    values=np.fromfile(stream,np.float64).reshape((*upper+1,5),order='F')
                if not np.array_equal(header,[*upper,5]):raise AssertionError('old MP1 RHS layout')
                return values
            convection,full=read_rhs('conv'),read_rhs('full')
            name=f'mp7_{backend}.rk{stage:02d}.bin'
            with (root/name).open('xb') as stream:
                np.array([*upper,halo,3,0,1,1,1],np.int32).tofile(stream)
                np.array([1.4,.1,.3,.001],np.float64).tofile(stream)
                for values in (q,p,metric_halo,jac_halo,gradient,np.zeros(nodes),
                    np.zeros(nodes,dtype=np.int8),convection):
                    values.ravel(order='F').tofile(stream)
            with (root/'steps'/name).open('xb') as stream:
                np.array([*upper,halo],np.int32).tofile(stream)
                for values in (base,full,post[physical]):values.ravel(order='F').tofile(stream)
    (root/'source.json').write_text(json.dumps(dict(reference=str(args.reference.resolve()),
        step=args.step,scope='Original CPU FP64 and old-GPU MP1 flux, MP7, 3 phases, dt=1e-4; only data copying'),indent=2)+'\n')
    for path in root.rglob('*'):path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


if __name__=='__main__':main()
