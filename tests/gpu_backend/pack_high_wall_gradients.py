"""Pack immutable high-wall stage data for the existing device-gradient probe."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from run_cpp_multi_tgv_gate import raw_field


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);records={}
    for name in ('cpu','gpu'):
        source=args.reference/name;target=out/(name+'_y');target.mkdir()
        with h5py.File(source/'outdat/new/resources/geometry.h5') as geometry:
            physical={f'q{c:04}':geometry[f'q{c:04}'][...].transpose(2,1,0) for c in range(5,14)}
        for stage in (1,2,3):
            suffix=f'.step00000000.rk{stage:02}.rank00000000.bin'
            header,p=raw_field(source/'outdat'/('rhs.pre_rhs_primitives'+suffix),primitive=True)
            dims=np.array(header[:3],dtype=int);h=int(header[3]);metric=np.zeros((*p.shape[:3],3,3),order='F')
            for b in range(3):
                for a in range(3):
                    metric[tuple(slice(h,h+n+1) for n in dims)+(a,b)]=physical[f'q{5+a+3*b:04}']
            nodes=int(np.prod(dims+1))
            with (source/'outdat'/('rhs.gradients'+suffix)).open('rb') as f:
                gh=np.fromfile(f,np.int32,5);values=np.fromfile(f,np.float64,nodes*12)
            if tuple(gh[:3])!=tuple(dims) or values.size!=nodes*12:raise AssertionError('high wall gradient record')
            with (target/f'gradient_input.rk{stage:02}.bin').open('xb') as f:
                np.array(header,np.int32).tofile(f);p.ravel(order='F').tofile(f);metric.ravel(order='F').tofile(f);values.tofile(f)
        records[name]=dict(source=str(source.resolve()),stages=[1,2,3])
    (out/'reference.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2))
    for path in out.rglob('*'):path.chmod(0o555 if path.is_dir() else 0o444)
    out.chmod(0o555)


if __name__=='__main__':main()
