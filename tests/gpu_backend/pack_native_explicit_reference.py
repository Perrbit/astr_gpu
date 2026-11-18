"""Partition frozen native-FP32 records; no numerical advancement."""
import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from compare_q_validation_snapshots import read_q_snapshot


def qrecord(path):
    record=read_q_snapshot(path,precision='fp32');im,jm,km,halo,components=record.header
    return record.values.reshape((im+2*halo+1,jm+2*halo+1,km+2*halo+1,components),order='F')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    upper=np.array([16,16,16]);halo=5;count=int(np.prod(upper+1))
    for source in ('cpu','gpu'):
      for filtering in (0,1):
        case=args.reference/f'filter{filtering}_{source}';raw=case/'outdat'
        with h5py.File(raw/'geometry.h5') as data:
            jac=data['jacob'][...].transpose(2,1,0)
            metric=np.stack([data[f'dxi{a}{b}'][...].transpose(2,1,0) for b in (1,2,3) for a in (1,2,3)],axis=-1).reshape((*jac.shape,3,3),order='F')
        initial=qrecord(raw/'rhs.initialized.step00000000.rk01.rank00000000.bin')
        base=qrecord(raw/'rhs.pre_rhs.step00000000.rk01.rank00000000.bin')
        indices=np.indices(initial.shape[:3])-halo;wrapped=tuple(indices[a]%upper[a] for a in range(3))
        jhalo=jac[wrapped];mhalo=metric[wrapped]
        physical=tuple(slice(halo,halo+n+1) for n in upper);jhalo[physical]=jac;mhalo[physical]=metric
        for label,topology in (('np1',(1,1,1)),('np2_x',(2,1,1)),('np2_y',(1,2,1)),('np2_z',(1,1,2))):
          destination=out/source/f'filter{filtering}'/label;destination.mkdir(parents=True)
          local=upper//topology
          for rank in range(int(np.prod(topology))):
            coordinate=np.array([rank%topology[0],rank//topology[0]%topology[1],rank//(topology[0]*topology[1])])
            origin=coordinate*local
            volume=tuple(slice(int(o),int(o+n+2*halo+1)) for o,n in zip(origin,local))
            body=tuple(slice(int(o),int(o+n+1)) for o,n in zip(origin,local))
            qbody=tuple(slice(int(o+halo),int(o+halo+n+1)) for o,n in zip(origin,local))
            with (destination/f'initial.rank{rank:08d}.bin').open('xb') as f:initial[volume].ravel(order='F').tofile(f)
            for stage in (1,2,3):
                suffix=f'.step00000000.rk{stage:02d}.rank00000000.bin'
                q=qrecord(raw/('rhs.pre_rhs'+suffix));following=qrecord(raw/('rhs.post_update'+suffix))
                with (raw/('rhs.pre_rhs_primitives'+suffix)).open('rb') as f:
                    header=np.fromfile(f,np.int32,4);p=np.fromfile(f,np.float32).reshape((*initial.shape[:3],6),order='F')
                    if tuple(header)!=(16,16,16,5):raise AssertionError('native primitive dimensions')
                with (raw/('rhs.gradients'+suffix)).open('rb') as f:
                    header=np.fromfile(f,np.int32,5)
                    grad=np.fromfile(f,np.float32,count*9).reshape((*upper+1,3,3),order='F')
                    thermal=np.fromfile(f,np.float32,count*3).reshape((*upper+1,3),order='F')
                    if tuple(header[:3])!=(16,16,16):raise AssertionError('native gradient dimensions')
                rhs=[]
                for name in ('conv','full'):
                    with (raw/('rhs.'+name+suffix)).open('rb') as f:
                        h=np.fromfile(f,np.int32,4);rhs.append(np.fromfile(f,np.float32).reshape((*upper+1,5),order='F'))
                        if tuple(h)!=(16,16,16,5):raise AssertionError('native RHS dimensions')
                with (destination/f'stage{stage}.rank{rank:08d}.bin').open('xb') as f:
                    np.array([*local,halo,*origin,stage,4],np.int32).tofile(f)
                    for array in (base[volume],q[volume],p[volume],mhalo[volume],jhalo[volume],grad[body],thermal[body],rhs[0][body],rhs[1][body],following[qbody]):
                        np.asarray(array,dtype=np.float32).ravel(order='F').tofile(f)
    (out/'source.json').write_text(json.dumps(dict(reference=str(args.reference.resolve()),precision='fp32',
        scope='Original CPU/GPU, filter0/1, three same-state RK stages and initialized q for coupled runs'),indent=2)+'\n')
    for path in out.rglob('*'):path.chmod(0o555 if path.is_dir() else 0o444)
    out.chmod(0o555)


if __name__=='__main__':main()
