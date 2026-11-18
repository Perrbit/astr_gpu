"""Replay existing native CURVE primitives/metrics through the device gradient ABI."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np

from run_cpp_multi_tgv_gate import raw_field
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from run_native_precision_curve_gate import read_gradients
from run_native_precision_tgv_gate import component_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('binary','reference','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    shutil.copy2(args.binary,root/'tested_probe');shutil.copy2(__file__,root/'runner.py')
    meta=json.loads((args.reference/'reference.json').read_text())
    partitions=np.loadtxt(args.reference/'cpu/datin/parallel.info',skiprows=3,dtype=int,ndmin=2)
    report=dict(precision='fp32',mapping=meta['mapping'],topology=meta['topology'],packets={},
        binary_sha256=hashlib.sha256((root/'tested_probe').read_bytes()).hexdigest())
    for backend in ('cpu','gpu'):
        source=args.reference/backend
        for row in partitions:
            rank,_,_,_,im,jm,km,i0,j0,k0=map(int,row)
            low=[0,0,0];high=[0,0,0]
            if meta['mapping']=='y-wavy':low[1]=int(j0==0);high[1]=int(j0+jm==32)
            for step,stage in ((0,1),(3,3)):
                suffix=f'step{step:08d}.rk{stage:02d}.rank{rank:08d}.bin'
                header,p=raw_field(source/('state.pre_rhs_primitives.'+suffix),primitive=True,precision='fp32')
                if tuple(header[:3])!=(im,jm,km) or p.dtype!=np.float32:raise AssertionError('native CURVE primitive packet')
                h=int(header[3]);metric=np.zeros(p.shape[:3]+(3,3),dtype=np.float32,order='F')
                region=(slice(h,h+im+1),slice(h,h+jm+1),slice(h,h+km+1))
                global_region=(slice(i0,i0+im+1),slice(j0,j0+jm+1),slice(k0,k0+km+1))
                with h5py.File(source/'geometry.h5') as geometry:
                    for a in range(3):
                        for b in range(3):
                            if geometry[f'dxi{a+1}{b+1}'].dtype!=np.float32:raise AssertionError('native metric kind')
                            metric[region+(a,b)]=geometry[f'dxi{a+1}{b+1}'][...].transpose(2,1,0)[global_region]
                label=f'{backend}.rank{rank}.step{step}.rk{stage}'
                packet=root/(label+'.input.bin')
                with packet.open('xb') as stream:
                    np.asarray([im,jm,km,h,*low,*high,4],np.int32).tofile(stream)
                    p.ravel(order='F').tofile(stream);metric.ravel(order='F').tofile(stream)
                with (root/(label+'.log')).open('w') as log:
                    subprocess.run([str(root/'tested_probe'),str(packet),str(root/label)],
                        stdout=log,stderr=subprocess.STDOUT,check=True)
                outputs=[]
                for debug in (0,1):
                    with (root/(label+f'.{debug}.bin')).open('rb') as stream:
                        size=np.fromfile(stream,np.int32,4);data=np.fromfile(stream,np.float32)
                    if tuple(size)!=(im,jm,km,4) or data.size!=(im+1)*(jm+1)*(km+1)*12:
                        raise AssertionError('native gradient output width/shape')
                    if not np.isfinite(data).all():raise AssertionError('nonfinite native gradient')
                    outputs.append(data.reshape((im+1,jm+1,km+1,12),order='F'))
                if outputs[0].tobytes()!=outputs[1].tobytes():raise AssertionError('gradient synchronization changed words')
                original,_=read_gradients(source/('state.gradients.'+suffix),'fp32')
                result=dict(original=component_report(original,outputs[0]),physical_low=low,physical_high=high,
                    input_sha256=hashlib.sha256(packet.read_bytes()).hexdigest(),sync_words_equal=True)
                if backend=='gpu':
                    same_state_cpu,_=read_gradients(source/('state.gradient_reference.'+suffix),'fp32')
                    result['same_state_cpu']=component_report(same_state_cpu,outputs[0])
                text=(root/(label+'.log')).read_text()
                for debug in (0,1):
                    if f'CPP_NATIVE_CURVE_GRADIENT_PASS 4 {debug}' not in text:raise AssertionError('gradient case completion')
                result['affine_records']=[line for line in text.splitlines() if line.startswith('CPP_NATIVE_CURVE_AFFINE')]
                report['packets'][label]=result
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(packets=len(report['packets']),maximum_scaled=max(row['max_scaled']
        for p in report['packets'].values() for row in p['original'])),indent=2))


if __name__=='__main__':main()
