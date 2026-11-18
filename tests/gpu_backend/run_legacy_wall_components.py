"""Device legacy wall components: analytic MPI and immutable actual wall CSVs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np


def restored_fields(path,halo):
    with h5py.File(path) as f:
        names=sorted(k for k in f if k.startswith('q') and len(k)==5)
        physical=np.stack([f[k][...].transpose(2,1,0) for k in names],axis=-1)
        shape=tuple(n+2*halo for n in physical.shape[:3])
        result=np.zeros((*shape,len(names)),dtype=physical.dtype,order='F')
        interior=tuple(slice(halo,halo+n) for n in physical.shape[:3])
        result[interior]=physical
        mask=np.ones(shape,bool,order='F');mask[interior]=False
        indices=np.flatnonzero(mask.ravel(order='F'))
        extras=f['rank_extras'][...]
        if extras.size!=indices.size*len(names):raise AssertionError('single-rank extras layout')
        for c in range(len(names)):
            values=result[...,c].ravel(order='F').copy()
            values[indices]=extras[c*indices.size:(c+1)*indices.size]
            result[...,c]=values.reshape(shape,order='F')
    return result


def packet(case,target):
    halo=5
    q=restored_fields(case/'outdat/new/checkpoints/step000000000004/state.h5',halo)[...,:5]
    geometry=restored_fields(case/'outdat/new/resources/geometry.h5',halo)
    R=q.dtype.type;cells=np.asarray(q.shape[:3])-2*halo-1
    if tuple(cells)!=(32,32,32):raise AssertionError('frozen wall reference shape')
    metric=geometry[...,4:13].reshape((*geometry.shape[:3],3,3),order='F')
    strip=np.zeros((33,33,2,9),dtype=R,order='F')
    coeff=[R(.75),R(-.15),R(1)/R(60)]
    for w,j in enumerate((0,32)):
        for k in range(33):
            for i in range(33):
                I,J,K=i+halo,j+halo,k+halo
                first=np.zeros(3,dtype=R);second=first.copy()
                for s,c in enumerate(coeff,1):
                    first+=c*(geometry[I+s,J,K,:3]-geometry[I-s,J,K,:3])
                    second+=c*(geometry[I,J,K+s,:3]-geometry[I,J,K-s,:3])
                normal=np.cross(first,second);normal/=np.sqrt(np.sum(normal*normal,dtype=R))
                inward=geometry[I,J+(1 if w==0 else -1),K,:3]-geometry[I,J,K,:3]
                if np.sum(normal*inward,dtype=R)<0:normal=-normal
                tangent=np.asarray([1,0,0],dtype=R)-normal[0]*normal
                tangent/=np.sqrt(np.sum(tangent*tangent,dtype=R))
                strip[i,k,w,:]=np.r_[geometry[I,J,K,:3],normal,tangent]
    gamma=R(1.4);mach=R(.1);temp=R(110.3)/R(273.15)
    material=np.asarray([gamma*(mach*mach),(gamma-R(1))*(mach*mach),R(1)/(gamma-R(1)),
        R(1600),R(.72),temp,R(1)+temp],dtype=R)
    with target.open('xb') as f:
        np.asarray([*cells,halo,q.itemsize,2,0,3,1],np.int32).tofile(f)
        for values in (material,q,metric,strip):values.ravel(order='F').tofile(f)
    return strip,q.dtype


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary64',type=Path,required=True);p.add_argument('--binary32',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mpiexec',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'binaries').mkdir()
    root=Path(__file__).resolve().parents[2]
    source_paths=['src_gpu/insitu_wall.cu','src_gpu/insitu_wall.h','src_gpu/insitu_wall.F90',
        'tests/gpu_backend/insitu_wall_probe.F90','tests/gpu_backend/run_legacy_wall_components.py']
    report=dict(analytic=[],frozen=[],binaries={},source_sha256={
        path:hashlib.sha256((root/path).read_bytes()).hexdigest() for path in source_paths})
    for kind,binary in (('fp64',a.binary64),('fp32',a.binary32)):
        original=binary.resolve();binary=out/'binaries'/kind
        shutil.copy2(original,binary);binary.chmod(0o555)
        report['binaries'][kind]=dict(path=str(binary),source=str(original),sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
        def run(name,ranks,args,mode='pinned'):
            env=os.environ|dict(OMPI_MCA_pml='ucx' if mode=='device-aware' else 'ob1',
                OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_osc='pt2pt',
                OMPI_MCA_opal_cuda_support='1' if mode=='device-aware' else '0')
            command=[str(a.mpiexec),'-n',str(ranks),str(binary),*args]
            with (out/(name+'.log')).open('x') as log:
                result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise AssertionError((name,'component failed',result.returncode))
            text=(out/(name+'.log')).read_text()
            if 'CPP_WALL_SAMPLE_PASS' not in text:raise AssertionError((name,'missing result'))
            return dict(name=name,command=command,log=name+'.log',maximum_scaled=float(text.split()[-1]))
        for dims,mode,closure in [('1,1,1','pinned','legacy'),('1,1,1','pinned','sixth'),
            ('1,1,1','pinned','curve'),('2,1,1','pinned','curve'),('1,2,1','pinned','sixth'),
            ('1,1,2','device-aware','curve')]:
            ranks=int(np.prod([int(v) for v in dims.split(',')]))
            item=run(kind+'_'+dims.replace(',','x')+'_'+mode+'_'+closure,ranks,[dims,mode,closure],mode)
            report['analytic'].append(item);print(item['name'],item['maximum_scaled'],flush=True)
        for implementation in ('cpu','gpu'):
            name=kind+'_'+implementation+'_frozen'
            case=a.reference/(implementation+'_np1_'+kind+('_csv' if implementation=='gpu' else ''))
            strip,dtype=packet(case,out/(name+'.input'))
            item=run(name,1,['packet',str(out/(name+'.input')),str(out/name)])
            values=[np.fromfile(str(out/name)+f'.{debug}.bin',dtype=dtype).reshape((33,33,2,4),order='F') for debug in (0,1)]
            if values[0].tobytes()!=values[1].tobytes():raise AssertionError('sync mode changed wall words')
            differences={}
            for w,side in enumerate(('lower','upper')):
                for c,field in enumerate(('wall_pressure','wall_shear_x','wall_heat_into_gas')):
                    records=list((case/'outdat/new/wall_products'/f'{side}-{field}-instantaneous').glob('*/segment*/frame.step000000000004.csv'))
                    if len(records)!=1:raise AssertionError('frozen wall product identity')
                    with records[0].open() as source:
                        csv=np.genfromtxt([line for line in source if not line.startswith('#')],delimiter=',',names=True)
                    xyz=np.stack([csv[v] for v in ('x','y','z')],axis=-1)
                    if not np.array_equal(strip[:,4,w,:3],xyz.astype(dtype)):raise AssertionError('frozen wall coordinates')
                    actual=values[0][:,4,w,c].astype(np.float64);expected=csv['value']
                    absolute=float(np.max(abs(actual-expected)));scaled=absolute/max(1.,float(np.max(abs(expected))))
                    if kind=='fp64' and scaled>2e-10:raise AssertionError((name,side,field,scaled))
                    differences[side+'_'+field]=dict(max_absolute=absolute,scaled=scaled)
            item.update(source=str(case.resolve()),quantities=differences)
            report['frozen'].append(item);print(name,max(v['scaled'] for v in differences.values()),flush=True)
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('legacy wall components: numerical/transport/observation checks complete')


if __name__=='__main__':main()
