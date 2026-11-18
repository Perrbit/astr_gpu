"""Reconciled marker -> actual MP7-LD flux on one shared interface per axis."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np


def reference(path):
    with path.open('rb') as stream:
        header=np.fromfile(stream,np.int32,3)
        locations=np.fromfile(stream,np.int32,6).reshape((3,2),order='F')
        constants=np.fromfile(stream,np.float64,3)
        stencil=np.fromfile(stream,np.float64,176).reshape((8,11,2),order='F')
        flux=np.fromfile(stream,np.float64,10).reshape((5,2),order='F')
        if stream.read(1):raise AssertionError('shared flux reference trailing data')
    if not np.all(header[1:]==1) or stencil[:,:,0].tobytes()!=stencil[:,:,1].tobytes():
        raise AssertionError('reference interface is not the canonical same-state contract')
    return header,locations,constants,flux


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('binary','references','packets','mpiexec','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--transport',choices=('pinned','device'),default='pinned')
    parser.add_argument('--case',choices=tuple(a+'_'+s for a in 'xyz' for s in ('shared','halo')))
    parser.add_argument('--sanitizer',choices=('memcheck','initcheck'))
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(args.binary.resolve(),out/'tested_probe');shutil.copyfile(__file__,out/'runner.py')
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
    report=dict(binary_sha256=hashlib.sha256((out/'tested_probe').read_bytes()).hexdigest(),
        transport=args.transport,sanitizer=args.sanitizer,cases={})
    for axis in 'xyz':
      for seed in ('shared','halo'):
        name=axis+'_'+seed
        if args.case and args.case!=name:continue
        case=out/name;case.mkdir();prefix='' if axis=='x' else axis+'_'
        cpu_path=args.references.resolve()/(prefix+'cpu_'+seed+'.shared_flux.bin')
        gpu_path=args.references.resolve()/(prefix+'gpu_'+seed+'.shared_flux.bin')
        cpu=reference(cpu_path);gpu=reference(gpu_path)
        if not np.array_equal(cpu[0],gpu[0]) or not np.array_equal(cpu[1],gpu[1]) or not np.array_equal(cpu[2],gpu[2]):
            raise AssertionError('CPU/GPU critical interface identity differs')
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n','2']
        if args.sanitizer:
            command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer','--tool',args.sanitizer,
                '--error-exitcode','99','--log-file',str(case/'sanitizer.%p.log')]
            if args.sanitizer=='memcheck':command+=['--leak-check','full']
        command += [str(out/'tested_probe'),str(args.packets.resolve()/('np2_'+axis)),str(cpu_path),seed,args.transport]
        with (case/'run.log').open('w') as log:subprocess.run(command,cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        text=(case/'run.log').read_text()
        values=re.findall(r'CPP_CRITICAL_SHARED_FLUX_VALUES\s+(\d+)\s+(\d+)\s+((?:[\d.Ee+\-]+\s*){5})',text)
        if len(values)!=4 or {(int(r),int(d)) for r,d,_ in values}!={(r,d) for r in range(2) for d in range(2)}:
            raise AssertionError('missing actual critical interface flux values')
        errors={key:np.zeros(5) for key in ('cpu','gpu')}
        for rank,debug,row in values:
            observed=np.array(list(map(float,row.split())))
            if not np.isfinite(observed).all():raise AssertionError('nonfinite critical flux')
            for key,ref in [('cpu',cpu),('gpu',gpu)]:
                errors[key]=np.maximum(errors[key],abs(observed-ref[3][:,int(rank)])/np.maximum(1.,np.max(abs(ref[3]),axis=1)))
        if any(max(v)>2e-10 for v in errors.values()):raise AssertionError('shared critical flux exceeds original gate')
        if args.sanitizer:
            logs=list(case.glob('sanitizer.*.log'))
            if len(logs)!=2 or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in logs):raise AssertionError('critical flux tool')
            if args.sanitizer=='memcheck' and any('LEAK SUMMARY: 0 bytes leaked' not in p.read_text() for p in logs):
                raise AssertionError('critical flux leaked')
        report['cases'][name]=dict(command=command,errors={k:v.tolist() for k,v in errors.items()},
            reference_cpu=str(cpu_path),reference_gpu=str(gpu_path),full_q_unchanged=True)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(name,'same-interface flux agrees with CPU/GPU')


if __name__=='__main__':main()
