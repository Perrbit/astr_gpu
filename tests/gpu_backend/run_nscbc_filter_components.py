"""Original six-order outlet filter versus thin-plane C++ filter and real halo."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from run_nscbc_outlet_components import compile_reference,ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-build',type=Path);parser.add_argument('--gpu-build',type=Path)
    parser.add_argument('--reference',type=Path);parser.add_argument('--binary',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--mpiexec',type=Path,required=True)
    parser.add_argument('--ranks',type=int,choices=[1,2],default=1)
    parser.add_argument('--axis',choices=list('xyz'),default='x')
    parser.add_argument('--transport',choices=['pinned','device'],default='pinned')
    parser.add_argument('--sanitizer',choices=['memcheck','initcheck'])
    parser.add_argument('--top',action='store_true',help='upper-y x/z sequence, not x+ outlet')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
    launch=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks)]
    source=Path(__file__).with_name('nscbc_filter_probe.F90')
    report=dict(ranks=args.ranks,axis=args.axis,transport=args.transport,kind='top' if args.top else 'outlet')
    if args.reference is None:
        if args.cpu_build is None or args.gpu_build is None or args.ranks!=1:
            parser.error('freeze requires CPU/old-GPU builds and NP=1')
        for name,build,gpu in [('cpu',args.cpu_build,False),('gpu',args.gpu_build,True)]:
            binary,info=compile_reference(build,out/name,gpu,source,['-DTEST_TOP_FILTER'] if args.top else ());report[name]=info
            with (out/f'{name}.log').open('w') as log:
                subprocess.run(launch+[str(binary),str(out)],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        report['errors']=[]
        with (out/'cpu.bin').open('rb') as a,(out/'gpu.bin').open('rb') as b:
            ga=np.fromfile(a,np.int32,3);gb=np.fromfile(b,np.int32,3)
            if not np.array_equal(ga,gb):raise AssertionError('filter reference header')
            shape=tuple(ga+11)+(5,);count=int(np.prod(shape));h=5
            physical=tuple(slice(h,h+n+1) for n in ga)
            for fixture in range(2):
                for trial in range(6 if args.top else 3):
                    aq=np.fromfile(a,np.float64,count).reshape(shape,order='F')
                    af=np.fromfile(a,np.float64,count).reshape(shape,order='F')
                    bq=np.fromfile(b,np.float64,count).reshape(shape,order='F')
                    bf=np.fromfile(b,np.float64,count).reshape(shape,order='F')
                    input_error=0.
                    if args.top and trial%2==1:
                        if aq[physical].tobytes()!=previous_a[physical].tobytes() or \
                                bq[physical].tobytes()!=previous_b[physical].tobytes():
                            raise AssertionError('z phase is not the preceding x-filtered state')
                        input_error=float(np.max(abs(aq-bq)))
                        if not np.isfinite(aq).all() or not np.isfinite(bq).all() or input_error>2e-10:
                            raise AssertionError('computed z inputs exceed the existing field gate')
                    elif not np.array_equal(aq,bq):raise AssertionError('filter input differs')
                    errors=[float(np.max(abs(af[physical+(c,)]-bf[physical+(c,)]))/
                        max(1.,float(np.max(abs(af[physical+(c,)]))))) for c in range(5)]
                    if not np.isfinite(af).all() or not np.isfinite(bf).all() or max(errors)>2e-10:
                        raise AssertionError(f'filter reference differs: {errors}')
                    constant_error=0.
                    if trial==0:
                        constant_error=float(np.max(abs(af-aq)))
                        if constant_error>2e-10:raise AssertionError('constant filter reference')
                    report['errors'].append(dict(fixture=fixture,trial=trial,scaled=errors,
                        constant_error=constant_error,computed_input_error=input_error))
                    previous_a=af;previous_b=bf
            if a.read(1) or b.read(1):raise AssertionError('filter reference trailing data')
        shutil.copyfile(source,out/'probe_source.F90')
        report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT/'src/filter.F90',ROOT/'src_gpu/boundary_gpu.cuf',source]}
    else:
        if args.binary is None:parser.error('comparison requires --binary')
        command=launch
        if args.sanitizer:
            command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
                '--tool',args.sanitizer,'--error-exitcode','99','--log-file',str(out/'sanitizer.%p.log')]
            if args.sanitizer=='memcheck':command+=['--leak-check','full']
        command += [str(args.binary.resolve()),str(args.reference.resolve()),str('xyz'.index(args.axis)+1),args.transport]
        with (out/'run.log').open('w') as log:
            subprocess.run(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        marker='CPP_NSCBC_TOP_FILTER_PASS' if args.top else 'CPP_NSCBC_FILTER_PASS'
        rows=re.findall(marker+r'\s+(\d+)\s+(\d+)\s+(\S+)',(out/'run.log').read_text())
        if sorted(int(row[0]) for row in rows)!=list(range(args.ranks)):raise AssertionError('missing filter rank')
        if any(int(row[1])!=(48 if args.top else 24) for row in rows):raise AssertionError('missing filter cases')
        report['rows']=[dict(rank=int(r),cases=int(n),max_error=float(e)) for r,n,e in rows]
        if args.sanitizer:
            logs=list(out.glob('sanitizer.*.log'))
            if len(logs)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in logs):
                raise AssertionError('missing clean per-rank tool result')
        report['command']=command;report['binary_sha256']=hashlib.sha256(args.binary.read_bytes()).hexdigest()
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('cpu','gpu','command')},indent=2))
    if args.reference is None:
        for p in out.rglob('*'):p.chmod(0o555 if p.is_dir() or p.name=='probe' else 0o444)
        out.chmod(0o555)


if __name__=='__main__':main()
