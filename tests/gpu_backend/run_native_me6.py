"""Native ME6 private-edge arithmetic, MPI payload and decomposition checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np


def fields(directory, ranks, debug):
    result={}
    for rank in range(ranks):
        path=directory/f'rank{rank:08d}.debug{debug}.bin'
        with path.open('rb') as stream:
            header=np.fromfile(stream,np.int32,11)
            if len(header)!=11 or header[1]!=rank or header[3]!=debug or header[10]!=16:
                raise AssertionError('ME6 periodic record identity')
            dtype={4:np.float32,8:np.float64}[int(header[0])]
            shape=tuple(header[4:7]+1)+(5,)
            data=np.fromfile(stream,dtype,np.prod(shape))
            if data.size!=np.prod(shape) or stream.read(1) or not np.isfinite(data).all():
                raise AssertionError('ME6 periodic record payload')
        result[rank]=(header,data.reshape(shape,order='F'))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('binary','mpiexec','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--precision',choices=('fp32','fp64'),required=True)
    parser.add_argument('--transport',choices=('pinned','device-aware'),required=True)
    parser.add_argument('--cases',nargs='+',choices=('np1','np2x','np2y','np2z'),default=['np1','np2x','np2y','np2z'])
    parser.add_argument('--compare',type=Path,help='same precision/fixture records from another transport')
    parser.add_argument('--sanitizer',choices=('memcheck','initcheck'))
    parser.add_argument('--analyze-only',action='store_true',help='read existing completed records without launching MPI')
    args=parser.parse_args();root=args.output.resolve()
    if args.analyze_only:
        if (root/'tested_probe').read_bytes()!=args.binary.read_bytes():
            raise AssertionError('ME6 analysis needs the unchanged tested executable')
        shutil.copyfile(__file__,root/'analysis.py')
    else:
        root.mkdir(parents=True,exist_ok=False)
        shutil.copyfile(__file__,root/'runner.py');shutil.copy2(args.binary,root/'tested_probe')
    binary=(root/'tested_probe').resolve();width=4 if args.precision=='fp32' else 8
    environment=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    if args.transport=='device-aware':environment.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
    report=dict(precision=args.precision,transport=args.transport,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),cases={})
    observed={}
    for name in args.cases:
        ranks=1 if name=='np1' else 2
        axis=1 if ranks==1 else 'xyz'.index(name[-1])+1
        directory=root/name
        if not args.analyze_only:directory.mkdir()
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(ranks)]
        if args.sanitizer:
            command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
                '--tool',args.sanitizer,'--error-exitcode','99','--log-file',str(directory/'sanitizer.%p.log')]
            if args.sanitizer=='memcheck':command+=['--leak-check','full']
        command+=[str(binary),'periodic',str(axis),args.transport,str(directory)]
        if not args.analyze_only:
            with (directory/'run.log').open('w') as log:
                subprocess.run(command,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        text=(directory/'run.log').read_text()
        rows=re.findall(r'^CPP_ME6_PERIODIC\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.Ee+-]+)\s+((?:\d+\s*){4})$',text,re.M)
        wires=re.findall(r'^CPP_ME6_WIRE\s+(\d+)\s+(\d+)\s+((?:\d+\s*){5})$',text,re.M)
        if len(rows)!=2*ranks or len(wires)!=ranks or text.count('CPP_ME6_PERIODIC_PASS ')!=ranks:
            raise AssertionError('incomplete ME6 rank/synchronization observations')
        a,b=fields(directory,ranks,0),fields(directory,ranks,1)
        for rank in a:
            if a[rank][0][0]!=width or not np.array_equal(a[rank][0][:3],b[rank][0][:3]) or a[rank][1].tobytes()!=b[rank][1].tobytes():
                raise AssertionError('ME6 type or synchronization words')
        capacity=0
        upper=a[0][0][4:7]
        if ranks>1 and axis==2:capacity=int(4*5*(upper[0]+11)*(upper[2]+1))
        if ranks>1 and axis==3:capacity=int(4*5*(upper[0]+11)*(upper[1]+11))
        workspace=(2*int(np.prod(upper+11))+162+4*capacity)*width
        for rank,direction,debug,error,values in rows:
            metrics=list(map(int,values.split()))
            expected=[workspace,4*capacity*width if args.transport=='pinned' else 0,
                2*capacity*width,4*capacity*width if args.transport=='pinned' else 0]
            if int(direction)!=axis or not np.isfinite(float(error)) or metrics!=expected:
                raise AssertionError(('ME6 private workspace/traffic observation',name,metrics,expected))
            if width==8 and float(error)>2e-10:raise AssertionError('ME6 FP64 field criterion')
        wire=[]
        for rank,actual_width,values in wires:
            actual=list(map(int,values.split()))
            expected=[4 if capacity else 0,4 if capacity else 0,4*capacity*width,4*capacity*width,0]
            if int(actual_width)!=width or actual!=expected:
                raise AssertionError(('ME6 actual packet MPI type/count/bytes',name,actual,expected))
            wire.append(dict(rank=int(rank),counters=actual))
        if args.compare:
            other=fields(args.compare/name,ranks,0)
            for rank in a:
                if not np.array_equal(a[rank][0],other[rank][0]) or a[rank][1].tobytes()!=other[rank][1].tobytes():
                    raise AssertionError('ME6 transport changed fields')
        if args.sanitizer:
            paths=list(directory.glob('sanitizer.*.log'))
            if len(paths)!=ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in paths):
                raise AssertionError('ME6 sanitizer records')
            if args.sanitizer=='memcheck' and any('LEAK SUMMARY: 0 bytes leaked' not in p.read_text() for p in paths):
                raise AssertionError('ME6 allocation leak')
        observed[name]=a
        report['cases'][name]=dict(command=command,field_scaled_max=max(float(row[3]) for row in rows),
            private_device_bytes=workspace,private_host_bytes=4*capacity*width if args.transport=='pinned' else 0,
            wire=wire,halo_all_words_equal=True,sync_words_equal=True,transport_words_equal=args.compare is not None,
            sanitizer=args.sanitizer)
        print(name,json.dumps(report['cases'][name]),flush=True)
    if 'np1' in observed:
        full=observed['np1'][0][1]
        for name,data in observed.items():
            for header,value in data.values():
                body=tuple(slice(int(o),int(o+n+1)) for o,n in zip(header[7:10],header[4:7]))
                if value.tobytes(order='F')!=full[body].tobytes(order='F'):
                    raise AssertionError(('ME6 decomposition words',name))
            report['cases'][name]['np1_words_equal']=True
    report['scope']='ME6 mapped/private-edge component; public face halos are exact manufactured inputs, no CFD main/restart/physical validation claim'
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
