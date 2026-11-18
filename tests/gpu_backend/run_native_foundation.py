"""Build-precision state and real MPI halo probes; no flow advancement."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('bin-directory','mpiexec','sanitizer','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--precision',choices=('fp32','fp64'),required=True)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    width=4 if args.precision=='fp32' else 8
    binaries={name:args.bin_directory.resolve()/f'astr_backend_{name}' for name in ('mb1','halo')}
    shutil.copyfile(__file__,root/'runner.py')
    for name,path in binaries.items():shutil.copyfile(path,root/f'tested_{name}')
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    plans=(('state_np1','mb1',1,0,'pinned','periodic',None),
        ('state_np2_memcheck','mb1',2,0,'pinned','periodic','memcheck'),
        ('halo_np1','halo',1,0,'pinned','periodic',None),
        ('halo_np2_x_pinned','halo',2,0,'pinned','periodic',None),
        ('halo_np2_y_device','halo',2,1,'device','periodic',None),
        ('halo_np2_z_physical_initcheck','halo',2,2,'pinned','physical','initcheck'),
        ('halo_np2_x_physical_device','halo',2,0,'device','physical',None))
    records=[]
    for name,kind,ranks,axis,transport,layout,tool in plans:
        case=root/name;case.mkdir()
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(ranks)]
        if tool:
            command += [str(args.sanitizer.resolve()),'--tool',tool,'--error-exitcode','99',
                '--log-file',str(case/'sanitizer.%p.log')]
            if tool=='memcheck':command += ['--leak-check','full']
        command += [str(binaries[kind])]
        if kind=='halo':command += [str(axis),transport,layout]
        execution=env if transport=='pinned' else env|dict(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
        with (case/'run.log').open('w') as log:
            subprocess.run(command,env=execution,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        text=(case/'run.log').read_text();wire=[]
        if kind=='mb1':
            rows=re.findall(r'MB1 rank=(\d+) halo=(\d+) debug=(\d+) element_bytes=(\d+) exact layout/copy/view PASS',text)
            if len(rows)!=2*ranks or any(int(w)!=width for _,_,_,w in rows):
                raise AssertionError('native state layout/type/copy evidence')
        else:
            rows=re.findall(r'HALO PASS rank/axis/hm/debug=(\d+) (\d+) (\d+) (\d+) ',text)
            if len(rows)!=4*ranks:raise AssertionError('native halo phase/lifecycle evidence')
            entries=re.findall(r'HALO_WIRE rank=(\d+) element_bytes=(\d+) calls_send_recv_bytes_send_recv_badtypes=\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)',text)
            if len(entries)!=ranks:raise AssertionError('missing PMPI wire observations')
            extents=(9,7,6);area=1
            for a in range(3):
                if a!=axis:area*=extents[a]+1
            expected_bytes=360*area*width if layout=='periodic' else 180*area*width
            for rank,w,send,receive,sent,received,bad in entries:
                expected=(0,0,0,0) if ranks==1 else (16,16,expected_bytes,expected_bytes)
                if int(w)!=width or int(bad)!=0 or tuple(map(int,(send,receive,sent,received)))!=expected:
                    raise AssertionError('actual MPI type/count/bytes do not match native precision')
                wire.append(dict(rank=int(rank),element_bytes=int(w),send_calls=int(send),recv_calls=int(receive),
                    sent_bytes=int(sent),received_bytes=int(received),bad_types=int(bad)))
        if tool:
            paths=list(case.glob('sanitizer.*.log'))
            if len(paths)!=ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in paths):
                raise AssertionError('native foundation memory-safety evidence')
            if tool=='memcheck' and any('LEAK SUMMARY: 0 bytes leaked' not in p.read_text() for p in paths):
                raise AssertionError('native state allocations leaked')
        record=dict(name=name,kind=kind,ranks=ranks,axis=axis,transport=transport,layout=layout,tool=tool,
            cases=len(rows),wire=wire,command=command)
        records.append(record);print(json.dumps(record),flush=True)
    report=dict(precision=args.precision,element_bytes=width,records=records,
        binaries={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in binaries.items()},
        scope='State/copy and shared-node/halo arithmetic/wire foundation only; no native FP32 CFD/RK/CFL/statistics/output claim')
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
