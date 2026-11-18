"""Run the actual C++ ordered/expanded repair against frozen old implementations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('binary','reference','output','mpiexec'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--mode',choices=['ordered','expanded','no-donor','reconciled'],default='expanded')
    parser.add_argument('--transport',choices=['pinned','device'],default='pinned')
    parser.add_argument('--layouts',nargs='+',default=['np1','np2_x','np2_y','np2_z'])
    parser.add_argument('--sanitizer',choices=['memcheck','initcheck','racecheck'])
    parser.add_argument('--precision',choices=['fp64','fp32'],default='fp64')
    args=parser.parse_args();meta=json.loads((args.reference/'reference.json').read_text())
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
    report=dict(binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),mode=args.mode,
        precision=args.precision,transport=args.transport,sanitizer=args.sanitizer,runs={})
    for name in args.layouts:
        item=meta['layouts'][name];ranks=item['ranks'];case=root/name;case.mkdir()
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(ranks)]
        if args.sanitizer:
            command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer','--tool',args.sanitizer,
                '--error-exitcode','99','--log-file',str(case/'sanitizer.%p.log')]
            if args.sanitizer=='memcheck':command+=['--leak-check','full']
        command += [str(args.binary.resolve()),str((args.reference/name).resolve()),args.mode,args.transport]
        with (case/'run.log').open('w') as log:
            subprocess.run(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        text=(case/'run.log').read_text()
        label='CPP_'+args.mode.upper().replace('-','_')+'_REPAIR_PASS'
        match=re.search(label+r' ranks=(\d+) local_cases=(\d+) max_scaled=\s*(\S+)',text)
        wanted=8 if args.mode=='ordered' else 2*item['cases']
        if args.mode=='no-donor':wanted=2
        if match is None or int(match[1])!=ranks or int(match[2])!=wanted:raise AssertionError('missing actual repair comparisons')
        result=dict(ranks=ranks,local_cases=wanted,max_scaled=float(match[3]))
        width=re.search(r'^CPP_REPAIR_REAL_BYTES=(\d+)$',text,re.M)
        if width is None or int(width[1])!=(4 if args.precision=='fp32' else 8):raise AssertionError('repair packet/scalar storage width')
        if report['precision']=='fp64' and result['max_scaled']>2e-10:raise AssertionError('FP64 repair difference')
        if args.mode=='expanded':
            actual={int(n):list(map(int,(rounds,total,radius,single))) for n,rounds,total,radius,single in
                re.findall(r'CPP_CRASHFIX_EXPANDED_CASE\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)',text)}
            expected={int(n):list(map(int,(rounds,total,radius,single))) for n,rounds,total,radius,single in
                re.findall(r'ASTR_GPU_CRASHFIX_EXPANDED\s+0\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)',
                    (args.reference/name/'run.log').read_text())}
            for n,values in expected.items():
                if actual.get(n)!=values:raise AssertionError(('repair traversal/progress differs',name,n,actual.get(n),values))
            result['expanded_traversal']=actual
        if args.sanitizer:
            logs=list(case.glob('sanitizer.*.log'))
            if len(logs)!=ranks:raise AssertionError('missing rank sanitizer log')
            for path in logs:
                data=path.read_text()
                expected='0 hazards displayed' if args.sanitizer=='racecheck' else 'ERROR SUMMARY: 0 errors'
                if expected not in data:raise AssertionError(path)
                if args.sanitizer=='memcheck' and 'LEAK SUMMARY: 0 bytes leaked' not in data:raise AssertionError(path)
        report['runs'][name]=result
        (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
