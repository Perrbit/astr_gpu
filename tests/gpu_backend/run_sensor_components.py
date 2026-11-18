"""Run frozen and analytic sensor checks on the existing MPI topology bridge."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def main():
    parser=argparse.ArgumentParser()
    for name in ('binary','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--analytic-only',action='store_true')
    parser.add_argument('--precision',choices=['fp64','fp32'],default='fp64')
    parser.add_argument('--compare',type=Path,help='matching topology/precision observations from another transport')
    parser.add_argument('--ranks',type=int,choices=[1,2],default=1)
    parser.add_argument('--axis',choices=['x','y','z'],default='x')
    parser.add_argument('--transport',choices=['pinned','device'],default='pinned')
    parser.add_argument('--sanitizer',choices=['memcheck','initcheck'])
    args=parser.parse_args()
    if args.analytic_only == (args.reference is not None):
        parser.error('select --reference or --analytic-only, not both')
    if args.precision=='fp32' and not args.analytic_only:
        parser.error('frozen original reference files are FP64')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(args.binary,output/'tested_probe');shutil.copy2(__file__,output/'runner.py')
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    if args.transport=='device':
        env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
    command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks)]
    if args.sanitizer:
        command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
            '--tool',args.sanitizer,'--error-exitcode','99','--log-file',str(output/'sanitizer.%p.log')]
        if args.sanitizer=='memcheck':
            command+=['--leak-check','full']
    command += [str(output/'tested_probe'),'analytic' if args.analytic_only else str(args.reference.resolve()),
        str('xyz'.index(args.axis)+1),args.transport]
    with (output/'run.log').open('w') as log:
        subprocess.run(command,cwd=output,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    log=(output/'run.log').read_text();report={}
    for kind,count in [('FROZEN',0 if args.analytic_only else 12),('ANALYTIC',10)]:
        rows=re.findall(r'CPP_SENSOR_'+kind+r'_PASS\s+(\d+)\s+(\d+)\s+(\S+)',log)
        if sorted(int(row[0]) for row in rows)!=list(range(args.ranks)) or any(int(row[1])!=count for row in rows):
            raise AssertionError('all expected rank/case sensor results required')
        report[kind.lower()]=[dict(rank=int(rank),cases=int(cases),max_error=float(error)) for rank,cases,error in rows]
    ties=re.findall(r'CPP_SENSOR_THRESHOLD_EXACT\s+(\d+)\s+(\d+)',log)
    widths=re.findall(r'CPP_SENSOR_ELEMENT_BYTES\s+(\d+)\s+(\d+)',log)
    width=4 if args.precision=='fp32' else 8
    if sorted((int(r),int(c)) for r,c in ties)!=[(r,6) for r in range(args.ranks)] or \
            sorted((int(r),int(w)) for r,w in widths)!=[(r,width) for r in range(args.ranks)]:
        raise AssertionError('native sensor threshold/element-width observations')
    if args.compare:
        for rank in range(args.ranks):
            name=f'observations.rank{rank:08d}.bin'
            if (output/name).read_bytes()!=(args.compare/name).read_bytes():
                raise AssertionError('sensor transport changed native records')
    if args.sanitizer:
        logs=list(output.glob('sanitizer.*.log'))
        if len(logs)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() or
                args.sanitizer=='memcheck' and 'LEAK SUMMARY: 0 bytes leaked' not in p.read_text() for p in logs):
            raise AssertionError('missing clean per-rank memory-tool evidence')
    report.update(ranks=args.ranks,axis=args.axis,transport=args.transport,sanitizer=args.sanitizer,
        raw_physical_words_unchanged_by_exchange=True,physical_byte_masks_exact=True,
        precision=args.precision,threshold_tie_cases_per_rank=6,synchronization_words_equal=True,
        transport_words_equal=args.compare is not None,
        binary_sha256=hashlib.sha256((output/'tested_probe').read_bytes()).hexdigest())
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
