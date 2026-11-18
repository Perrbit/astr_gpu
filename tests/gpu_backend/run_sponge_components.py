"""Freeze original sponge fields or compare the shared C++ implementation."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess

TOPOLOGIES=[(1,1,1),(2,1,1),(1,2,1),(1,1,2)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('binary','mpiexec','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--freeze-only',action='store_true')
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--precision',choices=('fp64','fp32'),default='fp64')
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_GPU_HALO_TRANSPORT='pinned',ASTR_GPU_SYNC_MODE='explicit')
    binary=args.binary.resolve();records=[]
    def run(name,topology,tail,device=False,memcheck=False):
        case=root/name;case.mkdir()
        execution=dict(env)
        if device:execution.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
        ranks=topology[0]*topology[1]*topology[2]
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(ranks)]
        if memcheck:command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
            '--tool','memcheck','--leak-check','full','--error-exitcode','99','--log-file',str(case/'memcheck.%p.log')]
        with (case/'run.log').open('w') as log:
            subprocess.run(command+[str(binary),*tail],cwd=case,env=execution,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        text=(case/'run.log').read_text()
        marker='ORIGINAL_SPONGE_PASS' if args.freeze_only else 'CPP_SPONGE_PASS'
        if text.count(marker)!=(3 if args.freeze_only else ranks):raise AssertionError('missing sponge numerical records')
        item=dict(name=name,topology=topology,transport='device' if device else 'pinned',memcheck=memcheck)
        if not args.freeze_only:
            widths=re.findall(r'CPP_SPONGE_REAL_BYTES\s+(\d+)\s+(\d+)',text)
            if sorted((int(r),int(w)) for r,w in widths)!=[(r,4 if args.precision=='fp32' else 8) for r in range(ranks)]:
                raise AssertionError('sponge actual native width')
            rows=re.findall(r'CPP_SPONGE_NATIVE_DIFFERENCE\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)',text)
            if len(rows)!=30*ranks:raise AssertionError('sponge native per-component observations')
            item['per_component_scaled_report_only']=[max(float(e) for r,m,d,c,e in rows if int(c)==component) for component in range(1,6)]
            item['constant_error']=[float(v) for v in re.findall(r'CPP_SPONGE_PASS\s+\d+\s+6\s+\S+\s+(\S+)',text)]
        if memcheck:
            logs=list(case.glob('memcheck.*.log'))
            if len(logs)!=ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() or
                'LEAK SUMMARY: 0 bytes leaked' not in p.read_text() for p in logs):raise AssertionError('sponge memory evidence')
        records.append(item)
        print(name,text.strip(),flush=True)
    for topology in TOPOLOGIES:
        label=''.join(map(str,topology));token=','.join(map(str,topology))
        if args.freeze_only:
            run(label,topology,[token,str(root/label/'reference')])
        else:
            if args.reference is None:parser.error('--reference required for C++ comparison')
            source=str((args.reference/label).resolve())
            run(label+'_pinned',topology,[source,token,'pinned'])
            if topology!=(1,1,1):run(label+'_device',topology,[source,token,'device'],device=True)
    if not args.freeze_only:
        run('121_memcheck',(1,2,1),[str((args.reference/'121').resolve()),'1,2,1','pinned'],memcheck=True)
    report=dict(binary=str(binary),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),precision=args.precision,cases=records,
        scope='linear five-array sponge fixture, not a thermodynamic CFD state')
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    shutil.copyfile(__file__,root/'preparation.py')
    if args.freeze_only:
        shutil.copyfile(binary,root/'original_sponge_probe')
        for p in root.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
        (root/'original_sponge_probe').chmod(0o555)
        root.chmod(0o555)


if __name__=='__main__':main()
