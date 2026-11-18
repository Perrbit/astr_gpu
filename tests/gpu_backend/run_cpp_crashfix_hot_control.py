"""Real input.run polling and device-history lifecycle, without CFD advancement."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from run_cpp_tgv_gate import prepare_case,namelist


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('binary','template','mpiexec','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--precision',choices=['fp64','fp32'],default='fp64')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=out/'tested_probe';shutil.copy2(args.binary.resolve(),binary);shutil.copy2(__file__,out/'runner.py')
    base={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    base.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_GPU_FILTER_WORKSPACE='scalar',
        ASTR_VALIDATION_CRASHFIX_HOT_CONTROL='1')
    report=dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),precision=args.precision,
        scope='real parser/poll/validator/apply and actual device repair; synthetic control positions, no RK advancement',cases={})
    for name,ranks,transport,sync,tool in (
        ('np1_pinned',1,'pinned','explicit',False),('np2_pinned',2,'pinned','dependency',False),
        ('np2_device',2,'device','explicit',False),('np2_pinned_memcheck',2,'pinned','dependency',True)):
        case=out/name
        prepare_case(case,args.template.resolve(),True,True,False,dt=1e-6,
            numerics=dict(conschm='643e',difschm='643e',lfilter=False))
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=4,reload_steps=1),
            timestep=dict(mode='fixed',fixed_dt=1e-6),logging=dict(list_steps=1),
            repair=dict(crashfix=False),statistics=dict(enabled=False))))
        env=base|dict(ASTR_FORCE_MPI_TOPOLOGY=f'{ranks},1,1',ASTR_GPU_HALO_TRANSPORT=transport,ASTR_GPU_SYNC_MODE=sync)
        if transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(ranks)]
        if tool:
            command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
                '--tool','memcheck','--leak-check','full','--error-exitcode','99','--log-file',str(case/'sanitizer.%p.log')]
        command += [str(binary),'run']
        with (case/'run.log').open('w') as log:
            subprocess.run(command,cwd=case,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        rows=re.findall(r'CPP_CRASHFIX_HOT_CONTROL_PASS\s+(\d+)\s+(\d+)\s+(\d+)\s+([TF])',(case/'run.log').read_text())
        if {(int(r),int(s),mode) for r,s,_,mode in rows}!={(r,s,'F' if s==2 else 'T') for r in range(ranks) for s in (1,2,3)} or len(rows)!=3*ranks:
            raise AssertionError('missing actual control transitions')
        widths=re.findall(r'^CPP_CRASHFIX_HOT_REAL_BYTES=(\d+)$',(case/'run.log').read_text(),re.M)
        if len(widths)!=ranks or any(int(width)!=(4 if args.precision=='fp32' else 8) for width in widths):
            raise AssertionError('hot-control numerical storage width')
        if tool:
            logs=list(case.glob('sanitizer.*.log'))
            if len(logs)!=ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() or
                'LEAK SUMMARY: 0 bytes leaked' not in p.read_text() for p in logs):raise AssertionError('control/history memory safety')
        report['cases'][name]=dict(command=command,rows=rows,disabled_unallocated=True,
            enabled_repaired=True,healthy_q_words_and_history_unchanged=True,memcheck=tool)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(name,'off/on/off/on history and full-q words checked')


if __name__=='__main__':main()
