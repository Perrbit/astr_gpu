"""Actual native-FP32 explicit components and four-step GPU composition."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from compare_q_validation_snapshots import read_q_snapshot
from run_native_precision_tgv_gate import component_report,physical_state


def analyze(case,reference,topology):
    ranks=int(np.prod(topology));text=(case/'run.log').read_text()
    rows=re.findall(r'CPP_NATIVE_EXPLICIT32_PASS\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.Ee+\-]+)',text)
    if len(rows)!=ranks or any(int(p)!=24 or int(c)!=8 for _,p,c,_ in rows):
        raise AssertionError('missing native frozen/coupled phase evidence')
    for key in ('CONVERSION','AFFINE','CONSTANT'):
        if text.count('CPP_NATIVE_EXPLICIT32_'+key)!=2*ranks:raise AssertionError('missing exact native invariant')
    differences=re.findall(r'CPP_NATIVE_EXPLICIT32_DIFFERENCE\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\w+)\s+((?:[\d.Ee+\-]+\s*){5})',text)
    if len(differences)!=72*ranks:raise AssertionError('missing same-state variable reports')
    phase={name:max(max(map(float,v.split())) for _,_,_,_,_,n,v in differences if n==name)
        for name in ('conv','full','updated')}
    field_rows={};per_rank_bits=True
    for source in (0,1):
      for filtering in (0,1):
        previous=reference/f'filter{filtering}_{"cpu" if source==0 else "gpu"}'/'outdat/rhs.post_update.step00000003.rk03.rank00000000.bin'
        expected=physical_state(previous,'fp32');actual=np.empty_like(expected)
        for rank in range(ranks):
            coordinate=np.array([rank%topology[0],rank//topology[0]%topology[1],rank//(topology[0]*topology[1])])
            cells=np.array([16,16,16])//topology;origin=coordinate*cells
            first=case/f'source{source}.filter{filtering}.debug0.rank{rank:08d}.bin'
            second=case/f'source{source}.filter{filtering}.debug1.rank{rank:08d}.bin'
            a=read_q_snapshot(first,precision='fp32');b=read_q_snapshot(second,precision='fp32')
            if a.values.dtype!=np.dtype('float32') or a.header!=b.header or a.values.tobytes()!=b.values.tobytes():
                raise AssertionError('native explicit/dependency full-buffer words differ')
            values=physical_state(first,'fp32')
            body=tuple(slice(int(o),int(o+n+1)) for o,n in zip(origin,cells))
            actual[body]=values
        field_rows[f'source{source}_filter{filtering}']=component_report(expected,actual)
    return dict(phase_rank_cases=sum(int(v[1]) for v in rows),coupled_rank_cases=sum(int(v[2]) for v in rows),
        same_state_scaled_report_only=phase,coupled_fields_report_only=field_rows,
        full_q_words_explicit_dependency_equal=per_rank_bits,
        exact_dyadic_conversion_affine_projection_and_constant_invariants=True,
        gate='Original native precision: arithmetic/invariant checks, finite positive states; differences reported, no added FP64 or cross-precision cap')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('binary','packets','reference','mpiexec','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--ranks',type=int,choices=(1,2),default=1)
    parser.add_argument('--axis',choices=('x','y','z'),default='x')
    parser.add_argument('--transport',choices=('pinned','device'),default='pinned')
    parser.add_argument('--workspace',choices=('scalar','full'),default='scalar')
    parser.add_argument('--sanitizer',choices=('memcheck','initcheck'))
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=args.binary.resolve();shutil.copyfile(binary,out/'tested_probe');shutil.copyfile(__file__,out/'runner.py')
    topology=[1,1,1];topology['xyz'.index(args.axis)]=args.ranks
    layout='np1' if args.ranks==1 else 'np2_'+args.axis
    command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks)]
    if args.sanitizer:
        command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
            '--tool',args.sanitizer,'--error-exitcode','99','--log-file',str(out/'sanitizer.%p.log')]
        if args.sanitizer=='memcheck':command+=['--leak-check','full']
    command += [str(binary),str(args.packets.resolve()),str(out),','.join(map(str,topology)),args.transport,args.workspace,layout]
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
    with (out/'run.log').open('w') as log:
        subprocess.run(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    report=analyze(out,args.reference.resolve(),np.array(topology))
    if args.sanitizer:
        logs=list(out.glob('sanitizer.*.log'))
        if len(logs)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in logs):
            raise AssertionError('native explicit memory-safety evidence')
        if args.sanitizer=='memcheck' and any('LEAK SUMMARY: 0 bytes leaked' not in p.read_text() for p in logs):
            raise AssertionError('native explicit context leaked')
    report.update(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),topology=topology,
        transport=args.transport,workspace=args.workspace,sanitizer=args.sanitizer,command=command,
        scope='Native explicit GPU components/composition; not ordinary-main/CFL/statistics/compact/output/restart qualification')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(topology=topology,workspace=args.workspace,phase=report['same_state_scaled_report_only'],
        coupled_max_scaled=max(r['max_scaled'] for rows in report['coupled_fields_report_only'].values() for r in rows)),indent=2))


if __name__=='__main__':main()
