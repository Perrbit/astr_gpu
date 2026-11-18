"""Full marker ownership/halo checks, not CFD or scaling benchmarks."""
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
    for key in ('binary','mpiexec','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--transport',choices=('pinned','device'),default='pinned')
    parser.add_argument('--sanitizer',choices=('memcheck','initcheck','racecheck'))
    parser.add_argument('--case',choices=('np1_periodic','np2_x_periodic','np2_y_wall','np2_z_wall','np4_xy_wall'))
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=args.binary.resolve();shutil.copy2(binary,out/'tested_probe');shutil.copyfile(__file__,out/'runner.py')
    cases=dict(np1_periodic=([1,1,1],[1,1,1]),np2_x_periodic=([2,1,1],[1,1,1]),
        np2_y_wall=([1,2,1],[0,0,1]),np2_z_wall=([1,1,2],[0,0,1]),np4_xy_wall=([2,2,1],[0,0,1]))
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
    report=dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),transport=args.transport,
        sanitizer=args.sanitizer,scope='Boolean ownership and face halos; GPU probes bind local rank modulo visible devices, so NP4 shares on this dual-GPU host; no CFD/performance claim',cases={})
    for name,(topology,periodic) in cases.items():
        if args.case and name!=args.case:continue
        case=out/name;case.mkdir();ranks=topology[0]*topology[1]*topology[2]
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(ranks)]
        if args.sanitizer:
            command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
                '--tool',args.sanitizer,'--error-exitcode','99','--log-file',str(case/'sanitizer.%p.log')]
            if args.sanitizer=='memcheck':command+=['--leak-check','full']
        command += [str(out/'tested_probe'),','.join(map(str,topology)),','.join(map(str,periodic)),args.transport]
        with (case/'run.log').open('w') as log:
            subprocess.run(command,cwd=case,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        rows=re.findall(r'CRITICAL_HISTORY_PASS\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)',(case/'run.log').read_text())
        if len(rows)!=2*ranks or {(int(r),int(d)) for r,d,_,_ in rows}!={(r,d) for r in range(ranks) for d in (0,1)}:
            raise AssertionError('missing marker rank/synchronization evidence')
        if args.sanitizer:
            logs=list(case.glob('sanitizer.*.log'))
            marker='0 hazards displayed' if args.sanitizer=='racecheck' else 'ERROR SUMMARY: 0 errors'
            if len(logs)!=ranks or any(marker not in p.read_text() for p in logs):raise AssertionError('marker tool result')
            if args.sanitizer=='memcheck' and any('LEAK SUMMARY: 0 bytes leaked' not in p.read_text() for p in logs):
                raise AssertionError('marker transport leaked')
        report['cases'][name]=dict(command=command,ranks=ranks,physical_nodes_per_rank=int(rows[0][2]),
            valid_halo_nodes_by_rank={r:int(n) for r,_,_,n in rows},passes_per_rank=4)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(name,'canonical markers and all hm faces exact')


if __name__=='__main__':main()
