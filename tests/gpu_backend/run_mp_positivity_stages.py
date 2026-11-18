"""Frozen full-stage MP comparisons, including an active anti-diffusive replay."""
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
    parser.add_argument('--transport',choices=['pinned','device'],default='pinned')
    parser.add_argument('--precision',choices=['fp64','fp32'],default='fp64')
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
    report=dict(binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),precision=args.precision,
        transport=args.transport,numeric_difference_limit=None if args.precision=='fp32' else 2e-10,cases={})
    pattern=re.compile(r'^CPP_MP_STAGE_PASS rank=(\d+) case=(\d+) transport=(\w+) limited=(\d+) '
        r'face=(\S+) delta=(\S+) ratio=(\S+) rhs=(\S+)$',re.M)
    for ranks,axis in [(1,'x'),(2,'x'),(2,'y'),(2,'z')]:
        reference=args.reference.resolve()/f'np{ranks}_{axis}'
        for kind in range(2):
            label=f'np{ranks}_{axis}_case{kind}'
            command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(ranks),str(args.binary.resolve()),
                str(reference),str(kind),args.transport]
            with (root/(label+'.log')).open('w') as log:
                subprocess.run(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
            rows=pattern.findall((root/(label+'.log')).read_text())
            assert len(rows)==ranks and sorted(int(row[0]) for row in rows)==list(range(ranks))
            assert all(int(row[1])==kind and row[2]==args.transport and (int(row[3])>0 if kind else int(row[3])==0) for row in rows)
            errors=[max(float(row[i]) for row in rows) for i in range(4,8)]
            if args.precision=='fp64':assert max(errors)<=2e-10
            storage=re.findall(r'^CPP_MP_STAGE_STORAGE real_bytes=(\d+) reference_limited=(\d+)$',
                (root/(label+'.log')).read_text(),re.M)
            assert len(storage)==ranks and all(int(row[0])==(4 if args.precision=='fp32' else 8) for row in storage)
            report['cases'][label]=dict(command=command,rank_passes=ranks,explicit_and_dependency_checked=True,
                limited=int(rows[0][3]),reference_limited=int(storage[0][1]),
                face_scaled=errors[0],delta_scaled=errors[1],ratio_max_abs=errors[2],rhs_scaled=errors[3])
            print(label,errors,'limited',rows[0][3],flush=True)
    (root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
