"""Freeze original ordered and expanded crashfix fields before C++ consumers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('binary','output','mpiexec'):parser.add_argument('--'+key,type=Path,required=True)
    variants=parser.add_mutually_exclusive_group()
    variants.add_argument('--reconciled',action='store_true',help='freeze the original shared-node repair lifecycle')
    variants.add_argument('--flux',action='store_true',help='freeze the original critical-node interface flux consumer')
    parser.add_argument('--precision',choices=['fp64','fp32'],default='fp64')
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_GPU_SYNC_MODE='explicit',
        ASTR_GPU_HALO_TRANSPORT='pinned')
    report=dict(binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),precision=args.precision,
        real_bytes=4 if args.precision=='fp32' else 8,reconciled=args.reconciled,flux=args.flux,layouts={})
    layouts=[('np1',[1,1,1]),('np2_x',[2,1,1]),('np2_y',[1,2,1]),('np2_z',[1,1,2])]
    if args.flux:layouts=layouts[:1]
    for name,topology in layouts:
        case=root/name;case.mkdir();ranks=max(topology)
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(ranks),
            str(args.binary.resolve()),','.join(map(str,topology)),str(case/'reference')]
        if args.reconciled:command+=['reconciled']
        if args.flux:command+=['flux']
        with (case/'run.log').open('w') as log:
            subprocess.run(command,cwd=case,
                env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        text=(case/'run.log').read_text()
        count=(6 if ranks==1 else 7)-int(args.reconciled)
        label='ORIGINAL_REPAIR_PASS';pattern='reference.case*.bin'
        if args.flux:count=6;label='ORIGINAL_CRITICAL_FLUX_PASS';pattern='reference.flux*.bin'
        if text.count(label)!=count or len(list(case.glob(pattern)))!=count*ranks:
            raise AssertionError('missing original repair scenarios')
        report['layouts'][name]=dict(topology=topology,ranks=ranks,cases=count)
    binary=root/'original_probe';shutil.copyfile(args.binary,binary);binary.chmod(0o555)
    (root/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
    shutil.copyfile(__file__,root/'preparation.py')
    shutil.copyfile(Path(__file__).resolve().parents[1]/'gpu_validation/crashfix_reference_probe.cuf',root/'original_probe.cuf')
    for path in root.rglob('*'):path.chmod(0o555 if path.is_dir() or path.name=='original_probe' else 0o444)
    root.chmod(0o555);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
