"""Compare actual C++ upwind/sensor operators with immutable solver packets."""
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
    for name in ('binary','reference','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--ranks',type=int,choices=[1,2],default=1)
    parser.add_argument('--axis',choices=['x','y','z'],default='x')
    parser.add_argument('--transport',choices=['pinned','device'],default='pinned')
    parser.add_argument('--sanitizer',choices=['memcheck','initcheck'])
    parser.add_argument('--steps-reference',type=Path)
    parser.add_argument('--compact-expanded',action='store_true')
    parser.add_argument('--mixed-flux',action='store_true')
    parser.add_argument('--characteristic-storage',action='store_true')
    parser.add_argument('--high-open',action='store_true')
    parser.add_argument('--precision',choices=['fp64','fp32'],default='fp64')
    parser.add_argument('--compare',type=Path,help='same precision/topology records from another transport')
    args=parser.parse_args()
    if (args.compact_expanded or args.high_open) and args.steps_reference:
        parser.error('expanded-mask/H1 component packets do not cover prepared RK stages')
    if sum((args.compact_expanded,args.mixed_flux,args.characteristic_storage,args.high_open))>1:
        parser.error('expanded masks, H1, MP1 flux and MP3 characteristic storage are separate component tests')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,output/'runner.py')
    shutil.copy2(args.binary.resolve(),output/'tested_probe')
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
    command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks)]
    if args.sanitizer:
        command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
            '--tool',args.sanitizer,'--error-exitcode','99','--log-file',str(output/'sanitizer.%p.log')]
        if args.sanitizer=='memcheck':command+=['--leak-check','full']
    command += [str(output/'tested_probe'),str(args.reference.resolve()),str('xyz'.index(args.axis)+1),args.transport]
    if args.steps_reference or args.compact_expanded or args.mixed_flux or args.characteristic_storage or args.high_open:
        command += [str(args.steps_reference.resolve()) if args.steps_reference else '']
    if args.compact_expanded or args.mixed_flux or args.characteristic_storage or args.high_open:
        command += ['high-open' if args.high_open else ('compact-expanded' if args.compact_expanded else '')]
    if args.mixed_flux:command += ['mixed-flux']
    if args.characteristic_storage:command += ['characteristic-storage']
    with (output/'run.log').open('w') as log:
        subprocess.run(command,cwd=output,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    rows=re.findall(r'CPP_UPWIND_PASS\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)',(output/'run.log').read_text())
    count=24 if args.mixed_flux or args.characteristic_storage or args.high_open else (12 if args.compact_expanded else 72)
    if sorted(int(r[0]) for r in rows)!=list(range(args.ranks)) or any(int(r[1])!=count for r in rows):
        raise AssertionError(f'all {count} upwind combinations per rank required')
    widths=re.findall(r'CPP_UPWIND_ELEMENT_BYTES\s+(\d+)\s+(\d+)',(output/'run.log').read_text())
    width=4 if args.precision=='fp32' else 8
    if sorted((int(r),int(w)) for r,w in widths)!=[(r,width) for r in range(args.ranks)]:
        raise AssertionError('upwind native storage/packet precision')
    if args.compare:
        for rank in range(args.ranks):
            name=f'observations.rank{rank:08d}.bin'
            if (output/name).read_bytes()!=(args.compare/name).read_bytes():
                raise AssertionError('upwind transport changed result records')
    if args.sanitizer:
        logs=list(output.glob('sanitizer.*.log'))
        if len(logs)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in logs):
            raise AssertionError('missing clean per-rank tool evidence')
    report=dict(ranks=args.ranks,axis=args.axis,transport=args.transport,sanitizer=args.sanitizer,
        binary_sha256=hashlib.sha256((output/'tested_probe').read_bytes()).hexdigest(),
        families=['mp7_roe'] if args.characteristic_storage else (['mp7'] if args.mixed_flux else
            (['mp7ld_flat'] if args.compact_expanded or args.high_open else ['mp7','mp7_roe','mp7ld_flat'])),
        mixed_flux=args.mixed_flux,
        compact_expanded=args.compact_expanded,rows=[dict(rank=int(r),cases=int(n),rhs_max_error=float(e),
        gradient_max_error=float(g)) for r,n,e,g in rows],sensor_masks_exact=True,q_bits_unchanged=True,
        interface_workspace_reused=True,gradient_modes=['frozen'] if args.compact_expanded else ['frozen','native'],
        precision=args.precision,synchronization_rhs_words_equal=True,transport_words_equal=args.compare is not None)
    if args.high_open:
        temperature=re.findall(r'CPP_HIGH_OPEN_TEMPERATURE_PASS\s+(\d+)\s+(\d+)\s+(\S+)',(output/'run.log').read_text())
        if sorted(int(row[0]) for row in temperature)!=list(range(args.ranks)) or any(int(row[1])!=12 for row in temperature):
            raise AssertionError('all H1 native temperature-gradient comparisons required')
        report.update(high_order_open=True,temperature_gradient=[dict(rank=int(r),cases=int(n),max_error=float(e))
            for r,n,e in temperature],scope='frozen H1 gradient/sensor/full convection including open physical faces; no complete NSCBC or full main')
    if args.steps_reference:
        steps=re.findall(r'CPP_UPWIND_STEP_PASS\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)',(output/'run.log').read_text())
        expected_steps=12 if args.mixed_flux or args.characteristic_storage else 36
        if sorted(int(r[0]) for r in steps)!=list(range(args.ranks)) or any(int(r[1])!=expected_steps for r in steps):
            raise AssertionError(f'all {expected_steps} prepared RK combinations per rank required')
        report['prepared_stages']=[dict(rank=int(r),cases=int(n),rhs_max_error=float(e),q_max_error=float(q)) for r,n,e,q in steps]
    if args.characteristic_storage:
        rows=re.findall(r'CPP_CHARACTERISTIC_STORAGE_PASS\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)',
            (output/'run.log').read_text())
        if sorted(int(row[0]) for row in rows)!=list(range(args.ranks)) or any(int(row[1])!=24 for row in rows):
            raise AssertionError('all MP3 conversion/sensor invariants required')
        report['characteristic_storage']=[dict(rank=int(r),cases=int(n),exact_float_stores=int(v),
            marked_interfaces=int(m),unmarked_interfaces=int(u),fp64_bytes=int(b),fp32_bytes=int(s))
            for r,n,v,m,u,b,s in rows]
        if sum(int(row[3]) for row in rows)==0 or sum(int(row[4]) for row in rows)==0:
            raise AssertionError('both characteristic and physical-space interface branches must be exercised')
        report.update(raw_sensor_bits_unchanged=True,integer_masks_unchanged=True,exact_fp32_store=True,
            exact_promoted_flux_difference=True,mp3_main_tested=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
