"""Exercise the real C/Fortran diagnostic ABI at the selected build precision."""
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
    for name in ('binary','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--precision',choices=('fp32','fp64'),required=True)
    parser.add_argument('--ranks',type=int,choices=(1,2),default=1)
    parser.add_argument('--axis',choices=tuple('xyz'),default='x')
    parser.add_argument('--sanitizer',choices=('memcheck','initcheck'))
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=out/'tested_probe';shutil.copy2(args.binary.resolve(),binary)
    shutil.copy2(__file__,out/'runner.py')
    shutil.copy2(Path(__file__).with_name('native_diagnostics_probe.F90'),out/'probe.F90')
    topology=[1,1,1];topology['xyz'.index(args.axis)]=args.ranks
    command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks)]
    if args.sanitizer:
        command += ['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
            '--tool',args.sanitizer,'--error-exitcode','99','--log-file',str(out/'sanitizer.%p.log')]
        if args.sanitizer=='memcheck':command += ['--leak-check','full']
    command += [str(binary),','.join(map(str,topology))]
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    with (out/'run.log').open('w') as log:
        subprocess.run(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    text=(out/'run.log').read_text()
    rows=re.findall(r'CPP_NATIVE_DIAGNOSTICS_PASS\s+(\d+)\s+(\d+)\s+(\d+)',text)
    width=4 if args.precision=='fp32' else 8
    if len(rows)!=args.ranks or {int(rank) for rank,_,_ in rows}!=set(range(args.ranks)):
        raise AssertionError('missing native diagnostic rank results')
    if any(int(kind)!=width or int(count)!=8 for _,kind,count in rows):
        raise AssertionError('wrong build kind or missing field/sync/dimensional cases')
    if text.count('CPP_NATIVE_DIAGNOSTICS_WALL')!=2*args.ranks:
        raise AssertionError('missing native constant channel/plate integral cases')
    if args.sanitizer:
        logs=list(out.glob('sanitizer.*.log'))
        if len(logs)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in logs):
            raise AssertionError('diagnostic memory-safety failure')
        if args.sanitizer=='memcheck' and any('LEAK SUMMARY: 0 bytes leaked' not in p.read_text() for p in logs):
            raise AssertionError('diagnostic allocations leaked')
    report=dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),precision=args.precision,
        topology=topology,command=command,sanitizer=args.sanitizer,rank_cases=8*args.ranks,
        analytic_directional_and_nodal_sum_maxima=True,deterministic_tie_keys=True,
        local_plane_selection_and_buffer_width=True,dyadic_constant_statistics_words_equal=True,
        constant_channel_and_plate_integrals_equal=True,
        authoritative_q_words_unchanged=True,nonphysical_spectrum_remains_latched=True,
        scope='CFL and instantaneous TGV scalar components; not ordinary-main timestep, temporal moments, compact, output or DCU qualification')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
