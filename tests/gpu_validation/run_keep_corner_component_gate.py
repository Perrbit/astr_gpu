"""Check the proposed nonreflecting corner with actual CPU/GPU face components."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mpiexec',type=Path,required=True)
    args=parser.parse_args()
    build=args.build.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    link=shlex.split((build/'src/CMakeFiles/astr.dir/link.txt').read_text())
    includes,=[line.split('=',1)[1] for line in (build/'src/CMakeFiles/astr.dir/flags.make').read_text().splitlines()
               if line.startswith('Fortran_INCLUDES =')]
    obj=out/'probe.o';exe=out/'probe'
    command=[link[0],'-Mpreprocess','-Mextend','-g','-traceback','-Mnostack_arrays','-O2',
             '-cuda','-gpu=lineinfo','-I'+str(build/'src'),*shlex.split(includes),'-c',
             str(Path(__file__).with_name('keep_nonreflecting_corner_probe.cuf')),'-o',str(obj)]
    subprocess.run(command,cwd=out,check=True)
    link[link.index('CMakeFiles/astr.dir/astr.F90.o')]=str(obj)
    link[link.index('-o')+1]=str(exe)
    link=[value for value in link if not value.startswith('-Wl,--dependency-file=')]
    subprocess.run(link,cwd=build/'src',check=True)
    env=dict(os.environ,OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
             OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    report={'status':'running','build':str(build),'cases':{},'meaning':'component only; no CFD trajectory'}
    for mode in ('uniform','outflow','inflow','grazing'):
        log=out/f'{mode}.log'
        with log.open('w') as stream:
            result=subprocess.run([str(args.mpiexec.resolve()),'--mca','coll_hcoll_enable','0',
                                   '-np','1',str(exe),mode],cwd=out,env=env,stdout=stream,stderr=subprocess.STDOUT)
        lines=[line for line in log.read_text().splitlines() if line.startswith('KEEP_NR_CORNER ')]
        if not lines:
            raise RuntimeError(f'Corner component exited {result.returncode} before diagnostic output: {log}')
        line,=lines
        names=('raw_cpu_gpu','projected_cpu_gpu','incoming_residual','outgoing_change',
               'independent_projection','constant_state_rhs')
        report['cases'][mode]=dict(zip(names,map(float,line.split()[2:])),returncode=result.returncode)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(line,flush=True)
        if result.returncode:
            raise RuntimeError(f'Corner component failed: {log}')
    report['status']='passed'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
