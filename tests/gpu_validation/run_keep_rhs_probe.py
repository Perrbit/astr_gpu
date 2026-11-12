"""Check additive compact KEEP RHS using actual root-CMake solver objects."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--mpiexec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',action='store_true')
    parser.add_argument('--repair',action='store_true')
    args=parser.parse_args()
    build=args.build.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    link=shlex.split((build/'src/CMakeFiles/astr.dir/link.txt').read_text())
    includes,=[line.split('=',1)[1] for line in (build/'src/CMakeFiles/astr.dir/flags.make').read_text().splitlines()
               if line.startswith('Fortran_INCLUDES =')]
    obj=out/'probe.o';exe=out/'probe'
    command=[link[0],'-Mpreprocess','-Mextend','-traceback','-g','-Mnostack_arrays','-O2',
             '-I'+str(build/'src'),*shlex.split(includes),'-c',
             str(Path(__file__).with_name('keep_rhs_production_probe.F90')),'-o',str(obj)]
    if args.gpu:
        command+=['-cuda','-gpu=lineinfo','-DTEST_CUDA']
    subprocess.run(command,cwd=out,check=True)
    link[link.index('CMakeFiles/astr.dir/astr.F90.o')]=str(obj)
    link[link.index('-o')+1]=str(exe)
    link=[value for value in link if not value.startswith('-Wl,--dependency-file=')]
    subprocess.run(link,cwd=build/'src',check=True)
    env=dict(os.environ,OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
             OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    report={'build':str(build),'gpu':args.gpu,'repair':args.repair,'compile':command,'link':link,'cases':[]}
    for ranks,axis in ((1,1),(2,1),(2,2),(2,3)):
        log=out/f'np{ranks}_axis{axis}.log'
        with log.open('w') as stream:
            subprocess.run([str(args.mpiexec.resolve()),'--mca','coll_hcoll_enable','0','-np',str(ranks),
                            str(exe),str(axis),'repair' if args.repair else 'rhs'],
                           cwd=out,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        line,=[line for line in log.read_text().splitlines() if line.startswith('KEEP_RHS_COMPACT')]
        metrics=list(map(float,line.split('=')[1].split()))
        report['cases'].append({'np':ranks,'axis':axis,'cpu_max_abs':metrics[2],'gpu_max_abs':metrics[3]})
        if args.repair:
            repair,=[line for line in log.read_text().splitlines() if line.startswith('KEEP_CRASHFIX')]
            fields=repair.split()
            report['cases'][-1].update(repaired_nodes=int(fields[1].split('=')[1]),repair_max_scaled=float(fields[-1]))
            print(repair,flush=True)
        print(line,flush=True)
    report['status']='passed'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
