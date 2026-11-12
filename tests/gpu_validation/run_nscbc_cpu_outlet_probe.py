"""Check the actual generic CPU x+ bc22 against analytic states and GPU kernels."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

MODES=['out_sup','out_sub','out_tangent','out_grazing','out_uniform','out_sup_gradient',
       'out_backflow','out_backflow_sup','out_backflow_pout']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mpiexec',type=Path,required=True)
    parser.add_argument('--gpu',action='store_true')
    parser.add_argument('--np',type=int,nargs='+',default=[1,2])
    parser.add_argument('--modes',nargs='+',choices=MODES,default=MODES)
    args=parser.parse_args()
    build=args.build.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    link=shlex.split((build/'src/CMakeFiles/astr.dir/link.txt').read_text())
    includes,=[line.split('=',1)[1] for line in (build/'src/CMakeFiles/astr.dir/flags.make').read_text().splitlines()
               if line.startswith('Fortran_INCLUDES =')]
    obj=out/'probe.o';exe=out/'probe'
    command=[link[0],'-Mpreprocess','-Mextend','-g','-traceback','-Mnostack_arrays','-O2',
             '-I'+str(build/'src'),*shlex.split(includes),'-c',
             str(Path(__file__).with_name('nscbc_cpu_outlet_probe.F90')),'-o',str(obj)]
    if args.gpu:
        command+=['-cuda','-gpu=lineinfo','-DTEST_CUDA']
    subprocess.run(command,cwd=out,check=True)
    link[link.index('CMakeFiles/astr.dir/astr.F90.o')]=str(obj)
    link[link.index('-o')+1]=str(exe)
    link=[value for value in link if not value.startswith('-Wl,--dependency-file=')]
    subprocess.run(link,cwd=build/'src',check=True)
    env=dict(os.environ,OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
             OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    report={'build':str(build),'gpu':args.gpu,'compile':command,'link':link,'cases':[]}
    for ranks in args.np:
        for mode in args.modes:
            log=out/f'np{ranks}_{mode}.log'
            with log.open('w') as stream:
                result=subprocess.run([str(args.mpiexec.resolve()),'--mca','coll_hcoll_enable','0','-np',str(ranks),
                                       str(exe),mode],cwd=out,env=env,stdout=stream,stderr=subprocess.STDOUT)
            lines=[line for line in log.read_text().splitlines() if line.startswith('NSCBC_CPU_OUTLET')]
            if not lines:
                raise RuntimeError(f'CPU outlet probe exited {result.returncode} before diagnostic output: {log}')
            line,=lines
            values=list(map(float,line.split()[3:]))
            record={'np':ranks,'mode':mode,'cpu_error':values[0],'gpu_error':values[1],
                    'cpu_gpu_error':values[2],'returncode':result.returncode}
            report['cases'].append(record)
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(line,flush=True)
            if result.returncode:
                raise RuntimeError(f'CPU outlet analytic check failed: {log}')
    report['status']='passed'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
