"""Manufactured ME6/nodal wall splice through root-CMake production objects."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mpiexec',type=Path,required=True)
    parser.add_argument('--gpu',action='store_true')
    parser.add_argument('--sizes',type=int,nargs='+',default=[16,32,64])
    parser.add_argument('--mpi',action='store_true')
    args=parser.parse_args()
    build=args.build.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    link=shlex.split((build/'src/CMakeFiles/astr.dir/link.txt').read_text())
    includes,=[line.split('=',1)[1] for line in (build/'src/CMakeFiles/astr.dir/flags.make').read_text().splitlines()
               if line.startswith('Fortran_INCLUDES =')]
    obj=out/'probe.o';exe=out/'probe'
    command=[link[0],'-Mpreprocess','-Mextend','-g','-traceback','-Mnostack_arrays','-O2',
             '-I'+str(build/'src'),*shlex.split(includes),'-c',
             str(Path(__file__).with_name('me6opti_wall_probe.F90')),'-o',str(obj)]
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
    matrix=[(n,1,1) for n in args.sizes]
    if args.mpi:
        matrix.extend((args.sizes[0],2,axis) for axis in (1,2,3))
    for n,ranks,axis in matrix:
        log=out/f'n{n}_np{ranks}_axis{axis}.log'
        with log.open('w') as stream:
            result=subprocess.run([str(args.mpiexec.resolve()),'--mca','coll_hcoll_enable','0','-np',str(ranks),
                                   str(exe),str(n),str(axis)],cwd=out,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'ME6 wall component exited {result.returncode}: {log}')
        line,=[line for line in log.read_text().splitlines() if line.startswith('ME6_WALL_ERRORS')]
        metrics=list(map(float,line.split('=')[1].split()))
        budget,=[line for line in log.read_text().splitlines() if line.startswith('ME6_WALL_NODAL_HYBRID_BUDGET')]
        values=list(map(float,budget.split('=')[1].split()))
        report['cases'].append({'n':n,'np':ranks,'axis':axis,'row_errors':metrics[3:8],
                                'cpu_gpu_scaled':metrics[8],'mapped_nodal_hybrid_sums':values})
        for norm in ('L1','L2','LINF'):
            norm_line,=[line for line in log.read_text().splitlines() if line.startswith('ME6_WALL_GLOBAL_'+norm+'=')]
            report['cases'][-1][norm]=list(map(float,norm_line.split('=')[1].split()))
        print(line,flush=True)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    levels=[case for case in report['cases'] if case['np']==1]
    report['orders']=[]
    for first,last in zip(levels,levels[1:]):
        orders=[math.log(a/b)/math.log(last['n']/first['n']) for a,b in zip(first['row_errors'],last['row_errors'])]
        report['orders'].append(orders)
        if min(orders)<=0:
            raise AssertionError(f'ME6 wall manufactured error does not decrease: {orders}')
    report['status']='passed'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
