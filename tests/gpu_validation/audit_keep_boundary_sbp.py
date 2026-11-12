"""Necessary diagonal-norm SBP checks on actual ASTR derivative matrices.

An obstruction to HD+D^T H=B is not a proof that a particular PDE closure is
unstable, and says nothing about all non-diagonal or constrained energy norms.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

import numpy as np


def obstruction(d):
    tolerance=128*np.finfo(float).eps*max(1,np.max(abs(d)))
    for i in range(1,len(d)-1):
        if abs(d[i,i])>tolerance:
            return dict(kind='interior_diagonal',i=i,d_ii=float(d[i,i]),
                        equation='2*h_i*D_ii=0 with h_i>0 is impossible')
    for i in range(len(d)):
        for j in range(i+1,len(d)):
            a,b=d[i,j],d[j,i]
            if (abs(a)>tolerance and abs(b)<=tolerance) or (abs(b)>tolerance and abs(a)<=tolerance):
                return dict(kind='one_sided_support',i=i,j=j,d_ij=float(a),d_ji=float(b),
                            equation='h_i*D_ij+h_j*D_ji=0 with positive weights is impossible')
            if a*b>tolerance*tolerance:
                return dict(kind='same_sign_pair',i=i,j=j,d_ij=float(a),d_ji=float(b),
                            equation='h_i*D_ij+h_j*D_ji=0 with positive weights is impossible')
    return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    build=args.build.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    link=shlex.split((build/'src/CMakeFiles/astr.dir/link.txt').read_text())
    includes,=[line.split('=',1)[1] for line in (build/'src/CMakeFiles/astr.dir/flags.make').read_text().splitlines()
               if line.startswith('Fortran_INCLUDES =')]
    obj=out/'probe.o';exe=out/'probe'
    command=[link[0],'-Mpreprocess','-Mextend','-traceback','-g','-Mnostack_arrays','-O2',
             '-I'+str(build/'src'),*shlex.split(includes),'-c',
             str(Path(__file__).with_name('keep_wall_derivative_matrix_probe.F90')),'-o',str(obj)]
    subprocess.run(command,cwd=out,check=True)
    link[link.index('CMakeFiles/astr.dir/astr.F90.o')]=str(obj)
    link[link.index('-o')+1]=str(exe)
    link=[value for value in link if not value.startswith('-Wl,--dependency-file=')]
    subprocess.run(link,cwd=build/'src',check=True)
    report=dict(scope='existing first-derivative matrices only, full-grid diagonal positive H and endpoint B',
                identity='H*D+D.T*H=diag(-1,0,...,0,1)',compile=command,cases={})
    for n in (16,32):
        path=out/f'matrices_n{n}.txt'
        with (out/f'extraction_n{n}.log').open('w') as log:
            subprocess.run([str(exe),str(n),str(path)],cwd=out,check=True,stdout=log,stderr=subprocess.STDOUT)
        rows=np.loadtxt(path)
        result={}
        for name,column in (('legacy_explicit',2),('legacy_compact',3),('existing_high_order_derivative',4)):
            d=np.zeros((n+1,n+1));d[rows[:,0].astype(int),rows[:,1].astype(int)]=rows[:,column]
            witness=obstruction(d)
            if witness is None:
                raise AssertionError(f'No expected obstruction found; re-examine the proposed diagonal-norm exclusion: {name}')
            result[name]=dict(constant_residual=float(np.max(abs(d@np.ones(n+1)))),witness=witness)
            if name=='existing_high_order_derivative':
                x=np.arange(n+1,dtype=float)/n
                polynomial=[]
                for degree in range(7):
                    exact=np.zeros(n+1) if degree==0 else degree*x**(degree-1)/n
                    polynomial.append(float(np.max(abs((d@x**degree-exact)[:3]))))
                if max(polynomial)>2e-10:
                    raise AssertionError('Extracted high-order rows do not reproduce their approved polynomial moments')
                result[name]['near_boundary_polynomial_errors_degree_0_to_6']=polynomial
        report['cases'][str(n)]=result
    report['status']='measured_necessary_condition_obstructions; not PDE instability acceptance'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['cases'],indent=2),flush=True)


if __name__=='__main__':
    main()
