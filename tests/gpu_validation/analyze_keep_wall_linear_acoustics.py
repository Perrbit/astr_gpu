"""Diagnose restricted wall-normal acoustics using production derivative matrices.

This is matrix analysis, not a CFD time integrator or full viscous stability proof.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

import h5py
import numpy as np


def acoustic_matrix(derivative, jacobian, mapped_area):
    n=len(jacobian)-1
    pressure=np.zeros((n+1,n-1))
    pressure[1:n]=np.eye(n-1)
    # Original bc41 computational normal-pressure derivative closure.
    pressure[0,0]=4/3;pressure[0,1]=-1/3
    pressure[n,-1]=4/3;pressure[n,-2]=-1/3
    velocity=np.zeros_like(pressure)
    velocity[1:n]=np.eye(n-1)
    gradient=derivative*mapped_area[None,:]/jacobian[:,None]
    zero=np.zeros((n-1,n-1))
    # rho0=1, c0=1/Ma, Ma=.3. Pressure and wall-normal velocity are unknowns.
    return np.block([[zero,-(1/.3**2)*gradient[1:n]@velocity],
                     [-gradient[1:n]@pressure,zero]])


def spectrum(matrix, dt):
    eigenvalues,eigenvectors=np.linalg.eig(matrix)
    index=int(np.argmax(eigenvalues.real))
    value=eigenvalues[index]
    vector=eigenvectors[:,index]
    vector=vector/np.max(abs(vector))
    z=dt*eigenvalues
    amplification=1+z+z*z/2+z*z*z/6
    residual=np.max(abs(matrix@vector-value*vector))/max(1,np.linalg.norm(matrix,np.inf))
    return dict(max_real=float(value.real),imag_at_max_real=float(value.imag),
                max_abs_eigenvalue=float(np.max(abs(eigenvalues))),
                rk3_spectral_radius=float(np.max(abs(amplification))),dt=dt,
                eigen_residual=float(residual),
                pressure_mode_abs=abs(vector[:len(vector)//2]).tolist(),
                velocity_mode_abs=abs(vector[len(vector)//2:]).tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--geometry',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    build=args.build.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    with h5py.File(args.geometry) as state:
        y=state['q0002'][0,:,0]
        jac=state['q0004'][0,:,0]
        eta_y=state['q0009'][0,:,0]
    if not np.isfinite(jac).all() or not (jac>0).all():
        raise AssertionError('Diagnostic geometry has nonpositive/nonfinite Jacobian')
    n=len(y)-1
    link=shlex.split((build/'src/CMakeFiles/astr.dir/link.txt').read_text())
    includes,=[line.split('=',1)[1] for line in (build/'src/CMakeFiles/astr.dir/flags.make').read_text().splitlines()
               if line.startswith('Fortran_INCLUDES =')]
    obj=out/'probe.o';exe=out/'probe'
    compile_command=[link[0],'-Mpreprocess','-Mextend','-traceback','-g','-Mnostack_arrays','-O2',
                     '-I'+str(build/'src'),*shlex.split(includes),'-c',
                     str(Path(__file__).with_name('keep_wall_derivative_matrix_probe.F90')),'-o',str(obj)]
    subprocess.run(compile_command,cwd=out,check=True)
    link[link.index('CMakeFiles/astr.dir/astr.F90.o')]=str(obj)
    link[link.index('-o')+1]=str(exe)
    link=[value for value in link if not value.startswith('-Wl,--dependency-file=')]
    subprocess.run(link,cwd=build/'src',check=True)
    with (out/'extraction.log').open('w') as log:
        subprocess.run([str(exe),str(n),str(out/'matrices.txt')],cwd=out,check=True,
                       stdout=log,stderr=subprocess.STDOUT)
    rows=np.loadtxt(out/'matrices.txt')
    matrices={}
    for name,column in (('explicit',2),('compact',3)):
        matrix=np.zeros((n+1,n+1))
        matrix[rows[:,0].astype(int),rows[:,1].astype(int)]=rows[:,column]
        matrices[name]=matrix
    geometries={'actual_stretched':(jac,jac*eta_y),
                'uniform_same_height':(np.full(n+1,(y[-1]-y[0])/n),np.ones(n+1))}
    report=dict(scope='restricted inviscid wall-normal acoustics; original pressure extrapolation and zero wall velocity; no CFD advancement',
                geometry=str(args.geometry.resolve()),intervals=n,compile=compile_command,
                samples={},interpretation='measured spectrum; no full nonlinear or viscous stability claim')
    for geometry,(j,area) in geometries.items():
        for scheme,derivative in matrices.items():
            operator=acoustic_matrix(derivative,j,area)
            entry={str(dt):spectrum(operator,dt) for dt in (.001,.0005,.0001)}
            report['samples'][geometry+'_'+scheme]=entry
            print(geometry,scheme,entry['0.001']['max_real'],entry['0.001']['rk3_spectral_radius'],flush=True)
    report['status']='measured'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
