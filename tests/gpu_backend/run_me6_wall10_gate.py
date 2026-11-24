"""Compare the isolated half-point candidate with an assembled wall operator."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from run_cpp_tgv_gate import compare

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gpu_validation'))
from check_me6_wall10_constant import closure
from check_me6_wall10_variable import viscosity
from check_me6_wall_halfpoint import rows


def manufactured_errors(result, n, profile='constant'):
    """Measure physical derivatives from the unreflected manufactured input."""
    distances = np.arange(1, 4, dtype=np.float64)/n
    y = np.stack((distances, 1-distances))
    b, db = viscosity(y,profile,np,float)
    exact_rhs = (-b*y*(y+3)+db*(1-y-y*y))*np.exp(y)
    exact_inward_flux = viscosity(np.array([0.,1.]),profile,np,float)[0]*[1.,np.e]
    q = result[-2]
    return dict(wall_rhs=float(np.max(abs(n*n*q[:,8:]-exact_rhs))),
                inward_wall_flux=float(np.max(abs(n*q[:,0]-exact_inward_flux))))


def run(binary, root, profile):
    root.mkdir(parents=True,exist_ok=False)
    gradient, divergence, _, _ = closure()
    interp_rows=np.asarray(rows()[0],dtype=np.float64)
    grad_rows=np.asarray(gradient,dtype=np.float64)
    div_rows=np.asarray(divergence,dtype=np.float64)
    report={}
    for n in (16,32,64,128):
        y=np.arange(n+1,dtype=np.float64)/n
        state=y*(1-y)*np.exp(y)
        mu,_=viscosity(y,profile,np,float)
        profiles=np.vstack((np.eye(n+1),np.ones_like(y),state,state[::-1]))
        fields=np.stack((profiles[:,:10],profiles[:,-1:-11:-1]),axis=1)
        face_mu=np.stack((interp_rows@mu[:8],interp_rows@mu[:-9:-1]))
        flux=face_mu[...,None]*grad_rows
        # Assemble each RHS row before applying it, independently of the probe's two-stage order.
        reference=np.concatenate((flux,div_rows@flux),axis=1)
        expected=np.einsum('csi,smi->csm',fields,reference)
        fixture=root/f'n{n}.bin'
        result=root/f'n{n}.result.bin'
        with fixture.open('xb') as stream:
            np.array([n,len(profiles)],dtype=np.int32).tofile(stream)
            interp_rows.tofile(stream)
            grad_rows.tofile(stream)
            div_rows.tofile(stream)
            mu.tofile(stream)
            profiles.tofile(stream)
        subprocess.run([str(binary),str(fixture),str(result)],check=True)
        cpu,gpu=np.fromfile(result,dtype=np.float64).reshape(2,len(profiles),2,11)
        report[n]=dict(reference_cpu=compare(cpu,expected),reference_gpu=compare(gpu,expected),cpu_gpu=compare(cpu,gpu),
                       bitwise_cpu_gpu=bool(np.array_equal(cpu.view(np.uint64),gpu.view(np.uint64))),
                       manufactured_cpu=manufactured_errors(cpu,n,profile),manufactured_gpu=manufactured_errors(gpu,n,profile),
                       min_wall_half_mu=float(face_mu.min()),inputs=len(profiles))
        if profile == 'constant':
            report[n].update(mirror_cpu=compare(cpu[-2],cpu[-1,::-1]),mirror_gpu=compare(gpu[-2],gpu[-1,::-1]))
        print(f'n={n}: {len(profiles)} profiles, both walls, max scaled CPU/GPU difference '
              f'{max(report[n]["cpu_gpu"]):.3e}',flush=True)
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile',choices=('constant','exp','sutherland'),default='constant')
    args=parser.parse_args()
    run(args.binary.resolve(),args.output.resolve(),args.profile)
