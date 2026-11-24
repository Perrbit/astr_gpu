"""Compare native W1 frozen momentum, total energy and wall fluxes against matrix algebra."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from run_keep_w1_normal_gate import audit, operators, sparse
from check_keep_w1_tensor import test


def run(executable, coefficients, output):
    left,inside = operators(coefficients)
    shift = audit(coefficients)['separate_viscous_d1_shift']
    coeffs = sparse(left,inside)
    output.mkdir(parents=True,exist_ok=False)
    report = {}
    for ny in (25,49,97,193):
        state = {}
        reference = test(ny,left,inside,shift,frozen=state)
        expected = state['expected']
        _,nx,_ = expected.shape
        mu,kappa,jac = (np.broadcast_to(state[key][:,None],(ny,nx)) for key in ('mu','kappa','jac'))
        for rotated in (False,True):
            vector = state['u'][:,:,::-1] if rotated else state['u']
            dims = [1,ny,nx] if rotated else [nx,ny,1]
            inv = [1.,1/state['he'],1/state['dx']] if rotated else [1/state['dx'],1/state['he'],1.]
            fields = (*np.moveaxis(vector,-1,0),state['t'],mu,kappa,jac)
            arrays = (*coeffs,state['h0'],np.array([state['gamma']]),state['dleft'],
                      np.array(dims,dtype=np.int32),np.array(inv),*(a.T if rotated else a for a in fields))
            tag = f'{"zy" if rotated else "xy"}-n{ny}'
            inp,out = output/f'{tag}.bin',output/f'{tag}_results.bin'
            with inp.open('xb') as file:
                for array in arrays:
                    np.ascontiguousarray(array).tofile(file)
            subprocess.run([str(executable),str(inp),str(out),'tensor'],check=True)
            raw = np.fromfile(out)
            if rotated:
                cpu,gpu = raw.reshape(2,nx,ny,8).transpose(0,2,1,3)[:,:,:,[2,1,0,3,6,5,4,7]]
            else:
                cpu,gpu = raw.reshape(2,ny,nx,8)
            errors = np.max(np.abs(np.stack((cpu-expected,gpu-expected,gpu-cpu))),axis=(1,2))/np.maximum(
                1.,np.max(np.abs(expected),axis=(0,1)))
            if not np.all(np.isfinite(errors)) or np.any(errors>2.e-10):
                raise AssertionError(f'W1 tensor phase mismatch at {tag}: {errors}')
            h = np.ones(ny)
            h[:6],h[-6:] = state['h0'],state['h0'][::-1]
            weight = h[:,None]*jac*state['he']*state['dx']
            for field in (cpu,gpu):
                rate = np.sum(weight[:,:,None]*state['u']*field[:,:,:3])
                if not np.isclose(rate,reference['kinetic_rate'],rtol=2.e-10):
                    raise AssertionError(f'W1 kinetic energy rate mismatch: {rate}')
                balance = abs(np.sum(weight*field[:,:,3])-state['dx']*np.sum(field[:,:,-1]))
                if balance>3.e-11:
                    raise AssertionError(f'W1 total energy balance mismatch: {balance}')
            report[tag] = dict(host=errors[0].tolist(),cuda=errors[1].tolist(),same_state=errors[2].tolist(),
                               reference_kinetic_rate=reference['kinetic_rate'])
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--executable',type=Path,required=True)
    p.add_argument('--coefficients',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(run(args.executable.resolve(),args.coefficients,args.output)))
