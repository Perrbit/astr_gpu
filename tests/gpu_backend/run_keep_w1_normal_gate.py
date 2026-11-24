"""Compare native W1 normal terms on frozen strided lines with a dense reference."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'gpu_validation'))
from check_keep_w1_scalar import audit, d2_matrix, operators
from check_keep_w1_shear_heat import normal_operators


def sparse(left, inside):
    begin, value_idx, coef_idx, charge = [0], [], [], []
    for row, origin in [(v, 0) for v in left]+[(inside, 3)]:
        i, j = np.nonzero(row)
        value_idx.extend(i-origin)
        coef_idx.extend(j-origin)
        charge.extend(row[i, j])
        begin.append(len(charge))
    if len(charge) != 485:
        raise AssertionError('the native W1 table has 485 nonzero terms')
    return (*[np.asarray(x, dtype=np.int32) for x in (begin, value_idx, coef_idx)], np.asarray(charge))


def verify(executable, coefficients, output):
    left, inside = operators(coefficients)
    shift = audit(coefficients)['separate_viscous_d1_shift']
    pack = sparse(left, inside)
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for n in (25, 49, 97, 193):
        h0, _, e, s, gamma = normal_operators(n, left, inside, shift)
        nh, lanes = n-1, 13
        eta = np.linspace(0., 1., n)[:, None]
        offset = np.arange(lanes)[None, :] / lanes
        mapping = .17+.2*offset
        y = eta+mapping*eta*(1-eta)
        jac = 1+mapping*(1-2*eta)
        mu = 1+.2*np.exp(y+offset)
        b = mu/jac
        values = .03+.11*eta+.1*offset+np.sin((1+offset)*np.pi*y)
        expected = np.zeros((n, lanes, 3))
        for lane in range(lanes):
            bv, jv, u = b[:, lane], jac[:, lane], values[:, lane]
            q = bv[[0,-1], None]*s
            p = 2*(bv[[0,-1]]/h0[[0,-1]]+gamma*bv[[0,-1]]**2/[bv[:6].min(),bv[-6:].min()])
            a = d2_matrix(bv,left,inside)
            weight = nh**2/(h0*jv)
            jump = u[[0,-1]]
            wall_work = u*u/2
            expected[:,lane,0] = weight*(h0*(a@u)+q.T@jump-e@(p*jump))
            expected[:,lane,1] = weight*(h0*(a@wall_work)-e@(q@wall_work))
            expected[[0,-1],lane,2] = nh*(q@u-p*jump)
        inp, out = output/f'n{n}.bin', output/f'n{n}_results.bin'
        with inp.open('wb') as f:
            for array in (*pack, h0[:6], np.array([gamma]), np.array([n,lanes],dtype=np.int32),
                          np.array([nh,0.,0.]), values,b,jac):
                np.ascontiguousarray(array).tofile(f)
        subprocess.run([str(executable),str(inp),str(out)],check=True)
        cpu, gpu = np.fromfile(out).reshape(2,n,lanes,3)
        scale = np.maximum(1.,np.max(np.abs(expected),axis=(0,1)))
        er = np.max(np.abs(np.stack([cpu-expected,gpu-expected,gpu-cpu])),axis=(1,2))/scale
        if not np.all(np.isfinite(er)) or np.any(er>2.e-10):
            raise AssertionError(f'W1 normal term mismatch at {n}: {er}')
        results[n] = dict(cpu=er[0].tolist(),gpu=er[1].tolist(),same_state=er[2].tolist())
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--coefficients',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    results = verify(args.executable.resolve(),args.coefficients,args.output)
    (args.output/'report.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results))
