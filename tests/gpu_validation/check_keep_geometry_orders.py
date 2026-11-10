"""Check G1 geometry/free-stream residuals and three-level analytic convergence."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports',type=Path,nargs=3,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    data=[json.loads(path.read_text()) for path in args.reports]
    errors=[]
    for item in data:
        if max(item['metric_identity_max_per_component'])>2e-10 or \
           max(item['constant_state_rhs_max_per_component'])>2e-10:
            raise AssertionError('actual metric identity or constant-state physical RHS exceeds G1 gate')
        accuracy=item['accuracy']
        errors.append([accuracy['jacobian_scaled_max_error'],accuracy['inverse_scaled_max_error'],
                       *accuracy['physical_gradient_max_errors']])
    sizes=np.array([x['intervals'] for x in data])
    if not np.array_equal(sizes[1:],2*sizes[:-1]):
        raise AssertionError('expected successive grid doubling')
    errors=np.array(errors)
    orders=np.log2(errors[:-1]/errors[1:])
    report=dict(reports=list(map(str,args.reports)),quantities=['J_scaled','G_scaled','grad_x','grad_y','grad_z'],
                intervals=sizes.tolist(),errors=errors.tolist(),orders=orders.tolist())
    if not np.isfinite(orders).all() or np.any(orders<5.5):
        raise AssertionError(f'geometry/physical scalar gradient lacks sixth-order asymptotic convergence: {orders}')
    report['status']='passed'
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
