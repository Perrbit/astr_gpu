"""Independent ideal-gas projector algebra; not a production boundary validation."""
import argparse
import json
from pathlib import Path

import numpy as np


def acoustic_projector(q,axis,sign,gamma):
    rho=q[0]
    velocity=q[1:4]/rho
    kinetic=0.5*np.dot(velocity,velocity)
    pressure=(gamma-1)*(q[4]-rho*kinetic)
    sound=np.sqrt(gamma*pressure/rho)
    enthalpy=(q[4]+pressure)/rho
    right=np.r_[1.0,velocity,enthalpy]
    right[axis+1]+=sign*sound
    right[4]+=sign*sound*velocity[axis]
    dp=np.r_[(gamma-1)*kinetic,-(gamma-1)*velocity,gamma-1]
    momentum=np.zeros(5)
    momentum[0]=-velocity[axis]
    momentum[axis+1]=1.0
    left=(dp+sign*sound*momentum)/(2*sound**2)
    return np.outer(right,left)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    gamma=1.4;mach=0.3
    report={'scope':'independent constant-state algebra only; no new closure is implemented or approved','cases':[]}
    identity=np.eye(5)
    raw=np.array([0.1,-0.2,0.3,-0.4,0.5])
    for v in (-1e-4,0.0,1e-4):
        q=np.array([1.0,1.0,v,0.0,1/(gamma*(gamma-1)*mach**2)+0.5*(1+v*v)])
        px=acoustic_projector(q,0,-1,gamma)
        py=acoustic_projector(q,1,-1,gamma)
        if v<0:
            py=identity-acoustic_projector(q,1,1,gamma)
        candidate=(identity-py)@raw
        xy=(identity-py)@(identity-px)@raw
        yx=(identity-px)@(identity-py)@raw
        record={'top_normal_velocity':v,'x_incoming_rank':int(np.linalg.matrix_rank(px)),
                'y_incoming_rank':int(np.linalg.matrix_rank(py)),
                'projector_identity_max_abs':float(max(np.max(abs(px@px-px)),np.max(abs(py@py-py)))),
                'commutator_max_abs':float(np.max(abs(px@py-py@px))),
                'sequential_order_difference_max_abs':float(np.max(abs(xy-yx))),
                'candidate_y_incoming_residual':float(np.max(abs(py@candidate))),
                'candidate_y_outgoing_change':float(np.max(abs((identity-py)@candidate-(identity-py)@raw))),
                'candidate_rhs':candidate.tolist()}
        report['cases'].append(record)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
