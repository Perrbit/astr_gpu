"""Explain the frozen low-order GRCBC outlet/top spanwise-transport discrepancy."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from compare_q_validation_snapshots import read_q_snapshot
from run_keep_y_farfield_gate import phase
from run_output_restart_validation import compare_fields


def physical_state(case):
    snapshot=read_q_snapshot(case/'outdat/rhs.pre_rhs.step00000000.rk01.rank00000000.bin')
    im,jm,km,hm,nq=snapshot.header
    values=snapshot.values.reshape((im+2*hm+1,jm+2*hm+1,km+2*hm+1,nq),order='F')
    return values[hm:hm+im+1,hm:hm+jm+1,hm:hm+km+1]


def analyze(cpu,gpu):
    q=physical_state(cpu);other=physical_state(gpu)
    input_error=float(np.max(abs(q-other)/np.maximum(1,np.maximum(abs(q),abs(other)))))
    compare_fields(cpu/'outdat/new/resources/geometry.h5',gpu/'outdat/new/resources/geometry.h5')
    with h5py.File(cpu/'outdat/new/resources/geometry.h5') as geometry:
        jacobian=geometry['q0004'][:].transpose(2,1,0)
        metric=np.empty(q.shape[:3]+(3,3))
        for axis in range(3):
            for component in range(3):
                metric[...,axis,component]=geometry[f'q{5+axis+3*component:04d}'][:].transpose(2,1,0)
    parameters=json.loads((cpu/'parameters.json').read_text())
    gamma=parameters['gamma'];n=q.shape[2]-1
    flux=[]
    for k in range(n+1):
        state=q[-1,-1,k];rho=state[0];velocity=state[1:4]/rho
        pressure=(gamma-1)*(state[4]-.5*np.dot(state[1:4],velocity))
        direction=metric[-1,-1,k,2];un=np.dot(direction,velocity)
        value=state*un;value[1:4]+=pressure*direction;value[4]+=pressure*un
        flux.append(jacobian[-1,-1,k]*value)
    flux=np.asarray(flux)
    dz=np.stack([.5*(flux[(k+1)%n]-flux[(k-1)%n]) for k in range(n+1)])
    observed=-phase(cpu,'boundary')-phase(gpu,'boundary')
    full=phase(cpu,'full')-phase(gpu,'full')
    predicted=[]
    for k in range(n+1):
        state=q[-1,-1,k];rho=state[0];velocity=state[1:4]/rho
        pressure=(gamma-1)*(state[4]-.5*np.dot(state[1:4],velocity))
        sound=np.sqrt(gamma*pressure/rho)
        normal=metric[-1,-1,k,1]/np.linalg.norm(metric[-1,-1,k,1])
        un=np.dot(velocity,normal)
        dp=np.r_[.5*(gamma-1)*np.dot(velocity,velocity),-(gamma-1)*velocity,gamma-1]
        dun=np.r_[-un,normal,0.]
        right=np.r_[1.,velocity-sound*normal,(state[4]+pressure)/rho-sound*un]
        incoming_acoustic=np.outer(right,(dp-sound*dun)/(2*sound**2))
        # The convective compatibility retains transverse entropy/shear transport.
        predicted.append((np.eye(5)-incoming_acoustic)@dz[k])
    diffusion=[phase(case,'incident_pre')-phase(case,'conv') for case in (cpu,gpu)]
    report=dict(scope='read-only frozen state; no new flow advance',
        hypothesis='CPU low-order GRCBC omits outlet/top z transport; GPU includes it once',
        input_max_scaled=input_error,
        raw_boundary_difference_max=float(np.max(abs(observed))),
        noncorner_boundary_difference_max=max(float(np.max(abs(observed[:-1]))),
                                              float(np.max(abs(observed[-1,:-1])))),
        missing_z_rhs_max=float(np.max(abs(dz))),
        raw_prediction_residual=float(np.max(abs(dz-observed[-1,-1]))),
        full_difference_max=float(np.max(abs(full))),
        complete_rhs_prediction_residual=float(np.max(abs(np.asarray(predicted)-full[-1,-1]))),
        diffusion_difference_max=float(np.max(abs(diffusion[0]-diffusion[1]))))
    if report['raw_boundary_difference_max']<=2e-10:
        raise AssertionError('Captured case does not exhibit the diagnosed discrepancy')
    if max(input_error,report['raw_prediction_residual'],report['complete_rhs_prediction_residual'],
           report['noncorner_boundary_difference_max'],report['diffusion_difference_max'])>2e-10:
        raise AssertionError(f'The missing-z hypothesis does not explain the frozen case: {report}')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cpu','gpu','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    report=analyze(args.cpu,args.gpu)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
