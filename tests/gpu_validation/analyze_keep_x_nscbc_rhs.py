"""Diagnose the uniform-tensor single-x NSCBC fixture without advancing any field."""
import argparse
import json
from pathlib import Path

import numpy as np

from compare_q_validation_snapshots import read_q_snapshot

SUFFIX = '.step00000000.rk01.rank00000000.bin'


def snapshot(case,phase):
    state = read_q_snapshot(case/'outdat'/('rhs.'+phase+SUFFIX))
    im,jm,km,hm,nq = state.header
    q = state.values.reshape((im+2*hm+1,jm+2*hm+1,km+2*hm+1,nq),order='F')
    return q[hm:hm+im+1,hm:hm+jm+1,hm:hm+km+1],(im,jm,km)


def boundary_rhs(case):
    with (case/'outdat'/('rhs.boundary'+SUFFIX)).open('rb') as stream:
        im,jm,km,nq = np.fromfile(stream,np.int32,4)
        return np.fromfile(stream,np.float64).reshape((im+1,jm+1,km+1,nq),order='F')


def euler_flux(q,axis):
    velocity = q[1:4]/q[0]
    pressure = .4*(q[4]-.5*np.dot(q[1:4],velocity))
    flux = q*velocity[axis]
    flux[axis+1] += pressure
    flux[4] += pressure*velocity[axis]
    return flux


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-case',type=Path,required=True)
    parser.add_argument('--gpu-case',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    current,dims = snapshot(args.cpu_case,'pre_rhs')
    other,other_dims = snapshot(args.gpu_case,'pre_rhs')
    initial,initial_dims = snapshot(args.cpu_case,'initialized')
    assert dims==other_dims==initial_dims
    im,jm,km = dims
    # The fixture is x,z in [0,2*pi], y in [0,pi], with uniform index spacing.
    areas = [None,(2*np.pi/im)*(2*np.pi/km),(2*np.pi/im)*(np.pi/jm)]
    observed = -boundary_rhs(args.cpu_case)-boundary_rhs(args.gpu_case)
    predicted = np.zeros_like(observed[im])
    for j in range(jm+1):
        for k in range(km+1):
            for axis in (1,2):
                difference = np.zeros(5)
                for side in (-1,1):
                    point = [im,j,k];point[axis] += side
                    wrapped = point.copy();wrapped[axis] %= dims[axis]
                    fresh = current[tuple(wrapped)]
                    stale = initial[tuple(wrapped)] if point[axis]<0 or point[axis]>dims[axis] else fresh
                    difference += side*(euler_flux(fresh,axis)-euler_flux(stale,axis))
                predicted[j,k] += .5*areas[axis]*difference
    residual = observed[im]-predicted
    report = dict(cpu_case=str(args.cpu_case),gpu_case=str(args.gpu_case),
        physical_q_max_abs=float(np.max(abs(current-other))),
        inlet_boundary_rhs_max_abs=float(np.max(abs(observed[0]))),
        outlet_boundary_rhs_max_abs=float(np.max(abs(observed[im]))),
        outlet_face_interior_max_abs=float(np.max(abs(observed[im,1:jm,1:km]))),
        predicted_outlet_stale_halo_max_abs=float(np.max(abs(predicted))),
        outlet_residual_after_stale_halo_correction=float(np.max(abs(residual))),
        interpretation='read-only analysis of saved first-stage data; inlet law mismatch and outlet stale tangential halo are separate')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    if report['physical_q_max_abs']>2e-10 or report['outlet_residual_after_stale_halo_correction']>2e-10:
        raise AssertionError(report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
