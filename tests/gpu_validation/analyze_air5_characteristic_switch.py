#!/usr/bin/env python3
"""Diagnose saved RK switch contributions; never advance a flow state."""
import argparse
import json
from pathlib import Path

import numpy as np

from air5_radau_reference import Air5RadauReference
from run_air5_characteristic_backend_gate import fields
from run_air5_sbli_preflight import ROOT


def convective_projection(q, vector, model):
    velocity = q[1:4]/q[0]
    partial = q[5:10]
    beta = (partial@model.gas_constant)/(partial@model.cv_tr)
    temperature = (q[4]-q[10]-.5*(q[1:4]@q[1:4])/q[0]-
                   partial@model.formation_energy)/(partial@model.cv_tr)
    pressure = temperature*(partial@model.gas_constant)
    sound = np.sqrt((1+beta)*pressure/q[0])
    dpdq = np.r_[.5*beta*(velocity@velocity), -beta*velocity, beta,
                 (model.gas_constant-beta*model.cv_tr)*temperature-
                 beta*model.formation_energy, -beta]
    dp = dpdq@vector
    dm = vector[2]-velocity[1]*vector[0]
    waves = np.tile(q/q[0], (2, 1))
    signs = np.array([-1., 1.])
    waves[:, 2] = velocity[1]+signs*sound
    waves[:, 4] = (q[4]+pressure)/q[0]+signs*sound*velocity[1]
    result = (vector-waves[0]*(dp-sound*dm)/(2*sound**2)-
              waves[1]*(dp+sound*dm)/(2*sound**2))
    if not np.isfinite(result).all():
        raise ValueError('nonfinite characteristic projection')
    return result


def analyze(root):
    contract = json.loads((root/'contract.json').read_text())
    model = Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json')
    records = []
    for name, steps in zip(('coarse', 'medium', 'fine'), contract['steps']):
        case = root/name
        meta = json.loads((case/'gate.json').read_text())
        profile = np.loadtxt(case/'datin/air5_hbl_profile.dat')[-1]
        _, rho, u, v, w, _, temperature, tv, *fractions = profile
        if any(value != 0 for value in (u, v, w)):
            raise ValueError('this diagnostic requires the stationary reservoir')
        partial = rho*np.array(fractions)
        ev = model.ev_from_tv(partial, tv)
        target = np.r_[rho, 0., 0., 0.,
                       model.q5_from_state(rho, np.zeros(3), partial, ev, temperature),
                       partial, ev]
        step = steps-1
        data = fields(case, (step,))
        points = []
        for i in (1, 15):
            increment = np.zeros(11)
            stages = []
            for stage, weight in ((1, 1/6), (2, 1/6), (3, 2/3)):
                q = data[f'{step}:pre_rhs:{stage}'][i, -1, 0]
                velocity = float(q[2]/q[0])
                rhs = convective_projection(q, -(q-target)/meta['tau'], model)
                if velocity < 0:
                    increment += meta['dt']*weight*rhs
                stages.append(dict(stage=stage, normal_velocity=velocity,
                                   incoming=velocity < 0, relaxation_rhs=rhs.tolist()))
            points.append(dict(i=i, k=0, stages=stages, increment=increment.tolist(),
                               final=data[f'{step}:post_chemistry:2'][i, -1, 0].tolist()))
        records.append(dict(case=name, dt=meta['dt'], last_step=step, points=points))
    differences = []
    for left, right in zip(records[:-1], records[1:]):
        for a, b in zip(left['points'], right['points']):
            predicted = np.array(a['increment'])-b['increment']
            observed = np.array(a['final'])-b['final']
            differences.append(dict(pair=f"{left['case']}/{right['case']}", i=a['i'],
                observed=observed.tolist(), switched_relaxation=predicted.tolist(),
                residual=(observed-predicted).tolist(),
                ratio=[float(p/o) if o != 0 else None for p, o in zip(predicted, observed)]))
    return dict(scope='saved-stage switch attribution, not a new acceptance gate',
                rk_weights=[1/6, 1/6, 2/3], records=records, differences=differences)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.root)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    for row in report['differences']:
        print(row['pair'], 'i=', row['i'], 'rho/E/Ev attribution=',
              [row['ratio'][j] for j in (0, 4, 10)])
