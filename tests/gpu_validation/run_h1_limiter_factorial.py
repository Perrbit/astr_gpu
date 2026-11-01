#!/usr/bin/env python3
"""Native two-factor H1 diagnosis; failed variants are evidence, not admitted cases."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import test_grcbc_oblique_shock as native


def frames(case):
    rows=[]
    for path in sorted((case/'outdat/new/fields').glob('segment*/step*/data.h5')):
        if '.tmp' in str(path):
            continue
        with h5py.File(path) as file:
            rho=file['density'][:]
            velocity=file['velocity'][:]
            pressure=file['pressure'][:]
            temperature=file['temperature'][:]
        q=np.concatenate([rho[...,None],rho[...,None]*velocity,
            (pressure/.4+.5*rho*np.sum(velocity**2,axis=-1))[...,None]],axis=-1)
        fluctuation=q[:-1]-q[:-1].mean(axis=0,keepdims=True)
        top_rho=rho[:-1,-3:].mean(axis=0)
        top_velocity=velocity[:-1,-3:, :,0].mean(axis=0)
        rows.append(dict(step=int(path.parent.name[4:]),
            top_three_l2=float(np.linalg.norm(fluctuation[:,-3:])),
            top_mean_density_min=float(top_rho.min()),top_mean_density_max=float(top_rho.max()),
            top_mean_u_min=float(top_velocity.min()),top_mean_u_max=float(top_velocity.max()),
            rho_min=float(rho.min()),temperature_min=float(temperature.min())))
    return rows


def run(args):
    native.EXE=str(args.executable.resolve())
    args.output.mkdir(parents=True)
    report=dict(scope='diagnostic factorial, not numerical or physical admission',
                executable=native.EXE,source=str(args.source.resolve()),steps=args.steps,
                dt=native.DT,topology=[2,1,1],variants={})
    for variant in ['legacy','internal-only','near-only','both']:
        case=args.output/variant
        completed=False
        try:
            native.launch(case,'gpu',n=192,steps=args.steps,topology=(2,1,1),
                field_source=args.source,volume_interval=8,checkpoint_interval=32,h1_variant=variant)
            completed=True
        except AssertionError:
            text=(case/'run.log').read_text()
            if 'GRCBC boundary status (4=subsonic outlet/top corner): 2' not in text:
                raise
        text=(case/'run.log').read_text()
        assert 'H1_DIAGNOSTIC_VARIANT '+variant in text
        timeline=frames(case)
        report['variants'][variant]=dict(completed=completed,
            last_archived_step=timeline[-1]['step'] if timeline else None,
            timeline=timeline)
        (args.output/'factorial.json').write_text(json.dumps(report,indent=2)+'\n')
        print(variant,'completed' if completed else 'numerical failure captured',
              timeline[-1] if timeline else {},flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=225)
    run(parser.parse_args())
