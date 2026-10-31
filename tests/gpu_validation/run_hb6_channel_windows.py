#!/usr/bin/env python3
"""Advance one approved HB6 case to t*=1,5,10 with actual ASTR, not Python CFD."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path

from run_hb6_channel_case import estimated_dt, run


def windows(args):
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    args.dt=estimated_dt(args.grid,args.perturbation,args.wall)
    args.checkpoint_interval=round(1/args.dt)
    args.restore=None
    args.snapshots=False
    args.workspace='scalar'
    args.wall_mode='hermite7'
    root=args.output
    results=[]
    for time in (1,5,10):
        args.output=root/f't{time}'
        args.steps=time*args.checkpoint_interval
        with (root/f't{time}.analysis.log').open('w') as log,redirect_stdout(log):
            record=run(args)
        record['target_time']=time
        results.append(record)
        (root/'windows.json').write_text(json.dumps(results,indent=2)+'\n')
        print(f'{args.backend} bc{args.wall} {args.grid} perturbation={args.perturbation:g} '
              f't={time}: CFL={record["maximum_cfl"]:.8g}, '
              f'rho_min={record["minimum_density"]:.8g}, T_min={record["minimum_temperature"]:.8g}',flush=True)
        if args.wall==41 and args.perturbation==0 and args.grid==(128,128,64):
            assert record['velocity_reference_scaled']<=1e-3,record
            assert record['temperature_reference_scaled']<=1e-3,record
            assert max(record['wall_shear_relative_errors']+record['wall_heat_relative_errors'])<=.01,record
        args.restore=Path(record['checkpoint'])
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend',choices=('cpu','gpu'),required=True)
    parser.add_argument('--grid',type=lambda s:tuple(map(int,s.split(','))),default=(32,32,16))
    parser.add_argument('--topology',type=lambda s:tuple(map(int,s.split(','))),default=(1,1,1))
    parser.add_argument('--perturbation',type=float,default=0.)
    parser.add_argument('--wall',type=int,choices=(41,42,421),default=41)
    windows(parser.parse_args())
