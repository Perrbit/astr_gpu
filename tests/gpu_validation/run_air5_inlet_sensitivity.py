#!/usr/bin/env python3
"""Three inlet-only restarts from one immutable ASTR Mach4 precursor checkpoint."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from run_air5_sbli_long import run, digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    baseline=args.baseline.resolve()
    output.mkdir(parents=True,exist_ok=False)
    checkpoint=baseline/'outdat/flowfield.h5'
    checksum=digest(checkpoint)
    results=[]
    for name,scale in [('control',1.),('thin',.9),('thick',1.1)]:
        if digest(checkpoint)!=checksum:
            raise ValueError('source checkpoint changed during inlet experiment')
        destination=output/name
        run(SimpleNamespace(output=destination,case='mach4-precursor',grid=None,
            baseline=baseline,executable=baseline.parent/'astr',updates=501,
            dt=2e-9,checkpoint=250,timeout=600.,inlet_thickness_scale=scale))
        status=json.loads((destination/'status.json').read_text())
        if status['status']!='completed-pending-final-field-review':
            raise ValueError('inlet experiment did not complete its bounded run')
        results.append(dict(case=name,scale=scale,status=status))
        (output/'matrix.json').write_text(json.dumps(dict(
            status='partial' if len(results)<3 else 'completed-pending-comparison',
            source_checkpoint=str(checkpoint),source_checkpoint_sha256=checksum,
            cases=results),indent=2)+'\n')


if __name__=='__main__':
    main()
