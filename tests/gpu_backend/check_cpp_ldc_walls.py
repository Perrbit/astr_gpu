"""Check all LDC physical faces and lid-owned intersections at every saved RK stage."""
import argparse
import json
from pathlib import Path

import numpy as np

from run_cpp_multi_tgv_gate import layouts,raw_field


def check(case):
    partition=layouts(case);global_upper=np.max([np.array(p['origin'])+p['upper'] for p in partition.values()],axis=0)
    report=dict(case=str(case.resolve()),ranks=[],max_absolute_error=0.)
    for rank,layout in partition.items():
        paths=sorted((case/'outdat').glob(f'rhs.pre_rhs_primitives.*.rank{rank:08d}.bin'))
        if len(paths)!=12:raise AssertionError('expected four complete LDC steps, three stages each')
        origin=np.array(layout['origin']);upper=np.array(layout['upper']);indices=np.indices(tuple(upper+1))+origin[:,None,None,None]
        lid=indices[1]==global_upper[1]
        walls=(indices[0]==0)|(indices[0]==global_upper[0])|(indices[1]==0)|lid
        expected=np.zeros((*tuple(upper+1),3));expected[lid,0]=1.
        maximum=0.
        for path in paths:
            header,values=raw_field(path,primitive=True);h=header[3]
            if tuple(header[:3])!=tuple(upper):raise AssertionError('LDC primitive layout')
            values=values[tuple(slice(h,h+n+1) for n in upper)+(slice(None),)]
            if not np.isfinite(values).all():raise AssertionError('nonfinite physical LDC primitives')
            error=max(float(np.max(abs(values[...,1:4][walls]-expected[walls]))),
                float(np.max(abs(values[...,5][walls]-1.))))
            if error>2e-10:raise AssertionError(f'LDC wall velocity/temperature error: {path.name}: {error}')
            maximum=max(maximum,error)
        report['ranks'].append(dict(rank=rank,stages=len(paths),physical_wall_nodes=int(walls.sum()),
            lid_nodes=int(lid.sum()),lid_side_intersection_nodes=int((lid&((indices[0]==0)|(indices[0]==global_upper[0]))).sum()),
            max_absolute_error=maximum))
        report['max_absolute_error']=max(report['max_absolute_error'],maximum)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',type=Path);parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();result=check(args.case)
    args.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
