"""Pack immutable solver RK observations; perform no time integration."""
import argparse
import json
from pathlib import Path

import numpy as np

from run_cpp_multi_tgv_gate import raw_field
from run_cpp_tgv_gate import compare


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);report={}
    for family in ('mp7','mp7_roe','mp7ld_flat'):
        data=[]
        for backend in ('cpu','gpu'):
            case=args.reference/(family+'_'+backend)
            base_header,base=raw_field(case/'outdat/rhs.pre_rhs.step00000000.rk01.rank00000000.bin')
            steps=[]
            for stage in (1,2,3):
                suffix=f'.step00000000.rk{stage:02d}.rank00000000.bin'
                header,post=raw_field(case/('outdat/rhs.post_update'+suffix))
                if header!=base_header:raise AssertionError('RK packet header changed')
                g=np.asarray(header[:3]);h=header[3]
                post=post[tuple(slice(h,h+n+1) for n in g)+(slice(None),)]
                with (case/('outdat/rhs.full'+suffix)).open('rb') as source:
                    rh=np.fromfile(source,np.int32,4)
                    rhs=np.fromfile(source,np.float64).reshape((*g+1,5),order='F')
                if tuple(rh)!=(*g,5):raise AssertionError('full RHS packet shape')
                if not np.isfinite(post).all() or not np.isfinite(rhs).all():raise AssertionError('nonfinite full RK reference')
                with (args.output/f'{family}_{backend}.rk{stage:02d}.bin').open('xb') as target:
                    np.asarray(base_header[:4],np.int32).tofile(target)
                    for values in (base,rhs,post):values.ravel(order='F').tofile(target)
                steps.append((rhs,post))
            data.append(steps)
        report[family]=[dict(full_rhs=compare(a[0],b[0]),post_update=compare(a[1],b[1])) for a,b in zip(*data)]
    (args.output/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
    for path in args.output.rglob('*'):path.chmod(0o555 if path.is_dir() else 0o444)
    args.output.chmod(0o555);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
