"""Define H1 convection after prescribed-face zeros using existing immutable snapshots."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from run_cpp_tgv_gate import compare


def phase(case,label,stage):
    with (case/f'outdat/rhs.{label}.step00000000.rk{stage:02d}.rank00000000.bin').open('rb') as stream:
        header=np.fromfile(stream,np.int32,4)
        return np.fromfile(stream,np.float64).reshape(tuple(header[:3]+1)+(header[3],),order='F')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.reference.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(source=str(source),phase='convection with prescribed inlet/lower-wall zeros, before complete characteristics',
        policy='preserve raw snapshots; project only the two existing prescribed faces in derived packets',cases={})
    for initial in ('entropy','shock'):
        (out/initial).mkdir();entries=[]
        for stage in (1,2,3):
            raw=[phase(source/f'{initial}_{b}','conv',stage) for b in ('cpu','gpu')]
            complete=[phase(source/f'{initial}_{b}','post_source',stage) for b in ('cpu','gpu')]
            free=np.ones(raw[0].shape[:3],bool);free[0,:,:]=False;free[:,0,:]=False
            for value in complete:
                if not np.isfinite(value).all() or np.any(value[~free]!=0):
                    raise AssertionError('existing complete H1 RHS does not zero prescribed faces')
            raw_error=compare(raw[1][free],raw[0][free]);complete_error=compare(complete[1],complete[0])
            entry=dict(stage=stage,raw_free_scaled=raw_error,complete_all_scaled=complete_error,
                raw_nonfinite_prescribed=[int((~np.isfinite(value[~free])).sum()) for value in raw],packets={})
            for backend,value in zip(('cpu','gpu'),raw):
                packet=source/initial/f'mp7ld_flat_{backend}.rk{stage:02d}.bin'
                with packet.open('rb') as stream:
                    header=np.fromfile(stream,np.int32,9);upper=header[:3];halo=header[3]
                    nodes=int(np.prod(upper+1));full=int(np.prod(upper+2*halo+1))
                    stream.seek(0);prefix_bytes=9*4+4*8+21*full*8+13*nodes*8+nodes
                    prefix=stream.read(prefix_bytes);previous=np.fromfile(stream,np.float64)
                if previous.size!=nodes*5 or value.shape!=tuple(upper+1)+(5,):
                    raise AssertionError('H1 frozen packet layout')
                derived=value.copy();derived[~free]=0
                target=out/initial/packet.name
                with target.open('xb') as stream:
                    stream.write(prefix);derived.ravel(order='F').tofile(stream)
                entry['packets'][backend]=dict(source_sha256=hashlib.sha256(packet.read_bytes()).hexdigest(),
                    derived_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
            entries.append(entry)
        report['cases'][initial]=entries
    shutil.copyfile(__file__,out/'packaging.py')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({name:[max(row['raw_free_scaled']) for row in rows] for name,rows in report['cases'].items()},indent=2))


if __name__=='__main__':main()
