"""Read NP=2 y-slab KEEP wall failure snapshots without advancing or repairing."""
import argparse
import json
from pathlib import Path

import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',type=Path)
    parser.add_argument('--step',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    records=[]
    for stage in (1,2,3):
        for phase in ('pre_rhs','post_update'):
            for rank in (0,1):
                path=args.case/'outdat'/f'rhs.{phase}.step{args.step:08d}.rk{stage:02d}.rank{rank:08d}.bin'
                # This diagnostic retains invalid values; the acceptance reader rejects them.
                with path.open('rb') as stream:
                    header=np.fromfile(stream,np.int32,5)
                    values=np.fromfile(stream,np.float64)
                if header.size!=5:
                    raise AssertionError('Incomplete failure snapshot header')
                im,jm,km,hm,nq=map(int,header)
                if nq!=5 or min(im,jm,km,hm)<0:
                    raise AssertionError('Failure analysis requires five conservative variables')
                full=values.reshape((im+2*hm+1,jm+2*hm+1,km+2*hm+1,nq),order='F')
                q=full[hm:hm+im+1,hm:hm+jm+1,hm:hm+km+1,:]
                rho=q[...,0]
                with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
                    internal=q[...,4]-.5*np.sum(q[...,1:4]*(q[...,1:4]/rho[...,None]),axis=-1)
                    pressure=.4*internal
                    temperature=(1.4*.3**2)*pressure/rho
                valid=np.isfinite(q).all(axis=-1)&np.isfinite(internal)&np.isfinite(temperature)
                valid&=(rho>0)&(internal>0)&(temperature>0)
                record={'stage':stage,'phase':phase,'rank':rank,'invalid_nodes':int(np.count_nonzero(~valid)),
                        'nonfinite_q_values':int(np.count_nonzero(~np.isfinite(q))),'minima':{}}
                for name,field in (('rho',rho),('internal_energy',internal),('pressure',pressure),('temperature',temperature)):
                    mask=np.isfinite(field)
                    if mask.any():
                        index=np.unravel_index(np.argmin(np.where(mask,field,np.inf)),field.shape)
                        record['minima'][name]={'value':float(field[index]),
                            'global_ijk':[int(index[0]),int(index[1]+rank*jm),int(index[2])],
                            'nonfinite_nodes':int(np.count_nonzero(~mask))}
                    else:
                        record['minima'][name]={'value':None,'global_ijk':None,'nonfinite_nodes':int(field.size)}
                if record['invalid_nodes']:
                    i,j,k=map(int,np.argwhere(~valid)[0]);global_j=j+rank*jm
                    record['first_invalid']={'global_ijk':[i,global_j,k],
                        'conservative_state':[float(x) if np.isfinite(x) else str(x) for x in q[i,j,k,:]]}
                    if (rank==0 and j==0) or (rank==1 and j==jm):
                        direction=1 if j==0 else -1
                        p1=float(pressure[i,j+direction,k]);p2=float(pressure[i,j+2*direction,k])
                        samples={'interior_p1':p1,'interior_p2':p2,
                            'predicted_p_wall':(4*p1-p2)/3,'actual_p_wall':float(pressure[i,j,k])}
                        record['wall_extrapolation']={key:float(value) if np.isfinite(value) else str(value)
                                                     for key,value in samples.items()}
                records.append(record)
    first=next((r for r in records if r['invalid_nodes']),None)
    report={'scope':'existing NP=2 y-slab snapshots, no repair or CFD replay','step':args.step,
            'case':str(args.case),'first_observed_invalid_record':first,'records':records}
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(first,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
