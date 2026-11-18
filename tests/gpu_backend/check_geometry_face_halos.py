"""Compare saved metric face halos with their published global physical nodes."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def check(path):
    report=dict(source=str(path.resolve()),ranks=[])
    with h5py.File(path) as source:
        shape=source['identity'][3:6].astype(int)
        if int(source['identity'][6])!=13:
            raise AssertionError('expected coordinates, Jacobian and nine metrics')
        global_fields=np.stack([source[f'q{c:04}'][...].transpose(2,1,0) for c in range(4,14)],axis=-1)
        word_type={4:np.uint32,8:np.uint64}[global_fields.dtype.itemsize]
        report['numerical_dtype']=str(global_fields.dtype)
        offset=0
        for rank,part in enumerate(source['partitions'][...].reshape(-1,8)):
            origin=part[:3].astype(int);cells=part[3:6].astype(int);halo=int(part[6]);count=int(part[7])
            full=tuple(cells+2*halo+1);owned=cells+(origin+cells==shape-1)
            indices=np.indices(full)
            owner=np.all((indices>=halo)&(indices<(halo+owned)[:,None,None,None]),axis=0)
            extra=~owner.ravel(order='F');width=int(extra.sum())
            if count!=13*width:
                raise AssertionError('geometry extras shape')
            values=source['rank_extras'][offset:offset+count].reshape((13,width));offset+=count
            local=np.empty((*full,10),dtype=global_fields.dtype,order='F')
            for c in range(10):
                local[...,c].ravel(order='F')[extra]=values[c+3]
                local[tuple(slice(halo,halo+n) for n in owned)+(c,)]=global_fields[
                    tuple(slice(o,o+n) for o,n in zip(origin,owned))+(c,)]
            physical=(indices>=halo)&(indices<=(halo+cells)[:,None,None,None])
            face=np.count_nonzero(~physical,axis=0)==1
            global_indices=indices-halo+origin[:,None,None,None]
            # No physical-exterior or wrapped-periodic assumption is needed:
            # compare only internal MPI halos with an actual global owner.
            selected=face&np.all((global_indices>=0)&(global_indices<shape[:,None,None,None]),axis=0)
            points=np.where(selected)
            expected=global_fields[tuple(axis[selected] for axis in global_indices)]
            actual=local[points]
            shared=np.all(physical,axis=0)&~owner
            shared_actual=local[np.where(shared)]
            shared_expected=global_fields[tuple(axis[shared] for axis in global_indices)]
            error=abs(actual-expected)
            maximum=error.max(axis=0) if error.size else np.zeros(10)
            scaled=maximum/np.maximum(1.,np.max(abs(global_fields),axis=(0,1,2)))
            record=dict(rank=rank,internal_face_nodes=int(selected.sum()),
                max_absolute_per_component=maximum.tolist(),max_scaled_per_component=scaled.tolist())
            record['owner_word_mismatches']=int(np.count_nonzero(actual.view(word_type)!=expected.view(word_type)))
            record['shared_nodes']=int(shared.sum())
            record['shared_owner_word_mismatches']=int(np.count_nonzero(shared_actual.view(word_type)!=shared_expected.view(word_type)))
            if error.size:
                row,c=np.unravel_index(error.argmax(),error.shape)
                record['worst']=dict(component=int(c+4),local_ijk=[int(p[row]-halo) for p in points],
                    actual=float(actual[row,c]),owner_value=float(expected[row,c]))
            report['ranks'].append(record)
        if offset!=source['rank_extras'].size:
            raise AssertionError('geometry extras length')
    report['max_scaled_error']=max(max(r['max_scaled_per_component']) for r in report['ranks'])
    report['passed']=report['max_scaled_error']<=2e-10
    report['owner_words_equal']=all(r['owner_word_mismatches']==0 for r in report['ranks'])
    report['shared_owner_words_equal']=all(r['shared_owner_word_mismatches']==0 for r in report['ranks'])
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('geometry',type=Path);parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();result=check(args.geometry)
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['passed']:
        raise SystemExit('metric face halos differ from the published owner state')
