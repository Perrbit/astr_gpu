"""Read immutable time blocks, coalescing exact partial snapshots of the same block."""
import argparse
import json
from pathlib import Path
import re

import numpy as np


def sampling_key(definition):
    temporal=definition['temporal']
    return json.dumps(dict(source=definition['source'],statistics_source=definition.get('statistics_source',''),
        sampling=definition['sampling'],fields=definition['fields'] if 'fields' in definition else [definition['field']],
        sample_steps=temporal.get('sample_steps',1),segment_length=temporal.get('segment_length',0),
        fluctuation=definition.get('fluctuation',False)),sort_keys=True,separators=(',',':'))


def read_histories(paths):
    groups={}
    for path in map(Path,paths):
        match=re.fullmatch(r'block\.g(\d+)\.n(\d+)\.(complete|partial\.step\d+)\.csv',path.name)
        if match is None: raise ValueError(f'Not a temporal block: {path}')
        with path.open() as stream:
            definition_line=stream.readline()
        prefix='# phase=completed_step; definition='
        if not definition_line.startswith(prefix): raise ValueError(f'Missing sampling definition: {path}')
        definition=json.loads(definition_line[len(prefix):])
        with path.open() as stream:
            data=np.genfromtxt((line for line in stream if not line.startswith('#')),
                delimiter=',',names=True,dtype=None,encoding='utf-8')
        data=np.atleast_1d(data)
        key=(definition['id'],int(match[1]),sampling_key(definition))
        block=int(match[2])
        slots=groups.setdefault(key,{})
        if block in slots:
            previous=slots[block]
            names=data.dtype.names
            if names!=previous.dtype.names: raise ValueError(f'Temporal fields differ: {path}')
            shorter,longer=(data,previous) if len(data)<=len(previous) else (previous,data)
            for field in names:
                if not np.array_equal(shorter[field],longer[field][:len(shorter)]):
                    raise ValueError(f'Conflicting temporal snapshot {path}: {field}')
            slots[block]=longer
        else:
            slots[block]=data
    result={}
    for key,blocks in groups.items():
        result[key]=np.concatenate([blocks[number] for number in sorted(blocks)])
        identities=np.column_stack([result[key]['step'],result[key]['node']])
        if len(np.unique(identities,axis=0))!=len(identities):
            raise ValueError(f'Duplicate samples across committed blocks: {key[:2]}')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths',nargs='+',type=Path)
    args=parser.parse_args()
    for key,data in read_histories(args.paths).items():
        print(json.dumps(dict(product=key[0],history_generation=key[1],rows=len(data),
            samples=len(np.unique(data['step'])),first_time=float(data['time'].min()),last_time=float(data['time'].max()),
            gap_steps=np.unique(data['step'][data['gap_before']!=0]).tolist())))


if __name__=='__main__': main()
