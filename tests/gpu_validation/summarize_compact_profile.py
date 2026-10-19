#!/usr/bin/env python3
"""Summarize recorded CUDA/MPI activity without treating API sums as wall time."""
import argparse
import json
from pathlib import Path
import sqlite3


def summarize(path):
    with sqlite3.connect(path) as db:
        def durations(table, name):
            query=(f'SELECT s.value,COUNT(*),SUM(a.end-a.start)*1e-9 FROM {table} a '
                   f'JOIN StringIds s ON a.{name}=s.id GROUP BY a.{name} ORDER BY SUM(a.end-a.start) DESC')
            return [dict(name=n,calls=count,seconds=seconds) for n,count,seconds in db.execute(query)]
        copies=[dict(kind=kind,calls=count,bytes=total,seconds=seconds) for kind,count,total,seconds in
            db.execute('SELECT copyKind,COUNT(*),SUM(bytes),SUM(end-start)*1e-9 '
                       'FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind')]
        mpi={table:[dict(name=name,calls=count,bytes=total,seconds=seconds) for name,count,total,seconds in
            db.execute(f'SELECT s.value,COUNT(*),SUM(a.size),SUM(a.end-a.start)*1e-9 FROM {table} a '
                       'JOIN StringIds s ON a.textId=s.id GROUP BY a.textId')]
            for table in ('MPI_P2P_EVENTS','MPI_COLLECTIVES_EVENTS')}
        return dict(file=str(path),cuda_api=durations('CUPTI_ACTIVITY_KIND_RUNTIME','nameId'),
                    kernels=durations('CUPTI_ACTIVITY_KIND_KERNEL','shortName'),copies=copies,mpi=mpi,
                    mpi_other=durations('MPI_OTHER_EVENTS','textId'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    paths=sorted(args.directory.glob('*.sqlite'))
    if not paths:
        raise ValueError('no Nsight SQLite exports in the specified directory')
    result=dict(scope='whole profiled process, including initialization; API and GPU times overlap',
                ranks=[summarize(path) for path in paths])
    destination=args.directory/'profile_summary.json'
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(destination)


if __name__=='__main__':
    main()
