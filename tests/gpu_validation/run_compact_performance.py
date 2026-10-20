#!/usr/bin/env python3
"""CP9 interleaved explicit/compact GPU performance matrix, local only."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
CASES={'tgv':('256,256,256',2.5e-4),'plate':('450,130,96',.005)}


def summarize(root,case):
    rows=[]
    reports={}
    identities=set()
    for path in sorted((root/'timing'/case).glob('*/summary.json')):
        record=json.loads(path.read_text())
        identities.add(tuple(record[name] for name in (
            'executable_sha256','crashfix','deltat','steps','warmup','workspace',
            'compact_sync','transport','batch_rhs'))+(tuple(record['grid_intervals']),))
        key=(record['scheme'],record['filter'],record['topology'])
        reports.setdefault(key,[]).append(record)
    expected={(scheme,filtering,f'{ranks},1,1') for scheme,filtering,ranks in
              itertools.product(('643e','643c'),('f','t'),(1,2))}
    if reports.keys()!=expected:
        raise ValueError('formal timing requires all eight configurations')
    if len(identities)!=1:
        raise ValueError('formal timing cannot mix executable, crashfix or runtime settings')
    for (scheme,filtering,topology),items in sorted(reports.items()):
        if len(items)!=3 or any(r['profiled'] or r['phase_timing'] or r['preflight'] for r in items):
            raise ValueError('formal timing requires three unprofiled repetitions')
        values=[r['rk_seconds_per_step']*1000 for r in items]
        rows.append(dict(case=case,scheme=scheme,filter=filtering,np=int(topology.split(',')[0]),
                         crashfix=items[0]['crashfix'],
                         median_ms=statistics.median(values),min_ms=min(values),max_ms=max(values),
                         samples_ms=values))
    by_key={(r['scheme'],r['filter'],r['np']):r for r in rows}
    for row in rows:
        row['compact_over_explicit']=row['median_ms']/by_key['643e',row['filter'],row['np']]['median_ms']
        row['speedup_2']=by_key[row['scheme'],row['filter'],1]['median_ms']/by_key[row['scheme'],row['filter'],2]['median_ms']
        row['efficiency_2']=row['speedup_2']/2
    (root/f'{case}_performance.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (root/f'{case}_performance.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=[k for k in rows[0] if k!='samples_ms'])
        writer.writeheader()
        writer.writerows({k:v for k,v in row.items() if k!='samples_ms'} for row in rows)
    print(json.dumps(rows,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--case',choices=CASES,required=True)
    parser.add_argument('--mode',choices=('preflight','timing','phases','summarize'),required=True)
    parser.add_argument('--crashfix',action='store_true',help='use lcracon=t in every plate configuration')
    args=parser.parse_args()
    if args.crashfix and args.case!='plate':
        parser.error('--crashfix is scoped to the profile plate')
    root=args.out.resolve()
    root.mkdir(parents=True,exist_ok=True)
    if args.mode=='summarize':
        summarize(root,args.case)
        return
    grid,dt=CASES[args.case]
    exe=args.exe.resolve(strict=True)
    with exe.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    configs=[(scheme,filtering,ranks) for filtering,ranks,scheme in
             itertools.product(('f','t'),(1,2),('643e','643c'))]
    if args.mode=='phases':
        configs=[item for item in configs if item[1]=='t']
    repeats=3 if args.mode=='timing' else 1
    for repeat in range(repeats):
        order=configs[repeat:]+configs[:repeat]
        if repeat==1:
            order=list(reversed(order))
        for scheme,filtering,ranks in order:
            name=f'{scheme}_filter{filtering}_np{ranks}'
            destination=root/args.mode/args.case/f'{name}_r{repeat+1}'
            if args.mode!='preflight':
                preflight=json.loads((root/'preflight'/args.case/f'{name}_r1/summary.json').read_text())
                if (preflight['executable_sha256']!=digest or preflight['status']!='pass' or
                    preflight['crashfix']!=args.crashfix):
                    raise ValueError('preflight does not match the frozen executable/crashfix mode')
            steps,warmup=(3,0) if args.mode=='preflight' else ((40,10) if args.mode=='timing' else (12,2))
            command=[sys.executable,str(ROOT/'tests/gpu_validation/run_compact_execution.py'),
                     '--exe',str(exe),'--out',str(destination),'--case',args.case,
                     '--grid',grid,'--deltat',str(dt),'--scheme',scheme,'--filter',filtering,
                     '--topology',f'{ranks},1,1','--steps',str(steps),'--warmup',str(warmup),
                     '--compact-sync','dependency','--transport','device-aware','--workspace','full']
            if args.crashfix:
                command+=['--crashfix']
            if args.mode=='preflight':
                command+=['--preflight']
            elif args.mode=='phases':
                command+=['--phase-timing']
            print(f'START {args.mode} {args.case} {name} round {repeat+1}',flush=True)
            subprocess.run(command,check=True)
    if args.mode=='timing':
        summarize(root,args.case)


if __name__=='__main__':
    main()
