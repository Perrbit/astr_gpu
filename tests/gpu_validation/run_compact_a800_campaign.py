#!/usr/bin/env python3
"""A800 FP64 TGV strong/weak scaling and CPU1--32 same-operator baselines.

Run `--mode manifest` before deployment. No Slurm submission is performed here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

from compact_benchmark import formal_configuration_matrix
from run_compact_a800_preflight import prepare_and_run


def case_name(row):
    return f"{row['family']}_{row['scheme']}_{row['backend']}_np{row['ranks']}_{''.join(map(str,row['topology']))}"


def summarize_runs(runs,repeats):
    groups={}
    for run in runs:
        groups.setdefault(case_name(run),[]).append(run)
    rows=[]
    for label,items in sorted(groups.items()):
        if len(items)!=repeats:
            continue
        costs=[item['median_rk_seconds'] for item in items]
        first=items[0]
        rows.append(dict(label=label,backend=first['backend'],scheme=first['scheme'],
            family=first['family'],ranks=first['ranks'],topology=first['topology'],grid=first['grid'],
            seconds=statistics.median(costs),minimum=min(costs),maximum=max(costs),repeats=repeats,
            samples_per_run=len(first['rank_max_rk_samples'])))
    cpu={(r['scheme'],r['ranks']):r['seconds'] for r in rows if r['backend']=='cpu'}
    single={(r['scheme'],r['family']):r['seconds'] for r in rows if r['backend']=='gpu' and r['ranks']==1}
    for row in rows:
        scheme=row['scheme'];n=row['ranks'];t=row['seconds']
        if row['backend']=='gpu':
            baseline=single.get((scheme,row['family']))
            if baseline is not None:
                row['speedup_or_weak_efficiency']=baseline/t
                row['efficiency']=baseline/t/(n if row['family']=='strong' else 1)
            if row['family']=='strong':
                row['speedup_vs_cpu']={str(cores):value/t for (s,cores),value in cpu.items() if s==scheme}
        elif (scheme,1) in cpu:
            row['cpu_speedup']=cpu[scheme,1]/t
            row['cpu_efficiency']=row['cpu_speedup']/n
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('manifest','timing','summarize'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cpu',type=Path)
    parser.add_argument('--gpu',type=Path)
    parser.add_argument('--backend',choices=('all','cpu','gpu'),default='all')
    parser.add_argument('--merge-from',type=Path,action='append',default=[],
                        help='other completed campaign directory for same-operator CPU/GPU ratios')
    parser.add_argument('--halo-transport',choices=('pinned','device-aware'),default='device-aware')
    parser.add_argument('--compact-transport',choices=('pinned','device-aware'),default='device-aware')
    parser.add_argument('--ucx-tls',default='self,sm,cuda_copy,cuda_ipc')
    args=parser.parse_args()
    args.out=args.out.resolve()
    if args.mode=='summarize':
        report=json.loads((args.out/'runs.json').read_text())
        for directory in args.merge_from:
            other=json.loads((directory/'runs.json').read_text())
            report['runs'].extend(other['runs'])
            report['failed'].update(other['failed'])
        summary=summarize_runs(report['runs'],5)
        (args.out/'performance.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(f'{len(summary)} complete configurations; {len(report["failed"])} failed/skipped')
        return
    matrix=formal_configuration_matrix(args.backend)
    if args.mode=='manifest':
        print(json.dumps(dict(configurations=len(matrix),backend=args.backend,
            gpu=sum(r['backend']=='gpu' for r in matrix),cpu=sum(r['backend']=='cpu' for r in matrix),repeats=5,
            gpu_steps=22,cpu_steps=7,warmup=2,matrix=matrix),indent=2))
        return
    if (args.backend in ('all','cpu') and args.cpu is None) or (args.backend in ('all','gpu') and args.gpu is None):
        parser.error('timing requires the executable for each selected backend')
    if args.cpu is not None:
        args.cpu=args.cpu.resolve(strict=True)
    if args.gpu is not None:
        args.gpu=args.gpu.resolve(strict=True)
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'matrix.json').write_text(json.dumps(matrix,indent=2)+'\n')
    report=dict(status='running',runs=[],failed={},
        cpu_sha256=hashlib.file_digest(args.cpu.open('rb'),'sha256').hexdigest() if args.cpu else None,
        gpu_sha256=hashlib.file_digest(args.gpu.open('rb'),'sha256').hexdigest() if args.gpu else None,
        backend=args.backend,
        halo_transport=args.halo_transport,compact_transport=args.compact_transport,ucx_tls=args.ucx_tls)
    for repeat in range(5):
        order=matrix[repeat:]+matrix[:repeat]
        if repeat%2:
            order=list(reversed(order))
        for row in order:
            label=case_name(row)
            if label in report['failed']:
                continue
            name=f'{label}_r{repeat+1}'
            print('START',name,flush=True)
            local=SimpleNamespace(**vars(args),steps=22 if row['backend']=='gpu' else 7)
            try:
                _,_,result=prepare_and_run(local,row,name,validation=False)
                if result['gpu_sharing_correctness_only']:
                    raise ValueError('shared GPUs are not valid formal scaling resources')
                result['repeat']=repeat+1
                report['runs'].append(result)
                print('PASS',name,flush=True)
            except Exception as error:
                report['failed'][label]=dict(run=name,error=str(error))
                print('FAILED; continuing independent configurations:',name,error,flush=True)
            (args.out/'runs.json').write_text(json.dumps(report,indent=2)+'\n')
    report['status']='pass' if not report['failed'] else 'partial'
    (args.out/'runs.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.out/'performance.json').write_text(json.dumps(summarize_runs(report['runs'],5),indent=2)+'\n')
    print(report['status'],flush=True)


if __name__=='__main__':
    main()
