#!/usr/bin/env python3
"""Small compute-node check before the split A800 timing campaigns."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from run_compact_a800_preflight import prepare_and_run, compare_pair
from run_compact_derivative_tgv import measure


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('cpu','gpu'),required=True)
    parser.add_argument('--cpu',type=Path,required=True)
    parser.add_argument('--gpu',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--ucx-tls',default='self,sm,cuda_copy,cuda_ipc')
    options=parser.parse_args()
    options.cpu=options.cpu.resolve(strict=True)
    if options.gpu:
        options.gpu=options.gpu.resolve(strict=True)
    options.out=options.out.resolve()
    options.out.mkdir(parents=True,exist_ok=False)
    args=SimpleNamespace(**vars(options),steps=10,halo_transport='device-aware',compact_transport='device-aware')
    results=[]
    for scheme in ('643e','643c'):
        baseline=None
        topologies=[(1,1,1),(4,4,2)] if options.backend=='cpu' else [(1,1,1),(2,2,1)]
        for topology in topologies:
            ranks=topology[0]*topology[1]*topology[2]
            row=dict(backend='cpu',scheme=scheme,family='preflight',grid=(32,32,32),topology=topology,ranks=ranks)
            case,field,record=prepare_and_run(args,row,f'{scheme}_cpu_{ranks}')
            if baseline is None:
                baseline=field
            if options.backend=='cpu':
                results.append(dict(scheme=scheme,ranks=ranks,**measure(baseline,field)))
            else:
                gpu,_,gpu_record=prepare_and_run(args,dict(row,backend='gpu'),f'{scheme}_gpu_{ranks}')
                if gpu_record['gpu_sharing_correctness_only']:
                    raise ValueError('compute-node preflight requires distinct GPUs')
                results.append(dict(scheme=scheme,ranks=ranks,**compare_pair(case,gpu,ranks,10)))
    (options.out/'result.json').write_text(json.dumps(dict(status='pass',checks=results),indent=2)+'\n')
    print('COMPUTE_NODE_PREFLIGHT=PASS',flush=True)


if __name__=='__main__':
    main()
