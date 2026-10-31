#!/usr/bin/env python3
"""Inspect diagnostic specialization; static PTX is not a timing result."""
import argparse
import json
import re
import subprocess

from pathlib import Path
from run_air5_performance_diagnosis import digest


def inspect(obj,executable):
    ptx=subprocess.check_output(['cuobjdump','--dump-ptx',str(obj)],text=True)
    result=dict(object_sha256=digest(obj),executable_sha256=digest(executable),kernels={})
    for role in ('half_step','report','quiet'):
        name=f'chemistry_coupling_gpu_air5_chemistry_{role}_kernel_'
        start=ptx.index('.visible .entry '+name)
        tail=ptx[start+1:]
        end=re.search(r'(?m)^\.visible \.(?:entry|func) ',tail)
        body=ptx[start:start+1+end.start()] if end else ptx[start:]
        resources=subprocess.check_output(['cuobjdump','--dump-resource-usage','--function',name,
                                           str(executable)],text=True)
        match=re.search(r'REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)',resources)
        if not match:
            raise ValueError(f'missing linked resources: {name}')
        result['kernels'][role]=dict(name=name,registers=int(match[1]),stack_bytes=int(match[2]),
            shared_bytes=int(match[3]),local_bytes=int(match[4]),
            diagnostic_min_atomics=body.count('atom.global.min.u64'),
            failure_cas=body.count('atom.global.cas'),ptx_lines=len(body.splitlines()))
    assert result['kernels']['quiet']['diagnostic_min_atomics']==0
    assert result['kernels']['quiet']['failure_cas']>0
    assert result['kernels']['report']['diagnostic_min_atomics']>0
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object',type=Path,required=True)
    parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=inspect(args.object,args.executable)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
