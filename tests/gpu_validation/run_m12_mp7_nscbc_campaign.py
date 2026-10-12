#!/usr/bin/env python3
"""Resume the frozen local M12 campaign serially, refusing incomplete directories."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from run_m12_mp7_nscbc_performance import compare_runs, digest, performance_summary

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_EXECUTABLE = '20f7da833963751e5c9905511b2b1fd324f1eab0302c0794827b13e7963a2f78'


def campaign(base):
    base = base.resolve(strict=True)
    executable = base/'device_prefix_fix/bin/astr'
    if digest(executable) != EXPECTED_EXECUTABLE:
        raise ValueError('campaign requires the frozen current executable and legal checkpoints')
    driver = ROOT/'tests/gpu_validation/run_m12_mp7_nscbc_performance.py'
    combo = ['--transport','device-aware','--face-layout','x-contiguous','--split-cache','on']
    cache = ['--face-layout','x-contiguous','--split-cache','on']
    active = base/'final_w1_reference_check/outdat/new/checkpoints/step000000001000'
    origin = base/'final_prepare0_880/outdat/new/checkpoints/step000000000880'
    suppression = ROOT/'tests/gpu_validation/compute_sanitizer_ucx_cuda_aware.supp.xml'
    base2 = base/'original_w0_round0/resources.json'
    base1 = base/'final_np1_reference10/resources.json'
    completed = []

    def run(label,options,baseline=base2):
        target = base/label
        receipt = target/'receipt.json'
        if receipt.exists():
            result = json.loads(receipt.read_text())
            if result['status'] != 'passed' or result['provenance']['executable_sha256'] != EXPECTED_EXECUTABLE:
                raise ValueError('invalid existing receipt: '+label)
            print('REUSE '+label,flush=True)
        else:
            if target.exists():
                raise ValueError('incomplete directory must not be overwritten: '+str(target))
            print('START '+label,flush=True)
            command = [sys.executable,str(driver),'--executable',str(executable),'--output',str(target)]+options
            if baseline:
                command += ['--resource-baseline',str(baseline)]
            subprocess.run(command,cwd=ROOT,check=True)
            result = json.loads(receipt.read_text())
        completed.append(label)
        (base/'campaign_completed_final.json').write_text(json.dumps(completed,indent=2)+'\n')
        print('DONE '+label+' RK='+str(result['timing']['rk_seconds_per_step']),flush=True)

    np1 = ['--topology','1,1,1','--steps','10','--warmup','0','--diagnostic','--checkpoint-every','10']
    run('final_np1_verify10_qualified',np1+combo+['--verify-faces'],baseline=base1)
    result = compare_runs(base/'final_np1_reference10',base/'final_np1_verify10_qualified',10)
    (base/'final_np1_equivalence10.json').write_text(json.dumps(result,indent=2)+'\n')
    run('final_np1_memcheck2',['--topology','1,1,1','--steps','2','--warmup','0','--diagnostic',
                             '--memcheck','--sanitizer-suppressions',str(suppression)]+combo,baseline=base1)
    for window,restore in [('w0',[]),('active',['--restore',str(active)])]:
        stop = '8' if window == 'w0' else '1008'
        for name,options in [('reference',[]),('combined',combo)]:
            run('final_nsys_'+window+'_'+name,restore+['--steps',stop,'--warmup','0','--profile']+options)
    # Profiler shadow buffers are compared against the equivalent profiled baseline.
    run('final_ncu_reference',['--steps','2','--warmup','0','--ncu'],baseline=base/'b1_ncu/resources.json')
    run('final_ncu_combined',['--steps','2','--warmup','0','--ncu']+combo,
        baseline=base/'final_ncu_reference/resources.json')
    run('final_layout_w0_isolation',['--face-layout','x-contiguous'])
    run('final_cache_w0_isolation',cache)
    run('final_cache_w1_isolation',['--restore',str(origin),'--steps','1000']+cache)
    for i in range(5):
        for window in (['w0','w1'] if i%2 == 0 else ['w1','w0']):
            restore = [] if window == 'w0' else ['--restore',str(origin),'--steps','1000']
            order = [('reference',[]),('combined',combo)]
            if i%2:
                order.reverse()
            for name,options in order:
                run(f'formal_{window}_{name}_r{i}',restore+options)
    for window in ('w0','w1'):
        result = performance_summary([base/f'formal_{window}_reference_r{i}' for i in range(5)],
                                     [base/f'formal_{window}_combined_r{i}' for i in range(5)])
        (base/f'formal_{window}_summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print('SUMMARY '+window+' '+json.dumps(result),flush=True)
    for name,options in [('reference',[]),('combined',combo)]:
        run('final_np1_cost_'+name,['--topology','1,1,1']+options,baseline=base1)
    print('M12 LOCAL CAMPAIGN COMPLETE',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    campaign(parser.parse_args().base)
