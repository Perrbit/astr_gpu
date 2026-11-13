"""Prepare or analyze the approved 32-GiB DCU scaling campaign; no scheduler calls."""
import argparse
import json
from pathlib import Path
import shutil
import statistics

import numpy as np

from run_cpp_tgv_gate import prepare_case, compare
from run_cpp_performance import summarize


def configurations():
    rows = []
    for topology in ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (4, 1, 1), (2, 2, 1)):
        rows.append(('strong_device_'+''.join(map(str, topology)), (384,)*3, topology, 'device'))
    for topology in ((2, 1, 1), (2, 2, 1)):
        rows.append(('strong_pinned_'+''.join(map(str, topology)), (384,)*3, topology, 'pinned'))
    for topology in ((2, 1, 1), (2, 2, 1)):
        rows.append(('weak_device_'+''.join(map(str, topology)), tuple(384*n for n in topology), topology, 'device'))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    p.add_argument('--analyze-only', action='store_true')
    a = p.parse_args()
    root = a.output.resolve()
    if a.analyze_only:
        plan = json.loads((root/'campaign.json').read_text())
        report = dict(configurations={})
        for name, grid, topology, transport in plan['configurations']:
            ranks = int(np.prod(topology))
            runs = []
            for repeat in range(1, 4):
                directory = root/'cases'/f'{name}_r{repeat}'
                runs.append(summarize(directory, ranks, 5, 10, 1))
                if transport == 'pinned':
                    peer = root/'cases'/f'{name.replace("pinned", "device")}_r{repeat}'
                    compare(np.loadtxt(directory/'flowstate.dat', skiprows=1, ndmin=2),
                            np.loadtxt(peer/'flowstate.dat', skiprows=1, ndmin=2))
            samples = [r['median_seconds_per_step'] for r in runs]
            report['configurations'][name] = dict(grid=grid, topology=topology, transport=transport,
                runs=runs, seconds_per_step=samples, median=statistics.median(samples),
                best=min(samples), maximum=max(samples))
        baseline = report['configurations']['strong_device_111']['median']
        for name, item in report['configurations'].items():
            ranks = int(np.prod(item['topology']))
            if item['transport'] == 'device':
                item['efficiency'] = baseline/item['median']/(ranks if name.startswith('strong') else 1)
                if name.startswith('strong'):
                    item['speedup'] = baseline/item['median']
            else:
                peer = report['configurations'][name.replace('pinned', 'device')]
                item['pinned_over_device'] = item['median']/peer['median']
        (root/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print('Ten configurations, three independent runs each analyzed')
        return
    if a.reference is None:
        p.error('--reference is required when preparing inputs')
    root.mkdir(parents=True, exist_ok=False)
    original = a.reference.resolve()/'evidence/standard_tgv_explicit10/cpu_np1_keep_x/datin'
    rows = configurations()
    manifest = []
    # Interleave configurations between repetitions rather than retaining a
    # single layout for all three runs; every run has its own warmup.
    for repeat in range(1, 4):
        for name, grid, topology, transport in rows:
            label = f'{name}_r{repeat}'
            prepare_case(root/'cases'/label, original, True, True, False, grid=grid, steps=15, dt=1e-4)
            manifest.append((label, int(np.prod(topology)), ','.join(map(str, topology)), transport))
    prepare_case(root/'preflight', original, True, True, False, steps=4, dt=1e-3)
    for backend in ('cpu', 'gpu'):
        shutil.copy2(a.reference.resolve()/f'evidence/standard_tgv_explicit10/{backend}_np1_keep_x/'
            'outdat/new/checkpoints/step000000000004/state.h5', root/f'reference_{backend}.h5')
    (root/'runs.tsv').write_text(''.join('\t'.join(map(str, row))+'\n' for row in manifest))
    (root/'campaign.json').write_text(json.dumps(dict(configurations=rows, repeats=3, warmup_steps=5,
        timed_steps=10, complete_steps=15, dt=1e-4, field_output=False, checkpoint=False,
        scalar_interval_steps=1, workspace='scalar', sync='dependency',
        weak_domain='fixed periodic box; fixed local interval counts, not a physical resolution comparison'), indent=2)+'\n')
    shutil.copy2(Path(__file__).with_name('run_cpp_scaling32_rocm.slurm'), root/'run.slurm')
    print('Prepared 30 independent runs, each 5 warmup + 10 measured steps')


if __name__ == '__main__':
    main()
