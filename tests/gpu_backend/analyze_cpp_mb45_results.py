"""Read completed MB4/5 batches; no solver invocation or scheduler operation."""
import argparse
import json
from pathlib import Path
import re

import h5py
import numpy as np

from run_cpp_tgv_gate import compare, checkpoint, logged_steps
from run_cpp_multi_tgv_gate import analyze, layouts, raw_field, physical, global_region, bits_equal
from run_cpp_restart_gate import exact_state
from run_cpp_performance import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--observed', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--native-results', type=Path, required=True)
    parser.add_argument('--automatic-reference', type=Path, required=True)
    args = parser.parse_args()
    root = args.observed.resolve()
    plan = json.loads((args.prepared/'batch.json').read_text())
    report = dict(gates={}, timing={}, archives='run_cpp_archive_gate.py --analyze-only separately')
    for name, ranks, topology, workspace, sync, mode, trace, steps, reference, check, left in plan['gates']:
        directory = root/'cases'/name
        ranks, steps = int(ranks), int(steps)
        if not name.startswith('io_'):
            report['gates'][name] = analyze(directory, args.frozen)
            continue
        restored = name.endswith('_resumed')
        clock = logged_steps(directory, 2 if restored else steps, first_step=3 if restored else 1)
        stats = np.loadtxt(directory/'flowstate.dat', skiprows=1, ndmin=2)
        expected_identity = np.arange(2 if restored else 1, steps)
        if not np.array_equal(stats[:, 0], expected_identity):
            raise AssertionError(name+': TGV scalar sampling identity')
        paths = {}
        if reference == 'fixed':
            paths = {b: args.prepared/f'references/{b}_exact/checkpoints/step000000000004/state.h5'
                     for b in ('cpu', 'gpu')}
            scalar_reference = args.frozen/'evidence/standard_tgv_explicit10/cpu_np1_keep_x'
        elif reference in ('automatic', 'nonexact_automatic'):
            paths = {b: args.prepared/f'references/{b}_{reference}.h5' for b in ('cpu', 'gpu')}
            scalar_reference = (args.automatic_reference/'cpu_automatic' if reference == 'automatic' else
                                args.native_results/'nonexact_auto_keep1_reference/nonexact_cpu_restored')
        else:
            scalar_reference = args.automatic_reference/'cpu_automatic'
        expected_clock = logged_steps(scalar_reference, 2, first_step=3) if reference == 'nonexact_automatic' else \
                         logged_steps(scalar_reference)[2:] if restored else logged_steps(scalar_reference)[:steps]
        item = dict(clock=compare(clock, expected_clock))
        baseline_stats = np.loadtxt(scalar_reference/'flowstate.dat', skiprows=1, ndmin=2)
        item['statistics'] = compare(stats, baseline_stats[np.isin(baseline_stats[:, 0], expected_identity)])
        if paths:
            fields = {b: checkpoint(p) for b, p in paths.items()}
            item['reference_pair'] = compare(fields['cpu'], fields['gpu'])
            item['physical_fields'] = {}
            for rank, layout in layouts(directory).items():
                _, values = raw_field(directory/f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin')
                item['physical_fields'][rank] = {b: compare(physical(values, layout), global_region(v, layout))
                                                for b, v in fields.items()}
        state = directory/f'outdat/new/checkpoints/step{steps:012d}/state.h5'
        if state.exists():
            q = checkpoint(state)
            pressure = .4*(q[..., 4]-.5*np.sum(q[..., 1:4]**2, axis=-1)/q[..., 0])
            if not np.isfinite(q).all() or not (q[..., 0] > 0).all() or not (pressure > 0).all():
                raise AssertionError(name+': invalid physical checkpoint')
            if paths:
                item['checkpoint_fields'] = {b: compare(q, v) for b, v in fields.items()}
            if check == 'exact':
                item['identical_datasets'] = exact_state(root/left, state)
            if 'nonexact' in name or 'cross' in name or name == 'io_auto_prefix':
                with h5py.File(state) as source:
                    if int(source['identity'][12]) != 11 or 'q0006' in source or source['rank_extras'].size:
                        raise AssertionError(name+': lean checkpoint layout')
                if (directory/'outdat/new/resources/geometry.h5').exists():
                    raise AssertionError(name+': lean checkpoint geometry resource')
        report['gates'][name] = item
    for rank, layout in layouts(root/'cases/device_np4_xy_plain').items():
        suffix = f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
        bits_equal(physical(raw_field(root/'cases/device_np4_xy_plain'/suffix)[1], layout),
                   physical(raw_field(root/'cases/device_np4_xy_trace'/suffix)[1], layout), 'NP4 observation isolation')
    for name, ranks, topology, mode in plan['timing']:
        directory = root/'cases'/name
        result = summarize(directory, int(ranks), 5, 5, 5)
        memory = re.findall(r'^ASTR_CPP_DEVICE_MEMORY (initialized|final)\s+(\d+)\s+(\d+)\s+(\d+)$',
                            (directory/'run.log').read_text(), re.MULTILINE)
        if len(memory) != 2*int(ranks) or any(not 0 <= int(free) <= int(total) or int(total) == 0
                                            for phase, rank, free, total in memory):
            raise AssertionError(name+': physical-device memory records')
        result['physical_device_memory'] = memory
        report['timing'][name] = result
    for mode in ('pinned', 'device'):
        first = report['timing'][f'perf_{mode}_np1_111']['median_seconds_per_step']
        for name, ranks, topology, transport in plan['timing']:
            if transport != mode:
                continue
            item = report['timing'][name]
            item['strong_speedup'] = first/item['median_seconds_per_step']
            item['strong_efficiency'] = item['strong_speedup']/int(ranks)
    for name, ranks, topology, mode in plan['timing']:
        if mode != 'device':
            continue
        peer = name.replace('device', 'pinned')
        report['timing'][name]['pinned_over_device_median'] = report['timing'][peer]['median_seconds_per_step'] / \
            report['timing'][name]['median_seconds_per_step']
        report['timing'][name]['scalar_transport_comparison'] = compare(
            np.loadtxt(root/'cases'/name/'flowstate.dat', skiprows=1),
            np.loadtxt(root/'cases'/peer/'flowstate.dat', skiprows=1))
    (root/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print('All scheduled fields, checkpoint/clock/statistics and ten timing runs checked')


if __name__ == '__main__':
    main()
