"""Prepare the approved DCU MB4/5 batch locally; never submit a scheduler job."""
import argparse
import json
from pathlib import Path
import shutil

from run_cpp_tgv_gate import prepare_case, namelist
from run_cpp_multi_tgv_gate import validation_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--native-results', type=Path, required=True)
    followup = parser.add_mutually_exclusive_group()
    followup.add_argument('--wire-followup', action='store_true')
    followup.add_argument('--archive-followup', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'references').mkdir()
    components = ([] if args.wire_followup or args.archive_followup else
                  [(mode, ranks, axis) for mode in ('pinned', 'device')
                   for ranks, axis in [(1, 0), (2, 0), (2, 1), (2, 2)]])
    (root/'components.tsv').write_text(''.join(f'{mode}\t{ranks}\t{axis}\n' for mode, ranks, axis in components))
    original = args.reference/'evidence/standard_tgv_explicit10/cpu_np1_keep_x/datin'
    references = args.reference/'evidence/standard_tgv_explicit10'
    for backend in ('cpu', 'gpu'):
        shutil.copytree(references/f'{backend}_np1_keep_x/outdat/new', root/'references'/f'{backend}_exact')
    auto = Path('results/multi_backend_mb2_20261112/native/main_automatic')
    for backend, folder in [('cpu', 'cpu_automatic'), ('gpu', 'old_gpu_automatic')]:
        shutil.copy2(auto/folder/'outdat/new/checkpoints/step000000000004/state.h5',
                     root/'references'/f'{backend}_automatic.h5')
    for backend in ('cpu', 'gpu'):
        shutil.copy2(args.native_results/f'nonexact_auto_keep1_reference/nonexact_{backend}_restored/'
                     'outdat/new/checkpoints/step000000000004/state.h5',
                     root/'references'/f'{backend}_nonexact_automatic.h5')
    rows = []

    def add(name, ranks, topology, transport, workspace='scalar', sync='dependency', trace=False,
            automatic=False, exact=None, restore=None, keep=2, steps=4, check='none', left='-'):
        if args.archive_followup and name not in (
                'device_np2_x_plain', 'pinned_np4_xy_plain', 'io_archive', 'io_archive_pinned_np4'):
            return None
        if args.wire_followup and not (name.startswith('io_') or name in
                ('device_np2_x_plain', 'device_np4_xy_trace', 'device_np4_xy_plain')):
            return None
        directory = root/'cases'/name
        prepare_case(directory, original, True, True, automatic, steps=steps)
        if exact is not None:
            options = dict(directory='outdat/new', exact_restart=exact, host_budget_bytes=-1,
                           device_budget_bytes=-1, device_reserve_bytes=0)
            if restore:
                options['restore_directory'] = args.remote_root+'/'+restore
            (directory/'datin/input.output').write_text(namelist(dict(output=options,
                checkpoint=dict(enabled=True, mode='steps', interval_steps=2, keep=keep),
                volume=dict(enabled=False), slices=dict(enabled=False))))
        ref = 'none' if steps != 4 else ('automatic' if automatic else 'fixed')
        if automatic and restore and exact is False:
            ref = 'nonexact_automatic'
        rows.append([name, str(ranks), ','.join(map(str, topology)), workspace, sync, transport,
                     str(int(trace)), str(steps), ref, check, left])
        return directory

    for transport in ('pinned', 'device'):
        for name, ranks, topology, trace, sync in validation_cases(True):
            add(transport+'_'+name, ranks, topology, transport, sync=sync, trace=trace)
    add('device_full_y', 2, (1, 2, 1), 'device', workspace='full', trace=True)
    state = 'outdat/new/checkpoints/step000000000004/state.h5'
    prefix = 'outdat/new/checkpoints/step000000000002'
    for label, exact, automatic, transport in [('exact', True, False, 'device'),
                                             ('nonexact', False, False, 'pinned'),
                                             ('automatic', True, True, 'device')]:
        cont = 'io_'+label+'_continuous'
        add(cont, 2, (2, 1, 1), transport, automatic=automatic, exact=exact)
        add('io_'+label+'_resumed', 2, (2, 1, 1), transport, automatic=automatic, exact=exact,
            restore='cases/'+cont+'/'+prefix, check='exact' if exact else 'physical',
            left='cases/'+cont+'/'+state)
    add('io_auto_prefix', 2, (2, 1, 1), 'device', automatic=True, exact=False, keep=1, steps=2)
    add('io_auto_nonexact_resumed', 2, (2, 1, 1), 'device', automatic=True, exact=False, keep=1,
        restore='cases/io_auto_prefix/'+prefix, check='physical', left='references/cpu_nonexact_automatic.h5')
    for backend in ('cpu', 'gpu'):
        add('io_'+backend+'_cross_resumed', 2, (2, 1, 1), 'device', exact=False,
            restore=f'references/{backend}_exact/checkpoints/step000000000002', check='physical',
            left=f'references/{backend}_exact/checkpoints/step000000000004/state.h5')
    archive = add('io_archive', 2, (2, 1, 1), 'device')
    source = args.native_results/'scalar_catalog_archives_np2/cpp_archive'
    shutil.copy2(source/'datin/input.output', archive/'datin/input.output')
    shutil.copy2(source/'input.products.jsonl', archive/'input.products.jsonl')
    if args.archive_followup:
        archive = add('io_archive_pinned_np4', 4, (2, 2, 1), 'pinned')
        shutil.copy2(source/'datin/input.output', archive/'datin/input.output')
        shutil.copy2(source/'input.products.jsonl', archive/'input.products.jsonl')
    (root/'gates.tsv').write_text('\n'.join('\t'.join(row) for row in rows)+'\n')
    timing = []
    for transport in (() if args.archive_followup else ('pinned', 'device')):
        for ranks, topology in [(1, (1, 1, 1)), (2, (2, 1, 1)), (2, (1, 2, 1)),
                                (2, (1, 1, 2)), (4, (2, 2, 1))]:
            name = f'perf_{transport}_np{ranks}_'+''.join(map(str, topology))
            prepare_case(root/'cases'/name, original, True, True, False, grid=128, steps=30, dt=2e-4)
            timing.append([name, str(ranks), ','.join(map(str, topology)), transport])
    (root/'timing.tsv').write_text('\n'.join('\t'.join(row) for row in timing)+'\n')
    (root/'batch.json').write_text(json.dumps(dict(remote_root=args.remote_root, gates=rows, timing=timing,
        performance=dict(grid=128, dt=2e-4, steps=30, warmup=5, windows=5, window_steps=5),
        source_baseline='ff7a320 plus current working changes'), indent=2)+'\n')
    print(len(rows), 'short runs and', len(timing), 'timing runs prepared')


if __name__ == '__main__':
    main()
