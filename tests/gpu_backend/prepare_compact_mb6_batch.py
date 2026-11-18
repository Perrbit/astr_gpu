"""Prepare compact migration inputs and immutable references; never submit jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from run_cpp_tgv_gate import prepare_case, namelist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--native-results', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    original = args.reference/'evidence/standard_tgv_explicit10/cpu_np1_keep_x/datin'
    folders = dict(compact_compact='flow_compact_compact_trace', compact_explicit='flow_compact_explicit',
                   explicit_compact='flow_explicit_compact')
    provenance = {}
    for form, folder in folders.items():
        for backend in ('cpu', 'old_gpu'):
            source = args.native_results/folder/backend
            state = source/'outdat/new/checkpoints/step000000000004/state.h5'
            target = root/'references'/form/backend
            target.mkdir(parents=True)
            shutil.copy2(state, target/'state.h5')
            shutil.copy2(source/'flowstate.dat', target/'flowstate.dat')
            shutil.copy2(source/'run.log', target/'run.log')
            provenance[form+'/'+backend] = dict(source=str(source.resolve()),
                state_sha256=hashlib.sha256(state.read_bytes()).hexdigest())
    topologies = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 1)]
    components = [(mode, dims) for mode in ('pinned', 'device-aware')
                  for dims in topologies+[(4, 1, 1)]]
    (root/'components.tsv').write_text(''.join(
        f'{mode}\t{a*b*c}\t{a}\t{b}\t{c}\n' for mode, (a, b, c) in components))
    rows = []

    def add(name, form, topology, mode, workspace='scalar', sync='explicit',
            exact=None, restore=None, check='none', left='-', archive=False):
        directory = root/'cases'/name
        derivative, filtering = form.split('_')
        scheme = '643c' if derivative == 'compact' else '643e'
        numerics = dict(conschm=scheme, difschm=scheme,
                        filter_scheme='compact10' if filtering == 'compact' else 'explicit10')
        prepare_case(directory, original, True, True, False, numerics=numerics)
        options = dict(directory='outdat/new', exact_restart=exact is not False,
                       host_budget_bytes=-1, device_budget_bytes=-1, device_reserve_bytes=0)
        if restore:
            options['restore_directory'] = args.remote_root+'/'+restore
        (directory/'datin/input.output').write_text(namelist(dict(output=options,
            checkpoint=dict(enabled=exact is not None, mode='steps', interval_steps=2, keep=2),
            volume=dict(enabled=archive, mode='steps', interval_steps=2), slices=dict(enabled=False))))
        if archive:
            products = [dict(catalog_schema='astr-insitu-products-v1'),
                dict(id='mid_z', kind='slice', enable_request=1, source='instantaneous',
                    fields=['density', 'velocity_x', 'velocity_y', 'velocity_z', 'pressure', 'temperature'],
                    sampling=dict(mode='grid_index', axis='k', index=8,
                                  index_ranges=dict(i=[0, 16], j=[0, 16])),
                    output=dict(channel='data', format='hdf5-xdmf'), schedule=dict(mode='steps', interval=2))]
            (directory/'input.products.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in products))
        rows.append([name, str(topology[0]*topology[1]*topology[2]), ','.join(map(str, topology)),
                     mode, workspace, sync, form, check, left])

    for mode in ('pinned', 'device-aware'):
        for dims in topologies:
            name = mode+'_np%d_%dx%dx%d' % (dims[0]*dims[1]*dims[2], *dims)
            add(name, 'compact_compact', dims, mode)
    add('explicit_compact_y', 'explicit_compact', (1, 2, 1), 'pinned')
    add('compact_explicit_z', 'compact_explicit', (1, 1, 2), 'device-aware')
    add('full_dependency_x', 'compact_compact', (2, 1, 1), 'device-aware', workspace='full', sync='dependency')
    for label, exact, dims, mode in [('exact', True, (2, 1, 1), 'device-aware'),
                                    ('nonexact', False, (2, 2, 1), 'pinned')]:
        continuous = label+'_continuous'
        add(continuous, 'compact_compact', dims, mode, exact=exact)
        add(label+'_resumed', 'compact_compact', dims, mode, exact=exact,
            restore=f'cases/{continuous}/outdat/new/checkpoints/step000000000002',
            check='exact' if exact else 'physical',
            left=f'cases/{continuous}/outdat/new/checkpoints/step000000000004/state.h5')
    add('primitive_archives', 'compact_compact', (2, 2, 1), 'device-aware', archive=True)
    (root/'gates.tsv').write_text(''.join('\t'.join(row)+'\n' for row in rows))
    (root/'batch.json').write_text(json.dumps(dict(reference=str(args.reference.resolve()),
        provenance=provenance, components=components, gates=rows, grid=16, steps=4, dt=.001,
        remote_root=args.remote_root), indent=2)+'\n')
    print(len(components), 'distributed component runs;', len(rows), 'ordinary ASTR runs prepared')


if __name__ == '__main__':
    main()
