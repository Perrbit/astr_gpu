"""Compare completed-step C++ field/slice archives to frozen solver binaries."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from run_cpp_tgv_gate import prepare_case, namelist, compare, snapshot
from run_cpp_restart_gate import datasets
from run_cpp_multi_tgv_gate import layouts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ranks', type=int, choices=[1, 2, 4], default=1)
    parser.add_argument('--analyze-only', action='store_true')
    parser.add_argument('--workspace', choices=['full', 'scalar'], default='full')
    parser.add_argument('--compact-form', choices=['compact_explicit', 'explicit_compact', 'compact_compact'])
    parser.add_argument('--primitive-only', action='store_true')
    parser.add_argument('--physical-fixture',type=Path)
    parser.add_argument('--profile-fixture',type=Path)
    parser.add_argument('--wall-axis',choices=['x','y','z'],default='y')
    parser.add_argument('--axis',choices=['x','y','z'],default='x')
    args = parser.parse_args()
    root = args.output.resolve()
    if not args.analyze_only:
        root.mkdir(parents=True, exist_ok=False)
    frozen = args.reference.resolve()
    original = frozen/'evidence/standard_tgv_explicit10/cpu_np1_keep_x/datin'
    env = {k: v for k, v in os.environ.items() if not k.startswith('ASTR_VALIDATION_')}
    env.update(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_osc='pt2pt',
               OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_opal_cuda_support='0',
               ASTR_GPU_FILTER_WORKSPACE=args.workspace, ASTR_GPU_SYNC_MODE='explicit',
               ASTR_VALIDATION_RHS_PREFIX='outdat/rhs', ASTR_VALIDATION_RHS_STEP='4')

    def run(name, binary, gpu, ranks, products):
        directory = root/name
        if args.profile_fixture:
            source=args.profile_fixture.resolve()/('old_gpu' if gpu else 'cpu')
            directory.mkdir();shutil.copytree(source/'datin',directory/'datin',copy_function=shutil.copyfile)
            (directory/'outdat/new').mkdir(parents=True)
        elif args.physical_fixture:
            source=args.physical_fixture.resolve()/(('gpu' if gpu else 'cpu')+'_'+args.wall_axis)
            directory.mkdir();shutil.copytree(source/'datin',directory/'datin')
            (directory/'outdat/new').mkdir(parents=True)
        else:
            numerics = None
            if args.compact_form:
                derivative, filtering = args.compact_form.split('_')
                scheme = '643c' if derivative == 'compact' else '643e'
                numerics = dict(conschm=scheme, difschm=scheme,
                                filter_scheme='compact10' if filtering == 'compact' else 'explicit10')
            prepare_case(directory, original, gpu, True, False, numerics=numerics)
        volume_derived=not args.primitive_only and args.physical_fixture is None and args.profile_fixture is None
        outputs = dict(output=dict(directory='outdat/new', host_budget_bytes=-1,
                                   device_budget_bytes=-1, device_reserve_bytes=0),
                       checkpoint=dict(enabled=False),
                       volume=dict(enabled=products, mode='steps', interval_steps=2,
                                   velocity_gradient=volume_derived, vorticity=volume_derived,
                                   qcriterion=volume_derived),
                       slices=dict(enabled=False))
        (directory/'datin/input.output').write_text(namelist(outputs))
        definitions = []
        if products:
            for axis in ('i', 'j', 'k'):
                for index in (0, 8, 16):
                    definitions.append(dict(id=f'{axis}{index}', kind='slice', enable_request=1,
                        source='instantaneous', fields=['density', 'velocity_x', 'velocity_y', 'velocity_z',
                            'pressure', 'temperature']+([] if args.primitive_only else
                            ['Q_rs', 'velocity_divergence', 'density_gradient_magnitude']),
                        sampling=dict(mode='grid_index', axis=axis, index=index,
                            index_ranges={a: [0, 16] for a in ('i', 'j', 'k') if a != axis}),
                        output=dict(channel='data', format='hdf5-xdmf'),
                        schedule=dict(mode='steps', interval=2)))
        (directory/'input.products.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in
            [dict(catalog_schema='astr-insitu-products-v1'), *definitions]))
        topology=[1,1,1];topology['xyz'.index(args.axis)]=ranks
        if ranks==4:topology=[2,2,1]
        selected = env | dict(ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)))
        command = [str(args.mpiexec.resolve()), '--bind-to', 'none', '-n', str(ranks), str(binary), 'run']
        with (directory/'run.log').open('w') as log:
            subprocess.run(command, cwd=directory, env=selected, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        return directory

    if args.analyze_only:
        cpu, old_gpu, cpp, plain = (root/name for name in ('frozen_cpu', 'frozen_gpu', 'cpp_archive', 'cpp_output_off'))
    else:
        cpu = run('frozen_cpu', frozen/'bin/astr_cpu', False, 1, True)
        old_gpu = run('frozen_gpu', frozen/'bin/astr_cuda_fortran', True, 1, True)
        cpp = run('cpp_archive', args.binary.resolve(), True, args.ranks, True)
        plain = run('cpp_output_off', args.binary.resolve(), True, args.ranks, False)
    for directory in (cpu, old_gpu, cpp):
        status = json.loads((directory/'outdat/new/products.status.json').read_text())
        if len(status['products']) != 9 or any(p['state'] != 'active' for p in status['products']):
            raise AssertionError(str(directory)+': nine active numerical slice products required')
    frames = [Path('fields')/'segment00000000'/f'step{step:012d}'/'data.h5' for step in (2, 4)]
    numerical = sorted((cpu/'outdat/new/slice_products').glob('*/*/segment*/step*.h5'))
    if len(numerical) != 27:
        raise AssertionError('nine numerical slices require initial, step-two and step-four frames')
    frames += [p.relative_to(cpu/'outdat/new') for p in numerical]
    report = dict(frames={},ranks=args.ranks,axis=args.axis,
        physical_fixture=str(args.physical_fixture) if args.physical_fixture else None,
        profile_fixture=str(args.profile_fixture) if args.profile_fixture else None,
        volume_derived=not args.primitive_only and args.physical_fixture is None and args.profile_fixture is None)
    def frame_path(directory, relative):
        if relative.parts[0] == 'slice_products':
            # Segment directory suffixes are exclusive-publication identities,
            # not quantities to match across independent solver processes.
            parent = directory/'outdat/new'/Path(*relative.parts[:3])
            matches = list(parent.glob('segment*/'+relative.name))
            if len(matches) != 1:
                raise AssertionError('missing or ambiguous numerical product frame')
            return matches[0]
        return directory/'outdat/new'/relative
    for relative in frames:
        sets = {name: datasets(frame_path(directory, relative))
                for name, directory in [('cpu', cpu), ('gpu', old_gpu), ('cpp', cpp)]}
        if not sets['cpu'] or any(values.keys() != sets['cpu'].keys() for values in sets.values()):
            raise AssertionError('field archive dataset identities')
        errors = {}
        for name, reference in sets['cpu'].items():
            if reference.dtype.kind == 'f':
                pair = compare(sets['gpu'][name].reshape(-1, 1), reference.reshape(-1, 1))
                errors[name] = dict(reference_pair=pair, cpu=compare(sets['cpp'][name].reshape(-1, 1), reference.reshape(-1, 1)),
                                    gpu=compare(sets['cpp'][name].reshape(-1, 1), sets['gpu'][name].reshape(-1, 1)))
            elif any(not np.array_equal(values[name], reference) for values in sets.values()):
                raise AssertionError('field archive metadata: '+name)
        report['frames'][str(relative)] = errors
    for rank in range(args.ranks):
        filename = f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
        a, b = snapshot(cpp/filename), snapshot(plain/filename)
        if not np.array_equal(a.view(np.uint64), b.view(np.uint64)):
            raise AssertionError('field archives changed physical q words')
    copies = re.findall(r'ASTR_OUTPUT_ARCHIVE rank=(\d+) product=(\w+).*download_bytes=(\d+)',
                        (cpp/'run.log').read_text())
    if len(copies) != 2*args.ranks:
        raise AssertionError('missing scheduled field publication records')
    # Index slices download their selected 2D values, not the volume cache.
    slice_copies = re.findall(r'ASTR_SLICE rank=(\d+).*field_d2h_bytes=(\d+)', (cpp/'run.log').read_text())
    if len(slice_copies) != 27*args.ranks:
        raise AssertionError('missing numerical slice publication records')
    layout=layouts(cpp)
    expected_volume=2*sum(int(np.prod(np.array(row['upper'])+2*row['halo'][0]+1))*6*8 for row in layout.values())
    if sum(int(n) for _,_,n in copies)!=expected_volume:
        raise AssertionError('scheduled volume primitive-cache transfer differs')
    expected_slices=27*17*17*(6 if args.primitive_only else 9)*8
    if sum(int(n) for _,n in slice_copies)!=expected_slices:
        raise AssertionError('numerical slices did not download exactly selected 2D result fields')
    report['expected_volume_download_bytes']=expected_volume
    report['expected_slice_result_download_bytes']=expected_slices
    report['publication_download_bytes'] = [list(x) for x in copies]
    report['slice_download_bytes'] = [list(x) for x in slice_copies]
    report['output_toggle_physical_q_identical'] = True
    (root/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Two volumes, 27 numerical slices, frozen references and output isolation passed')


if __name__ == '__main__':
    main()
