"""KEEP F0: three standard baselines with explicit viscosity and compact filtering."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np

from run_cpp_multi_tgv_gate import raw_field, physical
from run_cpp_tgv_gate import ROOT, checkpoint, compare, logged_steps, namelist, prepare_case

sys.path.insert(0, str(ROOT/'tests/gpu_validation'))
from run_keep_simple_boundary_gate import prepare as prepare_wall


def stage(directory, label, substep, rank):
    path = directory/f'outdat/rhs.{label}.step00000000.rk{substep:02}.rank{rank:08}.bin'
    if label.startswith('pre_rhs'):
        header, values = raw_field(path, primitive=label.endswith('primitives'))
        return physical(values, dict(upper=header[:3], halo=(header[3],)))
    with path.open('rb') as stream:
        shape = np.fromfile(stream, np.int32, 4)
        return np.fromfile(stream, np.float64).reshape((*map(int, shape[:3]+1), int(shape[3])), order='F')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True, help='ordinary C++ build with CPU runtime and diagnostics')
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    final = Path('outdat/new/checkpoints/step000000000002/state.h5')
    geometry = Path('outdat/new/resources/geometry.h5')
    base = os.environ | dict(ASTR_VALIDATION_RHS_PREFIX='outdat/rhs', ASTR_VALIDATION_RHS_STEP='0',
                            ASTR_GPU_COMPACT_TRANSPORT='pinned', ASTR_GPU_COMPACT_SYNC='explicit',
                            ASTR_GPU_HALO_TRANSPORT='pinned', ASTR_GPU_SYNC_MODE='explicit',
                            OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_osc='pt2pt',
                            OMPI_MCA_coll='^cuda,hcoll,ucc')

    def run(name, scheme, backend, dims, convection='standard', stretched=True, stationary=False):
        ranks = int(np.prod(list(map(int, dims.split(',')))))
        gpu = backend == 'gpu'
        case = output/f'{name}-np{ranks}-{backend}'
        if name.startswith('tgv'):
            prepare_case(case, ROOT/'examples/Taylor_Green_Vortex/datin', gpu, gpu, False, steps=2, dt=1e-5,
                         numerics=dict(conschm=scheme, difschm='643e', filter_scheme='compact10'))
            (case/'datin/input.output').write_text(namelist(dict(
                output=dict(directory='outdat/new', exact_restart=True, host_budget_bytes=-1,
                            device_budget_bytes=-1, device_reserve_bytes=0),
                checkpoint=dict(enabled=True, mode='steps', interval_steps=2, keep=2),
                volume=dict(enabled=False), slices=dict(enabled=False))))
        else:
            prepare_wall(case, '41y', backend, scheme, '643e', 'compact10', convection=convection,
                         stretched=stretched, steps=2)
            if stationary:
                with h5py.File(case/'datin/flowini3d.h5', 'r+') as field:
                    for key in ('ro', 't', 'u1', 'u2', 'u3'):
                        field[key][...] = float(key in ('ro', 't'))
        with (case/'run.log').open('w') as log:
            subprocess.run([str(args.mpiexec.resolve()), '-n', str(ranks), str(args.binary.resolve()), 'run'],
                           cwd=case, env=base | {'ASTR_FORCE_MPI_TOPOLOGY': dims},
                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
        logged_steps(case, count=2)
        return case

    report = {}
    for topology in ('1,1,1', 'split'):
        for name, scheme in (('tgv-c', '643c'), ('channel-c', '643c'), ('channel-e', '643e')):
            tgv = name.startswith('tgv')
            dims = topology if topology != 'split' else ('2,1,1' if tgv else '1,2,1')
            ranks = int(np.prod(list(map(int, dims.split(',')))))
            cpu, gpu = [run(name, scheme, backend, dims) for backend in ('cpu', 'gpu')]
            errors = dict(q=compare(checkpoint(gpu/final), checkpoint(cpu/final)))
            with h5py.File(gpu/geometry) as device, h5py.File(cpu/geometry) as host:
                errors['geometry'] = {key: compare(device[key][...].reshape(-1, 1), host[key][...].reshape(-1, 1))
                                      for key in (f'q{n:04}' for n in range(4, 14))}
            for rank in range(ranks):
                for rk in range(1, 4):
                    pair = {label: [stage(case, label, rk, rank) for case in (gpu, cpu)]
                            for label in ('pre_rhs', 'pre_rhs_primitives', 'conv', 'full')}
                    pair['viscous'] = [full-conv for full, conv in zip(pair['full'], pair['conv'])]
                    for label, values in pair.items():
                        errors[f'{label}/rk{rk}/rank{rank}'] = compare(*values)
            key = f'{name}/np{ranks}'
            report[key] = errors
            if not tgv:
                keep = run(f'{name}-keep', scheme, 'cpu', dims, convection='keep')
                with h5py.File(cpu/geometry) as standard, h5py.File(keep/geometry) as keep_geo:
                    if not np.all(standard['q0004'][...] > 0):
                        raise AssertionError('nonpositive channel Jacobian')
                    for n in range(1, 14):
                        key_geo = f'q{n:04}'
                        np.testing.assert_array_equal(standard[key_geo][...], keep_geo[key_geo][...], err_msg=key_geo)
            (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            print(f'{key}: CPU/CUDA C++ complete-step q, geometry and three RK operator stages match', flush=True)
    separation = {}
    for backend in ('cpu', 'gpu'):
        # A uniform tensor grid isolates the convective choice from the choice of metric stencil.
        compact, explicit = [run(f'uniform-channel-{form}', '643'+form, backend, '1,1,1', stretched=False)
                             for form in ('c', 'e')]
        compare(stage(compact, 'pre_rhs', 1, 0), stage(explicit, 'pre_rhs', 1, 0))
        convection = [stage(case, 'conv', 1, 0) for case in (compact, explicit)]
        viscosity = [stage(case, 'full', 1, 0)-conv for case, conv in zip((compact, explicit), convection)]
        separation[backend] = dict(viscous_error=compare(*viscosity),
                                   convective_difference=compare(*convection, limit=None))
        if max(separation[backend]['convective_difference']) <= 2e-10:
            raise AssertionError(f'{backend}: convection schemes are not distinguishable at the same state')
        stationary = run('channel-rest', '643c', backend, '1,1,1', stationary=True)
        initial = stage(stationary, 'pre_rhs', 1, 0)
        separation[backend]['stationary_state'] = compare(checkpoint(stationary/final), initial)
    (output/'separation.json').write_text(json.dumps(separation, indent=2)+'\n')


if __name__ == '__main__':
    main()
