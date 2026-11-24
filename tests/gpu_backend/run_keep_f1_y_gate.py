"""Check periodic and bc41 compact filters against independent line operators."""
import argparse
import json
from math import comb
import os
from pathlib import Path
import re
import subprocess

import h5py
import numpy as np

from run_keep_f0_gate import prepare_wall, stage
from run_keep_f1_wall5_gate import join, parts
from run_cpp_tgv_gate import ROOT, checkpoint, compare, logged_steps, namelist, prepare_case, snapshot


def periodic_y_reference(q, alpha):
    angle = 2*np.pi*np.fft.rfftfreq(q.shape[1])
    symbol = 1-(1-2*alpha)*np.sin(angle/2)**10/(1+2*alpha*np.cos(angle))
    return np.fft.irfft(np.fft.rfft(q, axis=1)*symbol[None, :, None, None], n=q.shape[1], axis=1)


def physical_y_reference(q, alpha):
    """Solve the unpartitioned physical-end system before the wall values are restored."""
    beta = .98
    axes = np.moveaxis(q, 1, 0)
    nodes = axes.shape[0]
    values = axes.reshape(nodes, -1)
    lhs = np.eye(nodes) + alpha*(np.eye(nodes, k=-1) + np.eye(nodes, k=1))
    lhs[0, 1] = lhs[-1, -2] = beta
    rhs = lhs @ values
    for j in range(nodes):
        distance = min(j, nodes-1-j)
        if distance <= 2:
            support = values[:7] if j == distance else values[:-8:-1]
            strength = (1-beta if distance == 0 else 1-2*alpha)*(-1)**(distance+1)
            order = 6
        else:
            radius = min(distance, 5)
            support = values[j-radius:j+radius+1]
            strength = (1-2*alpha)*(-1)**(radius+1)
            order = 2*radius
        diff = np.array([(-1)**i*comb(order, i) for i in range(order+1)])
        rhs[j] += (strength/2**order)*(diff @ support)
    filtered = np.linalg.solve(lhs, rhs)
    filtered[[0, -1]] = values[[0, -1]]
    return np.moveaxis(filtered.reshape(axes.shape), 0, 1)


def periodic_axis(q, axis, alpha):
    line = np.moveaxis(q, axis, 1)
    filtered = periodic_y_reference(line[:, :-1], alpha)
    return np.moveaxis(np.concatenate((filtered, filtered[:, :1]), axis=1), 1, axis)


def verify_channel(args, root, base):
    # F0 Case 1 uses all three directions; F1 Case 3 uses only y, on the same physical walls.
    report = {}
    final = Path('outdat/new/checkpoints/step000000000002/state.h5')
    for label, scheme, convection, scope in (('f0-channel-e-xyz', '643e', 'standard', 'xyz'),
                                             ('f1-channel-c-y', '643c', 'keep', 'y')):
        cases = {}
        entry = {}
        for backend in ('cpu', 'gpu'):
            case = root/f'{label}-{backend}'
            prepare_wall(case, '41y', backend, scheme, '643e', 'compact10', convection=convection, steps=2)
            cfg = case/'datin/input.case'
            cfg.write_text(cfg.read_text().replace("filter_scheme = 'compact10'",
                                                  f"filter_scheme = 'compact10'\n  filter_scope = '{scope}'"))
            # Excite both physical closure and tangential damping instead of relying on a smooth signal.
            n = 16
            k, j, i = np.indices((n+1, n+1, n+1))
            envelope = .5+.5*np.sin(np.pi*j/n)**2
            signal = .01*envelope*((-1.)**j + .7*(-1.)**i - .4*(-1.)**k)
            with h5py.File(case/'datin/flowini3d.h5', 'r+') as initial:
                initial['ro'][...] += signal
            with (case/'run.log').open('w') as log:
                subprocess.run([str(args.mpiexec.resolve()), '-n', '2', str(args.binary.resolve()), 'run'],
                               cwd=case, env=base | {'ASTR_FORCE_MPI_TOPOLOGY': '1,2,1'},
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
            logged_steps(case, count=2)
            if backend == 'gpu':
                axes = set(re.findall(r'ASTR_CPP_COMPACT_PHASE_TIMING\s+\d+\s+\d+\s+(\d+)\s+filter',
                                      (case/'run.log').read_text()))
                if axes != ({'2'} if scope == 'y' else {'1', '2', '3'}):
                    raise AssertionError(f'{label}: wrong active GPU filter axes: {axes}')
            entry[backend] = []
            cases[backend] = case
            with h5py.File(case/'outdat/new/resources/geometry.h5') as geometry:
                weight = geometry['q0004'][...].transpose(2, 1, 0)[:-1, :, :-1].copy()
            weight[:, [0, -1], :] *= .5
            for rk in range(1, 4):
                before = join(parts(case, 'pre_filter', rk, 2))
                after_blocks = parts(case, 'post_filter', rk, 2)
                after = join(after_blocks)
                y_input = periodic_axis(before, 0, .49) if scope == 'xyz' else before
                expected = physical_y_reference(y_input, .49)
                y_change = max(compare(expected, y_input, limit=None))
                if y_change <= 2e-10:
                    raise AssertionError('y reference did not exercise the physical closure')
                if scope == 'xyz':
                    expected = periodic_axis(expected, 2, .49)
                else:
                    np.testing.assert_array_equal(after[:, [0, -1]].view(np.uint64),
                                                  before[:, [0, -1]].view(np.uint64))
                error = compare(after, expected)
                shared = after_blocks[0].shape[1]-1
                copy_error = compare(after_blocks[1][:, 0], expected[:, shared])
                # For this fixture the update snapshot is the next substep's input, before any filter.
                input_error = None if rk == 1 else compare(before, join(parts(case, 'post_update', rk-1, 2)))
                delta = (after-before)[:-1, :, :-1, (0, 1, 4)]
                budget = np.einsum('ijk,ijkm->m', weight, delta).tolist()
                entry[backend].append(dict(reference=error, shared_copy=copy_error, y_signal=y_change,
                                           previous_rk_update=input_error, filter_budget=budget))
        entry['terminal'] = compare(checkpoint(cases['gpu']/final), checkpoint(cases['cpu']/final))
        report[label] = entry
        print(f'{label}: three physical-end filtered states and both rank copies verified', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_root = Path('outdat/new/checkpoints')
    base = os.environ | dict(ASTR_FORCE_MPI_TOPOLOGY='1,1,1', ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',
                            ASTR_VALIDATION_RHS_STEP='0', ASTR_GPU_COMPACT_PHASE_TIMING='1',
                            ASTR_GPU_COMPACT_TRANSPORT='pinned', ASTR_GPU_COMPACT_SYNC='explicit',
                            ASTR_GPU_SYNC_MODE='explicit', OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp',
                            OMPI_MCA_osc='pt2pt', OMPI_MCA_coll='^cuda,hcoll,ucc')

    def run(name, gpu, scope, steps, restore=None, mismatch=False):
        case = output/name
        numerics = dict(convection_form='keep', conschm='643c', difschm='643e',
                        filter_scheme='compact10', filter_scope=scope, alfa_filter=.49)
        prepare_case(case, ROOT/'examples/Taylor_Green_Vortex/datin', gpu, gpu, False, steps=steps, dt=1e-5,
                     numerics=numerics)
        (case/'datin/input.output').write_text(namelist(dict(
            output=dict(directory='outdat/new', restore_directory=str(restore or ''), exact_restart=True,
                        host_budget_bytes=-1, device_budget_bytes=-1, device_reserve_bytes=0),
            checkpoint=dict(enabled=True, mode='steps', interval_steps=2, keep=2),
            volume=dict(enabled=False), slices=dict(enabled=False))))
        with (case/'run.log').open('w') as log:
            result = subprocess.run([str(args.mpiexec.resolve()), '-n', '1', str(args.binary.resolve()), 'run'],
                                    cwd=case, env=base, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        log = (case/'run.log').read_text()
        if mismatch:
            if result.returncode == 0 or 'compact operator identity mismatch' not in log:
                raise AssertionError('xyz checkpoint was not explicitly rejected for y-only exact restart')
        else:
            result.check_returncode()
            logged_steps(case, count=steps-2 if restore else steps, first_step=3 if restore else 1)
            if gpu:
                axes = set(re.findall(r'ASTR_CPP_COMPACT_PHASE_TIMING\s+\d+\s+\d+\s+(\d+)\s+filter', log))
                if axes != ({'2'} if scope == 'y' else {'1', '2', '3'}):
                    raise AssertionError(f'{name}: wrong active GPU filter axes: {axes}')
        return case

    report = {}
    cpu, gpu = [run(f'{name}-y', device, 'y', 4) for name, device in (('cpu', False), ('gpu', True))]
    for rk in range(1, 4):
        prefix = cpu/'outdat/rhs'
        suffix = f'.step00000000.rk{rk:02}.rank00000000.bin'
        unfiltered = snapshot(Path(str(prefix)+'.pre_filter'+suffix))[:-1, :-1, :-1]
        expected = periodic_y_reference(unfiltered, .49)
        for backend, case in (('cpu', cpu), ('gpu', gpu)):
            observed = stage(case, 'pre_rhs', rk, 0)[:-1, :-1, :-1]
            report[f'{backend}/rk{rk}'] = compare(observed, expected)
    final = checkpoint_root/'step000000000004/state.h5'
    report['continuous'] = compare(checkpoint(gpu/final), checkpoint(cpu/final))
    for backend, device, case in (('cpu', False, cpu), ('gpu', True, gpu)):
        restored = run(f'{backend}-y-restored', device, 'y', 4, case/checkpoint_root/'step000000000002')
        np.testing.assert_array_equal(checkpoint(restored/final).view(np.uint64), checkpoint(case/final).view(np.uint64))
    xyz = run('gpu-xyz', True, 'xyz', 2)
    run('gpu-xyz-into-y', True, 'y', 4, xyz/checkpoint_root/'step000000000002', mismatch=True)
    report['bc41'] = verify_channel(args, output, base)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('y-only periodic symbol, CPU/GPU match, exact restart and scope rejection passed')


if __name__ == '__main__':
    main()
