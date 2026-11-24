"""Validate the bc41 wall-five compact filter, including the MPI-owned fifth plane."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np

from run_cpp_multi_tgv_gate import raw_field, physical
from run_cpp_tgv_gate import checkpoint, compare, logged_steps
from run_keep_f0_gate import stage, prepare_wall
from keep_wall5_reference import filter_wall5


def wall5_reference(field):
    result = field.copy()
    for line in (result, result[:, ::-1]):
        source = field if line is result else field[:, ::-1]
        samples = np.moveaxis(source[:, :11], 1, 0)
        values = filter_wall5(samples.reshape(11, -1)).reshape((5, *samples.shape[1:]))
        line[:, 1:6] = np.moveaxis(values, 0, 1)
    return result


def parts(case, label, rk, ranks):
    if label.startswith('pre_rhs'):
        return [stage(case, label, rk, rank) for rank in range(ranks)]
    blocks = []
    for rank in range(ranks):
        header, values = raw_field(case/f'outdat/rhs.{label}.step00000000.rk{rk:02}.rank{rank:08}.bin')
        blocks.append(physical(values, dict(upper=header[:3], halo=(header[3],))))
    return blocks


def join(blocks):
    return np.concatenate([blocks[0], *(b[:, 1:] for b in blocks[1:])], axis=1)


def verify_stages(case, ranks, constant=False, mirrored=False):
    with h5py.File(case/'outdat/new/resources/geometry.h5') as geometry:
        weight = geometry['q0004'][...].transpose(2, 1, 0)[:-1, :, :-1].copy()
    weight[:, [0, -1], :] *= .5

    def budget(after, before):
        delta = (after-before)[:-1, :, :-1, (0, 1, 4)]
        return list(np.einsum('ijk,ijkm->m', weight, delta))

    result = []
    for rk in range(1, 4):
        src = parts(case, 'pre_filter', rk, ranks)
        dst = parts(case, 'post_filter', rk, ranks)
        before, after = join(src), join(dst)
        expected = wall5_reference(before)
        errors = compare(after, expected)
        if constant:
            compare(after, before)
        first = 0
        total = before.shape[1]-1
        for a, b in zip(src, dst):
            last = first + a.shape[1]-1
            selected = np.arange(first, last+1)
            selected = ((selected >= 1) & (selected <= 5)) | ((selected >= total-5) & (selected < total))
            np.testing.assert_array_equal(b[:, ~selected].view(np.uint64), a[:, ~selected].view(np.uint64))
            compare(b, expected[:, first:last+1])
            first = last
        if ranks == 2:
            interface = src[0].shape[1]-1
            np.testing.assert_array_equal(dst[0][:, -1].view(np.uint64), dst[1][:, 0].view(np.uint64))
            if total == 11:
                if interface != 5:
                    raise AssertionError('fixture needs global j=5 to be shared')
                if not constant and np.array_equal(dst[1][:, 0], src[1][:, 0]):
                    raise AssertionError('fixture did not exercise a changed fifth plane')
        mirror = None
        if mirrored:
            # Reflect y and reverse the wall-normal momentum, leaving the other components unchanged.
            sign = np.array([1., 1., -1., 1., 1.])
            mirror = dict(input=compare(before, before[:, ::-1]*sign),
                          output=compare(after, after[:, ::-1]*sign))
            central = before[:, 6:total-5]
            if central.shape[1] == 0 or np.ptp(central[..., 0]) <= 2e-10:
                raise AssertionError('fixture needs a nonconstant unfiltered interior band')
            mirror['central_layers'] = central.shape[1]
            mirror['wall_signal'] = max(compare(after[:, 1:6], before[:, 1:6], limit=None))
            if mirror['wall_signal'] <= 2e-10:
                raise AssertionError('mirror fixture did not exercise the filter')
        pressure = .4*(after[..., 4]-.5*np.sum(after[..., 1:4]**2, axis=-1)/after[..., 0])
        if not (np.all(after[..., 0]>0) and np.all(pressure>0)):
            raise AssertionError('nonpositive density or pressure immediately after filter')
        pre_rhs = join(parts(case, 'pre_rhs', rk, ranks))
        primitive = join(parts(case, 'pre_rhs_primitives', rk, ranks))
        rho = pre_rhs[..., 0]
        velocity = pre_rhs[..., 1:4]/rho[..., None]
        prs = .4*(pre_rhs[..., 4]-.5*np.sum(pre_rhs[..., 1:4]*velocity, axis=-1))
        reference_primitive = np.concatenate((rho[..., None], velocity, prs[..., None],
                                              (1.4*.3**2*prs/rho)[..., None]), axis=-1)
        primitive_error = compare(primitive, reference_primitive)
        if ranks == 2:
            for rank in range(2):
                path = case/f'outdat/rhs.pre_rhs.step00000000.rk{rk:02}.rank{rank:08}.bin'
                (im, jm, km, halo, _), values = raw_field(path)
                j = slice(halo+jm+1, 2*halo+jm+1) if rank == 0 else slice(0, halo)
                by = pre_rhs[:, interface+1:interface+halo+1] if rank == 0 else pre_rhs[:, interface-halo:interface]
                compare(values[halo:halo+im+1, j, halo:halo+km+1], by)
        result.append(dict(reference=errors, primitives=primitive_error, filter=budget(after, before),
                           boundary=budget(pre_rhs, after), fixed_driver=0., mirror=mirror,
                           rk_update=budget(join(parts(case, 'post_update', rk, ranks)), pre_rhs)))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--mpiexec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    base = os.environ | dict(ASTR_VALIDATION_RHS_PREFIX='outdat/rhs', ASTR_VALIDATION_RHS_STEP='0',
                            ASTR_GPU_COMPACT_TRANSPORT='pinned', ASTR_GPU_COMPACT_SYNC='explicit',
                            ASTR_GPU_HALO_TRANSPORT='pinned', ASTR_GPU_SYNC_MODE='explicit',
                            OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_osc='pt2pt',
                            OMPI_MCA_coll='^cuda,hcoll,ucc')
    checkpoints = Path('outdat/new/checkpoints')
    final = checkpoints/'step000000000004/state.h5'

    def run(name, scheme, backend, ranks, restore=None, constant=False, mirrored=False):
        case = root/name
        n = 16 if mirrored else 11
        prepare_wall(case, '41y', backend, scheme, '643e', 'compact10', steps=4, intervals=n, restore=restore)
        cfg = case/'datin/input.case'
        cfg.write_text(cfg.read_text().replace("  filter_scheme = 'compact10'", "  filter_scheme = 'compact10'\n  filter_scope = 'y-wall5'"))
        if constant:
            with h5py.File(case/'datin/flowini3d.h5', 'r+') as field:
                for key in ('ro', 't', 'u1', 'u2', 'u3'):
                    field[key][...] = float(key in ('ro', 't'))
        if mirrored:
            k, j, i = np.indices((n+1, n+1, n+1))
            distance = np.minimum(j, n-j)
            wave = (-1.)**distance * (1+.1*np.cos(2*np.pi*i/n)*np.cos(2*np.pi*k/n))
            envelope = np.sin(np.pi*distance/n)**2
            with h5py.File(case/'datin/flowini3d.h5', 'r+') as field:
                field['ro'][...] = 1+.013*wave
                field['t'][...] = 1+.009*envelope*wave
                field['u1'][...] = .06*envelope*(1+.04*wave)
                field['u2'][...] = .004*np.sign(n/2-j)*envelope*wave
                field['u3'][...] = .015*envelope*(1-.04*wave)
        with (case/'run.log').open('w') as log:
            subprocess.run([str(args.mpiexec.resolve()), '-n', str(ranks), str(args.binary.resolve()), 'run'],
                           cwd=case, env=base | {'ASTR_FORCE_MPI_TOPOLOGY': f'1,{ranks},1'},
                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
        logged_steps(case, count=2 if restore else 4, first_step=3 if restore else 1)
        return case

    report = {}
    for scheme in ('643e', '643c'):
        for ranks in (1, 2):
            tag = f'{scheme}-np{ranks}'
            cases = {}
            entry = {}
            for backend in ('cpu', 'gpu'):
                case = run(f'{tag}-{backend}', scheme, backend, ranks)
                entry[backend] = verify_stages(case, ranks)
                cases[backend] = case
            entry['terminal'] = compare(checkpoint(cases['gpu']/final), checkpoint(cases['cpu']/final))
            if scheme == '643e' and ranks == 2:
                for backend, case in cases.items():
                    restored = run(f'{tag}-{backend}-restored', scheme, backend, ranks,
                                   restore=case/checkpoints/'step000000000002')
                    np.testing.assert_array_equal(checkpoint(restored/final).view(np.uint64), checkpoint(case/final).view(np.uint64))
                entry['exact_restart'] = 'bitwise CPU and GPU'
            report[tag] = entry
            print(f'{tag}: wall5 reference, interface, same-stage state and terminal CPU/GPU verified', flush=True)
    for backend in ('cpu', 'gpu'):
        case = run(f'constant-{backend}', '643e', backend, 2, constant=True)
        report[f'constant-{backend}'] = verify_stages(case, 2, constant=True)
        case = run(f'mirrored-{backend}', '643e', backend, 2, mirrored=True)
        report[f'mirrored-{backend}'] = verify_stages(case, 2, mirrored=True)
    (root/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
