#!/usr/bin/env python3
"""Exercise layered diffusion in ASTR; Python only prepares and checks fields."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import h5py
import numpy as np

from air5_radau_reference import Air5RadauReference
from check_air5_c5_hbl import _active_array
from check_air5_mach4_error_replay import primitive_metrics
from run_air5_sbli_preflight import ROOT, environment, set_value


def fixture_fields(kind, thermo):
    species = np.empty((49, 5))
    species[:] = [.70, .20, .03, .02, .05]
    tv = np.full(49, 350.)
    if kind == 'energy':
        tv[26] = 2000.
    elif kind == 'species':
        species[:] = [.78, .22, 0., 0., 0.]
        species[26] = [.77, .22, .01, 0., 0.]
        tv[:] = 1500.
    else:
        raise ValueError('unknown fixture')
    density = 1.e5/(4000.*(species@thermo.gas_constant))
    return species, density, tv


def scalar_halo(filename):
    with filename.open('rb') as stream:
        header = tuple(np.fromfile(stream, np.int32, 4))
        data = np.fromfile(stream, np.float64)
    if len(header) != 4 or min(header) < 0:
        raise ValueError('invalid coefficient header')
    im, jm, km, hm = header
    values = data.reshape((im+2*hm+1, jm+2*hm+1, km+2*hm+1), order='F')
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('invalid coefficient values')
    return header, values


def snap(case, label, stage, rank):
    return case/'validation'/f'air5.{label}.step00000000.rk{stage:02d}.rank{rank:08d}.bin'


def owned(case, label, stage, np_):
    return np.concatenate([_active_array(snap(case, label, stage, rank))[:-1, :-1, :-1]
                           for rank in range(np_)], axis=0)


def check_case(case, np_, thermo):
    result = dict(min_species=float('inf'), min_temperature=float('inf'),
                  min_vibrational_excess=float('inf'), coefficients={})
    ev_floor = np.array([thermo.species_vibrational_energy(s, 300.) for s in range(5)])
    ev_ceiling = np.array([thermo.species_vibrational_energy(s, 8000.) for s in range(5)])
    for label in ('pre_rhs', 'post_update'):
        for stage in range(1, 4):
            q = owned(case, label, stage, np_)
            rho, species, temp, pressure, _ = primitive_metrics(q, thermo)
            excess = q[..., 10]-species@ev_floor
            if (species.min() < 0 or rho.min() <= 0 or temp.min() < 1000 or
                temp.max() > 8000 or pressure.min() < thermo.pressure_bounds[0] or
                pressure.max() > thermo.pressure_bounds[1] or excess.min() < 0 or
                np.any(q[..., 10] > species@ev_ceiling)):
                raise ValueError(f'invalid accepted state: {case}/{label}/RK{stage}')
            np.testing.assert_allclose(species.sum(axis=-1), rho, atol=1e-14, rtol=2e-12)
            result['min_species'] = min(result['min_species'], float(species.min()))
            result['min_temperature'] = min(result['min_temperature'], float(temp.min()))
            result['min_vibrational_excess'] = min(result['min_vibrational_excess'], float(excess.min()))
    initial = owned(case, 'pre_rhs', 1, np_)
    final = owned(case, 'post_update', 3, np_)
    residual = abs(final.sum(axis=(0, 1, 2))-initial.sum(axis=(0, 1, 2)))
    scale = np.maximum(abs(initial).sum(axis=(0, 1, 2)), 1.)
    result['conservation_scaled'] = float((residual/scale).max())
    if result['conservation_scaled'] > 2e-11:
        raise ValueError('periodic conservative flux balance failed')
    for channel in ('species', 'energy'):
        minimum, seam_minimum = 1., 1.
        for stage in range(1, 4):
            fields = [scalar_halo(snap(case, f'diffusion_{channel}_ratio', stage, rank))
                      for rank in range(np_)]
            minimum = min(minimum, *(float(v.min()) for _, v in fields))
            if np_ == 2:
                (im, jm, km, hm), left = fields[0]
                right_header, right = fields[1]
                if right_header != (im, jm, km, hm):
                    raise ValueError('expected equal x slabs')
                # Compare the shared node and both adjacent halo/interior nodes.
                a = left[hm+im-1:hm+im+2, hm:hm+jm, hm:hm+km]
                b = right[hm-1:hm+2, hm:hm+jm, hm:hm+km]
                np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)
                seam_minimum = min(seam_minimum, float(a.min()))
        result['coefficients'][channel] = dict(minimum=minimum, seam_minimum=seam_minimum)
    return result


def run(args):
    if not np.isfinite((args.dt, args.ref_len)).all() or min(args.dt, args.ref_len) <= 0:
        raise ValueError('dt and ref_len must be finite and positive')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    thermo = Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json')
    baseline = args.baseline.resolve()
    report = dict(status='running', cases={}, dt=args.dt, ref_len=args.ref_len, grid=[48, 6, 6],
        scope='manufactured periodic limiter stress test, not physical validation',
        executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
        baseline_checkpoint_sha256=hashlib.sha256((baseline/'outdat/flowfield.h5').read_bytes()).hexdigest())
    shutil.copy2(args.executable.resolve(), root/'astr')
    try:
        for kind in args.fixture:
            for backend, np_ in [('cpu', 1), ('cpu', 2), ('gpu', 2)]:
                name = f'{kind}_{backend}_np{np_}'
                case = root/name
                shutil.copytree(baseline/'datin', case/'datin')
                (case/'outdat').mkdir()
                (case/'validation').mkdir()
                for filename in ('flowfield.h5', 'auxiliary.txt'):
                    shutil.copy2(baseline/'outdat'/filename, case/'outdat'/filename)
                species, density, tv = fixture_fields(kind, thermo)
                with h5py.File(case/'outdat/flowfield.h5', 'r+') as f:
                    if f['ro'].shape != (7, 7, 49):
                        raise ValueError('requires the 48x6x6 periodic ev-pulse checkpoint')
                    f['nstep'][...] = 0
                    f['time'][...] = 0.
                    for key in ('u1', 'u2', 'u3'):
                        f[key][...] = 0.
                    f['ro'][...] = np.broadcast_to(density, f['ro'].shape)
                    f['p'][...] = 1.e5
                    f['t'][...] = 4000.
                    f['tv'][...] = np.broadcast_to(tv, f['tv'].shape)
                    for s in range(5):
                        f[f'sp{s+1:03d}'][...] = np.broadcast_to(species[:, s], f['ro'].shape)
                inp = case/'datin/input.air5_c4'
                if '\nair5evpulse\n' not in inp.read_text():
                    raise ValueError('stress fixtures require frozen air5evpulse, not a reacting flowtype')
                set_value(inp, 'lrestar', 't')
                set_value(inp, 'ref_tem,ref_vel,ref_len,ref_den', f'1000.,10.,{args.ref_len:.17e},1.')
                set_value(inp, 'nondimen,diffterm,lfilter,lreadgrid,lfftz,limmbou,ltimrpt,lcomb',
                          'f,t,f,f,f,f,t,t,'+('t' if backend == 'gpu' else 'f'))
                set_value(case/'datin/controller', 'deltat', f'{args.dt:.17e}')
                set_value(case/'datin/controller', 'maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg',
                          '0,2,1000000,1000000,1,1000000')
                env = environment(f'{np_},1,1')
                env.update(ASTR_AIR5_CONVECTION_LIMITER='species_budget',
                    ASTR_AIR5_DIFFUSION_LIMITER='layered', ASTR_VALIDATION_RHS_PREFIX='validation/air5')
                command = ['timeout', '--kill-after=10s', '180s', shutil.which('mpirun'),
                           '--oversubscribe', '-np', str(np_), str(root/'astr'),
                           'run', 'datin/input.air5_c4']
                with (case/'run.log').open('w') as log:
                    process = subprocess.run(command, cwd=case, env=env, stdout=log, stderr=subprocess.STDOUT)
                report['cases'][name] = dict(returncode=process.returncode)
                process.check_returncode()
                log = (case/'run.log').read_text()
                cfl = [float(v) for v in re.findall(r'current CFL:\s+(\S+)', log)]
                if 'The job is done!' not in log or not cfl or not 0 <= max(cfl) < 1:
                    raise ValueError(f'completion/CFL gate failed: {case}')
                result = check_case(case, np_, thermo)
                report['cases'][name].update(result, max_cfl=max(cfl))
                if result['coefficients'][kind]['minimum'] >= 1:
                    raise ValueError(f'{kind} constraint was not activated')
                if np_ == 2 and result['coefficients'][kind]['seam_minimum'] >= 1:
                    raise ValueError(f'{kind} constraint missed the MPI interface')
                if np_ == 2:
                    reference = root/f'{kind}_cpu_np1'
                    error = 0.
                    for label in ('pre_rhs', 'post_update'):
                        for stage in range(1, 4):
                            a, b = owned(reference, label, stage, 1), owned(case, label, stage, 2)
                            np.testing.assert_allclose(b, a, atol=1e-9, rtol=1e-10)
                            error = max(error, float((abs(b-a)/np.maximum(abs(a), 1.)).max()))
                    result['np1_cpu_scaled_difference'] = error
                report['cases'][name].update(result, max_cfl=max(cfl))
                print(name, json.dumps(report['cases'][name]), flush=True)
        report['status'] = 'layered-stress-pass-not-physical-pass'
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        (root/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', nargs='+', choices=('energy', 'species'), default=['energy', 'species'])
    parser.add_argument('--ref-len', type=float, default=1e-4)
    parser.add_argument('--dt', type=float, default=5e-9)
    parser.add_argument('--executable', type=Path, default=ROOT/'tests/gpu_validation/out/c4_cuda_build_4/bin/astr')
    run(parser.parse_args())
