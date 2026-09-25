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
from check_air5_c5_frozen_transport import read_rhs_snapshot
from check_air5_mach4_error_replay import primitive_metrics, cartesian_jacobian
from run_air5_sbli_preflight import ROOT, environment, set_value


def fixture_fields(kind, thermo, contact_trace=1e-12):
    if not np.isfinite(contact_trace) or not 0 <= contact_trace <= .01:
        raise ValueError('contact trace mass fraction outside fixture range')
    species = np.empty((49, 5))
    species[:] = [.70, .20, .03, .02, .05]
    tv = np.full(49, 350.)
    if kind == 'energy':
        tv[26] = 2000.
    elif kind == 'species':
        species[:] = [.78, .22, 0., 0., 0.]
        species[26] = [.77, .22, .01, 0., 0.]
        tv[:] = 1500.
    elif kind in ('contact', 'low_n2'):
        wave = .5*(1.-np.cos(2*np.pi*np.arange(49)/48))
        if kind == 'contact':
            species[:] = 0.
            species[:, 0] = .77-.1*wave
            species[:, 1] = 1.-species[:, 0]
            species[26, 4] = contact_trace
            species[26, 1] -= species[26, 4]
        else:
            species[:, 0] = 1e-12*(1.+wave)
            species[:, 1] = .7-.1*wave
            species[:, 2] = .1+.05*wave
            species[:, 4] = .02
            species[:, 3] = 1.-species[:, [0, 1, 2, 4]].sum(axis=1)
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


def fixture_axis(case):
    marker = case/'validation/fixture_axis.json'
    return json.loads(marker.read_text()) if marker.exists() else 0


def canonical_field(values, axis, momentum=True):
    """Undo the fixture's x/axis permutation, including vector components."""
    result = np.swapaxes(values, 0, axis).copy()
    if momentum and axis:
        result[..., [1, 1+axis]] = result[..., [1+axis, 1]]
    return result


def owned(case, label, stage, np_):
    axis = fixture_axis(case)
    return np.concatenate([canonical_field(_active_array(snap(case, label, stage, rank)), axis)[:-1, :-1, :-1]
                           for rank in range(np_)], axis=0)


def check_case(case, np_, thermo, consistent_convection=False, diffusion=True):
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
    channels = {'species': 'diffusion_species_ratio', 'energy': 'diffusion_energy_ratio'} if diffusion else {}
    if consistent_convection:
        channels.update(convection_fluid='convection_fluid_ratio',
                        convection_species='convection_species_ratio')
    for channel, label in channels.items():
        minimum, seam_minimum = 1., 1.
        for stage in range(1, 4):
            fields = [scalar_halo(snap(case, label, stage, rank))
                      for rank in range(np_)]
            minimum = min(minimum, *(float(v.min()) for _, v in fields))
            if np_ == 2:
                (im, jm, km, hm), left = fields[0]
                right_header, right = fields[1]
                if right_header != (im, jm, km, hm):
                    raise ValueError('expected equal slabs')
                axis = fixture_axis(case)
                dims = [im, jm, km]
                dims[0], dims[axis] = dims[axis], dims[0]
                im, jm, km = dims
                left = canonical_field(left, axis, momentum=False)
                right = canonical_field(right, axis, momentum=False)
                # Compare the shared node and both adjacent halo/interior nodes.
                a = left[hm+im-1:hm+im+2, hm:hm+jm, hm:hm+km]
                b = right[hm-1:hm+2, hm:hm+jm, hm:hm+km]
                np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)
                seam_minimum = min(seam_minimum, float(a.min()))
        result['coefficients'][channel] = dict(minimum=minimum, seam_minimum=seam_minimum)
    return result


def uniform_contact_drift(case, np_, thermo, dt=None):
    initial = owned(case, 'pre_rhs', 1, np_)
    final = owned(case, 'post_update', 3, np_)
    a, b = primitive_metrics(initial, thermo), primitive_metrics(final, thermo)
    u0, u1 = initial[..., 1:4]/initial[..., :1], final[..., 1:4]/final[..., :1]
    np.testing.assert_allclose(a[2], 4000., atol=1e-9, rtol=0)
    np.testing.assert_allclose(a[3], 1e5, atol=1e-7, rtol=0)
    np.testing.assert_allclose(u0, np.broadcast_to([100., 0., 0.], u0.shape), atol=1e-10, rtol=0)
    result = dict(max_pressure_drift_pa=float(abs(b[3]-1e5).max()),
                max_temperature_drift_k=float(abs(b[2]-4000.).max()),
                max_velocity_drift_m_s=abs(u1-[100., 0., 0.]).max(axis=(0, 1, 2)).tolist(),
                scope='diagnostic of contact compatibility, not an asserted pressure-equilibrium pass')
    if dt is not None:
        jac = cartesian_jacobian(case)
        def rhs(label):
            fields = []
            for rank in range(np_):
                s = read_rhs_snapshot(snap(case, label, 1, rank))
                im, jm, km, nq = s.header
                field = s.values.reshape(im+1, jm+1, km+1, nq, order='F')
                fields.append(canonical_field(field, fixture_axis(case))[:-1, :-1, :-1]/jac)
            return np.concatenate(fields, axis=0)
        raw, limited = rhs('conv_raw'), rhs('conv')
        after = owned(case, 'post_update', 1, np_)
        np.testing.assert_allclose(after, initial+dt*limited, atol=1e-9, rtol=1e-12)
        def heat_invariant(q):
            return (q[..., 4]-q[..., 10]-100.*q[..., 1]+5000.*q[..., 0]
                    -q[..., 5:10]@(thermo.formation_energy+4000.*thermo.cv_tr))
        cv0, cv1 = initial[..., 5:10]@thermo.cv_tr, after[..., 5:10]@thermo.cv_tr
        velocity = after[..., 1:4]/after[..., :1]
        departure = .5*after[..., 0]*np.sum((velocity-[100., 0., 0.])**2, axis=-1)
        predicted = (heat_invariant(initial)+dt*heat_invariant(limited)-departure)/cv1
        actual = primitive_metrics(after, thermo)[2]-4000.
        np.testing.assert_allclose(predicted, actual, atol=1e-9, rtol=1e-10)
        result['first_stage'] = dict(
            raw_temperature_invariant_defect_k=float(abs(dt*heat_invariant(raw)/cv0).max()),
            limited_temperature_invariant_defect_k=float(abs(dt*heat_invariant(limited)/cv0).max()),
            measured_temperature_change_k=float(abs(actual).max()),
            predicted_temperature_closure_k=float(abs(predicted-actual).max()),
            limiter_increment_component_max_abs=(abs(dt*(limited-raw)).max(axis=(0, 1, 2))).tolist())
    return result


def require_contact_equilibrium(drift):
    # Same constant-state tolerances as the initial p/T/u checks above.
    if (drift['max_pressure_drift_pa'] > 1e-7 or
            drift['max_temperature_drift_k'] > 1e-9 or
            max(drift['max_velocity_drift_m_s']) > 1e-10):
        raise ValueError('constant pressure/temperature/velocity compatibility failed')


def symmetric_species_error(reference, candidate, initial):
    """Trace-relative comparison without a bulk-density absolute floor."""
    peak = np.max(abs(initial[..., 5:10]), axis=(0, 1, 2))
    error = np.max(abs(candidate[..., 5:10]-reference[..., 5:10]), axis=(0, 1, 2))
    relative = np.zeros(5)
    for s in range(5):
        if peak[s] == 0:
            if np.any(reference[..., 5+s] != 0) or np.any(candidate[..., 5+s] != 0):
                raise ValueError(f'initially absent frozen species {s} was generated')
        else:
            relative[s] = error[s]/peak[s]
    if np.any(relative > 1e-9):
        raise ValueError(f'species-relative CPU/MPI/GPU gate failed: {relative.tolist()}')
    return relative


def check_shared_face_payloads(case, topology):
    """Bitwise owner/receiver checks for complete planes, including signed zero."""
    topology = tuple(topology)
    if len(topology) != 3 or min(topology) < 1:
        raise ValueError('invalid shared-face topology')
    ranks = int(np.prod(topology))
    checked = 0
    for stage in range(1, 4):
        for axis in range(1, 4):
            payloads = []
            for rank in range(ranks):
                filename = case/'validation'/(
                    f'air5.shared_faces.axis{axis}.step00000000.rk{stage:02d}.rank{rank:08d}.bin')
                with filename.open('rb') as stream:
                    header = np.fromfile(stream, np.int32, 7)
                    values = np.fromfile(stream, np.float64)
                if (len(header) != 7 or tuple(header[:4]) != (0, stage, rank, axis)
                        or min(header[4:6]) < 1 or header[6] != 12
                        or len(values) != 4*int(np.prod(header[4:])) or not np.isfinite(values).all()):
                    raise ValueError(f'invalid shared-face record: {filename}')
                payloads.append(values.reshape(4, -1).view(np.uint64))
            for rank, own in enumerate(payloads):
                i = rank % topology[0]
                j = (rank//topology[0]) % topology[1]
                k = rank//(topology[0]*topology[1])
                previous = [i, j, k]
                following = previous.copy()
                previous[axis-1] = (previous[axis-1]-1) % topology[axis-1]
                following[axis-1] = (following[axis-1]+1) % topology[axis-1]
                def rank_of(node):
                    return node[0]+topology[0]*(node[1]+topology[1]*node[2])
                # First -> previous rank's plus face; last -> next rank's minus.
                if (not np.array_equal(own[0], payloads[rank_of(previous)][2]) or
                        not np.array_equal(own[1], payloads[rank_of(following)][3])):
                    raise ValueError(f'shared face is not bitwise identical: RK{stage}, axis{axis}, rank{rank}')
                checked += 2
    return dict(plane_pairs=checked, bitwise_identical=True)


def run(args):
    if not np.isfinite((args.dt, args.ref_len)).all() or min(args.dt, args.ref_len) <= 0:
        raise ValueError('dt and ref_len must be finite and positive')
    backends = getattr(args, 'backends', ('cpu', 'gpu'))
    if 'gpu' in backends and 'cpu' not in backends:
        raise ValueError('GPU comparison requires the CPU reference in the same gate')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    thermo = Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json')
    baseline = args.baseline.resolve()
    limiter = getattr(args, 'convection_limiter', 'species_budget')
    contact_trace = getattr(args, 'contact_trace', 1e-12)
    axis = 'xyz'.index(getattr(args, 'axis', 'x'))
    grid = [48, 6, 6]
    grid[0], grid[axis] = grid[axis], grid[0]
    report = dict(status='running', cases={}, dt=args.dt, ref_len=args.ref_len, grid=grid,
        decomposition_axis='xyz'[axis],
        convection_limiter=limiter, diffusion_limiter='layered',
        contact_trace_mass_fraction=contact_trace,
        scope='manufactured periodic limiter stress test, not physical validation',
        executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
        baseline_checkpoint_sha256=hashlib.sha256((baseline/'outdat/flowfield.h5').read_bytes()).hexdigest())
    shutil.copy2(args.executable.resolve(), root/'astr')
    try:
        for kind in args.fixture:
            contact = kind in ('contact', 'low_n2')
            for backend, np_ in [('cpu', 1), ('cpu', 2), ('gpu', 2)]:
                if backend not in getattr(args, 'backends', ('cpu', 'gpu')):
                    continue
                name = f'{kind}_{backend}_np{np_}'
                case = root/name
                shutil.copytree(baseline/'datin', case/'datin')
                (case/'outdat').mkdir()
                (case/'validation').mkdir()
                (case/'validation/fixture_axis.json').write_text(json.dumps(axis)+'\n')
                for filename in ('flowfield.h5', 'auxiliary.txt'):
                    shutil.copy2(baseline/'outdat'/filename, case/'outdat'/filename)
                species, density, tv = fixture_fields(kind, thermo, contact_trace)
                with h5py.File(case/'outdat/flowfield.h5', 'r+') as f:
                    if f['ro'].shape != (7, 7, 49):
                        raise ValueError('requires the 48x6x6 periodic ev-pulse checkpoint')
                    f['nstep'][...] = 0
                    f['time'][...] = 0.
                    for key in ('u1', 'u2', 'u3'):
                        f[key][...] = 0.
                    if contact:
                        f['u1'][...] = 100.
                    f['ro'][...] = np.broadcast_to(density, f['ro'].shape)
                    f['p'][...] = 1.e5
                    f['t'][...] = 4000.
                    f['tv'][...] = np.broadcast_to(tv, f['tv'].shape)
                    for s in range(5):
                        f[f'sp{s+1:03d}'][...] = np.broadcast_to(species[:, s], f['ro'].shape)
                    if axis:
                        fields = {key: np.swapaxes(f[key][...], 2, 2-axis)
                                  for key in f if f[key].ndim == 3}
                        fields['u1'], fields[f'u{axis+1}'] = fields[f'u{axis+1}'], fields['u1']
                        for key, values in fields.items():
                            del f[key]
                            f.create_dataset(key, data=values)
                inp = case/'datin/input.air5_c4'
                if '\nair5evpulse\n' not in inp.read_text():
                    raise ValueError('stress fixtures require frozen air5evpulse, not a reacting flowtype')
                set_value(inp, 'lrestar', 't')
                set_value(inp, 'im,jm,km', ','.join(map(str, grid)))
                set_value(inp, 'ref_tem,ref_vel,ref_len,ref_den', f'1000.,10.,{args.ref_len:.17e},1.')
                set_value(inp, 'nondimen,diffterm,lfilter,lreadgrid,lfftz,limmbou,ltimrpt,lcomb',
                          'f,'+('f' if contact else 't')+',f,f,f,f,t,t,'+('t' if backend == 'gpu' else 'f'))
                if contact:
                    set_value(inp, 'recon_schem, lchardecomp,bfacmpld,shkcrt', '3,f,0.3d0,0.05d0')
                set_value(case/'datin/controller', 'deltat', f'{args.dt:.17e}')
                set_value(case/'datin/controller', 'maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg',
                          '0,2,1000000,1000000,1,1000000')
                topology = [1, 1, 1]
                topology[axis] = np_
                env = environment(','.join(map(str, topology)))
                env.update(ASTR_AIR5_CONVECTION_LIMITER=limiter,
                    ASTR_AIR5_DIFFUSION_LIMITER='layered', ASTR_VALIDATION_RHS_PREFIX='validation/air5')
                if getattr(args, 'symmetric_face_probe', False):
                    env['ASTR_AIR5_SYMMETRIC_FACE_PROBE'] = '1'
                instrument = []
                memcheck = getattr(args, 'memcheck', False) and backend == 'gpu'
                if memcheck:
                    sanitizer = shutil.which('compute-sanitizer')
                    if sanitizer is None:
                        raise RuntimeError('compute-sanitizer is required for --memcheck')
                    instrument = [sanitizer, '--tool', 'memcheck', '--error-exitcode', '99',
                                  '--log-file', 'validation/memcheck.%p.log']
                command = ['timeout', '--kill-after=10s', '180s', shutil.which('mpirun'),
                           '--oversubscribe', '-np', str(np_), *instrument, str(root/'astr'),
                           'run', 'datin/input.air5_c4']
                with (case/'run.log').open('w') as log:
                    process = subprocess.run(command, cwd=case, env=env, stdout=log, stderr=subprocess.STDOUT)
                report['cases'][name] = dict(returncode=process.returncode)
                process.check_returncode()
                if memcheck:
                    logs = sorted((case/'validation').glob('memcheck.*.log'))
                    if len(logs) != np_ or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in logs):
                        raise ValueError(f'incomplete or failed memcheck reports: {case}')
                    report['cases'][name]['memcheck'] = dict(rank_logs=len(logs), errors=0)
                log = (case/'run.log').read_text()
                cfl = [float(v) for v in re.findall(r'current CFL:\s+(\S+)', log)]
                if 'The job is done!' not in log or not cfl or not 0 <= max(cfl) < 1:
                    raise ValueError(f'completion/CFL gate failed: {case}')
                result = check_case(case, np_, thermo, limiter == 'consistent_species', not contact)
                if limiter == 'symmetric_species' and getattr(args, 'symmetric_face_probe', False):
                    result['shared_faces'] = check_shared_face_payloads(case, topology)
                if contact:
                    result['contact_drift'] = uniform_contact_drift(case, np_, thermo, args.dt)
                report['cases'][name].update(result, max_cfl=max(cfl))
                if contact and getattr(args, 'require_contact_equilibrium', False):
                    require_contact_equilibrium(result['contact_drift'])
                if not contact and result['coefficients'][kind]['minimum'] >= 1:
                    raise ValueError(f'{kind} constraint was not activated')
                if not contact and np_ == 2 and result['coefficients'][kind]['seam_minimum'] >= 1:
                    raise ValueError(f'{kind} constraint missed the MPI interface')
                if limiter == 'consistent_species' and not contact:
                    channel = 'convection_fluid' if kind == 'energy' else 'convection_species'
                    if result['coefficients'][channel]['minimum'] >= 1 or (
                            np_ == 2 and result['coefficients'][channel]['seam_minimum'] >= 1):
                        raise ValueError(f'{channel} constraint was not activated at the required nodes')
                if np_ == 2:
                    reference = root/f'{kind}_cpu_np1'
                    error = 0.
                    species_error = np.zeros(5)
                    initial = owned(reference, 'pre_rhs', 1, 1)
                    for label in ('pre_rhs', 'post_update'):
                        for stage in range(1, 4):
                            a, b = owned(reference, label, stage, 1), owned(case, label, stage, 2)
                            np.testing.assert_allclose(b, a, atol=1e-9, rtol=1e-10)
                            error = max(error, float((abs(b-a)/np.maximum(abs(a), 1.)).max()))
                            if limiter == 'symmetric_species':
                                species_error = np.maximum(species_error, symmetric_species_error(a, b, initial))
                    result['np1_cpu_scaled_difference'] = error
                    if limiter == 'symmetric_species':
                        result['np1_cpu_species_relative_difference'] = species_error.tolist()
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
    parser.add_argument('--fixture', nargs='+', choices=('energy', 'species', 'contact', 'low_n2'),
                        default=['energy', 'species'])
    parser.add_argument('--ref-len', type=float, default=1e-4)
    parser.add_argument('--dt', type=float, default=5e-9)
    parser.add_argument('--contact-trace', type=float, default=1e-12,
                        help='initial localized NO mass fraction for the contact fixture, not a limiter floor')
    parser.add_argument('--require-contact-equilibrium', action='store_true',
                        help='fail if the constant-state p/T/u tolerances are violated after the update')
    parser.add_argument('--backends', nargs='+', choices=('cpu', 'gpu'), default=['cpu', 'gpu'])
    parser.add_argument('--symmetric-face-probe', action='store_true')
    parser.add_argument('--memcheck', action='store_true', help='run each GPU MPI rank under Compute Sanitizer')
    parser.add_argument('--axis', choices=('x', 'y', 'z'), default='x',
                        help='rotate the fixture and two-rank slab together; all gates stay unchanged')
    parser.add_argument('--convection-limiter',
                        choices=('full_state', 'species_budget', 'consistent_species', 'symmetric_species'),
                        default='species_budget')
    parser.add_argument('--executable', type=Path, default=ROOT/'tests/gpu_validation/out/c4_cuda_build_4/bin/astr')
    run(parser.parse_args())
