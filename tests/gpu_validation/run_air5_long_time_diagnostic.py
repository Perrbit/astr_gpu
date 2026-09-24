#!/usr/bin/env python3
"""Fresh NP=1 normal-shock diagnostic, not a physical acceptance driver."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

import h5py
import numpy as np

from air5_radau_reference import Air5RadauReference
from compare_q_validation_snapshots import read_q_snapshot
from prepare_air5_c4_case import next_data_line, replace_after_marker

ROOT = Path(__file__).resolve().parents[2]
DT = 8e-8
MAXSTEP = 624
WARNING = 2e-11
FIELDS = ('ro', 'u1', 'u2', 'u3', 'p', 't', 'tv',
          'sp001', 'sp002', 'sp003', 'sp004', 'sp005')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def volume_mean(a):
    """Uniform Cartesian x quadrature; periodic y/z endpoints excluded upstream."""
    weights = np.ones(a.shape[2])
    weights[[0, -1]] = .5
    return float(np.mean(np.average(a, axis=2, weights=weights)))


def metrics(fields, thermo, reference):
    """HDF arrays are (z,y,x); RMS uses unique periodic nodes, maxima use all."""
    if any(not np.isfinite(fields[k]).all() for k in FIELDS):
        raise ValueError('non-finite primitive state')
    rho = fields['ro']
    y = np.stack([fields[f'sp{i:03d}'] for i in range(1, 6)], axis=-1)
    if np.min(rho) <= 0 or np.min(y) < 0:
        raise ValueError('nonpositive density or negative accepted species')
    for key, bounds in (('t', thermo.temperature_bounds),
                        ('tv', thermo.temperature_bounds),
                        ('p', thermo.pressure_bounds)):
        if fields[key].min() < bounds[0] or fields[key].max() > bounds[1]:
            raise ValueError(f'{key} outside fixed mechanism property domain')
    vel = np.stack([fields[f'u{i}'] for i in (1, 2, 3)], axis=-1)
    species = rho[..., None]*y
    ev = np.zeros_like(rho)
    for i, theta in enumerate(thermo.theta_v):
        if theta > 0:
            ev += species[..., i]*thermo.gas_constant[i]*theta/np.expm1(theta/fields['tv'])
    energy = (.5*rho*np.sum(vel*vel, axis=-1) + (species@thermo.cv_tr)*fields['t']
              + species@thermo.formation_energy + ev)
    q = np.concatenate([rho[..., None], rho[..., None]*vel, energy[..., None],
                        species, ev[..., None]], axis=-1)
    if not np.isfinite(q).all():
        raise ValueError('non-finite reconstructed conservative state')
    scale = np.maximum(np.max(abs(q), axis=(0, 1, 2)), 1.)
    extrusion = float(np.max(abs(q-q[:1, :1])/scale))
    transverse_speed2 = np.sum(vel[..., 1:]**2, axis=-1)
    p = fields['p'][:-1, :-1]
    pprime = p-p.mean(axis=(0, 1), keepdims=True)
    unique_y = y[:-1, :-1]
    yprime = unique_y-unique_y.mean(axis=(0, 1), keepdims=True)
    uref, pref, rhoref = reference
    mean_p = p.mean(axis=(0, 1))
    return dict(
        reconstructed_q_extrusion=extrusion, extrusion_warning=extrusion > WARNING,
        max_transverse_velocity=float(np.sqrt(transverse_speed2.max())),
        max_transverse_velocity_over_uinf=float(np.sqrt(transverse_speed2.max())/uref),
        rms_transverse_velocity_over_uinf=float(np.sqrt(volume_mean(transverse_speed2[:-1, :-1]))/uref),
        transverse_kinetic_energy_over_dynamic_pressure=volume_mean(
            rho[:-1, :-1]*transverse_speed2[:-1, :-1])/(rhoref*uref**2),
        pressure_transverse_rms_over_pinf=float(np.sqrt(volume_mean(pprime**2))/pref),
        pressure_transverse_max_over_pinf=float(np.max(abs(fields['p']-fields['p'][:1, :1]))/pref),
        species_mass_fraction_transverse_rms=[float(np.sqrt(volume_mean(yprime[..., s]**2))) for s in range(5)],
        min_species_density=float(species.min()), min_density=float(rho.min()),
        min_temperature=float(fields['t'].min()), max_temperature=float(fields['t'].max()),
        min_tv=float(fields['tv'].min()), max_tv=float(fields['tv'].max()),
        min_pressure=float(fields['p'].min()), max_pressure=float(fields['p'].max()),
        mass_fraction_closure=float(np.max(abs(y.sum(axis=-1)-1))),
        shock_x=(int(np.argmax(abs(np.diff(mean_p))))+.5)*.02/(rho.shape[2]-1),
        outlet_pressure_mean=float(mean_p[-1]))


def final_fields(path, thermo):
    snapshot = read_q_snapshot(path)
    im, jm, km, hm, nq = snapshot.header
    q = snapshot.values.reshape((im+2*hm+1, jm+2*hm+1, km+2*hm+1, nq), order='F')
    q = q[hm:hm+im+1, hm:hm+jm+1, hm:hm+km+1].transpose(2, 1, 0, 3)
    rho, species = q[..., 0], q[..., 5:10]
    if np.any(rho <= 0) or np.any(species < 0):
        raise ValueError('invalid accepted final q')
    vel = q[..., 1:4]/rho[..., None]
    temp = (q[..., 4]-.5*rho*np.sum(vel*vel, axis=-1)-q[..., 10]
            -species@thermo.formation_energy)/(species@thermo.cv_tr)
    tv = np.array([thermo.tv_from_ev(s, e) for s, e in
                   zip(species.reshape(-1, 5), q[..., 10].ravel())]).reshape(rho.shape)
    result = dict(ro=rho, t=temp, tv=tv, p=(species@thermo.gas_constant)*temp)
    result.update({f'u{i+1}': vel[..., i] for i in range(3)})
    result.update({f'sp{i+1:03d}': species[..., i]/rho for i in range(5)})
    return result


def run(args):
    baseline = args.baseline.resolve()
    manifest = json.loads((baseline/'provenance.json').read_text())
    executable_manifest = json.loads(args.executable_manifest.read_text()) if args.executable_manifest else manifest
    exe = args.executable.resolve()
    lines = (baseline/'gpu/datin/input.air5_c4').read_text().splitlines()
    marker = next(i for i, line in enumerate(lines) if 'nondimen,diffterm,lfilter,lreadgrid' in line)
    flags = [v.strip().lower() for v in lines[next_data_line(lines, marker)].split(',')]
    if flags[:4] != ['f', 'f', 'f', 'f']:
        raise ValueError('diagnostic requires dimensional inviscid unfiltered generated grid')
    for path in (exe, ROOT/'src/chemistry_solver.F90', ROOT/'src_gpu/chemistry_solver_gpu.cuf',
                 *sorted((baseline/'gpu/datin').glob('*'))):
        if path.name == 'grid.h5':
            continue  # lreadgrid=f: ASTR overwrites this output, it is not an input.
        expected = manifest if path.parent == baseline/'gpu/datin' else executable_manifest
        if path.is_file() and expected.get(str(path.relative_to(ROOT))) != digest(path):
            raise ValueError(f'baseline provenance mismatch: {path}')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    case = out/'gpu'
    case.mkdir()
    shutil.copytree(baseline/'gpu/datin', case/'datin', ignore=shutil.ignore_patterns('grid.h5'))
    shutil.copy2(exe, out/'astr')
    shutil.copy2(ROOT/'chemMech/air5_kimjo12.json', out/'mechanism.json')
    (case/'validation').mkdir()
    for name, marker, value in (
        ('input.air5_c4', 'lrestar', 'f'),
        ('controller', 'maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg', '624,5,10,50,5,50'),
        ('controller', 'deltat', '8.d-8'),
    ):
        path = case/'datin'/name
        lines = path.read_text().splitlines()
        replace_after_marker(lines, marker, value)
        path.write_text('\n'.join(lines)+'\n')
    # Do not inherit experimental ASTR modes from an interactive shell.
    env = {k: v for k, v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(ASTR_FORCE_MPI_TOPOLOGY='1,1,1', ASTR_GPU_SYNC_MODE='explicit',
               ASTR_VALIDATION_RHS_PREFIX='validation/air5', ASTR_VALIDATION_RHS_STEP='0',
               ASTR_VALIDATION_RHS_STEP_SECONDARY=str(MAXSTEP), ASTR_AIR5_C4_CONSERVATION='f',
               ASTR_AIR5_SOURCE_MODE='coupled', OMPI_MCA_coll='^hcoll,ucc', OMPI_MCA_pml='ob1',
               OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_osc='pt2pt',
               OMPI_MCA_opal_cuda_support='0', UCX_MEMTYPE_CACHE='n', CUDA_VISIBLE_DEVICES='0')
    thermo = Air5RadauReference(out/'mechanism.json')
    upstream = np.loadtxt(case/'datin/air5_normal_shock_states.dat')[0]
    reference = (upstream[1], upstream[2], upstream[0])
    command = [shutil.which('mpirun'), '-np', '1', str(out/'astr'), 'run', 'datin/input.air5_c4']
    contract = dict(kind='long-time-diagnostic-not-physical-acceptance', updates=625,
                    dt=DT, target_time=625*DT, warning_threshold=WARNING, command=command,
                    baseline=str(baseline), input_baseline=manifest,
                    executable_baseline=executable_manifest,
                    environment={k: v for k, v in env.items() if k.startswith(('ASTR_', 'OMPI_', 'UCX_', 'CUDA_'))},
                    files={str(p.relative_to(out)): digest(p) for p in
                           (out/'astr', out/'mechanism.json', *sorted((case/'datin').glob('*'))) if p.is_file()},
                    driver_sha256=digest(__file__))
    (out/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    previous = -1
    start = time.monotonic()
    warnings = 0

    def record(fields, step, elapsed, phase):
        nonlocal warnings
        item = dict(step=step, time=elapsed, phase=phase,
                    wall_seconds=time.monotonic()-start, **metrics(fields, thermo, reference))
        warnings += int(item['extrusion_warning'])
        with (out/'statistics.jsonl').open('a') as stream:
            stream.write(json.dumps(item, allow_nan=False)+'\n')
        print(json.dumps(item, allow_nan=False), flush=True)
        return item

    def inspect_checkpoint():
        nonlocal previous
        path = case/'outdat/flowfield.h5'
        try:
            before = path.stat()
            if time.time()-before.st_mtime < 2:
                return
            data = path.read_bytes()
            if path.stat().st_mtime_ns != before.st_mtime_ns:
                return
            with h5py.File(io.BytesIO(data), 'r') as stream:
                if any(k not in stream for k in (*FIELDS, 'nstep', 'time')):
                    return
                step = int(stream['nstep'][()].item())
                if step <= previous:
                    return
                elapsed = float(stream['time'][()].item())
                fields = {k: stream[k][()] for k in FIELDS}
        except (OSError, BlockingIOError):
            return
        if not np.isclose(elapsed, step*DT, rtol=1e-10, atol=1e-18):
            raise ValueError('checkpoint step/time mismatch')
        if step in (5, 60, 65, 200, 620):
            (out/f'checkpoint_step{step:06d}.h5').write_bytes(data)
        record(fields, step, elapsed, 'checkpoint-completed-updates')
        previous = step

    def stop_signal(signum, frame):
        raise InterruptedError(f'service received signal {signum}')

    signal.signal(signal.SIGTERM, stop_signal)
    proc = None
    try:
        with (case/'run.log').open('w') as log:
            proc = subprocess.Popen(command, cwd=case, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            (out/'status.json').write_text(json.dumps(dict(status='running', pid=proc.pid))+'\n')
            while proc.poll() is None:
                inspect_checkpoint()
                if time.monotonic()-start > args.timeout:
                    raise TimeoutError('long-time diagnostic wall-time limit')
                time.sleep(2)
            if proc.returncode:
                raise RuntimeError(f'ASTR exit code {proc.returncode}')
        time.sleep(2)
        inspect_checkpoint()
        text = (case/'run.log').read_text()
        if 'The job is done!' not in text:
            raise RuntimeError('ASTR did not report completion')
        cfl = [float(v) for v in re.findall(r'current CFL:\s*(\S+)', text)]
        if not cfl or not np.isfinite(cfl).all() or max(cfl) >= 1:
            raise ValueError('CFL diagnostic safety gate failed')
        path = case/'validation'/f'air5.post_chemistry.step{MAXSTEP:08d}.rk02.rank00000000.bin'
        record(final_fields(path, thermo), 625, 625*DT, 'final-post-chemistry')
        status = dict(status='diagnostic-complete-not-physical-pass', warning_samples=warnings,
                      max_cfl=max(cfl), wall_seconds=time.monotonic()-start)
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        status = dict(status='stopped', reason=str(exc), wall_seconds=time.monotonic()-start)
        (out/'status.json').write_text(json.dumps(status, indent=2)+'\n')
        raise
    (out/'status.json').write_text(json.dumps(status, indent=2)+'\n')
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--executable', type=Path,
                        default=ROOT/'tests/gpu_validation/out/c4_cuda_build_4/bin/astr')
    parser.add_argument('--executable-manifest', type=Path,
                        help='Previously verified rebuild provenance, if different from input baseline')
    parser.add_argument('--timeout', type=float, default=43200)
    run(parser.parse_args())
