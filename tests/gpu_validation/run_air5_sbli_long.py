#!/usr/bin/env python3
"""Supervised two-GPU SBLI extension; all flow integration remains in ASTR."""
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
from prepare_air5_sbli_case import prepare
from prepare_air5_mach4_case import prepare as prepare_mach4, rescale_inlet_profile
from run_air5_sbli_preflight import ROOT, environment, set_value

FIELDS = ('ro','u1','u2','u3','p','t','tv','sp001','sp002','sp003','sp004','sp005')


def state_metrics(fields, thermo, uref, pref):
    if any(not np.isfinite(fields[k]).all() for k in FIELDS):
        raise ValueError('non-finite checkpoint state')
    rho = fields['ro']
    ys = np.stack([fields[f'sp{i:03d}'] for i in range(1, 6)], axis=-1)
    if rho.min() <= 0 or ys.min() < 0:
        raise ValueError('nonpositive density or negative accepted species')
    for key, bounds in (('t', thermo.temperature_bounds), ('tv', thermo.temperature_bounds),
                        ('p', thermo.pressure_bounds)):
        if fields[key].min() < bounds[0] or fields[key].max() > bounds[1]:
            raise ValueError(f'{key} outside the fixed mechanism domain')
    closure = float(abs(ys.sum(axis=-1)-1).max())
    if closure > 2e-12:
        raise ValueError(f'species closure failed: {closure}')
    # HDF uses z,y,x; the repeated periodic z endpoint is not an independent sample.
    p = fields['p'][:-1]
    pprime = p-p.mean(axis=0, keepdims=True)
    return dict(min_density=float(rho.min()), min_species_density=float((ys*rho[..., None]).min()),
        mass_fraction_closure=closure, min_temperature=float(fields['t'].min()),
        max_temperature=float(fields['t'].max()), min_tv=float(fields['tv'].min()),
        max_tv=float(fields['tv'].max()), min_pressure=float(fields['p'].min()),
        max_pressure=float(fields['p'].max()), wall_pressure_max=float(fields['p'][:,0,:].max()),
        max_spanwise_velocity_over_uinf=float(abs(fields['u3']).max()/uref),
        spanwise_pressure_rms_over_pinf=float(np.sqrt(np.mean(pprime*pprime))/pref))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_time(start_step, start_time, step, dt):
    if step < start_step or dt <= 0 or not np.isfinite((start_time, dt)).all():
        raise ValueError('invalid restart clock')
    return start_time + (step-start_step)*dt


def precursor_profiles(fields, domain):
    """Spanwise means on the uniform Cartesian precursor mesh; no flow integration."""
    averaged = {name: fields[name][:-1].mean(axis=0) for name in FIELDS}
    mass_flux = (fields['ro'][:-1]*fields['u1'][:-1]).mean(axis=0)
    ny, nx = averaged['ro'].shape
    y = np.linspace(0., domain[1], ny)
    samples = []
    for fraction in (.25, .5, .75):
        ix = round(fraction*(nx-1))
        edge_flux = mass_flux[-1,ix]
        if edge_flux <= 0:
            raise ValueError('invalid precursor edge mass flux')
        samples.append(dict(ix=ix, x=ix*domain[0]/(nx-1),
            displacement_thickness=float(np.trapezoid(1.-mass_flux[:,ix]/edge_flux,y)),
            wall_pressure=float(averaged['p'][0,ix]),
            fields={name: averaged[name][:,ix].tolist() for name in FIELDS}))
    return dict(y=y.tolist(), stations=samples)


def run(args):
    inlet_scale = getattr(args,'inlet_thickness_scale',1.)
    if not np.isfinite(inlet_scale) or inlet_scale <= 0:
        raise ValueError('invalid inlet thickness scale')
    if inlet_scale != 1. and (getattr(args,'case',None)!='mach4-precursor' or
                             getattr(args,'baseline',None) is None):
        raise ValueError('inlet-only diagnostics require a Mach4 checkpoint baseline')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    case = out/'gpu'
    start_step, start_time = 0, 0.
    precursor = getattr(args, 'case', 'legacy-sbli') == 'mach4-precursor'
    grid = getattr(args, 'grid', None)
    baseline = getattr(args, 'baseline', None)
    metadata_file = 'mach4_case_metadata.json' if precursor else 'incident_shock_metadata.json'
    if baseline is not None:
        baseline = baseline.resolve()
        meta = json.loads((baseline/metadata_file).read_text())
        if grid is not None and grid != meta['grid']:
            raise ValueError('restart grid differs from checkpoint case')
        grid = meta['grid']
        if digest(baseline.parent/'astr') != digest(args.executable.resolve()):
            raise ValueError('restart requires the archived baseline executable')
        if digest(baseline.parent/'mechanism.json') != digest(ROOT/'chemMech/air5_kimjo12.json'):
            raise ValueError('restart mechanism differs from baseline')
        shutil.copytree(baseline/'datin', case/'datin')
        (case/'outdat').mkdir()
        (case/'validation').mkdir()
        for name in ('flowfield.h5','auxiliary.txt'):
            shutil.copy2(baseline/'outdat'/name, case/'outdat'/name)
        shutil.copy2(baseline/metadata_file, case/metadata_file)
        with h5py.File(case/'outdat/flowfield.h5') as stream:
            start_step = int(stream['nstep'][()].item())
            start_time = float(stream['time'][()].item())
        if start_step < 0 or not np.isfinite(start_time) or start_time < 0:
            raise ValueError('invalid checkpoint origin')
        set_value(case/'datin/input.air5_c4', 'lrestar', 't')
        set_value(case/'datin/controller', 'deltat', f'{args.dt:.17e}')
    else:
        grid = grid or '31,31,7'
        preparer = prepare_mach4 if precursor else prepare
        preparer(case, grid, args.updates-1, f'{args.dt:.17e}', 't')
    maximum = start_step + args.updates-1
    if inlet_scale != 1.:
        rescale_inlet_profile(case/'datin/air5_hbl_profile.dat',inlet_scale)
    target_time = checkpoint_time(start_step,start_time,maximum+1,args.dt)
    set_value(case/'datin/controller', 'maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg',
              f'{maximum},{args.checkpoint},1000000,1000000,{args.checkpoint},1000000')
    shutil.copy2(args.executable.resolve(), out/'astr')
    shutil.copy2(Path(__file__).resolve(), out/'driver_at_launch.py')
    shutil.copy2(ROOT/'chemMech/air5_kimjo12.json', out/'mechanism.json')
    thermo = Air5RadauReference(out/'mechanism.json')
    meta = json.loads((case/metadata_file).read_text())
    uref = meta['upstream_q'][1]/meta['upstream_q'][0]
    env = environment('2,1,1')
    env.update(ASTR_VALIDATION_RHS_PREFIX='validation/air5',
               ASTR_VALIDATION_RHS_STEP=str(maximum))
    command = [shutil.which('mpirun'), '--oversubscribe', '-np', '2', str(out/'astr'),
               'run', 'datin/input.air5_c4']
    contract = dict(scope='long-time-engineering-diagnostic-not-physical-acceptance',
        updates=args.updates, dt=args.dt, target_time=target_time,
        start_step=start_step, start_time=start_time,
        baseline=str(baseline) if baseline else None,
        inlet_thickness_scale=inlet_scale,
        target_flow_through_times=target_time*uref/meta['domain'][0],
        checkpoint_frequency=args.checkpoint, topology=[2,1,1], grid=list(map(int,grid.split(','))),
        case='mach4-precursor' if precursor else 'legacy-sbli',
        driver_sha256=digest(__file__),
        transverse_error_policy='record-only; not accepted as physical validation',
        hard_gates=['solver failure','negative species','nonfinite state','model domain',
                    'mass closure','CFL >= 1'], command=command,
        environment={k:v for k,v in env.items() if k.startswith(('ASTR_','OMPI_','UCX_','CUDA_'))},
        files={str(p.relative_to(out)):digest(p) for p in
               (out/'astr',out/'mechanism.json',*sorted((case/'datin').glob('*')),
                *sorted((case/'outdat').glob('*'))) if p.is_file()})
    (out/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    start = time.monotonic()
    previous = -1
    proc = None
    latest = {}

    def status(state, **details):
        value = dict(status=state, wall_seconds=time.monotonic()-start, **details)
        temporary = out/'status.tmp'
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
        temporary.replace(out/'status.json')

    def inspect_checkpoint():
        nonlocal previous, latest
        path = case/'outdat/flowfield.h5'
        try:
            before = path.stat()
            if time.time()-before.st_mtime < 1:
                return
            data = path.read_bytes()
            if path.stat().st_mtime_ns != before.st_mtime_ns:
                return
            with h5py.File(io.BytesIO(data), 'r') as stream:
                if any(key not in stream for key in (*FIELDS,'nstep','time')):
                    return
                step = int(stream['nstep'][()].item())
                if step <= previous:
                    return
                elapsed = float(stream['time'][()].item())
                fields = {key:stream[key][()] for key in FIELDS}
        except OSError:
            return
        if not np.isclose(elapsed, checkpoint_time(start_step,start_time,step,args.dt),
                          rtol=1e-9, atol=1e-18):
            raise ValueError('checkpoint step/time mismatch')
        latest = dict(step=step, time=elapsed, phase='complete-RK-checkpoint',
            wall_seconds=time.monotonic()-start,
            **state_metrics(fields, thermo, uref, meta['pressure'][0]))
        if precursor and latest['min_temperature'] < meta['minimum_transport_temperature']:
            raise ValueError('precursor left the selected transport-fit temperature range')
        if precursor:
            profiles = precursor_profiles(fields,meta['domain'])
            profiles.update(step=step,time=elapsed)
            (out/f'profiles_step{step:06d}.json').write_text(json.dumps(profiles,indent=2)+'\n')
            latest['profile_stations'] = [{k:v for k,v in s.items() if k!='fields'}
                                           for s in profiles['stations']]
        if precursor or previous < 0 or step % 1000 == 0:
            (out/f'checkpoint_step{step:06d}.h5').write_bytes(data)
        with (out/'statistics.jsonl').open('a') as stream:
            stream.write(json.dumps(latest, allow_nan=False)+'\n')
        previous = step
        status('running', pid=proc.pid, latest=latest)
        print(json.dumps(latest, allow_nan=False), flush=True)

    def inspect_log():
        text = (case/'run.log').read_text(errors='replace')
        values = [float(s) for s in re.findall(r'current CFL:\s*(\S+)', text)]
        if values and (not np.isfinite(values).all() or max(values) >= 1):
            raise ValueError(f'CFL safety gate failed: {values[-1]}')
        return text, max(values) if values else None

    def interrupt(signum, frame):
        raise InterruptedError(f'service received signal {signum}')

    signal.signal(signal.SIGTERM, interrupt)
    try:
        with (case/'run.log').open('w') as log:
            proc = subprocess.Popen(command, cwd=case, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            status('running', pid=proc.pid, target_time=contract['target_time'])
            while proc.poll() is None:
                inspect_checkpoint()
                inspect_log()
                if time.monotonic()-start > args.timeout:
                    raise TimeoutError('long-run wall-time limit exceeded')
                time.sleep(2)
            if proc.returncode != 0:
                raise RuntimeError(f'ASTR returned {proc.returncode}; inspect gpu/run.log')
        time.sleep(1.1)
        inspect_checkpoint()
        text, cfl = inspect_log()
        if 'The job is done!' not in text or cfl is None:
            raise ValueError('solver completion or CFL record missing')
        status('completed-pending-final-field-review', latest=latest, max_cfl=cfl,
               target_time=contract['target_time'])
    except BaseException as error:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        status('stopped', reason=str(error), latest=latest)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--executable', type=Path,
        default=ROOT/'tests/gpu_validation/out/c4_cuda_build_4/bin/astr')
    parser.add_argument('--updates', type=int, default=10001)
    parser.add_argument('--case', choices=('legacy-sbli','mach4-precursor'), default='legacy-sbli')
    parser.add_argument('--grid', help='Grid upper bounds; fresh default 31,31,7; restart inherits')
    parser.add_argument('--baseline',type=Path, help='Archived case directory to restart, never modified')
    parser.add_argument('--inlet-thickness-scale',type=float,default=1.,
                        help='Mach4 restart diagnostic only; does not alter the interior checkpoint')
    parser.add_argument('--dt', type=float, default=5e-10)
    parser.add_argument('--checkpoint', type=int, default=50)
    parser.add_argument('--timeout', type=float, default=21600.)
    args = parser.parse_args()
    if args.updates < 2 or args.checkpoint < 1 or not np.isfinite(args.dt) or args.dt <= 0:
        parser.error('invalid update count, checkpoint frequency or time step')
    if not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('timeout must be finite and positive')
    run(args)
