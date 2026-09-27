#!/usr/bin/env python3
"""ASTR frozen-acoustic gate; Python only prepares and measures the flow."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess

import numpy as np

from air5_radau_reference import Air5RadauReference
from check_air5_c5_hbl import _active_array
from check_air5_mach4_error_replay import state_gate
from check_air5_mass_closure_stages import metrics
from prepare_air5_c4_case import prepare_case, replace_boundary_types
from run_air5_sbli_preflight import ROOT, environment, set_value

RHO, TEMP, EPSILON = .05, 1500., 1e-5
LX, LY, LZ = .08, .01, .002
Y0, SIGMA, YP = .0075, .0003125, .00875
FRACTIONS = np.array([.767, .233, 0., 0., 0.])


def prepare(output, executable, ny=256, dt=5e-9, tau_factor=1., mode='characteristic', extended=False,
            *, nz=8, pulse_y=Y0, short_updates=None, use_gpu=False,
            temperature0=TEMP, tv0=TEMP):
    output, executable = output.resolve(), executable.resolve()
    if output.exists():
        raise FileExistsError(output)
    if ny not in (128, 256) or dt not in (5e-9, 2.5e-9) or tau_factor not in (.5, 1., 2.):
        raise ValueError('configuration outside the approved acoustic matrix')
    model = Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json')
    gas = float(FRACTIONS @ model.gas_constant)
    gamma = 1 + gas / float(FRACTIONS @ model.cv_tr)
    pressure = RHO*gas*temperature0
    sound = np.sqrt(gamma*gas*temperature0)
    ly = LY*(2 if extended else 1)
    nj = ny*(2 if extended else 1)
    updates = round(7.5e-6/dt) if short_updates is None else short_updates
    input_file = prepare_case(ROOT/'examples/Taylor_Green_Vortex_SI/datin', output,
        f'16,{nj},{nz}', updates-1, f'{dt:.17e}', 'f', 'f', 't' if use_gpu else 'f', 'species-wave', list_frequency=50)
    set_value(input_file, 'flowtype', 'air5hbl')
    set_value(input_file, 'lihomo,ljhomo,lkhomo', 'f,f,t')
    set_value(input_file, 'lrestar', 'f')
    lines = input_file.read_text().splitlines()
    replace_boundary_types(lines, ('11,free', 50, f'41,{temperature0:.17e}', 51, 1, 1))
    input_file.write_text('\n'.join(lines)+'\n')
    (output/'datin/air5_hbl_domain.dat').write_text(f'air5_hbl_domain_v1\n{LX} {ly} {LZ}\n')
    def vibrational_temperature(y):
        return tv0+(temperature0-tv0)*max(0., 1.-y/(.1*ly))
    profile_y = (0., ly) if temperature0 == tv0 else (0., .1*ly, ly)
    rows = [[y, RHO, 0., 0., 0., pressure, temperature0, vibrational_temperature(y), *FRACTIONS]
            for y in profile_y]
    (output/'datin/air5_hbl_profile.dat').write_text('# Frozen acoustic reservoir\n# x_origin=1.0\n'+
        '\n'.join(' '.join(f'{v:.17e}' for v in row) for row in rows)+'\n')
    xs, ys = np.linspace(0., LX, 17), np.linspace(0., ly, nj+1)
    field_lines = ['# ASTR_AIR5_HBL_INITIAL_FIELD_V1', '# frozen acoustic simple wave', f'17 {nj+1}']
    for x in xs:
        edge_distance = min(x, LX-x)
        envelope = 1. if edge_distance >= .02 else .5*(1.-np.cos(np.pi*edge_distance/.02))
        for y in ys:
            ratio = 1 + EPSILON*envelope*np.exp(-.5*((y-pulse_y)/SIGMA)**2)
            rho = RHO*ratio**(1/gamma)
            v = 2*sound/(gamma-1)*(ratio**((gamma-1)/(2*gamma))-1)
            momentum = rho*np.array([0., v, 0.])
            partial = rho*FRACTIONS
            ev = model.ev_from_tv(partial, vibrational_temperature(y))
            temperature = pressure*ratio/(rho*gas)
            energy = model.q5_from_state(rho, momentum, partial, ev, temperature)
            row = [x, y, rho, *momentum, energy, *partial, ev]
            field_lines.append(' '.join(f'{value:.17e}' for value in row))
    (output/'datin/air5_hbl_initial_field.dat').write_text('\n'.join(field_lines)+'\n')
    jp = round(YP/(ly/nj))
    (output/'datin/monitor.dat').write_text('# i j k\n'+
        '\n'.join(f'{i} {jp} 4' for i in (8, 7, 9))+'\n')
    (output/'validation').mkdir()
    env_values = dict(ASTR_AIR5_SOURCE_MODE='frozen', ASTR_AIR5_COMPENSATION='on',
        ASTR_AIR5_TOP_MODE=mode, ASTR_AIR5_TOP_TAU=f'{tau_factor*LY/sound:.17e}',
        ASTR_VALIDATION_RHS_PREFIX='validation/air5', ASTR_VALIDATION_RHS_STEP=str(updates-1),
        ASTR_VALIDATION_RHS_STEP_SECONDARY='0')
    if use_gpu:
        env_values['ASTR_AIR5_TOP_GPU_VALIDATION'] = 'on'
        env_values['ASTR_AIR5_ACOUSTIC_PROBE'] = 'on'
    metadata = dict(ny=ny, grid_intervals=[16, nj, nz], domain=[LX, ly, LZ], dt=dt,
        pulse_y=pulse_y, use_gpu=use_gpu, temperature0=temperature0, tv0=tv0,
        updates=updates, mode=mode, extended=extended, tau_factor=tau_factor,
        tau=tau_factor*LY/sound, rho=RHO, pressure=pressure, sound_speed=float(sound), gamma=gamma,
        epsilon=EPSILON, probe_xyz=[.04, jp*ly/nj, .001], environment=env_values,
        incident_center=(YP-Y0)/sound, reflected_center=(2*LY-Y0-YP)/sound,
        window_halfwidth=3*SIGMA/sound, executable=str(executable),
        executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        sampling_phase='RK-first after frozen half-step and boundary preparation; interior step-start state')
    (output/'gate.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def read_monitor(path, expected_count, dt):
    # NVHPC default INTEGER*4, ten REAL*8 values, direct record length 88.
    dtype = np.dtype({'names': ['step', 'data'], 'formats': ['<i4', ('<f8', 10)],
                      'offsets': [0, 4], 'itemsize': 88})
    if path.stat().st_size != expected_count*dtype.itemsize:
        raise ValueError(f'wrong monitor length: {path}')
    data = np.fromfile(path, dtype=dtype)
    if not np.array_equal(data['step'], np.arange(expected_count)):
        raise ValueError('unexpected monitor step indices or record format')
    values = data['data']
    if not np.all(np.isfinite(values[:, :6])):
        raise ValueError('nonfinite monitor state')
    if not np.allclose(values[:, 0], np.arange(expected_count)*dt, atol=1e-17, rtol=0):
        raise ValueError('monitor phase/time mismatch')
    return values


def read_acoustic_probe(path, expected_count, dt):
    data = np.loadtxt(path, ndmin=2)
    if data.shape != (expected_count, 7) or not np.all(np.isfinite(data)):
        raise ValueError('wrong GPU probe shape or nonfinite data')
    if not np.array_equal(data[:, 0], np.arange(expected_count)):
        raise ValueError('unexpected GPU probe step indices')
    if not np.allclose(data[:, 1], np.arange(expected_count)*dt, atol=1e-17, rtol=0):
        raise ValueError('GPU probe phase/time mismatch')
    return data[:, 1:]


def load_monitor(output, meta, index):
    if meta.get('use_gpu', False):
        return read_acoustic_probe(output/f'monitor/acoustic{index:04d}.dat', meta['updates'], meta['dt'])
    return read_monitor(output/f'monitor/monitor{index:04d}.dat', meta['updates'], meta['dt'])


def analyze(output):
    meta = json.loads((output/'gate.json').read_text())
    model = Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json')
    state_report = state_gate(output, model)
    closure = 0.
    for label in ('pre_chemistry', 'post_chemistry', 'pre_rhs', 'post_update', 'post_transport'):
        files = list((output/'validation').glob(f'air5.{label}.*.bin'))
        if not files:
            raise ValueError(f'missing {label} states')
        for path in files:
            row = metrics(_active_array(path), (0, 0, 0))
            if row['transport_sum_violations'] or row['min_species_density'] < 0:
                raise ValueError(f'strict composition gate failed: {path}')
            closure = max(closure, abs(row['max_sum_error']))
    data = load_monitor(output, meta, 1)
    time, velocity, pressure = data[:, 0], data[:, 2], data[:, 5]
    plus = pressure-meta['pressure'] + meta['rho']*meta['sound_speed']*velocity
    minus = pressure-meta['pressure'] - meta['rho']*meta['sound_speed']*velocity
    incident = abs(time-meta['incident_center']) <= meta['window_halfwidth']
    reflected = abs(time-meta['reflected_center']) <= meta['window_halfwidth']
    integrate = getattr(np, 'trapezoid', np.trapz)
    peak = float(np.max(abs(plus[incident])))
    integral = float(integrate(plus[incident]**2, time[incident]))
    if peak <= 0 or integral <= 0:
        raise ValueError('no incident acoustic signal')
    rpeak = float(np.max(abs(minus[reflected]))/peak)
    rl2 = float(np.sqrt(integrate(minus[reflected]**2, time[reflected])/integral))
    lateral = 0.
    for index in (2, 3):
        adjacent = load_monitor(output, meta, index)
        ap = adjacent[:, 5]-meta['pressure']+meta['rho']*meta['sound_speed']*adjacent[:, 2]
        lateral = max(lateral, float(np.max(abs(ap-plus))/peak))
    threshold = .05 if meta['tau_factor'] == 1. else .10
    apply_gate = meta['mode'] == 'characteristic' and not meta['extended']
    report = dict(R_peak=rpeak, R_L2=rl2, incident_peak=peak,
        incident_L2=float(np.sqrt(integral)), adjacent_column_scaled_difference=lateral,
        sampled_state_gate=state_report, maximum_sequential_closure=closure,
        reflection_gate_passed=bool(rpeak<=threshold and rl2<=threshold) if apply_gate else None,
        scope='single-run reflection only; reference, sensitivity, refinement and physical-state gates remain separate')
    (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    if apply_gate and not report['reflection_gate_passed']:
        raise ValueError(f'acoustic reflection gate failed: {report}')
    return report


def run(output, metadata):
    if (output/'run.log').exists():
        raise FileExistsError('refuse to overwrite an existing acoustic run')
    if hashlib.sha256(Path(metadata['executable']).read_bytes()).hexdigest() != metadata['executable_sha256']:
        raise ValueError('prepared executable has changed')
    env = environment('1,1,1')
    env.update(metadata['environment'])
    command = [shutil.which('mpirun'), '-np', '1', metadata['executable'], 'run', 'datin/input.air5_c4']
    with (output/'run.log').open('w') as log:
        with subprocess.Popen(command, cwd=output, env=env, stdout=log,
                              stderr=subprocess.STDOUT, start_new_session=True) as process:
            try:
                code = process.wait(timeout=4*3600)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
    if code:
        raise subprocess.CalledProcessError(code, command)
    log = (output/'run.log').read_text()
    if 'The job is done!' not in log or 'ieee_invalid' in log.lower():
        raise RuntimeError('missing clean completion; inspect run.log')
    return analyze(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--executable', type=Path, default=ROOT/'build_cpu_probe/bin/astr')
    parser.add_argument('--ny', type=int, choices=(128, 256), default=256)
    parser.add_argument('--dt', type=float, choices=(5e-9, 2.5e-9), default=5e-9)
    parser.add_argument('--tau-factor', type=float, choices=(.5, 1., 2.), default=1.)
    parser.add_argument('--mode', choices=('prescribed', 'characteristic'), default='characteristic')
    parser.add_argument('--extended', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--run-prepared', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.run_prepared:
        print(json.dumps(run(output, json.loads((output/'gate.json').read_text())), indent=2))
    elif args.check_only:
        print(json.dumps(analyze(output), indent=2))
    else:
        meta = prepare(output, args.executable, args.ny, args.dt, args.tau_factor, args.mode, args.extended,
                       use_gpu=args.gpu)
        print(json.dumps(meta if args.prepare_only else run(output, meta), indent=2))
