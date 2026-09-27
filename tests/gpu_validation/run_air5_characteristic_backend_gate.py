#!/usr/bin/env python3
"""Short frozen pulse at the top: matched RK fields and MPI decomposition gate."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from air5_radau_reference import Air5RadauReference
from check_air5_c5_hbl import _assemble_global, _load_parallel_layout
from check_air5_mach4_error_replay import state_gate
from check_air5_mass_closure_stages import metrics
from run_air5_characteristic_acoustic import prepare, load_monitor
from run_air5_sbli_preflight import ROOT, environment, set_value

TOLERANCE = 2e-10
PHASES = [('post_chemistry', 1), ('pre_rhs', 1), ('post_update', 1),
          ('pre_rhs', 2), ('post_update', 2), ('pre_rhs', 3), ('post_update', 3),
          ('post_transport', 1), ('post_chemistry', 2)]


def fields(case, steps=(0, 2)):
    layouts = _load_parallel_layout(case/'datin/parallel.info')
    return {f'{step}:{label}:{stage}': _assemble_global(
        case/'validation/air5', label, stage, layouts, step)
        for step in steps for label, stage in PHASES}


def compare(reference, candidate):
    if not reference or reference.keys() != candidate.keys():
        raise ValueError('phase set mismatch')
    result = {}
    for name in reference:
        a, b = reference[name], candidate[name]
        if a.shape != b.shape:
            raise ValueError(f'{name}: shape mismatch')
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise ValueError(f'{name}: nonfinite state')
        scaled = abs(a-b)/np.maximum(1., abs(a))
        result[name] = dict(max_scaled=float(scaled.max()),
                           component_max_abs=abs(a-b).max(axis=(0, 1, 2)).tolist(),
                           top_max_scaled=float(scaled[:, -1].max()))
        if scaled.max() > TOLERANCE:
            raise ValueError(f'{name}: phase mismatch {result[name]}')
    return result


def execute(case, executable, gpu, topology, sanitizer=False, source_mode='frozen',
            viscous=False, temperature=1500., tv=1500., dt=None, face_probe=False,
            diffusion_limiter='full_state', convection_limiter='full_state'):
    meta = prepare(case, executable, ny=128, nz=16, pulse_y=.0096, short_updates=3, use_gpu=gpu,
                   temperature0=temperature, tv0=tv)
    if dt is not None:
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be positive and finite')
        set_value(case/'datin/controller', 'deltat', f'{dt:.17e}')
        meta['dt'] = dt
    set_value(case/'datin/input.air5_c4',
              'nondimen,diffterm,lfilter,lreadgrid,lfftz,limmbou,ltimrpt,lcomb',
              'f,'+('t' if viscous else 'f')+',f,f,f,f,t,t,'+('t' if gpu else 'f'))
    env = environment(topology)
    env.update(meta['environment'], ASTR_AIR5_TOP_GPU_VALIDATION='on')
    env['ASTR_AIR5_SOURCE_MODE'] = source_mode
    env['ASTR_AIR5_DIFFUSION_LIMITER'] = diffusion_limiter
    env['ASTR_AIR5_CONVECTION_LIMITER'] = convection_limiter
    if face_probe:
        env['ASTR_AIR5_SYMMETRIC_FACE_PROBE'] = '1'
    if source_mode != 'frozen':
        env.pop('ASTR_AIR5_ACOUSTIC_PROBE', None)
    command = [shutil.which('mpirun'), '--oversubscribe', '-np',
               str(int(np.prod([int(x) for x in topology.split(',')])))]
    if sanitizer:
        command += [shutil.which('compute-sanitizer'), '--tool', 'memcheck',
                    '--error-exitcode', '99', '--log-file', 'validation/memcheck.%p.log']
    command += [str(executable), 'run', 'datin/input.air5_c4']
    meta.update(topology=topology, use_gpu=gpu, source_mode=source_mode, viscous=viscous,
                diffusion_limiter=diffusion_limiter,
                convection_limiter=convection_limiter,
                gate='short backend equivalence, not acoustic acceptance',
                tolerance=TOLERANCE, environment={k:v for k,v in env.items() if k.startswith('ASTR_')})
    (case/'gate.json').write_text(json.dumps(meta, indent=2)+'\n')
    with (case/'run.log').open('w') as log:
        subprocess.run(['timeout', '--kill-after=10s', '900s', *command], cwd=case, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    log = (case/'run.log').read_text()
    if 'The job is done!' not in log or 'ieee_invalid' in log.lower():
        raise ValueError('solver did not complete cleanly')
    if sanitizer:
        logs = list((case/'validation').glob('memcheck.*.log'))
        if not logs or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in logs):
            raise ValueError('memcheck failed')
    state_gate(case, Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json'))
    data = fields(case)
    if source_mode == 'frozen':
        for index in (1, 2, 3):
            load_monitor(case, meta, index)
    for key, array in data.items():
        row = metrics(array, (0, 0, 0))
        if row['transport_sum_violations'] or row['min_species_density'] < 0:
            raise ValueError(f'{key}: strict composition failure')
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--topology', default='1,1,1')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--sanitizer', action='store_true')
    parser.add_argument('--source-mode', choices=('frozen', 'vt', 'coupled'), default='frozen')
    parser.add_argument('--viscous', action='store_true')
    parser.add_argument('--temperature', type=float, default=1500.)
    parser.add_argument('--tv', type=float, default=1500.)
    parser.add_argument('--dt', type=float, help='Override the prepared timestep in seconds')
    parser.add_argument('--face-probe', action='store_true')
    parser.add_argument('--diffusion-limiter', choices=('full_state', 'layered'), default='full_state')
    parser.add_argument('--convection-limiter', choices=('full_state', 'symmetric_species'), default='full_state')
    parser.add_argument('--executable', type=Path, default=ROOT/'build_gpu_probe/bin/astr')
    args = parser.parse_args()
    data = execute(args.output.resolve(), args.executable.resolve(), args.gpu, args.topology, args.sanitizer,
                   args.source_mode, args.viscous, args.temperature, args.tv, args.dt, args.face_probe,
                   args.diffusion_limiter, args.convection_limiter)
    report = dict(passed=True, source_mode=args.source_mode, viscous=args.viscous,
                  scope='short backend equivalence; spanwise homogeneous, not physical acceptance')
    source_delta = data['2:post_chemistry:2']-data['2:post_transport:1']
    report['second_half_top_component_max_abs'] = abs(source_delta[1:-1, -1, 1:-1]).max(axis=(0,1)).tolist()
    if args.source_mode != 'frozen' and report['second_half_top_component_max_abs'][10] == 0.:
        raise ValueError('source gate did not exercise vibrational evolution at the dynamic top')
    if args.reference:
        report['comparison'] = compare(fields(args.reference), data)
        old = json.loads((args.reference/'gate.json').read_text())
        new = json.loads((args.output/'gate.json').read_text())
        for key in ('temperature0', 'tv0', 'viscous', 'source_mode', 'dt'):
            if old.get(key) != new.get(key):
                raise ValueError(f'incompatible reference: {key}')
        if old.get('diffusion_limiter', 'full_state') != new['diffusion_limiter']:
            raise ValueError('incompatible reference: diffusion_limiter')
        if old.get('convection_limiter', 'full_state') != new['convection_limiter']:
            raise ValueError('incompatible reference: convection_limiter')
        monitor_error = 0.
        for index in ((1, 2, 3) if args.source_mode == 'frozen' else ()):
            a = load_monitor(args.reference, old, index)[:, :6]
            b = load_monitor(args.output, new, index)[:, :6]
            monitor_error = max(monitor_error, float((abs(a-b)/np.maximum(1.,abs(a))).max()))
        report['monitor_max_scaled'] = monitor_error if args.source_mode == 'frozen' else None
        if monitor_error > TOLERANCE:
            raise ValueError(f'monitor phase mismatch: {monitor_error}')
    (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
