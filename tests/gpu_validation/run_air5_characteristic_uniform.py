#!/usr/bin/env python3
"""Whole-solver stationary uniform gate; dynamic GPU admission remains guarded."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from air5_radau_reference import Air5RadauReference
from compare_q_validation_snapshots import read_q_snapshot
from prepare_air5_c4_case import prepare_case, replace_boundary_types
from run_air5_sbli_preflight import ROOT, environment, set_value

TOLERANCE = 2e-10


def run(output, executable, mode, trap_invalid=False, source_mode='vt', use_gpu=False):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    executable = executable.resolve()
    if trap_invalid and use_gpu:
        raise ValueError('host IEEE trap mode is CPU-only')
    input_file = prepare_case(ROOT/'examples/Taylor_Green_Vortex_SI/datin', output,
                              '15,15,7', 2, '1.d-9', 't', 'f', 't' if use_gpu else 'f', 'species-wave')
    set_value(input_file, 'flowtype', 'air5hbl')
    set_value(input_file, 'lihomo,ljhomo,lkhomo', 'f,f,t')
    set_value(input_file, 'lrestar', 'f')
    lines = input_file.read_text().splitlines()
    replace_boundary_types(lines, ('11,free', 50, '41,1500.d0', 51, 1, 1))
    input_file.write_text('\n'.join(lines)+'\n')
    model = Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json')
    fractions = np.array([.767, .233, 0., 0., 0.])
    rho, temperature = .05, 1500.
    pressure = rho * float(fractions @ model.gas_constant) * temperature
    rows = [[y, rho, 0., 0., 0., pressure, temperature, temperature, *fractions]
            for y in (0., .01)]
    (output/'datin/air5_hbl_profile.dat').write_text(
        '# Uniform stationary verification state\n# x_origin=1.0\n' +
        '\n'.join(' '.join(f'{v:.17e}' for v in row) for row in rows)+'\n')
    (output/'datin/air5_hbl_domain.dat').write_text('air5_hbl_domain_v1\n0.02 0.01 0.01\n')
    (output/'validation').mkdir(exist_ok=True)
    env = environment('1,1,1')
    env.update(ASTR_AIR5_TOP_MODE=mode, ASTR_AIR5_TOP_TAU='5.790834463433053e-6',
               ASTR_AIR5_SOURCE_MODE=source_mode, ASTR_AIR5_COMPENSATION='on',
               ASTR_VALIDATION_RHS_PREFIX='validation/air5', ASTR_VALIDATION_RHS_STEP='2',
               ASTR_VALIDATION_RHS_STEP_SECONDARY='0')
    command = [shutil.which('mpirun'), '-np', '1', str(executable), 'run', 'datin/input.air5_c4']
    if trap_invalid:
        command = [shutil.which('mpirun'), '-np', '1', shutil.which('gdb'), '--batch',
                   '-ex', 'break MAIN_', '-ex', 'run', '-ex', 'set language c',
                   '-ex', 'set $mxcsr = ($mxcsr & ~0xbf)',
                   '-ex', 'set $fctrl = ($fctrl & ~1)',
                   '-ex', 'continue', '-ex',
                   "python [gdb.execute(c) for c in ['bt', 'x/12i $pc-24', "
                   "'info registers mxcsr xmm0 xmm1 xmm2']] if gdb.selected_inferior().pid else print('TRAP_RUN_EXITED')",
                   '--args', str(executable),
                   'run', 'datin/input.air5_c4']
    (output/'gate.json').write_text(json.dumps(dict(mode=mode, source_mode=source_mode, use_gpu=use_gpu,
        tolerance=TOLERANCE,
        scope='stationary uniform physical domain, including dynamic top', command=command), indent=2)+'\n')
    with (output/'run.log').open('w') as log:
        result = subprocess.run(command, cwd=output, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=120)
    if trap_invalid:
        text = (output/'run.log').read_text()
        if 'SIGFPE' in text or 'arithmetic exception' in text:
            raise RuntimeError('floating-point trap captured; inspect run.log before further gates')
    result.check_returncode()
    if 'The job is done!' not in (output/'run.log').read_text():
        raise RuntimeError('missing normal completion')
    check_output(output, mode)


def check_output(output, mode):
    output = output.resolve()
    model = Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json')
    rho, temperature = .05, 1500.
    fractions = np.array([.767, .233, 0., 0., 0.])
    ev = model.ev_from_tv(rho*fractions, temperature)
    energy = model.q5_from_state(rho, np.zeros(3), rho*fractions, ev, temperature)
    analytic = np.array([rho, 0., 0., 0., energy, *(rho*fractions), ev])
    snapshots = sorted((output/'validation').glob('*.post_transport*.bin'))
    if not snapshots:
        raise RuntimeError('missing post-transport conservative snapshot')
    reference_files = sorted((output/'validation').glob('*.pre_chemistry*.bin'))
    if not reference_files:
        raise RuntimeError('missing pre-chemistry conservative snapshot')
    def physical(path):
        snapshot = read_q_snapshot(path)
        ni, nj, nk, hm, nq = snapshot.header
        return snapshot.values.reshape((ni+2*hm+1, nj+2*hm+1, nk+2*hm+1, nq), order='F')[
            hm:hm+ni+1, hm:hm+nj+1, hm:hm+nk+1, :]
    maximum = 0.
    for path in [*reference_files, *snapshots]:
        field = physical(path)
        maximum = max(maximum, float(np.max(np.abs(field-analytic)/np.maximum(1., np.abs(analytic)))))
    report = dict(passed=bool(maximum<=TOLERANCE),
                  maximum_scaled_drift=maximum,
                  tolerance=TOLERANCE, samples=len(reference_files)+len(snapshots), mode=mode)
    (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    if not report['passed']:
        raise RuntimeError(report)
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--executable', type=Path, default=ROOT/'build_cpu_probe/bin/astr')
    parser.add_argument('--mode', choices=('prescribed', 'characteristic'), default='characteristic')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--trap-invalid', action='store_true')
    parser.add_argument('--source-mode', choices=('vt', 'frozen'), default='vt')
    parser.add_argument('--use-gpu', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        check_output(args.output, args.mode)
    else:
        run(args.output, args.executable, args.mode, args.trap_invalid, args.source_mode, args.use_gpu)
