#!/usr/bin/env python3
"""Bounded GPU precursor windows; no automatic SBLI production admission."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from air5_radau_reference import Air5RadauReference
from check_air5_mach4_error_replay import state_gate
from check_air5_mass_closure_stages import metrics
from check_air5_compensation_checkpoint import read as read_checkpoint
from run_air5_characteristic_backend_gate import fields
from run_air5_sbli_preflight import ROOT, environment, set_value


def run(template, output, executable, updates, dt, sanitizer=False, restart_from=None, monitor_stride=0,
        topology='2,1,1', restart_qualification=None):
    if updates < 2 or not 0 < dt <= 2.5e-10:
        raise ValueError('require >=2 updates and 0 < dt <= 0.25 ns')
    template, output, executable = [p.resolve() for p in (template, output, executable)]
    if output.exists():
        raise FileExistsError(output)
    settings = json.loads((template/'environment.json').read_text())
    if topology not in ('2,1,1', '1,2,1', '1,1,2'):
        raise ValueError('this driver requires a two-rank slab')
    settings['ASTR_FORCE_MPI_TOPOLOGY'] = topology
    if settings['ASTR_AIR5_SOURCE_MODE'] != 'coupled':
        raise ValueError('precursor requires coupled chemistry')
    start_step, start_time = 0, 0.
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    if restart_from is not None:
        restart_from = restart_from.resolve()
        previous = json.loads((restart_from/'launch.json').read_text())
        if previous['executable_sha256'] != digest:
            if restart_qualification is None:
                raise ValueError('restart executable differs from the validated baseline')
            gate = json.loads(restart_qualification.read_text())
            expected = dict(passed=True, source_executable_sha256=previous['executable_sha256'],
                candidate_executable_sha256=digest,
                checkpoint_sha256=hashlib.sha256((restart_from/'outdat/flowfield.h5').read_bytes()).hexdigest(),
                dt=dt, topology=topology)
            if any(gate.get(k) != v for k,v in expected.items()):
                raise ValueError('restart qualification does not match executable/checkpoint/configuration')
        for key, value in settings.items():
            if previous['environment'].get(key) != value:
                raise ValueError(f'restart configuration differs: {key}')
        _, _, (start_step, start_time), _ = read_checkpoint(restart_from/'outdat/flowfield.h5')
    last_step = start_step + updates - 1
    shutil.copytree((restart_from or template)/'datin', output/'datin')
    if restart_from is not None:
        (output/'outdat').mkdir()
        for name in ('flowfield.h5', 'auxiliary.txt'):
            shutil.copy2(restart_from/'outdat'/name, output/'outdat'/name)
        set_value(output/'datin/input.air5_c4', 'lrestar', 't')
    (output/'validation').mkdir()
    set_value(output/'datin/controller', 'deltat', f'{dt:.17e}')
    set_value(output/'datin/controller', 'maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg',
              f'{last_step},{min(updates-1,1000)},1000000,1000000,10,1000000')
    env = environment('2,1,1')
    env.update(settings, ASTR_AIR5_TOP_GPU_VALIDATION='on',
        ASTR_VALIDATION_RHS_PREFIX='validation/air5',
        ASTR_VALIDATION_RHS_STEP=str(last_step), ASTR_VALIDATION_RHS_STEP_SECONDARY=str(start_step))
    if monitor_stride:
        if monitor_stride<1:
            raise ValueError('monitor stride must be positive')
        env['ASTR_AIR5_FLOW_MONITOR_STRIDE']=str(monitor_stride)
    if restart_from is not None:
        env.update(ASTR_AIR5_TOP_RESTART_VALIDATION='on', ASTR_AIR5_COMPENSATION_RESTART='restore')
    # One flow-through window can take several hours on the local two GPUs.
    command = ['timeout', '--kill-after=10s', '86400s', shutil.which('mpirun'),
               '--oversubscribe', '-np', '2']
    if sanitizer:
        command += [shutil.which('compute-sanitizer'), '--tool', 'memcheck',
                    '--error-exitcode', '99', '--log-file', 'validation/memcheck.%p.log']
    command += [str(executable), 'run', 'datin/input.air5_c4']
    report = dict(passed=False, updates=updates, dt=dt, end_time=start_time+updates*dt,
        start_step=start_step, start_time=start_time, last_step=last_step,
        restart_from=str(restart_from) if restart_from else None,
        checkpoint_sha256=hashlib.sha256((restart_from/'outdat/flowfield.h5').read_bytes()).hexdigest()
            if restart_from else None,
        restart_qualification=str(restart_qualification) if restart_qualification else None,
        scope='bounded precursor only; no developed inlet or SBLI admission',
        executable_sha256=digest,
        command=command, environment={k:v for k,v in env.items() if k.startswith('ASTR_')})
    (output/'launch.json').write_text(json.dumps(report, indent=2)+'\n')
    with (output/'run.log').open('w') as log:
        subprocess.run(command, cwd=output, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    log = (output/'run.log').read_text()
    if 'The job is done!' not in log or 'ieee_invalid' in log.lower():
        raise ValueError('unclean completion')
    if sanitizer:
        logs = list((output/'validation').glob('memcheck.*.log'))
        if len(logs) != 2 or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in logs):
            raise ValueError('two zero-error memcheck reports required')
        report['memcheck_errors'] = 0
    report['state'] = state_gate(output, Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json'))
    for phase, array in fields(output, (start_step, last_step)).items():
        row = metrics(array, (0, 0, 0))
        if row['transport_sum_violations'] or row['min_species_density'] < 0:
            raise ValueError(f'{phase}: strict composition failure')
    report['passed'] = True
    (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['state']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--executable', type=Path, default=ROOT/'build_gpu_probe/bin/astr')
    parser.add_argument('--updates', type=int, default=200)
    parser.add_argument('--dt', type=float, default=2.5e-10)
    parser.add_argument('--sanitizer', action='store_true')
    parser.add_argument('--restart-from', type=Path)
    parser.add_argument('--monitor-stride', type=int, default=0)
    parser.add_argument('--topology', choices=('2,1,1','1,2,1','1,1,2'), default='2,1,1')
    parser.add_argument('--restart-qualification', type=Path)
    args = parser.parse_args()
    run(args.template, args.output, args.executable, args.updates, args.dt, args.sanitizer,
        args.restart_from, args.monitor_stride, args.topology, args.restart_qualification)
