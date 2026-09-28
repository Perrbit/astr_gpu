#!/usr/bin/env python3
"""Qualify a monitor-only binary transition using archived phases and a live checkpoint."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from air5_radau_reference import Air5RadauReference
from check_air5_mach4_error_replay import state_gate
from check_air5_compensation_checkpoint import read, compare as compare_checkpoint
from check_air5_mass_closure_stages import metrics
from run_air5_characteristic_backend_gate import fields, compare
from run_air5_sbli_preflight import ROOT, environment, set_value


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(parent, case, executable, monitor):
    prior = json.loads((parent/'launch.json').read_text())
    _, _, (step, time), _ = read(parent/'outdat/flowfield.h5')
    shutil.copytree(parent/'datin', case/'datin')
    (case/'outdat').mkdir()
    (case/'validation').mkdir()
    for name in ('flowfield.h5', 'auxiliary.txt'):
        shutil.copy2(parent/'outdat'/name, case/'outdat'/name)
    set_value(case/'datin/input.air5_c4', 'lrestar', 't')
    set_value(case/'datin/controller', 'maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg',
              f'{step+1},1,1000000,1000000,1,1000000')
    env = environment('2,1,1')
    env.update(prior['environment'])
    env.update(ASTR_VALIDATION_RHS_STEP=str(step), ASTR_VALIDATION_RHS_STEP_SECONDARY=str(step+1),
               ASTR_AIR5_TOP_RESTART_VALIDATION='on', ASTR_AIR5_COMPENSATION_RESTART='restore')
    env.pop('ASTR_AIR5_FLOW_MONITOR_STRIDE', None)
    if monitor:
        env['ASTR_AIR5_FLOW_MONITOR_STRIDE'] = '1'
    command = ['timeout', '900s', shutil.which('mpirun'), '--oversubscribe', '-np', '2',
               str(executable), 'run', 'datin/input.air5_c4']
    (case/'launch.json').write_text(json.dumps(dict(command=command,
        environment={k:v for k,v in env.items() if k.startswith(('ASTR_', 'OMPI_', 'CUDA_', 'UCX_'))},
        executable_sha256=digest(executable), start_step=step, start_time=time), indent=2)+'\n')
    print(case.name, flush=True)
    with (case/'run.log').open('w') as log:
        subprocess.run(command, cwd=case, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    log = (case/'run.log').read_text()
    if 'The job is done!' not in log or 'ieee_invalid' in log.lower():
        raise ValueError('unclean restart')
    state = state_gate(case, Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json'))
    for label, data in fields(case, (step,step+1)).items():
        m = metrics(data, (0,0,0))
        if m['transport_sum_violations'] or m['min_species_density'] < 0:
            raise ValueError(f'{label}: strict composition failure')
    return step, state


def qualify(old_parent, archived, restart, output, executable):
    output.mkdir(parents=True, exist_ok=False)
    frozen = output/'astr'
    shutil.copy2(executable, frozen)
    prior = json.loads((restart/'launch.json').read_text())
    if json.loads((archived/'launch.json').read_text())['executable_sha256'] != prior['executable_sha256']:
        raise ValueError('archived reference is not from the checkpoint executable')
    step, replay_state = execute(old_parent, output/'archived_replay', frozen, False)
    replay = compare(fields(archived, (step,)), fields(output/'archived_replay', (step,)))
    step, off_state = execute(restart, output/'monitor_off', frozen, False)
    _, on_state = execute(restart, output/'monitor_on', frozen, True)
    a, b = fields(output/'monitor_off', (step,step+1)), fields(output/'monitor_on', (step,step+1))
    if any(not np.array_equal(a[k],b[k]) for k in a):
        raise ValueError('monitor changed a phase field')
    checkpoint = compare_checkpoint(output/'monitor_off/outdat/flowfield.h5',
                                    output/'monitor_on/outdat/flowfield.h5', exact=True)
    if not checkpoint['passed']:
        raise ValueError('monitor changed conservative or compensation checkpoint')
    for filename in ('air5_profiles.dat','air5_wall.dat'):
        data = np.loadtxt(output/'monitor_on/monitor'/filename)
        if not np.isfinite(data).all() or len(np.unique(data[:,1])) != 2:
            raise ValueError('invalid restarted monitor')
    report = dict(passed=True, source_executable_sha256=prior['executable_sha256'],
        candidate_executable_sha256=digest(frozen), checkpoint_sha256=digest(restart/'outdat/flowfield.h5'),
        dt=2.5e-10, topology='2,1,1', replay_comparison=replay,
        monitor_phases_bitwise_equal=True, checkpoint_comparison=checkpoint,
        states=[replay_state,off_state,on_state])
    (output/'qualification.json').write_text(json.dumps(report, indent=2)+'\n')
    print('PASS', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('old-parent','archived','restart','output','executable'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    qualify(a.old_parent.resolve(), a.archived.resolve(), a.restart.resolve(),
            a.output.resolve(), a.executable.resolve())
