#!/usr/bin/env python3
"""Same-mode dynamic-top checkpoint continuation and fail-closed metadata tests."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np

from run_air5_characteristic_acoustic import prepare
from run_air5_characteristic_backend_gate import fields, compare
from run_air5_sbli_preflight import ROOT, environment, set_value
from air5_radau_reference import Air5RadauReference
from check_air5_mach4_error_replay import state_gate
from check_air5_mass_closure_stages import metrics


def launch(case, executable, topology, env, log_name, expected_failure=False, mpi_launcher=None):
    command = ['timeout', '--kill-after=10s', '600s', str(mpi_launcher or shutil.which('mpirun')),
               '--oversubscribe', '-np', str(np.prod(list(map(int, topology.split(','))))),
               str(executable), 'run', 'datin/input.air5_c4']
    with (case/log_name).open('w') as stream:
        result = subprocess.run(command, cwd=case, env=env, stdout=stream, stderr=subprocess.STDOUT)
    log = (case/log_name).read_text()
    if expected_failure:
        if result.returncode != 91 or 'AIR5 top checkpoint mode/target/tau mismatch' not in log:
            raise ValueError(f'metadata rejection failed: {result.returncode}, {case}')
    else:
        result.check_returncode()
        if 'The job is done!' not in log or 'ieee_invalid' in log.lower():
            raise ValueError(f'unclean completion: {case}')


def run(output, executable, topology, use_gpu=True, filtering=False, filter_workspace='full'):
    output.mkdir(parents=True, exist_ok=False)
    cases = {key: output/key for key in ('continuous', 'split')}
    env = environment(topology)
    for key, case in cases.items():
        meta = prepare(case, executable, ny=128, nz=16, pulse_y=.0096,
                       short_updates=6, use_gpu=use_gpu, temperature0=3000., tv0=1500.)
        env.update(meta['environment'])
        env.pop('ASTR_AIR5_ACOUSTIC_PROBE', None)
        if filtering:
            env.update(ASTR_AIR5_FILTER_VALIDATION='on', ASTR_GPU_FILTER_WORKSPACE=filter_workspace)
            set_value(case/'datin/controller', 'deltat', '1.d-10')
        env.update(ASTR_AIR5_SOURCE_MODE='coupled', ASTR_AIR5_TOP_RESTART_VALIDATION='on',
                   ASTR_AIR5_CONVECTION_LIMITER='symmetric_species',
                   ASTR_AIR5_DIFFUSION_LIMITER='layered',
                   ASTR_AIR5_COMPENSATION_RESTART='restore', ASTR_VALIDATION_RHS_STEP='5')
        set_value(case/'datin/input.air5_c4',
                  'nondimen,diffterm,lfilter,lreadgrid,lfftz,limmbou,ltimrpt,lcomb',
                  'f,t,'+('t' if filtering else 'f')+',f,f,f,t,t,'+('t' if use_gpu else 'f'))
        set_value(case/'datin/controller', 'maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg',
                  f'{5 if key=="continuous" else 2},1,1000000,1000000,50,1000000')
        meta.update(source_mode='coupled', viscous=True, topology=topology,
                    filtering=filtering, filter_workspace=filter_workspace,
                    dt=1e-10 if filtering else meta['dt'],
                    updates=6 if key=='continuous' else 3,
                    environment={k:v for k,v in env.items() if k.startswith('ASTR_')},
                    sampling_phase='complete coupled chemistry and SSP-RK phase snapshots')
        (case/'gate.json').write_text(json.dumps(meta, indent=2)+'\n')
        launch(case, executable, topology, env, 'fresh.log')
    split = cases['split']
    seed = output/'seed'
    seed.mkdir()
    shutil.copytree(split/'datin', seed/'datin')
    shutil.copytree(split/'outdat', seed/'outdat')
    with h5py.File(seed/'outdat/flowfield.h5') as stream:
        if int(stream['air5_top_mode'][()].item()) != 1:
            raise ValueError('missing dynamic-top checkpoint metadata')
        if int(stream['air5_top_version'][()].item()) != 2:
            raise ValueError('checkpoint does not use inflow-transit relaxation')
        nonzero_carry = sum(np.count_nonzero(stream[f'acc{i:02}'][()]) for i in range(1,12))
        if nonzero_carry == 0:
            raise ValueError('restart seed did not exercise nonzero compensation')
    # AIR5 monitors deliberately refuse overwriting an existing segment.
    # Resume the checkpoint in a new directory and retain the original files.
    split = output/'resumed'
    shutil.copytree(seed, split)
    (split/'validation').mkdir()
    (split/'monitor').mkdir()
    cases['split'] = split
    set_value(split/'datin/input.air5_c4', 'lrestar', 't')
    set_value(split/'datin/controller', 'maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg',
              '5,1,1000000,1000000,50,1000000')
    launch(split, executable, topology, env, 'restart.log')
    sampled = {}
    for key, case in cases.items():
        state_gate(case, Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json'))
        sampled[key] = fields(case, (5,))
        for phase, array in sampled[key].items():
            row = metrics(array, (0, 0, 0))
            if row['transport_sum_violations'] or row['min_species_density'] < 0:
                raise ValueError(f'{key}/{phase}: strict composition failure')
    comparison = compare(sampled['continuous'], sampled['split'])
    bitwise = {}
    with h5py.File(cases['continuous']/'outdat/flowfield.h5') as a, \
         h5py.File(split/'outdat/flowfield.h5') as b:
        for name in ('nstep', 'time', *[f'{prefix}{i:02}' for prefix in ('acq','acc') for i in range(1,12)]):
            left, right = np.asarray(a[name][()]), np.asarray(b[name][()])
            bitwise[name] = (left.dtype == right.dtype and left.shape == right.shape
                             and left.tobytes() == right.tobytes())
    if not all(bitwise.values()):
        raise ValueError(f'checkpoint continuation is not bitwise: {bitwise}')
    for fault in ('tau', 'missing_metadata', 'compensation_off', 'old_top_policy'):
        case = output/fault
        shutil.copytree(seed, case)
        (case/'validation').mkdir()
        (case/'monitor').mkdir()
        set_value(case/'datin/input.air5_c4', 'lrestar', 't')
        bad_env = env.copy()
        if fault == 'tau':
            bad_env['ASTR_AIR5_TOP_TAU'] = str(2*float(env['ASTR_AIR5_TOP_TAU']))
        elif fault == 'compensation_off':
            bad_env['ASTR_AIR5_COMPENSATION'] = 'off'
        elif fault == 'old_top_policy':
            with h5py.File(case/'outdat/flowfield.h5', 'r+') as stream:
                stream['air5_top_version'][...] = 1
        else:
            with h5py.File(case/'outdat/flowfield.h5', 'r+') as stream:
                del stream['air5_top_version']
        launch(case, executable, topology, bad_env, 'rejection.log', expected_failure=True)
    report = dict(passed=True, topology=topology, use_gpu=use_gpu, comparison=comparison,
                  filtering=filtering, filter_workspace=filter_workspace,
                  checkpoint_bitwise=bitwise, seed_nonzero_carry=int(nonzero_carry),
                  rejected=['tau', 'missing_metadata', 'compensation_off', 'old_top_policy'])
    (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--topology', default='1,1,1')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--filter', action='store_true')
    parser.add_argument('--filter-workspace', choices=('full','scalar'), default='full')
    parser.add_argument('--executable', type=Path, default=ROOT/'build_gpu_probe/bin/astr')
    args = parser.parse_args()
    run(args.output.resolve(), args.executable.resolve(), args.topology, not args.cpu,
        args.filter, args.filter_workspace)
