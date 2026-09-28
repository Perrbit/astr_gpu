#!/usr/bin/env python3
"""Fail-fast, matched-phase CPU/GPU startup of the approved Mach-4 precursor."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np

from air5_radau_reference import Air5RadauReference
from check_air5_mach4_error_replay import state_gate
from check_air5_mass_closure_stages import metrics
from run_air5_characteristic_backend_gate import fields, compare
from run_air5_sbli_preflight import ROOT, environment, set_value


def run(template, output, executable, monitor_stride=0):
    template, output, executable = [p.resolve() for p in (template, output, executable)]
    if output.exists():
        raise FileExistsError(output)
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    settings = json.loads((template/'environment.json').read_text())
    output.mkdir(parents=True)
    report = dict(passed=False, executable_sha256=digest, scope='precursor startup only', cases={})
    model = Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json')
    reference = None
    for name, gpu, topology in [('cpu_np1', False, '1,1,1'), ('gpu_np2', True, '2,1,1')]:
        case = output/name
        shutil.copytree(template/'datin', case/'datin')
        (case/'validation').mkdir()
        set_value(case/'datin/input.air5_c4',
                  'nondimen,diffterm,lfilter,lreadgrid,lfftz,limmbou,ltimrpt,lcomb',
                  'f,t,f,f,f,f,t,t,'+('t' if gpu else 'f'))
        env = environment(topology)
        env.update(settings)
        env.update(ASTR_FORCE_MPI_TOPOLOGY=topology, ASTR_AIR5_TOP_GPU_VALIDATION='on',
                   ASTR_VALIDATION_RHS_PREFIX='validation/air5',
                   ASTR_VALIDATION_RHS_STEP='1', ASTR_VALIDATION_RHS_STEP_SECONDARY='0')
        if monitor_stride:
            env['ASTR_AIR5_FLOW_MONITOR_STRIDE'] = str(monitor_stride)
        command = ['timeout', '--kill-after=10s', '900s', shutil.which('mpirun'),
                   '--oversubscribe', '-np', '2' if gpu else '1',
                   str(executable), 'run', 'datin/input.air5_c4']
        (case/'launch.json').write_text(json.dumps(dict(command=command,
            executable_sha256=digest, environment={k:v for k,v in env.items()
            if k.startswith(('ASTR_', 'OMPI_', 'CUDA_', 'UCX_'))}), indent=2)+'\n')
        print(f'Starting {name}', flush=True)
        with (case/'run.log').open('w') as log:
            subprocess.run(command, cwd=case, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        log = (case/'run.log').read_text()
        if 'The job is done!' not in log or 'ieee_invalid' in log.lower():
            raise ValueError(f'{name}: unclean completion')
        state = state_gate(case, model)
        data = fields(case, steps=(0, 1))
        for phase, array in data.items():
            check = metrics(array, (0, 0, 0))
            if check['transport_sum_violations'] or check['min_species_density'] < 0:
                raise ValueError(f'{name}/{phase}: strict composition failure')
        result = dict(state=state, phases=len(data))
        if reference is None:
            reference = data
        else:
            result['comparison'] = compare(reference, data)
        report['cases'][name] = result
        (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(f'Passed {name}', flush=True)
    if monitor_stride:
        report['monitor_comparison'] = {}
        for name in ('air5_profiles.dat', 'air5_wall.dat'):
            a = np.loadtxt(output/'cpu_np1/monitor'/name, ndmin=2)
            b = np.loadtxt(output/'gpu_np2/monitor'/name, ndmin=2)
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError(f'{name}: incompatible monitor data')
            maximum = float(np.max(abs(a-b)/np.maximum(1.,abs(a))))
            if maximum > 2e-10:
                raise ValueError(f'{name}: CPU/GPU monitor mismatch {maximum}')
            report['monitor_comparison'][name] = maximum
    report['passed'] = True
    (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--executable', type=Path, default=ROOT/'build_gpu_probe/bin/astr')
    parser.add_argument('--monitor-stride', type=int, default=0)
    args = parser.parse_args()
    run(args.template, args.output, args.executable, args.monitor_stride)
