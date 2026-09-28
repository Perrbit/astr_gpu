#!/usr/bin/env python3
"""Check admitted AIR5 filtering and rejection of unimplemented combinations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from run_air5_sbli_preflight import environment

ROOT = Path(__file__).resolve().parents[2]


def compare_saved(output):
    from run_air5_characteristic_backend_gate import fields, compare
    report = json.loads((output/'result.json').read_text())
    if report['status'] != 'admitted-baseline-pass-production-combination-blocked':
        raise ValueError('baseline matrix must finish successfully before comparison')
    reference = fields(output/'1x1x1_full'/'gpu', steps=(0,))
    comparisons = {}
    for case in report['cases']:
        for backend in ('cpu', 'gpu'):
            name = case['name']+'/'+backend
            comparisons[name] = compare(reference, fields(output/name, steps=(0,)))
    result = dict(reference='1x1x1_full/gpu', tolerance=2e-10,
                  definition='abs(candidate-reference)/max(1,abs(reference))',
                  comparisons=comparisons, passed=True)
    (output/'cross_comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print('cross-workspace/backend/topology maximum scaled error:',
          max(phase['max_scaled'] for case in comparisons.values() for phase in case.values()))


def run(executable, output, mpirun):
    executable, output, mpirun = (p.resolve() for p in (executable, output, mpirun))
    if not executable.is_file() or not mpirun.is_file():
        raise ValueError('executable and MPI launcher must exist')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', production_ready=False,
                  scope='one-update prescribed-top filtering, compensation off',
                  executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                  cases=[], rejection_gates=[])
    def save():
        (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        for topology in ('1,1,1', '2,1,1', '1,2,1', '1,1,2'):
            for workspace in ('full', 'scalar'):
                name = topology.replace(',', 'x')+'_'+workspace
                case = output/name
                env = environment(topology)
                env.update(PATH=str(mpirun.parent)+os.pathsep+env['PATH'],
                           EXE=str(executable), OUT_DIR=str(case), GRID='31,31,15',
                           MAXSTEP='0', VALIDATION_STEP='0', VALIDATION_STEP_SECONDARY='',
                           DELTAT='1.d-10', MPI_NP=str(1 if topology=='1,1,1' else 2),
                           TOPOLOGY=topology, LFILTER='t', FILTER_WORKSPACE=workspace,
                           HBL_INITIAL_FIELD='uniform', ATOL='1e-9', RTOL='1e-10',
                           SAME_PHASE_SCALED_TOL='', EXTRUSION_SCALED_TOL='2e-10',
                           EXTRUSION_GATE='raw', ASTR_AIR5_COMPENSATION='off',
                           ASTR_AIR5_TOP_MODE='prescribed',
                           ASTR_AIR5_CONVECTION_LIMITER='symmetric_species',
                           ASTR_AIR5_DIFFUSION_LIMITER='layered')
                print(name, flush=True)
                with (output/(name+'.log')).open('w') as log:
                    subprocess.run(['bash', str(ROOT/'tests/gpu_validation/run_air5_c5_hbl_compare.sh')],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                report['cases'].append(dict(name=name, passed=True))
                save()
        # Production use remains opt-in until the combined lifecycle gates pass.
        for backend in ('cpu', 'gpu'):
            for mode, compensated, message in (
                ('characteristic', 'off', 'characteristic AIR5 filter requires explicit validation opt-in'),
                ('prescribed', 'on', 'compensated AIR5 filter requires explicit validation opt-in')):
                name = backend+'_'+mode+'_compensation_'+compensated
                case = output/name
                shutil.copytree(output/'1x1x1_full'/backend/'datin', case/'datin')
                env = environment('1,1,1')
                env.update(ASTR_AIR5_TOP_MODE=mode, ASTR_AIR5_COMPENSATION=compensated,
                           ASTR_AIR5_CONVECTION_LIMITER='symmetric_species',
                           ASTR_AIR5_DIFFUSION_LIMITER='layered')
                with (case/'run.log').open('w') as log:
                    result = subprocess.run(['timeout', '--kill-after=5s', '30s', str(mpirun),
                                             '--oversubscribe', '-np', '1', str(executable),
                                             'run', 'datin/input.air5_c4'], cwd=case, env=env,
                                            stdout=log, stderr=subprocess.STDOUT)
                text = (case/'run.log').read_text()
                if result.returncode in (0,124,137) or message not in text:
                    raise ValueError('expected admission rejection missing: '+name)
                if 'AIR5_CHEMISTRY half=' in text:
                    raise ValueError('rejected case entered chemistry: '+name)
                report['rejection_gates'].append(dict(name=name, passed=True))
                save()
    except BaseException as error:
        report.update(status='stopped', error=str(error))
        save()
        raise
    report.update(status='admitted-baseline-pass-production-combination-blocked')
    save()
    print(report['status'], flush=True)
    compare_saved(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mpirun', type=Path, required=True)
    args = parser.parse_args()
    run(args.executable, args.output, args.mpirun)
