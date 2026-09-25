#!/usr/bin/env python3
"""Small reacting-flow gates for same-state reuse and packed diagnostics."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np

from run_air5_sbli_preflight import ROOT, environment, set_value
from run_air5_layered_stress_gate import check_shared_face_payloads
from air5_radau_reference import Air5RadauReference
from compare_q_validation_snapshots import compare_snapshot_sets
from check_air5_c5_reacting_tgv import analyze


def check_trace_metrics(metrics):
    # A prescribed zero initial species is admissible; negative values never are.
    return (all(np.isfinite(v) for v in asdict(metrics).values()) and
            metrics.max_chemistry_constraint_change <= 1e-8 and
            metrics.minimum_chemistry_change >= 1e-12 and
            metrics.max_species_mass_closure <= 1e-10 and
            metrics.max_element_relative_change <= 2e-11 and
            metrics.minimum_species_density >= 0. and
            metrics.minimum_temperature >= 5000. and
            metrics.transport_change >= 1e-12 and
            metrics.x_velocity_range >= 1. and metrics.y_velocity_range >= 1. and
            metrics.z_modulation >= 1.)


def trace_restarts(args, root, reference):
    """Modify initial checkpoint composition only; ASTR advances both references."""
    thermo = Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json')
    reports = {}
    for trace in (0., 1e-12):
        cases = {}
        for backend in ('cpu', 'gpu'):
            name = f'trace_{trace:g}_{backend}'
            case = root/name
            for directory in ('datin', 'outdat'):
                shutil.copytree(reference/directory, case/directory)
            (case/'validation').mkdir()
            set_value(case/'datin/input.air5_c4', 'lrestar', 't')
            set_value(case/'datin/input.air5_c4',
                'nondimen,diffterm,lfilter,lreadgrid,lfftz,limmbou,ltimrpt,lcomb',
                'f,t,t,f,f,f,t,t,'+('t' if backend == 'gpu' else 'f'))
            fractions = np.array([.77, .23-trace, 0., 0., trace])
            with h5py.File(case/'outdat/flowfield.h5', 'r+') as f:
                f['nstep'][...] = 0
                f['time'][...] = 0.
                for s in range(5):
                    f[f'sp{s+1:03d}'][...] = fractions[s]
                f['p'][...] = f['ro'][...]*f['t'][...]*(fractions@thermo.gas_constant)
            env = environment('2,1,1')
            env.update(ASTR_AIR5_PRIMITIVE_REUSE='chemistry',
                ASTR_AIR5_CHEMISTRY_REDUCTIONS='packed',
                ASTR_AIR5_CONVECTION_LIMITER='symmetric_species',
                ASTR_AIR5_DIFFUSION_LIMITER='layered',
                ASTR_VALIDATION_RHS_PREFIX='validation/air5',
                ASTR_AIR5_C4_CONSERVATION='t' if backend == 'gpu' else 'f',
                ASTR_AIR5_SYMMETRIC_FACE_PROBE='1')
            with (case/'run.log').open('w') as log:
                subprocess.run(['mpirun', '-np', '2', str(args.executable.resolve()),
                    'run', 'datin/input.air5_c4'], cwd=case, env=env,
                    stdout=log, stderr=subprocess.STDOUT, timeout=90, check=True)
            metrics = analyze(case/'validation/air5', ROOT/'chemMech/air5_kimjo12.json')
            accepted = check_trace_metrics(metrics)
            (case/'physical_check.json').write_text(json.dumps(dict(passed=accepted, **asdict(metrics)), indent=2)+'\n')
            if not accepted:
                raise ValueError(f'zero/trace reacting state failed: {case}')
            cases[backend] = case
        comparisons = compare_snapshot_sets(cases['cpu']/'validation/air5', cases['gpu']/'validation/air5',
            ('pre_chemistry', 'post_chemistry', 'pre_rhs', 'post_update', 'post_transport'),
            1e-8, 2e-11, active_only=True)
        if not comparisons.passed or comparisons.file_count == 0:
            raise ValueError(f'trace restart mismatch: {trace}')
        subprocess.run([sys.executable, str(ROOT/'tests/gpu_validation/check_air5_c4_conservation.py'),
            '--input', str(cases['gpu']/'air5_c4_conservation.dat'),
            '--report', str(cases['gpu']/'conservation.txt'), '--allow-species-change'], check=True)
        reports[str(trace)] = dict(files=comparisons.file_count, max_abs=comparisons.max_abs,
            shared_faces=check_shared_face_payloads(cases['gpu'], (2, 1, 1)))
    return reports


def replay_gpu_matrix_case(args, root, name, env, topology, log):
    """CPU implementation and inputs are unchanged: reuse its immutable snapshots."""
    reference = args.reference_matrix.resolve()/name
    case = root/name/'gpu'
    shutil.copytree(reference/'gpu/datin', case/'datin')
    (case/'validation').mkdir()
    env.update(ASTR_VALIDATION_RHS_PREFIX='validation/air5', ASTR_AIR5_C4_CONSERVATION='t')
    with (case/'gpu.log').open('w') as output:
        subprocess.run(['mpirun', '-np', str(np.prod(tuple(map(int, topology.split(','))))),
            str(args.executable.resolve()), 'run', 'datin/input.air5_c4'], cwd=case,
            env=env, stdout=output, stderr=subprocess.STDOUT, timeout=180, check=True)
    labels = ('pre_chemistry', 'post_chemistry', 'pre_rhs', 'post_update', 'post_transport')
    for backend in ('cpu', 'gpu'):
        result = compare_snapshot_sets(reference/backend/'validation/air5',
            case/'validation/air5', labels, 1e-8, 2e-11, active_only=True)
        (root/name/f'{backend}_reference_compare.json').write_text(json.dumps(asdict(result), indent=2)+'\n')
        if not result.passed:
            raise ValueError(f'{backend} immutable reference mismatch: {name}')
    subprocess.run([sys.executable, str(ROOT/'tests/gpu_validation/check_air5_c5_reacting_tgv.py'),
        '--prefix', str(case/'validation/air5'), '--mechanism', str(ROOT/'chemMech/air5_kimjo12.json'),
        '--report', str(root/name/'gpu_reacting_tgv_contract.txt')], stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, str(ROOT/'tests/gpu_validation/check_air5_c4_conservation.py'),
        '--input', str(case/'air5_c4_conservation.dat'), '--report', str(root/name/'gpu_global_conservation.txt'),
        '--allow-species-change'], stdout=log, stderr=subprocess.STDOUT, check=True)


def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    modes = dict(off=('off', 'baseline'), reuse=('chemistry', 'baseline'),
                 packed=('off', 'packed'), combined=('chemistry', 'packed'))
    report = dict(status='running', cases={},
                  executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest())
    try:
        for mode in ([] if args.supplement_from else args.modes):
            for axis, topology in [('np1', '1,1,1'), ('x', '2,1,1'),
                                   ('y', '1,2,1'), ('z', '1,1,2')]:
                for workspace in args.filters:
                    name = f'{mode}_{axis}_{workspace}'
                    env = environment(topology)
                    env.update(EXE=str(args.executable.resolve()), OUT_DIR=str(root/name),
                        NP='1' if axis == 'np1' else '2', TOPOLOGY=topology,
                        MAXSTEP='1', GRID='12,12,12', LFILTER='f' if workspace == 'off' else 't',
                        ASTR_AIR5_PRIMITIVE_REUSE=modes[mode][0],
                        ASTR_AIR5_CHEMISTRY_REDUCTIONS=modes[mode][1],
                        ASTR_AIR5_CONVECTION_LIMITER='symmetric_species',
                        ASTR_AIR5_DIFFUSION_LIMITER='layered',
                        ASTR_AIR5_SYMMETRIC_FACE_PROBE='1',
                        ASTR_GPU_FILTER_WORKSPACE='full' if workspace == 'off' else workspace)
                    with (root/f'{name}.log').open('w') as log:
                        if args.reference_matrix:
                            replay_gpu_matrix_case(args, root, name, env, topology, log)
                        else:
                            subprocess.run(['bash', str(ROOT/'tests/gpu_validation/run_air5_c5_reacting_tgv_compare.sh')],
                                env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
                    text = (root/name/'gpu/gpu.log').read_text()
                    expected = ('AIR5_CHEMISTRY_OPTIONS reuse='+('T' if mode in ('reuse', 'combined') else 'F')+
                                ' packed_reductions='+('T' if mode in ('packed', 'combined') else 'F'))
                    if expected not in text:
                        raise ValueError(f'candidate not activated: {name}')
                    faces = check_shared_face_payloads(root/name/'gpu', tuple(map(int, topology.split(','))))
                    report['cases'][name] = dict(status='passed', topology=topology,
                                                options=modes[mode], shared_faces=faces)
                    print(name, 'passed', flush=True)
        report['status'] = 'small-reacting-gates-pass-not-long-physical-validation'
        reference = (args.supplement_from.resolve() if args.supplement_from else
                     root/f'{args.modes[0]}_x_{args.filters[0]}'/'gpu')
        report['supplement_reference'] = str(reference)
        if args.trace_restart:
            report['trace_restart'] = trace_restarts(args, root, reference)
        if args.fault_library:
            case = root/'packed_failure'
            shutil.copytree(reference/'datin', case/'datin')
            (case/'validation').mkdir()
            env = environment('2,1,1')
            env.update(ASTR_AIR5_PRIMITIVE_REUSE='chemistry',
                ASTR_AIR5_CHEMISTRY_REDUCTIONS='packed',
                ASTR_AIR5_CONVECTION_LIMITER='symmetric_species',
                ASTR_AIR5_DIFFUSION_LIMITER='layered',
                ASTR_VALIDATION_RHS_PREFIX='validation/air5',
                LD_PRELOAD=str(args.fault_library.resolve()))
            with (case/'run.log').open('w') as log:
                result = subprocess.run(['mpirun', '-np', '2', str(args.executable.resolve()),
                    'run', 'datin/input.air5_c4'], cwd=case, env=env,
                    stdout=log, stderr=subprocess.STDOUT, timeout=30)
            text = (case/'run.log').read_text()
            if (result.returncode != 99 or 'AIR5_TEST_INJECT' not in text or
                    'half 1 failed, status=99' not in text or
                    not list((case/'validation').glob('air5.pre_chemistry.*.bin')) or
                    list((case/'validation').glob('air5.post_chemistry.*.bin')) or
                    list((case/'validation').glob('air5.pre_rhs.*.bin'))):
                raise ValueError('injected rank failure did not stop before dependent stages')
            report['fault_injection'] = dict(returncode=result.returncode, status='fail-closed')
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        (root/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--modes', nargs='+', choices=('off', 'reuse', 'packed', 'combined'),
                        default=['off', 'reuse', 'packed', 'combined'])
    parser.add_argument('--filters', nargs='+', choices=('off', 'full', 'scalar'), default=['off'])
    parser.add_argument('--fault-library', type=Path, help='test-only MPI status injection shared library')
    parser.add_argument('--trace-restart', action='store_true', help='filtered reacting zero/trace restart gates')
    parser.add_argument('--supplement-from', type=Path,
                        help='reuse a completed reacting TGV GPU case for supplements only; skip the matrix')
    parser.add_argument('--reference-matrix', type=Path,
                        help='GPU-only rerun against immutable CPU/GPU snapshots when CPU and inputs are unchanged')
    run(parser.parse_args())
