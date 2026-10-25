#!/usr/bin/env python3
"""Local M12 outlet-backflow integration, partition and exact-restart checks."""

import argparse
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np

from prepare_m12_local_case import prepare
from run_m12_boundary_compare import active_state, configuration
from run_m12_mp7_nscbc_performance import compare_state, mpi_command, runtime
from insitu_cfl_gate import CflGate


ROOT = Path(__file__).resolve().parents[2]
SHAPE = (64, 32, 24)
GAMMA = 1.4


def seed_outlet_backflow(case, mach):
    """Keep wall/top states intact; prescribe normal Mach in an outlet patch."""
    with h5py.File(case / 'datin/grid.2d') as mesh:
        x = mesh['x'][...]
    s = np.clip(((x-x.min())/(x.max()-x.min())-.8)/.1, 0., 1.)
    x_weight = s*s*(3.-2.*s)
    eta = np.linspace(0., 1., x.shape[0])[:, None]
    y_weight = np.sin(np.pi*eta)**2
    y_weight[[0, -1]] = 0.
    weight = x_weight*y_weight
    with h5py.File(case / 'datin/flowini2d.h5', 'r+') as field:
        a = np.sqrt(GAMMA*field['p'][...]/field['ro'][...])
        u = field['u1'][...]
        field['u1'][...] = (1.-weight)*u-weight*mach*a
    return dict(normal_mach_target=-mach, x_start_fraction=.8, x_plateau_fraction=.9,
                y_weight='sin(pi*j/J)^2', y_target_node=x.shape[0]//2)


def final_state(case, ranks, axis, step=9):
    pieces = []
    for rank in range(ranks):
        path = case / f'diagnostics/state.post_update.step{step:08d}.rk03.rank{rank:08d}.bin'
        q = active_state(path)
        internal = q[..., 4] - .5*np.sum(q[..., 1:4]**2, axis=-1)/q[..., 0]
        if not np.isfinite(q).all() or np.min(q[..., 0]) <= 0 or np.min(internal) <= 0:
            raise ValueError(f'Invalid physical state: {path}')
        selection = [slice(None)]*4
        if rank < ranks-1:
            selection[axis] = slice(None, -1)
        pieces.append(q[tuple(selection)])
    return np.concatenate(pieces, axis=axis)


def run_case(args, label, topology, *, reverse=False, steps=10, restore=None,
             checkpoint=False, zero_forcing=True, pout=1., exact=True, expect_failure=None, outlet=22,
             backflow_mach=None):
    case = args.out / label
    prepare(args.source, case, 'gpu', SHAPE)
    configuration(case, 'nscbc', SHAPE, steps, archive=False,
                  checkpoint_interval=5 if checkpoint else None, restore=restore, exact_restart=exact)
    if pout != 1.:
        path = case / 'datin/input.dat'
        path.write_text(path.read_text().replace('22,1.d0', f'22,{pout:.16e}'))
    if outlet != 22:
        path = case / 'datin/input.dat'
        path.write_text(path.read_text().replace('22,1.d0', f'{outlet},1.d0').replace('\n52\n', '\n50\n'))
    if zero_forcing:
        path = case / 'datin/wallbs.dat'
        lines = path.read_text().splitlines()
        lines[2] = '0.0, 0.0, 20.0, 40.0, 60.0'
        path.write_text('\n'.join(lines)+'\n')
    if backflow_mach is not None and restore is None:
        seed = seed_outlet_backflow(case, backflow_mach)
        (case/'backflow_fixture.json').write_text(json.dumps(seed, indent=2)+'\n')
    elif reverse and restore is None:
        with h5py.File(case / 'datin/grid.2d') as mesh:
            x = mesh['x'][...]
        # Smooth manufactured reverse flow on the last fifth of the domain.
        s = np.clip((x-x.min())/(x.max()-x.min())*5.-4., 0., 1.)
        blend = s*s*(3.-2.*s)
        with h5py.File(case / 'datin/flowini2d.h5', 'r+') as field:
            field['u1'][...] *= 1.-1.03*blend
    ranks = int(np.prod(topology))
    env = runtime(','.join(map(str, topology)))
    env.update(ASTR_GPU_MP_POSITIVITY='off', ASTR_VALIDATION_RHS_PREFIX='diagnostics/state',
               ASTR_VALIDATION_RHS_STEP=str(steps-1))
    if backflow_mach is not None:
        env['ASTR_VALIDATION_RHS_STEP_SECONDARY'] = '0'
    command = [*mpi_command(args.mpirun, ranks, 'pinned'), str(args.exe.resolve()), 'run', 'datin/input.dat']
    result = subprocess.run(command, cwd=case, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (case / 'run.log').write_text(result.stdout)
    if expect_failure:
        if result.returncode == 0 or expect_failure not in result.stdout:
            raise ValueError(f'Missing expected restart diagnostic: {case}')
    elif result.returncode:
        raise ValueError(f'Run failed: {case}; returncode={result.returncode}')
    return case


def reverse_regime_checks(args, report):
    """Real RK integration, not just evaluation of a manufactured boundary RHS."""
    for regime, target in (('subsonic', .5), ('supersonic', 1.5)):
        reference = None
        for code in dict.fromkeys(['111', *args.topologies]):
            topology = tuple(map(int, code))
            axis = next((i for i, n in enumerate(topology) if n > 1), 0)
            ranks = int(np.prod(topology))
            case = run_case(args, regime+'_'+code, topology, steps=args.steps, backflow_mach=target)
            gate = CflGate(case/'run.log', 1., .02, 0, args.steps)
            gate.check(final=True)
            samples = []
            owners = [ranks-1] if axis == 0 else range(ranks)
            for step in (0, args.steps-1):
                for stage in (1, 2, 3):
                    states = []
                    for rank in owners:
                        name = f'state.post_update.step{step:08d}.rk{stage:02d}.rank{rank:08d}.bin'
                        q = active_state(case/'diagnostics'/name)
                        density = q[..., 0]
                        internal = q[..., 4]-.5*np.sum(q[..., 1:4]**2, axis=-1)/density
                        if not np.isfinite(q).all() or density.min() <= 0 or internal.min() <= 0:
                            raise ValueError(f'Invalid RK sample: {case/name}')
                        pressure = (GAMMA-1.)*internal
                        normal_mach = q[..., 1]/density/np.sqrt(GAMMA*pressure/density)
                        # Near the peak of the smooth patch, away from its sonic crossings.
                        j0 = rank*(SHAPE[1]//ranks) if axis == 1 else 0
                        jglobal = j0+np.arange(q.shape[1])
                        states.append(normal_mach[-1, (jglobal >= 15) & (jglobal <= 17), :-1].ravel())
                    values = np.concatenate(states)
                    lower, upper = float(values.min()), float(values.max())
                    valid = lower > -1. and upper < 0. if regime == 'subsonic' else upper < -1.
                    if not valid:
                        raise ValueError(f'{regime} sample changed regime: {case}, step={step}, stage={stage}, [{lower},{upper}]')
                    samples.append(dict(step=step, stage=stage, normal_mach_min=lower, normal_mach_max=upper))
            q = final_state(case, ranks, axis, args.steps-1)
            if reference is None:
                reference = q
            error = float(np.max(np.abs(q-reference)))
            if error > 2e-10:
                raise ValueError(f'Partition mismatch: {case}, {error}')
            entry = dict(case=case.name, topology=topology, steps=args.steps, dt=.02,
                         target_normal_mach=-target, max_cfl=gate.maximum, max_abs=error,
                         minimum_density=float(q[..., 0].min()),
                         minimum_internal_energy=float((q[..., 4]-.5*np.sum(q[..., 1:4]**2, axis=-1)/q[..., 0]).min()),
                         samples=samples)
            report['checks'].append(entry)
            print(json.dumps(entry), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('/home/dell/workspace/M12_C13'))
    parser.add_argument('--mpirun', default='/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--topologies', nargs='+', choices=('111', '211', '121', '112'),
                        default=('111', '211', '121', '112'))
    parser.add_argument('--regimes-only', action='store_true', help='Run sub/supersonic outlet patches without the old restart matrix')
    parser.add_argument('--steps', type=int, default=100, help='Complete RK steps for --regimes-only')
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(scope='local five-variable M12 short integration, not production validation', checks=[])
    try:
        if args.regimes_only:
            reverse_regime_checks(args, report)
            report['status'] = 'passed'
            return
        for reverse in (False, True):
            label = 'backflow' if reverse else 'outflow'
            reference = None
            selected = list(dict.fromkeys(['111', *args.topologies]))
            for code in selected:
                topology = tuple(map(int, code))
                axis = next((i for i, n in enumerate(topology) if n > 1), 0)
                ranks = int(np.prod(topology))
                case = run_case(args, label+'_'+''.join(map(str, topology)), topology, reverse=reverse)
                q = final_state(case, ranks, axis)
                if reverse and np.min(q[-1,...,1]/q[-1,...,0]) >= 0:
                    raise ValueError('Manufactured outlet no longer exercises backflow')
                if reference is None:
                    reference = q
                error = float(np.max(np.abs(q-reference)))
                if error > 2e-10:
                    raise ValueError(f'Partition mismatch: {case}, {error}')
                entry = dict(case=case.name, topology=topology, max_abs=error,
                             min_outlet_u=float(np.min(q[-1,...,1]/q[-1,...,0])))
                report['checks'].append(entry)
                print(json.dumps(entry), flush=True)
        continuous = run_case(args, 'restart_continuous', (2,1,1), reverse=True,
                              checkpoint=True, zero_forcing=False)
        first = run_case(args, 'restart_first', (2,1,1), reverse=True, steps=5,
                         checkpoint=True, zero_forcing=False)
        saved = first / 'outdat/new/checkpoints/step000000000005'
        resumed = run_case(args, 'restart_resumed', (2,1,1), restore=saved,
                           checkpoint=True, zero_forcing=False)
        final = Path('outdat/new/checkpoints/step000000000010')
        comparison = compare_state(continuous/final/'state.h5', resumed/final/'state.h5')
        if (continuous/final/'control.bin').read_bytes() != (resumed/final/'control.bin').read_bytes():
            raise ValueError('Exact restart control/RNG state differs')
        report['checks'].append(dict(case='exact_backflow_restart', datasets=comparison, exact=True))
        run_case(args, 'restart_changed_reference_exact', (2,1,1), restore=saved,
                 checkpoint=True, zero_forcing=False, pout=.95,
                 expect_failure='exact bc22 reference mismatch')
        changed = run_case(args, 'restart_changed_reference_nonexact', (2,1,1), restore=saved,
                           checkpoint=True, zero_forcing=False, pout=.95, exact=False)
        final_state(changed, 2, 0)
        report['checks'].append(dict(case='reference_change', exact_rejected=True, nonexact_completed=True))
        report['status'] = 'passed'
    except (ValueError, subprocess.SubprocessError) as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        (args.out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
