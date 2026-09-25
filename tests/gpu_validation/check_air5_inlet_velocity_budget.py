#!/usr/bin/env python3
"""Attribute existing ASTR SSPRK3 snapshots; never integrate or repair a field."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from check_air5_c5_frozen_transport import read_rhs_snapshot
from check_air5_c5_hbl import _active_array
from check_air5_mach4_error_replay import cartesian_jacobian, path
from check_air5_diffusion_probe import read_probe


WEIGHTS = (1/6, 1/6, 2/3)
RK = ((1., 0., 1.), (.75, .25, .25), (1/3, 2/3, 2/3))
UPDATE_WEIGHTS = (1/6, 2/3, 1.)


def velocity_terms(delta_terms, reference, candidate):
    """Exact endpoint identity, with all terms normalized using candidate rho."""
    u = reference[1]/reference[0]
    return {name: float((term[1]-u*term[0])/candidate[0])
            for name, term in delta_terms.items()}


def step_budget(case, step, dt, rank, node, jacobian, components=(0, 1)):
    components = tuple(components)
    if not components or len(set(components)) != len(components) or any(c not in range(11) for c in components):
        raise ValueError('invalid conservative component selection')
    suffix = 'rho_momentum' if components == (0, 1) else 'conservative'
    def state(label, stage):
        return _active_array(path(case, label, stage, rank, step))[node][list(components)]

    def rhs(label, stage):
        snapshot = read_rhs_snapshot(path(case, label, stage, rank, step))
        im, jm, km, nq = snapshot.header
        return snapshot.values.reshape((im+1, jm+1, km+1, nq), order='F')[node][list(components)]/jacobian

    initial = state('pre_chemistry', 1)
    origin = state('pre_rhs', 1)
    end = state('post_chemistry', 2)
    transport = state('post_transport', 1)
    terms = {key: np.zeros(len(components)) for key in ('convection_raw', 'convection_limiter',
        'diffusion', 'pre_rk_source_boundary', 'post_rk_source_boundary',
        'between_stage_change', 'post_rhs_update_residual', 'post_transport_change')}
    terms['pre_rk_source_boundary'] = origin-initial
    terms['post_rk_source_boundary'] = end-transport
    stages = []
    previous = None
    for stage, ((a, b, c), weight, update_weight) in enumerate(
            zip(RK, WEIGHTS, UPDATE_WEIGHTS), start=1):
        before = state('pre_rhs', stage)
        after = state('post_update', stage)
        raw, conv, full = [rhs(label, stage) for label in ('conv_raw', 'conv', 'full')]
        terms['convection_raw'] += dt*weight*raw
        terms['convection_limiter'] += dt*weight*(conv-raw)
        terms['diffusion'] += dt*weight*(full-conv)
        residual = after-(a*origin+b*before+c*dt*full)
        terms['post_rhs_update_residual'] += update_weight*residual
        if previous is not None:
            terms['between_stage_change'] += update_weight*b*(before-previous)
        stages.append({'stage': stage, f'raw_convection_{suffix}_rhs': raw.tolist(),
            f'limited_convection_{suffix}_rhs': conv.tolist(),
            f'diffusion_{suffix}_rhs': (full-conv).tolist(),
            'post_rhs_update_residual': residual.tolist()})
        previous = after
    terms['post_transport_change'] = transport-previous
    closure = end-initial-sum(terms.values())
    scale = abs(initial)+abs(end)+sum(abs(value) for value in terms.values())
    if not np.isfinite(scale).all() or np.any(abs(closure) > 256*np.finfo(float).eps*scale):
        raise ValueError('RK snapshot budget does not close')
    return initial, end, terms, {'step': step, 'stages': stages,
        f'closure_{suffix}': closure.tolist()}


def trajectory(root, backend, suffix, rank, node):
    directory = root/f'{backend}_{suffix}'
    return replay_trajectory(directory, rank, node)


def replay_trajectory(directory, rank, node, components=(0, 1)):
    contract = json.loads((directory/'contract.json').read_text())
    result = json.loads((directory/'result.json').read_text())
    if result['status'] != 'bounded-replay-completed-not-physical-pass':
        raise ValueError('cannot attribute an incomplete replay')
    case = directory/contract['backend']
    jacobian = cartesian_jacobian(case)
    terms, stages = {}, []
    initial = previous = None
    for step in range(contract['start_step'], contract['start_step']+contract['updates']):
        start, end, local, stage_report = step_budget(
            case, step, contract['dt'], rank, node, jacobian, components)
        if initial is None:
            initial = start
            terms = {key: np.zeros(len(components)) for key in local}
            terms['between_step_change'] = np.zeros(len(components))
        else:
            terms['between_step_change'] += start-previous
        for name, value in local.items():
            terms[name] += value
        previous = end
        stages.append(stage_report)
    return contract, initial, end, terms, stages


def compare_pair(coarse, fine):
    """Compare complete recorded RK budgets from a common restart at any step."""
    ca = json.loads((coarse/'contract.json').read_text())
    cb = json.loads((fine/'contract.json').read_text())
    for key in ('checkpoint_sha256', 'executable_sha256', 'start_step', 'start_time',
                'topology', 'backend', 'convection_limiter', 'diffusion_limiter'):
        if ca[key] != cb[key]:
            raise ValueError(f'incompatible replay {key}')
    if ca['dt'] != 2*cb['dt'] or ca['updates'] != 1 or cb['updates'] != 2:
        raise ValueError('expected one full step versus two half steps')
    if ca['topology'] != '2,1,1' or ca['target_time'] != cb['target_time']:
        raise ValueError('expected matched-time NP2 x-slab replay')
    report = dict(status='endpoint-RK-budget-identity-pass-not-physical-pass',
                  sign='two half steps minus one full step',
                  coarse=str(coarse), fine=str(fine), ranks=[])
    for rank in (0, 1):
        a = _active_array(path(coarse/ca['backend'], 'post_chemistry', 2, rank, ca['start_step']))
        b = _active_array(path(fine/cb['backend'], 'post_chemistry', 2, rank, cb['start_step']+1))
        difference = b[..., 1]/b[..., 0]-a[..., 1]/a[..., 0]
        node = tuple(map(int, np.unravel_index(abs(difference).argmax(), difference.shape)))
        _, qa, ea, ta, sa = replay_trajectory(coarse, rank, node)
        _, qb, eb, tb, sb = replay_trajectory(fine, rank, node)
        if not np.array_equal(qa, qb):
            raise ValueError('initial states differ at the diagnostic node')
        terms = velocity_terms({key: tb[key]-ta[key] for key in ta}, ea, eb)
        closure = float(difference[node]-sum(terms.values()))
        if abs(closure) > 512*np.finfo(float).eps*max(abs(ea[1]/ea[0]), abs(eb[1]/eb[0])):
            raise ValueError('endpoint velocity budget does not close')
        report['ranks'].append(dict(rank=rank, local_ijk=node,
            velocity_difference_m_s=float(difference[node]), velocity_contributions_m_s=terms,
            closure_m_s=closure, inlet_plane_max_abs_m_s=float(abs(difference[0]).max()) if rank == 0 else None,
            coarse_stages=sa, fine_stages=sb))
    return report


def stress_budget(log, steps, node):
    records = read_probe(log)
    if [(r['step'], r['stage']) for r in records] != [
            (step, stage) for step in steps for stage in range(1, 4)]:
        raise ValueError('incomplete or reordered stress probe stages')
    raw_total, limited_total = np.zeros(2), np.zeros(2)
    stages = []
    for record in records:
        if (record['rank'], record['i'], record['j'], record['k']) != (0, *node):
            raise ValueError('stress probe location mismatch')
        theta = {(entry['axis'], side): entry['theta'][side-1]
                 for entry in record['shared'] for side in (1, 2)}
        corrections = np.array([face['correction'] for face in record['faces']])
        coefficients = np.array([theta[face['axis'], face['side']] for face in record['faces']])
        if not np.isfinite(coefficients).all() or np.any((coefficients < 0) | (coefficients > 1)):
            raise ValueError('invalid shared coefficients')
        raw = corrections.sum(axis=0)
        limited = (coefficients[:, None]*corrections).sum(axis=0)
        weight = UPDATE_WEIGHTS[record['stage']-1]
        raw_total += weight*raw[:2]
        limited_total += weight*limited[:2]
        stages.append(dict(step=record['step'], stage=record['stage'],
            raw_momentum_increment_kg_m2_s=float(raw[1]),
            limited_momentum_increment_kg_m2_s=float(limited[1]),
            equivalent_limiter_velocity_increment_m_s=float(
                (limited[1]-raw[1])/record['base_SI'][0]),
            y_faces=[dict(side=f['side'], theta=theta[2, f['side']],
                raw_momentum_increment_kg_m2_s=f['correction'][1])
                for f in record['faces'] if f['axis'] == 2]))
    return raw_total, limited_total, stages


def checked_probe(root, original_root, suffix, node):
    replay = root/f'cpu_{suffix}'
    original = original_root/f'cpu_{suffix}'
    contract = json.loads((replay/'contract.json').read_text())
    reference = json.loads((original/'contract.json').read_text())
    for key in ('checkpoint_sha256', 'executable_sha256', 'dt', 'updates', 'start_step'):
        if contract[key] != reference[key]:
            raise ValueError(f'probe replay mismatch: {key}')
    if json.loads((replay/'result.json').read_text())['returncode'] != 0:
        raise ValueError('probe replay failed')
    files = sorted((original/'cpu/validation').glob('*.bin'))
    if {f.name for f in files} != {f.name for f in (replay/'cpu/validation').glob('*.bin')}:
        raise ValueError('probe changed the snapshot set')
    for filename in files:
        other = replay/'cpu/validation'/filename.name
        if hashlib.sha256(filename.read_bytes()).digest() != hashlib.sha256(other.read_bytes()).digest():
            raise ValueError(f'probe changed snapshot: {filename.name}')
    raw, limited, stages = stress_budget(replay/'cpu/run.log',
        range(contract['start_step'], contract['start_step']+contract['updates']), node)
    return raw, limited, dict(identical_snapshot_files=len(files), stages=stages)


def run(root, rank, node, probe_root=None):
    report = dict(status='snapshot-budget-diagnostic-not-physical-pass', rank=rank,
        local_ijk=node, sign='two 1ns steps minus one 2ns step', backends={})
    for backend in ('cpu', 'gpu'):
        coarse = trajectory(root, backend, 'dt2', rank, node)
        fine = trajectory(root, backend, 'dt1', rank, node)
        ca, qa, ea, ta, sa = coarse
        cb, qb, eb, tb, sb = fine
        for key in ('checkpoint_sha256', 'executable_sha256', 'start_step',
                    'topology', 'convection_limiter', 'start_time'):
            if ca[key] != cb[key]:
                raise ValueError(f'incompatible replay {key}')
        if ca.get('diffusion_limiter', 'full_state') != cb.get('diffusion_limiter', 'full_state'):
            raise ValueError('incompatible replay diffusion_limiter')
        if ca['dt'] != 2*cb['dt'] or ca['updates'] != 1 or cb['updates'] != 2:
            raise ValueError('expected one full step versus two half steps')
        if ca['topology'] != '2,1,1' or not np.array_equal(qa, qb):
            raise ValueError('expected matched NP2 x-slab checkpoint states')
        delta = {key: tb[key]-ta[key] for key in ta}
        velocity = velocity_terms(delta, ea, eb)
        observed = eb[1]/eb[0]-ea[1]/ea[0]
        if abs(observed-sum(velocity.values())) > 512*np.finfo(float).eps*max(
                abs(ea[1]/ea[0]), abs(eb[1]/eb[0])):
            raise ValueError('endpoint velocity budget does not close')
        # x=0 is prescribed inflow; x=1 is the first evolved interior plane.
        a = _active_array(path(root/f'{backend}_dt2'/backend, 'post_chemistry', 2, rank))
        b = _active_array(path(root/f'{backend}_dt1'/backend, 'post_chemistry', 2, rank, 1001))
        inlet_difference = b[0, ..., 1]/b[0, ..., 0]-a[0, ..., 1]/a[0, ..., 0]
        report['backends'][backend] = dict(observed_velocity_difference_m_s=float(observed),
            velocity_contributions_m_s=velocity,
            density_difference_kg_m3=float(eb[0]-ea[0]),
            momentum_difference_kg_m2_s=float(eb[1]-ea[1]),
            inlet_plane_velocity_max_abs_m_s=float(abs(inlet_difference).max()) if rank == 0 else None,
            initial_velocity_m_s=float(qa[1]/qa[0]),
            endpoint_velocities_m_s=[float(ea[1]/ea[0]), float(eb[1]/eb[0])],
            coarse_stages=sa, fine_stages=sb)
        if backend == 'cpu' and probe_root is not None:
            raw_a, limited_a, pa = checked_probe(probe_root, root, 'dt2', node)
            raw_b, limited_b, pb = checked_probe(probe_root, root, 'dt1', node)
            for actual, expected in ((limited_a, ta['diffusion']), (limited_b, tb['diffusion'])):
                if not np.allclose(actual, expected, rtol=1e-10, atol=1e-15):
                    raise ValueError('face probe disagrees with recorded diffusion RHS')
            split = dict(unlimited_stress_on_recorded_states=raw_b-raw_a,
                         direct_stress_limiting=(limited_b-raw_b)-(limited_a-raw_a))
            report['stress_attribution'] = dict(
                scope='algebra on actual trajectories, not a rerun with limiting disabled',
                velocity_contributions_m_s=velocity_terms(split, ea, eb),
                coarse=pa, fine=pb)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--root', type=Path)
    selection.add_argument('--coarse-case', type=Path)
    parser.add_argument('--fine-case', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--probe-root', type=Path)
    args = parser.parse_args()
    if args.coarse_case:
        if not args.fine_case or args.probe_root:
            parser.error('--coarse-case requires --fine-case and excludes --probe-root')
        result = compare_pair(args.coarse_case, args.fine_case)
    else:
        if args.fine_case:
            parser.error('--fine-case requires --coarse-case')
        result = run(args.root, rank=0, node=(1, 35, 1), probe_root=args.probe_root)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for backend, data in result.get('backends', {}).items():
        print(json.dumps(dict(backend=backend, **{k: v for k, v in data.items() if not k.endswith('stages')})))
    for data in result.get('ranks', []):
        print(json.dumps({key: value for key, value in data.items() if not key.endswith('stages')}))
    if 'stress_attribution' in result:
        print(json.dumps(result['stress_attribution']))
