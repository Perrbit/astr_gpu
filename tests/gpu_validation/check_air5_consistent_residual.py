#!/usr/bin/env python3
"""Read recorded ASTR states and RK budgets; do not integrate a flow solution."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from air5_radau_reference import Air5RadauReference
from check_air5_c5_hbl import _active_array
from check_air5_c5_normal_shock import primitive_metrics
from check_air5_inlet_velocity_budget import replay_trajectory
from check_air5_mach4_error_replay import COMPONENTS, check_window_contract, path, scalar


def mass_gas_interval(lower, upper, mass_flux, gas_constant):
    """Exact fractional-knapsack extrema on a box intersected by sum(f)=mass.

    This is a face-feasibility diagnostic, not a flux limiter. Using fsum avoids
    treating the LP solver's absolute feasibility tolerance as a species floor.
    """
    lower, upper, gas_constant = [np.asarray(v, dtype=float)
                                  for v in (lower, upper, gas_constant)]
    if (lower.shape != (5,) or upper.shape != (5,) or gas_constant.shape != (5,)
            or not np.isfinite([lower, upper, gas_constant]).all()
            or not np.isfinite(mass_flux) or np.any(lower > upper)):
        raise ValueError('invalid mass/gas flux interval')
    mass_bounds = [math.fsum(lower), math.fsum(upper)]
    if not mass_bounds[0] <= mass_flux <= mass_bounds[1]:
        return dict(mass_feasible=False, mass_bounds=mass_bounds, gas_bounds=None)
    # Start at the lower vertex; fill cheapest/most expensive species first.
    extrema = []
    for order in (np.argsort(gas_constant), np.argsort(-gas_constant)):
        flux = lower.copy()
        remaining = mass_flux-mass_bounds[0]
        for s in order:
            increment = min(remaining, upper[s]-lower[s])
            flux[s] += increment
            remaining = max(0., remaining-increment)
        extrema.append(math.fsum(flux*gas_constant))
    return dict(mass_feasible=True, mass_bounds=mass_bounds, gas_bounds=extrema)


def symmetric_face_probe(log, local_intervals, ranks):
    """Compare duplicate evaluations of the same oriented periodic x face."""
    if local_intervals < 1 or ranks < 1:
        raise ValueError('invalid periodic face topology')
    groups = {}
    seen_ranks = set()
    logs = [log] if isinstance(log, Path) else log
    for line in '\n'.join(p.read_text() for p in logs).splitlines():
        if not line.startswith('AIR5_SYMMETRIC_FACE '):
            continue
        tokens = line.split()[1:]
        if len(tokens) != 61:
            raise ValueError('incomplete symmetric face probe record')
        step, stage, rank, i, side = map(int, tokens[:5])
        values = np.array(list(map(float, tokens[5:])))
        if not np.isfinite(values).all() or rank not in range(ranks) or side not in (0, 1):
            raise ValueError('invalid symmetric face probe record')
        face = (rank*local_intervals+i+side-1) % (ranks*local_intervals)
        seen_ranks.add(rank)
        groups.setdefault((step, stage, face), []).append((rank, i, side, values))
    if not groups:
        raise ValueError('no symmetric face records')
    if seen_ranks != set(range(ranks)):
        raise ValueError('missing ranks in symmetric face records')
    report = []
    for (step, stage, face), rows in sorted(groups.items()):
        if len(rows) < 2:
            continue
        values = np.stack([r[3] for r in rows])
        spreads = np.ptp(values, axis=0)
        record = dict(step=step, stage=stage, global_face=face,
            evaluations=[list(r[:3]) for r in rows], beta=values[:, 0].tolist())
        for index, name in enumerate(('high', 'low', 'final', 'base_left', 'base_right')):
            section = slice(1+11*index, 12+11*index)
            record[name+'_max_abs_spread'] = spreads[section].tolist()
            record[name+'_values'] = values[:, section].tolist()
        report.append(record)
    return dict(scope='read-only duplicate face evaluation audit', comparisons=report)


def contact_face_feasibility(case, ranks, dt, thermo):
    """Audit first-stage periodic contact face budgets from recorded ASTR q.

    Reconstructed physical flux divergences must match the actual raw RHS.
    No projected flux is used to advance a solution or claim a physics pass.
    """
    from run_air5_layered_stress_gate import owned, snap
    from check_air5_c5_frozen_transport import read_rhs_snapshot
    from check_air5_mach4_error_replay import cartesian_jacobian

    if ranks not in (1, 2) or not np.isfinite(dt) or dt <= 0:
        raise ValueError('face audit requires one/two ranks and positive dt')
    full = owned(case, 'pre_rhs', 1, ranks)
    q = full[:, 0, 0, :]
    np.testing.assert_array_equal(full, np.broadcast_to(q[:, None, None, :], full.shape))
    rho, _, temperature, pressure, _ = primitive_metrics(q, thermo)
    velocity = q[:, 1:4]/rho[:, None]
    np.testing.assert_allclose(temperature, 4000., atol=1e-9, rtol=0)
    np.testing.assert_allclose(pressure, 1e5, atol=1e-7, rtol=0)
    np.testing.assert_allclose(velocity, np.broadcast_to([100., 0., 0.], velocity.shape),
                               atol=1e-10, rtol=0)
    # The fixture is a Cartesian periodic box of side 2*pi*1e-4 m. Refuse
    # other geometry rather than silently applying these metrics to SBLI.
    lengths = np.full(3, 2*np.pi*1e-4)
    spacing = lengths/np.array(full.shape[:3])
    jac = cartesian_jacobian(case)
    np.testing.assert_allclose(jac, np.prod(spacing), rtol=1e-13, atol=0)
    flux = q*velocity[:, :1]
    flux[:, 1] += pressure
    flux[:, 4] += pressure*velocity[:, 0]
    high = (37./60)*(flux+np.roll(flux, -1, axis=0))
    high -= (2./15)*(np.roll(flux, 1, axis=0)+np.roll(flux, -2, axis=0))
    high += (1./60)*(np.roll(flux, 2, axis=0)+np.roll(flux, -3, axis=0))
    gamma = 1.+(q[:, 5:10]@thermo.gas_constant)/(q[:, 5:10]@thermo.cv_tr)
    speed = abs(velocity[:, 0])+np.sqrt(gamma*pressure/rho)
    speed = np.maximum(speed, np.roll(speed, -1))
    low = .5*(flux+np.roll(flux, -1, axis=0))
    low -= .5*speed[:, None]*(np.roll(q, -1, axis=0)-q)
    fields = []
    for rank in range(ranks):
        s = read_rhs_snapshot(snap(case, 'conv_raw', 1, rank))
        im, jm, km, nq = s.header
        fields.append(s.values.reshape(im+1, jm+1, km+1, nq, order='F')[:-1, 0, 0]/jac)
    recorded = np.concatenate(fields)
    reconstructed = (np.roll(high, 1, axis=0)-high)/spacing[0]
    # Only the historical N2 remainder overwrite differs from the symmetric
    # reconstruction. Bound its rounding error by bulk-flux operands.
    scale = np.max(abs(flux), axis=0)/spacing[0]
    scale[2] = np.max(abs(pressure))/spacing[1]
    scale[3] = np.max(abs(pressure))/spacing[2]
    scale[5] += scale[0]+np.sum(scale[6:10])
    error = np.max(abs(recorded-reconstructed), axis=0)
    if np.any(error > 1024*np.finfo(float).eps*scale):
        raise ValueError(f'face reconstruction does not reproduce recorded central RHS: '
                         f'error={error.tolist()}, scale={scale.tolist()}')
    low_state = q+dt*(np.roll(low, 1, axis=0)-low)/spacing[0]
    if np.any(low_state[:, 5:10] < 0):
        raise ValueError('negative low-order species budget')
    delta = high[:, 5:10]-low[:, 5:10]
    minus = dt/spacing[0]*(np.minimum(np.roll(delta, 1, axis=0), 0.)
                            +np.minimum(-delta, 0.))
    ratio = np.ones_like(minus)
    np.divide(low_state[:, 5:10], -minus, out=ratio, where=minus < 0)
    ratio = np.minimum(ratio, 1.)
    # For a face oriented left -> right, positive corrections spend the left
    # cell's budget; negative corrections spend the right cell's budget.
    face_ratio = np.where(delta > 0., ratio, np.roll(ratio, -1, axis=0))
    endpoint = low[:, 5:10]+face_ratio*delta
    lower = np.minimum(low[:, 5:10], endpoint)
    upper = np.maximum(low[:, 5:10], endpoint)
    # Alternative sufficient six-face budget, without the extra requirement
    # that each species correction lie on its individual low/high segment.
    face_lower = low[:, 5:10]-np.roll(low_state[:, 5:10], -1, axis=0)*spacing[0]/(6*dt)
    face_upper = low[:, 5:10]+low_state[:, 5:10]*spacing[0]/(6*dt)
    rows = []
    for face in range(len(q)):
        interval = mass_gas_interval(lower[face], upper[face], high[face, 0], thermo.gas_constant)
        box = mass_gas_interval(face_lower[face], face_upper[face], high[face, 0], thermo.gas_constant)
        target = math.fsum(high[face, 5:10]*thermo.gas_constant)
        gas_scale = math.fsum(abs(high[face, 5:10]*thermo.gas_constant))
        margin = None if interval['gas_bounds'] is None else min(
            target-interval['gas_bounds'][0], interval['gas_bounds'][1]-target)
        rows.append(dict(face=face, **interval, gas_target=target,
            gas_margin=margin, gas_roundoff_bound=128*np.finfo(float).eps*gas_scale,
            candidate=high[face, 5:10].tolist(), lower=lower[face].tolist(),
            upper=upper[face].tolist(), species_ratio=face_ratio[face].tolist(),
            six_face_budget=box))
    return dict(scope='first-stage face feasibility only; no flow integration or acceptance',
        case=str(case), ranks=ranks, dt=dt, reconstructed_rhs_max_error=error.tolist(),
        mass_infeasible_faces=[r['face'] for r in rows if not r['mass_feasible']],
        gas_infeasible_beyond_roundoff=[r['face'] for r in rows
            if r['gas_margin'] is not None and r['gas_margin'] < -r['gas_roundoff_bound']],
        six_face_mass_infeasible=[r['face'] for r in rows
            if not r['six_face_budget']['mass_feasible']],
        six_face_gas_infeasible=[r['face'] for r in rows
            if r['six_face_budget']['gas_bounds'] is not None and not
            r['six_face_budget']['gas_bounds'][0]-r['gas_roundoff_bound'] <= r['gas_target'] <=
            r['six_face_budget']['gas_bounds'][1]+r['gas_roundoff_bound']],
        faces=rows)


def thermal_contributions(a, b, terms, thermo):
    """Exact finite-difference T,p identity, using fixed endpoint weights."""
    if a.shape != (11,) or b.shape != (11,) or not np.isfinite([a, b]).all():
        raise ValueError('invalid thermodynamic endpoint shape or values')
    if min(a[0], b[0], a[5:10]@thermo.cv_tr, b[5:10]@thermo.cv_tr) <= 0:
        raise ValueError('thermodynamic endpoint requires positive density and heat capacity')
    ta = float(primitive_metrics(a, thermo)[2])
    tb = float(primitive_metrics(b, thermo)[2])
    cvb = float(b[5:10]@thermo.cv_tr)
    rb = float(b[5:10]@thermo.gas_constant)
    ka = float(a[1:4]@a[1:4]/(2*a[0]))
    weights = np.zeros(11)
    weights[0] = ka/b[0]
    weights[1:4] = -(a[1:4]+b[1:4])/(2*b[0])
    weights[4], weights[10] = 1., -1.
    weights[5:10] = -thermo.formation_energy-ta*thermo.cv_tr
    values = {}
    for name, delta in terms.items():
        dt = float(weights@delta/cvb)
        dp = rb*dt+ta*float(delta[5:10]@thermo.gas_constant)
        values[name] = dict(temperature_k=dt, pressure_pa=dp)
    dp = float(primitive_metrics(b, thermo)[3]-primitive_metrics(a, thermo)[3])
    residual_t = tb-ta-sum(v['temperature_k'] for v in values.values())
    residual_p = dp-sum(v['pressure_pa'] for v in values.values())
    if abs(residual_t) > 1e-9 or abs(residual_p) > 1e-6:
        raise ValueError('thermodynamic endpoint identity does not close')
    return dict(temperature_difference_k=tb-ta, pressure_difference_pa=dp,
                terms=values, closure_temperature_k=residual_t, closure_pressure_pa=residual_p)


def endpoint_terms(a, b):
    result = {}
    for component, name in enumerate(COMPONENTS):
        delta = np.zeros(11)
        delta[component] = b[component]-a[component]
        result[name] = delta
    return result


def contact_counterfactuals(raw, limited, thermo):
    """Local EOS diagnostics only: not conservative fluxes or accepted states."""
    if raw.shape != limited.shape or raw.shape[-1] != 11:
        raise ValueError('contact state shape mismatch')
    if not np.isfinite([raw, limited]).all() or np.any(raw[..., 0] <= 0) or np.any(limited[..., 0] <= 0):
        raise ValueError('invalid contact states')
    ta, pa = primitive_metrics(raw, thermo)[2:4]
    cv = limited[..., 5:10]@thermo.cv_tr
    gas = limited[..., 5:10]@thermo.gas_constant
    if np.any(cv <= 0) or np.any(gas <= 0):
        raise ValueError('nonpositive contact heat capacity or gas density')

    def with_temperature(target):
        q = limited.copy()
        kinetic = np.sum(q[..., 1:4]**2, axis=-1)/(2*q[..., 0])
        q[..., 4] = kinetic+q[..., 10]+q[..., 5:10]@thermo.formation_energy+target*cv
        return q

    # Both keep species fixed. Restoring T and restoring p need different E
    # whenever the limiter changes sum(rho_s R_s).
    temperature_only = with_temperature(ta)
    pressure_only = with_temperature(pa/gas)
    selective = limited.copy()
    selective[..., 6] = raw[..., 6]
    selective[..., 5] = selective[..., 0]-selective[..., 6:10].sum(axis=-1)

    def metrics(q):
        temp, pressure = primitive_metrics(q, thermo)[2:4]
        return dict(max_temperature_difference_k=float(abs(temp-ta).max()),
                    max_pressure_difference_pa=float(abs(pressure-pa).max()),
                    min_species_density=float(q[..., 5:10].min()),
                    max_energy_change_j_m3=float(abs(q[..., 4]-limited[..., 4]).max()))

    prediction = ta*((limited[..., 5:10]-raw[..., 5:10])@thermo.gas_constant)
    actual = primitive_metrics(temperature_only, thermo)[3]-pa
    np.testing.assert_allclose(actual, prediction, atol=1e-7, rtol=1e-9)
    return dict(scope='pointwise counterfactuals, not a positivity or conservation proof',
        limiter_increment_max_abs=abs(limited-raw).reshape(-1, 11).max(axis=0).tolist(),
        original=metrics(limited), restore_temperature_only=metrics(temperature_only),
        restore_pressure_only=metrics(pressure_only),
        remove_o2_colimiting_with_n2_closure=metrics(selective),
        temperature_repair_pressure_identity_error_pa=float(abs(actual-prediction).max()))


def contact_case(case, ranks, dt, thermo):
    from run_air5_layered_stress_gate import owned, snap
    from check_air5_c5_frozen_transport import read_rhs_snapshot
    from check_air5_mach4_error_replay import cartesian_jacobian

    if ranks not in (1, 2) or not np.isfinite(dt) or dt <= 0:
        raise ValueError('contact diagnostic requires one/two ranks and positive dt')
    initial = owned(case, 'pre_rhs', 1, ranks)
    after = owned(case, 'post_update', 1, ranks)
    jac = cartesian_jacobian(case)
    states = []
    for label in ('conv_raw', 'conv'):
        fields = []
        for rank in range(ranks):
            s = read_rhs_snapshot(snap(case, label, 1, rank))
            im, jm, km, nq = s.header
            fields.append(s.values.reshape(im+1, jm+1, km+1, nq, order='F')[:-1, :-1, :-1])
        states.append(initial+dt*np.concatenate(fields, axis=0)/jac)
    np.testing.assert_allclose(states[1], after, atol=1e-9, rtol=1e-12)
    return dict(case=str(case), ranks=ranks, dt=dt,
                first_stage=contact_counterfactuals(*states, thermo))


def pair(coarse, fine, thermo, rk_budget=False):
    directories = (coarse, fine)
    contracts = [json.loads((p/'contract.json').read_text()) for p in directories]
    check_window_contract(*contracts)
    if contracts[0]['backend'] != contracts[1]['backend']:
        raise ValueError('paired backends differ')
    if rk_budget and [c['updates'] for c in contracts] != [1, 2]:
        raise ValueError('RK budget requires one step versus two half steps with all snapshots')
    for p in directories:
        if json.loads((p/'result.json').read_text())['status'] != 'bounded-replay-completed-not-physical-pass':
            raise ValueError('incomplete replay')
    report = dict(coarse=str(coarse), fine=str(fine), points=[], components=list(COMPONENTS),
                  scope='endpoint decomposition, not cumulative operator attribution')
    for rank in (0, 1):
        states, starts, coefficients = [], [], []
        for directory, c in zip(directories, contracts):
            case = directory/c['backend']
            last = c['start_step']+c['updates']-1
            states.append(_active_array(path(case, 'post_chemistry', 2, rank, last)))
            starts.append(_active_array(path(case, 'pre_chemistry', 1, rank, c['start_step'])))
            coefficients.append({label: [scalar(path(case, label, stage, rank, last))
                for stage in (1, 2, 3)] for label in (
                    'convection_fluid_ratio', 'convection_species_ratio',
                    'diffusion_species_ratio', 'diffusion_energy_ratio')})
        np.testing.assert_array_equal(*starts)
        a, b = states
        if a.shape != b.shape or not np.isfinite([a, b]).all():
            raise ValueError('invalid endpoint states')
        temp = [primitive_metrics(q, thermo)[2] for q in states]
        fields = {'rho_O2': abs(b[..., 6]-a[..., 6]), 'rho_NO': abs(b[..., 9]-a[..., 9]),
                  'temperature': abs(temp[1]-temp[0])}
        for field, difference in fields.items():
            node = tuple(map(int, np.unravel_index(difference.argmax(), difference.shape)))
            qa, qb = a[node], b[node]
            coeff = []
            for group in coefficients:
                coeff.append({label: [float(v[node]) for v in stages] for label, stages in group.items()})
            entry = dict(rank=rank, field=field, local_ijk=node, max_abs=float(difference[node]),
                         endpoints=[qa.tolist(), qb.tolist()], coefficients=coeff,
                         thermal=thermal_contributions(qa, qb, endpoint_terms(qa, qb), thermo))
            if rk_budget:
                _, a0, ae, at, _ = replay_trajectory(coarse, rank, node, tuple(range(11)))
                _, b0, be, bt, _ = replay_trajectory(fine, rank, node, tuple(range(11)))
                np.testing.assert_array_equal(a0, b0)
                delta = {name: bt[name]-at[name] for name in at}
                residual = be-ae-sum(delta.values())
                scale = abs(ae)+abs(be)+sum(abs(d) for d in delta.values())
                if np.any(abs(residual) > 1024*np.finfo(float).eps*scale):
                    raise ValueError('component RK difference budget does not close')
                entry['rk_delta_q_terms'] = {k: v.tolist() for k, v in delta.items()}
                entry['rk_thermal'] = thermal_contributions(ae, be, delta, thermo)
                entry['rk_closure'] = residual.tolist()
            report['points'].append(entry)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coarse', type=Path)
    parser.add_argument('--fine', type=Path)
    parser.add_argument('--contact-case', type=Path)
    parser.add_argument('--contact-feasibility', action='store_true')
    parser.add_argument('--symmetric-face-log', type=Path, nargs='+')
    parser.add_argument('--local-intervals', type=int)
    parser.add_argument('--np', type=int, default=2)
    parser.add_argument('--dt', type=float, default=5e-9)
    parser.add_argument('--rk-budget', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    thermo = Air5RadauReference(Path(__file__).resolve().parents[2]/'chemMech/air5_kimjo12.json')
    if args.symmetric_face_log:
        if args.contact_case or args.coarse or args.fine or args.local_intervals is None:
            parser.error('face log needs --local-intervals and is exclusive of state comparisons')
        result = symmetric_face_probe(args.symmetric_face_log, args.local_intervals, args.np)
    elif args.contact_case:
        if args.coarse or args.fine or args.rk_budget:
            parser.error('contact and paired replay modes are mutually exclusive')
        reader = contact_face_feasibility if args.contact_feasibility else contact_case
        result = reader(args.contact_case, args.np, args.dt, thermo)
    else:
        if args.contact_feasibility:
            parser.error('--contact-feasibility requires --contact-case')
        if not args.coarse or not args.fine:
            parser.error('paired replay requires --coarse and --fine')
        result = pair(args.coarse, args.fine, thermo, args.rk_budget)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    if args.contact_case:
        print(json.dumps({k: v for k, v in result.items() if k != 'faces'},
                         indent=2, allow_nan=False))
    if args.symmetric_face_log:
        for row in result['comparisons']:
            print(row['stage'], row['global_face'], row['beta'],
                  'final max spread', max(row['final_max_abs_spread']))
    for point in result.get('points', []):
        print(point['rank'], point['field'], point['local_ijk'], point['max_abs'],
              point.get('rk_thermal', point['thermal']))
