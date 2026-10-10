#!/usr/bin/env python3
"""Read-only M12 boundary sensitivity versus compressible laminar similarity."""

import json
from pathlib import Path

import h5py
import numpy as np
from scipy.integrate import cumulative_trapezoid

from analyze_curvilinear_hbl_physics import finite_difference_weights
from generate_compressible_blasius_profile import solve_profile, sutherland_ratio


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'tests/gpu_validation/out/m12_joint_nscbc_20261010/medium'
SOURCE = Path('/home/dell/workspace/M12_C13/datin')
# Adjust the output location here, not in the solver or the original case.
OUTPUT = RUN / 'blasius_comparison'
MACH, REYNOLDS, REFERENCE_T, WALL_T = 12., 1e4, 594.3, 10.
PRANDTL, GAMMA = .72, 1.4
KINDS = ('extrapolation', 'nscbc')
NAMES = ('velocity_x', 'temperature', 'density')
COLORS = {'extrapolation': '#c62828', 'nscbc': '#1565c0'}
LABELS = {'extrapolation': 'Original BC 21/50', 'nscbc': 'NSCBC 22/52'}


def crossing_height(y, velocity, threshold=.99):
    y, velocity = np.asarray(y), np.asarray(velocity)
    if (y.ndim != 1 or y.shape != velocity.shape or not np.isfinite(y).all()
            or not np.isfinite(velocity).all() or np.any(np.diff(y) <= 0)):
        raise ValueError('invalid profile coordinates or values')
    crossings = np.flatnonzero((velocity[:-1] < threshold) & (velocity[1:] >= threshold))
    if not crossings.size:
        raise ValueError('profile does not bracket the requested edge velocity')
    j = crossings[0]
    return float(y[j] + (y[j+1]-y[j])*(threshold-velocity[j])/(velocity[j+1]-velocity[j]))


def inlet_station(y, velocity, solution, reynolds):
    delta = crossing_height(y, velocity)
    height99 = crossing_height(solution['height'], solution['velocity'])
    return .5*reynolds*(delta/height99)**2, delta


def span_average(values, z):
    if values.shape[0] != len(z) or np.any(np.diff(z) <= 0):
        raise ValueError('invalid spanwise coordinates')
    # Trapezoidal endpoint weights integrate one physical periodic interval.
    return np.trapezoid(values, z, axis=0)/(z[-1]-z[0])


def derivative(values, coordinate, stencil):
    result = np.empty_like(values, dtype=float)
    for i in range(len(coordinate)):
        start = min(max(i-stencil//2, 0), len(coordinate)-stencil)
        offsets = coordinate[start:start+stencil]-coordinate[i]
        scale = np.max(abs(offsets))
        weights = finite_difference_weights(offsets/scale, 0.)/scale
        result[i] = weights @ values[start:start+stencil]
    return result


def wall_quantities(x, y, fields, stencil=7):
    weights = finite_difference_weights(y[:stencil]/y[stencil-1], 0.)/y[stencil-1]
    du_dy = weights @ fields['velocity_x'][:stencil]
    dt_dy = weights @ fields['temperature'][:stencil]
    dv_dx = derivative(fields['velocity_y'][0], x, stencil)
    viscosity = sutherland_ratio(fields['temperature'][0], REFERENCE_T)/REYNOLDS
    return {'cf': 2*viscosity*(du_dy+dv_dx),
            'qw': viscosity*dt_dy/(PRANDTL*(GAMMA-1)*MACH**2),
            'du_dy': du_dy, 'dv_dx': dv_dx, 'dT_dy': dt_dy}


def reference_solution(points=8001, eta_max=20.):
    eta, f, u, du, temperature = solve_profile(MACH, REFERENCE_T, WALL_T, eta_max, points)
    return dict(eta=eta, streamfunction=f, velocity=u, velocity_eta=du,
                temperature=temperature,
                height=cumulative_trapezoid(temperature, eta, initial=0.))


def reference_at(solution, y, station):
    if np.any(np.asarray(station) <= 0):
        raise ValueError('reference stations must be downstream of the virtual leading edge')
    coordinate = np.asarray(y)/np.sqrt(2*np.asarray(station)/REYNOLDS)
    temperature = np.interp(coordinate, solution['height'], solution['temperature'])
    return {'velocity_x': np.interp(coordinate, solution['height'], solution['velocity']),
            'temperature': temperature, 'density': 1/temperature}


def reference_wall(solution, stations):
    scale = WALL_T*np.sqrt(2*stations/REYNOLDS)
    weights = finite_difference_weights(solution['eta'][:7], 0.)
    dt = weights @ solution['temperature'][:7]
    viscosity = float(sutherland_ratio(np.array([WALL_T]), REFERENCE_T)[0])/REYNOLDS
    return {'cf': 2*viscosity*solution['velocity_eta'][0]/scale,
            'qw': viscosity*dt/scale/(PRANDTL*(GAMMA-1)*MACH**2)}


def relative_profile_deviation(y, values, reference, limit):
    inside = y < limit
    coordinate = np.append(y[inside], limit)
    a = np.append(values[inside], np.interp(limit, y, values))
    b = np.append(reference[inside], np.interp(limit, y, reference))
    return float(np.sqrt(np.trapezoid((a-b)**2, coordinate)/np.trapezoid(b*b, coordinate)))


def read_fields(case, z, expected_shape):
    path = case/'outdat/new/fields/segment00000000/step000000002000/data.h5'
    fields, spread = {}, {}
    with h5py.File(path) as archive:
        if archive.attrs['phase'] != b'completed_step':
            raise ValueError('field is not a complete RK state')
        identity = archive['metadata'][...]
        for name in (*NAMES, 'velocity_y', 'velocity_z', 'pressure'):
            values = archive[name][...]
            if values.shape != expected_shape or not np.isfinite(values).all():
                raise ValueError(f'invalid field: {path}/{name}')
            if name in ('density', 'temperature', 'pressure') and values.min() <= 0:
                raise ValueError(f'non-positive field: {path}/{name}')
            fields[name] = span_average(values, z)
            spread[name] = np.ptp(values, axis=0)
    return fields, spread, identity


def save_figure(fig, stem):
    for suffix in ('eps', 'jpeg'):
        fig.savefig(OUTPUT/f'{stem}.{suffix}', dpi=300, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_profiles(x, y, indices, fields, solution, offset, stem):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(indices), 3, figsize=(12., 3.3*len(indices)),
                             squeeze=False, constrained_layout=True)
    xlabels = (r'$\langle u\rangle_z/U_\infty$', r'$\langle T\rangle_z/T_\infty$',
               r'$\langle\rho\rangle_z/\rho_\infty$')
    for row, i in enumerate(indices):
        delta = crossing_height(solution['height'], solution['velocity'])*np.sqrt(2*(x[i]+offset)/REYNOLDS)
        ref = reference_at(solution, y, x[i]+offset)
        for column, name in enumerate(NAMES):
            ax = axes[row, column]
            for kind in KINDS:
                ax.plot(fields[kind][name][:, i], y/delta, color=COLORS[kind],
                        linestyle='-' if kind == 'extrapolation' else '--',
                        linewidth=1.8, label=LABELS[kind])
            ax.plot(ref[name], y/delta, color='black', linestyle='-.', linewidth=1.5,
                    label='Compressible similarity')
            ax.set_ylim(0, 1.5)
            if name == 'temperature':
                ax.set_xlim(0, 14)
            else:
                ax.set_xlim(0, 1.15)
            ax.set_xlabel(xlabels[column])
            if column == 0:
                ax.set_ylabel(r'$y/\delta_{99,\mathrm{ref}}$')
                ax.text(.05, .92, f'$x={x[i]:.2f}$', transform=ax.transAxes, fontsize=14)
    axes[0, 1].legend(frameon=False, loc='upper right', fontsize=12)
    save_figure(fig, stem)


def make_figures(x, y, inlet_y, inlet, fields, wall, initial_wall, refwall, solution, offset, indices, delta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401
    plt.style.use(['science', 'ieee', 'std-colors'])
    plt.rcParams['axes.grid'] = False
    plt.rcParams['grid.alpha'] = 0.
    plt.rcParams.update({'axes.labelsize': 16, 'xtick.labelsize': 14,
                         'ytick.labelsize': 14, 'legend.fontsize': 14})
    fig, axes = plt.subplots(1, 2, figsize=(9., 4.), constrained_layout=True)
    ref = reference_at(solution, inlet_y, offset)
    for ax, column, name, label in zip(axes, (1, 3), NAMES[:2],
                                      (r'$u/U_\infty$', r'$T/T_\infty$')):
        ax.plot(inlet[:, column], inlet_y/delta, color=COLORS['extrapolation'], linewidth=1.8,
                label='Original inlet')
        ax.plot(ref[name], inlet_y/delta, 'k-.', linewidth=1.5, label='Compressible similarity')
        ax.set_xlabel(label)
        ax.set_ylabel(r'$y/\delta_{99,\mathrm{in}}$')
        ax.set_ylim(0, 1.5)
    axes[1].legend(frameon=False)
    save_figure(fig, 'inlet_reference')
    plot_profiles(x, y, indices[:-1], fields, solution, offset, 'interior_profiles')
    plot_profiles(x, y, indices[-1:], fields, solution, offset, 'outlet_profiles')
    fig, axes = plt.subplots(2, 1, figsize=(10., 7.), sharex=True, constrained_layout=True)
    visible = x >= 270
    for ax, name, label in zip(axes, ('cf', 'qw'), (r'$C_f$', r'$q_w/(\rho_\infty U_\infty^3)$')):
        ax.axvspan(x[-51], x[-1], facecolor='#eeeeee', edgecolor='none')
        for kind in KINDS:
            ax.plot(x[visible], wall[kind][name][visible], color=COLORS[kind], linewidth=1.7,
                    linestyle='-' if kind == 'extrapolation' else '--', label=LABELS[kind])
        ax.plot(x[visible], initial_wall[name][visible], color='#777777', linestyle=':', linewidth=1.4,
                label='Original initial field')
        ax.plot(x[visible], refwall[name][visible], 'k-.', linewidth=1.5, label='Compressible similarity')
        ax.set_xlim(270, x[-1])
        ax.set_ylabel(label)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
        ax.text(.88, .94, 'Sponge', transform=ax.transAxes, fontsize=14)
    axes[0].legend(frameon=False, ncol=2, loc='upper left')
    axes[-1].set_xlabel(r'$x/L_{\mathrm{ref}}$')
    save_figure(fig, 'wall_quantities')


def main():
    if OUTPUT.exists():
        raise FileExistsError(f'choose a new output directory in this script: {OUTPUT}')
    with h5py.File(RUN/'extrapolation/datin/grid.2d') as archive:
        x, y = archive['x'][0, :], archive['y'][:, 0]
        if not (np.array_equal(archive['x'][...], np.broadcast_to(x, (len(y), len(x))))
                and np.array_equal(archive['y'][...], np.broadcast_to(y[:, None], (len(y), len(x))))):
            raise ValueError('M12 comparison requires separable Cartesian coordinates')
    with h5py.File(RUN/'extrapolation/outdat/new/slices/resources/data.h5') as archive:
        z = archive['j000000000000/coordinates'][:, 0, 2]
    with h5py.File(SOURCE/'grid.2d') as archive:
        inlet_y = archive['y'][:, 0]
    inlet = np.loadtxt(SOURCE/'inlet.prof', skiprows=4)
    if inlet.shape != (len(inlet_y), 4) or not np.isfinite(inlet).all():
        raise ValueError('unexpected original inlet format')
    solution, coarse, extended = reference_solution(), reference_solution(4001), reference_solution(12001, 30.)
    offset, delta = inlet_station(inlet_y, inlet[:, 1], solution, REYNOLDS)
    indices = [int(np.argmin(abs(x-target))) for target in (300., 600., 800., 1100.)]
    fields, spreads, walls, identities = {}, {}, {}, []
    for kind in KINDS:
        fields[kind], spreads[kind], identity = read_fields(RUN/kind, z, (len(z), len(y), len(x)))
        walls[kind] = wall_quantities(x, y, fields[kind])
        identities.append(identity)
    if not np.array_equal(*identities) or identities[0][1] != 2000:
        raise ValueError('unmatched final-field identities')
    with h5py.File(RUN/'extrapolation/datin/flowini2d.h5') as archive:
        initial = {name: archive[key][...] for name, key in
                   (('velocity_x', 'u1'), ('velocity_y', 'u2'), ('temperature', 't'), ('density', 'ro'))}
    initial_wall = wall_quantities(x, y, initial)
    refwall = reference_wall(solution, x+offset)
    inlet_ref = reference_at(solution, inlet_y, offset)
    reference_change = {}
    for name in NAMES:
        differences = [abs(reference_at(solution, y, x[i]+offset)[name]
                           - reference_at(coarse, y, x[i]+offset)[name]).max() for i in indices]
        reference_change[name] = float(max(differences))
    report = dict(scope='Transient similarity consistency and boundary sensitivity; not numerical error or steady validation',
        parameters=dict(mach=MACH, reynolds=REYNOLDS, reference_temperature_K=REFERENCE_T,
                        wall_temperature_ratio=WALL_T, prandtl=PRANDTL, gamma=GAMMA, sutherland_K=110.3),
        completed_step=2000, time=float(identities[0][2:3].view(np.float64)[0]),
        original_inlet_delta99=delta, inlet_effective_distance=offset, virtual_leading_edge=-offset,
        reference_checks=dict(points=8001, eta_max=20., coarse_points=4001,
            max_mapped_change_from_coarse=reference_change,
            eta_extension_max_temperature_change=float(abs(extended['temperature'][:8001]-solution['temperature']).max())),
        inlet_max_abs_deviation=dict(velocity=float(abs(inlet[:, 1]-inlet_ref['velocity_x']).max()),
                                    temperature=float(abs(inlet[:, 3]-inlet_ref['temperature']).max())),
        wall_derivative='Independent physical-coordinate seven-node polynomial differentiation, not the solver boundary closure',
        span_average='Instantaneous physical-length trapezoidal average, not time or ensemble average',
        source=str(SOURCE), runs={kind: str(RUN/kind) for kind in KINDS}, stations=[])
    OUTPUT.mkdir()
    columns, data = ['x', 'Cf_reference', 'qw_reference', 'Cf_initial', 'qw_initial'], [x, refwall['cf'], refwall['qw'], initial_wall['cf'], initial_wall['qw']]
    for kind in KINDS:
        lower = wall_quantities(x, y, fields[kind], stencil=5)
        for name in ('cf', 'qw'):
            columns += [f'{name}_{kind}', f'{name}_{kind}_five_node']
            data += [walls[kind][name], lower[name]]
    np.savetxt(OUTPUT/'wall_quantities.csv', np.column_stack(data), delimiter=',', header=','.join(columns), comments='')
    for i in indices:
        station = dict(global_i=i, x=float(x[i]), zone='outlet/sponge' if i == len(x)-1 else 'interior diagnostic', cases={})
        ref = reference_at(solution, y, x[i]+offset)
        refdelta = crossing_height(solution['height'], solution['velocity'])*np.sqrt(2*(x[i]+offset)/REYNOLDS)
        station['delta99_reference'] = float(refdelta)
        station['initial_profile_relative_deviation'] = {
            name: relative_profile_deviation(y, initial[name][:, i], ref[name], refdelta) for name in NAMES}
        station['initial_wall_relative_deviation'] = {
            name: float(initial_wall[name][i]/refwall[name][i]-1) for name in ('cf', 'qw')}
        columns, data = ['y'], [y]
        for name in NAMES:
            columns.append(name+'_reference')
            data.append(ref[name])
            for kind in KINDS:
                columns.append(name+'_'+kind)
                data.append(fields[kind][name][:, i])
        np.savetxt(OUTPUT/f'profile_i{i:04d}.csv', np.column_stack(data), delimiter=',', header=','.join(columns), comments='')
        for kind in KINDS:
            near = y <= 1.5*refdelta
            lower = wall_quantities(x, y, fields[kind], stencil=5)
            station['cases'][kind] = dict(
                profile_relative_deviation={name: relative_profile_deviation(y, fields[kind][name][:, i], ref[name], refdelta) for name in NAMES},
                wall_relative_deviation={name: float(walls[kind][name][i]/refwall[name][i]-1) for name in ('cf', 'qw')},
                wall_five_vs_seven_node_relative_change={name: float(lower[name][i]/walls[kind][name][i]-1) for name in ('cf', 'qw')},
                near_wall_max_span_u_range=float(spreads[kind]['velocity_x'][near, i].max()),
                near_wall_max_pressure_deviation_from_freestream=float(abs(fields[kind]['pressure'][near, i]*(GAMMA*MACH**2)-1).max()))
        report['stations'].append(station)
    (OUTPUT/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    make_figures(x, y, inlet_y, inlet, fields, walls, initial_wall, refwall, solution, offset, indices, delta)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
