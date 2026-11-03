"""One-dimensional thickness diagnostics on real completed-step mean profiles."""
from copy import deepcopy
import json
import os
from pathlib import Path
import re

import h5py
import numpy as np
import pytest

from run_output_restart_validation import compare_fields
from test_insitu_station_profiles import read_profile
from test_insitu_wall_transport_runtime import run_wall_fixture, wall_mean_definitions

pytestmark = pytest.mark.skipif(not os.environ.get('ASTR_SELECTED_RUNTIME_EXE') or
                               not os.environ.get('ASTR_SELECTED_RUNTIME_M12_SOURCE'),
                               reason='Select the root-CMake M12 build and read-only inputs')
TOPOLOGIES = ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 1), (2, 1, 2), (1, 2, 2))


def definitions():
    base = dict(id='native', kind='station_profile', source='favre', statistics_source='selected-time',
                enable_request=1, fields=['velocity_x', 'pressure', 'temperature'],
                sampling=dict(mode='station_span', axis='i', index=32,
                              index_ranges=dict(j=[0, 32], k=[0, 24])),
                output=dict(channel='data', format='csv'), schedule=dict(mode='steps', interval=3))
    records = [base, dict(deepcopy(base), id='reynolds', source='reynolds')]
    for name, options in [('delta99', dict(edge_band=[24, 28])),
                          ('both', dict(edge_band=[24, 28], explicit_upper=21.234)),
                          ('single', dict(edge_band=[28, 28], explicit_upper=21.234)),
                          ('bad-upper', dict(edge_band=[24, 28], explicit_upper=1000.)),
                          ('no-edge', dict(edge_band=[0, 0]))]:
        records.append(dict(deepcopy(base), id=name, thickness=options))
    return records


def latest(case, name):
    return next((case/'outdat/new/profile_products'/name).glob('*/segment*/frame.step000000000010.csv'))


def metadata_values(text):
    values = {}
    for line in text.splitlines():
        if not line.startswith('# thickness_') or '=' not in line:
            continue
        name, value = line[12:].split('=', 1)
        try:
            values[name] = float(value)
        except ValueError:
            values[name] = value
    return values


def integral(y, value, upper):
    kept = y < upper
    coordinates = np.r_[y[kept], upper]
    samples = np.r_[value[kept], np.interp(upper, y, value)]
    return np.trapezoid(samples, coordinates)


def check(case, records, record_property, expected_histories=1):
    native, _ = read_profile(latest(case, 'native'))
    reynolds, _ = read_profile(latest(case, 'reynolds'))
    y, rho, u = (native[name] for name in ('y', 'rho_time_mean', 'velocity_x'))
    worst = absolute = 0.

    def compare(actual, expected):
        nonlocal worst, absolute
        a, b = np.asarray(actual), np.asarray(expected)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        error = np.abs(a-b)
        scaled = error/np.maximum(1., np.maximum(np.abs(a), np.abs(b)))
        worst, absolute = max(worst, float(scaled.max())), max(absolute, float(error.max()))
        assert scaled.max() <= 2e-10, (scaled.max(), error.max())

    for item in records[2:]:
        if item['id'] == 'no-edge':
            status = json.loads((case/'outdat/new/products.status.json').read_text())
            rejected = next(p for p in status['products'] if p['id'] == item['id'])
            assert rejected['state'] == 'invalid-configuration', rejected
            assert 'above the wall' in rejected['reason'], rejected
            assert not list((case/'outdat/new/profile_products'/item['id']).glob('*/segment*/frame.step*.csv'))
            continue
        actual, text = read_profile(latest(case, item['id']))
        info = metadata_values(text)
        for field in native.dtype.names:
            np.testing.assert_array_equal(actual[field], native[field])
        assert 'p/T are Reynolds means' in text
        assert 'no extrapolation' in text
        first, last = item['thickness']['edge_band']
        indices = slice(first, last+1)
        if first == last:
            edge = [rho[first], u[first], reynolds['temperature'][first], reynolds['pressure'][first]]
        else:
            length = y[last]-y[first]
            rho_e = np.trapezoid(rho[indices], y[indices])/length
            edge = [rho_e, np.trapezoid((rho*u)[indices], y[indices])/(rho_e*length),
                    np.trapezoid(reynolds['temperature'][indices], y[indices])/length,
                    np.trapezoid(reynolds['pressure'][indices], y[indices])/length]
        if edge[1] <= 0:
            assert 'rho_edge' not in info and 'unavailable' in info
            assert np.all(actual['thickness_integrals_available'] == 0)
            assert np.isnan(actual['cumulative_displacement']).all()
            assert np.isnan(actual['cumulative_momentum']).all()
            continue
        rho_e, u_e, t_e, p_e = edge
        compare([info[k] for k in ('rho_edge', 'U_edge', 'T_edge', 'p_edge')], edge)
        sutherland = 110.3/594.3
        mu_e = t_e*np.sqrt(t_e)*(1+sutherland)/(t_e+sutherland)/1.e4
        compare(info['mu_edge'], mu_e)
        assert 'not mean instantaneous viscosity' in text
        fraction = rho*u/(rho_e*u_e)
        d, theta = 1-fraction, fraction*(1-u/u_e)
        for field, value in [('cumulative_displacement', d), ('cumulative_momentum', theta)]:
            compare(actual[field], np.r_[0., np.cumsum(.5*np.diff(y)*(value[1:]+value[:-1]))])
        assert np.all(actual['thickness_integrals_available'] == 1)
        above = u[:last+1] >= .99*u_e
        components = np.count_nonzero(above & ~np.r_[False, above[:-1]])
        assert info['threshold_components'] == components
        left = first
        if np.all(above[first:]):
            while left > 0 and above[left-1]:
                left -= 1
        if np.all(above[first:]) and left > 0:
            upper = y[left-1]+(y[left]-y[left-1])*(.99*u_e-u[left-1])/(u[left]-u[left-1])
            delta_star, momentum = integral(y, d, upper), integral(y, theta, upper)
            compare([info[k] for k in ('delta99', 'displacement99', 'momentum99', 'H99', 'Re_theta99')],
                    [upper-y[0], delta_star, momentum, delta_star/momentum, rho_e*u_e*momentum/mu_e])
        else:
            assert 'delta99' not in info and 'thickness_delta99_unavailable=' in text
        upper = item['thickness'].get('explicit_upper')
        if upper is not None and y[0] <= upper <= y[-1]:
            delta_star, momentum = integral(y, d, upper), integral(y, theta, upper)
            compare([info[k] for k in ('explicit_upper', 'displacement_explicit', 'momentum_explicit',
                                       'H_explicit', 'Re_theta_explicit')],
                    [upper, delta_star, momentum, delta_star/momentum, rho_e*u_e*momentum/mu_e])
        elif upper is not None:
            assert 'explicit_upper' not in info and 'thickness_explicit_unavailable=' in text
    histories = re.findall(r'ASTR_INSITU_SELECTED rank=\d+ histories=(\d+)', (case/'run.log').read_text())
    assert histories and set(histories) == {str(expected_histories)}, 'thickness must reuse the native selected history'
    record_property('max_scaled_error', worst)
    record_property('max_absolute_error', absolute)
    record_property('case_directory', str(case))


@pytest.mark.parametrize('backend', ('cpu', 'gpu'))
@pytest.mark.parametrize('topology', TOPOLOGIES)
def test_real_thickness_profiles(tmp_path, backend, topology, record_property):
    case, records = tmp_path/backend, definitions()
    run_wall_fixture(case, backend, topology, records, statistics=True, checkpoint=False)
    check(case, records, record_property)
    assert not list((case/'outdat/new/checkpoints').glob('step*'))


@pytest.mark.parametrize('backend', ('cpu', 'gpu'))
@pytest.mark.parametrize('grid_source', ('generated', 'read'))
def test_thickness_exact_restore_and_output_invariance(tmp_path, backend, grid_source, record_property):
    exact_restore_fixture(tmp_path, backend, grid_source, (1, 1, 2), record_property)


@pytest.mark.parametrize('backend', ('cpu', 'gpu'))
@pytest.mark.parametrize('topology', [t for t in TOPOLOGIES if t != (1, 1, 2)])
def test_read_cartesian_random_restart_topologies(tmp_path, backend, topology, record_property):
    exact_restore_fixture(tmp_path, backend, 'read', topology, record_property)


@pytest.mark.parametrize('backend', ('cpu', 'gpu'))
@pytest.mark.parametrize('geometry', ('nonseparable', 'descending_axes'))
def test_read_random_checkpoint_does_not_admit_curve(tmp_path, backend, geometry):
    source = os.environ.get('ASTR_SELECTED_RUNTIME_CARTESIAN_GRID')
    if not source:
        pytest.skip('Select the immutable Cartesian 64x32x24 grid')
    grid = tmp_path/'unsupported.h5'
    with h5py.File(source,'r') as original, h5py.File(grid,'x') as changed:
        for key in ('x','y','z'):
            values = original[key][...]
            if geometry == 'nonseparable' and key == 'x':
                values = values+1.e-6*original['y'][...]
            elif geometry == 'descending_axes' and key in ('x','z'):
                values = values.max()-values
            changed[key] = values
    message = ('finite axis-separable Cartesian coordinates' if geometry == 'nonseparable' else
               'increasing Cartesian axes agreeing at MPI shared nodes')
    run_wall_fixture(tmp_path/backend, backend, (1,1,2), definitions(), statistics=True,
                     read_grid=grid, reject=message)


@pytest.mark.parametrize('backend', ('cpu', 'gpu'))
def test_read_random_checkpoint_without_statistics(tmp_path, backend):
    grid = os.environ.get('ASTR_SELECTED_RUNTIME_CARTESIAN_GRID')
    if not grid:
        pytest.skip('Select the immutable Cartesian 64x32x24 grid')
    with_statistics, without_statistics = tmp_path/'with', tmp_path/'without'
    run_wall_fixture(with_statistics, backend, (1,1,2), definitions(), statistics=True, read_grid=grid)
    run_wall_fixture(without_statistics, backend, (1,1,2), [], statistics=False, read_grid=grid)
    final = Path('outdat/new/checkpoints/step000000000010')
    compare_fields(with_statistics/final/'state.h5', without_statistics/final/'state.h5')
    assert not (without_statistics/final/'statistics.h5').exists()
    assert (without_statistics/final/'control.bin').read_bytes().startswith(b'ASTROC09')
    log = (without_statistics/'run.log').read_text()
    assert log.count('ASTR_OUTPUT_PROFILE_GRID rank=') == 2
    assert 'ASTR_SELECTED_GEOMETRY' not in log, 'checkpoint proof must not depend on selected statistics'


def exact_restore_fixture(tmp_path, backend, grid_source, topology, record_property):
    records = definitions()
    grid = None
    if grid_source == 'read':
        grid = os.environ.get('ASTR_SELECTED_RUNTIME_CARTESIAN_GRID')
        if not grid:
            pytest.skip('Select the immutable Cartesian 64x32x24 grid')
    full, first, tail, raw = (tmp_path/name for name in ('full', 'prefix', 'tail', 'raw'))
    run_wall_fixture(full, backend, topology, records, statistics=True, read_grid=grid)
    run_wall_fixture(first, backend, topology, records, statistics=True, steps=5, read_grid=grid)
    changed = deepcopy(records)
    for item in changed[2:]:
        item['thickness'] = dict(edge_band=[1, 1], explicit_upper=999.)
    run_wall_fixture(tail, backend, topology, changed, statistics=True,
                     restore=first/'outdat/new/checkpoints/step000000000005', read_grid=grid,
                     omit_restart_inputs=grid is not None)
    run_wall_fixture(raw, backend, topology, records[:2], statistics=True, read_grid=grid)
    for name in ('state.h5', 'statistics.h5'):
        reference = Path('outdat/new/checkpoints/step000000000010')/name
        compare_fields(full/reference, tail/reference)
        compare_fields(full/reference, raw/reference)
    control = Path('outdat/new/checkpoints/step000000000010/control.bin')
    assert (full/control).read_bytes() == (tail/control).read_bytes(), 'RNG/control continuation differs'
    assert (full/control).read_bytes() == (raw/control).read_bytes(), 'thickness changed RNG/control'
    for item in records:
        if item['id'] == 'no-edge':
            continue
        source = latest(full, item['id']).parent
        restored = latest(tail, item['id']).parent
        for frame in restored.glob('frame.step*.csv'):
            assert frame.read_bytes() == (source/frame.name).read_bytes()
    if grid is not None:
        for case in (full, first, tail, raw):
            assert len(re.findall(r'ASTR_SELECTED_GEOMETRY rank=\d+ cartesian=T', (case/'run.log').read_text())) == np.prod(topology)
            assert len(re.findall(r'ASTR_OUTPUT_PROFILE_GRID rank=\d+ cartesian=T', (case/'run.log').read_text())) == np.prod(topology)
            resources = case/'outdat/new/resources'
            assert (resources/'grid.h5').read_bytes() == Path(grid).read_bytes()
            assert not (resources/'grid.2d').exists()
            assert (resources/'wallbs.dat').is_file()
    else:
        for case in (full, first, tail, raw):
            resources = case/'outdat/new/resources'
            assert (resources/'grid.2d').read_bytes() == (case/'datin/grid.2d').read_bytes()
            assert not (resources/'grid.h5').exists()
            assert 'ASTR_OUTPUT_PROFILE_GRID' not in (case/'run.log').read_text()
    check(full, records, record_property)


def test_thickness_composes_with_wall_units(tmp_path, record_property):
    case, records = tmp_path/'gpu', definitions()
    wall = wall_mean_definitions()[1]
    wall.update(id='wall-reference', wall_reference={})
    records[3]['wall_units'] = wall['id']
    run_wall_fixture(case, 'gpu', (1, 1, 2), [*records, wall], statistics=True, checkpoint=False)
    check(case, records, record_property, expected_histories=2)
    actual, _ = read_profile(latest(case, 'both'))
    path = next((case/'outdat/new/wall_profile_products/wall-reference').glob('*/segment*/frame.step000000000010.csv'))
    values, _ = read_profile(path)
    w = values[values['global_i'] == 32][0]
    utau = np.sqrt(np.hypot(w['wall_shear_x'], w['wall_shear_z'])/w['rho_time_mean'])
    length = w['dynamic_viscosity']/w['rho_time_mean']/utau
    assert np.all(actual['wall_scale_available'] == 1)
    for name, expected in [('y_plus', (actual['y']-actual['y'][0])/length),
                            ('velocity_x_plus', actual['velocity_x']/utau)]:
        error = np.abs(actual[name]-expected)/np.maximum(1., np.maximum(np.abs(actual[name]), np.abs(expected)))
        assert error.max() <= 2e-10
