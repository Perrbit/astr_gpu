"""Wall reference products from real complete-step moments; no new CFD model."""
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
from test_insitu_wall_transport_runtime import run_wall_fixture, wall_mean_definitions, check_wall_means

pytestmark = pytest.mark.skipif(not os.environ.get('ASTR_SELECTED_RUNTIME_EXE') or
                               not os.environ.get('ASTR_SELECTED_RUNTIME_M12_SOURCE'),
                               reason='Select the existing root build and read-only M12 resources')


def definitions():
    records = wall_mean_definitions()
    for name, reference in [('wall-scales', {}), ('recovery-temperature', {'recovery_temperature': 26.92}),
                            ('recovery-factor', {'recovery_factor': .9}), ('recovery-zero', {'recovery_factor': 0})]:
        item = deepcopy(records[1])
        item.update(id=name, fields=['wall_shear_x'], wall_reference=reference)
        item['schedule']['interval'] = 3
        records.append(item)
    for name, source in [('partial-scales', records[2]), ('line-scales', records[3])]:
        item = deepcopy(source)
        item.update(id=name, fields=['wall_shear_x'], wall_reference={})
        item['schedule']['interval'] = 3
        records.append(item)
    return records


def final_profile(case, name):
    path = next((case / 'outdat/new/wall_profile_products' / name).glob('*/segment*/frame.step000000000010.csv'))
    return read_profile(path)


def compare(actual, expected):
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    if actual.size == 0:
        return 0., 0.
    error = np.abs(actual-expected)
    scaled = error/np.maximum(1., np.maximum(np.abs(actual), np.abs(expected)))
    assert scaled.max() <= 2e-10, (scaled.max(), error.max())
    return float(scaled.max()), float(error.max())


@pytest.mark.parametrize('backend', ('cpu', 'gpu'))
@pytest.mark.parametrize('topology', ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2),
                                     (2, 2, 1), (2, 1, 2), (1, 2, 2)))
def test_real_wall_references(tmp_path, backend, topology, record_property):
    case = tmp_path / backend
    records = definitions()
    run_wall_fixture(case, backend, topology, records, statistics=True)
    check_wall_means(case, records[:7], record_property)
    with h5py.File(case / 'outdat/new/resources/geometry.h5') as geometry:
        x = geometry['q0001'][0, 0, :]
        y = geometry['q0002'][0, :, 0]
        z = geometry['q0003'][:, 0, 0]
    grid_columns = ['y_first_plus', 'dx_minus_plus', 'dx_plus_plus', 'dz_min_plus', 'dz_max_plus']
    dx, dz = np.diff(x), np.diff(z)
    worst = absolute = 0.
    for item in records[7:]:
        lo, hi = item['sampling']['index_ranges']['k']
        raw_id = 'point-wall' if lo == hi else 'partial-wall' if lo > 0 else 'mean-wall'
        raw, _ = final_profile(case, raw_id)
        rho, mu = raw['rho_time_mean'], raw['dynamic_viscosity']
        magnitude = np.hypot(raw['wall_shear_x'], raw['wall_shear_z'])
        scale = magnitude > 0
        utau = np.sqrt(magnitude[scale])/np.sqrt(rho[scale])
        nu = mu[scale]/rho[scale]
        data, metadata = final_profile(case, item['id'])
        for key in raw.dtype.names:
            np.testing.assert_array_equal(data[key], raw[key])
        np.testing.assert_array_equal(data['wall_scale_available'], scale)
        assert np.all(data['cf_available'] == 1) and np.all(data['cq_available'] == 1)
        for name, expected in [('u_tau', utau), ('nu_wall', nu), ('viscous_length', nu/utau)]:
            scaled, error = compare(data[name][scale], expected)
            worst, absolute = max(worst, scaled), max(absolute, error)
            assert np.isnan(data[name][~scale]).all()
        # The first/last nonperiodic x nodes do not have both adjacent cells.
        length = data['viscous_length']
        selected_dz = dz[lo:hi] if lo != hi else dz[[(lo-1) % len(dz), lo % len(dz)]]
        expected_grid = [(y[1]-y[0])/length, np.r_[np.nan, dx]/length,
                         np.r_[dx, np.nan]/length, selected_dz.min()/length, selected_dz.max()/length]
        for name, expected in zip(grid_columns, expected_grid):
            actual = data[name]
            available = scale & np.isfinite(expected)
            np.testing.assert_array_equal(np.isfinite(actual), available)
            scaled, error = compare(actual[available], expected[available])
            worst, absolute = max(worst, scaled), max(absolute, error)
        assert 'first off-wall node' in metadata and 'cells contained in the selected span' in metadata
        for name, expected in [('Cf', 2*raw['wall_shear_x']), ('Cq_inf', raw['wall_heat_into_wall'])]:
            scaled, error = compare(data[name], expected)
            worst, absolute = max(worst, scaled), max(absolute, error)
        assert 'fixed_primary_free_stream' in metadata and 'U_inf is fixed speed' in metadata
        assert 'cp_inf=1/((gamma-1)*Mach_inf^2)' in metadata
        if not item['wall_reference']:
            assert 'St_stat' not in data.dtype.names
            assert 'Stanton number not requested' in metadata
        else:
            recovery = (item['wall_reference']['recovery_temperature'] if
                        'recovery_temperature' in item['wall_reference'] else
                        1+item['wall_reference']['recovery_factor']*.4*12**2/2)
            denominator = recovery-raw['temperature']
            assert np.all(data['stanton_available'] == 1)
            expected = raw['wall_heat_into_wall']/((1/(.4*144))*denominator)
            scaled, error = compare(data['St_stat'], expected)
            worst, absolute = max(worst, scaled), max(absolute, error)
            assert 'not mean(St_instantaneous)' in metadata
        files = list((case / 'outdat/new/wall_profile_products' / item['id']).glob('*/segment*/frame.step*.csv'))
        assert sorted(int(p.stem.split('step')[1]) for p in files) == [1, 4, 7, 10]
    history_counts = re.findall(r'ASTR_INSITU_SELECTED rank=\d+ histories=(\d+)', (case / 'run.log').read_text())
    assert history_counts and set(history_counts) == {'6'}, 'derived reference changes must not allocate duplicate histories'
    record_property('reference_max_scaled_error', worst)
    record_property('reference_max_absolute_error', absolute)
    record_property('case_directory', str(case))


@pytest.mark.parametrize('backend', ('cpu', 'gpu'))
def test_wall_reference_exact_restart_and_unchanged_history(tmp_path, backend):
    records = definitions()
    topology = (1, 1, 2)
    full, prefix, tail, baseline = [tmp_path / name for name in ('full', 'prefix', 'tail', 'raw-history')]
    run_wall_fixture(full, backend, topology, records, statistics=True)
    run_wall_fixture(prefix, backend, topology, records, steps=5, statistics=True)
    conflicting = deepcopy(records)
    for item in conflicting[7:]:
        if item['wall_reference']:
            item['wall_reference'] = {'recovery_temperature': 30.}
    run_wall_fixture(tail, backend, topology, conflicting, statistics=True,
                     restore=prefix / 'outdat/new/checkpoints/step000000000005')
    run_wall_fixture(baseline, backend, topology, records[:7], statistics=True)
    final = Path('outdat/new/checkpoints/step000000000010')
    for filename in ('state.h5', 'statistics.h5'):
        compare_fields(full / final / filename, tail / final / filename)
        compare_fields(full / final / filename, baseline / final / filename)
    for item in records[1:]:
        before = next((full / 'outdat/new/wall_profile_products' / item['id']).glob('*/segment*'))
        after = next((tail / 'outdat/new/wall_profile_products' / item['id']).glob('*/segment*'))
        for path in after.glob('frame.step*.csv'):
            assert path.read_bytes() == (before / path.name).read_bytes()
