"""Traditional wall units compose live profiles with matching molecular histories."""
from copy import deepcopy
import os
from pathlib import Path
import re

import numpy as np
import pytest

from run_output_restart_validation import compare_fields
from test_insitu_station_profiles import read_profile
from test_insitu_wall_transport_runtime import run_wall_fixture, wall_mean_definitions
from test_runtime_catalog_images import catalog_replay_library

pytestmark = pytest.mark.skipif(not os.environ.get('ASTR_SELECTED_RUNTIME_EXE') or
                               not os.environ.get('ASTR_SELECTED_RUNTIME_M12_SOURCE'),
                               reason='Select the root-CMake M12 build and read-only inputs')


def definitions():
    records = wall_mean_definitions()
    for source in records[1:4]:
        item = deepcopy(source)
        item.update(id=source['id']+'-reference', wall_reference={})
        item['schedule']['interval'] = 5
        records.append(item)
    for name, source, mode, span, wall, fields in [
            ('reynolds', 'reynolds', 'station_span', [0, 24], 'mean-wall', ['velocity_x', 'pressure', 'temperature']),
            ('favre', 'favre', 'station_span', [0, 24], 'mean-wall', ['velocity_x', 'pressure', 'temperature']),
            ('partial', 'favre', 'station_span', [12, 24], 'partial-wall', ['velocity_x', 'pressure', 'temperature']),
            ('point', 'reynolds', 'station_line', [24, 24], 'point-wall', ['velocity_x', 'pressure', 'temperature']),
            ('scalars', 'reynolds', 'station_span', [0, 24], 'mean-wall', ['pressure', 'temperature'])]:
        item = dict(id=name, kind='station_profile', enable_request=1, source=source,
                    statistics_source='selected-time', fields=fields,
                    sampling=dict(mode=mode, axis='i', index=32, index_ranges=dict(j=[0, 32], k=span)),
                    output=dict(channel='data', format='csv'), schedule=dict(mode='steps', interval=3))
        records.append(deepcopy(item))
        item = deepcopy(item)
        item.update(id=name+'-plus', wall_units=wall+'-reference')
        records.append(item)
    return records


def final_profile(case, name, directory='profile_products',steps=10):
    return read_profile(next((case / 'outdat/new' / directory / name).glob(f'*/segment*/frame.step{steps:012d}.csv')))


def compare(actual, expected):
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    difference = np.abs(actual-expected)
    scaled = difference/np.maximum(1., np.maximum(np.abs(actual), np.abs(expected)))
    assert scaled.max() <= 2e-10, (scaled.max(), difference.max())
    return float(scaled.max()), float(difference.max())


@pytest.mark.parametrize('backend', ('cpu', 'gpu'))
@pytest.mark.parametrize('topology', ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2),
                                     (2, 2, 1), (2, 1, 2), (1, 2, 2)))
def test_real_wall_unit_profiles(tmp_path, backend, topology, record_property):
    case = tmp_path / backend
    records = definitions()
    run_wall_fixture(case, backend, topology, records, statistics=True, checkpoint=False)
    assert not list((case / 'outdat/new/checkpoints').glob('step*'))
    assert not (case / 'outdat/new/resources/geometry.h5').exists()
    check_wall_unit_profiles(case, records, record_property)


def check_wall_unit_profiles(case, records, record_property, *, steps=10,expected_histories=10):
    worst = absolute = rms_raw = 0.
    for item in records:
        if 'wall_units' not in item:
            continue
        observed, metadata = final_profile(case, item['id'],steps=steps)
        native, _ = final_profile(case, item['id'].removesuffix('-plus'),steps=steps)
        wall_id = item['wall_units'].removesuffix('-reference')
        wall, _ = final_profile(case, wall_id, 'wall_profile_products',steps=steps)
        wall = wall[wall['global_i'] == item['sampling']['index']][0]
        rho = wall['rho_time_mean']
        utau = np.sqrt(np.hypot(wall['wall_shear_x'], wall['wall_shear_z'])/rho)
        length = (wall['dynamic_viscosity']/rho)/utau
        for field in native.dtype.names:
            np.testing.assert_array_equal(observed[field], native[field])
        assert np.all(observed['wall_scale_available'] == 1)
        assert 'matching positive-weight sampling identity required' in metadata
        assert 'statistical_wall_reference' in metadata
        expected = {'y_plus': (native['y']-native['y'][0])/length}
        for field in native.dtype.names:
            if field.startswith(('velocity_', 'rms_velocity_')):
                expected[field+'_plus'] = native[field]/utau
            elif field.startswith('covariance_'):
                expected[field+'_plus'] = native[field]/utau**2
            elif field.startswith('rho_covariance_f_'):
                expected[field+'_plus'] = native[field]/(rho*utau**2)
        assert set(observed.dtype.names) == {*native.dtype.names, 'wall_scale_available', *expected}
        for field, value in expected.items():
            actual = observed[field]
            if field.startswith('rms_velocity_'):
                variance = native['covariance_'+field[13] * 2]/utau**2
                rms_raw = max(rms_raw, float(np.max(np.abs(actual-np.sqrt(variance)))))
                actual, value = actual**2, variance
            error, raw = compare(actual, value)
            worst, absolute = max(worst, error), max(absolute, raw)
        frames = list((case / 'outdat/new/profile_products' / item['id']).glob('*/segment*/frame.step*.csv'))
        assert sorted(int(p.stem.split('step')[1]) for p in frames) == list(range(1,steps+1,item['schedule']['interval']))
    histories = re.findall(r'ASTR_INSITU_SELECTED rank=\d+ histories=(\d+)', (case / 'run.log').read_text())
    assert histories and set(histories) == {str(expected_histories)}, 'normalization must share the native histories'
    record_property('max_scaled_error', worst)
    record_property('max_absolute_error', absolute)
    record_property('rms_raw_maxabs', rms_raw)
    record_property('case_directory', str(case))


@pytest.mark.parametrize('backend', ('cpu', 'gpu'))
@pytest.mark.parametrize('topology', ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2),
                                     (2, 2, 1), (2, 1, 2), (1, 2, 2)))
def test_read_grid_wall_unit_profiles(tmp_path, backend, topology, record_property):
    grid = os.environ.get('ASTR_SELECTED_RUNTIME_CARTESIAN_GRID')
    if not grid:
        pytest.skip('Select the immutable Cartesian 64x32x24 grid from an earlier M12 fixture')
    case = tmp_path / backend
    records = definitions()
    run_wall_fixture(case, backend, topology, records, statistics=True, checkpoint=False, read_grid=grid)
    check_wall_unit_profiles(case, records, record_property)
    text = (case / 'run.log').read_text()
    proof = re.findall(r'ASTR_SELECTED_GEOMETRY rank=(\d+) cartesian=T axis_directions=1 1 1', text)
    assert len(proof) == np.prod(topology), 'one actual geometry proof per rank, not a file-read flag bypass'


@pytest.mark.parametrize('backend', ('cpu', 'gpu'))
def test_wall_unit_profile_exact_restart_and_authority(tmp_path, backend):
    records = definitions()
    full, first, tail, native = [tmp_path / n for n in ('full', 'prefix', 'tail', 'native')]
    topology = (1, 1, 2)
    run_wall_fixture(full, backend, topology, records, statistics=True)
    run_wall_fixture(first, backend, topology, records, steps=5, statistics=True)
    conflicting = deepcopy(records)
    for item in conflicting:
        if 'wall_units' in item:
            item['wall_units'] = 'unconsumed-disk-definition'
    run_wall_fixture(tail, backend, topology, conflicting, statistics=True,
                     restore=first / 'outdat/new/checkpoints/step000000000005')
    without = deepcopy(records)
    for item in without:
        item.pop('wall_units', None)
    run_wall_fixture(native, backend, topology, without, statistics=True)
    final = Path('outdat/new/checkpoints/step000000000010')
    for filename in ('state.h5', 'statistics.h5'):
        compare_fields(full / final / filename, tail / final / filename)
        compare_fields(full / final / filename, native / final / filename)
    for item in records:
        if item['kind'] != 'station_profile':
            continue
        before = next((full / 'outdat/new/profile_products' / item['id']).glob('*/segment*'))
        after = next((tail / 'outdat/new/profile_products' / item['id']).glob('*/segment*'))
        for frame in after.glob('frame.step*.csv'):
            assert frame.read_bytes() == (before / frame.name).read_bytes()


@pytest.mark.parametrize('backend', ('cpu', 'gpu'))
def test_unavailable_wall_reference_retains_raw_profile(tmp_path, backend):
    records = definitions()
    source = next(p for p in records if p['id'] == 'reynolds')
    for name, reference in [('missing', 'not-present'), ('wrong-span', 'partial-wall-reference')]:
        item = deepcopy(source)
        item.update(id=name, wall_units=reference)
        records.append(item)
    case = tmp_path / backend
    run_wall_fixture(case, backend, (2, 1, 1), records, statistics=True)
    native, _ = final_profile(case, 'reynolds')
    for name in ('missing', 'wrong-span'):
        data, metadata = final_profile(case, name)
        assert np.all(data['wall_scale_available'] == 0)
        for field in native.dtype.names:
            np.testing.assert_array_equal(data[field], native[field])
        for field in data.dtype.names:
            if field.endswith('_plus'):
                assert np.isnan(data[field]).all()
        assert 'wall_units_unavailable=' in metadata
        assert ('no active wall_profile history' if name == 'missing' else 'same span and spatial average') in metadata


@pytest.mark.parametrize('backend', ('cpu', 'gpu'))
@pytest.mark.parametrize('keep_shared', (False, True))
def test_actual_gap_identity_and_shared_consumers(tmp_path, backend, keep_shared, catalog_replay_library):
    records = definitions()
    wall = next(deepcopy(p) for p in records if p['id'] == 'mean-wall-reference')
    profile = next(deepcopy(p) for p in records if p['id'] == 'reynolds-plus')
    wall['schedule']['interval'] = profile['schedule']['interval'] = 1
    keepers = []
    if keep_shared:
        for source in (wall, profile):
            item = deepcopy(source)
            item['id'] += '-keeper'
            item['schedule']['interval'] = 1000
            keepers.append(item)
    # Poll 1 observes step 0. The two histories have different real sampling gaps.
    publications = {4: [wall, *keepers], 5: keepers, 6: [profile, *keepers], 7: [wall, profile, *keepers]}
    case = tmp_path / backend
    run_wall_fixture(case, backend, (1, 1, 2), [wall, profile, *keepers], statistics=True,
                     catalog_replay=(catalog_replay_library, publications))
    data, metadata = final_profile(case, profile['id'])
    assert np.all(data['wall_scale_available'] == int(keep_shared))
    for field in data.dtype.names:
        if field.endswith('_plus'):
            assert np.isfinite(data[field]).all() if keep_shared else np.isnan(data[field]).all()
        else:
            assert np.isfinite(data[field]).all()
    if keep_shared:
        assert 'statistical_wall_reference' in metadata and 'sampling_segments=1' in metadata
    else:
        assert 'sampling histories differ' in metadata and 'sampling_segments=2' in metadata
    text = (case / 'run.log').read_text()
    for poll in (4, 5, 6, 7):
        assert f'ASTR_TEST_CATALOG_PUBLISH poll={poll}' in text
