"""Real completed-step molecular wall diagnostics, with explicit checkpoint oracles."""
import json
import os
from fractions import Fraction
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pytest

from prepare_m12_local_case import prepare, replace_value
from test_index_slice_runtime import ROOT, arguments, product, segments, frames
from test_insitu_selected_runtime import configuration
from test_insitu_station_profiles import read_profile, endpoint_weights
from run_output_restart_validation import compare_fields

EXE = os.environ.get('ASTR_SELECTED_RUNTIME_EXE')
SOURCE = os.environ.get('ASTR_SELECTED_RUNTIME_M12_SOURCE')
pytestmark = pytest.mark.skipif(not EXE or not SOURCE, reason='Select the built M12 executable and read-only resources')
GRADIENTS = ['velocity_gradient_'+c+d for d in 'xyz' for c in 'xyz']
THERMAL = ['temperature_gradient_'+d for d in 'xyz']
WALL = ['wall_shear_x', 'wall_shear_z', 'wall_heat_into_wall', 'dynamic_viscosity']


def derivative_weights():
    coefficients = np.zeros((7, 7))
    for offset in range(7):
        for s in range(7):
            denominator = np.prod([s-j for j in range(7) if j != s])
            value = Fraction(0)
            for m in range(7):
                if m == s:
                    continue
                numerator = np.prod([offset-j for j in range(7) if j != s and j != m])
                value += Fraction(int(numerator), int(denominator))
            coefficients[offset, s] = float(value)
    return coefficients


def expected_checkpoint_gradients(case):
    with h5py.File(case / 'outdat/new/checkpoints/step000000000010/state.h5') as state:
        # These are the saved complete-step primitive caches, not an earlier RK state.
        values = np.stack([state[f'q{n:04d}'][...] for n in (7, 8, 9, 11)])
    # The provider folds periodic support nodes, including tangential derivative lines.
    # Keep the output terminal node in the comparison; reconstruct its support from k=0.
    values[:, -1] = values[:, 0]
    with h5py.File(case / 'outdat/new/resources/geometry.h5') as geometry:
        metrics = np.empty((3, 3, 25, 65))
        for d in range(3):
            for p in range(3):
                metrics[d, p] = geometry[f'q{5+d+3*p:04d}'][:, 0, :]
    weights = derivative_weights()
    i = np.arange(65)
    first = np.clip(i-3, 0, 58)
    sample_i = first[:, None]+np.arange(7)
    sample_k = (np.arange(25)[:, None]-3+np.arange(7)) % 24
    directional = np.empty((4, 3, 25, 65))
    for c, value in enumerate(values):
        directional[c, 0] = np.einsum('kis,is->ki', value[:, 0, sample_i], weights[i-first])
        directional[c, 1] = np.einsum('kji,j->ki', value[:, :7, :], weights[0])
        directional[c, 2] = np.einsum('ksi,s->ki', value[sample_k, 0, :], weights[3])
    return np.einsum('cdki,dpki->cpki', directional, metrics)


def check(case, record_property):
    segment = next(p for p in segments(case) if p.parent.parent.name == 'wall-complete')
    expected = expected_checkpoint_gradients(case)
    with h5py.File(frames(segment)[-1]) as data:
        assert int(data['step'][0]) == 10
        arrays = {key: data[key][...] for key in [*GRADIENTS, *THERMAL, *WALL]}
        temperature = data['temperature'][...]
        temperature[-1] = temperature[0]
        reference_t, reynolds, mach = 594.3, 1.e4, 12.
        # The fixture input is immutable; do not silently reuse these constants for a new case.
        lines = (case / 'datin/input.dat').read_text().splitlines()
        marker = next(i for i, line in enumerate(lines) if line.startswith('# ref_t,reynolds,mach'))
        parsed = [float(v.strip().lower().replace('d', 'e')) for v in lines[marker+1].split(',')]
        np.testing.assert_array_equal(parsed, [reference_t, reynolds, mach])
        transport_source = (ROOT / 'src/perfect_gas_transport.F90').read_text()
        assert 'sutherland_k=110.3e0_real_kind' in transport_source
        sutherland = 110.3/reference_t
        mu = temperature*np.sqrt(temperature)*(1+sutherland)/(temperature+sutherland)/reynolds
        kappa = (mu/.72)/(.4*mach*mach)
        targets = {name: expected[c, d] for d in range(3) for c, name in
                   enumerate(GRADIENTS[3*d:3*d+3])}
        targets.update({name: expected[3, d] for d, name in enumerate(THERMAL)})
        targets.update(wall_shear_x=mu*(expected[0, 1]+expected[1, 0]),
                       wall_shear_z=mu*(expected[2, 1]+expected[1, 2]),
                       wall_heat_into_wall=kappa*expected[3, 1], dynamic_viscosity=mu)
        worst = absolute = 0.
        for key, target in targets.items():
            actual = arrays[key]
            assert np.isfinite(actual).all(), key
            error = np.abs(actual-target)
            scaled = error/np.maximum(1., np.maximum(np.abs(actual), np.abs(target)))
            worst = max(worst, float(scaled.max()))
            absolute = max(absolute, float(error.max()))
            assert scaled.max() <= 2e-10, (key, scaled.max(), error.max())
        for key, units in [('temperature_gradient_y', 'solver-dimensionless'),
                           ('wall_shear_x', 'solver-dimensionless')]:
            assert units in str(data[key].attrs['units'])
        # Blowing tangent derivatives must be visible and must contribute to the tensor.
        cross = mu*expected[1, 0]
        assert np.max(np.abs(cross)) > 1e-15
        record_property('blowing_tangent_shear_maxabs', float(np.max(np.abs(cross))))
    status = json.loads((case / 'outdat/new/products.status.json').read_text())
    assert status['products'][0]['state'] == 'active', status
    record_property('max_scaled_error', worst)
    record_property('max_absolute_error', absolute)
    record_property('case_directory', str(case))


def run_wall_fixture(case, backend, topology, records, steps=10, statistics=False, restore=None, catalog_replay=None,
                     checkpoint=True, read_grid=None, omit_restart_inputs=False, reject=None):
    prepare(Path(SOURCE), case, backend)
    if read_grid is not None:
        input_file = case / 'datin/input.dat'
        lines = input_file.read_text().splitlines()
        replace_value(lines, 'nondimen,diffterm', 't,t,f,t,f,f,f,f,'+('t' if backend == 'gpu' else 'f'))
        (case / 'datin/read_grid.h5').symlink_to(Path(read_grid).resolve(strict=True))
        replace_value(lines, 'gridfile', 'datin/read_grid.h5')
        input_file.write_text('\n'.join(lines)+'\n')
    (case / 'input.products.jsonl').write_text('\n'.join(json.dumps(row) for row in
        [{'catalog_schema': 'astr-insitu-products-v1'}, *records])+'\n')
    controller = case / 'datin/controller'
    lines = controller.read_text().splitlines()
    replace_value(lines, 'maxstep,feqchkpt', f'{steps-1},1,9999,9999,1,9999')
    controller.write_text('\n'.join(lines)+'\n')
    output = case / 'datin/input.output'
    reference = f", restore_directory='{restore}'" if restore else ''
    text = output.read_text().replace(" directory='outdat/new',", f" directory='outdat/new', exact_restart=t{reference},")
    interval = 5 if statistics else 10
    if checkpoint:
        text = text.replace('&checkpoint\n enabled=f', f"&checkpoint\n enabled=t,mode='steps',interval_steps={interval},keep=2")
    if catalog_replay is not None:
        text = text.replace('exact_restart=t,', 'exact_restart=t,product_poll_steps=1,')
    output.write_text(text)
    if omit_restart_inputs:
        assert restore is not None and read_grid is not None
        # Delete only this newly prepared test's copies, never the frozen source.
        for filename in ('read_grid.h5', 'grid.2d', 'inlet.prof', 'wallbs.dat', 'flowini2d.h5'):
            (case/'datin'/filename).unlink()
    (case / 'outdat/new').mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str, topology)), ASTR_WALL_BLOWING_MODE='legacy_random',
               ASTR_GPU_HALO_TRANSPORT='pinned', ASTR_GPU_SYNC_MODE='explicit', ASTR_INSITU_TIMING='1')
    if catalog_replay is not None:
        library, publications = catalog_replay
        directory = case / 'catalog_replay'
        directory.mkdir()
        for poll, definitions in publications.items():
            (directory / f'catalog{poll:06d}.jsonl').write_text('\n'.join(json.dumps(row) for row in
                [{'catalog_schema': 'astr-insitu-products-v1'}, *definitions])+'\n')
        env.update(LD_PRELOAD=str(Path(library).resolve()), ASTR_TEST_CATALOG_FILE='input.products.jsonl',
                   ASTR_TEST_CATALOG_REPLAY=str(directory.resolve()))
    if statistics:
        (case / 'datin/input.insitu').write_text(configuration((.01, .19)))
        env['ASTR_INSITU_CONFIG'] = 'datin/input.insitu'
    args = arguments(case.parent)
    from named_fixture import prepare_named_run
    primary=prepare_named_run(case,'datin/input.dat',env)
    with (case / 'run.log').open('w') as log:
        result = subprocess.run([str(args.mpiexec), '--mca', 'coll_hcoll_enable', '0', '--oversubscribe',
                                 '-np', str(np.prod(topology)), str(Path(EXE).resolve()), 'run', primary],
                                cwd=case, env=env, stdout=log, stderr=subprocess.STDOUT)
    log = (case/'run.log').read_text()
    if reject is not None:
        assert result.returncode != 0 and reject in log, log
        assert 'time cost at step' not in log and not list((case/'outdat/new/checkpoints').glob('step*'))
    else:
        assert result.returncode == 0 and 'The job is done!' in log, log


@pytest.mark.parametrize('backend', ('cpu', 'gpu'))
@pytest.mark.parametrize('topology', ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2),
                                     (2, 2, 1), (2, 1, 2), (1, 2, 2)))
def test_real_complete_wall_transport(tmp_path, backend, topology, record_property):
    case = tmp_path / backend
    definition = product('wall-complete', ['density', 'velocity_x', 'velocity_y', 'velocity_z', 'temperature',
                                          *GRADIENTS, *THERMAL, *WALL])
    definition['sampling'] = dict(mode='grid_index', axis='j', index=0,
                                  index_ranges=dict(i=[0, 64], k=[0, 24]))
    definition['schedule']['interval'] = 5
    run_wall_fixture(case, backend, topology, [definition])
    check(case, record_property)


def wall_mean_definitions():
    raw = product('raw-wall', ['density', 'temperature', *WALL])
    raw['sampling'] = dict(mode='grid_index', axis='j', index=0, index_ranges=dict(i=[0, 64], k=[0, 24]))
    raw['schedule']['interval'] = 1
    records = [raw]
    for name, mode, krange in [('mean-wall', 'wall_span', [0, 24]),
                               ('partial-wall', 'wall_span', [12, 24]),
                               ('point-wall', 'wall_line', [24, 24])]:
        records.append(dict(id=name, kind='wall_profile', enable_request=1, source='reynolds',
                            statistics_source='selected-time', fields=raw['fields'],
                            sampling=dict(mode=mode, axis='j', index=0, index_ranges=dict(i=[0, 64], k=krange)),
                            output=dict(channel='data', format='csv'), schedule=dict(mode='steps', interval=1)))
    for name, fields, mode, krange in [
            ('viscosity-only', ['dynamic_viscosity'], 'wall_span', [12, 24]),
            ('shear-only', ['wall_shear_x', 'wall_shear_z'], 'wall_span', [0, 24]),
            ('thermal-only', ['temperature', 'wall_heat_into_wall'], 'wall_line', [24, 24])]:
        records.append(dict(id=name, kind='wall_profile', enable_request=1, source='reynolds',
                            statistics_source='selected-time', fields=fields,
                            sampling=dict(mode=mode, axis='j', index=0, index_ranges=dict(i=[0, 64], k=krange)),
                            output=dict(channel='data', format='csv'), schedule=dict(mode='steps', interval=1)))
    return records


def check_wall_means(case, records, record_property, *, steps=10,dt=.02,window=(.01,.19),ni=65,nk=24):
    raw = next(p for p in segments(case) if p.parent.parent.name == 'raw-wall')
    fields = ['density', 'temperature', *WALL]
    samples = []
    for path in frames(raw):
        with h5py.File(path) as data:
            samples.append(np.stack([data[key][...] for key in fields]))
    samples = np.stack(samples)
    assert samples.shape == (steps+1, 6, nk+1, ni)
    weights = endpoint_weights(steps, dt, window)
    temporal = np.einsum('t,tfki->fki', weights, samples)/weights.sum()
    with h5py.File(raw / 'geometry.h5') as geometry:
        xyz = geometry['coordinates'][...]
    worst = absolute = 0.
    for item in records[1:]:
        path = next((case / 'outdat/new/wall_profile_products' / item['id']).glob(f'*/segment*/frame.step{steps:012d}.csv'))
        observed, metadata = read_profile(path)
        zweight = np.zeros(nk)
        a, b = item['sampling']['index_ranges']['k']
        if a == b:
            zweight[a % nk] = 1
        else:
            for k in range(a, b+1):
                zweight[k % nk] += .5*(xyz[min(b,k+1), 0, 2]-xyz[max(a,k-1), 0, 2])
        target = np.einsum('k,fki->fi', zweight, temporal[:, :nk])/zweight.sum()
        for f, name in enumerate(['rho_time_mean', *fields[1:]]):
            if f>0 and fields[f] not in item['fields']:
                continue
            actual = observed[name]
            error = np.abs(actual-target[f]);scaled = error/np.maximum(1., np.maximum(np.abs(actual), np.abs(target[f])))
            assert scaled.max() <= 2e-10, (item['id'], name, scaled.max())
            worst, absolute = max(worst, float(scaled.max())), max(absolute, float(error.max()))
        np.testing.assert_array_equal(observed['global_i'], np.arange(ni))
        assert set(observed.dtype.names) == {'global_i', 'x', 'rho_time_mean', *set(item['fields'])-{'density'}}
        np.testing.assert_allclose(observed['x'], xyz[0, :, 0], rtol=0, atol=2e-10)
        assert 'source=reynolds' in metadata and 'sampling_segments=1' in metadata
    status = json.loads((case / 'outdat/new/products.status.json').read_text())
    assert all(item['state'] == 'active' for item in status['products']), status
    record_property('max_scaled_error', worst)
    record_property('max_absolute_error', absolute)
    record_property('case_directory', str(case))


@pytest.mark.parametrize('backend', ('cpu', 'gpu'))
@pytest.mark.parametrize('topology', ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2),
                                     (2, 2, 1), (2, 1, 2), (1, 2, 2)))
def test_completed_wall_means(tmp_path, backend, topology, record_property):
    case = tmp_path / backend
    records = wall_mean_definitions()
    run_wall_fixture(case, backend, topology, records, statistics=True)
    check_wall_means(case, records, record_property)


@pytest.mark.parametrize('backend', ('cpu', 'gpu'))
def test_wall_mean_exact_restart_and_unchanged_authority(tmp_path, backend):
    records = wall_mean_definitions()
    topology = (1, 1, 2)
    full, prefix, tail = [tmp_path / name for name in ('full', 'prefix', 'tail')]
    run_wall_fixture(full, backend, topology, records, statistics=True)
    run_wall_fixture(prefix, backend, topology, records, steps=5, statistics=True)
    run_wall_fixture(tail, backend, topology, records, statistics=True,
                     restore=prefix / 'outdat/new/checkpoints/step000000000005')
    final = Path('outdat/new/checkpoints/step000000000010')
    for filename in ('state.h5', 'statistics.h5'):
        compare_fields(full / final / filename, tail / final / filename)
    baseline = tmp_path / 'no-wall-history'
    run_wall_fixture(baseline, backend, topology, records[:1])
    compare_fields(baseline / final / 'state.h5', full / final / 'state.h5')
    for item in records[1:]:
        before = next((full / 'outdat/new/wall_profile_products' / item['id']).glob('*/segment*'))
        after = next((tail / 'outdat/new/wall_profile_products' / item['id']).glob('*/segment*'))
        for path in after.glob('frame.step*.csv'):
            assert path.read_bytes() == (before / path.name).read_bytes()
