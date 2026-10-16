"""Lightweight/default checkpoint and independent cumulative-statistics I/O."""
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pytest

from prepare_m12_local_case import prepare, replace_value
from run_m12_boundary_compare import configuration
from run_m12_mp7_nscbc_performance import mpi_command, runtime
from run_output_restart_validation import run_case, compare_fields
from test_output_repartition_runtime import ROOT, arguments

EXE = os.environ.get('ASTR_OUTPUT_RUNTIME_EXE')
MPIEXEC = '/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec'
pytestmark = pytest.mark.skipif(not EXE, reason='set ASTR_OUTPUT_RUNTIME_EXE to the rebuilt solver')


def launch(tmp_path, backend, ranks, name, steps, **kwargs):
    args = arguments(tmp_path)
    args.executable = Path(EXE).resolve()
    return run_case(args, ROOT, backend, ranks, name, steps, checkpoint_interval=2, **kwargs)[0]


def batch(case, step):
    return case / f'outdat/new/checkpoints/step{step:012d}'


def assert_physical_fields_close(reference, actual):
    with h5py.File(reference) as a, h5py.File(actual) as b:
        for n in range(1, 6):
            key = f'q{n:04d}'
            assert np.isfinite(b[key][...]).all()
            np.testing.assert_allclose(a[key][...], b[key][...], rtol=0, atol=2e-10)
        assert (b['q0001'][...] > 0).all()


@pytest.mark.parametrize('backend,ranks', [('cpu', 1), ('cpu', 2), ('gpu', 1), ('gpu', 2)])
def test_lightweight_checkpoint_and_restart(tmp_path, backend, ranks):
    continuous = launch(tmp_path, backend, ranks, 'continuous', 4, exact_restart=False)
    source = batch(continuous, 2)
    with h5py.File(source / 'state.h5') as state:
        assert state['identity'][12] == 11
        assert sorted(key for key in state if key.startswith('q')) == [f'q{n:04d}' for n in range(1, 6)]
        assert state['rank_extras'].size == 0
        assert (state['partitions'][...].reshape(-1, 8)[:, 6:] == 0).all()
    assert not (continuous / 'outdat/new/resources/geometry.h5').exists()
    assert 'geometry.h5' not in (source / 'RESOURCES').read_text()
    assert (source / 'statistics.h5').is_file()
    resumed = launch(tmp_path, backend, ranks, 'resumed', 4, restore=source, exact_restart=False)
    assert_physical_fields_close(batch(continuous, 4) / 'state.h5', batch(resumed, 4) / 'state.h5')
    with h5py.File(batch(continuous, 4) / 'statistics.h5') as a, \
            h5py.File(batch(resumed, 4) / 'statistics.h5') as b:
        np.testing.assert_array_equal(a['q0008'][...], b['q0008'][...])


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
def test_explicit_exact_checkpoint_stays_bitwise(tmp_path, backend):
    continuous = launch(tmp_path, backend, 2, 'continuous', 4)
    resumed = launch(tmp_path, backend, 2, 'resumed', 4, restore=batch(continuous, 2))
    for name in ('state.h5', 'statistics.h5'):
        compare_fields(batch(continuous, 4) / name, batch(resumed, 4) / name)
    assert (continuous / 'outdat/new/resources/geometry.h5').is_file()


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
@pytest.mark.parametrize('saved_statistics', [False, True])
def test_statistics_switch_starts_new_coverage_without_changing_flow(tmp_path, backend, saved_statistics):
    reference = launch(tmp_path, backend, 2, 'reference', 4)
    disabled = launch(tmp_path, backend, 2, 'disabled', 4, checkpoint_statistics=False)
    compare_fields(batch(reference, 4) / 'state.h5', batch(disabled, 4) / 'state.h5')
    assert not (batch(disabled, 4) / 'statistics.h5').exists()
    assert 'statistics.h5' not in (batch(disabled, 4) / 'MANIFEST').read_text()
    source = batch(reference if saved_statistics else disabled, 2)
    # If the source omitted statistics, re-enable saving. Otherwise discard its history.
    resumed = launch(tmp_path, backend, 2, 'resumed', 4, restore=source,
                     checkpoint_statistics=not saved_statistics)
    compare_fields(batch(reference, 4) / 'state.h5', batch(resumed, 4) / 'state.h5')
    if not saved_statistics:
        with h5py.File(batch(resumed, 4) / 'statistics.h5') as state:
            # Device statistics intentionally omit duplicated periodic endpoints.
            np.testing.assert_allclose(state['q0008'][...][:-1, :-1, :-1], 0.002, rtol=0, atol=2e-17)
    else:
        assert not (batch(resumed, 4) / 'statistics.h5').exists()


def test_lightweight_source_cannot_be_claimed_exact(tmp_path):
    first = launch(tmp_path, 'gpu', 1, 'first', 2, exact_restart=False)
    launch(tmp_path, 'gpu', 1, 'rejected', 4, restore=batch(first, 2),
           reject='exact_restart=t cannot restore a non-exact checkpoint')


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
def test_full_source_can_be_resumed_in_lightweight_mode(tmp_path, backend):
    reference = launch(tmp_path, backend, 2, 'reference', 4)
    resumed = launch(tmp_path, backend, 2, 'resumed', 4, restore=batch(reference, 2), exact_restart=False)
    assert_physical_fields_close(batch(reference, 4) / 'state.h5', batch(resumed, 4) / 'state.h5')
    with h5py.File(batch(resumed, 4) / 'state.h5') as state:
        assert state['identity'][12] == 11


def test_lightweight_device_pack_restore_memcheck(tmp_path):
    first = launch(tmp_path, 'gpu', 2, 'first', 2, exact_restart=False, memcheck=True)
    resumed = launch(tmp_path, 'gpu', 2, 'resumed', 4, restore=batch(first, 2),
                     exact_restart=False, memcheck=True)
    assert (batch(resumed, 4) / 'COMPLETE').is_file()


@pytest.mark.parametrize('backend,ranks', [('cpu', 1), ('gpu', 2)])
def test_m12_lightweight_random_blowing_and_crashfix_history(tmp_path, backend, ranks):
    def run(name, steps, restore=None):
        case = tmp_path / name
        prepare(Path('/home/dell/workspace/M12_C13'), case, backend)
        configuration(case, 'nscbc' if backend == 'gpu' else 'extrapolation', (64, 32, 24), steps,
                      archive=False, checkpoint_interval=5, restore=restore, exact_restart=False)
        controller = case / 'datin/controller'
        lines = controller.read_text().splitlines()
        replace_value(lines, 'lwsequ,lwslic', 'f,f,f,t')
        controller.write_text('\n'.join(lines) + '\n')
        # Verify the user-facing default, not only an explicitly false value.
        config = case / 'datin/input.output'
        config.write_text(config.read_text().replace(' exact_restart=f,', ' '))
        env = runtime(f'{ranks},1,1')
        command = [*mpi_command(MPIEXEC, ranks, 'pinned'), str(Path(EXE).resolve()), 'run', 'datin/input.dat']
        with (case / 'run.log').open('w') as log:
            result = subprocess.run(command, cwd=case, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        assert result.returncode == 0, (case / 'run.log').read_text()[-8000:]
        return case
    reference = run('reference', 10)
    resumed = run('resumed', 10, batch(reference, 5))
    assert_physical_fields_close(batch(reference, 10) / 'state.h5', batch(resumed, 10) / 'state.h5')
    with h5py.File(batch(reference, 10) / 'state.h5') as a, h5py.File(batch(resumed, 10) / 'state.h5') as b:
        np.testing.assert_array_equal(a['crashfix_history'][...], b['crashfix_history'][...])
