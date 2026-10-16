"""Checkpoint modes on channel and static/dynamic curvilinear flatplates."""
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from run_output_restart_validation import run_case, compare_fields
from test_output_repartition_runtime import (ROOT, air5_arguments, channel_arguments,
                                           curve_arguments, frozen_channel_initial, clocks)

GPU = Path(os.environ.get('ASTR_OUTPUT_RUNTIME_EXE', ROOT / 'build_gpu_probe/bin/astr')).resolve()
CPU = Path(os.environ.get('ASTR_OUTPUT_CPU_EXE', ROOT / 'build_release_restart_cpu/bin/astr')).resolve()
ATOL = 2e-10


def batch(case, step=10):
    return case / f'outdat/new/checkpoints/step{step:012d}'


def channel_statistics():
    return """&insitu_run
 enabled=t,statistics=t,render=f,products='channel_walls',
 statistics_window=0.0005,0.0115,output_directory='outdat/render',
 host_budget_bytes=67108864,device_budget_bytes=67108864,device_reserve_bytes=1073741824
/
"""


def setup(output, backend, case_name, axis):
    extra = {}
    if case_name == 'channel':
        args = channel_arguments(output, backend, axis)
        args.legacy_statistics = False
        extra['initial_resource'] = frozen_channel_initial(output / 'initial.h5')
        extra['insitu_config'] = channel_statistics()
    else:
        args = curve_arguments(output, backend, axis)
        args.case = case_name
        args.legacy_statistics = True
        if case_name == 'dynamic':
            args.inflow_count = 12
    args.executable = GPU if backend == 'gpu' else CPU
    return args, extra


def compare_close(left, right):
    errors = {}
    with h5py.File(left) as a, h5py.File(right) as b:
        names_a, names_b = [], []
        a.visit(names_a.append); b.visit(names_b.append)
        assert names_a == names_b
        for name in names_a:
            if isinstance(a[name], h5py.Group):
                continue
            aa, bb = a[name][...], b[name][...]
            assert aa.shape == bb.shape and aa.dtype == bb.dtype, name
            if aa.dtype.kind != 'f':
                # The velocity-statistics metadata embeds FP64 regional moments.
                if name == 'metadata' and int(a['identity'][12]) in (3, 4):
                    np.testing.assert_array_equal(aa[:4], bb[:4])
                    aa, bb = aa[4:].view(np.float64), bb[4:].view(np.float64)
                else:
                    np.testing.assert_array_equal(aa, bb, err_msg=name)
                    continue
            assert np.isfinite(aa).all() and np.isfinite(bb).all(), name
            error = float(np.max(np.abs(aa-bb), initial=0.))
            errors[name] = error
            assert error <= ATOL, (name, error)
    return max(errors.values(), default=0.)


def assert_positive(path):
    with h5py.File(path) as state:
        q = np.stack([state[f'q{n:04d}'][...] for n in range(1, 6)], axis=-1)
    assert np.isfinite(q).all()
    assert (q[..., 0] > 0).all()
    internal = q[..., 4] - .5*np.sum(q[..., 1:4]**2, axis=-1)/q[..., 0]
    assert np.isfinite(internal).all() and (internal > 0).all()


@pytest.mark.parametrize('case_name', ['channel', 'curve', 'dynamic'])
@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
@pytest.mark.parametrize('ranks,axis', [(1, 'z'), (2, 'x'), (2, 'z')])
@pytest.mark.parametrize('exact', [False, True], ids=['lean', 'exact'])
def test_other_case_restart(tmp_path, case_name, backend, ranks, axis, exact):
    args, extra = setup(tmp_path, backend, case_name, axis)
    common = dict(checkpoint_interval=5, buffer_bytes=4096, exact_restart=exact, **extra)
    reference, _ = run_case(args, ROOT, backend, ranks, 'continuous', 10, **common)
    source = batch(reference, 5)
    resumed, _ = run_case(args, ROOT, backend, ranks, 'resumed', 10, restore=source, **common)
    with h5py.File(source / 'state.h5') as state:
        assert int(state['identity'][12]) == (2 if exact else 11)
        if not exact:
            assert state['rank_extras'].size == 0
            assert len([name for name in state if name.startswith('q')]) == 5
    assert (reference / 'outdat/new/resources/geometry.h5').exists() == exact
    state_error = compare_close(batch(reference) / 'state.h5', batch(resumed) / 'state.h5')
    stats_error = compare_close(batch(reference) / 'statistics.h5', batch(resumed) / 'statistics.h5')
    if exact:
        for name in ('state.h5', 'statistics.h5'):
            compare_fields(batch(reference) / name, batch(resumed) / name)
    for name in ('control.bin', 'insitu_control.bin'):
        assert (batch(reference) / name).read_bytes() == (batch(resumed) / name).read_bytes()
    assert clocks(resumed) == clocks(reference)[5:]
    if case_name == 'dynamic':
        compare_fields(batch(reference) / 'inflow.h5', batch(resumed) / 'inflow.h5')
        with h5py.File(source / 'inflow.h5') as a, h5py.File(batch(resumed) / 'inflow.h5') as b:
            assert int(b['metadata'][2]) > int(a['metadata'][2])
    assert_positive(batch(resumed) / 'state.h5')
    (tmp_path / 'result.json').write_text(json.dumps(dict(case=case_name, backend=backend, ranks=ranks,
        axis=axis, exact=exact, steps=10, restart_step=5, state_maxabs=state_error,
        statistics_maxabs=stats_error, continuous=str(reference), resumed=str(resumed)), indent=2) + '\n')


@pytest.mark.parametrize('case_name', ['channel', 'curve', 'dynamic'])
@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
@pytest.mark.parametrize('source_statistics', [False, True], ids=['enable-on-restart', 'disable-on-restart'])
def test_other_case_statistics_io(tmp_path, case_name, backend, source_statistics):
    args, extra = setup(tmp_path, backend, case_name, 'z')
    common = dict(checkpoint_interval=5, buffer_bytes=4096, exact_restart=False, **extra)
    reference, _ = run_case(args, ROOT, backend, 2, 'continuous', 10, **common)
    disabled, _ = run_case(args, ROOT, backend, 2, 'disabled', 10, checkpoint_statistics=False, **common)
    compare_fields(batch(reference) / 'state.h5', batch(disabled) / 'state.h5')
    assert not (batch(disabled) / 'statistics.h5').exists()
    assert 'statistics.h5' not in (batch(disabled) / 'MANIFEST').read_text()
    source = batch(reference if source_statistics else disabled, 5)
    resumed, _ = run_case(args, ROOT, backend, 2, 'resumed', 10, restore=source,
                          checkpoint_statistics=not source_statistics, **common)
    state_error = compare_close(batch(reference) / 'state.h5', batch(resumed) / 'state.h5')
    assert (batch(resumed) / 'statistics.h5').exists() != source_statistics
    if not source_statistics:
        with h5py.File(batch(resumed) / 'statistics.h5') as state:
            metadata = state['metadata'][...]
            if case_name == 'channel':
                # Five elapsed intervals, seeded at completed step five.
                np.testing.assert_allclose(state['q0008'][...][:-1, :, :-1], .005, atol=2e-17, rtol=0)
                duration = np.array([state['wall_statistics/metadata'][9]], dtype=np.int64).view(np.float64)[0]
                assert abs(duration-.005) <= 2e-17
            else:
                count_index = 7 if backend == 'gpu' else 1
                start_index = 8 if backend == 'gpu' else 2
                assert int(metadata[count_index]) == 5
                assert int(metadata[start_index]) == 5
    assert_positive(batch(resumed) / 'state.h5')
    (tmp_path / 'result.json').write_text(json.dumps(dict(case=case_name, backend=backend,
        source_statistics=source_statistics, state_maxabs=state_error), indent=2) + '\n')


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
@pytest.mark.parametrize('exact', [False, True], ids=['lean', 'exact'])
def test_channel_wall_normal_partition(tmp_path, backend, exact):
    test_other_case_restart(tmp_path, 'channel', backend, 2, 'y', exact)


@pytest.mark.parametrize('case_name,axis,transport', [
    ('channel', 'y', 'pinned'), ('curve', 'x', 'device-aware')])
def test_other_case_restore_memcheck(tmp_path, case_name, axis, transport):
    args, extra = setup(tmp_path, 'gpu', case_name, axis)
    common = dict(checkpoint_interval=2, buffer_bytes=4096, exact_restart=False,
                  solver_halo_transport=transport, memcheck=True, **extra)
    first, _ = run_case(args, ROOT, 'gpu', 2, 'first', 2, **common)
    restored, _ = run_case(args, ROOT, 'gpu', 2, 'restored', 4, restore=batch(first, 2), **common)
    assert_positive(batch(restored, 4) / 'state.h5')


@pytest.mark.parametrize('case_name', ['hbl', 'sbli'])
@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
def test_air5_exact_restart_and_mean_statistics_io(tmp_path, case_name, backend):
    from run_output_air5_restart_validation import launch, state_sanity
    args = air5_arguments(tmp_path, backend, 'z' if case_name == 'hbl' else 'x')
    args.executable = GPU  # CUDA/AIR5 binary also contains the CPU reference.
    args.case = case_name
    args.mean_statistics = True
    args.sample_interval = 1
    reference, _ = launch(args, backend, 2, 'continuous', 4, interval=2)
    resumed, _ = launch(args, backend, 2, 'resumed', 4, interval=2, restore=batch(reference, 2))
    for name in ('state.h5', 'statistics.h5'):
        compare_fields(batch(reference, 4) / name, batch(resumed, 4) / name)
    disabled, _ = launch(args, backend, 2, 'disabled', 4, interval=2, checkpoint_statistics=False)
    compare_fields(batch(reference, 4) / 'state.h5', batch(disabled, 4) / 'state.h5')
    assert not (batch(disabled, 4) / 'statistics.h5').exists()
    fresh, _ = launch(args, backend, 2, 'fresh_statistics', 4, interval=2,
                      restore=batch(disabled, 2))
    compare_fields(batch(reference, 4) / 'state.h5', batch(fresh, 4) / 'state.h5')
    with h5py.File(batch(fresh, 4) / 'statistics.h5') as state:
        assert int(state['metadata'][1]) == 2 and int(state['metadata'][2]) == 2
    skipped, _ = launch(args, backend, 2, 'skipped_statistics', 4, interval=2,
                        restore=batch(reference, 2), checkpoint_statistics=False)
    compare_fields(batch(reference, 4) / 'state.h5', batch(skipped, 4) / 'state.h5')
    assert not (batch(skipped, 4) / 'statistics.h5').exists()
    sanity = state_sanity(batch(fresh, 4) / 'state.h5')
    (tmp_path / 'result.json').write_text(json.dumps(dict(case=case_name, backend=backend,
        exact=True, steps=4, restart_step=2, mean_samples_after_fresh_restart=2,
        state_bitwise_equal=True, sanity=sanity), indent=2) + '\n')


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
def test_air5_native_wall_and_volume_statistics_fresh_restart(tmp_path, backend):
    from run_output_air5_restart_validation import launch
    args = air5_arguments(tmp_path, backend, 'z')
    args.executable = GPU
    config = """&insitu_run
 enabled=t,statistics=t,render=f,products='air5_walls',
 air5_volume_statistics=t,air5_volume_reduction=t,
 statistics_window=5.d-11,3.5d-10,output_directory='outdat/render',
 host_budget_bytes=67108864,device_budget_bytes=67108864,device_reserve_bytes=1073741824
/
"""
    reference, _ = launch(args, backend, 2, 'continuous', 4, interval=2,
                          insitu_config=config, directory_budget_bytes=256*1024**2)
    disabled, _ = launch(args, backend, 2, 'disabled', 4, interval=2,
                         insitu_config=config, checkpoint_statistics=False, directory_budget_bytes=256*1024**2)
    compare_fields(batch(reference, 4) / 'state.h5', batch(disabled, 4) / 'state.h5')
    resumed, _ = launch(args, backend, 2, 'resumed', 4, interval=2, restore=batch(disabled, 2),
                        insitu_config=config, directory_budget_bytes=256*1024**2)
    compare_fields(batch(reference, 4) / 'state.h5', batch(resumed, 4) / 'state.h5')
    with h5py.File(batch(resumed, 4) / 'statistics.h5') as state:
        assert int(state['metadata'][8]) == 3  # Seed at step two, then steps three/four.
        duration = state['metadata'][9:10].view(np.float64)[0]
        assert abs(duration-1.5e-10) <= 1e-25
        assert 'air5_volume_statistics' in state
        assert np.isfinite(state['air5_volume_statistics/q0008'][...]).all()
    (tmp_path / 'result.json').write_text(json.dumps(dict(case='hbl-native-statistics', backend=backend,
        state_bitwise_equal=True, covered_time=duration), indent=2) + '\n')


def test_air5_lightweight_checkpoint_is_explicitly_rejected(tmp_path):
    from run_output_air5_restart_validation import launch
    args = air5_arguments(tmp_path, 'gpu', 'z')
    args.executable = GPU
    launch(args, 'gpu', 2, 'rejected', 1, exact_restart=False,
           reject='AIR5 checkpoint currently requires exact_restart=t')
