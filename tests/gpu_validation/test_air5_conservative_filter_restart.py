"""T3-P3 bounded coupled-step, workspace and exact-restart qualification."""
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from air5_checkpoint_contract import FILTER_IDENTITY, read_air5_contract
from run_air5_performance_diagnosis import digest
from run_output_air5_restart_validation import launch, state_sanity
from run_output_restart_validation import compare_fields
from test_air5_source_evaluation_restart import modified_batch
from test_output_air5_conservation import arguments, FINAL


@pytest.fixture(scope='module', params=[(1, 'x'), (2, 'x'), (2, 'y'), (2, 'z')])
def coupled(request, tmp_path_factory):
    ranks, axis = request.param
    root = tmp_path_factory.mktemp(f't3_p3_np{ranks}_{axis}')
    records, cases, seeds = {}, {}, {}
    for backend in ('cpu', 'gpu'):
        args = arguments(root, case='sbli', axis=axis)
        args.air5_filter_mode = 'conservative-flux'
        args.runtime_timeout_seconds = 300
        full, full_size = launch(args, backend, ranks, 'continuous_scalar', 12)
        first, first_size = launch(args, backend, ranks, 'first', 5)
        seed = first/'outdat/new/checkpoints/step000000000005'
        resumed, resumed_size = launch(args, backend, ranks, 'resumed', 12, restore=seed)
        exact = {}
        for name in ('state.h5', 'statistics.h5'):
            exact[name] = compare_fields(full/FINAL/name, resumed/FINAL/name)
        args.filter_workspace = 'full'
        whole, whole_size = launch(args, backend, ranks, 'continuous_full', 12)
        for name in ('state.h5', 'statistics.h5'):
            compare_fields(full/FINAL/name, whole/FINAL/name)
        config = (resumed/FINAL/'air5_config.bin').read_bytes()
        assert config == (full/FINAL/'air5_config.bin').read_bytes()
        assert read_air5_contract(config)['filter_identity'] == FILTER_IDENTITY
        records[backend] = dict(exact_datasets=exact, sanity=state_sanity(full/FINAL/'state.h5'),
            directory_bytes=[full_size, first_size, resumed_size, whole_size],
            scalar_full_bitwise_equal=True)
        cases[backend], seeds[backend] = full, seed
    errors = {}
    with h5py.File(cases['cpu']/FINAL/'state.h5') as cpu, h5py.File(cases['gpu']/FINAL/'state.h5') as gpu:
        for m in range(1, 35):
            name = f'q{m:04}'
            error = float(np.max(abs(cpu[name][:]-gpu[name][:])))
            errors[name] = error
            assert error <= 2e-10, (name, error)
    evidence = dict(executable_sha256=digest(args.executable), ranks=ranks, axis=axis,
        grid_intervals=[16]*3, steps=12, restart_step=5, dt=1e-10,
        filter_identity=list(FILTER_IDENTITY), backends=records, cpu_gpu_max_absolute_errors=errors)
    (root/'summary.json').write_text(json.dumps(evidence, indent=2, allow_nan=False)+'\n')
    return args, ranks, seeds, evidence


def test_coupled_step_and_restart(coupled, record_property):
    record_property('t3_p3', json.dumps(coupled[-1], allow_nan=False))


@pytest.mark.parametrize('fault', ['mode', 'missing', 'partial', 'identity'])
def test_filter_identity_required(coupled, tmp_path, fault):
    original, ranks, seeds, _ = coupled
    if ranks != 1:
        pytest.skip('contract parser is topology independent')
    args = SimpleNamespace(**vars(original))
    args.output = tmp_path
    seed = seeds['gpu']
    config = (seed/'air5_config.bin').read_bytes()
    offset = read_air5_contract(config)['filter_offset']
    message = 'AIR5 filter control contract mismatch'
    if fault == 'mode':
        args.air5_filter_mode = 'legacy'
    else:
        payload = config[:offset] if fault == 'missing' else config[:-1] if fault == 'partial' else config[:-8]+bytes(8)
        seed = modified_batch(seed, tmp_path/'modified', payload)
        if fault == 'partial':
            message = 'AIR5 filter control version/size'
    launch(args, 'gpu', 1, fault, 6, restore=seed, reject=message)


@pytest.mark.parametrize('backend', ['cpu','gpu'])
def test_failed_face_transaction_preserves_q_and_carry(tmp_path, backend):
    args = arguments(tmp_path,case='sbli')
    args.air5_filter_mode='conservative-flux'
    args.mean_statistics=False
    case, _ = launch(args,backend,2,'transaction',1,filter_failure_rank=1,
        reject='injected failure before commit')
    log=(case/'run.log').read_text()
    for rank in (0,1):
        assert f'AIR5_FILTER_TRANSACTION rank={rank} unchanged=T' in log


@pytest.mark.parametrize('backend', ['cpu','gpu'])
def test_default_is_legacy(tmp_path,backend):
    args=arguments(tmp_path,case='sbli')
    args.mean_statistics=False
    args.air5_filter_mode=None
    default,_=launch(args,backend,1,'default',2)
    args.air5_filter_mode='legacy'
    explicit,_=launch(args,backend,1,'legacy',2)
    relative='outdat/new/checkpoints/step000000000002'
    compare_fields(default/relative/'state.h5',explicit/relative/'state.h5')
    assert (default/relative/'air5_config.bin').read_bytes()==(explicit/relative/'air5_config.bin').read_bytes()
