"""CP7: real ASTR checkpoint identity and same-topology compact continuation."""
import json
import os
from pathlib import Path
import shutil
import struct

import h5py
import numpy as np
import pytest

from run_output_restart_validation import run_case, compare_fields
from test_output_repartition_runtime import ROOT, arguments

CPU = os.environ.get('ASTR_COMPACT_CPU_EXE')
GPU = os.environ.get('ASTR_COMPACT_GPU_EXE')
pytestmark = pytest.mark.skipif(not CPU or not GPU, reason='set both rebuilt compact executables')
TAG = b'ASTRCM01'


def launch(output, backend, ranks, name, steps, filtering=True, workspace='scalar', scheme='643c', **kwargs):
    args = arguments(output, workspace=workspace)
    args.executable = Path(GPU if backend == 'gpu' else CPU).resolve()
    args.statistics = False
    args.wall_samples = False
    args.directory_budget_bytes = shutil.disk_usage(output).free
    return run_case(args, ROOT, backend, ranks, name, steps, grid='32,32,32',
                    checkpoint_interval=5, lfilter=filtering, scheme=scheme, **kwargs)[0]


def batch(case, step):
    return case / f'outdat/new/checkpoints/step{step:012d}'


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
@pytest.mark.parametrize('ranks', [1, 2])
@pytest.mark.parametrize('filtering', [False, True])
def test_compact_exact_continuation(tmp_path, backend, ranks, filtering):
    workspace = 'full' if backend == 'gpu' and ranks == 2 else 'scalar'
    reference = launch(tmp_path, backend, ranks, 'continuous', 12, filtering, workspace)
    payload = (batch(reference, 12)/'control.bin').read_bytes()
    assert TAG in payload, 'compact checkpoint lacks operator identity'
    first = launch(tmp_path, backend, ranks, 'first', 5, filtering, workspace)
    resumed = launch(tmp_path, backend, ranks, 'resumed', 12, filtering, workspace,
                     restore=batch(first, 5))
    datasets = compare_fields(batch(reference, 12)/'state.h5', batch(resumed, 12)/'state.h5')
    with h5py.File(batch(resumed, 12)/'state.h5') as state:
        assert state['identity'][8] == 12
        assert abs(state['identity'][9:10].view(np.float64)[0]-0.012) < 1e-16
        for name in (f'q{i:04d}' for i in range(1, 12)):
            assert np.isfinite(state[name][...]).all()
        assert (state['q0001'][...] > 0).all()
    text = (resumed/'run.log').read_text()
    assert 'ASTR_COMPACT_RESTART exact operator identity matched' in text
    assert 'COMPACT GLOBAL:' in text
    assert not any('compact' in p.name.lower() for p in batch(resumed, 12).iterdir())
    (tmp_path/'result.json').write_text(json.dumps(dict(status='pass', backend=backend, ranks=ranks,
        filtering=filtering, workspace=workspace, completed_steps=12, split=[5,7],
        bitwise_datasets=datasets), indent=2)+'\n')


def clone_source(case, tmp_path, fault):
    root = tmp_path/fault
    shutil.copytree(case/'outdat/new', root)
    source = root/'checkpoints/step000000000005'
    path = source/'control.bin'
    data = path.read_bytes()
    offset = data.index(TAG)
    if fault == 'missing':
        data = data[:offset]
    elif fault == 'mismatch':
        data = data[:offset+8]+data[offset+8:].replace(b'global-compact-v1', b'global-compact-v0', 1)
    elif fault == 'coefficient':
        data = data[:-24]+struct.pack('<d', 0.4)+data[-16:]
    else:
        raise ValueError(fault)
    path.write_bytes(data)
    return source


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
@pytest.mark.parametrize('fault', ['missing', 'mismatch', 'coefficient'])
def test_compact_unknown_identity_requires_inexact(tmp_path, backend, fault):
    first = launch(tmp_path, backend, 1, 'first', 5)
    source = clone_source(first, tmp_path, fault)
    launch(tmp_path, backend, 1, 'rejected', 6, restore=source,
           reject='compact operator identity '+('missing' if fault == 'missing' else 'mismatch'))
    resumed = launch(tmp_path, backend, 1, 'inexact', 6, restore=source, exact_restart=False)
    assert 'ASTR_COMPACT_RESTART non-exact' in (resumed/'run.log').read_text()
    with h5py.File(batch(resumed, 6)/'state.h5') as state:
        assert state['identity'][12] == 11
        assert all(np.isfinite(state[f'q{i:04d}'][...]).all() for i in range(1,6))


def test_compact_storage_is_not_operator_identity(tmp_path):
    reference = launch(tmp_path, 'gpu', 2, 'continuous', 12, workspace='scalar')
    first = launch(tmp_path, 'gpu', 2, 'first', 5, workspace='scalar')
    resumed = launch(tmp_path, 'gpu', 2, 'resumed', 12, workspace='full',
                     restore=batch(first, 5))
    compare_fields(batch(reference, 12)/'state.h5', batch(resumed, 12)/'state.h5')


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
def test_explicit_checkpoint_contract_unchanged(tmp_path, backend):
    reference = launch(tmp_path, backend, 2, 'continuous', 12, scheme='643e')
    first = launch(tmp_path, backend, 2, 'first', 5, scheme='643e')
    assert TAG not in (batch(first, 5)/'control.bin').read_bytes()
    resumed = launch(tmp_path, backend, 2, 'resumed', 12, scheme='643e', restore=batch(first, 5))
    compare_fields(batch(reference, 12)/'state.h5', batch(resumed, 12)/'state.h5')


def test_compact_restore_memcheck(tmp_path):
    first = launch(tmp_path, 'gpu', 2, 'first', 5, workspace='full')
    resumed = launch(tmp_path, 'gpu', 2, 'resumed', 12, workspace='full',
                     restore=batch(first, 5), memcheck=True)
    assert batch(resumed, 12).is_dir()
