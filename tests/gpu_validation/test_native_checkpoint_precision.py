"""Native MPI/HDF5 precision identity and conversion, without a CFD advancer."""

import os
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
PROBES = {
    4: ROOT / 'build_native_fp0_fp32_cpu/bin/checkpoint_state_probe',
    8: ROOT / 'build_native_fp0_fp64_gpu/bin/checkpoint_state_probe',
}
MPIEXEC = os.environ.get('ASTR_OUTPUT_MPIEXEC',
    '/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec')


def run(path, width, mode, ranks=2, axis=1, success=True):
    probe = Path(os.environ.get(f'ASTR_CHECKPOINT_FP{width * 8}_PROBE', PROBES[width]))
    if not probe.is_file():
        pytest.skip(f'build native FP{width * 8} checkpoint_state_probe through root CMake')
    env = os.environ.copy()
    env.update(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp',
               OMPI_MCA_osc='pt2pt', OMPI_MCA_coll='^cuda,hcoll,ucc',
               OMPI_MCA_opal_cuda_support='0')
    result = subprocess.run(
        [MPIEXEC, '--oversubscribe', '-n', str(ranks), str(probe), str(path),
         mode, str(axis), '5', '1'], env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=45)
    if success:
        assert result.returncode == 0, result.stdout
        assert 'PASS' in result.stdout
    else:
        assert result.returncode != 0, result.stdout
        assert 'checkpoint state rejected:' in result.stdout
    return result.stdout


@pytest.mark.parametrize('width', [4, 8])
def test_native_exact(tmp_path, width):
    path = tmp_path / 'state.h5'
    run(path, width, 'write')
    run(path, width, 'exact')
    with h5py.File(path) as state:
        assert state['identity'][1] == 3
        assert state['numerical_kind_bytes'][:].tolist() == [width]
        assert state['q0001'].dtype == np.dtype(f'<f{width}')
        assert state['rank_extras'].dtype == np.dtype(f'<f{width}')
        assert np.signbit(state['q0001'][0, 0, 0])


@pytest.mark.parametrize('source,target', [(4, 8), (8, 4)])
def test_cross_precision_is_nonexact(tmp_path, source, target):
    path = tmp_path / 'state.h5'
    run(path, source, 'write')
    before = path.read_bytes()
    log = run(path, target, 'exact', success=False)
    assert 'numerical precision mismatch' in log
    run(path, target, 'repartition', ranks=1)
    run(path, target, 'repartition', ranks=2, axis=2)
    assert path.read_bytes() == before


@pytest.mark.parametrize('target', [4, 8])
def test_published_fp64_version2(tmp_path, target):
    path = tmp_path / 'state.h5'
    run(path, 8, 'write')
    with h5py.File(path, 'r+') as state:
        state['identity'][1] = 2
        del state['numerical_kind_bytes']
    run(path, target, 'exact' if target == 8 else 'repartition')


def test_unpublished_fp32_prototype_is_not_legacy_fp64(tmp_path):
    path = tmp_path / 'state.h5'
    run(path, 4, 'write')
    with h5py.File(path, 'r+') as state:
        state['identity'][1] = 2
        del state['numerical_kind_bytes']
    run(path, 8, 'exact', success=False)
    run(path, 4, 'repartition', success=False)


@pytest.mark.parametrize('defect', ['missing', 'unsupported', 'payload'])
def test_precision_identity_is_enforced(tmp_path, defect):
    path = tmp_path / 'state.h5'
    run(path, 4, 'write')
    with h5py.File(path, 'r+') as state:
        if defect == 'missing':
            del state['numerical_kind_bytes']
        elif defect == 'unsupported':
            state['numerical_kind_bytes'][0] = 16
        else:
            data = state['q0002'][:].astype(np.float64)
            del state['q0002']
            state['q0002'] = data
    run(path, 4, 'exact', success=False)
