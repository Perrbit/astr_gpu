"""Small synthetic private face-transport tests, not AIR5 physical validation."""
import os
from pathlib import Path
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROBE = Path(os.environ.get('ASTR_INSITU_BOUNDARY_PROBE',
                            ROOT / 'build_insitu_device_render/bin/insitu_device_boundary_probe'))
PREFIX = os.environ.get('ASTR_INSITU_DEVICE_MPI_PREFIX')


def launch(ranks, axis, mode, fixture, scenario=''):
    assert PREFIX, 'Set ASTR_INSITU_DEVICE_MPI_PREFIX to the matching MPI installation'
    env = dict(os.environ, OMPI_MCA_pml='ucx', OMPI_MCA_coll='^hcoll,ucc,cuda',
               OMPI_MCA_coll_hcoll_enable='0', UCX_MEMTYPE_CACHE='n',
               UCX_CUDA_COPY_ENABLE_FABRIC='no', UCX_CUDA_COPY_DMABUF='no',
               UCX_CUDA_IPC_ENABLE_MNNVL='no', UCX_TLS='self,sm,cuda_copy,cuda_ipc',
               ASTR_GPU_HALO_TRANSPORT='device-aware' if mode == 'pinned' else 'pinned')
    result = subprocess.run([str(Path(PREFIX) / 'bin/mpirun'), '--prefix', PREFIX,
                             '-np', str(ranks), str(PROBE), str(axis), mode, fixture, scenario],
                            env=env, capture_output=True, text=True, timeout=60)
    print(result.stdout + result.stderr)
    return result


@pytest.mark.parametrize('ranks,axis', [(1, 1), (2, 1), (2, 2), (2, 3)])
@pytest.mark.parametrize('mode', ['device-aware', 'pinned'])
@pytest.mark.parametrize('fixture', ['channel', 'air5', 'physical'])
def test_physical_faces(ranks, axis, mode, fixture):
    result = launch(ranks, axis, mode, fixture)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert 'PASS physical endpoints, halos, nine metrics, source and slot lifecycle' in output
    assert all(float(value) <= 2e-10 for value in re.findall(r'max_error=\s*(\S+)', output))
    assert 'errors=0' in output


def test_periodicity_requires_collective_agreement():
    result = launch(2, 1, 'pinned', 'channel', 'inconsistent-periodic')
    assert result.returncode != 0, result.stdout + result.stderr
    assert 'transport periodicity/components agreement' in result.stdout + result.stderr


@pytest.mark.parametrize('ranks,axis', [(1, 1), (2, 1), (2, 2), (2, 3)])
@pytest.mark.parametrize('mode', ['device-aware', 'pinned'])
@pytest.mark.parametrize('fixture', ['channel', 'physical'])
@pytest.mark.parametrize('scenario', ['wall-sixth', 'wall-affine'])
def test_private_sixth_order_walls(ranks, axis, mode, fixture, scenario, record_property):
    result = launch(ranks, axis, mode, fixture, scenario)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert 'PASS physical endpoints, halos, nine metrics, source and slot lifecycle' in output
    errors = [float(value) for value in re.findall(r'max_error=\s*(\S+)', output)]
    assert len(errors) == ranks and max(errors) <= 2e-10, output
    record_property('private_sixth_gradient_q_maxabs', max(errors))


@pytest.mark.parametrize('scenario,message', [
    ('coordinate-nonfinite', 'nonfinite physical coordinate'),
    ('coordinate-budget', 'coordinate snapshot retained budget'),
])
def test_coordinate_rejection(scenario, message):
    result = launch(2, 1, 'pinned', 'channel', scenario)
    assert result.returncode != 0, result.stdout + result.stderr
    assert message in result.stdout + result.stderr
