"""Bounded device snapshot tests for independent, fixed-slot face transport."""
import os
from pathlib import Path
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROBE = Path(os.environ.get('ASTR_INSITU_VELOCITY_PROBE', ROOT / 'build_insitu_gpu/bin/insitu_device_velocity_probe'))
PREFIX = os.environ.get('ASTR_INSITU_DEVICE_MPI_PREFIX')


def launch(ranks, axis, mode, scenario='', probe=PROBE, mapping=''):
    assert PREFIX, 'Set ASTR_INSITU_DEVICE_MPI_PREFIX to the matching MPI installation'
    env = dict(os.environ, OMPI_MCA_pml='ucx', OMPI_MCA_coll='^hcoll,ucc,cuda',
               OMPI_MCA_coll_hcoll_enable='0', UCX_MEMTYPE_CACHE='n',
               UCX_CUDA_COPY_ENABLE_FABRIC='no', UCX_CUDA_COPY_DMABUF='no',
               UCX_CUDA_IPC_ENABLE_MNNVL='no', UCX_TLS='self,sm,cuda_copy,cuda_ipc',
               ASTR_GPU_HALO_TRANSPORT='device-aware' if mode == 'pinned' else 'pinned')
    return subprocess.run([str(Path(PREFIX) / 'bin/mpirun'), '--prefix', PREFIX,
                           '-np', str(ranks), str(probe), str(axis), mode, scenario, str(mapping)],
                          env=env, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize('ranks,axis', [(1, 1), (2, 1), (2, 2), (2, 3)])
@pytest.mark.parametrize('mode', ['device-aware', 'pinned'])
def test_fixed_slots_snapshot(ranks, axis, mode):
    result = launch(ranks, axis, mode)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert 'mismatches=0 source_unchanged=1' in output
    assert 'PASS transport slots reused, released and rebuilt' in output
    match = re.search(r'frames=3 generation=1 host_bytes=(\d+) device_bytes=(\d+) d2h_bytes=(\d+) h2d_bytes=(\d+)', output)
    assert match, output
    host, device, d2h, h2d = map(int, match.groups())
    capacity_bytes = 4 * 33 * 33 * 8
    slot_bytes = 9 * 39 * 39 * 8
    assert device == 2 * slot_bytes
    assert host == (2 * slot_bytes if mode == 'pinned' and ranks == 2 else 0)
    expected = 3 * capacity_bytes if mode == 'pinned' and ranks == 2 else 0
    assert d2h == h2d == expected


@pytest.mark.parametrize('mode,scenario,message', [
    ('', '', 'explicit consistent transport required'),
    ('automatic', '', 'explicit consistent transport required'),
    ('pinned', 'inconsistent', 'explicit consistent transport required'),
    ('pinned', 'host-budget', 'transport face-buffer budget'),
    ('pinned', 'node-host-budget', 'transport node face-buffer budget'),
    ('device-aware', 'device-budget', 'transport face-buffer budget'),
    ('pinned', 'nested', 'transport shape, lifecycle or budget'),
    ('device-aware', 'shape', 'snapshot transport identity/shape changed'),
])
def test_transport_rejection(mode, scenario, message):
    result = launch(2, 1, mode, scenario)
    assert result.returncode != 0, result.stdout + result.stderr
    assert message in result.stdout + result.stderr


@pytest.mark.parametrize('ranks,axis', [(1, 1), (2, 1), (2, 2), (2, 3)])
@pytest.mark.parametrize('mode', ['device-aware', 'pinned'])
def test_periodic_halo_and_q(ranks, axis, mode):
    result = launch(ranks, axis, mode, probe=PROBE.with_name('insitu_device_fields_probe'))
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert 'halo_errors=0 source_errors=0' in output
    match = re.search(r'max_error=\s*(\S+)', output)
    assert match and float(match.group(1)) <= 2e-10, output


@pytest.mark.parametrize('ranks,axis', [(1, 1), (2, 1), (2, 2), (2, 3)])
@pytest.mark.parametrize('mode', ['device-aware', 'pinned'])
@pytest.mark.parametrize('mapping', [1, 2], ids=['periodic', 'y-wavy'])
def test_physical_coordinate_halo(ranks, axis, mode, mapping, record_property):
    result = launch(ranks, axis, mode, 'coordinates', mapping=mapping)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert f'Physical coordinate halo mapping={mapping}' in output, output
    assert 'mismatches=0 source_unchanged=1 outer_wrap=0 frames=3' in output, output
    match = re.search(r'coordinate_d2h_bytes=(\d+) coordinate_h2d_bytes=(\d+)', output)
    assert match, output
    expected = 3 * 9 * 39 * 39 * 8 if ranks == 2 and mode == 'pinned' else 0
    assert int(match[1]) == int(match[2]) == expected, output
    record_property('coordinate_halo_output', output)
