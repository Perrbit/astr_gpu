"""X4 cutting primitive, not solver admission or whole-mesh topology proof."""
import os
from pathlib import Path
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
BIN = Path(os.environ.get('ASTR_INSITU_DEVICE_PLANE_PROBE',
    ROOT / 'build_insitu_device_render/bin/insitu_device_plane_probe'))


@pytest.mark.parametrize('ranks,axis', [(1, 0), (2, 0), (2, 1), (2, 2)])
def test_plane_core(ranks, axis, tmp_path):
    prefix = os.environ.get('ASTR_INSITU_DEVICE_MPI_PREFIX')
    assert prefix, 'Set ASTR_INSITU_DEVICE_MPI_PREFIX'
    result = subprocess.run([str(Path(prefix) / 'bin/mpirun'), '--prefix', prefix,
        '--mca', 'pml', 'ob1', '--mca', 'btl', 'self,tcp', '--mca', 'osc', 'pt2pt',
        '--mca', 'coll_hcoll_enable', '0', '--mca', 'coll_ucc_enable', '0',
        '--mca', 'opal_cuda_support', '0', '-np', str(ranks), str(BIN),str(axis)],
        cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout, end='')
    rows = [line for line in result.stdout.splitlines() if line.startswith('X4_PLANE_CORE ')]
    assert len(rows) == ranks, result.stdout
    for line in rows:
        values = dict(re.findall(r'(\w+)=([^\s]+)', line))
        assert int(values['cases']) == 168
        for name in ('invalid', 'topology_errors', 'winding_errors', 'zero_area',
                     'identity_errors', 'source_errors'):
            assert int(values[name]) == 0, line
        assert float(values['plane_error']) <= 2e-10
        assert float(values['field_error']) <= 2e-10
        assert values['test_geometry_readback'] == '1'
        assert values['runtime_admission'] == '0'
    assert not list(tmp_path.iterdir())
    meshes = [line for line in result.stdout.splitlines() if line.startswith('X4_PLANE_MESH ')]
    assert len(meshes) == 4, result.stdout
    for line in meshes:
        values = dict(re.findall(r'(\w+)=([^\s]+)', line))
        for name in ('invalid', 'topology_errors', 'winding_errors', 'zero_area', 'identity_errors'):
            assert int(values[name]) == 0, line
        for name in ('plane_error', 'field_error', 'area_error'):
            assert float(values[name]) <= 2e-10, line
        assert int(values['triangles']) > 0 and int(values['points']) > 0
