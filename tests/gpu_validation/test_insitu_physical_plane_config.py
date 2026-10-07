"""Physical plane identity is agreed before rank-zero normalization."""
import subprocess
from pathlib import Path

import numpy as np
import pytest

from test_insitu_run_config import PROBE, COLLECTIVE, MPIEXEC, DEVICE


PLANE = DEVICE.replace('statistics=t', 'statistics=f').replace('step_interval=2',
    "step_interval=2,products='velocity_slice',slice_definition='plane',"
    'slice_origin=0.5,0.,0.,slice_normal=1.,0.25,-0.125')


@pytest.mark.skipif(not PROBE, reason='Set ASTR_INSITU_CONFIG_PROBE')
@pytest.mark.parametrize('text,accepted', [
    (PLANE, True),
    (PLANE.replace('slice_normal=1.,0.25,-0.125', 'slice_normal=0.,0.,0.'), False),
    (PLANE.replace('slice_origin=0.5,0.,0.', 'slice_origin=NaN,0.,0.'), False),
    (PLANE.replace('slice_normal=1.,0.25,-0.125', 'slice_normal=Inf,0.,0.'), False),
    (PLANE.replace("slice_definition='plane'", "slice_definition='index'"), False),
    (PLANE.replace("slice_definition='plane'", "slice_definition='unknown'"), False),
    (PLANE.replace('statistics=f', 'statistics=t'), False),
    (PLANE.replace("products='velocity_slice'", "products='all'"), False),
    (PLANE.replace("slice_definition='plane'", "slice_definition='plane',slice_index=16"), False),
    (PLANE.replace('step_interval=2', "step_interval=2,rendering_pipeline='compatible'"), False),
    (PLANE.replace('step_interval=2',
        "product_ids='velocity_slice.image',product_modes='steps',product_steps=2"), False),
])
def test_physical_plane_parser(tmp_path, text, accepted):
    path = tmp_path / 'plane.nml'
    path.write_text(text)
    result = subprocess.run([str(Path(PROBE).resolve()), str(path)],
                            capture_output=True, text=True, timeout=10)
    assert (result.returncode == 0) == accepted, result.stdout + result.stderr


@pytest.mark.skipif(not COLLECTIVE or not MPIEXEC, reason='Set collective probe and MPI launcher')
@pytest.mark.parametrize('normal', ['1.,0.25,-0.125', '-1.,-0.25,0.125',
    '1.d300,2.5d299,-1.25d299', '1.d-300,2.5d-301,-1.25d-301', '-1.,1.,0.'])
def test_root_canonical_plane(tmp_path, normal):
    text = PLANE.replace('slice_normal=1.,0.25,-0.125', 'slice_normal=' + normal)
    for rank in range(2):
        (tmp_path / f'rank{rank}.nml').write_text(text)
    result = subprocess.run([MPIEXEC, '-np', '2', str(Path(COLLECTIVE).resolve()), str(tmp_path / 'rank')],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [np.asarray(line.split()[3:], dtype=float) for line in result.stdout.splitlines()
            if line.startswith('PLANE rank ')]
    assert len(rows) == 2 and all(row.size == 6 for row in rows), result.stdout
    np.testing.assert_array_equal(rows[0], rows[1])
    np.testing.assert_array_equal(rows[0][:3], [.5, 0., 0.])
    assert abs(np.linalg.norm(rows[0][3:]) - 1.) <= 4 * np.finfo(float).eps
    assert rows[0][3 + np.argmax(abs(rows[0][3:]))] > 0.


@pytest.mark.skipif(not COLLECTIVE or not MPIEXEC, reason='Set collective probe and MPI launcher')
@pytest.mark.parametrize('other', [
    PLANE.replace('slice_origin=0.5,0.,0.', 'slice_origin=0.6,0.,0.'),
    PLANE.replace('slice_normal=1.,0.25,-0.125', 'slice_normal=-1.,-0.25,0.125'),
    PLANE.replace('slice_normal=1.,0.25,-0.125', 'slice_normal=1.,0.3,-0.125'),
])
def test_raw_plane_rank_mismatch(tmp_path, other):
    (tmp_path / 'rank0.nml').write_text(PLANE)
    (tmp_path / 'rank1.nml').write_text(other)
    result = subprocess.run([MPIEXEC, '-np', '2', str(Path(COLLECTIVE).resolve()), str(tmp_path / 'rank')],
                            capture_output=True, text=True, timeout=30)
    output = result.stdout + result.stderr
    assert result.returncode != 0 and output.count('REJECT rank ') == 2, output
    assert 'configuration values differ' in output


@pytest.mark.skipif(not COLLECTIVE or not MPIEXEC, reason='Set collective probe and MPI launcher')
def test_unrepresentable_unit_component_is_rejected(tmp_path):
    text = PLANE.replace('slice_normal=1.,0.25,-0.125', 'slice_normal=1.d300,1.d-300,0.')
    for rank in range(2):
        (tmp_path / f'rank{rank}.nml').write_text(text)
    result = subprocess.run([MPIEXEC, '-np', '2', str(Path(COLLECTIVE).resolve()), str(tmp_path / 'rank')],
                            capture_output=True, text=True, timeout=30)
    output = result.stdout + result.stderr
    assert result.returncode != 0 and output.count('REJECT rank ') == 2, output
    assert 'unit normal is not representable' in output
