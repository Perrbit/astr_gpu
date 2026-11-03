"""Run root-CMake-built wall transport and one-dimensional diagnostic operators."""
import os
from pathlib import Path
import re
import subprocess

import pytest

PROBE = os.environ.get('ASTR_BL_DIAGNOSTIC_PROBE')
MPIEXEC = os.environ.get('ASTR_INSITU_MPIEXEC', '/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpirun')
pytestmark = pytest.mark.skipif(not PROBE, reason='Select the root-CMake-built boundary-layer probe')


@pytest.mark.parametrize('ranks', (1, 2, 4))
def test_wall_and_thickness_operators(tmp_path, ranks, record_property):
    result = subprocess.run([MPIEXEC, '--mca', 'coll_hcoll_enable', '0', '--oversubscribe', '-np', str(ranks),
                             str(Path(PROBE).resolve())], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    errors = re.findall(r'PASS boundary layer diagnostics rank=(\d+) max_scaled_error=\s*(\S+) max_absolute_error=\s*(\S+)',
                        result.stdout)
    assert {int(rank) for rank, _, _ in errors} == set(range(ranks))
    assert max(float(scaled) for _, scaled, _ in errors) <= 2e-10
    record_property('max_scaled_error', max(float(scaled) for _, scaled, _ in errors))
    record_property('max_absolute_error', max(float(absolute) for _, _, absolute in errors))
    assert 'diagnostic_grid_errors=' in result.stdout
