"""Exercise the actual Fortran zero-crossing implementation, not a second solver."""
import os
from pathlib import Path
import subprocess

import pytest

PROBE=Path(os.environ.get('ASTR_INSITU_WALL_SEPARATION_PROBE',
    Path(__file__).resolve().parents[2]/'build_insitu_check/bin/insitu_wall_separation_probe'))


def test_synthetic_crossing_gates():
    if not PROBE.is_file():
        pytest.fail('Build insitu_wall_separation_probe through the root CMake first')
    result=subprocess.run([str(PROBE.resolve())],capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'PASS: wall separation synthetic gates' in result.stdout
