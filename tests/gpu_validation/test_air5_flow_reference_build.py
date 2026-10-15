"""Test-only tighter CPU references must not alter the production presets."""
from pathlib import Path
import subprocess

import pytest

from run_air5_flow_control_gate import CONTROLS

ROOT = Path(__file__).resolve().parents[2]
COMPILER = '/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/nvfortran'


@pytest.mark.parametrize('level', [None, 11, 12])
def test_reference_preprocessing_is_opt_in_and_has_distinct_checkpoint_id(level):
    command = [COMPILER, '-Mpreprocess', '-E']
    if level is not None:
        command += [f'-DASTR_AIR5_REFERENCE_LEVEL={level}']
    text = subprocess.check_output([*command, str(ROOT / 'src/chemistry_runtime.F90')], text=True)
    if level is None:
        assert 'AIR5_TEST_REFERENCE' not in text
        assert 'air5_tolerance_choice=local(1)' in text
        assert 'air5_chemistry_rtol=1.0e-11_real64' not in text
        assert 'air5_chemistry_rtol=1.0e-12_real64' not in text
    else:
        assert f'air5_tolerance_choice={level}' in text
        assert f'air5_chemistry_rtol=1.0e-{level}_real64' in text
        assert f'air5_chemistry_atol_factor=1.0e-{level+4}_real64' in text
        assert 'if(any(local(1:2)/=0)) local(3)=1' in text


def test_reference_build_is_cpu_air5_test_only_and_does_not_add_candidate_controls():
    cmake = (ROOT / 'src/CMakeLists.txt').read_text()
    assert 'set(ASTR_AIR5_REFERENCE_LEVEL "off" CACHE STRING' in cmake
    assert 'NOT BUILD_TESTING OR NOT ASTR_WITH_AIR5_CHEMISTRY OR ASTR_WITH_CUDA' in cmake
    assert CONTROLS == (('baseline', 'off'), ('loose', 'off'), ('tight', 'off'),
                        ('baseline', 'previous'), ('loose', 'previous'))
