"""Native analysis components only; not a claim of full FP32 rendering support."""

import os
from pathlib import Path
import re
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
BUILD32 = Path(os.environ.get('ASTR_NATIVE_INSITU_FP32_BUILD', ROOT / 'build_native_fp7_fp32_device'))
BUILD64 = Path(os.environ.get('ASTR_NATIVE_INSITU_FP64_BUILD', ROOT / 'build_insitu_device_render'))


def execute(command):
    result = subprocess.run([str(value) for value in command], text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.parametrize('width,build', [(4, BUILD32), (8, BUILD64)])
def test_native_bridge_real_array_abi(width, build, record_property):
    output = execute([build / 'bin/native_insitu_abi_probe'])
    assert f'Fortran/C++ ABI bytes={width}' in output
    record_property('probe', output.strip())


@pytest.mark.parametrize('target', ['native_precision_rk45_probe', 'native_precision_rk45_gpu_probe'])
def test_native_fp32_rk45(target, record_property):
    output = execute([BUILD32 / 'bin' / target])
    assert 'PASS native RK45 bytes=4' in output
    record_property('probe', output.strip())


def test_fp64_rk45_matches_original_vtk(record_property):
    step = execute([BUILD64 / 'bin/insitu_device_rk45_probe'])
    trace = execute([BUILD64 / 'bin/insitu_device_trace_probe'])
    assert 'matched=26 max_error=0' in step
    assert 'counter_error=0' in trace and 'exact_state_resume=1' in trace
    record_property('step', step.strip())
    record_property('trace', trace.strip())


def test_fp32_kernel_has_no_fp64_arithmetic(tmp_path, record_property):
    tool = Path('/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/13.1/bin/cuobjdump')
    pattern = r'\b(?:DADD|DMUL|DFMA|DSETP|DSET|DMNMX)\b|\b(?:F2F|F2I|I2F)\.[^;\n]*\bF64\b'
    counts = {}
    for width, build in [(4, BUILD32), (8, BUILD64)]:
        output = execute([tool, '--dump-sass', build / 'bin/native_precision_rk45_gpu_probe'])
        (tmp_path / f'fp{width * 8}.sass').write_text(output)
        assert 'Function :' in output and 'check_kernel' in output
        counts[width] = len(re.findall(pattern, output))
    assert counts[4] == 0 and counts[8] > 0
    record_property('fp32_fp64_instruction_count', counts[4])
    record_property('fp64_fp64_instruction_count', counts[8])
