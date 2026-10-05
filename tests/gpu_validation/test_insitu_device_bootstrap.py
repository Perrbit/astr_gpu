"""Component probes only; not a device visualization or MPI acceptance test."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BIN = Path(os.environ.get("ASTR_INSITU_DEVICE_PROBE_BIN", ROOT / "build_insitu_device_probes/bin"))


def test_fp64_cuda_compiler():
    result = subprocess.run([str(BIN / "insitu_device_compiler_probe")],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CUDA compiler probe FP64 max_error=" in result.stdout


def test_device_rk45_matches_original_vtk():
    result = subprocess.run([str(BIN / "insitu_device_rk45_probe")],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "GPU RK45 versus original VTK matched=26" in result.stdout


def test_accepted_length_reference_regression():
    original = subprocess.run([str(BIN / "insitu_streamtracer_length_probe"),
                               "--require-accepted"], capture_output=True,
                              text=True, timeout=60)
    assert original.returncode == 2, original.stdout + original.stderr
    fixed = subprocess.run([str(BIN / "insitu_streamtracer_length_fixed_probe"),
                            "--require-accepted"], capture_output=True,
                           text=True, timeout=60)
    assert fixed.returncode == 0, fixed.stdout + fixed.stderr
    assert "StreamTracer accepted-length audit" in fixed.stdout


def test_device_accepted_length_and_state_resume():
    result = subprocess.run([str(BIN / "insitu_device_trace_probe")],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "exact_state_resume=1 ownership_resume=1" in result.stdout
    assert "subminimum_stop=1 unrelated_errors_rejected=1" in result.stdout
