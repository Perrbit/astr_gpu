"""Bounded Mach-2 checks for the minimum halo-safe 2D extrusion, not HCW physics."""
import numpy as np
import pytest

from test_sbli_joint_nscbc import EXE, compare_cpu_gpu, fields, launch

pytestmark = pytest.mark.skipif(not EXE, reason="Set ASTR_SBLI_JOINT_EXE to a root-CMake FP64 executable")


@pytest.mark.parametrize("scheme", ["543e", "743e"])
def test_five_periodic_intervals_match_eight_and_pass_memcheck(tmp_path, scheme):
    cpu = launch(tmp_path / "cpu_k5", "cpu", scheme, 12, (1, 1, 1), km=5)
    gpu = launch(tmp_path / "gpu_k5", "gpu", scheme, 12, (1, 1, 1), km=5, memcheck=True)
    compare_cpu_gpu(cpu, gpu)
    reference = launch(tmp_path / "gpu_k8", "gpu", scheme, 12, (1, 1, 1), km=8)
    minimum, baseline = fields(gpu, 12), fields(reference, 12)
    assert minimum["q"].shape[:3] == (6, 49, 65)
    for name, expected in baseline.items():
        actual = minimum[name]
        plane_error = np.max(np.abs(actual[0] - expected[0]) / (1.0 + np.abs(expected[0])))
        extrusion_error = np.max(np.abs(actual - actual[0]) / (1.0 + np.abs(actual[0])))
        assert plane_error <= 2.0e-10, (name, plane_error)
        assert extrusion_error <= 2.0e-10, (name, extrusion_error)
