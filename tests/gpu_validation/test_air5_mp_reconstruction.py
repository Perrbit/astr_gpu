"""Exercise the production pure Fortran MP helpers, not a Python reimplementation."""
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
RECORDED_STENCIL = np.array([
    2.230535679136144e-14, -1.1212422454719421e-12,
    -6.060399899679019e-13, 1.0968519751849646e-12,
    -2.155979738964063e-12, 1.072271952788764e-12,
    7.336413162521745e-13,
])


def helper(source: str, name: str) -> str:
    pattern = rf"^  (?:pure|attributes\(device\)) real\(real64\) function {name}\(.*?^  end function {name}\b"
    match = re.search(pattern, source, re.MULTILINE | re.DOTALL)
    assert match is not None, name
    return match.group(0)


@pytest.fixture(scope="module")
def reconstruct(tmp_path_factory):
    compiler = shutil.which("gfortran")
    if compiler is None:
        pytest.skip("gfortran is required for the isolated pure-function probe")
    directory = tmp_path_factory.mktemp("air5_mp")
    source = (ROOT / "src/chemistry_solver.F90").read_text()
    functions = "\n".join(helper(source, name) for name in (
        "air5_minmod2", "air5_minmod4", "air5_mp5", "air5_mp7"))
    probe = directory / "probe.f90"
    probe.write_text("""program probe
  use iso_fortran_env, only: real64
  implicit none
  real(real64) :: values(7)
  integer :: status
  do
    read(*,*,iostat=status) values
    if(status<0) exit
    if(status/=0) error stop 'invalid probe input'
    write(*,'(2(es26.17e3,1x))') air5_mp5(values(2:6)),air5_mp7(values)
  enddo
contains
""" + functions + "\nend program probe\n")
    executable = directory / "probe"
    subprocess.run([compiler, "-std=f2008", "-O0", "-fcheck=all",
                    "-ffpe-trap=invalid,zero,overflow", str(probe),
                    "-o", str(executable)], check=True, capture_output=True, text=True)

    def evaluate(stencils):
        rows = np.atleast_2d(stencils)
        result = subprocess.run([str(executable)], input="\n".join(
            " ".join(f"{value:.17e}" for value in row) for row in rows) + "\n",
            check=True, capture_output=True, text=True)
        return np.array([[float(value) for value in row.split()]
                         for row in result.stdout.splitlines()])

    return evaluate


def test_recorded_tiny_stencil_uses_full_mp_bounds(reconstruct):
    result = reconstruct(RECORDED_STENCIL)[0]
    np.testing.assert_allclose(result, RECORDED_STENCIL[3], rtol=2e-14, atol=0)


def test_constants_linear_fields_and_smooth_extrema(reconstruct):
    x = np.arange(-3.0, 4.0)
    stencils = np.array([np.zeros(7), np.full(7, 2.0),
                         2.0 + 0.25*x, 2.0 - x*x, x*x - 2.0])
    expected = np.array([0.0, 2.0, 2.125, 2.0-1.0/6.0, 1.0/6.0-2.0])
    np.testing.assert_allclose(reconstruct(stencils),
                               np.repeat(expected[:, None], 2, axis=1),
                               rtol=2e-14, atol=0)


def test_reconstruction_is_homogeneous_across_flux_scales(reconstruct):
    rng = np.random.default_rng(20260923)
    stencils = np.vstack([RECORDED_STENCIL / np.max(abs(RECORDED_STENCIL)),
                           rng.uniform(-1, 1, (128, 7))])
    expected = reconstruct(stencils)
    for scale in (1e-200, 1e-12, 1e-6, 1.0, 1e6, 1e150, -1e-200, -1e12):
        np.testing.assert_allclose(reconstruct(stencils * scale) / scale,
                                   expected, rtol=2e-13, atol=2e-14)


def test_cpu_gpu_helper_algebra_matches():
    cpu = (ROOT / "src/chemistry_solver.F90").read_text()
    gpu = (ROOT / "src_gpu/chemistry_solver_gpu.cuf").read_text()
    for name in ("air5_minmod2", "air5_minmod4", "air5_mp5", "air5_mp7"):
        cpu_body = helper(cpu, name)
        gpu_body = helper(gpu, name + "_gpu").replace("_gpu", "")
        gpu_body = gpu_body.replace("attributes(device)", "pure").replace(", value", "")
        # Device helpers use the common seven-point scratch buffer for both orders.
        cpu_body = cpu_body.replace(", intent(in)", "")
        if name == "air5_mp5":
            cpu_body = cpu_body.replace(":: values(5)", ":: values(7)")
        strip = lambda text: "".join(re.sub(r"!.*", "", text).split())
        assert strip(cpu_body) == strip(gpu_body)
