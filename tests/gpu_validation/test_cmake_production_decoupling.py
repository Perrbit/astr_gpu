"""Opt-in CMake integration checks using a source copy without tests/."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
COMPILER = os.environ.get("ASTR_TEST_CMAKE_COMPILER")
pytestmark = pytest.mark.skipif(
    not COMPILER, reason="Set ASTR_TEST_CMAKE_COMPILER for CMake integration checks"
)


@pytest.fixture(scope="module")
def source_copy(tmp_path_factory):
    root = tmp_path_factory.mktemp("astr-production-source")
    shutil.copy2(ROOT / "CMakeLists.txt", root)
    for name in ("src", "src_gpu", "user_define_module", "examples"):
        shutil.copytree(ROOT / name, root / name)
    assert not (root / "tests").exists()
    return root


def configure(source, build, *options):
    result = subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build),
         f"-DCMAKE_Fortran_COMPILER={COMPILER}", *options],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return result


@pytest.mark.parametrize("cuda", ["OFF", "ON"])
@pytest.mark.parametrize("air5", ["OFF", "ON"])
def test_production_without_test_sources(source_copy, tmp_path, cuda, air5):
    build = tmp_path / "build"
    result = configure(
        source_copy, build, "-DBUILD_TESTING=OFF",
        f"-DASTR_WITH_CUDA={cuda}", f"-DASTR_WITH_AIR5_CHEMISTRY={air5}",
    )
    assert result.returncode == 0, result.stdout
    targets = (build / "CMakeFiles/TargetDirectories.txt").read_text()
    assert "probe.dir" not in targets
    assert "halo_exchange_contract_test.dir" not in targets
    flags = (build / "src/CMakeFiles/astr.dir/flags.make").read_text()
    assert "ASTR_BUILD_TESTING" not in flags
    assert ("ASTR_AIR5_CHEMISTRY" in flags) == (air5 == "ON")
    if cuda == "ON":
        cache = (build / "CMakeCache.txt").read_text()
        assert "ASTR_HAS_MPI_CUDA_QUERY:INTERNAL=" in cache
        if "ASTR_HAS_MPI_CUDA_QUERY:INTERNAL=1" in cache:
            assert "ASTR_MPI_CUDA_AWARE_QUERY" in flags
    if os.environ.get("ASTR_TEST_CMAKE_BUILD") == "1":
        result = subprocess.run(
            ["cmake", "--build", str(build), "--target", "astr", "-j", "2"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        assert result.returncode == 0, result.stdout


def test_trace_requires_test_sources(source_copy, tmp_path):
    result = configure(
        source_copy, tmp_path / "build", "-DBUILD_TESTING=OFF",
        "-DASTR_WITH_CUDA=ON", "-DASTR_BUILD_MPI_COMPLETION_TRACE=ON",
    )
    assert result.returncode != 0
    assert "ASTR_BUILD_MPI_COMPLETION_TRACE requires BUILD_TESTING=ON" in result.stdout


@pytest.mark.parametrize("cuda", ["OFF", "ON"])
def test_development_targets_remain_available(tmp_path, cuda):
    build = tmp_path / "build"
    result = configure(
        ROOT, build, "-DBUILD_TESTING=ON", f"-DASTR_WITH_CUDA={cuda}",
        "-DASTR_WITH_AIR5_CHEMISTRY=ON",
    )
    assert result.returncode == 0, result.stdout
    targets = (build / "CMakeFiles/TargetDirectories.txt").read_text()
    assert "boundary_contract_cpu_probe.dir" in targets
    flags = (build / "src/CMakeFiles/astr.dir/flags.make").read_text()
    assert "ASTR_BUILD_TESTING" in flags
    if cuda == "ON":
        flags = (build / "src/CMakeFiles/halo_exchange_contract_test.dir/flags.make").read_text()
        assert "ASTR_BUILD_TESTING" in flags
        assert "ASTR_AIR5_CHEMISTRY" in flags
    if os.environ.get("ASTR_TEST_CMAKE_BUILD") == "1":
        result = subprocess.run(
            ["cmake", "--build", str(build), "--target", "astr",
             "boundary_contract_cpu_probe", "-j", "2"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        assert result.returncode == 0, result.stdout
