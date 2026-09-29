"""Opt-in CMake integration checks using a source copy without tests/."""

import os
import hashlib
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
# Explicitly approved production addition, pending the next authorized commit.
PENDING_PRODUCTION_INPUTS = ("src_gpu/checkpoint_gpu.cuf",)
COMPILER = os.environ.get("ASTR_TEST_CMAKE_COMPILER")
pytestmark = pytest.mark.skipif(
    not COMPILER, reason="Set ASTR_TEST_CMAKE_COMPILER for CMake integration checks"
)


@pytest.fixture(scope="module")
def source_copy(tmp_path_factory):
    root = tmp_path_factory.mktemp("astr-production-source")
    # Copy tracked production inputs, including pending fixes, but not local runs.
    paths = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "CMakeLists.txt", "src", "src_gpu",
         "user_define_module", "examples"], cwd=ROOT).decode().split("\0")
    for name in sorted(set(filter(None, paths)) | set(PENDING_PRODUCTION_INPUTS)):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
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
def test_release_install_prefixes(source_copy, tmp_path, cuda):
    if os.environ.get("ASTR_TEST_CMAKE_BUILD") != "1":
        pytest.skip("Set ASTR_TEST_CMAKE_BUILD=1 for build/install verification")

    def inventory(root):
        return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob("*") if p.is_file()}

    before = inventory(source_copy)
    build = tmp_path / "build"
    prefix = tmp_path / "configured-install"
    override = tmp_path / "override-install"
    result = configure(source_copy, build, "-DBUILD_TESTING=OFF", "-DCHEMISTRY=OFF",
                       "-DASTR_WITH_AIR5_CHEMISTRY=OFF", f"-DASTR_WITH_CUDA={cuda}",
                       f"-DCMAKE_INSTALL_PREFIX={prefix}")
    (tmp_path / "configure.log").write_text(result.stdout)
    assert result.returncode == 0, result.stdout
    result = subprocess.run(["cmake", "--build", str(build), "--target", "astr", "-j", "2"],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (tmp_path / "build.log").write_text(result.stdout)
    assert result.returncode == 0, result.stdout
    for destination, options in [(prefix, []), (override, ["--prefix", str(override)])]:
        result = subprocess.run(["cmake", "--install", str(build), *options],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (tmp_path / (destination.name+".log")).write_text(result.stdout)
        assert result.returncode == 0, result.stdout
        assert (destination / "bin/astr").is_file()
        for case in ("Taylor_Green_Vortex", "Taylor_Green_Vortex_2D"):
            expected = inventory(source_copy / "examples" / case / "datin")
            assert expected
            assert inventory(destination / "examples" / case / "datin") == expected
        assert inventory(source_copy) == before, "installation modified source tree"

    default_build = tmp_path / "default-build"
    result = configure(source_copy, default_build, "-DBUILD_TESTING=OFF", "-DCHEMISTRY=OFF",
                       "-DASTR_WITH_AIR5_CHEMISTRY=OFF", "-DASTR_WITH_CUDA=OFF")
    (tmp_path / "default-configure.log").write_text(result.stdout)
    assert result.returncode == 0, result.stdout
    script = (default_build / "cmake_install.cmake").read_text()
    assert f'set(CMAKE_INSTALL_PREFIX "{default_build}/opt")' in script


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
