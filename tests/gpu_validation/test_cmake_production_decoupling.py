"""Opt-in CMake integration checks using a source copy without tests/."""

import os
import hashlib
import re
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
# Explicitly approved production addition, pending the next authorized commit.
PENDING_PRODUCTION_INPUTS = (
    "src_gpu/checkpoint_gpu.cuf", "src/insitu_runtime.F90", "src/catalyst_adapter.cpp",
    "src/insitu_resource_budget.F90",
    "src/insitu_run_config.F90", "src/insitu_config_collective.F90",
    "src/insitu_fields.F90", "src_gpu/insitu_sample_gpu.cuf",
    "src/insitu_session.F90", "src/insitu_schedule.F90", "src/insitu_checkpoint_batch.F90",
    "src/insitu_time_integral.F90", "src/insitu_velocity_statistics.F90", "src/insitu_spatial_statistics.F90",
    "src_gpu/insitu_statistics_gpu.cuf",
    "src/insitu_device_map.cpp",
    "src/insitu_mesh_adapter.cpp",
    "src/insitu_resource_observer.cpp",
    "scripts/insitu/tgv_pipeline.py", "scripts/insitu/tgv_streamlines.py",
    "scripts/insitu/egl_identity.py",
    "scripts/insitu/image_publication.py", "scripts/insitu/device_render_pipeline.py",
    "src/adaptive_output.F90", "src/insitu_product_schedule.F90",
    "src/insitu_product_dispatch.h", "scripts/insitu/product_dispatch.py",
    "src_gpu/insitu_wall_fields_gpu.cuf", "src_gpu/insitu_device_wall.h",
    "src_gpu/insitu_device_plane.h", "src_gpu/insitu_device_structured.h",
    "src_gpu/insitu_device_curve_trace.h",
    "src_gpu/insitu_plane_products.cu",
)
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
    assert "astr_catalyst_adapter.dir" not in targets
    cache = (build / "CMakeCache.txt").read_text()
    assert "ASTR_WITH_CATALYST:BOOL=OFF" in cache
    assert "catalyst_DIR:" not in cache
    assert "CMAKE_CXX_COMPILER:" not in cache
    flags = (build / "src/CMakeFiles/astr.dir/flags.make").read_text()
    assert "ASTR_BUILD_TESTING" not in flags
    rules = (build / "src/CMakeFiles/astr.dir/build.make").read_text()
    assert "insitu_sample_validation" not in rules
    assert "insitu_fields.F90" in rules
    assert ("insitu_sample_gpu.cuf" in rules) == (cuda == "ON")
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


def test_explicit_cuda_architecture(source_copy, tmp_path):
    result = configure(source_copy, tmp_path / "build", "-DBUILD_TESTING=OFF",
                       "-DASTR_WITH_CUDA=ON", "-DASTR_CUDA_ARCHITECTURES=80;89")
    assert result.returncode == 0, result.stdout
    flags = (tmp_path / "build/src/CMakeFiles/astr.dir/flags.make").read_text()
    assert "-gpu=rdc,cc80,cc89" in flags
    cache = (tmp_path / "build/CMakeCache.txt").read_text()
    assert "CMAKE_CUDA_ARCHITECTURES:STRING=80;89" in cache


def test_invalid_cuda_architecture_is_rejected(source_copy, tmp_path):
    result = configure(source_copy, tmp_path / "build", "-DBUILD_TESTING=OFF",
                       "-DASTR_WITH_CUDA=ON", "-DASTR_CUDA_ARCHITECTURES=native")
    assert result.returncode != 0
    assert "ASTR_CUDA_ARCHITECTURES requires numeric compute capabilities" in result.stdout


@pytest.mark.parametrize("cuda", ["OFF", "ON"])
def test_native_catalyst_without_test_sources(source_copy, tmp_path, cuda):
    catalyst = os.environ.get("ASTR_TEST_CATALYST_DIR")
    if not catalyst:
        pytest.skip("Set ASTR_TEST_CATALYST_DIR for native Catalyst configuration checks")
    build = tmp_path / "build"
    result = configure(source_copy, build, "-DBUILD_TESTING=OFF",
                       "-DASTR_WITH_CATALYST=ON", f"-DASTR_WITH_CUDA={cuda}",
                       f"-Dcatalyst_DIR={catalyst}")
    assert result.returncode == 0, result.stdout
    targets = (build / "CMakeFiles/TargetDirectories.txt").read_text()
    assert "astr_catalyst_adapter.dir" in targets
    assert "insitu_allocation_fault.dir" not in targets
    assert "insitu_device_map_probe.dir" not in targets
    rules = (build / "src/CMakeFiles/astr_catalyst_adapter.dir/build.make").read_text()
    assert "insitu_mesh_adapter.cpp" in rules
    assert str(source_copy / "tests") not in rules
    assert ("insitu_device_map.cpp" in rules) == (cuda == "ON")
    install = (build / "src/cmake_install.cmake").read_text()
    for name in ("tgv_pipeline.py", "tgv_streamlines.py", "egl_identity.py"):
        assert name in install
    assert "insitu_allocation_fault" not in install


@pytest.mark.parametrize('air5', ('OFF', 'ON'))
def test_strict_device_products_without_test_sources(source_copy, tmp_path, air5):
    cache_path = os.environ.get('ASTR_TEST_RESIDENT_CMAKE_CACHE')
    if not cache_path:
        pytest.skip('Select the validated strict-render dependency CMake cache')
    cache = {}
    for line in Path(cache_path).read_text().splitlines():
        match = re.match(r'^([^#/][^:]*):[^=]+=(.*)$', line)
        if match:
            cache[match[1]] = match[2]
    keys = ('CMAKE_CXX_COMPILER', 'CMAKE_CUDA_COMPILER', 'CMAKE_CUDA_HOST_COMPILER',
            'Viskores_DIR', 'ParaView_DIR', 'VTK_DIR', 'catalyst_DIR',
            'ASTR_INSITU_FLYING_EDGES_SOURCE', 'ASTR_NVML_INCLUDE_DIR')
    assert all(cache.get(key) for key in keys), 'Incomplete validated dependency cache'
    build = tmp_path / 'build'
    result = configure(source_copy, build, '-DBUILD_TESTING=OFF', '-DASTR_WITH_CUDA=ON',
        '-DASTR_WITH_CATALYST=ON', '-DASTR_WITH_INSITU_DEVICE=ON',
        '-DASTR_WITH_INSITU_DEVICE_RENDERING=ON',
        '-DASTR_WITH_AIR5_CHEMISTRY='+air5,
        '-DCMAKE_CUDA_FLAGS='+cache.get('CMAKE_CUDA_FLAGS', ''), *[f'-D{key}={cache[key]}' for key in keys])
    assert result.returncode == 0, result.stdout
    targets = (build / 'CMakeFiles/TargetDirectories.txt').read_text()
    assert 'astr_insitu_device_products.dir' in targets and 'probe.dir' not in targets
    rules = (build / 'src/CMakeFiles/astr_insitu_device_products.dir/build.make').read_text()
    assert 'insitu_plane_products.cu' in rules and str(source_copy / 'tests') not in rules
    flags = (build / 'src/CMakeFiles/astr_insitu_device_products.dir/flags.make').read_text()
    assert 'ASTR_INSITU_DEVICE_RENDERING' in flags and 'ASTR_BUILD_TESTING' not in flags
    assert 'ASTR_BUILD_TESTING' not in rules
    assert '--fmad=false' in flags
    if os.environ.get('ASTR_TEST_CMAKE_BUILD') == '1':
        result = subprocess.run(['cmake', '--build', str(build), '--target', 'astr', '-j', '2'],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert result.returncode == 0, result.stdout
        assert (build / 'bin/astr').is_file()
    prefix = tmp_path / 'installed'
    installed = subprocess.run(['cmake', '--install', str(build), '--component', 'InsituTools',
        '--prefix', str(prefix)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert installed.returncode == 0, installed.stdout
    for name in ('tgv_pipeline.py', 'tgv_streamlines.py', 'egl_identity.py',
                 'image_publication.py', 'product_dispatch.py', 'device_render_pipeline.py'):
        assert (prefix / 'share/astr/insitu' / name).read_bytes() == (ROOT / 'scripts/insitu' / name).read_bytes()


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
