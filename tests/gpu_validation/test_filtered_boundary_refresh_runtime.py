"""Bounded non-reacting filter-refresh regression using existing case drivers."""
import os
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from compare_flowfield_h5 import read_flowfield, reconstruct_q
from run_output_restart_validation import run_case

ROOT = Path(__file__).resolve().parents[2]
CPU = Path(os.environ.get("ASTR_OUTPUT_CPU_EXE", ROOT / "build_release_restart_cpu/bin/astr"))
GPU = Path(os.environ.get("ASTR_OUTPUT_RUNTIME_EXE", ROOT / "build_gpu_probe/bin/astr"))
ATOL = 2e-10


def test_explicit_filter_refresh_is_boundary_independent():
    expected = ".not.lcomb.and.numq==5.and.num_species==0.and.conschm(4:4)=='e'.and.difschm(4:4)=='e'"
    for filename, count in (("src/mainloop.F90", 1), ("src_gpu/mainloop_gpu.cuf", 2)):
        source = (ROOT / filename).read_text()
        expressions = re.findall(r"refresh_filtered_primitives=([^\n]+)&\s*\n([^\n]+)", source)
        assert len(expressions) == count
        for lines in expressions:
            assert re.sub(r"\s+", "", "".join(lines)) == expected


CASES = [
    ("wall41_x", "run_wall41_phased_compare.sh", "scalar", "2,1,1", {"WALL_AXIS": "x"}),
    ("wall41_y", "run_wall41_phased_compare.sh", "full", "2,1,1", {"WALL_AXIS": "y"}),
    ("wall41_z", "run_wall41_phased_compare.sh", "scalar", "1,1,2", {"WALL_AXIS": "z"}),
    ("wall42_x", "run_xextrap_phaseb_compare.sh", "full", "2,1,1", {"BC_KIND": "adiabaticwall", "ZERO_AXIS": "x"}),
    ("wall42_y", "run_xextrap_phaseb_compare.sh", "scalar", "2,1,1", {"BC_KIND": "adiabaticwall", "ZERO_AXIS": "y"}),
    ("wall411_y", "run_xextrap_phaseb_compare.sh", "full", "1,1,2", {"BC_KIND": "slipisotwall", "ZERO_AXIS": "y"}),
    ("wall421_y", "run_xextrap_phaseb_compare.sh", "scalar", "2,1,1", {"BC_KIND": "slipadibwall", "ZERO_AXIS": "y"}),
    ("symmetry_x", "run_xextrap_phaseb_compare.sh", "full", "1,2,1", {"BC_KIND": "symmetry", "ZERO_AXIS": "x"}),
    ("zeroextrap_z", "run_xextrap_phaseb_compare.sh", "scalar", "1,1,2", {"BC_KIND": "zeroextrap", "ZERO_AXIS": "z"}),
    ("ldc", "run_ldcavity_phaseia_compare.sh", "full", "2,1,1", {}),
    ("rti", "run_rti_phasej_compare.sh", "scalar", "2,1,1", {"COMPARE_FIELD": "f"}),
    ("curve_wall41_y", "run_curvilinear_tgv_compare.sh", "scalar", "2,1,1",
     {"MAPPING": "y-wavy", "HOMOGENEOUS": "t,f,t", "BCTYPE": "1;1;41,273.15d0;41,273.15d0;1;1"}),
    ("curve_wall42_x", "run_curvilinear_tgv_compare.sh", "full", "1,2,1",
     {"MAPPING": "x-wavy", "HOMOGENEOUS": "f,t,t", "BCTYPE": "42;42;1;1;1;1"}),
]


@pytest.mark.parametrize("name,driver,workspace,topology,extra", CASES,
                         ids=[case[0] for case in CASES])
def test_physical_boundary_filtered_state(name, driver, workspace, topology, extra,
                                          tmp_path, record_property):
    env = {key: value for key, value in os.environ.items() if not key.startswith("ASTR_")}
    env.update(CPU_EXE=str(CPU), GPU_EXE=str(GPU), OUT_DIR=str(tmp_path),
               GRID="16,16,16", MAXSTEP="3", FEQCHKPT="3", DELTAT="1.d-4",
               LFILTER="t", DIFFTERM="t", SCHEME="643e", NP="2", TOPOLOGY=topology,
               COMPARE_STATS="t", COMPARE_FIELD="t", STATS_ATOL=str(ATOL), STATS_RTOL="0",
               FIELD_ATOL=str(ATOL), FIELD_RTOL="0", FILTER_WORKSPACE=workspace,
               GPU_SYNC_MODE="explicit", GPU_HALO_TRANSPORT="pinned",
               OMPI_MCA_coll_hcoll_enable="0",
               ASTR_GPU_FILTER_WORKSPACE=workspace, ASTR_GPU_SYNC_MODE="explicit",
               ASTR_GPU_HALO_TRANSPORT="pinned", ASTR_GPU_PRECISION_MODE="fp64",
               ASTR_VALIDATION_RK_SNAPSHOT="outdat/rk_complete_snapshot.h5")
    env.update(extra)
    if driver == "run_curvilinear_tgv_compare.sh":
        # The generated 16-cubed grid replaces grid.h5; do not copy the unused
        # large example grid into either bounded test directory.
        fixture = tmp_path / "source/datin"
        fixture.mkdir(parents=True)
        for filename in ("input.tgv", "controller", "parallel.info"):
            source = ROOT / "examples/Taylor_Green_Vortex/datin" / filename
            if source.exists():
                shutil.copyfile(source, fixture / filename)
        env["CASE_DIR"] = str(fixture.parent)
    with (tmp_path / "driver.log").open("wb") as log:
        subprocess.run(["bash", str(ROOT / "tests/gpu_validation" / driver)], cwd=ROOT,
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    # RTI's legacy driver compares a different CPU output phase; use the common
    # complete-RK snapshot explicitly, as the other drivers already do.
    left = read_flowfield(tmp_path / "cpu/outdat/rk_complete_snapshot.h5")
    right = read_flowfield(tmp_path / "gpu")
    left.update(reconstruct_q(left, 2.5))
    right.update(reconstruct_q(right, 2.5))
    errors = {}
    for field in left:
        assert left[field].shape == right[field].shape
        assert np.isfinite(left[field]).all() and np.isfinite(right[field]).all()
        errors[field] = float(np.max(np.abs(left[field] - right[field])))
        assert errors[field] <= ATOL, (name, field, errors[field])
    record_property("field_errors", errors)
    record_property("filter_workspace", workspace)
    total = sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file())
    assert total < 64 * 1024**2
    record_property("test_root_bytes", total)


@pytest.mark.parametrize("axis,workspace", [("x", "scalar"), ("y", "full")])
def test_curved_profile_flatplate_filtered_state(axis, workspace, tmp_path, record_property):
    args = SimpleNamespace(output=tmp_path, executable=CPU, mpiexec=Path(os.environ.get(
        "ASTR_OUTPUT_MPIEXEC", "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec")),
        case="curve", mode="steps", restart_step=5, initial_dimension=0,
        legacy_statistics=False, statistics=False, initial_restart=False,
        filter_workspace=workspace, force="fixed", axis=axis, no_samples=True)
    cpu, _ = run_case(args, ROOT, "cpu", 2, "comparison", 4, buffer_bytes=4096)
    args.executable = GPU
    gpu, _ = run_case(args, ROOT, "gpu", 2, "comparison", 4, buffer_bytes=4096)
    checkpoint = "outdat/new/checkpoints/step000000000004/state.h5"
    errors = {}
    with h5py.File(cpu / checkpoint) as a, h5py.File(gpu / checkpoint) as b:
        assert np.isfinite(a["rank_extras"][:]).all() and np.isfinite(b["rank_extras"][:]).all()
        fields = sorted(name for name in a if re.fullmatch(r"q\d{4}", name))
        assert len(fields) == 11
        for name in fields:
            left, right = a[name][:], b[name][:]
            assert left.shape == right.shape and left.dtype == right.dtype
            assert np.isfinite(left).all() and np.isfinite(right).all()
            errors[name] = float(np.max(np.abs(left-right)))
            assert errors[name] <= ATOL, (name, errors[name])
    record_property("field_errors", errors)
    assert sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file()) < 64*1024**2


def test_nscbc52_global_filter_rejected(tmp_path):
    # The NSCBC face's transverse filter does not admit global lfilter=t.
    validation = ROOT / "tests/gpu_validation"
    subprocess.run([
        "python3", str(validation / "prepare_s1_flatplate_case.py"),
        "--dst-case", str(tmp_path), "--use-gpu", "t",
        "--im", "16", "--jm", "128", "--km", "16",
        "--conschm", "643e", "--diffterm", "t", "--lfilter", "t",
        "--upper-bctype", "52", "--reynolds", "1.83052e6", "--mach", "5",
        "--reference-temperature", "226.65", "--wall-temperature", "5.191440547760865",
        "--x-min", "-1", "--x-max", "10", "--y-stretch", "5",
        "--maxstep", "2", "--feqchkpt", "2", "--deltat", "1e-6",
    ], cwd=ROOT, check=True, capture_output=True, timeout=60)
    subprocess.run([
        "python3", str(validation / "generate_compressible_blasius_profile.py"),
        "--grid", str(tmp_path / "datin/grid.flatplate.h5"),
        "--output", str(tmp_path / "datin/inlet.prof"),
        "--mach", "5", "--reynolds", "1.83052e6",
        "--reference-temperature", "226.65", "--wall-temperature", "5.191440547760865",
        "--station-x", "1", "--points", "201",
    ], cwd=ROOT, check=True, capture_output=True, timeout=60)
    env = {key: value for key, value in os.environ.items() if not key.startswith("ASTR_")}
    env.update(ASTR_FORCE_MPI_TOPOLOGY="2,1,1", ASTR_GPU_FILTER_WORKSPACE="scalar",
               ASTR_GPU_SYNC_MODE="explicit", ASTR_GPU_HALO_TRANSPORT="pinned",
               ASTR_GPU_PRECISION_MODE="fp64", OMPI_MCA_coll_hcoll_enable="0")
    mpiexec = os.environ.get("ASTR_OUTPUT_MPIEXEC",
        "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec")
    with (tmp_path / "gpu.log").open("wb") as log:
        result = subprocess.run([mpiexec, "-np", "2", str(GPU), "run", "datin/input.flatplate"],
                                cwd=tmp_path, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=60)
    assert result.returncode != 0
    assert "GPU S1 flatplate requires an explicit 3D S1-A capability contract" in (
        tmp_path / "gpu.log").read_text()
    with h5py.File(tmp_path / "outdat/flowfield.h5") as initial:
        assert np.array_equal(initial["nstep"][:], [0])
        assert np.array_equal(initial["time"][:], [0.0])
    assert sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file()) < 64*1024**2
