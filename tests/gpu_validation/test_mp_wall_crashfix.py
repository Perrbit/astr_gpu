"""Synthetic bc41 inputs through production CPU/GPU wall, repair and flux code."""
import os
from pathlib import Path
import subprocess

import pytest
import numpy as np

from run_m12_mp7_nscbc_performance import mpi_command, runtime
from prepare_m12_local_case import prepare, replace_value
from run_m12_boundary_compare import configuration
from compare_q_validation_snapshots import read_q_snapshot, comparison_values


PROBE = os.environ.get("ASTR_MP_WALL_PROBE")
EXECUTABLE = os.environ.get("ASTR_M12_CRASHFIX_EXECUTABLE")
MPIEXEC = os.environ.get("ASTR_TEST_MPIEXEC", "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec")
pytestmark = pytest.mark.skipif(not PROBE, reason="set ASTR_MP_WALL_PROBE to the built production-module probe")


def run_probe(ranks, binding, *extra, transport="pinned"):
    env = runtime(f"{ranks},1,1",transport)
    env.update(ASTR_GPU_MP_FACE_LAYOUT="x-contiguous", ASTR_GPU_MP_SPLIT_CACHE="on")
    command = mpi_command(MPIEXEC, ranks, transport)
    return subprocess.run([*command, str(Path(PROBE).resolve()), binding, "wall-blowing", *extra], env=env,
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)


@pytest.mark.parametrize("ranks,binding,transport", [(1, "0", "pinned"), (1, "1", "pinned"),
    (2, "rank", "pinned"), (4, "rank", "pinned"), (2, "rank", "device-aware")])
def test_boundary_generated_nonpositive_state_is_repaired_before_mp_flux(ranks, binding, transport):
    result = run_probe(ranks, binding, transport=transport)
    assert result.returncode == 0, result.stdout
    assert "MP wall blowing input repair: PASS" in result.stdout, result.stdout
    assert "ieee_invalid is signaling" not in result.stdout, result.stdout
    rows = [line.split() for line in result.stdout.splitlines() if line.startswith("MP_WALL_REPAIR ")]
    assert len(rows) == 8, result.stdout
    for fields in rows:
        _, forcing, ratio, callbacks, error, vmin, vmax, rho, pressure, temperature = fields
        assert int(callbacks) == int(float(ratio) >= 4.0)
        assert float(error) <= 2e-10
        assert min(map(float, (rho, pressure, temperature))) > 0.0
        if forcing == "legacy_random":
            assert float(vmin) < 0.0 < float(vmax)


def test_disabled_crashfix_still_reports_the_invalid_mp_input():
    result = run_probe(1, "0", "detect-only")
    assert result.returncode != 0, result.stdout
    assert "ASTR_MP_POSITIVITY_FAILURE" in result.stdout, result.stdout
    assert "phase=face input/flux" in result.stdout, result.stdout
    assert "MP wall blowing input repair: PASS" not in result.stdout


@pytest.mark.parametrize("ranks,binding",[(1,"0"),(2,"rank")])
def test_no_valid_neighbor_still_fails_with_a_nonzero_exit(ranks,binding):
    result = run_probe(ranks,binding,"no-neighbor")
    assert result.returncode != 0,result.stdout
    assert "GPU crashfix left a nonfinite or non-positive physical state" in result.stdout,result.stdout
    assert "MP wall blowing input repair: PASS" not in result.stdout


@pytest.mark.parametrize("ranks", [1, 2, 4])
def test_gpu_ordered_repair_matches_original_cpu_and_preserves_history(ranks):
    env = runtime(f"{ranks},1,1")
    command = [*mpi_command(MPIEXEC,ranks,"pinned"),str(Path(PROBE).resolve()),"rank","crashfix-order"]
    result = subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
    assert result.returncode == 0,result.stdout
    assert "GPU ordered crashfix: PASS" in result.stdout,result.stdout
    rows = [line.split() for line in result.stdout.splitlines() if line.startswith("GPU_CRASHFIX_ORDER ")]
    assert len(rows) == 4,result.stdout
    assert all(float(row[-1]) <= 2e-10 for row in rows),result.stdout


@pytest.mark.skipif(not EXECUTABLE, reason="set ASTR_M12_CRASHFIX_EXECUTABLE to the M12 production binary")
@pytest.mark.parametrize("ranks", [1, 2])
def test_m12_valid_ten_step_trajectory_is_unchanged_by_enabling_repair(tmp_path,ranks):
    executable = str(Path(EXECUTABLE).resolve())
    finals = []
    for repair in (False,True):
        case = tmp_path/("repair" if repair else "detect")
        prepare(Path("/home/dell/workspace/M12_C13"),case,"gpu")
        configuration(case,"nscbc",(64,32,24),10,archive=False)
        controller = case/"datin/controller"
        lines = controller.read_text().splitlines()
        replace_value(lines,"lwsequ,lwslic","f,f,f,"+("t" if repair else "f"))
        controller.write_text("\n".join(lines)+"\n")
        env = runtime(f"{ranks},1,1")
        env.update(ASTR_GPU_MP_FACE_LAYOUT="x-contiguous",ASTR_GPU_MP_SPLIT_CACHE="on",
                   ASTR_VALIDATION_RHS_PREFIX="diagnostics/state",ASTR_VALIDATION_RHS_STEP="9")
        command = [*mpi_command(MPIEXEC,ranks,"pinned"),executable,"run","datin/input.dat"]
        with (case/"run.log").open("w") as log:
            result = subprocess.run(command,cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        text = (case/"run.log").read_text()
        assert result.returncode == 0,text[-6000:]
        assert "ASTR_MP_POSITIVITY_FAILURE" not in text
        assert "ASTR_MP_INPUT_CRASHFIX" not in text,"the valid baseline unexpectedly needed recovery"
        label = "post_crashfix" if repair else "post_update"
        paths = sorted((case/"diagnostics").glob(f"state.{label}.step00000009.rk03.rank*.bin"))
        assert len(paths) == ranks,text[-6000:]
        finals.append([comparison_values(read_q_snapshot(path),True) for path in paths])
    for reference,candidate in zip(*finals):
        assert np.isfinite(reference).all() and np.isfinite(candidate).all()
        np.testing.assert_array_equal(reference,candidate)
