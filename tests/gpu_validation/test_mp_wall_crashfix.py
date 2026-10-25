"""Synthetic bc41 inputs through production CPU/GPU wall, repair and flux code."""
import os
from pathlib import Path
import subprocess

import pytest
import numpy as np
import h5py

from run_m12_mp7_nscbc_performance import mpi_command, runtime
from prepare_m12_local_case import prepare, replace_value
from run_m12_boundary_compare import configuration
from compare_q_validation_snapshots import read_q_snapshot, comparison_values
from run_output_restart_validation import compare_fields
from compare_flowstate import read_flowstate


PROBE = os.environ.get("ASTR_MP_WALL_PROBE")
EXECUTABLE = os.environ.get("ASTR_M12_CRASHFIX_EXECUTABLE")
MPIEXEC = os.environ.get("ASTR_TEST_MPIEXEC", "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec")
pytestmark = pytest.mark.skipif(not PROBE, reason="set ASTR_MP_WALL_PROBE to the built production-module probe")


def run_probe(ranks, binding, *extra, transport="pinned", cwd=None, positivity="flux"):
    env = runtime(f"{ranks},1,1",transport)
    env.update(ASTR_GPU_MP_FACE_LAYOUT="x-contiguous", ASTR_GPU_MP_SPLIT_CACHE="on",
               ASTR_GPU_MP_POSITIVITY=positivity)
    command = mpi_command(MPIEXEC, ranks, transport)
    return subprocess.run([*command, str(Path(PROBE).resolve()), binding, "wall-blowing", *extra], env=env,cwd=cwd,
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


@pytest.mark.parametrize("ranks,binding", [(1,"0"),(2,"rank")])
def test_boundary_repair_remains_active_without_the_mp_flux_limiter(ranks,binding):
    result = run_probe(ranks,binding,positivity="off")
    assert result.returncode == 0,result.stdout
    assert "ASTR_GPU_MP_POSITIVITY=off" in result.stdout,result.stdout
    assert "MP wall blowing input repair: PASS" in result.stdout,result.stdout
    rows = [line.split() for line in result.stdout.splitlines() if line.startswith("MP_WALL_REPAIR ")]
    assert len(rows) == 8,result.stdout
    assert all(int(row[3]) == int(float(row[2]) >= 4.0) for row in rows),result.stdout
    assert "ieee_invalid is signaling" not in result.stdout,result.stdout


@pytest.mark.parametrize("ranks,binding",[(1,"0"),(2,"rank")])
def test_no_valid_neighbor_still_fails_with_a_nonzero_exit(ranks,binding):
    result = run_probe(ranks,binding,"no-neighbor")
    assert result.returncode != 0,result.stdout
    assert "GPU crashfix left a nonfinite or non-positive physical state" in result.stdout,result.stdout
    assert "MP wall blowing input repair: PASS" not in result.stdout


@pytest.mark.parametrize("ranks", [1, 2, 4])
def test_expanding_repair_reaches_distant_and_remote_valid_states(ranks,tmp_path):
    env = runtime(f"{ranks},1,1")
    env['ASTR_GPU_MP_POSITIVITY'] = 'off'
    command = [*mpi_command(MPIEXEC,ranks,'pinned'),str(Path(PROBE).resolve()),'rank','crashfix-expand']
    result = subprocess.run(command,env=env,cwd=tmp_path,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
    (tmp_path/'expanded.log').write_text(result.stdout)
    assert result.returncode == 0,result.stdout
    assert 'GPU expanding crashfix: PASS' in result.stdout,result.stdout
    assert 'ASTR_GPU_CRASHFIX_EXPANDED' in result.stdout,result.stdout
    rows = [line.split() for line in result.stdout.splitlines() if line.startswith('GPU_CRASHFIX_EXPAND ')]
    assert {int(row[1]) for row in rows} == ({1,3} if ranks == 1 else {1,2,3}),result.stdout
    expanded = [line.split() for line in result.stdout.splitlines() if line.startswith('ASTR_GPU_CRASHFIX_EXPANDED ')]
    assert max(int(row[5]) for row in expanded) > 1,result.stdout
    if ranks > 1:
        assert next(int(row[3]) for row in expanded if int(row[2]) == 2) == ranks-1,result.stdout


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


@pytest.mark.parametrize("ranks,transport", [(1,"pinned"),(2,"pinned"),(4,"pinned"),(2,"device-aware")])
def test_actual_repaired_history_roundtrip_and_mp_flux_are_exact(tmp_path,ranks,transport):
    result = run_probe(ranks,"rank","history",transport=transport,cwd=tmp_path)
    assert result.returncode == 0,result.stdout
    assert result.stdout.count("GPU repaired history roundtrip: PASS") == 2,result.stdout
    for mode,name in ((1,"enabled"),(0,"disabled")):
        with h5py.File(tmp_path/f"history-{name}.h5") as state:
            layout=state["crashfix_layout"][...]
            counts=state["crashfix_partitions"][...].reshape(-1,2)
            words=state["crashfix_history"][...]
            np.testing.assert_array_equal(layout[:3],[1,mode,ranks])
            assert words.dtype.kind == "i" and words.dtype.itemsize == 8
            assert len(words) == layout[3] == counts[:,1].sum()
            offset=0
            for nodes,count in counts:
                assert words[offset] == -1  # All 64 bits, including the sign bit.
                assert int(words[offset+count-1]) & (1 << int((nodes-1)%64))
                offset+=int(count)


@pytest.mark.parametrize("option,message",[("history-missing","checkpoint lacks required crashfix history"),
    ("history-mode","crashfix mode/layout mismatch"),("history-topology","crashfix history requires the same MPI topology")])
def test_history_restart_contract_rejects_missing_or_incompatible_state(tmp_path,option,message):
    result = run_probe(2,"rank",option,cwd=tmp_path)
    assert result.returncode != 0,result.stdout
    assert message in result.stdout,result.stdout


@pytest.mark.skipif(not EXECUTABLE, reason="set ASTR_M12_CRASHFIX_EXECUTABLE to the M12 production binary")
@pytest.mark.parametrize("backend,ranks",[("gpu",1),("gpu",2),("gpu",4),("cpu",1),("cpu",2)])
def test_m12_nonzero_history_twelve_steps_match_five_plus_seven(tmp_path,backend,ranks):
    executable=str(Path(EXECUTABLE).resolve())
    def launch(name,steps,restore=None):
        case=tmp_path/name
        prepare(Path("/home/dell/workspace/M12_C13"),case,backend)
        configuration(case,"nscbc" if backend=="gpu" else "extrapolation",(64,32,24),steps,
                      archive=False,checkpoint_interval=5,restore=restore)
        controller=case/"datin/controller"
        lines=controller.read_text().splitlines()
        replace_value(lines,"lwsequ,lwslic","f,f,f,t")
        controller.write_text("\n".join(lines)+"\n")
        env=runtime(f"{ranks},1,1")
        env.update(ASTR_GPU_MP_FACE_LAYOUT="x-contiguous",ASTR_GPU_MP_SPLIT_CACHE="on")
        command=[*mpi_command(MPIEXEC,ranks,"pinned"),executable,"run","datin/input.dat"]
        with (case/"run.log").open("w") as log:
            result=subprocess.run(command,cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        text=(case/"run.log").read_text()
        assert result.returncode == 0,text[-8000:]
        assert "ASTR_MP_INPUT_CRASHFIX" not in text,text[-8000:]
        return case
    seed=launch("seed",1)/"outdat/new/checkpoints/step000000000001"
    # Add explicit diagnostic critical nodes to a valid completed-step seed.
    # Actual negative-wall recovery and its saved history are covered above.
    with h5py.File(seed/"state.h5","r+") as state:
        words=state["crashfix_history"][...].view(np.uint64)
        offset=0
        for partition in state["partitions"][...].reshape(-1,8):
            cells=partition[3:6]; halo=int(partition[6]); extents=cells+2*halo+1
            nodes=int(np.prod(extents)); count=(nodes+63)//64
            points=[0,nodes-1]
            for i in (int(cells[0]//2),int(cells[0])):
                points.append(int(i+halo+extents[0]*(cells[1]//2+halo+extents[1]*(cells[2]//2+halo))))
            for point in points:
                words[offset+point//64] |= np.uint64(1) << np.uint64(point%64)
            offset+=count
        state["crashfix_history"][...]=words.view(np.int64)
        initial_history=words.copy()
    continuous=launch("continuous",13,seed)
    first=launch("first",6,seed)
    resumed=launch("resumed",13,first/"outdat/new/checkpoints/step000000000006")
    final=Path("outdat/new/checkpoints/step000000000013")
    compare_fields(continuous/final/"state.h5",resumed/final/"state.h5")
    for filename in ("control.bin","insitu_control.bin"):
        assert (continuous/final/filename).read_bytes() == (resumed/final/filename).read_bytes()
    with h5py.File(resumed/final/"state.h5") as state:
        np.testing.assert_array_equal(state["crashfix_history"][...].view(np.uint64),initial_history)
    header,values=read_flowstate(continuous)
    first_header,first_values=read_flowstate(first)
    resumed_header,resumed_values=read_flowstate(resumed)
    assert header == first_header == resumed_header
    np.testing.assert_array_equal(values,np.concatenate([first_values,resumed_values]))


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
