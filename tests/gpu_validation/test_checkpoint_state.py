"""Exercise the actual MPI/HDF5 state primitive; no flow solver is modeled here."""
import os
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np
import pytest


PROBE = Path(os.environ.get("ASTR_CHECKPOINT_STATE_PROBE", "build_insitu_gpu/bin/checkpoint_state_probe")).resolve()
MPIEXEC = os.environ.get("ASTR_OUTPUT_MPIEXEC", shutil.which("mpiexec") or "mpiexec")


@pytest.mark.parametrize("ranks", [1, 2])
@pytest.mark.parametrize("mode", ["context_reject", "context_clear"])
def test_failure_context(tmp_path, ranks, mode):
    log = run(tmp_path / "unused.h5", mode, ranks, success=False)
    assert "injected single-rank failure" in log
    if mode == "context_reject":
        assert "checkpoint context: batch=test-batch; last_complete=test-last-complete" in log
    else:
        assert "checkpoint context:" not in log


@pytest.mark.parametrize("ranks,axis", [(1, 1), (2, 1), (2, 2)])
def test_air5_cache_carry_provider(tmp_path, ranks, axis):
    path = tmp_path / "air5.h5"
    run(path, "air5_write", ranks, axis, components=34)
    run(path, "air5_exact", ranks, axis, components=34)
    run(path, "air5_compensation_off", ranks, axis, components=34, success=False)
    run(path, "statistics_exact", ranks, axis, components=34, success=False)
    with h5py.File(path) as state:
        assert state["identity"][12] == 7
        assert state["metadata"][:].tolist() == [1, 1]
        assert len(state) == 38


@pytest.mark.parametrize("ranks,axis", [(1, 1), (2, 1), (2, 2)])
def test_statistics_state_continuation(tmp_path, ranks, axis):
    path = tmp_path / "statistics.h5"
    run(path, "statistics_write", ranks, axis, components=34)
    run(path, "statistics_exact", ranks, axis, components=34)
    run(path, "statistics_wrong_role", ranks, axis, components=34, success=False)
    with h5py.File(path, "r") as state:
        assert state["identity"][12] == 3
        assert len(state) == 38
        assert np.all(state["q0034"][:] == 1)
        assert state["metadata"][:].tolist() == [1, -(1 << 63), 4602678819172646912, 2147483650]


@pytest.mark.parametrize("defect", ["missing", "shape", "type"])
def test_statistics_metadata_rejection(tmp_path, defect):
    path = tmp_path / "statistics.h5"
    run(path, "statistics_write", 2, components=34)
    with h5py.File(path, "r+") as state:
        data = state["metadata"][:]
        del state["metadata"]
        if defect == "shape":
            state.create_dataset("metadata", data=data[:-1])
        elif defect == "type":
            state.create_dataset("metadata", data=data.astype(np.float64))
    run(path, "statistics_exact", 2, components=34, success=False)


def run(path, mode, ranks=1, axis=1, components=5, halo=1, success=True):
    if not PROBE.is_file():
        pytest.skip("build checkpoint_state_probe through the root CMake first")
    env = os.environ.copy()
    env.update(OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
    result = subprocess.run(
        [MPIEXEC, "--oversubscribe", "-n", str(ranks), str(PROBE), str(path), mode,
         str(axis), str(components), str(halo)],
        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=45,
    )
    if success:
        assert result.returncode == 0, result.stdout
        assert "PASS" in result.stdout, result.stdout
    else:
        assert result.returncode != 0, result.stdout
        assert "checkpoint state rejected:" in result.stdout, result.stdout
    assert sum(p.stat().st_size for p in path.parent.iterdir() if p.is_file()) <= 4 * 1024**2
    return result.stdout


@pytest.mark.parametrize("components", [5, 11])
@pytest.mark.parametrize("halo", [0, 1])
@pytest.mark.parametrize("ranks,axis", [(1, 1), (2, 1), (2, 2)])
def test_exact_and_repartition(tmp_path, components, halo, ranks, axis):
    path = tmp_path / "state.h5"
    run(path, "write", ranks, axis, components, halo)
    run(path, "exact", ranks, axis, components, halo)
    for new_ranks, new_axis in [(1, 1), (2, 1), (2, 2)]:
        run(path, "repartition", new_ranks, new_axis, components, halo)
    with h5py.File(path, "r") as f:
        assert f["q0001"].shape == (5, 7, 9)
        assert np.signbit(f["q0001"][0, 0, 0])
        assert f["identity"][8] == 2147483650
        assert len(f) == components + 3
        if ranks == 1 and halo == 0:
            assert f["rank_extras"].shape == (0,)


@pytest.mark.parametrize("defect", ["version", "phase", "shape", "missing_q", "fp32_q", "clock",
                                    "extras_count", "missing_extras", "saved_overlap"])
def test_reject_damaged_state(tmp_path, defect):
    path = tmp_path / "state.h5"
    run(path, "write", 2)
    with h5py.File(path, "r+") as f:
        if defect in {"version", "phase", "shape", "clock"}:
            index, value = {"version": (1, 99), "phase": (2, 0), "shape": (3, 10),
                            "clock": (11, 0)}[defect]
            f["identity"][index] = value
        elif defect == "extras_count":
            f["partitions"][7] = -1
        elif defect == "saved_overlap":
            f["partitions"][8] = 0
        elif defect == "missing_extras":
            del f["rank_extras"]
        elif defect == "missing_q":
            del f["q0002"]
        else:
            data = f["q0002"][:].astype("float32")
            del f["q0002"]
            f.create_dataset("q0002", data=data)
    run(path, "exact", 2, success=False)
    run(path, "repartition", 1, success=False)


def test_reject_wrong_exact_partition_and_existing_file(tmp_path):
    path = tmp_path / "state.h5"
    run(path, "write", 2)
    original = path.read_bytes()
    run(path, "write", 2, success=False)
    assert path.read_bytes() == original
    run(path, "exact", 1, success=False)
    run(path, "exact", 2, axis=2, success=False)
    run(path, "exact", 2)


@pytest.mark.parametrize("ranks,axis", [(1, 1), (2, 1), (2, 2)])
def test_perfect_gas_cache_provider(tmp_path, ranks, axis):
    path = tmp_path / "state.h5"
    run(path, "cache_write", ranks, axis, components=11)
    run(path, "cache_exact", ranks, axis, components=11)
    # The same component count must not silently alias a different field role.
    run(path, "exact", ranks, axis, components=11, success=False)
    with h5py.File(path, "r") as f:
        assert f["identity"][1] == 2
        assert f["identity"][12] == 2
        assert len(f) == 14
