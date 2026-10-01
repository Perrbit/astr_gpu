"""Real 16-cubed TGV failure/recovery with the unchanged production time integrator."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_checkpoint_bundle import fault_library
from run_output_restart_validation import compare_fields, run_case

ROOT = Path(__file__).resolve().parents[2]
EXE = Path(os.environ.get("ASTR_OUTPUT_RUNTIME_EXE", ROOT / "build_gpu_probe/bin/astr")).resolve()
MPIEXEC = Path(os.environ.get("ASTR_OUTPUT_MPIEXEC",
    "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec")).resolve()


@pytest.fixture(scope="module", params=["cpu", "gpu"])
def reference(request, tmp_path_factory):
    if not EXE.is_file():
        pytest.fail("build the root CUDA/AIR5 solver for the CPU/GPU runtime gate")
    args = SimpleNamespace(output=tmp_path_factory.mktemp("publication_" + request.param),
        executable=EXE, mpiexec=MPIEXEC, case="tgv", mode="steps", restart_step=5,
        initial_dimension=0, legacy_statistics=False, statistics=False,
        initial_restart=False, filter_workspace="scalar", force="feedback", axis="x", no_samples=True)
    continuous, _ = run_case(args, ROOT, request.param, 2, "continuous", 12, buffer_bytes=4096)
    seed, _ = run_case(args, ROOT, request.param, 2, "seed", 5, buffer_bytes=4096)
    return args, request.param, continuous, seed


@pytest.mark.parametrize("phase", ["batch_create", "batch_rename", "latest_rename", "retire_marker", "retire_payload"])
def test_runtime_failure_then_exact_recovery(reference, fault_library, phase, tmp_path):
    args, backend, continuous, seed = reference
    args = SimpleNamespace(**vars(args))
    args.output = tmp_path
    label = "cannot create checkpoint candidate" if phase == "batch_create" else "cannot publish/retain checkpoint"
    failed, _ = run_case(args, ROOT, backend, 2, phase, 12, buffer_bytes=4096, checkpoint_keep=1,
        publication_fault=(fault_library, phase, 10, 5), reject=label)
    checkpoints = failed / "outdat/new/checkpoints"
    recovery_step = 10 if phase.startswith("retire_") else 5
    recovery = checkpoints / f"step{recovery_step:012d}"
    log = (failed / "run.log").read_text()
    assert "batch=outdat/new/checkpoints/step000000000010" in log
    assert f"last_complete=outdat/new/checkpoints/step{recovery_step:012d}" in log
    assert "checkpoint context: batch=" in log
    assert (checkpoints / "LATEST").read_text() == recovery.name + "\n"
    assert not (checkpoints / "step000000000012").exists()
    expected = seed / "outdat/new/checkpoints/step000000000005" if recovery_step == 5 else \
        continuous / "outdat/new/checkpoints/step000000000010"
    compare_fields(recovery / "state.h5", expected / "state.h5")
    restored, _ = run_case(args, ROOT, backend, 2, phase + "_restored", 12, restore=recovery,
                          buffer_bytes=4096, checkpoint_keep=1)
    final = "outdat/new/checkpoints/step000000000012/state.h5"
    compare_fields(restored / final, continuous / final)
    assert sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file()) < 64 * 1024**2


def test_first_resumed_write_reports_source(reference, fault_library, tmp_path):
    args, backend, _, seed = reference
    args = SimpleNamespace(**vars(args))
    args.output = tmp_path
    source = seed / "outdat/new/checkpoints/step000000000005"
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    failed, _ = run_case(args, ROOT, backend, 2, "resumed_create_failure", 12, restore=source,
        buffer_bytes=4096, checkpoint_keep=1,
        publication_fault=(fault_library, "batch_create", 10, 5), reject="cannot create checkpoint candidate")
    log = (failed / "run.log").read_text()
    assert "checkpoint context: batch=outdat/new/checkpoints/step000000000010" in log
    assert f"last_complete={source}" in log
    assert not list((failed / "outdat/new/checkpoints").rglob("COMPLETE"))
    assert not (failed / "outdat/new/checkpoints/LATEST").exists()
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    assert sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file()) < 64 * 1024**2
