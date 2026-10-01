"""Checkpoint-bound GPU diagnostic baseline, preserving the existing transport phase."""
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from run_output_air5_restart_validation import launch
from run_output_restart_validation import compare_fields, archive_schedule_payload
from run_output_archive_validation import groups, frames, check_frame, check_accounting
from test_checkpoint_bundle import fault_library
from test_output_series_repair import seal_file_records, repair

ROOT = Path(__file__).resolve().parents[2]
EXE = Path(os.environ.get("ASTR_OUTPUT_RUNTIME_EXE", ROOT / "build_gpu_probe/bin/astr")).resolve()
MPIEXEC = Path(os.environ.get("ASTR_OUTPUT_MPIEXEC",
    "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec")).resolve()
FINAL = "outdat/new/checkpoints/step000000000012"


def arguments(output, case="hbl", axis="x", initial=False, workspace="scalar"):
    return SimpleNamespace(output=output, executable=EXE, mpiexec=MPIEXEC, case=case,
        reconstruction=3 if case == "sbli" else 5, mean_statistics=True, sample_interval=2,
        mode="steps", axis=axis, initial_restart=initial,
        top_mode="characteristic", filter_workspace=workspace)


def state(path):
    raw = path.read_bytes()
    assert len(raw) == 168 and raw[:8] == b"ASTRA5C1"
    clock = np.frombuffer(raw[8:40], dtype="<i8")
    history = np.frombuffer(raw[40:80], dtype="<i8")
    baseline = np.frombuffer(raw[80:], dtype="<f8")
    assert np.isfinite(baseline).all()
    return raw, clock, history, baseline


def diagnostic(case, origin):
    path = case / f"outdat/new/air5_conservation_from_step{origin:012d}.dat"
    values = np.loadtxt(path)
    assert values.ndim == 2 and values.shape[1] == 21 and np.isfinite(values).all()
    return values


@pytest.fixture(scope="module", params=[(1, "x", "hbl", "scalar"), (2, "z", "sbli", "full")])
def reference(request, tmp_path_factory, fault_library):
    ranks, axis, case, workspace = request.param
    args = arguments(tmp_path_factory.mktemp("conservation_" + case), case, axis, workspace=workspace)
    reference, _ = launch(args, "gpu", ranks, "continuous", 12, conservation=True,
                          publication_hook=(fault_library, 5))
    source = reference / "outdat/new/checkpoints/step000000000005"
    assert (source / "PROTECT").is_file() and (source / "COMPLETE").is_file()
    return args, ranks, reference, source


def test_gpu_exact_conservation_baseline_continuation(reference, tmp_path, record_property):
    args, ranks, continuous, source = reference
    args = SimpleNamespace(**vars(args))
    args.output = tmp_path
    old = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    resumed, size = launch(args, "gpu", ranks, "resumed", 12, restore=source, conservation=True)
    for name in ("state.h5", "statistics.h5"):
        compare_fields(continuous / FINAL / name, resumed / FINAL / name)
    for name in ("control.bin", "air5_config.bin", "air5_conservation.bin"):
        assert (continuous / FINAL / name).read_bytes() == (resumed / FINAL / name).read_bytes()
    assert archive_schedule_payload(continuous / FINAL / "archives.bin") == archive_schedule_payload(
        resumed / FINAL / "archives.bin")
    _, _, history, baseline = state(resumed / FINAL / "air5_conservation.bin")
    assert history.tolist() == [1, 1, 0, 12, 12]
    assert baseline.tobytes() == state(source / "air5_conservation.bin")[3].tobytes()
    earlier, later = diagnostic(continuous, 0), diagnostic(resumed, 5)
    assert earlier[0, :2].tolist() == later[0, :2].tolist() == [0, 0]
    assert earlier[0].tobytes() == later[0].tobytes()
    assert earlier[6:].tobytes() == later[1:].tobytes()
    assert later[1:, 0].tolist() == list(range(6, 13)) and np.all(later[1:, 1] == 1)
    assert earlier[1:6, 0].tolist() + later[1:, 0].tolist() == list(range(1, 13))
    for values in later[1:]:
        for component, column in ((0, 16), (4, 17)):
            expected = (values[component + 2] - baseline[component]) / max(
                abs(baseline[component]), np.finfo(np.float64).tiny)
            # Two FP64 operations on the same recorded totals and persisted base.
            assert abs(values[column] - expected) <= 8 * np.finfo(np.float64).eps * max(
                abs(expected), np.finfo(np.float64).tiny)
    assert old == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    assert size < 64 * 1024**2
    record_property("directory_bytes", size)
    record_property("conservation_state_bytes", 168)


def test_conservation_switch_change_is_rejected(reference, tmp_path):
    args, ranks, _, source = reference
    args = SimpleNamespace(**vars(args))
    args.output = tmp_path
    launch(args, "gpu", ranks, "disabled", 12, restore=source,
           reject="AIR5 source/limiter/compensation/top/resource contract mismatch")


def test_cpu_does_not_silently_enable_gpu_diagnostic(tmp_path):
    launch(arguments(tmp_path), "cpu", 1, "unsupported", 5, conservation=True,
           reject="AIR5 conservation diagnostic is GPU-only")


@pytest.mark.parametrize("ranks,axis,workspace", [(1, "x", "scalar"), (2, "y", "full")])
def test_step_zero_empty_history_exact_initial_restart(tmp_path, ranks, axis, workspace):
    args = arguments(tmp_path, initial=True, axis=axis, workspace=workspace)
    continuous, _ = launch(args, "gpu", ranks, "continuous", 12, conservation=True)
    # Use a dedicated one-step seed because ordinary keep=2 rotates step zero.
    seed, _ = launch(args, "gpu", ranks, "seed", 1, conservation=True)
    source = seed / "outdat/new/checkpoints/step000000000000"
    _, clock, history, baseline = state(source / "air5_conservation.bin")
    assert clock[0] == 0 and history.tolist() == [1, 0, 0, 0, 0] and np.all(baseline == 0)
    resumed, size = launch(args, "gpu", ranks, "resumed", 12, restore=source, conservation=True)
    for name in ("state.h5", "statistics.h5"):
        compare_fields(continuous / FINAL / name, resumed / FINAL / name)
    assert (continuous / FINAL / "air5_conservation.bin").read_bytes() == (
        resumed / FINAL / "air5_conservation.bin").read_bytes()
    assert state(resumed / FINAL / "air5_conservation.bin")[2].tolist() == [1, 1, 0, 12, 12]
    assert diagnostic(resumed, 0)[:, 0].tolist() == diagnostic(continuous, 0)[:, 0].tolist()
    assert diagnostic(resumed, 0).tobytes() == diagnostic(continuous, 0).tobytes()
    assert size < 64 * 1024**2


@pytest.mark.parametrize("defect", ["missing", "clock", "samples", "nan", "empty", "tail",
                                  "control_tail", "archive_tail", "config_tail"])
def test_invalid_conservation_history_is_rejected(reference, tmp_path, defect):
    args, ranks, _, source = reference
    args = SimpleNamespace(**vars(args))
    args.output = tmp_path
    run_root = tmp_path / "modified_source"
    shutil.copytree(source.parent.parent / "resources", run_root / "resources")
    batch = run_root / "checkpoints" / source.name
    shutil.copytree(source, batch)
    payload = batch / {"control_tail": "control.bin", "archive_tail": "archives.bin",
                       "config_tail": "air5_config.bin"}.get(defect, "air5_conservation.bin")
    raw = bytearray(payload.read_bytes())
    message = "AIR5 conservation sample identity"
    if defect == "missing":
        payload.unlink()
        message = "invalid new checkpoint bundle"
    else:
        if defect == "clock":
            raw[8:16] = np.array([6], dtype="<i8").tobytes()
            message = "AIR5 conservation state clock/version"
        elif defect == "samples":
            raw[72:80] = np.array([4], dtype="<i8").tobytes()
        elif defect == "nan":
            raw[80:88] = np.array([np.nan], dtype="<f8").tobytes()
            message = "AIR5 conservation activation/nonfinite baseline"
        elif defect == "empty":
            raw[48:] = bytes(len(raw) - 48)
            message = "AIR5 conservation empty history"
        elif defect == "tail":
            raw.extend(b"tail")
            message = "AIR5 conservation state size"
        elif defect.endswith("_tail"):
            raw.extend(b"x")
            message = {"control_tail": "control metadata tail", "archive_tail": "schedule history tail",
                       "config_tail": "AIR5 configuration size/tail"}[defect]
        payload.write_bytes(raw)
        names = [row.split()[0] for row in (batch / "MANIFEST").read_text().splitlines()[2:]]
        (batch / "MANIFEST").write_text(seal_file_records(batch, names, "ASTR_CHECKPOINT_BUNDLE 1"))
        size, crc = repair.fingerprint(batch / "MANIFEST")
        (batch / "COMPLETE").write_text(f"ASTR_COMPLETE_1 {size} {crc:016X}\n")
    before = {p.name: p.read_bytes() for p in batch.iterdir() if p.is_file()}
    launch(args, "gpu", ranks, "rejected", 12, restore=batch, conservation=True, reject=message)
    assert before == {p.name: p.read_bytes() for p in batch.iterdir() if p.is_file()}
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 64 * 1024**2


def test_conservation_fixed_order_memcheck(tmp_path):
    args = arguments(tmp_path)
    args.mean_statistics = False
    case, size = launch(args, "gpu", 1, "sanitized", 2, conservation=True, memcheck=True)
    log = (case / "run.log").read_text()
    assert "ERROR SUMMARY: 0 errors" in log and "fixed_order=yes" in log
    checkpoint = case / "outdat/new/checkpoints/step000000000002/air5_conservation.bin"
    assert state(checkpoint)[2].tolist() == [1, 1, 0, 2, 2]
    assert size < 64 * 1024**2


@pytest.mark.parametrize("archive,budget,message", [
    (False, 5631, "AIR5 conservation diagnostic device budget/allocation"),
    (True, 5632 + 95, "device budget cannot hold one basic output node"),
])
def test_conservation_and_archive_share_device_budget(tmp_path, archive, budget, message):
    args = arguments(tmp_path)
    args.mean_statistics = False
    launch(args, "gpu", 1, "too_small", 2, conservation=True, device_budget_bytes=budget,
           archive_groups=groups("steps") if archive else None, reject=message)


def test_conservation_with_native_archives_preserves_flow(tmp_path, record_property):
    args = arguments(tmp_path)
    args.mean_statistics = False
    plain, _ = launch(args, "gpu", 1, "plain", 2, conservation=True)
    archived, size = launch(args, "gpu", 1, "archived", 2, conservation=True,
                            archive_groups=groups("steps"), buffer_bytes=4096)
    checkpoint = "outdat/new/checkpoints/step000000000002"
    compare_fields(plain / checkpoint / "state.h5", archived / checkpoint / "state.h5")
    for name in ("control.bin", "air5_config.bin", "air5_conservation.bin"):
        assert (plain / checkpoint / name).read_bytes() == (archived / checkpoint / name).read_bytes()
    assert diagnostic(plain, 0).tobytes() == diagnostic(archived, 0).tobytes()
    geometry = archived / "outdat/new/resources/geometry.h5"
    for product in ("fields", "slices"):
        check_frame(frames(archived, product)[2], product, archived / checkpoint, geometry, "gpu")
    accounting = check_accounting(archived, "gpu", 1)
    assert accounting["max_device_tile_bytes"] + 5632 <= 67108864
    assert size < 64 * 1024**2
    record_property("directory_bytes", size)
    record_property("archive_accounting", accounting)
