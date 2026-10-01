"""Real EGL/Catalyst continuation from completed-step native checkpoints."""
import csv
import json
import os
from pathlib import Path
import re
import shutil
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from run_output_restart_validation import run_case, compare_fields, archive_schedule_payload
from test_checkpoint_bundle import fault_library
from test_output_series_repair import seal_file_records, repair

ROOT = Path(__file__).resolve().parents[2]
EXE = Path(os.environ.get("ASTR_OUTPUT_INSITU_EXE", ROOT / "build_insitu_gpu/bin/astr")).resolve()
MPIEXEC = Path(os.environ.get("ASTR_OUTPUT_MPIEXEC",
    "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec")).resolve()
BACKEND = Path(os.environ.get("ASTR_OUTPUT_INSITU_BACKEND",
    "/home/dell/workspace/astr_dependencies/install/paraview-6.1.1-hpcx-gcc13/lib/catalyst")).resolve()
FINAL = "outdat/new/checkpoints/step000000000012"


def arguments(output, axis="x"):
    return SimpleNamespace(output=output, executable=EXE, mpiexec=MPIEXEC,
        case="tgv", mode="steps", restart_step=5, initial_dimension=0,
        legacy_statistics=False, statistics=True, initial_restart=False,
        filter_workspace="scalar", force="feedback", axis=axis, no_samples=True)


def configuration(render=True, statistics=True, mode="steps", interval=8):
    schedule = f"step_interval={interval}" if mode == "steps" else f"time_interval={interval * 0.001:.17e}"
    return f"""&insitu_run
 enabled=t, statistics={'.true.' if statistics else '.false.'}, render={'.true.' if render else '.false.'},
 statistics_window=0.0005,0.0115, output_directory='outdat/render',
 schedule_mode='{mode}', {schedule}, initial_frame=f, final_frame=t,
 host_budget_bytes=4294967296, device_budget_bytes=2147483648, device_reserve_bytes=1073741824,
 implementation_path='{BACKEND}', pipeline_file='{ROOT / 'scripts/insitu/tgv_pipeline.py'}'
/
"""


def check_render(case, ranks, steps, statistics=True):
    output = case / "outdat/render"
    for rank in range(ranks):
        with (output / f"resources.rank{rank:08d}.csv").open() as stream:
            device = stream.readline().split()[1].removeprefix("device=GPU-")
            resources = list(csv.DictReader(stream))
        expected_stages = ["baseline", "initialized"] + ["frame_before", "frame_after"] * len(steps) + [
            "finalize_before", "finalize_after"]
        if statistics:
            expected_stages.append("statistics_exported")
        expected_stages.append("session_released")
        assert [row["stage"] for row in resources] == expected_stages
        for row in resources:
            assert int(row["host_increment_bytes"]) <= 4 * 1024**3
            assert int(row["device_increment_bytes"]) <= 2 * 1024**3
            assert int(row["device_free_bytes"]) >= 1024**3
        lifecycle = json.loads((output / f"lifecycle_rank{rank}.json").read_text())
        assert lifecycle["finalized"] is True
        assert [frame[0] for frame in lifecycle["frames"]] == steps
        for step in steps:
            receipt = json.loads((output / f"mesh_step{step:08d}_rank{rank}.json").read_text())
            assert receipt["step"] == step and np.isfinite(receipt["time"])
            assert receipt["crossing_maxabs"] <= 2e-10 if rank == 0 else True
            assert all(item["egl_uuid"] == device for item in receipt["products"].values())
    pictures = sorted(output.glob("*.jpeg"))
    assert pictures and all(path.with_suffix(".eps").is_file() for path in pictures)
    for path in pictures:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
            assert pixels.shape == (600, 800, 3)
            assert np.any(pixels < 245), path
    assert not list((case / "outdat").glob("restart_q.*"))
    assert not list((case / "outdat").glob("insitu_cpu_q.*"))
    flows = re.findall(r"ASTR_INSITU_GPU_FLOW rank=(\d+) frame_downloads=(\d+)", (case / "run.log").read_text())
    assert len(flows) == ranks and all(int(count) == len(steps) for _, count in flows)
    return pictures


@pytest.fixture(scope="module", params=[(1, "x", "steps"), (2, "x", "time")])
def reference(request, tmp_path_factory, fault_library):
    ranks, axis, mode = request.param
    args = arguments(tmp_path_factory.mktemp("native_render"), axis)
    config = configuration(mode=mode)
    continuous, size = run_case(args, ROOT, "gpu", ranks, "continuous", 12, insitu_config=config,
                                publication_fault=(fault_library, "protect_batch", 5, 0))
    check_render(continuous, ranks, [8, 12])
    source = continuous / "outdat/new/checkpoints/step000000000005"
    assert (source / "PROTECT").is_file()
    assert len((source / "insitu_control.bin").read_bytes()) == 313
    return args, ranks, mode, continuous, source, size


def test_same_topology_exact_render_continuation(reference, tmp_path, record_property):
    args, ranks, mode, continuous, source, reference_size = reference
    args = SimpleNamespace(**vars(args)); args.output = tmp_path
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    resumed, size = run_case(args, ROOT, "gpu", ranks, "resumed", 12, restore=source,
                            insitu_config=configuration(mode=mode))
    for name in ("state.h5", "statistics.h5"):
        compare_fields(continuous / FINAL / name, resumed / FINAL / name)
    for name in ("control.bin", "insitu_control.bin"):
        assert (continuous / FINAL / name).read_bytes() == (resumed / FINAL / name).read_bytes()
    assert archive_schedule_payload(continuous / FINAL / "archives.bin") == archive_schedule_payload(
        resumed / FINAL / "archives.bin")
    pictures = check_render(resumed, ranks, [8, 12])
    for path in pictures:
        with Image.open(path) as actual, Image.open(continuous / "outdat/render" / path.name) as expected:
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    geometry = sorted((resumed / "outdat/render").rglob("*.vtp"))
    assert geometry
    for path in geometry:
        assert path.read_bytes() == (continuous / "outdat/render" / path.relative_to(resumed / "outdat/render")).read_bytes()
    for rank in range(ranks):
        name = f"sample.statistics.step00000012.rank{rank:08d}.bin"
        assert (resumed / "outdat/render" / name).read_bytes() == (continuous / "outdat/render" / name).read_bytes()
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    record_property("directory_bytes", [reference_size, size])


def test_render_disabled_preserves_flow_and_statistics(reference, tmp_path):
    args, ranks, mode, continuous, _, _ = reference
    args = SimpleNamespace(**vars(args)); args.output = tmp_path
    plain, _ = run_case(args, ROOT, "gpu", ranks, "plain", 12,
                        insitu_config=configuration(render=False, mode=mode))
    for name in ("state.h5", "statistics.h5"):
        compare_fields(continuous / FINAL / name, plain / FINAL / name)
    assert len((plain / FINAL / "insitu_control.bin").read_bytes()) == 144
    assert not list((plain / "outdat/render").glob("*.jpeg"))


def test_finished_render_schedule_can_continue(reference, tmp_path):
    args, ranks, mode, continuous, _, _ = reference
    args = SimpleNamespace(**vars(args)); args.output = tmp_path
    source = continuous / FINAL
    plain, _ = run_case(args, ROOT, "gpu", ranks, "continuous13", 13,
                        insitu_config=configuration(render=False, mode=mode))
    resumed, _ = run_case(args, ROOT, "gpu", ranks, "resumed13", 13, restore=source,
                          insitu_config=configuration(mode=mode))
    final = "outdat/new/checkpoints/step000000000013"
    for name in ("state.h5", "statistics.h5"):
        compare_fields(plain / final / name, resumed / final / name)
    check_render(resumed, ranks, [13])
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 64 * 1024**2


def test_finished_checkpoint_without_advancement_does_not_duplicate_frame(reference, tmp_path):
    args, ranks, mode, continuous, _, _ = reference
    args = SimpleNamespace(**vars(args)); args.output = tmp_path
    source = continuous / FINAL
    resumed, _ = run_case(args, ROOT, "gpu", ranks, "already_finished", 12, restore=source,
                          insitu_config=configuration(mode=mode))
    output = resumed / "outdat/render"
    assert not list(output.glob("*.jpeg"))
    assert not list(output.glob("lifecycle_rank*.json"))
    assert not list((resumed / "outdat/new/checkpoints").glob("step*/COMPLETE"))
    for rank in range(ranks):
        with (output / f"resources.rank{rank:08d}.csv").open() as stream:
            stream.readline()
            resources = list(csv.DictReader(stream))
        assert [row["stage"] for row in resources] == ["baseline", "statistics_exported", "session_released"]
        name = f"sample.statistics.step00000012.rank{rank:08d}.bin"
        assert (output / name).read_bytes() == (continuous / "outdat/render" / name).read_bytes()
    flows = re.findall(r"ASTR_INSITU_GPU_FLOW rank=(\d+) frame_downloads=(\d+)", (resumed / "run.log").read_text())
    assert len(flows) == ranks and all(int(count) == 0 for _, count in flows)


def test_override_initial_frame_does_not_resample_statistics(reference, tmp_path):
    args, ranks, mode, continuous, source, _ = reference
    args = SimpleNamespace(**vars(args)); args.output = tmp_path
    config = configuration(mode=mode).replace("initial_frame=f", "initial_frame=t")
    resumed, _ = run_case(args, ROOT, "gpu", ranks, "new_initial_frame", 12, restore=source,
                          override=True, insitu_config=config)
    for name in ("state.h5", "statistics.h5"):
        compare_fields(continuous / FINAL / name, resumed / FINAL / name)
    check_render(resumed, ranks, [5, 12])


def test_enable_renderer_requires_override_and_preserves_statistics(reference, tmp_path):
    args, ranks, mode, continuous, _, _ = reference
    args = SimpleNamespace(**vars(args)); args.output = tmp_path
    inactive, _ = run_case(args, ROOT, "gpu", ranks, "inactive", 5,
                           insitu_config=configuration(render=False, mode=mode))
    source = inactive / "outdat/new/checkpoints/step000000000005"
    config = configuration(mode=mode)
    run_case(args, ROOT, "gpu", ranks, "rejected_enable", 12, restore=source, insitu_config=config,
             reject="native render configuration differs; select explicit override")
    resumed, _ = run_case(args, ROOT, "gpu", ranks, "enabled", 12, restore=source,
                          override=True, insitu_config=config)
    for name in ("state.h5", "statistics.h5"):
        compare_fields(continuous / FINAL / name, resumed / FINAL / name)
    check_render(resumed, ranks, [12])
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 64 * 1024**2


@pytest.mark.parametrize("change", ["interval", "disable"])
def test_render_changes_require_explicit_override(reference, tmp_path, change):
    args, ranks, mode, _, source, _ = reference
    args = SimpleNamespace(**vars(args)); args.output = tmp_path
    config = configuration(render=change != "disable", mode=mode, interval=7 if change == "interval" else 8)
    run_case(args, ROOT, "gpu", ranks, "rejected", 12, restore=source, insitu_config=config,
             reject="native render configuration differs; select explicit override")


@pytest.mark.parametrize("disable", [False, True])
def test_render_checkpoint_repartition_is_not_admitted(reference, tmp_path, disable):
    args, ranks, mode, _, source, _ = reference
    args = SimpleNamespace(**vars(args)); args.output = tmp_path; args.axis = "y"
    config = configuration(render=not disable, mode=mode)
    message = "exact restore rank count" if ranks == 1 else "exact restore partition mismatch"
    run_case(args, ROOT, "gpu", 2, "rejected_partition", 12, restore=source,
             insitu_config=config, override=disable, reject=message)


def test_render_enablement_during_inactive_checkpoint_repartition_is_rejected(reference, tmp_path):
    args, ranks, mode, _, _, _ = reference
    args = SimpleNamespace(**vars(args)); args.output = tmp_path
    inactive, _ = run_case(args, ROOT, "gpu", ranks, "inactive", 5,
                           insitu_config=configuration(render=False, mode=mode))
    source = inactive / "outdat/new/checkpoints/step000000000005"
    args.axis = "y"
    message = "exact restore rank count" if ranks == 1 else "exact restore partition mismatch"
    run_case(args, ROOT, "gpu", 2, "rejected_enable", 12, restore=source,
             insitu_config=configuration(mode=mode), override=True, reject=message)


@pytest.mark.parametrize("change", ["interval", "disable"])
def test_render_override_preserves_numerical_history(reference, tmp_path, change):
    args, ranks, mode, continuous, source, _ = reference
    args = SimpleNamespace(**vars(args)); args.output = tmp_path
    config = configuration(render=change != "disable", mode=mode, interval=7 if change == "interval" else 8)
    resumed, _ = run_case(args, ROOT, "gpu", ranks, "overridden", 12, restore=source,
                          override=True, insitu_config=config)
    for name in ("state.h5", "statistics.h5"):
        compare_fields(continuous / FINAL / name, resumed / FINAL / name)
    if change == "interval":
        check_render(resumed, ranks, [12])
    else:
        assert not list((resumed / "outdat/render").glob("*.jpeg"))


@pytest.mark.parametrize("defect", ["missing", "clock", "flags", "tail", "index", "signature"])
def test_corrupt_render_control_is_rejected(reference, tmp_path, defect):
    args, ranks, mode, _, source, _ = reference
    args = SimpleNamespace(**vars(args)); args.output = tmp_path
    root = tmp_path / "modified_source"
    shutil.copytree(source.parent.parent / "resources", root / "resources")
    batch = root / "checkpoints" / source.name
    shutil.copytree(source, batch)
    payload = batch / "insitu_control.bin"
    raw = bytearray(payload.read_bytes())
    messages = {"missing": "missing native render control", "clock": "native render clock/version mismatch",
                "flags": "invalid native render flags", "tail": "native render control tail",
                "index": "invalid native render schedule history",
                "signature": "native render configuration differs; select explicit override"}
    if defect == "missing":
        payload.unlink()
        # Remove the member from the sealed list to exercise the mandatory provider.
        names = [row.split()[0] for row in (batch / "MANIFEST").read_text().splitlines()[2:]
                 if row.split()[0] != "insitu_control.bin"]
    else:
        if defect == "tail":
            raw.extend(b"x")
        else:
            offset = {"clock": 8, "flags": 40, "index": 285, "signature": 112}[defect]
            value = {"clock": 6, "flags": 2, "index": 100, "signature": 1}[defect]
            raw[offset:offset + 8] = np.array([value], dtype="<i8").tobytes()
        payload.write_bytes(raw)
        names = [row.split()[0] for row in (batch / "MANIFEST").read_text().splitlines()[2:]]
    (batch / "MANIFEST").write_text(seal_file_records(batch, names, "ASTR_CHECKPOINT_BUNDLE 1"))
    size, crc = repair.fingerprint(batch / "MANIFEST")
    (batch / "COMPLETE").write_text(f"ASTR_COMPLETE_1 {size} {crc:016X}\n")
    before = {p.name: p.read_bytes() for p in batch.iterdir() if p.is_file()}
    run_case(args, ROOT, "gpu", ranks, "rejected", 12, restore=batch, insitu_config=configuration(mode=mode),
             reject=messages[defect])
    assert before == {p.name: p.read_bytes() for p in batch.iterdir() if p.is_file()}
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 64 * 1024**2


def test_render_only_exact_continuation_without_statistics(tmp_path):
    args = arguments(tmp_path)
    args.statistics = False
    config = configuration(statistics=False)
    seed, _ = run_case(args, ROOT, "gpu", 1, "seed", 5, insitu_config=config)
    source = seed / "outdat/new/checkpoints/step000000000005"
    resumed, _ = run_case(args, ROOT, "gpu", 1, "resumed", 12, restore=source, insitu_config=config)
    args.output = tmp_path / "plain"
    args.output.mkdir()
    plain, _ = run_case(args, ROOT, "gpu", 1, "plain", 12,
                        insitu_config="&insitu_run enabled=f, statistics=f, render=f /\n")
    compare_fields(plain / FINAL / "state.h5", resumed / FINAL / "state.h5")
    assert not (resumed / FINAL / "statistics.h5").exists()
    check_render(seed, 1, [5], statistics=False)
    check_render(resumed, 1, [8, 12], statistics=False)
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 64 * 1024**2
