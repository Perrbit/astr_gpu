"""Bounded OR6 GPU basic-output reserve, exact restart and fail-closed gates."""
import hashlib
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from run_output_archive_validation import check_accounting, check_frame, check_series, frames, groups
from run_output_restart_validation import (archive_schedule_payload, compare_fields,
                                           compare_statistics, controlled_checkpoint_buffers, run_case)

ROOT = Path(__file__).resolve().parents[2]
EXE = Path(os.environ.get("ASTR_OUTPUT_RUNTIME_EXE", ROOT / "build_gpu_probe/bin/astr")).resolve()
MPIEXEC = Path(os.environ.get("ASTR_OUTPUT_MPIEXEC",
    "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec")).resolve()
RESERVE = 1073741824
BUDGET = 64 * 1024**2
FINAL = "outdat/new/checkpoints/step000000000012"
GUARD = re.compile(r"ASTR_OUTPUT_DEVICE_RESERVE rank=(\d+) phase=(\S+) free_bytes=(\d+)"
                   r" allocation_bytes=(\d+) reserve_bytes=(\d+)")


def arguments(output, statistics=True):
    return SimpleNamespace(output=output, executable=EXE, mpiexec=MPIEXEC, case="tgv", mode="steps",
        restart_step=5, initial_dimension=0, legacy_statistics=False, statistics=statistics,
        initial_restart=False, filter_workspace="scalar", force="feedback", axis="x", no_samples=True)


def reserve_records(case, ranks, expected=RESERVE):
    records = GUARD.findall((case / "run.log").read_text())
    assert records, "missing native GPU reserve measurements"
    minima = []
    for rank in range(ranks):
        selected = [row for row in records if int(row[0]) == rank]
        assert len(selected) > 0 and len(selected) % 2 == 0
        assert [row[1] for row in selected] == ["before_allocate", "after_release"] * (len(selected) // 2)
        for _, phase, free, allocation, reserve in selected:
            free, allocation, reserve = map(int, (free, allocation, reserve))
            assert reserve == expected and free - allocation >= reserve
            assert (allocation > 0) == (phase == "before_allocate")
            assert allocation <= 4096
            minima.append(free - allocation)
    assert {int(row[0]) for row in records} == set(range(ranks))
    accounting = check_accounting(case, "gpu", ranks)
    events = (case / "run.log").read_text().count("ASTR_OUTPUT_ARCHIVE rank=")
    assert len(records) == 2 * events
    return dict(measurements=len(records), min_free_after_planned_allocation=min(minima), **accounting)


@pytest.mark.parametrize("ranks", [1, 2])
def test_basic_reserve_exact_restart_and_output_invariance(ranks, tmp_path, record_property):
    args = arguments(tmp_path)
    options = dict(archive_groups=groups("steps"), buffer_bytes=4096, device_reserve_bytes=RESERVE,
                   checkpoint_interval=99)
    continuous, _ = run_case(args, ROOT, "gpu", ranks, "continuous", 12, **options)
    seed, _ = run_case(args, ROOT, "gpu", ranks, "seed", 5, **options)
    source = seed / "outdat/new/checkpoints/step000000000005"
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    resumed, _ = run_case(args, ROOT, "gpu", ranks, "resumed", 12, restore=source, **options)
    default, _ = run_case(args, ROOT, "gpu", ranks, "default", 12,
                          archive_groups=groups("steps"), buffer_bytes=4096, checkpoint_interval=99)
    off, _ = run_case(args, ROOT, "gpu", ranks, "off", 12, buffer_bytes=4096, checkpoint_interval=99)
    for other in (resumed, default, off):
        for name in ("state.h5", "statistics.h5"):
            compare_fields(continuous / FINAL / name, other / FINAL / name)
    for other in (resumed, default):
        assert archive_schedule_payload(continuous / FINAL / "archives.bin") == archive_schedule_payload(
            other / FINAL / "archives.bin")
        assert (continuous / FINAL / "control.bin").read_bytes() == (other / FINAL / "control.bin").read_bytes()
    compare_statistics(continuous, resumed, off, "gpu", 5, "tgv")
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    for product in ("fields", "slices"):
        original = frames(continuous, product)
        assert set(original) == ({0, 2, 4, 6, 8, 10, 12} if product == "fields" else {0, 3, 6, 9, 12})
        assert set(frames(resumed, product)) == {step for step in original if step > 5}
        for other in (resumed, default):
            for step, frame in frames(other, product).items():
                compare_fields(original[step] / "data.h5", frame / "data.h5")
        check_frame(original[12], product, continuous / FINAL,
                    continuous / "outdat/new/resources/geometry.h5", "gpu")
        for case in (continuous, seed, resumed, default):
            check_series(case, product)
    assert not GUARD.findall((default / "run.log").read_text())
    assert not GUARD.findall((off / "run.log").read_text())
    measurements = [reserve_records(case, ranks) for case in (continuous, seed, resumed)]
    directory_bytes = [sum(p.stat().st_size for p in case.rglob("*") if p.is_file())
                       for case in (continuous, seed, resumed, default, off)]
    assert max(directory_bytes) <= BUDGET
    # Formal statistics are an independent resident allocation, not a tile buffer.
    buffers = controlled_checkpoint_buffers(continuous / FINAL / "state.h5", 23 * 17**3 * 8)
    result = dict(executable_sha256=hashlib.sha256(EXE.read_bytes()).hexdigest(), np=ranks,
        reserve_bytes=RESERVE, exact_restart=True, output_switch_unchanged=True, default_unchanged=True,
        directory_bytes=directory_bytes, controlled_host_bytes=buffers, reserve_measurements=measurements)
    (tmp_path / "reserve_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    test_bytes = sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file())
    assert test_bytes <= BUDGET, test_bytes
    record_property("test_directory_bytes", test_bytes)
    record_property("reserve_validation", json.dumps(result))


@pytest.mark.parametrize("ranks", [1, 2])
def test_impossible_reserve_rejected_before_allocation_and_publication(ranks, tmp_path):
    args = arguments(tmp_path, statistics=False)
    case, _ = run_case(args, ROOT, "gpu", ranks, "impossible", 1, enabled=False,
        archive_groups=groups("steps"), buffer_bytes=4096, device_reserve_bytes=2**63 - 1,
        reject="output GPU free memory reserve before_allocate")
    log = (case / "run.log").read_text()
    records = GUARD.findall(log)
    assert {int(row[0]) for row in records} == set(range(ranks))
    assert all(row[1] == "before_allocate" and int(row[4]) == 2**63 - 1 for row in records)
    assert "after_release" not in log and "ASTR_OUTPUT_ARCHIVE rank=" not in log
    assert not list((case / "outdat/new").rglob("COMPLETE"))
    for ledger in (case / "outdat/new").rglob("series.frames"):
        assert ledger.read_text().splitlines() == ["ASTR_FRAME_SERIES_1"]
    assert not list((case / "outdat/new").rglob("series.xdmf"))
    assert not list((case / "outdat/new").glob("fields/segment*/step*/data.h5"))


def test_basic_reserve_memcheck_np2(tmp_path, record_property):
    args = arguments(tmp_path, statistics=False)
    case, _ = run_case(args, ROOT, "gpu", 2, "memcheck", 2, enabled=False,
        archive_groups=groups("steps"), buffer_bytes=4096, device_reserve_bytes=RESERVE, memcheck=True)
    record_property("reserve_measurements", json.dumps(reserve_records(case, 2)))


def registered_case(output, case_name):
    from test_output_repartition_runtime import (air5_arguments, channel_arguments,
                                                 curve_arguments, frozen_channel_initial)
    if case_name == "channel":
        args = channel_arguments(output, "gpu", "x", "full")
        return args, dict(initial_resource=frozen_channel_initial(output / "initial.h5"))
    if case_name in ("curve", "dynamic"):
        args = curve_arguments(output, "gpu", "x" if case_name == "curve" else "z")
        args.case = case_name
        args.inflow_count = 12
        return args, {}
    args = air5_arguments(output, "gpu")
    args.case = "hbl" if case_name == "air5hbl" else "sbli"
    args.reconstruction = 5 if args.case == "hbl" else 3
    args.axis = "z" if args.case == "hbl" else "x"
    return args, {}


def run_registered(args, name, steps, *, restore=None, archive=True, reserve=RESERVE, **extra):
    options = dict(archive_groups=groups("steps") if archive else None, buffer_bytes=4096,
                   device_reserve_bytes=reserve, restore=restore, **extra)
    if args.case in ("hbl", "sbli"):
        from run_output_air5_restart_validation import launch
        return launch(args, "gpu", 2, name, steps, interval=99, **options)[0]
    return run_case(args, ROOT, "gpu", 2, name, steps, checkpoint_interval=99, **options)[0]


@pytest.mark.parametrize("case_name", ["channel", "curve", "dynamic", "air5hbl", "air5sbli"])
def test_registered_basic_reserve_exact_continuation(case_name, tmp_path, record_property):
    args, extra = registered_case(tmp_path, case_name)
    continuous = run_registered(args, "continuous", 12, **extra)
    seed = run_registered(args, "seed", 5, **extra)
    source = seed / "outdat/new/checkpoints/step000000000005"
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    resumed = run_registered(args, "resumed", 12, restore=source, **extra)
    off = run_registered(args, "off", 12, archive=False, reserve=0, **extra)
    for other in (resumed, off):
        compare_fields(continuous / FINAL / "state.h5", other / FINAL / "state.h5")
        if case_name == "dynamic":
            compare_fields(continuous / FINAL / "inflow.h5", other / FINAL / "inflow.h5")
    controls = ["control.bin"] + (["air5_config.bin"] if case_name.startswith("air5") else [])
    for name in controls:
        assert (continuous / FINAL / name).read_bytes() == (resumed / FINAL / name).read_bytes()
    assert archive_schedule_payload(continuous / FINAL / "archives.bin") == archive_schedule_payload(
        resumed / FINAL / "archives.bin")
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    for product in ("fields", "slices"):
        original = frames(continuous, product)
        assert set(frames(resumed, product)) == {step for step in original if step > 5}
        for step, frame in frames(resumed, product).items():
            compare_fields(original[step] / "data.h5", frame / "data.h5")
        check_frame(original[12], product, continuous / FINAL,
                    continuous / "outdat/new/resources/geometry.h5", "gpu")
        for case in (continuous, seed, resumed):
            check_series(case, product)
    measurements = [reserve_records(case, 2) for case in (continuous, seed, resumed)]
    assert not GUARD.findall((off / "run.log").read_text())
    test_bytes = sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file())
    assert test_bytes <= BUDGET, test_bytes
    record_property("result", json.dumps(dict(case=case_name, np=2, axis=args.axis,
        executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
        exact_continuation=True, output_switch_unchanged=True, test_directory_bytes=test_bytes,
        reserve_measurements=measurements)))


@pytest.mark.parametrize("case_name", ["curve", "air5sbli"])
def test_registered_impossible_reserve_refused(case_name, tmp_path):
    args, extra = registered_case(tmp_path, case_name)
    case = run_registered(args, "impossible", 1, reserve=2**63-1,
        reject="output GPU free memory reserve before_allocate", **extra)
    log = (case / "run.log").read_text()
    records = GUARD.findall(log)
    assert {int(row[0]) for row in records} == {0, 1}
    assert all(row[1] == "before_allocate" and int(row[4]) == 2**63-1 for row in records)
    assert "after_release" not in log and "ASTR_OUTPUT_ARCHIVE rank=" not in log
    assert not list((case / "outdat/new").rglob("COMPLETE"))
    assert not list((case / "outdat/new").glob("fields/segment*/step*/data.h5"))
    assert not list((case / "outdat/new").rglob("series.xdmf"))
