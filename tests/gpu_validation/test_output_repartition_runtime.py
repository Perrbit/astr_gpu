"""Approved bounded periodic TGV rank-count and NP=2 slab-direction gates."""
import os
from pathlib import Path
import re
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from run_output_archive_validation import groups, check_series, check_frame, check_series_reader
from run_output_restart_validation import archive_schedule_payload, run_case, controlled_checkpoint_buffers, compare_fields

ROOT = Path(__file__).resolve().parents[2]
EXE = Path(os.environ.get("ASTR_OUTPUT_RUNTIME_EXE", ROOT / "build_gpu_probe/bin/astr")).resolve()
MPIEXEC = Path(os.environ.get("ASTR_OUTPUT_MPIEXEC",
    "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec")).resolve()
ATOL = 2e-10
FINAL = "outdat/new/checkpoints/step000000000012"


def arguments(output, backend_axis="x", workspace="scalar"):
    return SimpleNamespace(output=output, executable=EXE, mpiexec=MPIEXEC,
        case="tgv", mode="steps", restart_step=5, initial_dimension=0,
        legacy_statistics=False, statistics=True, initial_restart=False,
        filter_workspace=workspace, force="feedback", axis=backend_axis, no_samples=True)


def clocks(case):
    records = re.findall(r"ASTR_CFL complete_step=(\d+) state_time=\s*(\S+) dt=\s*(\S+)",
                         (case / "run.log").read_text())
    return [(int(step), float(time), float(dt)) for step, time, dt in records]


def compare_global_state(reference, actual, statistics=False):
    with h5py.File(reference, "r") as left, h5py.File(actual, "r") as right:
        # The final target partition is identical; its halos are rebuilt, not replayed.
        assert left["identity"][:].tobytes() == right["identity"][:].tobytes()
        assert left["partitions"][:].tobytes() == right["partitions"][:].tobytes()
        fields = sorted(name for name in left if re.fullmatch(r"q\d{4}", name))
        assert fields == sorted(name for name in right if re.fullmatch(r"q\d{4}", name))
        assert len(fields) == (34 if statistics else 11)
        maxima = {}
        for name in fields:
            a, b = left[name][:], right[name][:]
            assert a.shape == b.shape and a.dtype == b.dtype
            assert np.isfinite(a).all() and np.isfinite(b).all()
            maxima[name] = float(np.max(np.abs(a - b)))
            assert maxima[name] <= ATOL, (name, maxima[name])
            if statistics and int(name[1:]) in (1, 2, 3, 8, 34):
                # Window bounds, last sample time, covered duration and active flag.
                assert a.tobytes() == b.tobytes(), name
        if statistics:
            a, b = left["metadata"][:], right["metadata"][:]
            assert a[:4].tobytes() == b[:4].tobytes()
            a, b = a[4:].view(np.float64), b[4:].view(np.float64)
            assert np.isfinite(a).all() and np.isfinite(b).all()
            maxima["regional"] = float(np.max(np.abs(a - b)))
            assert maxima["regional"] <= ATOL
            assert a[[0, 1, 2, 7, 33]].tobytes() == b[[0, 1, 2, 7, 33]].tobytes()
        return maxima


@pytest.mark.parametrize("backend,workspace", [("cpu", "scalar"), ("gpu", "scalar"), ("gpu", "full")])
@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_filtered_tgv_partition_independence(backend, workspace, axis, tmp_path, record_property):
    args = arguments(tmp_path, axis, workspace)
    single, _ = run_case(args, ROOT, backend, 1, "baseline", 4)
    divided, _ = run_case(args, ROOT, backend, 2, "baseline", 4)
    assert clocks(single) == clocks(divided)
    for name in ("state.h5", "statistics.h5"):
        checkpoint = "outdat/new/checkpoints/step000000000004/" + name
        with h5py.File(single / checkpoint) as a, h5py.File(divided / checkpoint) as b:
            errors = []
            for field in (k for k in a if re.fullmatch(r"q\d{4}", k)):
                left, right = a[field][:], b[field][:]
                assert left.shape == right.shape and left.dtype == right.dtype
                assert np.isfinite(left).all() and np.isfinite(right).all()
                errors.append(float(np.max(np.abs(left - right))))
            record_property(name + "_max_abs", max(errors))
            assert max(errors) <= ATOL, (name, max(errors))


@pytest.mark.parametrize("ranks,axis,workspace", [(1, "x", "scalar"), (2, "x", "scalar"),
    (2, "y", "scalar"), (2, "z", "full")])
def test_completed_step_cpu_gpu_filter_state(ranks, axis, workspace, tmp_path, record_property):
    args = arguments(tmp_path, axis, workspace)
    cpu, _ = run_case(args, ROOT, "cpu", ranks, "comparison", 12)
    gpu, _ = run_case(args, ROOT, "gpu", ranks, "comparison", 12)
    assert clocks(cpu) == clocks(gpu)
    for name in ("state.h5", "statistics.h5"):
        with h5py.File(cpu / FINAL / name) as a, h5py.File(gpu / FINAL / name) as b:
            maxima = []
            for field in (k for k in a if re.fullmatch(r"q\d{4}", k)):
                # GPU statistics own unique periodic nodes; upper endpoint
                # padding is not a statistics value and is not compared to CPU.
                nodes = (slice(0, 16),) * 3 if name == "statistics.h5" else np.s_[:]
                left, right = a[field][nodes], b[field][nodes]
                assert np.isfinite(left).all() and np.isfinite(right).all()
                maxima.append(float(np.max(np.abs(left - right))))
            assert max(maxima) <= ATOL, (name, max(maxima))
            if name == "statistics.h5":
                left, right = a["metadata"][4:].view(np.float64), b["metadata"][4:].view(np.float64)
                assert np.isfinite(left).all() and np.isfinite(right).all()
                error = float(np.max(np.abs(left - right)))
                assert error <= ATOL, ("regional", error)
                record_property("regional_max_abs", error)
            record_property(name + "_max_abs", max(maxima))


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("source_ranks,target_ranks", [(1, 2), (2, 1)])
def test_tgv_repartition(backend, axis, source_ranks, target_ranks, tmp_path, record_property):
    args = arguments(tmp_path, axis)
    archive = groups("steps")
    reference, a_bytes = run_case(args, ROOT, backend, target_ranks, "continuous", 12,
                                  archive_groups=archive)
    seed, b_bytes = run_case(args, ROOT, backend, source_ranks, "seed", 5,
                             archive_groups=archive)
    source = seed / "outdat/new/checkpoints/step000000000005"
    restored, c_bytes = run_case(args, ROOT, backend, target_ranks, "restored", 12,
                                 restore=source, archive_groups=archive)
    assert "ASTR_OUTPUT_REPARTITION" in (restored / "run.log").read_text()
    assert clocks(seed) + clocks(restored) == clocks(reference)
    assert [row[0] for row in clocks(reference)] == list(range(12))
    for name in ("control.bin", "archives.bin"):
        read = archive_schedule_payload if name == "archives.bin" else Path.read_bytes
        assert read(reference / FINAL / name) == read(restored / FINAL / name)
    for name, statistics in (("state.h5", False), ("statistics.h5", True)):
        maxima = compare_global_state(reference / FINAL / name, restored / FINAL / name, statistics)
        record_property(name + "_max_abs", max(maxima.values()))
    for product in ("fields", "slices"):
        expected = check_series(reference, product)
        earlier = check_series(seed, product)
        later = check_series(restored, product)
        # Seed's forced final frame at step five is not in the regular schedule.
        assert [t for t in earlier if t < 0.005] + later == expected
    for case in (reference, seed, restored):
        checkpoint = case / (FINAL if case != seed else "outdat/new/checkpoints/step000000000005")
        assert max(controlled_checkpoint_buffers(checkpoint / "state.h5", 0, backend)) <= 64 * 1024**2
        with h5py.File(checkpoint / "statistics.h5") as state:
            for row in state["partitions"][:].reshape(-1, 8):
                # Host packing and a conservative equally sized temporary.
                bound = 16 * int(np.prod(row[3:6] + 1)) * 34 + 4096
                assert bound <= 64 * 1024**2
    record_property("case_bytes", [a_bytes, b_bytes, c_bytes])


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_repartition_keeps_exact_restore(backend, tmp_path):
    args = arguments(tmp_path)
    reference, _ = run_case(args, ROOT, backend, 2, "continuous", 12)
    seed, _ = run_case(args, ROOT, backend, 2, "seed", 5)
    source = seed / "outdat/new/checkpoints/step000000000005"
    restored, _ = run_case(args, ROOT, backend, 2, "restored", 12, restore=source)
    for name in ("state.h5", "statistics.h5"):
        compare_fields(reference / FINAL / name, restored / FINAL / name)
    for name in ("control.bin", "archives.bin"):
        read = archive_schedule_payload if name == "archives.bin" else Path.read_bytes
        assert read(reference / FINAL / name) == read(restored / FINAL / name)


def check_slab_direction(backend, source_axis, target_axis, tmp_path, record_property, workspace="scalar"):
    args = arguments(tmp_path, target_axis, workspace)
    archive = groups("steps")
    reference, a_bytes = run_case(args, ROOT, backend, 2, "continuous", 12, archive_groups=archive)
    args.axis = source_axis
    seed, b_bytes = run_case(args, ROOT, backend, 2, "seed", 5, archive_groups=archive)
    source = seed / "outdat/new/checkpoints/step000000000005"
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    args.axis = target_axis
    restored, c_bytes = run_case(args, ROOT, backend, 2, "restored", 12, restore=source, archive_groups=archive)
    assert "ASTR_OUTPUT_REPARTITION" in (restored / "run.log").read_text()
    assert clocks(seed) + clocks(restored) == clocks(reference)
    for name in ("state.h5", "statistics.h5"):
        maxima = compare_global_state(reference / FINAL / name, restored / FINAL / name, name == "statistics.h5")
        record_property(name + "_max_abs", max(maxima.values()))
    assert (reference / FINAL / "control.bin").read_bytes() == (restored / FINAL / "control.bin").read_bytes()
    assert (reference / FINAL / "insitu_control.bin").read_bytes() == (restored / FINAL / "insitu_control.bin").read_bytes()
    assert archive_schedule_payload(reference / FINAL / "archives.bin") == archive_schedule_payload(restored / FINAL / "archives.bin")
    for product in ("fields", "slices"):
        expected, earlier, later = (check_series(case, product) for case in (reference, seed, restored))
        assert [t for t in earlier if t < 0.005] + later == expected
        frame = restored / "outdat/new" / product / "segment00000000/step000000000012"
        check_frame(frame, product, restored / FINAL, restored / "outdat/new/resources/geometry.h5", backend)
        if (backend, source_axis, target_axis) in (("cpu", "x", "z"), ("gpu", "z", "y")):
            assert check_series_reader(restored, product)["frames"] == len(later)
    for case in (reference, seed, restored):
        final = "outdat/new/checkpoints/step000000000005" if case == seed else FINAL
        assert max(controlled_checkpoint_buffers(case / final / "state.h5", 0, backend)) <= 64 * 1024**2
        with h5py.File(case / final / "statistics.h5") as state:
            for row in state["partitions"][:].reshape(-1, 8):
                assert 16 * int(np.prod(row[3:6] + 1)) * 34 + 4096 <= 64 * 1024**2
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    record_property("case_bytes", [a_bytes, b_bytes, c_bytes])
    total = sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file())
    assert total < 64 * 1024**2
    record_property("test_root_bytes", total)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("source_axis,target_axis", [("x", "y"), ("x", "z"), ("y", "x"),
                                                    ("y", "z"), ("z", "x"), ("z", "y")])
def test_tgv_slab_direction_repartition(backend, source_axis, target_axis, tmp_path, record_property):
    check_slab_direction(backend, source_axis, target_axis, tmp_path, record_property)


def test_tgv_slab_direction_full_workspace(tmp_path, record_property):
    check_slab_direction("gpu", "z", "y", tmp_path, record_property, workspace="full")


def test_unapproved_larger_rank_count_is_rejected(tmp_path):
    args = arguments(tmp_path)
    seed, _ = run_case(args, ROOT, "cpu", 1, "seed", 5)
    # Each local extent stays above the existing five-layer halo requirement.
    run_case(args, ROOT, "cpu", 4, "rejected", 12, topology=(2, 2, 1),
             restore=seed / "outdat/new/checkpoints/step000000000005",
             reject="validated repartition gate requires at most two ranks")


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_unvalidated_no_filter_repartition_is_rejected(backend, tmp_path):
    args = arguments(tmp_path)
    seed, _ = run_case(args, ROOT, backend, 1, "seed", 5, lfilter=False)
    run_case(args, ROOT, backend, 2, "rejected", 12, lfilter=False,
             restore=seed / "outdat/new/checkpoints/step000000000005",
             reject="exact restore rank count")
