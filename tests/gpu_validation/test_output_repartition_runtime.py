"""Approved bounded TGV, channel, static-CURVE and AIR5 repartition gates."""
import hashlib
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from run_output_archive_validation import groups, check_series, check_frame, check_series_reader, frames
from run_output_restart_validation import (archive_schedule_payload, run_case, controlled_checkpoint_buffers,
                                            compare_fields, check_geometry_padding)

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


def channel_arguments(output, backend, axis, workspace="scalar"):
    args = arguments(output, axis, workspace)
    args.case = "channel"
    args.initial_dimension = 3
    args.statistics = False
    args.legacy_statistics = backend == "cpu"
    args.force = "fixed"
    if backend == "cpu":
        args.executable = Path(os.environ.get("ASTR_OUTPUT_CPU_EXE",
            ROOT / "build_release_restart_cpu/bin/astr")).resolve()
    return args


def frozen_channel_initial(path):
    k, j, i = np.indices((17, 17, 17))
    x, z, eta = 2*np.pi*i/16, 2*np.pi*k/16, j/8-1
    envelope = 1-eta**2
    perturbation = envelope * np.cos(x) * np.cos(z)
    fields = {"ro": 1+0.001*perturbation, "t": 1+0.002*perturbation,
              "u1": 1.5*envelope+0.005*perturbation,
              "u2": 0.001*envelope*np.sin(x)*np.cos(z),
              "u3": 0.001*envelope*np.cos(x)*np.sin(z)}
    with h5py.File(path, "x") as handle:
        for name, values in fields.items():
            values[:, :, -1] = values[:, :, 0]
            values[-1, :, :] = values[0, :, :]
            assert np.isfinite(values).all()
            if name in ("ro", "t"):
                assert np.min(values) > 0
            else:
                assert np.all(values[:, [0, 16], :] == 0)
            handle[name] = values.astype("<f8")
    assert path.stat().st_size < 2*1024**2
    return path


def compare_channel_control(reference, actual, exact=False):
    left, right = reference.read_bytes(), actual.read_bytes()
    assert left[:8] == right[:8] == b"ASTROC04"
    assert len(left) == len(right) and len(left) >= 652
    if exact:
        assert left == right
    # The six driver reals follow contract/clock/schedules and three 128-byte strings.
    assert left[:604] == right[:604] and left[652:] == right[652:]
    a, b = (np.frombuffer(payload[604:652], dtype="<f8") for payload in (left, right))
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert np.array_equal(a[:3], [1e-4, 0, 0]) and np.array_equal(b[:3], a[:3])
    assert a[-1:].tobytes() == b[-1:].tobytes()
    error = float(np.max(np.abs(a-b)))
    assert error <= ATOL
    return error


def compare_channel_mean(reference, actual):
    errors = {}
    with h5py.File(reference, "r") as a, h5py.File(actual, "r") as b:
        for name in ("identity", "partitions", "metadata"):
            assert a[name][:].tobytes() == b[name][:].tobytes(), name
        assert a["identity"][12] == 6 and a["metadata"][1] > 0
        names = sorted(name for name in a if re.fullmatch(r"q\d{4}", name))
        assert len(names) == 44
        for name in names:
            left, right = a[name][:], b[name][:]
            assert left.shape == right.shape and left.dtype == right.dtype
            assert np.isfinite(left).all() and np.isfinite(right).all()
            errors[name] = float(np.max(np.abs(left-right)))
            assert errors[name] <= ATOL, (name, errors[name])
    return errors


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("axis", ["x", "z"])
def test_channel_filtered_partition_independence(backend, axis, tmp_path, record_property):
    initial = frozen_channel_initial(tmp_path / "frozen_channel.h5")
    args = channel_arguments(tmp_path, backend, axis)
    args.initial_restart = True
    cases = [run_case(args, ROOT, backend, ranks, "fresh", 1, buffer_bytes=4096,
                      initial_resource=initial)[0] for ranks in (1, 2)]
    errors = {}
    for step in (0, 1):
        relative = f"outdat/new/checkpoints/step{step:012d}/state.h5"
        with h5py.File(cases[0] / relative) as a, h5py.File(cases[1] / relative) as b:
            for name in (k for k in a if re.fullmatch(r"q\d{4}", k)):
                left, right = a[name][:], b[name][:]
                assert np.isfinite(left).all() and np.isfinite(right).all()
                if step == 0:
                    assert left.tobytes() == right.tobytes(), name
                else:
                    errors[name] = float(np.max(np.abs(left-right)))
    record_property("one_step_errors", errors)
    assert max(errors.values()) <= ATOL, errors
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 64*1024**2


@pytest.mark.parametrize("ranks,axis,workspace", [(1, "x", "scalar"), (1, "x", "full"),
                                                  (2, "x", "scalar"), (2, "z", "full")])
def test_channel_completed_step_cpu_gpu(ranks, axis, workspace, tmp_path, record_property):
    initial = frozen_channel_initial(tmp_path / "frozen_channel.h5")
    cpu_args = channel_arguments(tmp_path, "cpu", axis, workspace)
    gpu_args = channel_arguments(tmp_path, "gpu", axis, workspace)
    cpu, _ = run_case(cpu_args, ROOT, "cpu", ranks, "comparison", 12,
                      buffer_bytes=4096, initial_resource=initial)
    gpu, _ = run_case(gpu_args, ROOT, "gpu", ranks, "comparison", 12,
                      buffer_bytes=4096, initial_resource=initial)
    assert clocks(cpu) == clocks(gpu)
    errors = compare_global_state(cpu / FINAL / "state.h5", gpu / FINAL / "state.h5")
    record_property("state_errors", errors)
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 64*1024**2


@pytest.mark.parametrize("axis,workspace", [("x", "scalar"), ("z", "full")])
def test_channel_filtered_memcheck(axis, workspace, tmp_path):
    initial = frozen_channel_initial(tmp_path / "frozen_channel.h5")
    args = channel_arguments(tmp_path, "gpu", axis, workspace)
    case, _ = run_case(args, ROOT, "gpu", 2, "memcheck", 3,
                       buffer_bytes=4096, initial_resource=initial, memcheck=True)
    assert [step for step, _, _ in clocks(case)] == [0, 1, 2]


def check_channel_repartition(backend, source_ranks, source_axis, target_ranks, target_axis,
                              tmp_path, record_property, exact=False, workspace="scalar"):
    initial = frozen_channel_initial(tmp_path / "frozen_channel.h5")
    initial_bytes = initial.read_bytes()
    args = channel_arguments(tmp_path, backend, target_axis, workspace)
    selected = groups("steps")
    reference, _ = run_case(args, ROOT, backend, target_ranks, "continuous", 12,
        archive_groups=selected, buffer_bytes=4096, initial_resource=initial)
    args.axis = source_axis
    seed, _ = run_case(args, ROOT, backend, source_ranks, "seed", 5,
        archive_groups=selected, buffer_bytes=4096, initial_resource=initial)
    source = seed / "outdat/new/checkpoints/step000000000005"
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    args.axis = target_axis
    restored, _ = run_case(args, ROOT, backend, target_ranks, "restored", 12,
        restore=source, archive_groups=selected, buffer_bytes=4096, initial_resource=initial)
    assert ("ASTR_OUTPUT_REPARTITION" in (restored / "run.log").read_text()) != exact
    assert clocks(seed) + clocks(restored) == clocks(reference)
    assert [row[0] for row in clocks(reference)] == list(range(12))
    errors = compare_global_state(reference / FINAL / "state.h5", restored / FINAL / "state.h5")
    if exact:
        compare_fields(reference / FINAL / "state.h5", restored / FINAL / "state.h5")
    driver_error = compare_channel_control(reference / FINAL / "control.bin", restored / FINAL / "control.bin", exact)
    mean_errors = {}
    if backend == "cpu":
        mean_errors = compare_channel_mean(reference / FINAL / "statistics.h5", restored / FINAL / "statistics.h5")
        if exact:
            compare_fields(reference / FINAL / "statistics.h5", restored / FINAL / "statistics.h5")
    for filename in ("archives.bin", "insitu_control.bin"):
        read = archive_schedule_payload if filename == "archives.bin" else Path.read_bytes
        assert read(reference / FINAL / filename) == read(restored / FINAL / filename)
    for product in ("fields", "slices"):
        expected, earlier, later = (check_series(case, product) for case in (reference, seed, restored))
        assert [t for t in earlier if t < 0.005] + later == expected
        frame = restored / "outdat/new" / product / "segment00000000/step000000000012"
        check_frame(frame, product, restored / FINAL, restored / "outdat/new/resources/geometry.h5", backend)
    budgets = []
    for case in (reference, seed, restored):
        checkpoint = case / ("outdat/new/checkpoints/step000000000005" if case == seed else FINAL)
        assert max(controlled_checkpoint_buffers(checkpoint / "state.h5", 0, backend)) <= 64*1024**2
        if backend == "cpu":
            with h5py.File(checkpoint / "statistics.h5") as state:
                for row in state["partitions"][:].reshape(-1, 8):
                    assert 16 * int(np.prod(row[3:6] + 1)) * 44 + 4096 <= 64*1024**2
        frozen = case / "outdat/new/resources/flowini3d.h5"
        assert frozen.read_bytes() == initial_bytes
        if case == restored:
            assert not (case / "datin/flowini3d.h5").exists()
        budgets.append(sum(p.stat().st_size for p in case.rglob("*") if p.is_file()))
    assert initial.read_bytes() == initial_bytes
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    total = sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file())
    assert total < 64*1024**2
    with args.executable.open("rb") as stream:
        executable_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    result = dict(backend=backend, source=[source_ranks, source_axis], target=[target_ranks, target_axis],
        state_max_abs=max(errors.values()), driver_max_abs=driver_error,
        mean_max_abs=max(mean_errors.values(), default=0), exact=exact, filter_workspace=workspace,
        frozen_initial_sha256=hashlib.sha256(initial_bytes).hexdigest(), test_root_bytes=total,
        case_bytes=budgets, executable_sha256=executable_sha256)
    (tmp_path / "channel_repartition.json").write_text(json.dumps(result, indent=2)+"\n")
    record_property("result", result)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("source_ranks,source_axis,target_ranks,target_axis", [
    (1, "x", 2, "x"), (2, "x", 1, "x"), (1, "z", 2, "z"), (2, "z", 1, "z"),
    (2, "x", 2, "z"), (2, "z", 2, "x")])
def test_channel_repartition(backend, source_ranks, source_axis, target_ranks, target_axis, tmp_path, record_property):
    check_channel_repartition(backend, source_ranks, source_axis, target_ranks, target_axis, tmp_path, record_property)


@pytest.mark.parametrize("backend,axis,workspace", [("cpu", "x", "scalar"), ("gpu", "z", "full")])
def test_channel_repartition_keeps_exact_restore(backend, axis, workspace, tmp_path, record_property):
    check_channel_repartition(backend, 2, axis, 2, axis, tmp_path, record_property,
                              exact=True, workspace=workspace)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("source_ranks,source_axis,target_ranks,target_axis,rejection", [
    (1, "x", 2, "y", "exact restore rank count"),
    (2, "y", 1, "x", "repartition nonperiodic direction must stay unpartitioned")])
def test_channel_y_repartition_is_rejected(backend, source_ranks, source_axis, target_ranks, target_axis,
                                         rejection, tmp_path):
    initial = frozen_channel_initial(tmp_path / "frozen_channel.h5")
    args = channel_arguments(tmp_path, backend, source_axis)
    seed, _ = run_case(args, ROOT, backend, source_ranks, "seed", 5, initial_resource=initial)
    args.axis = target_axis
    run_case(args, ROOT, backend, target_ranks, "rejected", 12, initial_resource=initial,
        restore=seed / "outdat/new/checkpoints/step000000000005", reject=rejection)


def curve_arguments(output, backend, axis="z"):
    args = arguments(output, axis, "full" if backend == "gpu" else "scalar")
    args.case = "curve"
    args.statistics = False
    if backend == "cpu":
        args.executable = Path(os.environ.get("ASTR_OUTPUT_CPU_EXE",
            ROOT / "build_release_restart_cpu/bin/astr")).resolve()
    return args


def compare_curve_physical(reference, actual, components):
    errors = {}
    with h5py.File(reference) as a, h5py.File(actual) as b:
        assert np.array_equal(a["identity"][3:6], b["identity"][3:6])
        for component in range(1, components+1):
            name = f"q{component:04d}"
            left, right = a[name][:], b[name][:]
            assert left.dtype == right.dtype and left.shape == right.shape
            assert np.isfinite(left).all() and np.isfinite(right).all()
            errors[name] = float(np.max(np.abs(left-right)))
            assert errors[name] <= ATOL, (name, errors[name])
    return errors


def compare_curve_target_halos(reference, actual, scales=None):
    """Compare shared nodes, interior x face halos and periodic-z face halos."""
    maximum, normalized, checked, offset = 0.0, 0.0, 0, 0
    with h5py.File(reference) as a, h5py.File(actual) as b:
        assert np.array_equal(a["partitions"][:], b["partitions"][:])
        global_shape = a["identity"][3:6]
        components = int(a["identity"][6])
        assert components == int(b["identity"][6])
        assert components in (11, 13) if scales is None else len(scales) == components
        for row in a["partitions"][:].reshape(-1, 8):
            origin, cells, halo, count = row[:3], row[3:6], int(row[6]), int(row[7])
            indices = np.indices(tuple(cells+2*halo+1))
            physical = (indices >= halo) & (indices <= (halo+cells)[:, None, None, None])
            owned = cells+(origin+cells == global_shape-1)
            owner = np.all((indices >= halo) &
                (indices < (halo+owned)[:, None, None, None]), axis=0)
            global_indices = indices+(origin-halo)[:, None, None, None]
            interior_x = (global_indices[0] >= 0) & (global_indices[0] < global_shape[0])
            wanted = np.all(physical, axis=0) | (physical[0] & physical[1] & ~physical[2]) | \
                (~physical[0] & interior_x & physical[1] & physical[2])
            mask = wanted.ravel(order="F")[~owner.ravel(order="F")]
            assert count == components*mask.size
            left, right = (f["rank_extras"][offset:offset+count].reshape(components, -1) for f in (a, b))
            assert np.isfinite(left).all() and np.isfinite(right).all()
            differences = np.max(np.abs(left[:, mask]-right[:, mask]), axis=1)
            error = float(np.max(differences))
            if scales is None:
                assert error <= ATOL, error
            else:
                assert np.all(differences[scales == 0] == 0)
                scaled = float(np.max(differences/np.where(scales > 0, scales, 1.)))
                assert scaled <= ATOL, scaled
                normalized = max(normalized, scaled)
            maximum = max(maximum, error)
            checked += components*int(np.count_nonzero(mask))
            offset += count
        assert offset == a["rank_extras"].size == b["rank_extras"].size and checked > 0
    return dict(max_abs=maximum, max_normalized=normalized, values_checked=checked)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("axis", ["z", "x"])
def test_curve_partition_independence(backend, axis, tmp_path, record_property):
    args = curve_arguments(tmp_path, backend, axis)
    one, _ = run_case(args, ROOT, backend, 1, "one", 1, buffer_bytes=4096)
    two, _ = run_case(args, ROOT, backend, 2, "two", 1, buffer_bytes=4096)
    checkpoint = "outdat/new/checkpoints/step000000000001/state.h5"
    errors = compare_curve_physical(one / checkpoint, two / checkpoint, 11)
    geometry = "outdat/new/resources/geometry.h5"
    geometry_errors = compare_curve_physical(one / geometry, two / geometry, 13)
    for case in (one, two):
        with h5py.File(case / checkpoint) as state:
            assert np.isfinite(state["rank_extras"][:]).all()
    record_property("state_max_abs", max(errors.values()))
    record_property("geometry_max_abs", max(geometry_errors.values()))
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 64*1024**2


def check_curve_repartition(backend, source_ranks, target_ranks, tmp_path, record_property,
                            exact=False, memcheck=False, source_axis="z", target_axis="z"):
    args = curve_arguments(tmp_path, backend, target_axis)
    archive = groups("steps", dt=1e-5)
    reference, _ = run_case(args, ROOT, backend, target_ranks, "continuous", 12,
        archive_groups=archive, buffer_bytes=4096)
    args.axis = source_axis
    seed, _ = run_case(args, ROOT, backend, source_ranks, "seed", 5,
        archive_groups=archive, buffer_bytes=4096)
    source = seed / "outdat/new/checkpoints/step000000000005"
    before = {str(p.relative_to(seed / "outdat/new")): p.read_bytes()
        for p in (seed / "outdat/new").rglob("*") if p.is_file()}
    args.axis = target_axis
    restored, _ = run_case(args, ROOT, backend, target_ranks, "restored", 12,
        restore=source, archive_groups=archive, buffer_bytes=4096, memcheck=memcheck)
    assert ("ASTR_OUTPUT_REPARTITION bl" in (restored / "run.log").read_text()) != exact
    assert clocks(seed)+clocks(restored) == clocks(reference)
    errors = compare_global_state(reference / FINAL / "state.h5", restored / FINAL / "state.h5")
    halos = compare_curve_target_halos(reference / FINAL / "state.h5", restored / FINAL / "state.h5")
    geometry_errors = compare_curve_physical(reference / "outdat/new/resources/geometry.h5",
        restored / "outdat/new/resources/geometry.h5", 13)
    geometry_halos = compare_curve_target_halos(reference / "outdat/new/resources/geometry.h5",
        restored / "outdat/new/resources/geometry.h5")
    for filename in ("control.bin", "archives.bin", "insitu_control.bin"):
        read = archive_schedule_payload if filename == "archives.bin" else Path.read_bytes
        assert read(reference / FINAL / filename) == read(restored / FINAL / filename)
    if exact:
        compare_fields(reference / FINAL / "state.h5", restored / FINAL / "state.h5")
        compare_fields(reference / "outdat/new/resources/geometry.h5", restored / "outdat/new/resources/geometry.h5")
    readers = {}
    for product in ("fields", "slices"):
        expected, earlier, later = (frames(case, product) for case in (reference, seed, restored))
        assert sorted(step for step in earlier if step < 5)+sorted(later) == sorted(expected)
        assert [t for step, t in zip(sorted(earlier), check_series(seed, product)) if step < 5]+check_series(
            restored, product) == check_series(reference, product)
        for step in later:
            with h5py.File(later[step] / "data.h5") as a, h5py.File(expected[step] / "data.h5") as b:
                names = [None] if product == "fields" else ["i000000000008", "j000000000000", "k000000000016"]
                for name in names:
                    left, right = (a[name], b[name]) if name else (a, b)
                    for field in ("density", "velocity_x", "velocity_y", "velocity_z", "pressure", "temperature", "velocity"):
                        assert np.isfinite(left[field][:]).all() and np.isfinite(right[field][:]).all()
                        assert np.max(np.abs(left[field][:]-right[field][:])) <= ATOL
        check_frame(later[12], product, restored / FINAL, restored / "outdat/new/resources/geometry.h5", backend)
        readers[product] = check_series_reader(restored, product)
        assert readers[product]["frames"] == len(later)
    for case in (reference, seed, restored):
        final = case / ("outdat/new/checkpoints/step000000000005" if case == seed else FINAL)
        assert max(controlled_checkpoint_buffers(final / "state.h5", 0, backend)) <= 64*1024**2
        assert max(controlled_checkpoint_buffers(case / "outdat/new/resources/geometry.h5", 0)) <= 64*1024**2
        assert check_geometry_padding(case / "outdat/new/resources/geometry.h5", (False, False, True)) > 0
        assert not (final / "statistics.h5").exists()
        if case == restored:
            assert not (case / "datin/grid.flatplate.h5").exists()
            assert not (case / "datin/inlet.prof").exists()
    assert before == {str(p.relative_to(seed / "outdat/new")): p.read_bytes()
        for p in (seed / "outdat/new").rglob("*") if p.is_file()}
    total = sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file())
    assert total < 64*1024**2
    with args.executable.open("rb") as stream:
        executable_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    record = dict(backend=backend, source_np=source_ranks, target_np=target_ranks, axis=target_axis,
        source_axis=source_axis, target_axis=target_axis,
        exact=exact, memcheck=memcheck, filter_workspace=args.filter_workspace, state_max_abs=max(errors.values()),
        geometry_max_abs=max(geometry_errors.values()), target_halos=halos, geometry_halos=geometry_halos, readers=readers,
        executable_sha256=executable_sha256, test_root_bytes=total)
    (tmp_path / "curve_repartition.json").write_text(json.dumps(record, indent=2)+"\n")
    record_property("result", record)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("source_ranks,target_ranks", [(1, 2), (2, 1)])
def test_curve_repartition(backend, source_ranks, target_ranks, tmp_path, record_property):
    check_curve_repartition(backend, source_ranks, target_ranks, tmp_path, record_property)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("axis", ["z", "x"])
def test_curve_repartition_keeps_exact_restore(backend, axis, tmp_path, record_property):
    check_curve_repartition(backend, 2, 2, tmp_path, record_property, exact=True,
                            source_axis=axis, target_axis=axis)


@pytest.mark.parametrize("axis", ["z", "x"])
def test_curve_repartition_memcheck(axis, tmp_path, record_property):
    check_curve_repartition("gpu", 1, 2, tmp_path, record_property, memcheck=True,
                            source_axis=axis, target_axis=axis)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("source_ranks,source_axis,target_ranks,target_axis", [
    (1, "x", 2, "x"), (2, "x", 1, "x"), (2, "x", 2, "z"), (2, "z", 2, "x")])
def test_curve_x_repartition(backend, source_ranks, source_axis, target_ranks, target_axis,
                             tmp_path, record_property):
    check_curve_repartition(backend, source_ranks, target_ranks, tmp_path, record_property,
                            source_axis=source_axis, target_axis=target_axis)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("source_ranks,source_axis,target_ranks,target_axis,rejection", [
    (1, "z", 2, "y", "exact restore rank count"),
    (2, "y", 1, "z", "repartition nonperiodic direction must stay unpartitioned")])
def test_curve_nonperiodic_repartition_is_rejected(backend, source_ranks, source_axis,
        target_ranks, target_axis, rejection, tmp_path):
    args = curve_arguments(tmp_path, backend, source_axis)
    seed, _ = run_case(args, ROOT, backend, source_ranks, "seed", 5, buffer_bytes=4096)
    args.axis = target_axis
    run_case(args, ROOT, backend, target_ranks, "rejected", 12, buffer_bytes=4096,
        restore=seed / "outdat/new/checkpoints/step000000000005", reject=rejection)


def air5_arguments(output, backend, axis="z"):
    args = curve_arguments(output, backend, axis)
    if backend == "cpu":
        args.executable = Path(os.environ.get("ASTR_OUTPUT_AIR5_CPU_EXE",
            ROOT / "build_cpu_probe/bin/astr")).resolve()
    args.case = "hbl"
    args.reconstruction = 5
    args.mean_statistics = False
    args.sample_interval = 2
    args.top_mode = "characteristic"
    return args


def air5_scales():
    from air5_radau_reference import Air5RadauReference
    model = Air5RadauReference(ROOT / "chemMech/air5_kimjo12.json")
    ys = np.array([.767, .233, 0., 0., 0.])
    gas, cv = float(ys @ model.gas_constant), float(ys @ model.cv_tr)
    rho, temperature = .05, 3000.
    sound = float(np.sqrt((1 + gas/cv) * gas * temperature))
    pressure = rho * gas * temperature
    conserved = np.array([rho, *([rho*sound]*3), rho*sound**2,
                          *([rho]*5), rho*sound**2])
    scales = np.r_[conserved, conserved, rho, [sound]*3, pressure,
                   temperature, temperature, [1.]*5]
    assert scales.shape == (34,) and np.isfinite(scales).all() and np.all(scales > 0)
    return scales, dict(rho=rho, sound=sound, pressure=pressure, temperature=temperature)


def frozen_geometry_scales(path):
    with h5py.File(path) as geometry:
        return np.array([np.max(np.abs(geometry[f"q{m:04d}"][:])) for m in range(1, 14)])


def compare_air5_scaled(reference, actual, scales, same_layout=False):
    errors, normalized = {}, {}
    with h5py.File(reference) as a, h5py.File(actual) as b:
        assert np.array_equal(a["identity"][3:7], b["identity"][3:7])
        assert np.array_equal(a["identity"][12:], b["identity"][12:])
        if same_layout:
            assert a["identity"][:].tobytes() == b["identity"][:].tobytes()
            assert a["partitions"][:].tobytes() == b["partitions"][:].tobytes()
        for m, scale in enumerate(scales, 1):
            name = f"q{m:04d}"
            left, right = a[name][:], b[name][:]
            assert left.shape == right.shape and left.dtype == right.dtype
            assert np.isfinite(left).all() and np.isfinite(right).all()
            error = float(np.max(np.abs(left-right)))
            errors[name] = error
            normalized[name] = error/float(scale) if scale else error
            assert (normalized[name] <= ATOL if scale else error == 0), (name, error, scale)
        if len(scales) == 34:
            assert np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant
            for m in range(1, 12):
                left = a[f"q{m:04d}"][:].astype(np.longdouble)-a[f"q{m+11:04d}"][:]
                right = b[f"q{m:04d}"][:].astype(np.longdouble)-b[f"q{m+11:04d}"][:]
                error = float(np.max(np.abs(left-right)))
                name = f"extended_q_minus_carry_{m:02d}"
                errors[name], normalized[name] = error, error/float(scales[m-1])
                assert normalized[name] <= ATOL, (name, error)
        for state in (a, b):
            assert np.isfinite(state["rank_extras"][:]).all()
    return dict(max_normalized=max(normalized.values()), dimensional=errors, normalized=normalized)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_air5_partition_independence(backend, tmp_path, record_property):
    from run_output_air5_restart_validation import launch, state_sanity
    scales, physical = air5_scales()
    args = air5_arguments(tmp_path, backend)
    one, _ = launch(args, backend, 1, "one", 1, buffer_bytes=4096)
    geometry = "outdat/new/resources/geometry.h5"
    geometry_scales = frozen_geometry_scales(one / geometry)
    two, _ = launch(args, backend, 2, "two", 1, buffer_bytes=4096)
    checkpoint = "outdat/new/checkpoints/step000000000001/state.h5"
    results = compare_air5_scaled(one / checkpoint, two / checkpoint, scales)
    geometry_results = compare_air5_scaled(one / geometry, two / geometry, geometry_scales)
    for case in (one, two):
        state_sanity(case / checkpoint)
    total = sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file())
    assert total < 64*1024**2
    record_property("state", results)
    record_property("geometry", geometry_results)
    record_property("physical_scales", physical)
    record_property("test_root_bytes", total)


def check_air5_repartition(backend, source_ranks, target_ranks, tmp_path, record_property,
                           exact=False, memcheck=False):
    from run_output_air5_restart_validation import launch, state_sanity, compare_air5_resources
    scales, physical_scales = air5_scales()
    args = air5_arguments(tmp_path, backend)
    reference, _ = launch(args, backend, target_ranks, "continuous", 12, buffer_bytes=4096)
    geometry = "outdat/new/resources/geometry.h5"
    geometry_scales = frozen_geometry_scales(reference / geometry)
    seed, _ = launch(args, backend, source_ranks, "seed", 5, buffer_bytes=4096)
    source = seed / "outdat/new/checkpoints/step000000000005"
    original = {str(p.relative_to(seed / "outdat/new")): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (seed / "outdat/new").rglob("*") if p.is_file()}
    with h5py.File(source / "state.h5") as state:
        assert state["metadata"][:].tolist() == [1, 1] and int(state["identity"][12]) == 7
        assert sum(np.count_nonzero(state[f"q{m:04d}"][:]) for m in range(12, 23)) > 0
    pure, _ = launch(args, backend, target_ranks, "pure_restore", 5, restore=source,
                     buffer_bytes=4096, restore_probe=True)
    assert clocks(pure) == [] and not list((pure / "outdat/new/checkpoints").glob("*/COMPLETE"))
    probe = pure / "outdat/restore_probe.h5"
    with h5py.File(source / "state.h5") as a, h5py.File(probe) as b:
        for m in range(1, 35):
            name = f"q{m:04d}"
            assert a[name][:].tobytes() == b[name][:].tobytes(), ("pure_restore", name)
        assert a["identity"][8:13].tobytes() == b["identity"][8:13].tobytes()
    restored, _ = launch(args, backend, target_ranks, "restored", 12, restore=source,
        buffer_bytes=4096, restore_probe=True, memcheck=memcheck)
    with h5py.File(probe) as a, h5py.File(restored / "outdat/restore_probe.h5") as b:
        for m in range(1, 35):
            name = f"q{m:04d}"
            assert a[name][:].tobytes() == b[name][:].tobytes(), ("advancing_restore", name)
    assert ("ASTR_OUTPUT_REPARTITION air5hbl" in (restored / "run.log").read_text()) != exact
    assert clocks(seed)+clocks(restored) == clocks(reference)
    assert [entry[0] for entry in clocks(reference)] == list(range(12))
    errors = compare_air5_scaled(reference / FINAL / "state.h5", restored / FINAL / "state.h5", scales, True)
    halos = compare_curve_target_halos(reference / FINAL / "state.h5", restored / FINAL / "state.h5", scales)
    geometry_errors = compare_air5_scaled(reference / geometry, restored / geometry, geometry_scales, True)
    geometry_halos = compare_curve_target_halos(reference / geometry, restored / geometry, geometry_scales)
    if exact:
        compare_fields(reference / FINAL / "state.h5", restored / FINAL / "state.h5")
        compare_fields(reference / geometry, restored / geometry)
    for filename in ("control.bin", "air5_config.bin", "archives.bin", "insitu_control.bin", "air5_conservation.bin"):
        read = archive_schedule_payload if filename == "archives.bin" else Path.read_bytes
        if filename == "air5_conservation.bin" and backend == "cpu":
            continue
        assert read(reference / FINAL / filename) == read(restored / FINAL / filename), filename
    resources = compare_air5_resources(reference, restored, False)
    controlled = []
    sanity = []
    for case, final in ((reference, FINAL), (seed, "outdat/new/checkpoints/step000000000005"),
                        (pure, "outdat/restore_probe.h5"), (restored, FINAL)):
        path = case / final
        state_file = path if case == pure else path / "state.h5"
        controlled.extend(controlled_checkpoint_buffers(state_file, 0, backend))
        with h5py.File(state_file) as state:
            for row in state["partitions"][:].reshape(-1, 8):
                cells, halo = row[3:6], int(row[6])
                packed = int(np.prod(cells+2*halo+1))*34*8
                carry = int(np.prod(cells+1))*11*8 if backend == "gpu" else 0
                face = max(int((cells[a]+1)*(cells[b]+1)) for a, b in ((0, 1), (0, 2), (1, 2)))
                controlled.append(packed+carry+6*face*halo*34*8)
            q = np.stack([state[f"q{m:04d}"][:] for m in range(1, 12)], axis=-1).astype(np.longdouble)
            carry = np.stack([state[f"q{m:04d}"][:] for m in range(12, 23)], axis=-1).astype(np.longdouble)
            represented = q-carry
            assert np.all(represented[..., 0] > 0) and np.all(represented[..., 5:10] >= 0)
            closure = np.abs(represented[..., 5:10].sum(axis=-1)-represented[..., 0])/represented[..., 0]
            assert np.max(closure) <= 128*np.finfo(float).eps
        sanity.append(state_sanity(state_file))
        if case != pure:
            assert not (path / "statistics.h5").exists()
        if case in (pure, restored):
            assert not any((case / f"datin/air5_hbl_{name}.dat").exists()
                           for name in ("domain", "profile", "initial_field"))
    assert max(controlled) < 64*1024**2
    assert original == {str(p.relative_to(seed / "outdat/new")): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (seed / "outdat/new").rglob("*") if p.is_file()}
    total = sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file())
    assert total < 64*1024**2
    record = dict(backend=backend, source_np=source_ranks, target_np=target_ranks, axis="z",
        exact=exact, memcheck=memcheck, filter_workspace=args.filter_workspace, state=errors,
        geometry=geometry_errors, target_halos=halos, geometry_halos=geometry_halos,
        pure_restore="bitwise_all_34_physical_components", physical_scales=physical_scales,
        geometry_scales=geometry_scales.tolist(), resources=resources, sanity=sanity,
        controlled_host_max_bytes=max(controlled), test_root_bytes=total,
        executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest())
    (tmp_path / "air5_repartition.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    record_property("result", record)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("source_ranks,target_ranks", [(1, 2), (2, 1)])
def test_air5_repartition(backend, source_ranks, target_ranks, tmp_path, record_property):
    check_air5_repartition(backend, source_ranks, target_ranks, tmp_path, record_property)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_air5_repartition_keeps_exact_restore(backend, tmp_path, record_property):
    check_air5_repartition(backend, 2, 2, tmp_path, record_property, exact=True)


def test_air5_repartition_memcheck(tmp_path, record_property):
    check_air5_repartition("gpu", 1, 2, tmp_path, record_property, memcheck=True)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("source_ranks,target_ranks", [(1, 2), (2, 1)])
def test_air5_nonperiodic_repartition_is_rejected(backend, axis, source_ranks, target_ranks, tmp_path):
    from run_output_air5_restart_validation import launch
    args = air5_arguments(tmp_path, backend, axis if source_ranks == 2 else "z")
    seed, _ = launch(args, backend, source_ranks, "seed", 5, buffer_bytes=4096)
    args.axis = axis if target_ranks == 2 else "z"
    rejection = "exact restore rank count" if target_ranks == 2 else "repartition nonperiodic direction must stay unpartitioned"
    launch(args, backend, target_ranks, "rejected", 12, buffer_bytes=4096,
           restore=seed / "outdat/new/checkpoints/step000000000005", reject=rejection)
