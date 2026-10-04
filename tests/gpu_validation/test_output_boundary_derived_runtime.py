"""Complete-step private gradients at registered walls and static-grid boundaries."""
import hashlib
import json

import h5py
import numpy as np
import pytest

from run_output_archive_validation import check_frame, check_series, check_series_reader, frames
from run_output_restart_validation import archive_schedule_payload, compare_fields, run_case
from test_output_derived_runtime import ROOT, FINAL, archive, accounting, directory_budget
from test_output_fields import DERIVED_NAMES
from test_output_repartition_runtime import channel_arguments, curve_arguments, frozen_channel_initial


def arguments(output, backend, case_name, axis):
    if case_name == "channel":
        args = channel_arguments(output, backend, axis)
        args.legacy_statistics = False
        initial = frozen_channel_initial(output / "initial.h5")
    else:
        args = curve_arguments(output, backend, axis)
        args.case = case_name
        args.inflow_count = 12
        initial = None
    return args, dict(initial_resource=initial)


def directional(values, axis, periodic):
    result = np.empty_like(values)
    size = values.shape[axis]
    for index in range(size):
        row = [slice(None)]*values.ndim
        row[axis] = index
        if periodic or 3 <= index < size-3:
            weights = [(-3, -1/60), (-2, 3/20), (-1, -3/4), (1, 3/4), (2, -3/20), (3, 1/60)]
        elif index == 0:
            weights = [(0, -1.5), (1, 2.), (2, -0.5)]
        elif index == size-1:
            weights = [(-2, 0.5), (-1, -2.), (0, 1.5)]
        elif index in (1, size-2):
            weights = [(-1, -0.5), (1, 0.5)]
        else:
            weights = [(-2, 1/12), (-1, -2/3), (1, 2/3), (2, -1/12)]
        result[tuple(row)] = sum(weight*np.take(values, (index+offset) % (size-1) if periodic
            else index+offset, axis=axis) for offset, weight in weights)
    return result


def reference(case, step, case_name):
    checkpoint = case / f"outdat/new/checkpoints/step{step:012d}"
    with h5py.File(checkpoint / "state.h5") as state, h5py.File(case / "outdat/new/resources/geometry.h5") as grid:
        velocity = np.stack([state[f"q{c:04d}"][:] for c in (7, 8, 9)], axis=-1)
        gradient = np.zeros((*velocity.shape, 3))
        for computational, axis in enumerate((2, 1, 0)):
            derivative = directional(velocity, axis, computational == 2 or
                                     (case_name == "channel" and computational == 0))
            for physical in range(3):
                metric = grid[f"q{5+computational+3*physical:04d}"][:]
                gradient[..., :, physical] += derivative*metric[..., None]
    fields = [gradient[..., c, d] for d in range(3) for c in range(3)]
    fields += [-0.5*np.einsum("...ij,...ji->...", gradient, gradient),
               np.trace(gradient, axis1=-2, axis2=-1), gradient[..., 2, 1]-gradient[..., 1, 2],
               gradient[..., 0, 2]-gradient[..., 2, 0], gradient[..., 1, 0]-gradient[..., 0, 1]]
    return fields


def check(case, step, case_name, backend):
    expected = reference(case, step, case_name)
    checkpoint = case / f"outdat/new/checkpoints/step{step:012d}"
    maximum = 0.
    for product in ("fields", "slices"):
        frame = frames(case, product)[step]
        check_frame(frame, product, checkpoint, case / "outdat/new/resources/geometry.h5",
                    backend, extra_names=DERIVED_NAMES)
        with h5py.File(frame / "data.h5") as data:
            leaves = [(data, np.s_[:])] if product == "fields" else [
                (data["i000000000008"], np.s_[:, :, 8]),
                (data["j000000000000"], np.s_[:, 0, :]),
                (data["k000000000016"], np.s_[16, :, :])]
            for leaf, selection in leaves:
                for name, field in zip(DERIVED_NAMES, expected):
                    value = leaf[name][:]
                    assert np.all(np.isfinite(value))
                    error = float(np.max(np.abs(value-field[selection])))
                    assert error <= 2e-10, (case_name, backend, step, product, name, error)
                    maximum = max(maximum, error)
    return maximum


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("case_name,ranks,axis", [("channel", 1, "x"), ("channel", 2, "x"),
    ("channel", 2, "z"), ("curve", 1, "x"), ("curve", 2, "x"), ("curve", 2, "z"), ("dynamic", 2, "z")])
def test_boundary_derivatives_exact_restart(case_name, ranks, axis, backend, tmp_path, record_property):
    args, resources = arguments(tmp_path, backend, case_name, axis)
    options = dict(archive_groups=archive(), buffer_bytes=4096, checkpoint_interval=99, **resources)
    full, _ = run_case(args, ROOT, backend, ranks, "full", 12, **options)
    maximum = check(full, 12, case_name, backend)
    seed, _ = run_case(args, ROOT, backend, ranks, "seed", 5, **options)
    maximum = max(maximum, check(seed, 5, case_name, backend))
    source = seed / "outdat/new/checkpoints/step000000000005"
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    restored, _ = run_case(args, ROOT, backend, ranks, "restored", 12, restore=source, **options)
    maximum = max(maximum, check(restored, 12, case_name, backend))
    off, _ = run_case(args, ROOT, backend, ranks, "off", 12,
                       buffer_bytes=4096, checkpoint_interval=99, **resources)
    names = ["state.h5"] + (["inflow.h5"] if case_name == "dynamic" else [])
    for other in (restored, off):
        for name in names:
            compare_fields(full / FINAL / name, other / FINAL / name)
    assert (full / FINAL / "control.bin").read_bytes() == (restored / FINAL / "control.bin").read_bytes()
    assert archive_schedule_payload(full / FINAL / "archives.bin") == archive_schedule_payload(restored / FINAL / "archives.bin")
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    for product in ("fields", "slices"):
        assert sorted(frames(full, product)) == [0, 5, 10, 12]
        assert sorted(frames(restored, product)) == [10, 12]
        check_series(restored, product)
        for step, frame in frames(restored, product).items():
            compare_fields(frame / "data.h5", frames(full, product)[step] / "data.h5")
    result = dict(case=case_name, backend=backend, np=ranks, axis=axis, max_reference_error=maximum,
        executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
        accounting=accounting(restored, backend), exact_continuation=True, output_switch_unchanged=True,
        directory_bytes=directory_budget(tmp_path))
    (tmp_path / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    directory_budget(tmp_path)
    record_property("result", json.dumps(result))


@pytest.mark.parametrize("case_name,axis", [("channel", "x"), ("curve", "x"), ("dynamic", "z")])
def test_boundary_cpu_gpu_fields(case_name, axis, tmp_path, record_property):
    cases = []
    for backend in ("cpu", "gpu"):
        root = tmp_path / backend
        root.mkdir()
        args, resources = arguments(root, backend, case_name, axis)
        case, _ = run_case(args, ROOT, backend, 2, "compare", 12,
            archive_groups=archive(), buffer_bytes=4096, checkpoint_interval=99, **resources)
        check(case, 12, case_name, backend)
        cases.append(case)
    maximum = 0.
    for product in ("fields", "slices"):
        for step, frame in frames(cases[0], product).items():
            with h5py.File(frame / "data.h5") as cpu, h5py.File(frames(cases[1], product)[step] / "data.h5") as gpu:
                names = [""] if product == "fields" else ["i000000000008", "j000000000000", "k000000000016"]
                for name in names:
                    left, right = (cpu[name], gpu[name]) if name else (cpu, gpu)
                    for field in DERIVED_NAMES:
                        error = float(np.max(np.abs(left[field][:]-right[field][:])))
                        assert error <= 2e-10, (case_name, step, field, error)
                        maximum = max(maximum, error)
        assert check_series_reader(cases[1], product, extra_names=DERIVED_NAMES)["frames"] == 4
    record_property("max_cpu_gpu_difference", maximum)
    record_property("directory_bytes", directory_budget(tmp_path))


def test_boundary_private_derivatives_memcheck(tmp_path, record_property):
    args, resources = arguments(tmp_path, "gpu", "curve", "x")
    case, _ = run_case(args, ROOT, "gpu", 2, "memcheck", 2, archive_groups=archive(),
        buffer_bytes=4096, checkpoint_interval=99, memcheck=True, **resources)
    record_property("max_reference_error", check(case, 2, "curve", "gpu"))
    record_property("directory_bytes", directory_budget(tmp_path))
