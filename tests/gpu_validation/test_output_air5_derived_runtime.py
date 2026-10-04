"""Approved OR5 SI AIR5 private derivatives and complete-step continuation."""
import hashlib
import json

import h5py
import numpy as np
import pytest

from run_output_air5_restart_validation import launch, state_sanity
from run_output_archive_validation import check_frame, check_series_reader, frames
from run_output_restart_validation import archive_schedule_payload, compare_fields
from test_output_boundary_derived_runtime import directional
from test_output_boundary_lifecycle import arguments
from test_output_derived_runtime import FINAL, archive, accounting, directory_budget
from test_output_fields import DERIVED_NAMES

GRADIENT_SCALE = 110038.42893098433
SCALES = np.full(14, GRADIENT_SCALE)
SCALES[9] = 12108455841.599289


def reference(case, step):
    checkpoint = case / f"outdat/new/checkpoints/step{step:012d}"
    with h5py.File(checkpoint / "state.h5") as state, h5py.File(case / "outdat/new/resources/geometry.h5") as grid:
        velocity = np.stack([state[f"q{c:04d}"][:] for c in (24, 25, 26)], axis=-1)
        gradient = np.zeros((*velocity.shape, 3))
        for computational, axis in enumerate((2, 1, 0)):
            derivative = directional(velocity, axis, computational == 2)
            for physical in range(3):
                gradient[..., :, physical] += derivative*grid[f"q{5+computational+3*physical:04d}"][:][..., None]
    return [gradient[..., c, d] for d in range(3) for c in range(3)] + [
        -0.5*np.einsum("...ij,...ji->...", gradient, gradient),
        np.trace(gradient, axis1=-2, axis2=-1), gradient[..., 2, 1]-gradient[..., 1, 2],
        gradient[..., 0, 2]-gradient[..., 2, 0], gradient[..., 1, 0]-gradient[..., 0, 1]]


def check(case, step, backend):
    expected = reference(case, step)
    maxima = np.zeros(14)
    checkpoint = case / f"outdat/new/checkpoints/step{step:012d}"
    state_sanity(checkpoint / "state.h5")
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
                for index, name in enumerate(DERIVED_NAMES):
                    value = leaf[name][:]
                    assert value.dtype == np.dtype("<f8") and np.isfinite(value).all()
                    error = float(np.max(np.abs(value-expected[index][selection])))
                    assert error/SCALES[index] <= 2e-10, (product, name, error, error/SCALES[index])
                    assert leaf[name].attrs["quantity_units"].decode() == ("s^-2" if index == 9 else "s^-1")
                    assert leaf[name].attrs["coordinate_space"].decode() == "physical"
                    assert leaf[name].attrs["time_dimension_exponent"].decode() == ("-2" if index == 9 else "-1")
                    maxima[index] = max(maxima[index], error)
    return maxima


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("case_name,ranks", [("hbl", 1), ("hbl", 2), ("sbli", 1), ("sbli", 2)])
def test_air5_derivatives_exact_restart(case_name, ranks, backend, tmp_path, record_property):
    args = arguments(tmp_path, backend, case_name)
    options = dict(archive_groups=archive(), buffer_bytes=4096, interval=99,
                   conservation=backend == "gpu")
    full, _ = launch(args, backend, ranks, "full", 12, **options)
    errors = check(full, 12, backend)
    seed, _ = launch(args, backend, ranks, "seed", 5, **options)
    errors = np.maximum(errors, check(seed, 5, backend))
    source = seed / "outdat/new/checkpoints/step000000000005"
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    restored, _ = launch(args, backend, ranks, "restored", 12, restore=source, **options)
    errors = np.maximum(errors, check(restored, 12, backend))
    off, _ = launch(args, backend, ranks, "off", 12, buffer_bytes=4096, interval=99,
                    conservation=backend == "gpu")
    for other in (restored, off):
        for name in ("state.h5", "statistics.h5"):
            compare_fields(full / FINAL / name, other / FINAL / name)
        for name in ["air5_config.bin"] + (["air5_conservation.bin"] if backend == "gpu" else []):
            assert (full / FINAL / name).read_bytes() == (other / FINAL / name).read_bytes()
    assert (full / FINAL / "control.bin").read_bytes() == (restored / FINAL / "control.bin").read_bytes()
    assert archive_schedule_payload(full / FINAL / "archives.bin") == archive_schedule_payload(restored / FINAL / "archives.bin")
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    for product in ("fields", "slices"):
        assert sorted(frames(full, product)) == [0, 5, 10, 12]
        assert sorted(frames(restored, product)) == [10, 12]
        for step, frame in frames(restored, product).items():
            compare_fields(frame / "data.h5", frames(full, product)[step] / "data.h5")
    report = dict(case=case_name, backend=backend, np=ranks, axis=args.axis,
        reference_si_errors=dict(zip(DERIVED_NAMES, errors.tolist())),
        max_normalized_reference_error=float(np.max(errors/SCALES)),
        gradient_scale=GRADIENT_SCALE, q_scale=float(SCALES[9]),
        exact_continuation=True, exact_statistics=True, output_switch_unchanged=True,
        executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
        accounting=accounting(restored, backend), directory_bytes=directory_budget(tmp_path))
    (tmp_path / "result.json").write_text(json.dumps(report, indent=2)+"\n")
    directory_budget(tmp_path)
    record_property("result", json.dumps(report))


@pytest.mark.parametrize("case_name", ["hbl", "sbli"])
def test_air5_derived_cpu_gpu_readback(case_name, tmp_path, record_property):
    cases = []
    for backend in ("cpu", "gpu"):
        root = tmp_path / backend
        root.mkdir()
        args = arguments(root, backend, case_name)
        case, _ = launch(args, backend, 2, "compare", 12,
            archive_groups=archive(), buffer_bytes=4096, interval=99,
            conservation=backend == "gpu")
        check(case, 12, backend)
        cases.append(case)
    maxima = np.zeros(14)
    for product in ("fields", "slices"):
        for step, frame in frames(cases[0], product).items():
            with h5py.File(frame / "data.h5") as cpu, h5py.File(frames(cases[1], product)[step] / "data.h5") as gpu:
                names = [""] if product == "fields" else ["i000000000008", "j000000000000", "k000000000016"]
                for name in names:
                    left, right = (cpu[name], gpu[name]) if name else (cpu, gpu)
                    for index, field in enumerate(DERIVED_NAMES):
                        error = float(np.max(np.abs(left[field][:]-right[field][:])))
                        assert error/SCALES[index] <= 2e-10, (case_name, step, field, error)
                        maxima[index] = max(maxima[index], error)
        assert check_series_reader(cases[1], product, extra_names=DERIVED_NAMES)["frames"] == 4
    record_property("result", json.dumps(dict(cpu_gpu_si_errors=dict(zip(DERIVED_NAMES, maxima.tolist())),
        max_normalized_cpu_gpu_error=float(np.max(maxima/SCALES)), directory_bytes=directory_budget(tmp_path))))


@pytest.mark.parametrize("case_name", ["hbl", "sbli"])
def test_air5_derived_memcheck(case_name, tmp_path, record_property):
    args = arguments(tmp_path, "gpu", case_name)
    case, _ = launch(args, "gpu", 2, "memcheck", 2, archive_groups=archive(),
        buffer_bytes=4096, interval=99, conservation=True, memcheck=True)
    errors = check(case, 2, "gpu")
    record_property("max_normalized_reference_error", float(np.max(errors/SCALES)))
    record_property("directory_bytes", directory_budget(tmp_path))


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("case_name", ["hbl", "sbli"])
def test_air5_unregistered_derived_axis_is_rejected(case_name, backend, tmp_path):
    args = arguments(tmp_path, backend, case_name)
    args.axis = "x" if case_name == "hbl" else "z"
    case, _ = launch(args, backend, 2, "rejected", 2, archive_groups=archive(),
        buffer_bytes=4096, interval=99,
        reject="derived fields require registered 16-cubed explicit TGV/channel/CURVE/AIR5 NP<=2")
    assert not list((case / "outdat/new").rglob("COMPLETE"))
    directory_budget(tmp_path)
