"""Approved OR5 real TGV derivatives, exact continuation and bounded readback."""
import hashlib
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from run_output_archive_validation import check_frame, check_series, check_series_reader, frames, groups
from run_output_restart_validation import archive_schedule_payload, compare_fields, run_case
from test_output_archive_segments import check_parent_reader
from test_output_fields import DERIVED_NAMES

ROOT = Path(__file__).resolve().parents[2]
EXE = Path(os.environ.get("ASTR_OUTPUT_RUNTIME_EXE", ROOT / "build_gpu_probe/bin/astr")).resolve()
CPU_EXE = Path(os.environ.get("ASTR_OUTPUT_CPU_EXE", ROOT / "build_release_restart_cpu/bin/astr")).resolve()
MPIEXEC = Path(os.environ.get("ASTR_OUTPUT_MPIEXEC",
    "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec")).resolve()
FINAL = "outdat/new/checkpoints/step000000000012"
SELECTIONS = {"all": list(range(14)), "gradient": list(range(9)), "curl": list(range(11, 14)), "q": [9, 10]}


def arguments(output, backend, axis):
    return SimpleNamespace(output=output, executable=EXE if backend == "gpu" else CPU_EXE,
        mpiexec=MPIEXEC, case="tgv", mode="steps", restart_step=5, initial_dimension=0,
        legacy_statistics=False, statistics=True, initial_restart=False, filter_workspace="scalar",
        force="feedback", axis=axis, no_samples=True)


def archive(mode="all", volume=True):
    options = {"all": "velocity_gradient=t,vorticity=t,qcriterion=t",
               "gradient": "velocity_gradient=t", "curl": "vorticity=t", "q": "qcriterion=t"}[mode]
    return groups("steps", volume=volume).replace("interval_steps=2", "interval_steps=5").replace(
        "interval_steps=3", "interval_steps=5").replace("initial_frame=.true.", options+", initial_frame=.true.")


def discrete_reference(checkpoint, geometry):
    """Independent periodic stencil on the saved complete-step physical velocity."""
    with h5py.File(checkpoint / "state.h5") as state, h5py.File(geometry) as grid:
        velocity = np.stack([state[f"q{c:04d}"][:16, :16, :16] for c in (7, 8, 9)], axis=-1)
        gradient = np.empty((16, 16, 16, 3, 3))
        for d, array_axis in enumerate((2, 1, 0)):
            derivative = sum(weight*(np.roll(velocity, -offset, axis=array_axis)-
                np.roll(velocity, offset, axis=array_axis)) for offset, weight in ((1, 3/4), (2, -3/20), (3, 1/60)))
            for c in range(3):
                metric = grid[f"q{5+c+3*d:04d}"][:]
                np.testing.assert_allclose(metric, 16/(2*np.pi) if c == d else 0, rtol=0, atol=2e-13)
            gradient[..., :, d] = derivative*(16/(2*np.pi))
    result = [gradient[..., c, d] for d in range(3) for c in range(3)]
    result += [-0.5*np.einsum("...ij,...ji->...", gradient, gradient),
        np.trace(gradient, axis1=-2, axis2=-1), gradient[..., 2, 1]-gradient[..., 1, 2],
        gradient[..., 0, 2]-gradient[..., 2, 0], gradient[..., 1, 0]-gradient[..., 0, 1]]
    return [np.pad(value, [(0, 1)]*3, mode="wrap") for value in result]


def check_derivatives(case, step, mode, backend, volume=True):
    checkpoint = case / f"outdat/new/checkpoints/step{step:012d}"
    geometry = case / "outdat/new/resources/geometry.h5"
    reference = discrete_reference(checkpoint, geometry)
    indices = SELECTIONS[mode]
    names = [DERIVED_NAMES[n] for n in indices]
    maxima = {}
    for product in (["fields", "slices"] if volume else ["slices"]):
        frame = frames(case, product)[step]
        check_frame(frame, product, checkpoint, geometry, backend, extra_names=names)
        with h5py.File(frame / "data.h5") as data:
            assert list(map(int, data.attrs["derived_layout"].decode().split())) == [n+1 for n in indices]+[0]*(14-len(indices))
            leaves = [(data, np.s_[:])] if product == "fields" else [
                (data["i000000000008"], np.s_[:, :, 8]),
                (data["j000000000000"], np.s_[:, 0, :]),
                (data["k000000000016"], np.s_[16, :, :])]
            for leaf, selection in leaves:
                for n in indices:
                    values = leaf[DERIVED_NAMES[n]][:]
                    assert np.all(np.isfinite(values)) and values.dtype == np.dtype("<f8")
                    error = float(np.max(np.abs(values-reference[n][selection])))
                    assert error <= 2e-10, (backend, product, DERIVED_NAMES[n], error)
                    assert leaf[DERIVED_NAMES[n]].attrs["quantity_units"].decode() == "dimensionless"
                    assert leaf[DERIVED_NAMES[n]].attrs["coordinate_space"].decode() == "physical"
                    assert leaf[DERIVED_NAMES[n]].attrs["time_dimension_exponent"].decode() == ("-2" if n == 9 else "-1")
                    maxima[DERIVED_NAMES[n]] = max(maxima.get(DERIVED_NAMES[n], 0), error)
        declaration = (frame / "FRAME").read_text().splitlines()
        assert declaration[0] == "ASTR_DERIVED_FRAME_1" and len(declaration) == 5
        assert list(map(int, declaration[4].split())) == [n+1 for n in indices]+[0]*(14-len(indices))
    return maxima


def accounting(case, backend):
    rows = re.findall(r"ASTR_OUTPUT_ARCHIVE rank=(\d+) product=(fields|slices) field_bytes=(\d+) download_bytes=(\d+)"
        r"host_peak_bytes=(\d+) device_peak_bytes=(\d+)", (case / "run.log").read_text())
    assert rows
    downloads = {}
    for rank, product, size, copied, host, device in rows:
        size, copied, host, device = map(int, (size, copied, host, device))
        assert 0 < host <= 64*1024**2 and 0 <= device <= 64*1024**2
        assert copied == (size if backend == "gpu" else 0)
        downloads[product] = downloads.get(product, 0)+copied
    return dict(downloads=downloads, max_host_bytes=max(int(row[4]) for row in rows),
                max_device_bytes=max(int(row[5]) for row in rows))


def directory_budget(path):
    size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    assert size < 64*1024**2, size
    return size


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("ranks,axis", [(1, "x"), (2, "x"), (2, "y"), (2, "z")])
def test_real_derived_exact_restart(backend, ranks, axis, tmp_path, record_property):
    args = arguments(tmp_path, backend, axis)
    continuous, _ = run_case(args, ROOT, backend, ranks, "continuous", 12,
        archive_groups=archive(), buffer_bytes=4096, checkpoint_interval=99)
    errors = check_derivatives(continuous, 12, "all", backend)
    seed, _ = run_case(args, ROOT, backend, ranks, "seed", 5,
        archive_groups=archive(), buffer_bytes=4096, checkpoint_interval=99)
    check_derivatives(seed, 5, "all", backend)
    source = seed / "outdat/new/checkpoints/step000000000005"
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    restored, _ = run_case(args, ROOT, backend, ranks, "restored", 12, restore=source,
        archive_groups=archive(), buffer_bytes=4096, checkpoint_interval=99)
    check_derivatives(restored, 12, "all", backend)
    baseline, _ = run_case(args, ROOT, backend, ranks, "off", 12, buffer_bytes=4096, checkpoint_interval=99)
    for other in (restored, baseline):
        for name in ("state.h5", "statistics.h5"):
            compare_fields(continuous / FINAL / name, other / FINAL / name)
    for name in ("control.bin", "insitu_control.bin"):
        assert (continuous / FINAL / name).read_bytes() == (restored / FINAL / name).read_bytes()
    assert archive_schedule_payload(continuous / FINAL / "archives.bin") == archive_schedule_payload(restored / FINAL / "archives.bin")
    for product in ("fields", "slices"):
        assert sorted(frames(continuous, product)) == [0, 5, 10, 12]
        assert sorted(frames(seed, product)) == [0, 5]
        assert sorted(frames(restored, product)) == [10, 12]
        check_series(continuous, product)
        check_series(restored, product)
        for step, frame in frames(restored, product).items():
            compare_fields(frame / "data.h5", frames(continuous, product)[step] / "data.h5")
        assert check_series_reader(restored, product, extra_names=DERIVED_NAMES)["frames"] == 2
    slice_only, _ = run_case(args, ROOT, backend, ranks, "slice_only", 12, restore=source, override=True,
        archive_groups=archive(volume=False), buffer_bytes=4096, checkpoint_interval=99)
    check_derivatives(slice_only, 12, "all", backend, volume=False)
    assert not frames(slice_only, "fields") and sorted(frames(slice_only, "slices")) == [10, 12]
    for name in ("state.h5", "statistics.h5"):
        compare_fields(continuous / FINAL / name, slice_only / FINAL / name)
    counts = {name: accounting(case, backend) for name, case in
              (("continuous", continuous), ("restored", restored), ("slice_only", slice_only))}
    if backend == "gpu":
        assert counts["slice_only"]["downloads"] == {"slices": 2*3*17**2*20*8}
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    with args.executable.open("rb") as stream:
        binary = hashlib.file_digest(stream, "sha256").hexdigest()
    report = dict(backend=backend, np=ranks, axis=axis, errors=errors, accounting=counts,
        exact_continuation=True, output_switch_unchanged=True, file_bytes=directory_budget(tmp_path), executable_sha256=binary)
    (tmp_path / "result.json").write_text(json.dumps(report, indent=2)+"\n")
    record_property("result", report)


@pytest.mark.parametrize("ranks,axis", [(1, "x"), (2, "x"), (2, "y"), (2, "z")])
def test_cpu_gpu_derived_fields(ranks, axis, tmp_path, record_property):
    cases = []
    for backend in ("cpu", "gpu"):
        args = arguments(tmp_path, backend, axis)
        args.statistics = False
        case, _ = run_case(args, ROOT, backend, ranks, "cross", 12,
            archive_groups=archive(), buffer_bytes=4096, checkpoint_interval=99)
        check_derivatives(case, 12, "all", backend)
        cases.append(case)
    maximum = 0.0
    for product in ("fields", "slices"):
        for step, frame in frames(cases[0], product).items():
            with h5py.File(frame / "data.h5") as cpu, h5py.File(frames(cases[1], product)[step] / "data.h5") as gpu:
                labels = [""] if product == "fields" else ["i000000000008", "j000000000000", "k000000000016"]
                for label in labels:
                    left, right = (cpu[label], gpu[label]) if label else (cpu, gpu)
                    for name in DERIVED_NAMES:
                        error = float(np.max(np.abs(left[name][:]-right[name][:])))
                        assert error <= 2e-10, (product, step, name, error)
                        maximum = max(maximum, error)
    record_property("max_absolute_difference", maximum)
    record_property("file_bytes", directory_budget(tmp_path))


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("mode", ["gradient", "curl", "q"])
def test_real_selected_derivatives(backend, mode, tmp_path):
    args = arguments(tmp_path, backend, "z")
    args.statistics = False
    case, _ = run_case(args, ROOT, backend, 2, mode, 1,
        archive_groups=archive(mode), buffer_bytes=4096, checkpoint_interval=99)
    check_derivatives(case, 1, mode, backend)
    for product in ("fields", "slices"):
        assert check_series_reader(case, product, extra_names=[DERIVED_NAMES[n] for n in SELECTIONS[mode]])["frames"] == 2
    directory_budget(tmp_path)


def test_derived_layout_override_and_parent(tmp_path):
    args = arguments(tmp_path, "gpu", "z")
    args.statistics = False
    seed, _ = run_case(args, ROOT, "gpu", 2, "seed", 5, archive_groups=archive(), buffer_bytes=4096, checkpoint_interval=99)
    source = seed / "outdat/new/checkpoints/step000000000005"
    run_case(args, ROOT, "gpu", 2, "reject_change", 12, restore=source, archive_groups=archive("q"),
        buffer_bytes=4096, checkpoint_interval=99, reject="product options changed without override")
    restored, _ = run_case(args, ROOT, "gpu", 2, "same", 12, restore=source, archive_groups=archive(),
        buffer_bytes=4096, checkpoint_interval=99)
    for product in ("fields", "slices"):
        segment = restored / "outdat/new" / product / "segment00000000"
        report = check_parent_reader(segment, tmp_path / (product+"_combined"), extra_names=DERIVED_NAMES)
        assert report["segments"] == 2 and report["frames"] == 4
    changed, _ = run_case(args, ROOT, "gpu", 2, "override", 12, restore=source, override=True,
        archive_groups=archive("q"), buffer_bytes=4096, checkpoint_interval=99)
    check_derivatives(changed, 12, "q", "gpu")
    for name in ("state.h5",):
        compare_fields(restored / FINAL / name, changed / FINAL / name)
    import sys
    sys.path.insert(0, str(ROOT / "scripts/output"))
    from combine_series import combine_series
    for product in ("fields", "slices"):
        with pytest.raises((ValueError, RuntimeError), match="layout|identity|invariant|metadata"):
            combine_series(changed / "outdat/new" / product / "segment00000000", output=tmp_path / (product+"_bad"))
    directory_budget(tmp_path)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_unvalidated_derived_wall_partition_rejected(backend, tmp_path):
    args = arguments(tmp_path, backend, "y")
    args.statistics = False
    args.case = "channel"
    run_case(args, ROOT, backend, 2, "rejected", 1, archive_groups=archive(),
        buffer_bytes=4096, reject="derived fields require registered 16-cubed explicit TGV/channel/CURVE/AIR5 NP<=2")
    directory_budget(tmp_path)
