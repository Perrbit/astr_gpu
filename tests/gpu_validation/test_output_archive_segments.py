"""Real bounded continuation into a new segment of an existing output root."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from run_output_archive_validation import check_frame, check_series, check_series_reader, field_names, groups, run_case
from run_output_restart_validation import archive_schedule_payload, compare_fields

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/output"))
from combine_series import combine_series
from repair_series import fingerprint, segment_record
EXE = Path(os.environ.get("ASTR_OUTPUT_RUNTIME_EXE", ROOT / "build_gpu_probe/bin/astr")).resolve()
MPIEXEC = Path(os.environ.get("ASTR_OUTPUT_MPIEXEC",
    "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec")).resolve()


def fingerprints(root):
    result = {}
    for p in root.rglob("*"):
        if p.is_file():
            with p.open("rb") as stream:
                result[str(p.relative_to(root))] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def check_parent_reader(head, output, extra_names=()):
    report = combine_series(head, output=output)
    for row in (output / "lineage.frames").read_text().splitlines()[1:]:
        step, time, source = row.split(maxsplit=2)
        frame = output / source
        reference = {"time": np.array(float(time))}
        with h5py.File(frame / "data.h5") as data, h5py.File(frame.parent.parent / "resources/data.h5") as geometry:
            volume = "metadata" in data
            labels = [""] if volume else sorted(name for name in data if name != "frame_identity")
            if not volume:
                reference["labels"] = np.array(labels)
            for label in labels:
                leaf, coords = (data[label], geometry[label]) if label else (data, geometry)
                names = (*field_names(int(leaf["metadata"][-1])), *extra_names)
                reference["field_names"] = np.array(names)
                prefix = label + "_" if label else ""
                reference[prefix + "coordinates"] = coords["coordinates"][:].reshape(-1, 3)
                for name in names:
                    reference[prefix + name] = leaf[name][:].ravel()
                reference[prefix + "velocity"] = leaf["velocity"][:].reshape(-1, 3)
        assert sum(value.nbytes for value in reference.values()) < 2 * 1024**2
        directory = output / f"step{int(step):012d}"
        directory.mkdir()
        np.savez(directory / "reader_reference.npz", **reference)
    result = subprocess.run(["/usr/bin/pvpython", "--no-mpi",
        str(Path(__file__).with_name("run_checkpoint_export_validation.py")),
        "--native-lineage-check", str(output)], stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, timeout=60)
    (output / "reader.log").write_text(result.stdout)
    assert result.returncode == 0 and "Error" not in result.stdout and "Traceback" not in result.stdout, result.stdout
    reader = json.loads(result.stdout.strip().splitlines()[-1])
    assert reader["frames"] == report["frames"]
    return report


def make_reference(output, backend, case="tgv", axis="x"):
    args = SimpleNamespace(output=output,
        executable=EXE, mpiexec=MPIEXEC, case=case, mode="steps", restart_step=5,
        initial_dimension=0, legacy_statistics=case != "tgv", statistics=case == "tgv",
        initial_restart=False, filter_workspace="scalar", force="feedback", axis=axis, no_samples=True)
    continuous, _ = run_case(args, ROOT, backend, 2, "continuous", 12,
                             buffer_bytes=4096, archive_groups=groups("steps"))
    seed, _ = run_case(args, ROOT, backend, 2, "seed", 5,
                       buffer_bytes=4096, archive_groups=groups("steps"))
    assert sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) < 64 * 1024**2
    return args, backend, continuous, seed


@pytest.fixture(scope="module", params=["cpu", "gpu"])
def reference(request, tmp_path_factory):
    return make_reference(tmp_path_factory.mktemp("segment_" + request.param), request.param)


@pytest.fixture(scope="module", params=[("cpu", "curve", "y"), ("gpu", "air5hbl", "z")])
def boundary_reference(request, tmp_path_factory):
    backend, case, axis = request.param
    return make_reference(tmp_path_factory.mktemp("segment_" + case), backend, case, axis)


def assert_same_root_continuation(reference, tmp_path):
    args, backend, continuous, seed = reference
    args = SimpleNamespace(**vars(args))
    args.output = tmp_path
    seed_root = seed / "outdat/new"
    source = seed_root / "checkpoints/step000000000005"
    before = fingerprints(seed_root)
    restored, _ = run_case(args, ROOT, backend, 2, "restored", 12, restore=source,
        reuse_root=seed_root, buffer_bytes=4096, checkpoint_keep=1, archive_groups=groups("steps"))
    actual = restored / "outdat/new"
    final = "checkpoints/step000000000012"
    for name in ("state.h5", "statistics.h5"):
        compare_fields(actual / final / name, continuous / "outdat/new" / final / name)
    assert archive_schedule_payload(actual / final / "archives.bin") == archive_schedule_payload(
        continuous / "outdat/new" / final / "archives.bin")
    ids, sizes, crcs = np.frombuffer((actual / final / "archives.bin").read_bytes()[-48:], dtype="<i8").reshape(3, 2)
    assert ids.tolist() == [1, 1]
    assert before == fingerprints(seed_root)
    after = fingerprints(actual)
    for product, steps in (("fields", [6, 8, 10, 12]), ("slices", [6, 9, 12])):
        for name, checksum in before.items():
            if name.startswith(product + "/") or name.startswith("resources/") or name.startswith("checkpoints/step"):
                assert after[name] == checksum
        segment = actual / product / "segment00000001"
        rows = (segment / "series.frames").read_text().splitlines()
        assert [int(row.split()[0]) for row in rows[1:]] == steps
        assert (segment / "SEGMENT").read_text().splitlines()[1].split()[0] == "5"
        record = segment_record(segment, require_parent=True)
        assert (segment / record["parent"]).resolve() == (actual / product / "segment00000000").resolve()
        assert record["parent_fingerprint"] == fingerprint(actual / product / "segment00000000/SEGMENT")
        position = 0 if product == "fields" else 1
        assert (int(sizes[position]), int(crcs[position]) & ((1 << 64) - 1)) == record["fingerprint"]
        assert not (segment / "step000000000005").exists()
        check_frame(segment / "step000000000012", product, actual / final,
                    actual / "resources/geometry.h5", backend)
        assert len(check_series(restored, product, "segment00000001")) == len(steps)
        assert check_series_reader(restored, product, "segment00000001")["frames"] == len(steps)
        before_combined = fingerprints(actual)
        combined = check_parent_reader(segment, restored / (product + "_combined"))
        assert combined["segments"] == 2 and combined["excluded_parent_frames"] == 0
        assert combined["frames"] == (8 if product == "fields" else 6)
        assert before_combined == fingerprints(actual)
    assert (actual / "checkpoints/LATEST").read_text() == "step000000000012\n"
    assert (actual / "checkpoints/step000000000005/COMPLETE").is_file()
    assert not (actual / "checkpoints/step000000000010").exists()
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 64 * 1024**2


def test_same_root_new_segment_exact_continuation(reference, tmp_path):
    assert_same_root_continuation(reference, tmp_path)


def test_native_historical_parent_cutoff_and_external_root(reference, tmp_path):
    args, backend, continuous, _ = reference
    args = SimpleNamespace(**vars(args))
    args.output = tmp_path
    parent = continuous / "outdat/new"
    before = fingerprints(parent)
    restored, _ = run_case(args, ROOT, backend, 2, "historical", 12,
        restore=parent / "checkpoints/step000000000010", buffer_bytes=4096,
        archive_groups=groups("steps"))
    final = "outdat/new/checkpoints/step000000000012"
    for name in ("state.h5", "statistics.h5"):
        compare_fields(restored / final / name, continuous / final / name)
    for product in ("fields", "slices"):
        head = restored / "outdat/new" / product / "segment00000000"
        report = check_parent_reader(head, restored / (product + "_combined"))
        assert report["segments"] == 2 and report["excluded_parent_frames"] == 1
        rows = (restored / (product + "_combined/lineage.frames")).read_text().splitlines()[1:]
        assert int(rows[-1].split()[0]) == 12
        assert (restored / (product + "_combined") / rows[-1].split(maxsplit=2)[2]).resolve().parent == head.resolve()
    assert fingerprints(parent) == before
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 64 * 1024**2


def test_native_three_segment_chain(reference, tmp_path):
    args, backend, continuous, seed = reference
    args = SimpleNamespace(**vars(args))
    args.output = tmp_path
    source_root = seed / "outdat/new"
    middle, _ = run_case(args, ROOT, backend, 2, "middle", 10,
        restore=source_root / "checkpoints/step000000000005", reuse_root=source_root,
        buffer_bytes=4096, checkpoint_keep=1, archive_groups=groups("steps"))
    middle_root = middle / "outdat/new"
    before = fingerprints(middle_root)
    head_case, _ = run_case(args, ROOT, backend, 2, "head", 12,
        restore=middle_root / "checkpoints/step000000000010", buffer_bytes=4096,
        archive_groups=groups("steps"))
    for name in ("state.h5", "statistics.h5"):
        relative = "outdat/new/checkpoints/step000000000012/" + name
        compare_fields(head_case / relative, continuous / relative)
    for product in ("fields", "slices"):
        head = head_case / "outdat/new" / product / "segment00000000"
        report = check_parent_reader(head, head_case / (product + "_combined"))
        assert report["segments"] == 3 and report["excluded_parent_frames"] == 0
        assert report["frames"] == (8 if product == "fields" else 7)
    assert fingerprints(middle_root) == before
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 64 * 1024**2


def test_restore_rejects_tampered_parent_identity(reference, tmp_path):
    args, backend, _, seed = reference
    args = SimpleNamespace(**vars(args))
    args.output = tmp_path
    source_root = tmp_path / "tampered_source"
    shutil.copytree(seed / "outdat/new", source_root)
    with (source_root / "fields/segment00000000/SEGMENT").open("a") as stream:
        stream.write("changed provenance\n")
    before = fingerprints(source_root)
    rejected, _ = run_case(args, ROOT, backend, 2, "rejected_parent", 12,
        restore=source_root / "checkpoints/step000000000005", buffer_bytes=4096,
        archive_groups=groups("steps"), reject="cannot record segment input/parent identity")
    assert not list((rejected / "outdat/new/fields/segment00000000").glob("step*"))
    assert before == fingerprints(source_root)
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 64 * 1024**2


def test_boundary_root_new_segment_exact_continuation(boundary_reference, tmp_path):
    assert_same_root_continuation(boundary_reference, tmp_path)


def test_changed_planes_require_new_geometry_root(reference, tmp_path):
    args, backend, _, seed = reference
    args = SimpleNamespace(**vars(args))
    args.output = tmp_path
    seed_root = seed / "outdat/new"
    before = fingerprints(seed_root)
    failed, _ = run_case(args, ROOT, backend, 2, "mismatched_planes", 12,
        restore=seed_root / "checkpoints/step000000000005", reuse_root=seed_root, buffer_bytes=4096,
        override=True, archive_groups=groups("steps").replace("i_indices=8", "i_indices=7"),
        reject="field output: open or create field group")
    actual = failed / "outdat/new"
    after = fingerprints(actual)
    assert all(after[name] == checksum for name, checksum in before.items())
    assert before == fingerprints(seed_root)
    assert not (actual / "slices/segment00000001").exists()
    assert not (actual / "checkpoints/step000000000012").exists()
    assert not list((actual / "fields/segment00000001").glob("step*"))
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 64 * 1024**2
