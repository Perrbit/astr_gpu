"""Rebuild copied indexes from immutable, previously accepted native artifacts."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np
import pytest

from run_output_archive_validation import check_accounting, check_frame, check_series, check_series_reader, frames, groups
from run_output_restart_validation import archive_schedule_payload, compare_fields, controlled_checkpoint_buffers, run_case
from test_output_archive_segments import check_parent_reader, fingerprints
from test_output_derived_runtime import (FINAL, accounting, archive, arguments, directory_budget,
                                         discrete_reference)
from test_output_fields import DERIVED_NAMES
from test_output_repartition_runtime import channel_arguments


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/output"))
spec = importlib.util.spec_from_file_location("series_repair_runtime", ROOT / "scripts/output/repair_series.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def fingerprint_tree(root):
    return {str(path.relative_to(root)): (repair.fingerprint(path), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("reference,backend,expected_fields", [
    ("or6_archive_context_cpu_20261001", "cpu", 7),
    ("or6_archive_context_air5_20261001", "gpu", 7),
])
@pytest.mark.parametrize("product", ["fields", "slices"])
def test_native_archive_index_recovery(tmp_path, reference, backend, expected_fields, product):
    references = Path(os.environ.get("ASTR_OUTPUT_REPAIR_REFERENCE_ROOT", ROOT / "tests/gpu_validation/out"))
    report_path = references / reference / "summary.json"
    if not report_path.is_file():
        pytest.skip(f"immutable accepted reference required: {report_path}")
    reference_report = json.loads(report_path.read_text())
    assert reference_report["status"] == "passed"
    source = references / reference / f"{backend}_np2_continuous/outdat/new" / product
    original = fingerprint_tree(source)
    target = tmp_path / "case/outdat/new" / product
    shutil.copytree(source, target)
    before = fingerprint_tree(target)
    segment = target / "segment00000000"
    # A complete frame can exist even when neither index records its clock.
    (segment / "series.frames").write_text("ASTR_FRAME_SERIES_1\n")
    (segment / "series.xdmf").write_text("interrupted index generation\n")
    repaired = repair.repair_segment(segment, publish=True, catalog_bytes=4096)
    expected = expected_fields if product == "fields" else 5
    assert repaired["frames"] == expected and repaired["field_array_read_bytes"] == 0
    assert len(check_series(tmp_path / "case", product)) == expected
    actual_reader = check_series_reader(tmp_path / "case", product)
    assert actual_reader["status"] == "passed" and actual_reader["frames"] == expected
    after = fingerprint_tree(target)
    for name, value in before.items():
        if Path(name).name not in repair.INDEX_NAMES:
            assert after[name] == value
    assert fingerprint_tree(source) == original
    report = {**repaired, "reader": actual_reader, "source": str(source),
              "directory_bytes": sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file())}
    (tmp_path / "repair_report.json").write_text(json.dumps(report, indent=2) + "\n")
    assert sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file()) < 64 * 1024**2


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_relocated_derived_restart_rotation_and_index_recovery(tmp_path, backend, record_property):
    reference_root = Path(os.environ.get("ASTR_OUTPUT_DERIVED_REFERENCE_ROOT",
        ROOT / "tests/gpu_validation/out/or5_real_derived_runtime_20261001"))
    references = []
    for path in reference_root.glob("test_real_derived_exact_restar[0-9]*/result.json"):
        report = json.loads(path.read_text())
        if report["backend"] == backend and report["np"] == 2 and report["axis"] == "z":
            references.append((path.parent, report))
    assert len(references) == 1, "prepare the accepted immutable OR5 NP=2 z reference"
    reference, report = references[0]
    args = arguments(tmp_path, backend, "z")
    with args.executable.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == report["executable_sha256"], \
            "build changed: regenerate the OR5 reference, do not bypass its identity"
    assert report["exact_continuation"] and report["output_switch_unchanged"]
    continuous = reference / (backend+"_np2_continuous")
    seed = reference / (backend+"_np2_seed/outdat/new")
    original = fingerprints(seed)
    before_move = tmp_path / "before_move"
    shutil.copytree(seed, before_move)
    moved_seed = tmp_path / "relocated seed"
    before_move.rename(moved_seed)
    assert not before_move.exists() and fingerprints(moved_seed) == original
    source = moved_seed / "checkpoints/step000000000005"
    run_case(args, ROOT, backend, 2, "reject_schedule", 12, restore=source,
        reuse_root=moved_seed, checkpoint_keep=1, checkpoint_interval=5,
        archive_groups=archive(), buffer_bytes=4096,
        reject="saved output schedule differs; select explicit override")
    resumed, _ = run_case(args, ROOT, backend, 2, "resumed", 12, restore=source,
        reuse_root=moved_seed, checkpoint_keep=1, checkpoint_interval=5,
        archive_groups=archive(), buffer_bytes=4096, override=True)
    for name in ("state.h5", "statistics.h5"):
        compare_fields(continuous / FINAL / name, resumed / FINAL / name)
    assert archive_schedule_payload(continuous / FINAL / "archives.bin") == archive_schedule_payload(
        resumed / FINAL / "archives.bin")
    expected_control = (continuous / FINAL / "control.bin").read_bytes()
    actual_control = (resumed / FINAL / "control.bin").read_bytes()
    assert expected_control[:8] == actual_control[:8] == b"ASTROC04"
    # Contract, completed clock, loop state and last save match; the explicit
    # checkpoint schedule override begins after this fixed-format prefix.
    assert expected_control[:168] == actual_control[:168]
    assert (continuous / FINAL / "insitu_control.bin").read_bytes() == (resumed / FINAL / "insitu_control.bin").read_bytes()
    counts = accounting(resumed, backend)
    relocated = tmp_path / "relocated complete case"
    resumed.rename(relocated)
    root = relocated / "outdat/new"
    assert (root / "checkpoints/LATEST").read_text() == "step000000000012\n"
    assert sorted(p.name for p in (root / "checkpoints").glob("step*")) == [
        "step000000000005", "step000000000012"]
    # The active restore point is protected; the intermediate point is retired.
    assert not (root / "checkpoints/step000000000010").exists()
    restored_snapshot = fingerprints(root)
    for name, checksum in original.items():
        if name.startswith(("resources/", "fields/", "slices/", "checkpoints/step")):
            assert restored_snapshot[name] == checksum
    derivative_errors = {}
    readers = {}
    for product in ("fields", "slices"):
        segment = root / product / "segment00000001"
        assert sorted(p.name for p in segment.glob("step*")) == [
            "step000000000010", "step000000000012"]
        for step in (10, 12):
            compare_fields(segment / f"step{step:012d}/data.h5",
                continuous / f"outdat/new/{product}/segment00000000/step{step:012d}/data.h5")
        immutable = fingerprints(root / product)
        (segment / "series.frames").write_text("ASTR_FRAME_SERIES_1\n")
        (segment / "series.xdmf").write_text("interrupted index generation\n")
        repaired = repair.repair_segment(segment, publish=True, catalog_bytes=4096)
        assert repaired["frames"] == 2 and repaired["field_array_read_bytes"] == 0
        assert check_series(relocated, product, "segment00000001") == check_series(continuous, product)[2:]
        after = fingerprints(root / product)
        for name, checksum in immutable.items():
            if name not in {"segment00000001/"+n for n in repair.INDEX_NAMES}:
                assert after[name] == checksum
        # The single-segment reader copies files only into its private test tree.
        direct = check_series_reader(relocated, product, "segment00000001", extra_names=DERIVED_NAMES)
        assert direct["frames"] == 2
        parent = check_parent_reader(segment, tmp_path / (product+"_combined"), extra_names=DERIVED_NAMES)
        assert parent["segments"] == 2 and parent["frames"] == 4
        readers[product] = {"segment_frames": direct["frames"], "lineage_frames": parent["frames"]}
        assert after == fingerprints(root / product)
    final = root / "checkpoints/step000000000012"
    expected = discrete_reference(final, root / "resources/geometry.h5")
    with h5py.File(root / "fields/segment00000001/step000000000012/data.h5") as data:
        for name, values in zip(DERIVED_NAMES, expected):
            derivative_errors[name] = float(np.max(np.abs(data[name][:]-values)))
            assert derivative_errors[name] <= 2e-10
    assert fingerprints(seed) == original and fingerprints(moved_seed) == original
    result = dict(backend=backend, np=2, axis="z", exact_continuation=True,
        moved_resources=True, checkpoint_override=True, protected_restore_step=5, retired_step=10,
        retained_archive_steps=[0, 5, 10, 12], readers=readers,
        derivative_errors=derivative_errors, accounting=counts,
        reference=str(reference), executable_sha256=report["executable_sha256"],
        test_root_bytes=directory_budget(tmp_path))
    (tmp_path / "relocation_report.json").write_text(json.dumps(result, indent=2)+"\n")
    record_property("result", result)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_relocated_channel_restart_rotation_and_index_recovery(tmp_path, backend, record_property):
    reference_root = Path(os.environ.get("ASTR_OUTPUT_CHANNEL_REFERENCE_ROOT",
        ROOT / "tests/gpu_validation/out/or4_channel_repartition_fixed_20261001"))
    references = []
    for path in reference_root.glob("test_channel_repartition_keeps[0-9]*/channel_repartition.json"):
        report = json.loads(path.read_text())
        if report["backend"] == backend and report["exact"]:
            references.append((path.parent, report))
    assert len(references) == 1, "prepare the accepted exact channel reference for this backend"
    reference, report = references[0]
    args = channel_arguments(tmp_path, backend, report["source"][1], report["filter_workspace"])
    with args.executable.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == report["executable_sha256"], \
            "build changed: regenerate the channel reference, do not bypass its identity"
    assert report["source"] == report["target"] and report["source"][0] == 2
    assert report["state_max_abs"] == report["driver_max_abs"] == report["mean_max_abs"] == 0
    continuous = reference / (backend+"_np2_continuous")
    seed = reference / (backend+"_np2_seed/outdat/new")
    original = fingerprints(seed)
    before_move = tmp_path / "before_move"
    shutil.copytree(seed, before_move)
    moved_seed = tmp_path / "relocated channel seed"
    before_move.rename(moved_seed)
    assert not before_move.exists() and fingerprints(moved_seed) == original
    source = moved_seed / "checkpoints/step000000000005"
    initial = moved_seed / "resources/flowini3d.h5"
    run_case(args, ROOT, backend, 2, "reject_schedule", 12, restore=source,
        reuse_root=moved_seed, checkpoint_keep=1, checkpoint_interval=1,
        archive_groups=groups("steps"), buffer_bytes=4096, initial_resource=initial,
        reject="saved output schedule differs; select explicit override")
    resumed, _ = run_case(args, ROOT, backend, 2, "resumed", 12, restore=source,
        reuse_root=moved_seed, checkpoint_keep=1, checkpoint_interval=1,
        archive_groups=groups("steps"), buffer_bytes=4096, initial_resource=initial, override=True)
    compare_fields(continuous / FINAL / "state.h5", resumed / FINAL / "state.h5")
    if backend == "cpu":
        compare_fields(continuous / FINAL / "statistics.h5", resumed / FINAL / "statistics.h5")
    assert archive_schedule_payload(continuous / FINAL / "archives.bin") == archive_schedule_payload(
        resumed / FINAL / "archives.bin")
    expected_control = (continuous / FINAL / "control.bin").read_bytes()
    actual_control = (resumed / FINAL / "control.bin").read_bytes()
    assert expected_control[:8] == actual_control[:8] == b"ASTROC04"
    assert expected_control[:168] == actual_control[:168]
    # Output schedule override changes control bytes, not the saved driver state.
    assert expected_control[220:652] == actual_control[220:652]
    assert (continuous / FINAL / "insitu_control.bin").read_bytes() == (
        resumed / FINAL / "insitu_control.bin").read_bytes()
    buffers = controlled_checkpoint_buffers(resumed / FINAL / "state.h5", 0, backend)
    if backend == "cpu":
        with h5py.File(resumed / FINAL / "statistics.h5") as mean:
            buffers.extend(16 * int(np.prod(row[3:6]+1)) * 44 + 4096
                for row in mean["partitions"][:].reshape(-1, 8))
    assert max(buffers) <= 64*1024**2
    counts = check_accounting(resumed, backend, 2)
    relocated = tmp_path / "relocated complete channel"
    resumed.rename(relocated)
    root = relocated / "outdat/new"
    assert (root / "checkpoints/LATEST").read_text() == "step000000000012\n"
    assert sorted(p.name for p in (root / "checkpoints").glob("step*")) == [
        "step000000000005", "step000000000012"]
    assert not (root / "checkpoints/step000000000010").exists()
    assert not (root / "checkpoints/step000000000011").exists()
    after_move = fingerprints(root)
    for name, checksum in original.items():
        if name.startswith(("resources/", "fields/", "slices/", "checkpoints/step")):
            assert after_move[name] == checksum
    readers = {}
    for product, steps in (("fields", [6, 8, 10, 12]), ("slices", [6, 9, 12])):
        segment = root / product / "segment00000001"
        assert sorted(frames(relocated, product, "segment00000001")) == steps
        for step in steps:
            compare_fields(segment / f"step{step:012d}/data.h5",
                continuous / f"outdat/new/{product}/segment00000000/step{step:012d}/data.h5")
        check_frame(segment / "step000000000012", product, root / "checkpoints/step000000000012",
                    root / "resources/geometry.h5", backend)
        immutable = fingerprints(root / product)
        (segment / "series.frames").write_text("ASTR_FRAME_SERIES_1\n")
        (segment / "series.xdmf").write_text("interrupted index generation\n")
        repaired = repair.repair_segment(segment, publish=True, catalog_bytes=4096)
        assert repaired["frames"] == len(steps) and repaired["field_array_read_bytes"] == 0
        assert check_series(relocated, product, "segment00000001") == check_series(continuous, product)[-len(steps):]
        after = fingerprints(root / product)
        for name, checksum in immutable.items():
            if name not in {"segment00000001/"+n for n in repair.INDEX_NAMES}:
                assert after[name] == checksum
        direct = check_series_reader(relocated, product, "segment00000001")
        assert direct["frames"] == len(steps)
        parent = check_parent_reader(segment, tmp_path / (product+"_combined"))
        assert parent["segments"] == 2 and parent["frames"] == (8 if product == "fields" else 6)
        readers[product] = {"segment_frames": direct["frames"], "lineage_frames": parent["frames"]}
        assert after == fingerprints(root / product)
    assert fingerprints(seed) == original and fingerprints(moved_seed) == original
    result = dict(backend=backend, np=2, axis=args.axis, filter_workspace=args.filter_workspace,
        exact_continuation=True, cpu_mean44_exact=backend=="cpu", moved_resources=True,
        checkpoint_override=True, protected_restore_step=5, retired_step=10,
        readers=readers, accounting=counts, controlled_checkpoint_bytes=max(buffers),
        reference=str(reference), executable_sha256=report["executable_sha256"],
        test_root_bytes=directory_budget(tmp_path))
    (tmp_path / "channel_relocation_report.json").write_text(json.dumps(result, indent=2)+"\n")
    record_property("result", result)
