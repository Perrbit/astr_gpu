"""OR6 dynamic-inflow/AIR5 relocation, retention and immutable index recovery."""
import hashlib
import json
import shutil
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from run_output_air5_restart_validation import launch, state_sanity
from run_output_archive_validation import check_accounting, check_frame, check_series, check_series_reader, frames, groups
from run_output_restart_validation import archive_schedule_payload, compare_fields, run_case
from test_output_archive_segments import check_parent_reader, fingerprints
from test_output_derived_runtime import ROOT, FINAL, directory_budget
from test_output_repartition_runtime import air5_arguments, curve_arguments
from test_output_series_repair_runtime import repair


def arguments(output, backend, case_name):
    if case_name == "dynamic":
        args = curve_arguments(output, backend, "z")
        args.case = "dynamic"
        args.inflow_count = 12
        args.legacy_statistics = True
    else:
        args = air5_arguments(output, backend, "z" if case_name == "hbl" else "x")
        args.case = case_name
        args.reconstruction = 5 if case_name == "hbl" else 3
        args.mean_statistics = True
    return args


def execute(args, backend, name, steps, ranks=2, **options):
    common = dict(buffer_bytes=4096, archive_groups=groups("steps"))
    common.update(options)
    interval = common.pop("interval", 99)
    if args.case == "dynamic":
        return run_case(args, ROOT, backend, ranks, name, steps, checkpoint_interval=interval, **common)
    return launch(args, backend, ranks, name, steps, interval=interval,
                  conservation=backend == "gpu", **common)


@pytest.fixture(scope="module", params=[(case, backend) for case in ("dynamic", "hbl", "sbli")
                                      for backend in ("cpu", "gpu")])
def reference(request, tmp_path_factory):
    case_name, backend = request.param
    output = tmp_path_factory.mktemp(f"lifecycle_{case_name}_{backend}")
    args = arguments(output, backend, case_name)
    continuous, _ = execute(args, backend, "continuous", 12)
    seed, _ = execute(args, backend, "seed", 5)
    final = continuous / FINAL
    with h5py.File(final / "statistics.h5") as stats:
        assert int(stats["identity"][12]) == (5 if case_name == "dynamic" and backend == "gpu" else 6)
        stats.visititems(lambda key, value: assert_finite_dataset(value))
        assert np.any(stats["q0002"][:]), "a zero-history fixture cannot test statistics preservation"
    if case_name != "dynamic":
        state_sanity(final / "state.h5")
        with h5py.File(final / "state.h5") as state:
            assert int(state["identity"][12]) == 7
            assert any(np.any(state[f"q{m:04d}"][:]) for m in range(12, 23)), "carry must be exercised"
    report = dict(case=case_name, backend=backend, axis=args.axis, np=2,
        executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
        directory_bytes=directory_budget(output))
    (output / "reference.json").write_text(json.dumps(report, indent=2)+"\n")
    directory_budget(output)
    return args, backend, continuous, seed, report


def test_boundary_relocated_retention_and_index_recovery(reference, tmp_path, record_property):
    original_args, backend, continuous, seed, report = reference
    args = SimpleNamespace(**vars(original_args)); args.output = tmp_path
    assert hashlib.sha256(args.executable.read_bytes()).hexdigest() == report["executable_sha256"]
    source_tree = seed / "outdat/new"
    original = fingerprints(source_tree)
    before_move = tmp_path / "before_move"
    shutil.copytree(source_tree, before_move)
    moved = tmp_path / "relocated input tree"
    before_move.rename(moved)
    assert not before_move.exists() and fingerprints(moved) == original
    source = moved / "checkpoints/step000000000005"
    resumed, _ = execute(args, backend, "resumed", 12, restore=source, reuse_root=moved,
                         interval=1, checkpoint_keep=1, override=True)
    names = ["state.h5", "statistics.h5"] + (["inflow.h5"] if args.case == "dynamic" else [])
    for name in names:
        compare_fields(continuous / FINAL / name, resumed / FINAL / name)
    if args.case != "dynamic":
        for name in ["air5_config.bin"] + (["air5_conservation.bin"] if backend == "gpu" else []):
            assert (continuous / FINAL / name).read_bytes() == (resumed / FINAL / name).read_bytes()
        state_sanity(resumed / FINAL / "state.h5")
    assert archive_schedule_payload(continuous / FINAL / "archives.bin") == archive_schedule_payload(
        resumed / FINAL / "archives.bin")
    expected = (continuous / FINAL / "control.bin").read_bytes()
    actual = (resumed / FINAL / "control.bin").read_bytes()
    assert expected[:8] == actual[:8] == b"ASTROC04" and expected[:168] == actual[:168]
    assert (continuous / FINAL / "insitu_control.bin").read_bytes() == (resumed / FINAL / "insitu_control.bin").read_bytes()
    accounting = check_accounting(resumed, backend, 2)
    relocated = tmp_path / "relocated complete run"
    resumed.rename(relocated)
    root = relocated / "outdat/new"
    assert (root / "checkpoints/LATEST").read_text() == "step000000000012\n"
    assert sorted(p.name for p in (root / "checkpoints").glob("step*")) == [
        "step000000000005", "step000000000012"]
    current = fingerprints(root)
    for name, digest in original.items():
        if name.startswith(("resources/", "fields/", "slices/", "checkpoints/step")):
            assert current[name] == digest, name
    if args.case == "dynamic":
        assert not (relocated / "inflow").exists()
        assert not (relocated / "datin/inlet.prof").exists()
        assert not (relocated / "datin/grid.flatplate.h5").exists()
    else:
        assert not list((relocated / "datin").glob("air5_*dat"))
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
        result = repair.repair_segment(segment, publish=True, catalog_bytes=4096)
        assert result["frames"] == len(steps) and result["field_array_read_bytes"] == 0
        assert check_series(relocated, product, "segment00000001") == check_series(continuous, product)[-len(steps):]
        after = fingerprints(root / product)
        for name, digest in immutable.items():
            if name not in {"segment00000001/"+n for n in repair.INDEX_NAMES}:
                assert after[name] == digest
        direct = check_series_reader(relocated, product, "segment00000001")
        parent = check_parent_reader(segment, tmp_path / (product+"_combined"))
        assert direct["frames"] == len(steps)
        assert parent["segments"] == 2 and parent["frames"] == (8 if product == "fields" else 6)
        readers[product] = dict(segment=direct["frames"], lineage=parent["frames"])
        assert after == fingerprints(root / product)
    assert fingerprints(source_tree) == fingerprints(moved) == original
    result = dict(report, exact_continuation=True, exact_statistics=True,
        resources_moved=True, restore_protected=True, intermediate_checkpoints_retired=True,
        readers=readers, accounting=accounting, reference_directory_bytes=report["directory_bytes"],
        directory_bytes=directory_budget(tmp_path))
    (tmp_path / "lifecycle.json").write_text(json.dumps(result, indent=2)+"\n")
    directory_budget(tmp_path)
    record_property("result", json.dumps(result))


def test_boundary_schedule_change_requires_override(reference, tmp_path):
    original_args, backend, _, seed, _ = reference
    args = SimpleNamespace(**vars(original_args)); args.output = tmp_path
    source_tree = seed / "outdat/new"
    original = fingerprints(source_tree)
    rejected, _ = execute(args, backend, "rejected", 12,
        restore=source_tree / "checkpoints/step000000000005", reuse_root=source_tree,
        interval=1, checkpoint_keep=1, reject="saved output schedule differs; select explicit override")
    after = fingerprints(rejected / "outdat/new")
    assert all(after[name] == digest for name, digest in original.items())
    assert set(name for name in after if name.endswith("/COMPLETE")) == set(
        name for name in original if name.endswith("/COMPLETE"))
    assert fingerprints(source_tree) == original
    directory_budget(tmp_path)


@pytest.mark.parametrize("target", ["one_rank", "other_axis"])
def test_unregistered_history_repartition_is_rejected(reference, tmp_path, target, record_property):
    original_args, backend, _, seed, report = reference
    args = SimpleNamespace(**vars(original_args))
    if args.case == "dynamic":
        pytest.skip("x/z migration now registered; dynamic y refusal is covered separately")
    if args.case in ("hbl", "sbli") and target == "one_rank":
        pytest.skip("bounded AIR5 history migration now registered; wrong-axis refusals covered separately")
    args.output = tmp_path
    assert hashlib.sha256(args.executable.read_bytes()).hexdigest() == report["executable_sha256"]
    source_tree = seed / "outdat/new"
    original = fingerprints(source_tree)
    ranks = 1 if target == "one_rank" else 2
    if target == "other_axis":
        args.axis = "x" if original_args.axis == "z" else "z"
    message = "exact restore rank count" if ranks == 1 else "exact restore partition mismatch"
    rejected, _ = execute(args, backend, "history_rejected", 12, ranks=ranks,
        restore=source_tree / "checkpoints/step000000000005", reject=message)
    assert not list((rejected / "outdat/new").rglob("COMPLETE"))
    assert fingerprints(source_tree) == original
    record_property("refusal", json.dumps(dict(case=args.case, backend=backend,
        source_axis=original_args.axis, target_axis=args.axis, target_ranks=ranks,
        error=message, executable_sha256=report["executable_sha256"],
        directory_bytes=directory_budget(tmp_path))))


def assert_finite_dataset(value):
    if isinstance(value, h5py.Dataset):
        assert np.isfinite(value[:]).all()
