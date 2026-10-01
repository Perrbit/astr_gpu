"""Explicit lineage, independent branches and bounded immutable index construction."""
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import h5py
import pytest

from test_output_series_repair import make_segment, repair, reseal, snapshot
from combine_series import combine_series


def register(segment, parent=None, step=0):
    input_size, input_crc = repair.fingerprint(segment / "input.txt")
    if parent is None:
        parent_id, parent_size, parent_crc, relative = -1, 0, 0, "none"
    else:
        parent_id = int(parent.name[7:])
        parent_size, parent_crc = repair.fingerprint(parent / "SEGMENT")
        relative = os.path.relpath(parent, segment)
    (segment / "SEGMENT").write_text(
        f"ASTR_OUTPUT_SEGMENT_2\n{step} {step * .001:.17g}\n{1 if parent else 0} 0000000000000000\n"
        f"{parent_id} {parent_size} {parent_crc:016X}\n{input_size} {input_crc:016X}\n{relative}\n")
    size, crc = repair.fingerprint(segment / "SEGMENT")
    geom_size, geom_crc = repair.fingerprint(segment.parent / "resources/data.h5")
    for frame in segment.glob("step*"):
        (frame / "RESOURCES").write_text(
            f"ASTR_SHARED_RESOURCES 1\n2\nSEGMENT {size} {crc:016X}\ndata.h5 {geom_size} {geom_crc:016X}\n")
        reseal(frame)


@pytest.fixture(params=[(6, []), (12, [("i", 4), ("j", 2), ("k", 1)])])
def branch(request, tmp_path):
    root = make_segment(tmp_path / "seed", *request.param)
    register(root)
    child = root.parent / "segment00000001"
    shutil.copytree(root, child)
    for step in (0, 2):
        shutil.rmtree(child / f"step{step:012d}")
    register(child, root, step=2)
    # An unrelated branch must not be scanned or included, even if corrupt.
    unrelated = root.parent / "segment00000002"
    unrelated.mkdir()
    (unrelated / "SEGMENT").write_text("unrelated incomplete branch\n")
    return root, child


def test_exact_parent_cutoff_no_branch_guessing(branch, tmp_path):
    root, child = branch
    before = snapshot(root.parent)
    dry = combine_series(child)
    assert dry["frames"] == 3 and dry["segments"] == 2 and dry["excluded_parent_frames"] == 1
    assert not dry["published"] and snapshot(root.parent) == before
    out = tmp_path / "combined"
    report = combine_series(child, output=out)
    assert report["field_array_read_bytes"] == 0 and report["catalog_peak_bytes"] < 16384
    rows = (out / "lineage.frames").read_text().splitlines()[1:]
    assert [int(row.split()[0]) for row in rows] == [0, 2, 4]
    assert Path(rows[-1].split(maxsplit=2)[2]).parts[-2] == child.name
    for item in ET.parse(out / "lineage.xdmf").findall(".//DataItem"):
        path, dataset = item.text.split(":", 1)
        with h5py.File(out / path) as handle:
            assert handle[dataset].shape == tuple(map(int, item.attrib["Dimensions"].split()))
    assert snapshot(root.parent) == before
    with pytest.raises(FileExistsError):
        combine_series(child, output=out)
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 4 * 1024**2


def test_v2_single_segment_repair_and_metadata_binding(branch):
    root, child = branch
    assert repair.repair_segment(root)["frames"] == 3
    assert repair.repair_segment(child, publish=True)["frames"] == 1
    (child / "SEGMENT").write_text((child / "SEGMENT").read_text().replace("2 0.002", "2 0.001"))
    with pytest.raises(ValueError, match="coordinate fingerprint"):
        repair.repair_segment(child)


@pytest.mark.parametrize("defect", ["parent_crc", "input", "missing_parent", "symlink_parent", "clock",
                                  "budget", "legacy", "inside", "damaged_frame", "duplicate"])
def test_reject_invalid_chain_without_source_mutation(branch, tmp_path, defect):
    root, child = branch
    kwargs = {}
    if defect == "parent_crc":
        (root / "SEGMENT").write_text((root / "SEGMENT").read_text() + "corruption\n")
    elif defect == "input":
        (root / "input.txt").write_text("modified input\n")
    elif defect == "missing_parent":
        root.rename(root.with_name("missing"))
    elif defect == "symlink_parent":
        moved = root.with_name("moved")
        root.rename(moved)
        root.symlink_to(moved, target_is_directory=True)
    elif defect == "clock":
        register(child, root, step=1)
        # Step two would be omitted, leaving a valid chain: use an inconsistent time instead.
        rows = (child / "SEGMENT").read_text().splitlines()
        rows[1] = "2 0.001"
        (child / "SEGMENT").write_text("\n".join(rows) + "\n")
    elif defect == "budget":
        kwargs["catalog_bytes"] = 64
    elif defect == "legacy":
        (child / "SEGMENT").write_text("ASTR_OUTPUT_SEGMENT_1\n2 .002\n0 0000000000000000\n")
    elif defect == "inside":
        kwargs["output"] = child / "combined"
    elif defect == "damaged_frame":
        (child / "step000000000004/COMPLETE").write_text("bad seal\n")
    elif defect == "duplicate":
        shutil.copytree(root / "step000000000002", child / "step000000000002")
        register(child, root, step=2)
    before = snapshot(child.parent)
    with pytest.raises((ValueError, FileNotFoundError)):
        combine_series(child, **kwargs)
    assert snapshot(child.parent) == before


def test_relative_chain_survives_tree_move(branch, tmp_path):
    root, child = branch
    moved = tmp_path / "moved"
    root.parent.parent.rename(moved)
    assert combine_series(moved / root.parent.name / child.name)["frames"] == 3


def test_external_parent_root(branch, tmp_path):
    root, child = branch
    external = tmp_path / "external" / child.parent.name
    external.mkdir(parents=True)
    shutil.copytree(child.parent / "resources", external / "resources")
    head = external / "segment00000000"
    shutil.copytree(child, head)
    register(head, root, step=2)
    assert combine_series(head)["frames"] == 3


def test_deep_chain_applies_each_restore_cutoff(branch):
    root, child = branch
    head = root.parent / "segment00000003"
    shutil.copytree(child, head)
    register(head, child, step=3)
    report = combine_series(head)
    assert report["segments"] == 3 and report["frames"] == 3
    assert report["excluded_parent_frames"] == 2


def test_cycle_is_not_replaced_with_directory_order(branch, monkeypatch):
    root, child = branch
    import combine_series as module
    original = module.segment_record
    # A checksum-valid cycle is not constructible by ordinary fixtures; exercise
    # traversal's independent guard with trusted metadata reader output.
    def cyclic(path, **options):
        record = original(path, **options)
        if path == root.resolve():
            record["parent"] = os.path.relpath(child, root)
            record["parent_fingerprint"] = original(child)["fingerprint"]
        return record
    monkeypatch.setattr(module, "segment_record", cyclic)
    with pytest.raises(ValueError, match="cycle"):
        module.combine_series(child)


def test_changing_layout_rejects_chain(branch, tmp_path):
    root, _ = branch
    components = 12 if root.parent.name == "fields" else 6
    planes = [] if components == 12 else [("i", 4), ("j", 2), ("k", 1)]
    other = make_segment(tmp_path / "different", components, planes)
    for step in (0, 2):
        shutil.rmtree(other / f"step{step:012d}")
    # Product must match to isolate the field signature rejection.
    if other.parent.name != root.parent.name:
        pytest.fail("fixture selected a different product")
    register(other, root, step=2)
    with pytest.raises(ValueError, match="layout changes"):
        combine_series(other)


@pytest.mark.parametrize("changed", [False, True])
def test_derived_layout_in_parent_chain(tmp_path, changed):
    root = make_segment(tmp_path / "parent", 6, [], derived=(10, 11))
    register(root)
    child = make_segment(tmp_path / "child", 6, [], derived=(12, 13, 14) if changed else (10, 11))
    for step in (0, 2):
        shutil.rmtree(child / f"step{step:012d}")
    register(child, root, step=2)
    before = snapshot(root.parent), snapshot(child.parent)
    if changed:
        with pytest.raises(ValueError, match="layout changes"):
            combine_series(child)
    else:
        report = combine_series(child, output=tmp_path / "combined")
        assert report["frames"] == 3 and report["segments"] == 2
        for grid in ET.parse(tmp_path / "combined/lineage.xdmf").findall(".//Grid[@GridType='Uniform']"):
            assert [a.attrib["Name"] for a in grid.findall("Attribute")] == list(repair.FIELDS)+["Q_rs", "velocity_divergence", "velocity"]
    assert before == (snapshot(root.parent), snapshot(child.parent))
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 4*1024**2
