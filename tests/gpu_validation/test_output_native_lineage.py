"""Bounded native parent catalog and metadata gate; no solver advancement."""
import shutil
import os
import hashlib
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest

from test_output_fields import probe, run_probe, MPIEXEC
from test_output_parent_series import register
from test_output_series_repair import make_segment, repair, reseal, snapshot
from combine_series import combine_series


def source_snapshot(root):
    """Record deliberately invalid aliases without accepting them as native files."""
    result = {}
    for path in root.rglob("*"):
        info = path.lstat()
        if path.is_symlink():
            value = ("symlink", os.readlink(path))
        elif path.is_file():
            with path.open("rb") as stream:
                value = ("file", hashlib.file_digest(stream, "sha256").hexdigest())
        else:
            continue
        result[str(path.relative_to(root))] = (info.st_mode, info.st_nlink, info.st_mtime_ns, value)
    return result


def invoke(probe, head, components, planes, derived=(), decomposition=1):
    return run_probe([MPIEXEC, "--mca", "coll_hcoll_enable", "0", "-n", "2", str(probe),
        str(head), str(components), str(decomposition), "native_prepare_"+("planes" if planes else "volume"),
        "all" if derived else "none"])


def make_chain(tmp_path, components=6, planes=(), derived=()):
    parent = make_segment(tmp_path / "parent & branch", components, planes, derived)
    register(parent)
    head = parent.parent / "segment00000001"
    shutil.copytree(parent, head)
    for step in (0, 2):
        shutil.rmtree(head / f"step{step:012d}")
    register(head, parent, step=2)
    repair.repair_segment(head, publish=True)
    return parent, head


@pytest.mark.parametrize("components,planes,derived", [
    (6, (), ()), (12, (), ()), (6, (("i", 4), ("j", 2), ("k", 1)), ()),
    (12, (("i", 4), ("j", 2), ("k", 1)), ()), (6, (), tuple(range(1, 15))),
    (12, (("i", 4), ("j", 2), ("k", 1)), tuple(range(1, 15)))])
def test_native_cutoff_and_layout_reference(probe, tmp_path, components, planes, derived):
    parent, head = make_chain(tmp_path, components, planes, derived)
    before = snapshot(parent)
    result = invoke(probe, head, components, planes, derived)
    assert result.returncode == 0 and "OUTPUT_NATIVE_LINEAGE_COMPATIBLE" in result.stdout, result.stdout+result.stderr
    rows = (head / "lineage.frames.tmp").read_text().splitlines()
    assert rows[0] == "ASTR_NATIVE_LINEAGE_1" and [int(row.split()[0]) for row in rows[1::3]] == [0, 2, 4]
    for item in ET.parse(head / "lineage.xdmf.tmp").findall(".//DataItem"):
        path, dataset = item.text.strip().split(":", 1)
        with h5py.File(head / path) as handle:
            assert handle[dataset].shape == tuple(map(int, item.attrib["Dimensions"].split()))
    report = combine_series(head)
    assert report["frames"] == 3 and report["excluded_parent_frames"] == 1
    assert snapshot(parent) == before
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 4*1024**2


@pytest.mark.parametrize("override", ["derived", "planes"])
def test_valid_native_layout_override_does_not_abort(probe, tmp_path, override):
    planes = (("i", 4), ("j", 2), ("k", 1)) if override == "planes" else ()
    parent, head = make_chain(tmp_path, planes=(("k", 1),) if planes else ())
    before = snapshot(parent)
    result = invoke(probe, head, 6, planes, tuple(range(1, 15)) if override == "derived" else ())
    assert result.returncode == 0 and "OUTPUT_NATIVE_LINEAGE_INCOMPATIBLE" in result.stdout, result.stdout+result.stderr
    assert not (head / "lineage.frames.tmp").exists() and not (head / "lineage.xdmf.tmp").exists()
    assert snapshot(parent) == before


def test_native_unsorted_plane_selection_is_same_layout(probe, tmp_path):
    planes = (("i", 4), ("i", 5), ("j", 2), ("k", 1))
    parent, head = make_chain(tmp_path, planes=planes)
    before = snapshot(parent)
    result = invoke(probe, head, 6, planes, decomposition=2)
    assert result.returncode == 0 and "OUTPUT_NATIVE_LINEAGE_COMPATIBLE" in result.stdout, result.stdout+result.stderr
    for item in ET.parse(head / "lineage.xdmf.tmp").findall(".//DataItem"):
        path, dataset = item.text.strip().split(":", 1)
        with h5py.File(head / path) as handle:
            assert handle[dataset].shape == tuple(map(int, item.attrib["Dimensions"].split()))
    assert snapshot(parent) == before


@pytest.mark.parametrize("fault", ["parent_crc", "input_crc", "missing_parent", "symlink_parent", "legacy",
    "damaged_frame", "metadata_clock", "fp32", "inventory", "derived_definition", "missing_metadata", "clock_cutoff",
    "symlink_payload", "hardlink_payload", "unexpected_member", "unbound_segment"])
def test_invalid_native_source_is_not_a_layout_override(probe, tmp_path, fault):
    derived = tuple(range(1, 15))
    parent, head = make_chain(tmp_path, derived=derived)
    frame = parent / "step000000000002"
    if fault == "parent_crc":
        with (parent / "SEGMENT").open("a") as stream:
            stream.write("corrupt\n")
    elif fault == "input_crc":
        (parent / "input.txt").write_text("corrupt\n")
    elif fault in ("missing_parent", "symlink_parent"):
        moved = parent.with_name("moved")
        parent.rename(moved)
        if fault == "symlink_parent":
            parent.symlink_to(moved, target_is_directory=True)
    elif fault == "legacy":
        (parent / "SEGMENT").write_text("ASTR_OUTPUT_SEGMENT_1\n0 0\n0 0000000000000000\n")
        register(head, parent, step=2)
    elif fault == "damaged_frame":
        (frame / "COMPLETE").write_text("bad\n")
    elif fault in ("symlink_payload", "hardlink_payload"):
        external = tmp_path / "payload.h5"
        shutil.move(frame / "data.h5", external)
        if fault == "symlink_payload":
            (frame / "data.h5").symlink_to(external)
        else:
            os.link(external, frame / "data.h5")
    elif fault == "unexpected_member":
        (frame / "extra.txt").write_text("not part of the frame\n")
    elif fault == "unbound_segment":
        rows = (frame / "RESOURCES").read_text().splitlines()
        (frame / "RESOURCES").write_text(rows[0]+"\n1\n"+rows[-1]+"\n")
        reseal(frame)
    elif fault == "clock_cutoff":
        rows = (head / "SEGMENT").read_text().splitlines()
        rows[1] = "2 0.001"
        (head / "SEGMENT").write_text("\n".join(rows)+"\n")
    else:
        with h5py.File(frame / "data.h5", "r+") as handle:
            if fault == "metadata_clock":
                meta = handle["metadata"][:]; meta[1] = 3; handle["metadata"][:] = meta
            elif fault == "fp32":
                values = handle["density"][:]; del handle["density"]; handle["density"] = values.astype("<f4")
            elif fault == "inventory":
                handle["extra"] = np.array([1.])
            elif fault == "missing_metadata":
                del handle["metadata"]
            else:
                del handle["Q_rs"].attrs["definition"]
                handle["Q_rs"].attrs["definition"] = np.bytes_("bad")
        reseal(frame)
    source = parent if parent.is_dir() and not parent.is_symlink() else parent.with_name("moved")
    before = source_snapshot(source)
    result = invoke(probe, head, 6, (), derived)
    assert result.returncode != 0, result.stdout+result.stderr
    assert "OUTPUT_NATIVE_LINEAGE_INCOMPATIBLE" not in result.stdout
    assert not (head / "lineage.xdmf.tmp").exists()
    assert source_snapshot(source) == before
