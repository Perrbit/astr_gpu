"""Small immutable native-layout fixtures for explicit offline index recovery."""
import importlib.util
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/output"))
spec = importlib.util.spec_from_file_location("series_repair", ROOT / "scripts/output/repair_series.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def seal_file_records(root, names, magic):
    rows = [magic, str(len(names))]
    for name in names:
        size, crc = repair.fingerprint(root / name)
        rows.append(f"{name} {size} {crc:016X}")
    return "\n".join(rows) + "\n"


def reseal(frame):
    files = ("data.h5", "data.xdmf", "FRAME", "RESOURCES")
    (frame / "MANIFEST").write_text(seal_file_records(frame, files, "ASTR_CHECKPOINT_BUNDLE 1"))
    size, crc = repair.fingerprint(frame / "MANIFEST")
    (frame / "COMPLETE").write_text(f"ASTR_COMPLETE_1 {size} {crc:016X}\n")


def snapshot(path):
    return {str(p.relative_to(path)): (repair.fingerprint(p), p.stat().st_mtime_ns)
            for p in path.rglob("*") if p.is_file()}


def make_segment(root, components, planes, derived=()):
    product = root / ("slices" if planes else "fields")
    segment = product / "segment00000000"
    resources = product / "resources"
    resources.mkdir(parents=True)
    segment.mkdir()
    units = "si" if components == 12 else "dimensionless"
    fields = tuple(repair.AIR5_FIELDS if components == 12 else repair.FIELDS)
    fields += tuple(repair.DERIVED_NAMES[n-1] for n in derived)
    (segment / "SEGMENT").write_text("ASTR_OUTPUT_SEGMENT_1\n0 0.0\n0 0000000000000000\n")
    (segment / "input.txt").write_text(f"synthetic\n8,6,4\nt,t,t\n{'f' if units == 'si' else 't'},f,f,f,f,f,f,f\n")
    labels = [(axis + f"{index:012d}", "ijk".index(axis) + 1, index) for axis, index in planes] or [("", 0, 0)]
    shape = (5, 7, 9)
    encoding = np.arange(np.prod(shape), dtype="<f8").reshape(shape)
    with h5py.File(resources / "data.h5", "w") as geometry:
        geometry.attrs.update(units=np.bytes_(units), center=np.bytes_("Node"), phase=np.bytes_("completed_step"))
        for label, axis, index in labels:
            leaf = geometry.create_group(label) if label else geometry
            selected_shape = tuple(n for d, n in enumerate(shape) if axis == 0 or d != 3 - axis)
            meta = np.array([1, 0, *np.array([0., 0., 1.], dtype="<f8").view("<i8"),
                             axis, index, 9, 7, 5, components], dtype="<i8")
            leaf["metadata"] = meta
            leaf["coordinates"] = np.arange(np.prod(selected_shape) * 3, dtype="<f8").reshape(selected_shape + (3,))
    for step in (0, 2, 4):
        frame = segment / f"step{step:012d}"
        frame.mkdir()
        total_bytes = 0
        with h5py.File(frame / "data.h5", "w") as handle:
            handle.attrs.update(units=np.bytes_(units), center=np.bytes_("Node"), phase=np.bytes_("completed_step"))
            if derived:
                handle.attrs["derived_layout"] = np.bytes_(" ".join(map(str, tuple(derived)+(0,)*(14-len(derived)))))
            for label, axis, index in labels:
                leaf = handle.create_group(label) if label else handle
                meta = np.array([1, step, *np.array([step * .001, .001 if step else 0., .001], dtype="<f8").view("<i8"),
                                 axis, index, 9, 7, 5, components], dtype="<i8")
                leaf["metadata"] = meta
                selected = [slice(None)] * 3
                if axis:
                    selected[3 - axis] = index
                data = encoding[tuple(selected)]
                for m, name in enumerate(fields):
                    leaf[name] = data + m / 16
                for n in derived:
                    dataset = leaf[repair.DERIVED_NAMES[n-1]]
                    quantity_units = "dimensionless" if units == "dimensionless" else ("s^-2" if n == 10 else "s^-1")
                    dataset.attrs.update(quantity_units=np.bytes_(quantity_units),
                        definition=np.bytes_(repair.DERIVED_DEFINITIONS[n-1]), coordinate_space=np.bytes_("physical"),
                        time_dimension_exponent=np.bytes_("-2" if n == 10 else "-1"))
                leaf["velocity"] = np.stack([data + m / 16 for m in (1, 2, 3)], axis=-1)
                total_bytes += data.size * len(fields) * 8
            if planes:
                handle["frame_identity"] = np.array([*meta[:5], *meta[7:11], int(units == "si")], dtype="<i8")
        magic = "ASTR_DERIVED_FRAME_1" if derived else "ASTR_BASIC_FRAME_1"
        declaration = f"{magic}\n{step} {step * .001:.17g}\n{components} {units}\n{total_bytes} {total_bytes}\n"
        if derived:
            declaration += " ".join(map(str, tuple(derived)+(0,)*(14-len(derived))))+"\n"
        (frame / "FRAME").write_text(declaration)
        (frame / "data.xdmf").write_text("sealed XML is not used as a source of unvalidated HDF paths\n")
        (frame / "RESOURCES").write_text(seal_file_records(resources, ["data.h5"], "ASTR_SHARED_RESOURCES 1"))
        reseal(frame)
    (segment / "series.frames").write_text("ASTR_FRAME_SERIES_1\n0 0\n")
    (segment / "series.xdmf").write_text("old truncated index\n")
    return segment


@pytest.fixture(params=[(6, []), (12, [("i", 4), ("j", 2), ("k", 1)])])
def segment(request, tmp_path):
    return make_segment(tmp_path, *request.param)


def test_dry_run_and_explicit_repair(segment):
    before = snapshot(segment.parent)
    report = repair.repair_segment(segment)
    assert not report["published"] and report["frames"] == 3
    assert report["crc_scan_bytes"] > report["geometry_fingerprint_bytes"]
    assert report["field_array_read_bytes"] == 0 and snapshot(segment.parent) == before
    repaired = repair.repair_segment(segment, publish=True)
    assert repaired["published"] and repaired["catalog_peak_bytes"] < 4096
    rows = (segment / "series.frames").read_text().splitlines()
    assert [int(row.split()[0]) for row in rows[1:]] == [0, 2, 4]
    grids = ET.parse(segment / "series.xdmf").findall("./Domain/Grid/Grid")
    assert [float(grid.find("Time").attrib["Value"]) for grid in grids] == [0, .002, .004]
    for item in ET.parse(segment / "series.xdmf").findall(".//DataItem"):
        path, name = item.text.split(":", 1)
        with h5py.File(segment / path) as handle:
            assert name in handle
            assert tuple(map(int, item.attrib["Dimensions"].split())) == handle[name].shape
    after = snapshot(segment.parent)
    for name, fingerprint in before.items():
        if Path(name).name not in repair.INDEX_NAMES:
            assert after[name] == fingerprint
    stable = {name: (segment / name).read_bytes() for name in repair.INDEX_NAMES}
    repair.repair_segment(segment, publish=True)
    assert all((segment / name).read_bytes() == data for name, data in stable.items())
    assert sum(p.stat().st_size for p in segment.parent.rglob("*") if p.is_file()) < 4 * 1024**2


@pytest.mark.parametrize("components,planes,selection", [
    (6, [], tuple(range(1, 15))), (6, [("i", 4), ("j", 2), ("k", 1)], tuple(range(1, 10))),
    (6, [], (10, 11)), (6, [("i", 4)], (12, 13, 14)), (12, [("k", 1)], tuple(range(1, 15)))])
def test_derived_index_layout(tmp_path, components, planes, selection):
    segment = make_segment(tmp_path, components, planes, derived=selection)
    before = snapshot(segment.parent)
    report = repair.repair_segment(segment)
    assert report["frames"] == 3 and report["field_array_read_bytes"] == 0
    assert snapshot(segment.parent) == before
    repair.repair_segment(segment, publish=True)
    expected = list(repair.FIELDS if components == 6 else repair.AIR5_FIELDS)+[repair.DERIVED_NAMES[n-1] for n in selection]+["velocity"]
    for grid in ET.parse(segment / "series.xdmf").findall(".//Grid[@GridType='Uniform']"):
        assert [a.attrib["Name"] for a in grid.findall("Attribute")] == expected
        for item in grid.findall(".//DataItem"):
            path, name = item.text.split(":", 1)
            with h5py.File(segment / path) as handle:
                assert tuple(map(int, item.attrib["Dimensions"].split())) == handle[name].shape
    assert sum(p.stat().st_size for p in segment.parent.rglob("*") if p.is_file()) < 4 * 1024**2


@pytest.mark.parametrize("fault", ["missing_layout", "layout_gap", "duplicate", "unsorted", "range", "vlen_layout",
    "missing_field", "fp32", "units", "definition", "coordinate_space", "time_power", "nonscalar_definition", "bytes"])
def test_derived_index_rejects_bad_contract(tmp_path, fault):
    segment = make_segment(tmp_path, 6, [], derived=(10, 11))
    frame = segment / "step000000000004"
    with h5py.File(frame / "data.h5", "r+") as handle:
        if fault == "missing_layout":
            del handle.attrs["derived_layout"]
        elif fault == "vlen_layout":
            del handle.attrs["derived_layout"]
            handle.attrs["derived_layout"] = "10 11 "+"0 "*11+"0"
        elif fault == "missing_field":
            del handle["Q_rs"]
        elif fault == "fp32":
            values = handle["Q_rs"][:]
            del handle["Q_rs"]
            handle["Q_rs"] = values.astype("<f4")
        elif fault in ("units", "definition", "coordinate_space", "time_power", "nonscalar_definition"):
            attribute = {"units":"quantity_units", "time_power":"time_dimension_exponent",
                         "nonscalar_definition":"definition"}.get(fault, fault)
            del handle["Q_rs"].attrs[attribute]
            handle["Q_rs"].attrs[attribute] = (np.array([np.bytes_(repair.DERIVED_DEFINITIONS[9])])
                if fault == "nonscalar_definition" else np.bytes_("bad"))
    rows = (frame / "FRAME").read_text().splitlines()
    if fault in ("layout_gap", "duplicate", "unsorted", "range"):
        selection = {"layout_gap":[10,0,11]+[0]*11, "duplicate":[10,10]+[0]*12,
                     "unsorted":[11,10]+[0]*12, "range":[10,15]+[0]*12}[fault]
        rows[4] = " ".join(map(str, selection))
    elif fault == "bytes":
        rows[3] = "1 1"
    (frame / "FRAME").write_text("\n".join(rows)+"\n")
    reseal(frame)
    indexes = {name: (segment / name).read_bytes() for name in repair.INDEX_NAMES}
    with pytest.raises(ValueError):
        repair.repair_segment(segment, publish=True)
    assert all((segment / name).read_bytes() == old for name, old in indexes.items())


def test_derived_layout_change_not_silently_joined(tmp_path):
    segment = make_segment(tmp_path / "original", 6, [], derived=(10, 11))
    other = make_segment(tmp_path / "other", 6, [], derived=(12, 13, 14))
    frame = segment / "step000000000004"
    for path in (other / frame.name).iterdir():
        shutil.copyfile(path, frame / path.name)
    (frame / "RESOURCES").write_text(seal_file_records(segment.parent / "resources", ["data.h5"], "ASTR_SHARED_RESOURCES 1"))
    reseal(frame)
    # This frame is valid in isolation; rejection must come from the segment layout.
    repair.validate_frame(frame, segment.parent / "resources/data.h5", repair.fingerprint(segment.parent / "resources/data.h5"))
    indexes = {name: (segment / name).read_bytes() for name in repair.INDEX_NAMES}
    with pytest.raises(ValueError, match="changing field/plane layout"):
        repair.repair_segment(segment, publish=True)
    assert all((segment / name).read_bytes() == old for name, old in indexes.items())


@pytest.mark.parametrize("defect", ["incomplete", "crc", "resource", "clock", "fp32", "external", "units", "byte_count",
                                  "nonscalar_attribute", "vlen_attribute", "virtual"])
def test_invalid_frame_never_replaces_indexes(segment, defect):
    frame = segment / "step000000000004"
    if defect == "incomplete":
        (frame / "COMPLETE").unlink()
    elif defect == "crc":
        with (frame / "data.h5").open("ab") as handle:
            handle.write(b"corrupted")
    elif defect == "resource":
        with (segment.parent / "resources/data.h5").open("ab") as handle:
            handle.write(b"corrupted")
    elif defect == "byte_count":
        lines = (frame / "FRAME").read_text().splitlines()
        lines[-1] = "1 1"
        (frame / "FRAME").write_text("\n".join(lines) + "\n")
        reseal(frame)
    else:
        with h5py.File(frame / "data.h5", "r+") as handle:
            leaf = handle if "metadata" in handle else handle["i000000000004"]
            if defect == "clock":
                leaf["metadata"][2] = np.array([.003], dtype="<f8").view("<i8")[0]
            elif defect == "units":
                handle.attrs["units"] = np.bytes_("other")
            elif defect == "nonscalar_attribute":
                del handle.attrs["center"]
                handle.attrs["center"] = np.array([np.bytes_("Node")])
            elif defect == "vlen_attribute":
                del handle.attrs["center"]
                handle.attrs["center"] = "Node"
            else:
                values = leaf["density"][:]
                del leaf["density"]
                if defect == "fp32":
                    leaf["density"] = values.astype("<f4")
                elif defect == "external":
                    leaf["density"] = h5py.ExternalLink("missing.h5", "/density")
                else:
                    layout = h5py.VirtualLayout(shape=values.shape, dtype="<f8")
                    layout[...] = h5py.VirtualSource("missing.h5", "/density", shape=values.shape)
                    leaf.create_virtual_dataset("density", layout)
        reseal(frame)
    indexes = {name: (segment / name).read_bytes() for name in repair.INDEX_NAMES}
    with pytest.raises((ValueError, FileNotFoundError)):
        repair.repair_segment(segment, publish=True)
    assert all((segment / name).read_bytes() == before for name, before in indexes.items())
    assert (frame / "data.h5").exists()


def test_partial_candidates_are_retained_not_indexed(segment):
    candidate = segment / "step000000000006.tmp"
    candidate.mkdir()
    (candidate / "partial").write_bytes(b"diagnostic remains")
    native_temporary = segment / "series.xdmf.tmp"
    native_temporary.write_bytes(b"failed native index remains")
    report = repair.repair_segment(segment, publish=True)
    assert report["ignored_candidates"] == 1 and report["frames"] == 3
    assert (candidate / "partial").read_bytes() == b"diagnostic remains"
    assert native_temporary.read_bytes() == b"failed native index remains"


@pytest.mark.parametrize("fault", ["symlink", "hardlink", "staging", "budget", "origin", "empty"])
def test_preflight_and_bounds(segment, tmp_path, fault):
    if fault in ("symlink", "hardlink"):
        target = segment / "series.frames"
        target.unlink()
        victim = tmp_path / "external"
        victim.write_bytes(b"do not change")
        if fault == "symlink":
            target.symlink_to(victim)
        else:
            target.hardlink_to(victim)
    elif fault == "staging":
        (segment / repair.STAGED_NAMES[0]).write_bytes(b"prior repair diagnostic")
    elif fault == "origin":
        (segment / "SEGMENT").write_text("ASTR_OUTPUT_SEGMENT_1\n10 1.0\n0 0000000000000000\n")
    elif fault == "empty":
        for frame in segment.glob("step*"):
            shutil.rmtree(frame)
    with pytest.raises((ValueError, FileExistsError)):
        repair.repair_segment(segment, publish=True, catalog_bytes=100 if fault == "budget" else 1048576)
    if fault in ("symlink", "hardlink"):
        assert victim.read_bytes() == b"do not change"


def test_second_index_replace_failure_preserves_sealed_frames(segment, monkeypatch):
    before = snapshot(segment.parent)
    replace = repair.os.replace
    def fail_second(source, target):
        if Path(target).name == "series.xdmf":
            raise OSError("injected second index replacement failure")
        return replace(source, target)
    monkeypatch.setattr(repair.os, "replace", fail_second)
    with pytest.raises(OSError, match="second index"):
        repair.repair_segment(segment, publish=True)
    assert len((segment / "series.frames").read_text().splitlines()) == 4
    assert (segment / "series.xdmf").read_text() == "old truncated index\n"
    assert (segment / repair.STAGED_NAMES[1]).is_file()
    after = snapshot(segment.parent)
    for name, state in before.items():
        if Path(name).name not in repair.INDEX_NAMES:
            assert after[name] == state
