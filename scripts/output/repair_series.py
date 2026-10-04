"""Validate or explicitly rebuild one stopped native archive segment's indexes."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from export_checkpoint import (AIR5_FIELDS, FIELDS, entries, fingerprint,
                               native_dataset, plain_file, small_text, validate_units)


FRAME_NAME = re.compile(r"step[0-9]{12}")
PLANE_NAME = re.compile(r"([ijk])([0-9]{12})")
INDEX_NAMES = ("series.frames", "series.xdmf")
STAGED_NAMES = tuple(name + ".repair.tmp" for name in INDEX_NAMES)
DERIVED_NAMES = ("velocity_gradient_xx", "velocity_gradient_yx", "velocity_gradient_zx",
                 "velocity_gradient_xy", "velocity_gradient_yy", "velocity_gradient_zy",
                 "velocity_gradient_xz", "velocity_gradient_yz", "velocity_gradient_zz",
                 "Q_rs", "velocity_divergence", "vorticity_x", "vorticity_y", "vorticity_z")
DERIVED_DEFINITIONS = ("du_x/dx", "du_y/dx", "du_z/dx", "du_x/dy", "du_y/dy", "du_z/dy",
    "du_x/dz", "du_y/dz", "du_z/dz", "-0.5*tr(A*A); A(i,j)=du_i/dx_j; full strain",
    "tr(A); A(i,j)=du_i/dx_j", "du_z/dy-du_y/dz", "du_x/dz-du_z/dx", "du_y/dx-du_x/dy")


def directory(path):
    if not stat.S_ISDIR(path.lstat().st_mode):
        raise ValueError(f"not a plain directory: {path}")


def optional_plain(path):
    try:
        plain_file(path)
    except FileNotFoundError:
        pass


def frame_catalog(segment, budget):
    names, ignored, name_bytes = [], 0, 0
    allowed = {"SEGMENT", "input.txt", "LATEST", ".LATEST.tmp",
               "lineage.parent", "lineage.frames", "lineage.xdmf", "lineage.frames.tmp", "lineage.xdmf.tmp",
               *INDEX_NAMES, *(name + ".tmp" for name in INDEX_NAMES)}
    with os.scandir(segment) as members:
        for member in members:
            name = member.name
            if FRAME_NAME.fullmatch(name):
                directory(segment / name)
                names.append(name)
                name_bytes += sys.getsizeof(name)
                # Include a conservative bound for list.sort's pointer scratch.
                if name_bytes + sys.getsizeof(names) + len(names) * 8 > budget:
                    raise ValueError("frame catalog exceeds its explicit byte budget")
            elif name in allowed:
                plain_file(segment / name)
            elif re.fullmatch(r"step[0-9]{12}\.tmp", name):
                directory(segment / name)
                ignored += 1
            else:
                raise ValueError(f"unexpected segment member: {name}")
    peak = name_bytes + sys.getsizeof(names) + len(names) * 8
    if peak > budget:
        raise ValueError("frame catalog exceeds its explicit byte budget")
    names.sort()
    return names, ignored, peak


def text_attribute(handle, name, expected):
    if name not in handle.attrs:
        raise ValueError(f"missing native archive attribute: {name}")
    declaration = handle.attrs.get_id(name)
    if declaration.shape != () or declaration.dtype.kind != "S" or declaration.dtype.itemsize != len(expected):
        raise ValueError(f"invalid native archive attribute layout: {name}")
    value = handle.attrs.get(name)
    if not isinstance(value, (bytes, np.bytes_)) or value.decode("ascii") != expected:
        raise ValueError(f"invalid native archive attribute: {name}")


def attributes(handle, units):
    for name, expected in (("units", units), ("center", "Node"), ("phase", "completed_step")):
        text_attribute(handle, name, expected)


def derived_selection(declaration):
    tokens = declaration.split()
    if len(tokens) != 14 or any(not re.fullmatch(r"0|[1-9][0-9]*", token) for token in tokens):
        raise ValueError("invalid derived field selection")
    values = tuple(map(int, tokens))
    selected = tuple(value for value in values if value)
    if (not selected or any(value > 14 for value in selected) or
            any(a >= b for a, b in zip(selected, selected[1:])) or
            values != selected+(0,)*(14-len(selected))):
        raise ValueError("invalid derived field selection")
    return selected


def validate_derived_dataset(dataset, index, units):
    quantity_units = "dimensionless" if units == "dimensionless" else ("s^-2" if index == 10 else "s^-1")
    text_attribute(dataset, "quantity_units", quantity_units)
    text_attribute(dataset, "definition", DERIVED_DEFINITIONS[index-1])
    text_attribute(dataset, "coordinate_space", "physical")
    text_attribute(dataset, "time_dimension_exponent", "-2" if index == 10 else "-1")


def metadata(group):
    values = native_dataset(group, "metadata", (11,), "<i8")[:]
    time, dt_used, dt_next = values[2:5].view("<f8")
    if (values[0] != 1 or not 0 <= values[1] < 10**12 or
            not np.isfinite([time, dt_used, dt_next]).all() or
            time < 0 or dt_used < 0 or dt_next <= 0 or
            (values[1] > 0 and dt_used <= 0) or
            np.any(values[7:10] <= 1) or values[10] not in (6, 12)):
        raise ValueError("invalid native archive clock/layout metadata")
    return values


def child_group(handle, name):
    if not isinstance(handle.get(name, getlink=True), h5py.HardLink) or not isinstance(handle[name], h5py.Group):
        raise ValueError(f"missing or nonlocal native plane group: {name}")
    return handle[name]


def grid_xml(parent, label, meta, fields, frame_name, prefix, units):
    shape = tuple(int(n) for n in meta[7:10][::-1])
    if meta[5] > 0:
        shape = tuple(n for axis, n in enumerate(shape) if axis != 3 - meta[5])
    dimensions = " ".join(map(str, shape))
    grid = ET.SubElement(parent, "Grid", Name=label, GridType="Uniform")
    ET.SubElement(grid, "Time", Value=format(meta[2:3].view("<f8")[0], ".17g"))
    ET.SubElement(grid, "Information", Name="units", Value=units)
    ET.SubElement(grid, "Topology", TopologyType=f"{len(shape)}DSMesh", Dimensions=dimensions)
    geometry = ET.SubElement(grid, "Geometry", GeometryType="XYZ")
    item = ET.SubElement(geometry, "DataItem", Dimensions=dimensions + " 3",
                         NumberType="Float", Precision="8", Format="HDF")
    item.text = f"../resources/data.h5:/{prefix}coordinates"
    for name in (*fields, "velocity"):
        vector = name == "velocity"
        attribute = ET.SubElement(grid, "Attribute", Name=name, Center="Node",
                                  AttributeType="Vector" if vector else "Scalar")
        item = ET.SubElement(attribute, "DataItem", Dimensions=dimensions + (" 3" if vector else ""),
                             NumberType="Float", Precision="8", Format="HDF")
        item.text = f"{frame_name}/data.h5:/{prefix}{name}"


def segment_record(segment, *, require_parent=False):
    rows = small_text(segment / "SEGMENT").splitlines()
    version = rows[0] if rows else ""
    if (version not in ("ASTR_OUTPUT_SEGMENT_1", "ASTR_OUTPUT_SEGMENT_2") or
            len(rows) != (3 if version.endswith("_1") else 6) or
            any(len(row.split()) != 2 for row in rows[1:3])):
        raise ValueError("invalid segment provenance declaration")
    step, time = int(rows[1].split()[0]), float(rows[1].split()[1])
    size, crc = rows[2].split()
    if (not 0 <= step < 10**12 or not math.isfinite(time) or time < 0 or
            not re.fullmatch(r"[0-9]+", size) or not re.fullmatch(r"[0-9A-F]{16}", crc)):
        raise ValueError("invalid segment origin/checkpoint identity")
    result = dict(version=version, step=step, time=time, parent=None,
                  fingerprint=fingerprint(segment / "SEGMENT"))
    result["crc_scan_bytes"] = result["fingerprint"][0]
    plain_file(segment / "input.txt")
    if version.endswith("_1"):
        if require_parent:
            raise ValueError("legacy segment has no explicit parent edge; cannot infer a chain")
        return result
    if len(rows[3].split()) != 3 or len(rows[4].split()) != 2:
        raise ValueError("invalid explicit segment parent/input identity")
    parent_id, parent_size, parent_crc = rows[3].split()
    input_size, input_crc = rows[4].split()
    if (not re.fullmatch(r"-1|[0-9]+", parent_id) or not re.fullmatch(r"[0-9]+", parent_size) or
            not re.fullmatch(r"[0-9A-F]{16}", parent_crc) or not re.fullmatch(r"[0-9]+", input_size) or
            not re.fullmatch(r"[0-9A-F]{16}", input_crc) or
            fingerprint(segment / "input.txt") != (int(input_size), int(input_crc, 16))):
        raise ValueError("segment input/parent fingerprint is invalid")
    result["crc_scan_bytes"] += int(input_size)
    number = int(parent_id)
    result["parent_fingerprint"] = (int(parent_size), int(parent_crc, 16))
    if number == -1:
        if rows[5] != "none" or result["parent_fingerprint"] != (0, 0):
            raise ValueError("parentless segment declares a parent reference")
    else:
        path = rows[5]
        if (not 0 <= number <= 99999999 or int(parent_size) <= 0 or int(size) <= 0 or
                not path or Path(path).is_absolute() or ":" in path or
                any(ord(c) < 32 or ord(c) == 127 for c in path) or
                Path(path).name != f"segment{number:08d}"):
            raise ValueError("invalid explicit relative parent path")
        result["parent"] = path
    return result


def validate_frame(frame, geometry_path, geometry_fingerprint, segment_fingerprint=None):
    directory(frame)
    members = entries(frame / "MANIFEST", "ASTR_CHECKPOINT_BUNDLE 1")
    if set(members) != {"data.h5", "data.xdmf", "FRAME", "RESOURCES"}:
        raise ValueError("unsupported archive frame members")
    size, crc = fingerprint(frame / "MANIFEST")
    scan_bytes = size
    if small_text(frame / "COMPLETE") != f"ASTR_COMPLETE_1 {size} {crc:016X}\n":
        raise ValueError("frame completion marker does not bind its manifest")
    for member in frame.iterdir():
        if member.name not in {*members, "MANIFEST", "COMPLETE"}:
            raise ValueError("unexpected files in sealed archive frame")
    for name, expected in members.items():
        if fingerprint(frame / name) != expected:
            raise ValueError(f"archive frame checksum mismatch: {name}")
        scan_bytes += expected[0]
    resources = {"data.h5": geometry_fingerprint}
    if segment_fingerprint is not None:
        resources["SEGMENT"] = segment_fingerprint
    if entries(frame / "RESOURCES", "ASTR_SHARED_RESOURCES 1") != resources:
        raise ValueError("shared archive coordinate fingerprint mismatch")
    rows = small_text(frame / "FRAME").splitlines()
    derived = rows[0] == "ASTR_DERIVED_FRAME_1" if rows else False
    if (len(rows) != (5 if derived else 4) or
            rows[0] not in ("ASTR_BASIC_FRAME_1", "ASTR_DERIVED_FRAME_1") or
            any(len(row.split()) != 2 for row in rows[1:4])):
        raise ValueError("invalid archive FRAME declaration")
    selection = derived_selection(rows[4]) if derived else ()
    step, time = int(rows[1].split()[0]), float(rows[1].split()[1])
    components, units = int(rows[2].split()[0]), rows[2].split()[1]
    byte_counts = tuple(int(n) for n in rows[3].split())
    if (step != int(frame.name[4:]) or not math.isfinite(time) or time < 0 or
            components not in (6, 12) or units not in ("dimensionless", "si") or
            (components == 12 and units != "si") or min(byte_counts) < 0):
        raise ValueError("invalid archive FRAME identity")
    fields = tuple(FIELDS if components == 6 else AIR5_FIELDS)+tuple(DERIVED_NAMES[n-1] for n in selection)
    root = ET.Element("Root")
    with h5py.File(frame / "data.h5", "r") as data, h5py.File(geometry_path, "r") as geometry:
        attributes(data, units)
        attributes(geometry, units)
        if selection:
            text_attribute(data, "derived_layout", " ".join(map(str, selection+(0,)*(14-len(selection)))))
        elif "derived_layout" in data.attrs:
            raise ValueError("basic frame declares derived fields")
        volume = "metadata" in data
        if len(data) > 769:
            raise ValueError("invalid archive plane count")
        labels = [""] if volume else sorted(name for name in data if name != "frame_identity")
        if not labels or len(labels) > 768:
            raise ValueError("invalid archive plane count")
        parent = root
        if not volume:
            identity = native_dataset(data, "frame_identity", (10,), "<i8")[:]
            parent = ET.SubElement(root, "Grid", Name=frame.name, GridType="Collection", CollectionType="Spatial")
            ET.SubElement(parent, "Time", Value=format(time, ".17g"))
        expected_bytes, signature = 0, []
        for label in labels:
            source = data if volume else child_group(data, label)
            coordinates = geometry if volume else child_group(geometry, label)
            meta, geom = metadata(source), metadata(coordinates)
            if meta[1] != step or meta[2:3].view("<f8")[0] != time or meta[10] != components:
                raise ValueError("FRAME/HDF5 clock or components disagree")
            if not np.array_equal(meta[[0, 5, 6, 7, 8, 9, 10]], geom[[0, 5, 6, 7, 8, 9, 10]]):
                raise ValueError("frame/shared geometry layouts disagree")
            axis, index = int(meta[5]), int(meta[6])
            if volume:
                if axis != 0 or index != 0:
                    raise ValueError("invalid volume layout")
            else:
                match = PLANE_NAME.fullmatch(label)
                if (not match or axis != "ijk".index(match[1]) + 1 or index != int(match[2]) or
                        not 0 <= index < meta[6 + axis]):
                    raise ValueError("invalid plane index/layout")
                expected = np.array([*meta[:5], *meta[7:11], int(units == "si")], dtype="<i8")
                if not np.array_equal(identity, expected):
                    raise ValueError("grouped frame identity differs from plane metadata")
            shape = tuple(int(n) for d, n in enumerate(meta[7:10][::-1]) if axis == 0 or d != 3 - axis)
            if (len(source) != len(fields) + 2 or len(coordinates) != 2 or
                    set(source) != {"metadata", "velocity", *fields} or set(coordinates) != {"metadata", "coordinates"}):
                raise ValueError("unsupported native field/coordinate inventory")
            native_dataset(coordinates, "coordinates", shape + (3,), "<f8")
            for name in fields:
                native_dataset(source, name, shape, "<f8")
            for n in selection:
                validate_derived_dataset(source[DERIVED_NAMES[n-1]], n, units)
            native_dataset(source, "velocity", shape + (3,), "<f8")
            expected_bytes += math.prod(shape) * len(fields) * 8
            signature.append((label, tuple(int(n) for n in meta[5:])))
            grid_xml(parent, frame.name if volume else label, meta, fields, frame.name,
                     label + "/" if label else "", units)
        if byte_counts[0] != expected_bytes or byte_counts[1] not in (0, expected_bytes):
            raise ValueError("FRAME field/download byte counts disagree with layout")
    return step, time, (components, units, tuple(signature), selection), root[0], scan_bytes


def repair_segment(segment, *, publish=False, catalog_bytes=1048576):
    """Publish only with an explicit stopped-writer contract; never delete source files."""
    segment = Path(segment).absolute()
    if not re.fullmatch(r"segment[0-9]{8}", segment.name) or catalog_bytes <= 0:
        raise ValueError("invalid segment name or catalog budget")
    for path in (segment, segment.parent, segment.parent / "resources"):
        directory(path)
    record = segment_record(segment)
    origin_step, origin_time = record["step"], record["time"]
    segment_fp = record["fingerprint"] if record["version"].endswith("_2") else None
    for name in INDEX_NAMES:
        optional_plain(segment / name)
    for name in STAGED_NAMES:
        if os.path.lexists(segment / name):
            raise FileExistsError(f"inspect previous repair staging before retry: {segment / name}")
    names, ignored, peak = frame_catalog(segment, catalog_bytes)
    if publish and not names:
        raise ValueError("no sealed frames to publish; leave the empty segment unchanged")
    geometry_path = segment.parent / "resources/data.h5"
    geometry_fingerprint = fingerprint(geometry_path)
    scan_bytes = geometry_fingerprint[0] + record["crc_scan_bytes"]
    xml_stream = ledger_stream = None
    previous_time, signature = -1.0, None
    try:
        if publish:
            ledger_stream = (segment / STAGED_NAMES[0]).open("x", encoding="ascii")
            xml_stream = (segment / STAGED_NAMES[1]).open("xb")
            ledger_stream.write("ASTR_FRAME_SERIES_1\n")
            xml_stream.write(b'<?xml version="1.0"?><Xdmf Version="3.0"><Domain>'
                             b'<Grid Name="series" GridType="Collection" CollectionType="Temporal">\n')
        for name in names:
            step, time, layout, grid, frame_bytes = validate_frame(segment / name, geometry_path, geometry_fingerprint, segment_fp)
            scan_bytes += frame_bytes
            if step < origin_step or time < origin_time:
                raise ValueError("frame clock predates segment origin")
            if time <= previous_time or (signature is not None and layout != signature):
                raise ValueError("nonmonotonic clocks or changing field/plane layout in segment")
            if signature is None:
                validate_units(segment / "input.txt", layout[1])
            if publish:
                ledger_stream.write(f"{step} {time:.17g}\n")
                xml_stream.write(ET.tostring(grid, encoding="utf-8") + b"\n")
            previous_time, signature = time, layout
        if publish:
            xml_stream.write(b"</Grid></Domain></Xdmf>\n")
    finally:
        if ledger_stream is not None:
            ledger_stream.close()
        if xml_stream is not None:
            xml_stream.close()
    if publish:
        for staged, target in zip(STAGED_NAMES, INDEX_NAMES):
            plain_file(segment / staged)
            optional_plain(segment / target)
            os.replace(segment / staged, segment / target)
    return dict(format="ASTR_ARCHIVE_INDEX_REPAIR_1", published=publish, frames=len(names),
                ignored_candidates=ignored, catalog_peak_bytes=peak,
                field_array_read_bytes=0, crc_scan_bytes=scan_bytes,
                geometry_fingerprint_bytes=geometry_fingerprint[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("segment", type=Path)
    parser.add_argument("--publish-stopped-segment", action="store_true",
                        help="explicitly assert that no solver/writer is using this segment; replace its two indexes")
    parser.add_argument("--catalog-bytes", type=int, default=1048576)
    args = parser.parse_args()
    report = repair_segment(args.segment, publish=args.publish_stopped_segment, catalog_bytes=args.catalog_bytes)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
