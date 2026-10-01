"""Read-only basic-field export from completed-step ASTR checkpoint bundles."""
import argparse
import json
from pathlib import Path
import re
import stat
import xml.etree.ElementTree as ET

import h5py
import numpy as np


MAGIC = 0x4153545251533031
FIELDS = {"density": 6, "velocity_x": 7, "velocity_y": 8,
          "velocity_z": 9, "pressure": 10, "temperature": 11}
AIR5_SPECIES = ("N2", "O2", "N", "O", "NO")
AIR5_FIELDS = {"density": 23, "velocity_x": 24, "velocity_y": 25,
               "velocity_z": 26, "pressure": 27, "temperature": 28,
               "vibrational_temperature": 29,
               **{f"mass_fraction_{name}": 30 + index for index, name in enumerate(AIR5_SPECIES)}}
COORDINATES = {"x": 1, "y": 2, "z": 3}
MASK = (1 << 64) - 1


def crc_table():
    table = []
    for byte in range(256):
        value = byte << 56
        for _ in range(8):
            value = ((value << 1) ^ (0x42F0E1EBA9EA3693 if value >> 63 else 0)) & MASK
        table.append(value)
    return table


CRC_TABLE = crc_table()


def plain_file(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f"not a private regular file: {path}")
    return info


def fingerprint(path):
    info = plain_file(path)
    size, crc = 0, 0
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(65536), b""):
            size += len(data)
            for byte in data:
                crc = ((crc << 8) & MASK) ^ CRC_TABLE[(crc >> 56) ^ byte]
    if size != info.st_size:
        raise ValueError(f"file size changed during validation: {path}")
    return size, crc


def small_text(path):
    if plain_file(path).st_size > 16384:
        raise ValueError(f"oversized bundle metadata: {path}")
    return path.read_text(encoding="ascii")


def entries(path, magic):
    lines = small_text(path).splitlines()
    if len(lines) < 2 or lines[0] != magic or not re.fullmatch(r"[1-9][0-9]*", lines[1]):
        raise ValueError(f"invalid bundle metadata: {path}")
    count = int(lines[1])
    if count > 64 or len(lines) != count + 2:
        raise ValueError(f"invalid bundle member count: {path}")
    result = {}
    for line in lines[2:]:
        match = re.fullmatch(r"([A-Za-z0-9_-][A-Za-z0-9_.-]{0,127}) ([0-9]+) ([0-9A-F]{16})", line)
        if not match:
            raise ValueError(f"invalid bundle entry: {path}")
        name, size, crc = match.groups()
        if name in result or name in {"MANIFEST", "COMPLETE"} or str(int(size)) != size:
            raise ValueError(f"duplicate or reserved bundle entry: {name}")
        result[name] = (int(size), int(crc, 16))
    return result


def validate_bundle(batch):
    # Keep the same relative resource contract as the native restart reader.
    for directory in (batch, batch.parent, batch.parent.parent, batch.parent.parent / "resources"):
        if not stat.S_ISDIR(directory.lstat().st_mode):
            raise ValueError(f"not a plain bundle directory: {directory}")
    marker = small_text(batch / "COMPLETE")
    size, crc = fingerprint(batch / "MANIFEST")
    if marker != f"ASTR_COMPLETE_1 {size} {crc:016X}\n":
        raise ValueError("checkpoint completion marker does not bind its manifest")
    members = entries(batch / "MANIFEST", "ASTR_CHECKPOINT_BUNDLE 1")
    if not {"state.h5", "control.bin", "RESOURCES"}.issubset(members):
        raise ValueError("checkpoint lacks mandatory members")
    for name, expected in members.items():
        if fingerprint(batch / name) != expected:
            raise ValueError(f"checkpoint checksum mismatch: {name}")
    resources = entries(batch / "RESOURCES", "ASTR_SHARED_RESOURCES 1")
    if not {"geometry.h5", "input.txt"}.issubset(resources):
        raise ValueError("checkpoint lacks geometry/input resources")
    resource_root = batch.parent.parent / "resources"
    for name, expected in resources.items():
        if fingerprint(resource_root / name) != expected:
            raise ValueError(f"resource checksum mismatch: {name}")
    return resource_root


def validate_units(input_path, units):
    records = []
    for line in small_text(input_path).splitlines():
        data = line.split("#", 1)[0].split("!", 1)[0].strip()
        if data:
            records.append(data)
        if len(records) == 4:
            break
    if len(records) != 4:
        raise ValueError("cannot identify the frozen input nondimen flag")
    flag = re.split(r"[,\s]+", records[3], maxsplit=1)[0].lower().strip(".")
    if flag not in {"t", "true", "f", "false"}:
        raise ValueError("invalid frozen input nondimen flag")
    if (flag in {"t", "true"}) != (units == "dimensionless"):
        raise ValueError("export unit declaration disagrees with the frozen input")


def native_dataset(handle, name, shape, dtype):
    if not isinstance(handle.get(name, getlink=True), h5py.HardLink):
        raise ValueError(f"missing or nonlocal HDF5 dataset: {name}")
    data = handle[name]
    if not isinstance(data, h5py.Dataset) or data.shape != shape or data.dtype != np.dtype(dtype):
        raise ValueError(f"invalid HDF5 dataset layout: {name}")
    if data.is_virtual or data.external:
        raise ValueError(f"external HDF5 storage is not allowed: {name}")
    return data


def identity(handle, components, role):
    header = native_dataset(handle, "identity", (13,), "<i8")[:]
    if tuple(header[:3]) != (MAGIC, 2, 1) or header[6] != components or header[12] != role:
        raise ValueError("unsupported state version, phase or component role")
    if np.any(header[3:6] <= 1) or header[7] <= 0 or header[8] < 0:
        raise ValueError("invalid state dimensions or step")
    times = header[9:12].view("<f8")
    if not np.isfinite(times).all() or times[0] < 0 or times[1] < 0 or times[2] <= 0:
        raise ValueError("invalid state clock")
    if header[8] > 0 and times[1] <= 0:
        raise ValueError("missing completed-step dt")
    shape = tuple(int(v) for v in header[3:6][::-1])
    for component in range(1, components + 1):
        native_dataset(handle, f"q{component:04d}", shape, "<f8")
    native_dataset(handle, "partitions", (8 * int(header[7]),), "<i8")
    return header, shape


def field_layout(handle):
    header = native_dataset(handle, "identity", (13,), "<i8")[:]
    if header[12] == 2:
        header, shape = identity(handle, 11, 2)
        return header, shape, FIELDS
    if header[12] == 7:
        header, shape = identity(handle, 34, 7)
        metadata = native_dataset(handle, "metadata", (2,), "<i8")[:]
        if metadata[0] != 1 or metadata[1] not in (0, 1):
            raise ValueError("invalid AIR5 state metadata")
        return header, shape, AIR5_FIELDS
    raise ValueError("unsupported state component role")


def selections(shape, buffer_bytes):
    # One FP64 array and its finite mask coexist. HDF5 internals are not included.
    capacity = buffer_bytes // 9
    if capacity < 1:
        raise ValueError("buffer must hold one FP64 value and its finite mask")
    block = [1] * len(shape)
    for axis in reversed(range(len(shape))):
        block[axis] = min(shape[axis], capacity)
        capacity //= block[axis]
    for origin in np.ndindex(*(int((n + b - 1) // b) for n, b in zip(shape, block))):
        yield tuple(slice(o*b, min((o+1)*b, n)) for o, b, n in zip(origin, block, shape))


def copy_field(source, target, shape, buffer_bytes, fixed=None, vector=None, component=None, target_component=None):
    peak = read_bytes = 0
    for selected in selections(shape, buffer_bytes):
        source_selection = list(selected)
        if fixed is not None:
            source_selection.insert(*fixed)
        values = source[tuple(source_selection)]
        finite = np.isfinite(values)
        peak = max(peak, values.nbytes + finite.nbytes)
        if not finite.all():
            raise ValueError(f"nonfinite physical field: {source.name}")
        target[selected if target_component is None else selected+(target_component,)] = values
        if vector is not None:
            vector[selected + (component,)] = values
        read_bytes += values.nbytes
        del values, finite
    return peak, read_bytes


def xdmf_grid(parent, label, shape, prefix, file_name, time, fields):
    grid = ET.SubElement(parent, "Grid", Name=label, GridType="Uniform")
    ET.SubElement(grid, "Time", Value=format(time, ".17g"))
    dimensions = " ".join(map(str, shape))
    ET.SubElement(grid, "Topology", TopologyType=f"{len(shape)}DSMesh", Dimensions=dimensions)
    geometry = ET.SubElement(grid, "Geometry", GeometryType="XYZ")
    node = ET.SubElement(geometry, "DataItem", Dimensions=dimensions + " 3",
                         NumberType="Float", Precision="8", Format="HDF")
    node.text = f"{file_name}:/{prefix}coordinates"
    for name in fields:
        attribute = ET.SubElement(grid, "Attribute", Name=name, AttributeType="Scalar", Center="Node")
        node = ET.SubElement(attribute, "DataItem", Dimensions=dimensions,
                             NumberType="Float", Precision="8", Format="HDF")
        node.text = f"{file_name}:/{prefix}{name}"
    attribute = ET.SubElement(grid, "Attribute", Name="velocity", AttributeType="Vector", Center="Node")
    node = ET.SubElement(attribute, "DataItem", Dimensions=dimensions + " 3",
                         NumberType="Float", Precision="8", Format="HDF")
    node.text = f"{file_name}:/{prefix}velocity"


def write_xml(path, shape, time, groups=None, fields=FIELDS):
    root = ET.Element("Xdmf", Version="3.0")
    domain = ET.SubElement(root, "Domain")
    temporal = ET.SubElement(domain, "Grid", Name="series", GridType="Collection", CollectionType="Temporal")
    if groups is None:
        xdmf_grid(temporal, "volume", shape, "", "volume.h5", time, fields)
    else:
        collection = ET.SubElement(temporal, "Grid", Name="slices", GridType="Collection", CollectionType="Spatial")
        ET.SubElement(collection, "Time", Value=format(time, ".17g"))
        for label, plane_shape in groups:
            xdmf_grid(collection, label, plane_shape, label + "/", "slices.h5", time, fields)
    ET.indent(root)
    with path.open("xb") as stream:
        ET.ElementTree(root).write(stream, encoding="utf-8", xml_declaration=True)


def export_checkpoint(batch, output, *, volume=True, slices=(), buffer_bytes=1048576, units="dimensionless"):
    batch, output = Path(batch).absolute(), Path(output).absolute()
    if buffer_bytes < 9 or units not in {"dimensionless", "si"}:
        raise ValueError("invalid explicit export buffer or unit system")
    if not volume and not slices:
        raise ValueError("select a volume or at least one slice")
    resources = validate_bundle(batch)
    validate_units(resources / "input.txt", units)
    if output.resolve().is_relative_to(batch.parent.parent.resolve()):
        raise ValueError("export output must be outside its source run")
    # No solver imports, state reconstruction, gradient calls or source writes.
    with h5py.File(batch / "state.h5", "r") as state, h5py.File(resources / "geometry.h5", "r") as geometry:
        header, shape, fields = field_layout(state)
        if int(header[12]) == 7 and units != "si":
            raise ValueError("AIR5 export currently requires dimensional SI input")
        grid_header, grid_shape = identity(geometry, 13, 1)
        if shape != grid_shape or header[7] != grid_header[7]:
            raise ValueError("field/geometry layout mismatch")
        requested = sorted(set(slices))
        for axis, index in requested:
            if axis not in "ijk" or len(axis) != 1 or not 0 <= index < shape[2-"ijk".index(axis)]:
                raise ValueError("slice index lies outside the global node grid")
        time, dt_used, dt_next = header[9:12].view("<f8")
        output.mkdir(parents=True, exist_ok=False)
        report = dict(format="ASTR_BASIC_EXPORT_1", complete_step=int(header[8]), time=float(time),
                      dt_used=float(dt_used), dt_next=float(dt_next), units=units, center="Node",
                      axis_order="k,j,i; i varies fastest", source_phase="completed_step",
                      source_manifest=(batch / "COMPLETE").read_text().strip(),
                      buffer_bytes=buffer_bytes, peak_controlled_array_bytes=0, field_read_bytes=0,
                      checksum_buffer_bytes=65536, third_party_memory_included=False,
                      source_role=int(header[12]), fields=list(fields),
                      species_order=list(AIR5_SPECIES) if int(header[12]) == 7 else [],
                      restartable=False, volume=volume, slices=[list(item) for item in requested])
        inputs = [(geometry, name, component) for name, component in COORDINATES.items()]
        inputs += [(state, name, component) for name, component in fields.items()]
        products = [("volume.h5", [("", shape, None)])] if volume else []
        plane_groups = []
        if requested:
            planes = []
            for axis, index in requested:
                numpy_axis = 2 - "ijk".index(axis)
                plane_shape = shape[:numpy_axis] + shape[numpy_axis+1:]
                label = f"{axis}{index:012d}"
                planes.append((label, plane_shape, (numpy_axis, index)))
                plane_groups.append((label, plane_shape))
            products.append(("slices.h5", planes))
        for filename, groups in products:
            with h5py.File(output / filename, "x") as target:
                target.attrs.update(complete_step=report["complete_step"], time=time,
                                    units=units, axis_order=report["axis_order"], center="Node")
                for label, product_shape, fixed in groups:
                    group = target.create_group(label) if label else target
                    # Materialize the vector for readers whose 2D JOIN path writes scratch data.
                    vector = group.create_dataset("velocity", shape=product_shape+(3,), dtype="<f8", chunks=True)
                    coordinates = group.create_dataset("coordinates", shape=product_shape+(3,), dtype="<f8", chunks=True)
                    for source, name, component in inputs:
                        coordinate_index = component - 1 if name in COORDINATES else None
                        if coordinate_index is not None:
                            dataset = coordinates
                        else:
                            dataset = group.create_dataset(name, shape=product_shape, dtype="<f8", chunks=True)
                        vector_index = "xyz".index(name[-1]) if name.startswith("velocity_") else None
                        peak, size = copy_field(source[f"q{component:04d}"], dataset,
                                                product_shape, buffer_bytes, fixed,
                                                vector if vector_index is not None else None, vector_index, coordinate_index)
                        report["peak_controlled_array_bytes"] = max(report["peak_controlled_array_bytes"], peak)
                        report["field_read_bytes"] += size
        if volume:
            write_xml(output / "volume.xdmf", shape, float(time), fields=fields)
        if requested:
            write_xml(output / "slices.xdmf", shape, float(time), plane_groups, fields=fields)
        # A failed export leaves no completion record and never replaces an older export.
        report["data_file_bytes"] = sum(p.stat().st_size for p in output.iterdir())
        completion = output / "COMPLETE.json.tmp"
        with completion.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        completion.rename(output / "COMPLETE.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--units", required=True, choices=("dimensionless", "si"),
                        help="Explicit solver unit system; no conversion is performed")
    parser.add_argument("--slices-only", action="store_true")
    parser.add_argument("--slice", action="append", default=[], metavar="i:INDEX")
    parser.add_argument("--buffer-bytes", type=int, default=1048576)
    args = parser.parse_args()
    slices = []
    for item in args.slice:
        match = re.fullmatch(r"([ijk]):([0-9]+)", item)
        if not match:
            parser.error("slices must have the form i:INDEX, j:INDEX or k:INDEX")
        slices.append((match[1], int(match[2])))
    result = export_checkpoint(args.checkpoint, args.output, volume=not args.slices_only,
                               slices=slices, buffer_bytes=args.buffer_bytes, units=args.units)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
