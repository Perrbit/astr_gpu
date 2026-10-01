"""Bounded read-only export and asymmetric-axis checks; no flow integration."""
import importlib.util
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("checkpoint_export", ROOT / "scripts/output/export_checkpoint.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def seal(path, names, magic):
    lines = [magic, str(len(names))]
    for name in names:
        size, crc = export.fingerprint(path / name)
        lines.append(f"{name} {size} {crc:016X}")
    return "\n".join(lines) + "\n"


def reseal(batch):
    names = ["state.h5", "control.bin", "RESOURCES"]
    (batch / "MANIFEST").write_text(seal(batch, names, "ASTR_CHECKPOINT_BUNDLE 1"))
    size, crc = export.fingerprint(batch / "MANIFEST")
    (batch / "COMPLETE").write_text(f"ASTR_COMPLETE_1 {size} {crc:016X}\n")


@pytest.fixture
def bundle(tmp_path):
    resource = tmp_path / "run/resources"
    batch = tmp_path / "run/checkpoints/step000000000012"
    resource.mkdir(parents=True)
    batch.mkdir(parents=True)
    k, j, i = np.indices((5, 7, 9))
    encoding = 100*i + 10*j + k
    for path, components, role in [(batch / "state.h5", 11, 2), (resource / "geometry.h5", 13, 1)]:
        times = np.array([0.012, 0.001, 0.001], dtype="<f8")
        if role == 1:
            times = np.array([0., 0., 1.], dtype="<f8")
        with h5py.File(path, "w") as handle:
            handle["identity"] = np.array([export.MAGIC, 2, 1, 9, 7, 5, components, 1,
                                            12 if role == 2 else 0, *times.view("<i8"), role], dtype="<i8")
            handle["partitions"] = np.array([0, 0, 0, 8, 6, 4, 0, 0], dtype="<i8")
            handle["rank_extras"] = np.empty(0, dtype="<f8")
            for component in range(1, components + 1):
                handle[f"q{component:04d}"] = (encoding + component/16).astype("<f8")
    (resource / "input.txt").write_text("synthetic\n8,6,4\nt,t,t\nt,f,f,f,f,f,f,f\n")
    (batch / "control.bin").write_bytes(b"synthetic control is not executed\n")
    (batch / "RESOURCES").write_text(seal(resource, ["geometry.h5", "input.txt"], "ASTR_SHARED_RESOURCES 1"))
    reseal(batch)
    return batch


def test_crc_ecma_reference():
    crc = 0
    for byte in b"123456789":
        crc = ((crc << 8) & export.MASK) ^ export.CRC_TABLE[(crc >> 56) ^ byte]
    assert crc == 0x6C40DF5F0B497347


def test_volume_and_slices_exact_readback(bundle, tmp_path):
    original = {p: (export.fingerprint(p), p.stat().st_mtime_ns) for p in bundle.parent.parent.rglob("*") if p.is_file()}
    output = tmp_path / "export"
    report = export.export_checkpoint(bundle, output, slices=[("i", 4), ("j", 2), ("k", 1), ("k", 1)], buffer_bytes=113)
    assert report["complete_step"] == 12 and report["time"] == 0.012
    assert report["peak_controlled_array_bytes"] <= 113
    assert report["slices"] == [["i", 4], ["j", 2], ["k", 1]]
    with h5py.File(bundle / "state.h5") as source, h5py.File(output / "volume.h5") as volume, h5py.File(output / "slices.h5") as planes:
        for name, component in export.FIELDS.items():
            expected = source[f"q{component:04d}"][:]
            assert np.array_equal(volume[name][:], expected)
            assert np.array_equal(planes["i000000000004"][name][:], expected[:, :, 4])
            assert np.array_equal(planes["j000000000002"][name][:], expected[:, 2, :])
            assert np.array_equal(planes["k000000000001"][name][:], expected[1, :, :])
        assert volume["density"].shape == (5, 7, 9)
    xml = ET.parse(output / "volume.xdmf")
    assert xml.find(".//Topology").attrib == {"TopologyType": "3DSMesh", "Dimensions": "5 7 9"}
    assert xml.find(".//Geometry").attrib["GeometryType"] == "XYZ"
    assert xml.find(".//Attribute[@Name='velocity']/DataItem").attrib["Dimensions"] == "5 7 9 3"
    assert all((export.fingerprint(p), p.stat().st_mtime_ns) == before for p, before in original.items())
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) <= 4*1024**2
    with pytest.raises(FileExistsError):
        export.export_checkpoint(bundle, output)


def test_slice_only_is_bounded_and_independent(bundle, tmp_path):
    output = tmp_path / "planes"
    report = export.export_checkpoint(bundle, output, volume=False, slices=[("i", 8)], buffer_bytes=18)
    assert report["field_read_bytes"] == 5*7*9*8  # 3 coordinates plus 6 basic fields.
    assert report["peak_controlled_array_bytes"] == 18
    assert not (output / "volume.h5").exists()
    assert (output / "COMPLETE.json").is_file()
    moved = tmp_path / "moved"
    shutil.copytree(bundle.parent.parent, moved)
    export.export_checkpoint(moved / "checkpoints" / bundle.name, tmp_path / "moved_export", volume=False, slices=[("j", 0)])


@pytest.fixture
def air5_bundle(bundle):
    with h5py.File(bundle / "state.h5", "r+") as handle:
        encoding = handle["q0001"][:] - 1/16
        handle["identity"][6] = 34
        handle["identity"][12] = 7
        handle["metadata"] = np.array([1, 1], dtype="<i8")
        for component in range(12, 35):
            handle[f"q{component:04d}"] = (encoding + component/16).astype("<f8")
    resource = bundle.parent.parent / "resources"
    (resource / "input.txt").write_text("synthetic\n8,6,4\nt,t,t\nf,f,f,f,f,f,f,f\n")
    (bundle / "RESOURCES").write_text(seal(resource, ["geometry.h5", "input.txt"], "ASTR_SHARED_RESOURCES 1"))
    reseal(bundle)
    return bundle


@pytest.mark.parametrize("volume", [True, False])
def test_air5_cached_fields_exact_readback(air5_bundle, tmp_path, volume):
    original = {p: (export.fingerprint(p), p.stat().st_mtime_ns)
                for p in air5_bundle.parent.parent.rglob("*") if p.is_file()}
    output = tmp_path / "air5_export"
    planes = [("i", 4), ("j", 2), ("k", 1)]
    report = export.export_checkpoint(air5_bundle, output, volume=volume, slices=planes,
                                      buffer_bytes=113, units="si")
    assert report["fields"] == list(export.AIR5_FIELDS)
    assert report["species_order"] == ["N2", "O2", "N", "O", "NO"]
    assert report["source_role"] == 7 and report["peak_controlled_array_bytes"] <= 113
    shape = (5, 7, 9)
    nodes = int(np.prod(shape)) if volume else 0
    nodes += 5*7 + 5*9 + 7*9
    assert report["field_read_bytes"] == nodes * (3+12)*8
    with h5py.File(air5_bundle / "state.h5") as source, h5py.File(output / "slices.h5") as slices:
        for name, component in export.AIR5_FIELDS.items():
            expected = source[f"q{component:04d}"][:]
            assert np.array_equal(slices["i000000000004"][name][:], expected[:, :, 4])
            assert np.array_equal(slices["j000000000002"][name][:], expected[:, 2, :])
            assert np.array_equal(slices["k000000000001"][name][:], expected[1, :, :])
        expected_velocity = np.stack([source[f"q{component:04d}"][:] for component in (24, 25, 26)], axis=-1)
        assert np.array_equal(slices["i000000000004/velocity"][:], expected_velocity[:, :, 4])
        if volume:
            with h5py.File(output / "volume.h5") as fields:
                for name, component in export.AIR5_FIELDS.items():
                    assert np.array_equal(fields[name][:], source[f"q{component:04d}"][:])
                assert np.array_equal(fields["velocity"][:], expected_velocity)
    xml = ET.parse(output / "slices.xdmf")
    names = [a.attrib["Name"] for a in xml.findall(".//Grid[@Name='i000000000004']/Attribute")]
    assert names == list(export.AIR5_FIELDS) + ["velocity"]
    assert all((export.fingerprint(p), p.stat().st_mtime_ns) == before for p, before in original.items())
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) <= 4*1024**2


@pytest.mark.parametrize("defect", ["metadata", "components", "nan", "dimensionless"])
def test_invalid_air5_source_rejected(air5_bundle, tmp_path, defect):
    with h5py.File(air5_bundle / "state.h5", "r+") as source:
        if defect == "metadata":
            source["metadata"][1] = 2
        elif defect == "components":
            source["identity"][6] = 33
        elif defect == "nan":
            source["q0029"][0, 0, 0] = np.nan
    units = "si"
    if defect == "dimensionless":
        resource = air5_bundle.parent.parent / "resources"
        (resource / "input.txt").write_text("synthetic\n8,6,4\nt,t,t\nt,f,f,f,f,f,f,f\n")
        (air5_bundle / "RESOURCES").write_text(seal(resource, ["geometry.h5", "input.txt"], "ASTR_SHARED_RESOURCES 1"))
        units = "dimensionless"
    reseal(air5_bundle)
    output = tmp_path / "invalid_air5"
    with pytest.raises(ValueError):
        export.export_checkpoint(air5_bundle, output, units=units)
    assert not (output / "COMPLETE.json").exists()


@pytest.mark.parametrize("defect", ["crc", "resource", "missing", "phase", "role", "fp32", "nan", "symlink", "external"])
def test_invalid_source_rejected(bundle, tmp_path, defect):
    if defect in {"crc", "resource"}:
        path = bundle / "state.h5" if defect == "crc" else bundle.parent.parent / "resources/input.txt"
        with path.open("ab") as stream:
            stream.write(b"corrupt")
    elif defect == "missing":
        (bundle / "COMPLETE").unlink()
    elif defect == "symlink":
        data = bundle / "state.h5"
        data.rename(bundle / "real_state.h5")
        data.symlink_to("real_state.h5")
    else:
        with h5py.File(bundle / "state.h5", "r+") as handle:
            if defect == "phase":
                handle["identity"][2] = 0
            elif defect == "role":
                handle["identity"][12] = 1
            elif defect == "nan":
                handle["q0006"][0, 0, 0] = np.nan
            else:
                data = handle["q0006"][:]
                del handle["q0006"]
                if defect == "fp32":
                    handle["q0006"] = data.astype("<f4")
                else:
                    handle["q0006"] = h5py.ExternalLink("other.h5", "data")
        reseal(bundle)
    output = tmp_path / "bad_export"
    with pytest.raises((ValueError, FileNotFoundError)):
        export.export_checkpoint(bundle, output, buffer_bytes=1024)
    assert not (output / "COMPLETE.json").exists()


@pytest.mark.parametrize("slice", [("i", 9), ("j", -1), ("k", 5), ("x", 0), ("ij", 0)])
def test_slice_bounds_rejected(bundle, tmp_path, slice):
    with pytest.raises(ValueError):
        export.export_checkpoint(bundle, tmp_path / "bad_export", slices=[slice])
    assert not (tmp_path / "bad_export").exists()


def test_unit_mismatch_and_source_tree_output_rejected(bundle, tmp_path):
    with pytest.raises(ValueError, match="unit declaration"):
        export.export_checkpoint(bundle, tmp_path / "wrong_units", units="si")
    with pytest.raises(ValueError, match="outside its source run"):
        export.export_checkpoint(bundle, bundle / "nested_export")
    assert not (tmp_path / "wrong_units").exists()
    assert not (bundle / "nested_export").exists()


def test_failed_completion_is_not_published(bundle, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("injected completion write failure")
    monkeypatch.setattr(export.json, "dump", fail)
    output = tmp_path / "failed_export"
    with pytest.raises(OSError, match="injected completion"):
        export.export_checkpoint(bundle, output, volume=False, slices=[("i", 4)])
    assert not (output / "COMPLETE.json").exists()
