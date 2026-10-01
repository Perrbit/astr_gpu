"""Export existing bounded checkpoints and verify real ParaView point values."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def reader_check(directory):
    import paraview
    from paraview import servermanager
    from paraview.simple import Xdmf3ReaderS
    from vtk.util.numpy_support import vtk_to_numpy
    def leaves(data):
        if data is None:
            return []
        if data.IsA("vtkMultiBlockDataSet"):
            return [leaf for i in range(data.GetNumberOfBlocks()) for leaf in leaves(data.GetBlock(i))]
        if data.IsA("vtkMultiPieceDataSet"):
            return [leaf for i in range(data.GetNumberOfPieces()) for leaf in leaves(data.GetPiece(i))]
        return [data]
    with np.load(directory / "reference.npz") as reference:
        for product in ("volume", "slices"):
            reader = Xdmf3ReaderS(FileName=[str(directory / "export" / (product + ".xdmf"))])
            reader.UpdatePipeline()
            if np.atleast_1d(reader.TimestepValues).tolist() != [float(reference["time"])]:
                raise AssertionError("ParaView time identity mismatch")
            data = servermanager.Fetch(reader)
            blocks = leaves(data)
            labels = ["volume"] if product == "volume" else list(reference["slice_labels"])
            if len(blocks) != len(labels):
                raise AssertionError(f"ParaView block inventory mismatch: {product}: {len(blocks)} versus {len(labels)}")
            for block, label in zip(blocks, labels):
                points = vtk_to_numpy(block.GetPoints().GetData())
                if not np.array_equal(points, reference[label + "_coordinates"]):
                    raise AssertionError(f"ParaView coordinate/axis mismatch: {label}")
                for name in [*reference["field_names"], "velocity"]:
                    actual = vtk_to_numpy(block.GetPointData().GetArray(name))
                    expected = reference[label + "_" + name]
                    if actual.dtype != np.float64 or actual.tobytes() != expected.tobytes():
                        raise AssertionError(f"ParaView field mismatch: {label}/{name}")
    print(json.dumps(dict(status="passed", paraview_version=paraview.__version__)))


def native_frame_reader_check(directory, series=False, lineage=False):
    """Read a bounded writer-probe frame; this is not checkpoint validation."""
    import paraview
    from paraview import servermanager
    from paraview.simple import Xdmf3ReaderS
    from vtk.util.numpy_support import vtk_to_numpy
    series = series or lineage
    stem = "lineage" if lineage else "series"
    reader = Xdmf3ReaderS(FileName=[str(directory / (stem + ".xdmf" if series else "data.xdmf"))])
    reader.UpdatePipeline()
    def leaves(data):
        if data is None:
            return []
        if data.IsA("vtkMultiBlockDataSet"):
            return [leaf for i in range(data.GetNumberOfBlocks()) for leaf in leaves(data.GetBlock(i))]
        if data.IsA("vtkMultiPieceDataSet"):
            return [leaf for i in range(data.GetNumberOfPieces()) for leaf in leaves(data.GetPiece(i))]
        return [data]
    if series:
        rows = (directory / (stem + ".frames")).read_text().splitlines()[1:]
        expected_times = [float(row.split()[1]) for row in rows]
        reference_paths = [directory / f"step{int(row.split()[0]):012d}" / "reader_reference.npz" for row in rows]
    else:
        reference_paths = [directory / "reader_reference.npz"]
        with np.load(reference_paths[0]) as reference:
            expected_times = [float(reference["time"])]
    if np.atleast_1d(reader.TimestepValues).tolist() != expected_times:
        raise AssertionError("native output time sequence mismatch")
    for time, reference_path in zip(expected_times, reference_paths):
        reader.UpdatePipeline(time=time)
        blocks = leaves(servermanager.Fetch(reader))
        with np.load(reference_path) as reference:
            if time != float(reference["time"]):
                raise AssertionError("native frame time mismatch")
            prefixes = [str(label) + "_" for label in reference["labels"]] if "labels" in reference else [""]
            if len(blocks) != len(prefixes):
                raise AssertionError("native frame block inventory mismatch")
            for data, prefix in zip(blocks, prefixes):
                points = vtk_to_numpy(data.GetPoints().GetData())
                if points.dtype != np.float64 or points.tobytes() != reference[prefix + "coordinates"].tobytes():
                    raise AssertionError("native frame coordinates mismatch")
                for name in [*reference["field_names"], "velocity"]:
                    actual = vtk_to_numpy(data.GetPointData().GetArray(name))
                    expected = reference[prefix + name]
                    if actual.dtype != np.float64 or actual.tobytes() != expected.tobytes():
                        raise AssertionError(f"native frame field mismatch: {prefix}{name}")
    print(json.dumps(dict(status="passed", frames=len(expected_times), paraview_version=paraview.__version__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader-check", type=Path)
    parser.add_argument("--native-frame-check", type=Path)
    parser.add_argument("--native-series-check", type=Path)
    parser.add_argument("--native-lineage-check", type=Path)
    parser.add_argument("--checkpoint", type=Path, action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pvpython", type=Path, default=Path("/usr/bin/pvpython"))
    parser.add_argument("--units", choices=("dimensionless", "si"), default="dimensionless")
    args = parser.parse_args()
    if args.native_lineage_check:
        native_frame_reader_check(args.native_lineage_check, lineage=True)
        return
    if args.native_series_check:
        native_frame_reader_check(args.native_series_check, series=True)
        return
    if args.native_frame_check:
        native_frame_reader_check(args.native_frame_check)
        return
    if args.reader_check:
        reader_check(args.reader_check)
        return
    if not args.checkpoint or args.output is None:
        parser.error("checkpoint(s) and a new output directory are required")
    import h5py
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("checkpoint_export", root / "scripts/output/export_checkpoint.py")
    export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", checks=[])
    try:
        for index, source in enumerate(args.checkpoint):
            source = source.resolve(strict=True)
            case = args.output / f"case{index:02d}"
            case.mkdir()
            before = {p: (export.fingerprint(p), p.stat().st_mtime_ns)
                      for p in source.parent.parent.rglob("*") if p.is_file()}
            planes = [("i", 8), ("j", 0), ("k", 8)]
            result = export.export_checkpoint(source, case / "export", slices=planes, buffer_bytes=4096, units=args.units)
            references = {"time": np.array(result["time"]),
                          "field_names": np.array(result["fields"]),
                          "slice_labels": np.array([f"{axis}{number:012d}" for axis, number in planes])}
            with h5py.File(source / "state.h5") as state, h5py.File(source.parent.parent / "resources/geometry.h5") as grid:
                _, _, fields = export.field_layout(state)
                selections = [("volume", (slice(None),)*3)]
                for axis, number in planes:
                    selected = [slice(None)]*3
                    selected[2-"ijk".index(axis)] = number
                    selections.append((f"{axis}{number:012d}", tuple(selected)))
                for label, selected in selections:
                    references[label + "_coordinates"] = np.column_stack([
                        grid[f"q{component:04d}"][selected].ravel() for component in (1, 2, 3)])
                    for name, component in fields.items():
                        references[label + "_" + name] = state[f"q{component:04d}"][selected].ravel()
                    references[label + "_velocity"] = np.column_stack([
                        references[label + "_velocity_" + axis] for axis in "xyz"])
            controlled_reference_bytes = sum(v.nbytes for v in references.values())
            if controlled_reference_bytes > 2*1024**2:
                raise RuntimeError("small readback reference exceeds its 2 MiB array budget")
            np.savez(case / "reference.npz", **references)
            process = subprocess.run([str(args.pvpython), "--no-mpi", str(Path(__file__).resolve()),
                                      "--reader-check", str(case.resolve())],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
            (case / "paraview.log").write_text(process.stdout)
            if process.returncode or "Error" in process.stdout or "Traceback" in process.stdout:
                raise RuntimeError(f"ParaView readback failed: {case / 'paraview.log'}")
            reader = json.loads(process.stdout.strip().splitlines()[-1])
            slice_result = export.export_checkpoint(source, case / "slices_only", volume=False,
                                                     slices=planes, buffer_bytes=4096, units=args.units)
            with h5py.File(case / "export/slices.h5") as all_output, h5py.File(case / "slices_only/slices.h5") as only:
                for group in all_output:
                    for field in all_output[group]:
                        if all_output[group][field][:].tobytes() != only[group][field][:].tobytes():
                            raise AssertionError("slice-only data differs from combined export")
            if not all((export.fingerprint(p), p.stat().st_mtime_ns) == original for p, original in before.items()):
                raise AssertionError("export or reader modified its source checkpoint/resources")
            disk_bytes = sum(p.stat().st_size for p in case.rglob("*") if p.is_file())
            if disk_bytes > 64*1024**2:
                raise RuntimeError("readback test directory exceeds 64 MiB")
            report["checks"].append(dict(source=str(source.relative_to(root)), export=result,
                                          slice_only=slice_result, reader=reader,
                                          reference_array_bytes=controlled_reference_bytes, directory_bytes=disk_bytes,
                                          source_unchanged=True))
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
