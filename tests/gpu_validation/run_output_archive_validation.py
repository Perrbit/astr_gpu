"""Bounded real ASTR gate for supported completed-step basic-field archives."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from run_output_restart_validation import archive_schedule_payload, compare_fields, run_case as run_nonreacting_case

FIELD_NAMES = ("density", "velocity_x", "velocity_y", "velocity_z", "pressure", "temperature")
AIR5_FIELDS = (*FIELD_NAMES, "vibrational_temperature", "mass_fraction_N2", "mass_fraction_O2",
               "mass_fraction_N", "mass_fraction_O", "mass_fraction_NO")
PLANES = (("i", 8), ("j", 0), ("k", 16))


def field_names(components):
    if components not in (6, 12):
        raise AssertionError("unknown native field component count")
    return FIELD_NAMES if components == 6 else AIR5_FIELDS


def run_case(args, root, backend, ranks, name, steps, restore=None, enabled=True, **options):
    if args.case not in ("air5hbl", "air5sbli"):
        return run_nonreacting_case(args, root, backend, ranks, name, steps, restore=restore,
                                   enabled=enabled, **options)
    from run_output_air5_restart_validation import launch
    air5 = SimpleNamespace(**vars(args))
    air5.case = "sbli" if args.case == "air5sbli" else "hbl"
    air5.mean_statistics = args.legacy_statistics
    air5.sample_interval = 2
    air5.top_mode = "characteristic"
    air5.reconstruction = 3 if air5.case == "sbli" else 5
    return launch(air5, backend, ranks, name, steps, restore=restore,
                  interval=5 if enabled else 1000000, **options)


def groups(mode, volume=True, slices=True, dt=0.001):
    intervals = ("interval_steps=2", "interval_steps=3") if mode == "steps" else (
        f"interval_time={2*dt:.17e}", f"interval_time={3*dt:.17e}")
    return f"""&volume
 enabled={'.true.' if volume else '.false.'}, mode='{mode}', {intervals[0]},
 initial_frame=.true., final_frame=.true.
/
&slices
 enabled={'.true.' if slices else '.false.'}, mode='{mode}', {intervals[1]},
 initial_frame=.true., final_frame=.true., i_indices=8, j_indices=0, k_indices=16
/
"""


def frames(case, product, segment_name="segment00000000"):
    segment = case / "outdat/new" / product / segment_name
    return {int(path.name[4:]): path for path in segment.glob("step*") if not path.name.endswith(".tmp")}


def equal_array(actual, expected, message):
    if actual.dtype != expected.dtype or actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
        raise AssertionError(message)


def check_series(case, product, segment_name="segment00000000"):
    inventory = frames(case, product, segment_name)
    segment = case / "outdat/new" / product / segment_name
    if not inventory:
        if (segment / "series.xdmf").exists():
            raise AssertionError("empty segment has a time-series index")
        return []
    rows = (segment / "series.frames").read_text().splitlines()
    if rows[0] != "ASTR_FRAME_SERIES_1" or len(rows) != len(inventory) + 1:
        raise AssertionError("series ledger inventory mismatch")
    grids = ET.parse(segment / "series.xdmf").findall("./Domain/Grid/Grid")
    if len(grids) != len(inventory):
        raise AssertionError("time-series frame inventory mismatch")
    times = []
    for row, grid, (step, path) in zip(rows[1:], grids, sorted(inventory.items())):
        if not (path / "COMPLETE").is_file():
            raise AssertionError("series references an incomplete frame")
        tokens = row.split()
        if len(tokens) != 2 or int(tokens[0]) != step or grid.attrib["Name"] != path.name:
            raise AssertionError("time-series ordering mismatch")
        with h5py.File(path / "data.h5") as data:
            identity = data["metadata"] if product == "fields" else data["frame_identity"]
            time = identity[2:3].view(np.float64)[0]
        if float(tokens[1]) != time or float(grid.find("Time").attrib["Value"]) != time:
            raise AssertionError("series ledger/XML/frame clocks differ")
        if times and time <= times[-1]:
            raise AssertionError("series time is not strictly increasing")
        times.append(float(time))
        for item in grid.findall(".//DataItem"):
            filename, dataset = item.text.strip().split(":", 1)
            if not filename.startswith((path.name + "/", "../resources/")):
                raise AssertionError("series references another frame")
            with h5py.File(segment / filename) as data:
                if dataset not in data:
                    raise AssertionError("series dataset reference missing")
    if list(segment.glob("*.tmp")):
        raise AssertionError("staged series metadata leaked after successful publication")
    return times


def check_series_reader(case, product, segment_name="segment00000000", extra_names=()):
    source = case / "outdat/new" / product
    target_name = product if segment_name == "segment00000000" else product + "_" + segment_name
    target = case / "series_reader" / target_name
    segment = target / segment_name
    segment.mkdir(parents=True, exist_ok=False)
    (target / "resources").mkdir()
    shutil.copyfile(source / "resources/data.h5", target / "resources/data.h5")
    for filename in ("series.frames", "series.xdmf"):
        shutil.copyfile(source / segment_name / filename, segment / filename)
    with h5py.File(target / "resources/data.h5") as geometry:
        for step, path in sorted(frames(case, product, segment_name).items()):
            frame = segment / path.name
            frame.mkdir()
            shutil.copyfile(path / "data.h5", frame / "data.h5")
            labels = [""] if product == "fields" else [f"{axis}{index:012d}" for axis, index in PLANES]
            references = {"field_names": np.array(FIELD_NAMES)}
            if product == "slices":
                references["labels"] = np.array(labels)
            with h5py.File(path / "data.h5") as data:
                identity = data["metadata"] if product == "fields" else data["frame_identity"]
                references["time"] = identity[2:3].view(np.float64)[0]
                names = (*field_names(int(identity[-1] if product == "fields" else identity[-2])), *extra_names)
                references["field_names"] = np.array(names)
                for label in labels:
                    leaf = data[label] if label else data
                    coords = geometry[label] if label else geometry
                    prefix = label + "_" if label else ""
                    references[prefix + "coordinates"] = coords["coordinates"][:].reshape(-1, 3)
                    for name in names:
                        references[prefix + name] = leaf[name][:].ravel()
                    references[prefix + "velocity"] = leaf["velocity"][:].reshape(-1, 3)
            if sum(np.asarray(value).nbytes for value in references.values()) > 2 * 1024**2:
                raise RuntimeError("series readback reference exceeds 2 MiB")
            np.savez(frame / "reader_reference.npz", **references)
            del references
    command = ["/usr/bin/pvpython", "--no-mpi", str(Path(__file__).with_name("run_checkpoint_export_validation.py")),
               "--native-series-check", str(segment)]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
    (segment / "reader.log").write_text(result.stdout)
    if result.returncode or "Error" in result.stdout or "Traceback" in result.stdout:
        raise RuntimeError(f"native series ParaView readback failed: {segment / 'reader.log'}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def check_frame(path, product, checkpoint, geometry, backend, extra_names=()):
    """Compare written physical-node fields to authoritative same-phase cached state."""
    if not (path / "COMPLETE").is_file() or len(list(path.glob("*.h5"))) != 1:
        raise AssertionError("frame was not published as one sealed HDF5 product")
    shared_path = path.parents[1] / "resources/data.h5"
    with h5py.File(path / "data.h5") as data, h5py.File(checkpoint / "state.h5") as state, \
            h5py.File(geometry) as grid, h5py.File(shared_path) as shared:
        labels = [None] if product == "fields" else [f"{axis}{index:012d}" for axis, index in PLANES]
        role = int(state["identity"][12])
        names, first_cache = (AIR5_FIELDS, 23) if role == 7 else (FIELD_NAMES, 6)
        if product == "slices" and set(data) != set(labels) | {"frame_identity"}:
            raise AssertionError("grouped slice inventory mismatch")
        for position, label in enumerate(labels):
            selected = [slice(None)] * 3
            if label is not None:
                axis, index = PLANES[position]
                selected[2 - "ijk".index(axis)] = index
            selected = tuple(selected)
            leaf = data if label is None else data[label]
            coordinates_leaf = shared if label is None else shared[label]
            if "coordinates" in leaf or set(coordinates_leaf) != {"coordinates", "metadata"}:
                raise AssertionError("frame repeated coordinates or resource contains flow fields")
            identity = leaf["metadata"][:]
            equal_array(identity[1:5], state["identity"][8:12], "frame/checkpoint clock mismatch")
            for component, name in enumerate(names, first_cache):
                equal_array(leaf[name][:], state[f"q{component:04d}"][selected], f"field mismatch: {label}/{name}")
            coordinates = np.stack([grid[f"q{c:04d}"][selected] for c in (1, 2, 3)], axis=-1)
            velocity = np.stack([leaf["velocity_" + axis][:] for axis in "xyz"], axis=-1)
            equal_array(coordinates_leaf["coordinates"][:], coordinates, "physical-coordinate mismatch")
            equal_array(leaf["velocity"][:], velocity, "velocity-vector mismatch")
        xml = ET.parse(path / "data.xdmf")
        uniform = xml.findall(".//Grid[@GridType='Uniform']")
        if len(uniform) != len(labels):
            raise AssertionError("XDMF plane inventory mismatch")
        if any(not (path / item.text.strip().split(":")[0]).is_file() for item in xml.findall(".//DataItem")):
            raise AssertionError("XDMF references missing data")
    counts = (17**3 if product == "fields" else 3 * 17**2) * (len(names)+len(extra_names)) * 8
    record = (path / "FRAME").read_text().splitlines()
    if list(map(int, record[3].split())) != [counts, counts if backend == "gpu" else 0]:
        raise AssertionError("actual download/field payload differs from selected physical nodes")
    return counts


def check_accounting(case, backend, np_ranks):
    rows = re.findall(r"ASTR_OUTPUT_ARCHIVE rank=(\d+) product=(fields|slices) field_bytes=(\d+) download_bytes=(\d+)"
                      r"host_peak_bytes=(\d+) device_peak_bytes=(\d+)", (case / "run.log").read_text())
    if not rows:
        raise AssertionError("missing archive buffer/transfer accounting")
    totals = {}
    for rank, product, fields, download, host, device in rows:
        rank, fields, download, host, device = map(int, (rank, fields, download, host, device))
        if rank >= np_ranks or not 0 < host <= 4096 or device > 4096:
            raise AssertionError("archive workspace/index budget exceeded")
        if download != (fields if backend == "gpu" else 0):
            raise AssertionError("downloaded more than selected fields")
        totals[product] = totals.get(product, 0) + download
    return dict(downloaded_field_bytes=totals, max_host_tile_bytes=max(int(r[4]) for r in rows),
                max_device_tile_bytes=max(int(r[5]) for r in rows))


def check_reader(path, product, checkpoint, geometry):
    readback = checkpoint.parents[3] / "reader" / product / "segment00000000" / "step000000000012"
    readback.mkdir(parents=True, exist_ok=False)
    for filename in ("data.h5", "data.xdmf"):
        shutil.copyfile(path / filename, readback / filename)
    resources = readback.parents[1] / "resources"
    resources.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(path.parents[1] / "resources/data.h5", resources / "data.h5")
    references = {"field_names": np.array(FIELD_NAMES)}
    labels = [""] if product == "fields" else [f"{axis}{index:012d}" for axis, index in PLANES]
    if product == "slices":
        references["labels"] = np.array(labels)
    with h5py.File(checkpoint / "state.h5") as state, h5py.File(geometry) as grid:
        role = int(state["identity"][12])
        names, first_cache = (AIR5_FIELDS, 23) if role == 7 else (FIELD_NAMES, 6)
        references["field_names"] = np.array(names)
        references["time"] = state["identity"][9:10].view(np.float64)[0]
        for position, label in enumerate(labels):
            selected = [slice(None)] * 3
            if product == "slices":
                axis, index = PLANES[position]
                selected[2 - "ijk".index(axis)] = index
            selected = tuple(selected)
            prefix = label + "_" if label else ""
            references[prefix + "coordinates"] = np.column_stack([
                grid[f"q{component:04d}"][selected].ravel() for component in (1, 2, 3)])
            for component, name in enumerate(names, first_cache):
                references[prefix + name] = state[f"q{component:04d}"][selected].ravel()
            references[prefix + "velocity"] = np.column_stack([
                references[prefix + "velocity_" + axis] for axis in "xyz"])
    if sum(np.asarray(value).nbytes for value in references.values()) > 2 * 1024**2:
        raise RuntimeError("readback reference exceeds 2 MiB")
    np.savez(readback / "reader_reference.npz", **references)
    command = ["/usr/bin/pvpython", "--no-mpi", str(Path(__file__).with_name("run_checkpoint_export_validation.py")),
               "--native-frame-check", str(readback)]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
    (readback / "reader.log").write_text(result.stdout)
    if result.returncode or "Error" in result.stdout or "Traceback" in result.stdout:
        raise RuntimeError(f"native ParaView readback failed: {readback / 'reader.log'}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--mpiexec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", choices=("cpu", "gpu"), nargs="+", default=["cpu", "gpu"])
    parser.add_argument("--ranks", type=int, choices=(1, 2), nargs="+", default=[1, 2])
    parser.add_argument("--axis", choices=("x", "y", "z"), default="x")
    parser.add_argument("--mode", choices=("steps", "time"), default="steps")
    parser.add_argument("--case", choices=("tgv", "channel", "curve", "dynamic", "air5hbl", "air5sbli"), default="tgv")
    parser.add_argument("--force", choices=("feedback", "fixed", "frozen"), default="feedback")
    parser.add_argument("--legacy-statistics", action="store_true")
    parser.add_argument("--filter-workspace", choices=("scalar", "full"), default="scalar")
    parser.add_argument("--inflow-count", type=int, default=12)
    parser.add_argument("--statistics", action="store_true")
    parser.add_argument("--readback", action="store_true")
    parser.add_argument("--initial-restart", action="store_true")
    parser.add_argument("--schedule-checks", action="store_true")
    parser.add_argument("--budget-checks", action="store_true")
    parser.add_argument("--keep", type=int, choices=(1, 2), default=2)
    parsed = parser.parse_args()
    if parsed.case != "tgv" and parsed.statistics:
        parser.error("formal TGV statistics require --case tgv")
    if parsed.statistics and parsed.legacy_statistics:
        parser.error("formal and legacy statistics are separate acceptance paths")
    if parsed.legacy_statistics and parsed.case in ("tgv", "channel") and "gpu" in parsed.backends:
        parser.error("GPU legacy statistics require CURVE or AIR5")
    if parsed.case.startswith("air5") and (parsed.schedule_checks or parsed.budget_checks):
        parser.error("AIR5 first archive gate does not include schedule/budget rejections")
    if parsed.inflow_count < 12 or parsed.inflow_count > 256:
        parser.error("bounded 12-step gate requires 12:256 inflow frames, not a production limit")
    if parsed.initial_restart and parsed.keep != 2:
        parser.error("step-0 source preparation requires keep=2")
    if parsed.schedule_checks and (parsed.initial_restart or parsed.mode != "steps"):
        parser.error("schedule overrides gate uses the step-5 steps-mode source")
    if parsed.budget_checks and (parsed.statistics or parsed.legacy_statistics):
        parser.error("small packing-budget positive gate requires statistics off")
    parsed.output = parsed.output.resolve()
    parsed.output.mkdir(parents=True, exist_ok=False)
    args = SimpleNamespace(**vars(parsed), restart_step=5, initial_dimension=0, no_samples=True)
    dt = {"tgv": 0.001, "channel": 0.001, "curve": 1e-5, "dynamic": 6e-6,
          "air5hbl": 1e-10, "air5sbli": 1e-10}[args.case]
    args.executable = args.executable.resolve(strict=True)
    args.mpiexec = args.mpiexec.resolve(strict=True)
    root = Path(__file__).resolve().parents[2]
    report = dict(status="running", checks=[])
    try:
        for backend in args.backends:
            for ranks in args.ranks:
                archive = groups(args.mode, dt=dt)
                continuous, a_size = run_case(args, root, backend, ranks, "continuous", 12,
                                             archive_groups=archive, buffer_bytes=4096, checkpoint_keep=args.keep)
                first, b_size = run_case(args, root, backend, ranks, "first", 1 if args.initial_restart else 5,
                                        archive_groups=archive, buffer_bytes=4096, checkpoint_keep=args.keep)
                restart_step = 0 if args.initial_restart else 5
                source = first / f"outdat/new/checkpoints/step{restart_step:012d}"
                resumed, c_size = run_case(args, root, backend, ranks, "resumed", 12, restore=source,
                                          archive_groups=archive, buffer_bytes=4096, checkpoint_keep=args.keep)
                baseline, d_size = run_case(args, root, backend, ranks, "baseline", 12, buffer_bytes=4096,
                                           checkpoint_keep=args.keep)
                final = "outdat/new/checkpoints/step000000000012"
                for other in (resumed, baseline):
                    compare_fields(continuous / final / "state.h5", other / final / "state.h5")
                    if args.statistics or args.legacy_statistics:
                        compare_fields(continuous / final / "statistics.h5", other / final / "statistics.h5")
                    if args.case == "dynamic":
                        compare_fields(continuous / final / "inflow.h5", other / final / "inflow.h5")
                controls = ["control.bin", "archives.bin"]
                if args.case.startswith("air5"):
                    controls.append("air5_config.bin")
                for name in controls:
                    read = archive_schedule_payload if name == "archives.bin" else Path.read_bytes
                    if read(continuous / final / name) != read(resumed / final / name):
                        raise AssertionError("restart schedule/control mismatch: " + name)
                product_info = {}
                for product, stride in (("fields", 2), ("slices", 3)):
                    cf, rf = frames(continuous, product), frames(resumed, product)
                    expected = {0, *range(stride, 13, stride), 12}
                    if args.mode == "time":
                        times = np.cumsum(np.full(12, dt, dtype=np.float64))
                        targets = np.arange(1, 14, dtype=np.float64) * (stride * dt)
                        crossed = (times[:, None] >= targets[None, :]).sum(axis=1)
                        expected = {0, 12, *[index + 1 for index in range(12)
                                             if crossed[index] > (crossed[index-1] if index else 0)]}
                    if set(cf) != expected:
                        raise AssertionError(f"unexpected {product} schedule: {sorted(cf)}")
                    if set(rf) != {step for step in cf if step > restart_step}:
                        raise AssertionError("restored point was repeated or archive history was lost")
                    for step in rf:
                        compare_fields(cf[step] / "data.h5", rf[step] / "data.h5")
                    payload = check_frame(cf[12], product, continuous / final,
                                          continuous / "outdat/new/resources/geometry.h5", backend)
                    if not all((path / "COMPLETE").exists() for path in cf.values()):
                        raise AssertionError("checkpoint retention removed a field archive")
                    product_info[product] = dict(continuous_steps=sorted(cf), resumed_steps=sorted(rf),
                                                 final_field_bytes=payload,
                                                 continuous_times=check_series(continuous, product),
                                                 resumed_times=check_series(resumed, product))
                    if args.readback:
                        product_info[product]["reader"] = check_reader(cf[12], product, continuous / final,
                            continuous / "outdat/new/resources/geometry.h5")
                        product_info[product]["series_reader"] = check_series_reader(continuous, product)
                standalone, e_size = run_case(args, root, backend, ranks, "slices_only", 5, enabled=False,
                                             archive_groups=groups(args.mode, volume=False, dt=dt), buffer_bytes=4096)
                # Formal statistics keep checkpointing active in the shared test runner.
                if args.case == "tgv" and not args.statistics and not args.legacy_statistics:
                    if any((standalone / "outdat/new" / name).exists() for name in ("resources", "checkpoints", "fields")):
                        raise AssertionError("slice-only path forced checkpoint/full-volume resources")
                sf = frames(standalone, "slices")
                common = set(sf) & set(frames(continuous, "slices"))
                for step in common:
                    compare_fields(sf[step] / "data.h5", frames(continuous, "slices")[step] / "data.h5")
                if not args.initial_restart:
                    compare_fields(sf[5] / "data.h5", frames(first, "slices")[5] / "data.h5")
                for case in (continuous, first, resumed, standalone):
                    accounting = check_accounting(case, backend, ranks)
                    for product in ("fields", "slices"):
                        if (case / "outdat/new" / product).exists():
                            check_series(case, product)
                report["checks"].append(dict(backend=backend, np=ranks, axis=args.axis, mode=args.mode, case=args.case,
                                             exact_state_restart=True, output_switch_unchanged=True,
                                             restart_step=restart_step,
                                             checkpoint_keep=args.keep,
                                             exact_statistics=args.statistics or args.legacy_statistics, products=product_info,
                                             slice_only_accounting=accounting,
                                             directory_bytes=[a_size, b_size, c_size, d_size, e_size]))
                if args.keep == 1 and set(p.name for p in (continuous / "outdat/new/checkpoints").glob("step*")) != {
                        "step000000000012"}:
                    raise AssertionError("keep=1 did not retire old checkpoints")
                if args.budget_checks and backend == "gpu":
                    small, _ = run_case(args, root, backend, ranks, "small_device", 5, enabled=False,
                                        archive_groups=groups(args.mode, volume=False, dt=dt), buffer_bytes=4096,
                                        device_budget_bytes=512)
                    compare_fields(frames(small, "slices")[5] / "data.h5", sf[5] / "data.h5")
                    small_counts = check_accounting(small, backend, ranks)
                    if small_counts["max_device_tile_bytes"] > 512:
                        raise AssertionError("device planner exceeded the reduced budget")
                    run_case(args, root, backend, ranks, "reject_device", 5, enabled=False,
                             archive_groups=groups(args.mode, volume=False, dt=dt), buffer_bytes=4096,
                             device_budget_bytes=47, reject="device budget cannot hold one basic output node")
                    report["checks"][-1]["reduced_device_budget"] = small_counts
                if args.schedule_checks:
                    changed = archive.replace("interval_steps=2", "interval_steps=4")
                    run_case(args, root, backend, ranks, "reject_changed", 12, restore=source,
                             archive_groups=changed, buffer_bytes=4096,
                             reject="product options changed without override")
                    overridden, _ = run_case(args, root, backend, ranks, "override", 12, restore=source,
                                             archive_groups=changed, buffer_bytes=4096, override=True)
                    if set(frames(overridden, "fields")) != {9, 12} or set(frames(overridden, "slices")) != {6, 9, 12}:
                        raise AssertionError("override reset unchanged schedules or repeated the restored point")
                    compare_fields(overridden / final / "state.h5", continuous / final / "state.h5")
                    if args.statistics or args.legacy_statistics:
                        compare_fields(overridden / final / "statistics.h5", continuous / final / "statistics.h5")
                    stopped, _ = run_case(args, root, backend, ranks, "no_advance", 5, restore=source,
                                          archive_groups=archive, buffer_bytes=4096)
                    if frames(stopped, "fields") or frames(stopped, "slices") or list((stopped / "outdat/new").rglob("COMPLETE")):
                        raise AssertionError("zero-advance restore duplicated a completed-step product")
                    for product in ("fields", "slices"):
                        check_series(overridden, product)
                        check_series(stopped, product)
                    derived_args = SimpleNamespace(**(vars(args) | {"case": "channel"}))
                    run_case(derived_args, root, backend, ranks, "reject_derived", 12,
                             archive_groups=archive.replace("initial_frame=.true.", "qcriterion=.true., initial_frame=.true."),
                             buffer_bytes=4096, lfilter=False,
                             reject="derived fields require registered 16-cubed explicit TGV/channel/CURVE/AIR5 NP<=2")
                    run_case(args, root, backend, ranks, "reject_buffer", 12, enabled=False,
                             archive_groups=groups(args.mode, volume=False, dt=dt), buffer_bytes=1,
                             reject="host budget cannot hold one basic node")
                    report["checks"][-1]["schedule_override_and_rejections"] = "passed"
                sizes = [sum(path.stat().st_size for path in case.rglob("*") if path.is_file())
                         for case in (continuous, first, resumed, baseline, standalone)]
                if max(sizes) > 64 * 1024**2:
                    raise RuntimeError("archive/readback case exceeds approved 64 MiB directory budget")
                report["checks"][-1]["directory_bytes_with_readback"] = sizes
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
