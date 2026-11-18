"""Compare AIR5 wall fields, clipped wall moments and exact continuation."""
import argparse
import csv
import json
import os
import re
from math import prod
from pathlib import Path

import h5py
import numpy as np

from run_air5_filter_fields import command
from run_cpp_air5_output import FIELDS, archive_frame, checkpoint, statistics
from run_cpp_air5_strang import prepare
from run_output_restart_validation import compare_fields
from test_insitu_air5_walls import FIELD_SCALES
from test_insitu_wall_scalar_statistics import read_statistics
from test_insitu_air5_volume_statistics import read_statistics as read_volume, SCALAR_SCALES


DT = 2.5e-10
WINDOW = (0.5 * DT, 3.5 * DT)
WALL_FIELDS = {"wall_pressure": 6, "wall_shear_x": 12, "wall_heat_into_gas": 16}
ATOL = 2.e-10


def wall_frames(work, field, step):
    paths = list((work / "outdat/new/wall_products" / field).rglob(f"frame.step{step:012d}.csv"))
    assert len(paths) == 1, (work, field, step, paths)
    return paths[0]


def wall_oracle(work, rank, step, legacy=False):
    path = work / f"outdat/sample.air5_wall.step{step:08d}.rank{rank:08d}.bin"
    with path.open("rb") as stream:
        assert stream.read(8) == b"ASTRAW01"
        # The frozen NVHPC oracle omitted jg0/kg0 from its mixed-kind constructor.
        header = np.fromfile(stream, "<i4", 7 if legacy else 9)
        clock = np.fromfile(stream, "<f8", 2)
        shape = tuple(header[3:6])
        nodes = prod(shape)
        xyz = np.fromfile(stream, "<f8", nodes * 3).reshape((*shape, 3), order="F")
        values = np.fromfile(stream, "<f8", nodes * 18).reshape((*shape, 18), order="F")
        owned = np.fromfile(stream, "<i4", nodes).reshape(shape, order="F")
        assert not stream.read(1)
    assert tuple(header[:3]) == (1, step, rank)
    np.testing.assert_allclose(clock, [step*DT, DT if step else 0], atol=1e-25, rtol=0)
    assert np.isfinite(values).all() and np.isfinite(xyz).all()
    return header, xyz, values, owned


def trapezoid_weights(times):
    integrated_weights = np.zeros(len(times), dtype=np.longdouble)
    duration = 0.
    for n, (t0, t1) in enumerate(zip(times[:-1], times[1:])):
        a, b = max(t0, WINDOW[0]), min(t1, WINDOW[1])
        if b <= a:
            continue
        position = 0.5*((a-t0)/(t1-t0)) + 0.5*((b-t0)/(t1-t0))
        left, right = (b-a)*(1-position), (b-a)*position
        integrated_weights[n] += left
        integrated_weights[n+1] += right
        duration += left
        duration += right
    return integrated_weights / integrated_weights.sum(), duration


def oracle_moments(fields, times):
    weights, duration = trapezoid_weights(times)
    values = fields.astype(np.longdouble)
    mean = np.tensordot(weights, values, axes=(0, 0))
    variance = np.tensordot(weights, (values-mean)**2, axes=(0, 0))
    return mean, variance, duration


def check_separation(oracle, continuous, resumed, ranks, length):
    worst_x, worst_shear = 0., 0.
    for step in range(5):
        rel = f"outdat/render/sample.wall_separation.step{step:08d}.csv"
        lines = (continuous / rel).read_text().splitlines()
        ref_u, span_period, pairs = map(float, lines[1].split()[-3:])
        assert ref_u > 0 and pairs == 0
        np.testing.assert_allclose(span_period, length, atol=2.e-18, rtol=0)
        rows = list(csv.DictReader(line for line in lines if not line.startswith("#")))
        assert len(rows) == 17 and all(
            row["record"] == "profile" and row["status"] == "eligible_positive_x_inflow" for row in rows
        ), rows
        np.testing.assert_array_equal([int(row["step"]) for row in rows], step)
        np.testing.assert_array_equal([float(row["time"]) for row in rows], step*DT)
        points = np.empty((17, 16, 3))
        owned_nodes = np.zeros((17, 16), dtype=int)
        for rank in range(ranks):
            header, xyz, fields, owned = wall_oracle(oracle, rank, step)
            for i, k, w in np.argwhere(owned):
                g = (i+header[6], k+header[8])
                points[g] = xyz[i, k, w, 0], xyz[i, k, w, 2], fields[i, k, w, 12]
                owned_nodes[g] += 1
        assert np.all(owned_nodes == 1)
        z = points[0, :, 1]
        weights = .5*(np.r_[z[1:], z[0]+length] - np.r_[z[-1]-length, z[:-1]])
        np.testing.assert_allclose(weights.sum(), length, rtol=0, atol=2.e-18)
        expected_x = points[..., 0] @ weights / length
        expected_shear = points[..., 2] @ weights / length
        assert np.all(expected_shear > 0), expected_shear
        actual_x = [float(row["x_lower"]) for row in rows]
        actual_shear = [float(row["value"]) for row in rows]
        worst_x = max(worst_x, float(np.max(np.abs(actual_x-expected_x))))
        worst_shear = max(worst_shear, float(np.max(np.abs(actual_shear-expected_shear))))
        if step > 2:
            assert (continuous / rel).read_bytes() == (resumed / rel).read_bytes()
    assert worst_x <= ATOL and worst_shear <= ATOL, (worst_x, worst_shear)
    return dict(x_max_abs=worst_x, shear_max_abs=worst_shear, eligible=True, complete_pairs=0,
                exact_continuation=True, frames=5)


def check_volume(oracle, full, resumed, ranks, lengths, reduction):
    ids = (0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11)
    records = []
    for step in range(5):
        with h5py.File(archive_frame(oracle, "fields", step)) as frame:
            records.append(np.stack([frame[FIELDS[i]][:-1].T for i in ids], axis=-1))
    fields = np.array(records, dtype=np.longdouble)
    wt, duration = trapezoid_weights(np.arange(5)*DT)
    density, velocity, scalars = fields[..., 0], fields[..., 1:4], fields[..., 4:]
    def avg(values):
        return np.tensordot(wt, values, axes=(0, 0))
    mean_r, mean_density, mean_s = avg(velocity), avg(density), avg(scalars)
    mean_f = avg(density[..., None]*velocity)/mean_density[..., None]
    dr, df = velocity-mean_r, velocity-mean_f
    cr = avg(dr[..., :, None]*dr[..., None, :])
    density_stress = avg(density[..., None, None]*df[..., :, None]*df[..., None, :])
    cf = density_stress/mean_density[..., None, None]
    var_s = avg((scalars-mean_s)**2)
    ownership = np.zeros((17, 17, 16), dtype=int)
    measures = np.empty_like(ownership, dtype=np.float64)
    coords = np.empty((*ownership.shape, 3))
    errors = dict(velocity_statistics=0., scalar_mean=0., scalar_variance=0.)
    for rank in range(ranks):
        name = f"outdat/render/sample.air5_statistics.step00000004.rank{rank:08d}.bin"
        header, clocks, count, _, xyz, weights, a = read_volume(full / name)
        assert (full / name).read_bytes() == (resumed / name).read_bytes()
        assert int(header[9]) == int(reduction) and count == 5
        np.testing.assert_array_equal(clocks[:4], [4*DT, *WINDOW, duration])
        np.testing.assert_allclose(clocks[4], prod(lengths), rtol=0, atol=2.e-18)
        region = tuple(slice(origin, origin+n) for origin, n in zip(header[6:9], header[3:6]))
        ownership[region] += 1
        measures[region], coords[region] = weights, xyz
        np.testing.assert_array_equal(a[..., 0], duration)
        for start, end, target in ((1, 2, mean_density[..., None]), (2, 5, mean_r), (5, 8, mean_f),
                                   (14, 23, cr), (23, 32, cf), (32, 41, density_stress)):
            actual = a[..., start:end]
            if start >= 14:
                actual = actual.reshape((*actual.shape[:3], 3, 3), order="F")
            errors["velocity_statistics"] = max(errors["velocity_statistics"],
                                                 float(np.max(np.abs(actual-target[region]))))
        for start, target in ((8, cr), (11, cf)):
            difference = a[..., start:start+3]**2-np.diagonal(target[region], axis1=-2, axis2=-1)
            errors["velocity_statistics"] = max(errors["velocity_statistics"], float(np.max(np.abs(difference))))
        errors["scalar_mean"] = max(errors["scalar_mean"], float(
            np.max(np.abs(a[..., 41:48]-mean_s[region])/SCALAR_SCALES)))
        errors["scalar_variance"] = max(errors["scalar_variance"], float(
            np.max(np.abs(a[..., 48:55]-var_s[region])/SCALAR_SCALES**2)))
    assert np.all(ownership == 1)
    wx, wy, wz = (coords[:, 0, 0, 0], coords[0, :, 0, 1], coords[0, 0, :, 2])
    def cell_width(line, period=None):
        left = line[0] if period is None else line[-1]-period
        right = line[-1] if period is None else line[0]+period
        return .5*(np.r_[line[1:], right]-np.r_[left, line[:-1]])
    expected = cell_width(wx)[:, None, None]*cell_width(wy)[None, :, None]*cell_width(wz, lengths[2])[None, None, :]
    np.testing.assert_allclose(measures, expected, atol=2.e-21, rtol=0)
    np.testing.assert_allclose(measures.sum(), prod(lengths), atol=2.e-18, rtol=0)
    filename = "outdat/render/sample.air5_volume_rms.step00000004.csv"
    if reduction:
        path = full / filename
        assert path.read_bytes() == (resumed / filename).read_bytes()
        with path.open() as stream:
            rows = list(csv.DictReader(line for line in stream if not line.startswith("#")))
        assert [row["component"] for row in rows] == ["u", "v", "w"]
        spatial = measures / measures.sum()
        signal = np.einsum("ijk,tijkc->tc", spatial, velocity)
        signal_var = avg((signal-avg(signal))**2)
        averaged_var = np.einsum("ijk,ijkc->c", spatial, np.diagonal(cr, axis1=-2, axis2=-1))
        for key, value in (("regional_signal_variance", signal_var), ("local_variance_volume_mean", averaged_var)):
            errors[key] = float(np.max(np.abs([float(row[key]) for row in rows]-value)))
    else:
        assert not (full / filename).exists()
    assert max(errors.values()) <= ATOL, errors
    return dict(nodes=int(ownership.size), physical_volume=float(measures.sum()), errors=errors, reduction=reduction)


def render_products(lengths):
    lx, ly, lz = lengths
    wall_bounds = [edge for length in lengths for edge in (0., 1.01*length)]
    outputs = dict(channel="image", formats=["jpeg"], image_size=[800, 600], jpeg_quality=95, dpi=300)
    definitions = []
    for name, field, source, limits in (
            ("air5-pressure-chart", "wall_pressure", "instantaneous", [19990., 20040.]),
            ("air5-mean-shear-chart", "wall_shear_x", "reynolds", [612., 615.])):
        item = dict(id=name, kind="wall_curve", enable_request=1, field=field, source=source,
                    sampling=dict(mode="wall", axis="j", side="lower", spanwise_index=4, bounds=wall_bounds),
                    display=dict(y_range=limits, line_width=2.), output=outputs, schedule=dict(mode="steps", interval=1))
        if source == "reynolds":
            item["statistics_source"] = "wall-time"
        definitions.append(item)
    for name, field, limits in (("air5-tv-slice", "vibrational_temperature", [1490., 3010.]),
                                ("air5-no-slice", "mass_fraction_NO", [0., 2.e-20])):
        sampling = dict(mode="grid_index", axis="k", index=4, index_ranges=dict(i=[0, 16], j=[0, 16]))
        definitions.append(dict(id=name, kind="slice", enable_request=1, field=field, source="instantaneous",
                                sampling=sampling, output=outputs, schedule=dict(mode="steps", interval=2),
                                display=dict(camera=[lx/2, 1.1*ly, lz/4+lx], focal_point=[lx/2, 1.1*ly, lz/4],
                                             view_up=[0., 1., 0.], projection="parallel", parallel_scale=.45*lx,
                                             color_range=limits, color_map="diverging_blue_red", lighting=False)))
        definitions.append(dict(id=name+"-data", kind="slice", enable_request=1, fields=[field], source="instantaneous",
                                sampling=sampling, output=dict(channel="data", format="hdf5-xdmf"),
                                schedule=dict(mode="steps", interval=2)))
    return definitions


def check_rendering(full, resumed, definitions, pipeline, topology, transport):
    from PIL import Image
    images = {}
    ranks = prod(topology)
    for item in definitions:
        if item["output"]["channel"] != "image":
            continue
        name, field, source = item["id"], item["field"], item["source"]
        raw = item["kind"] == "slice"
        def segment(work):
            segments = list((work / "outdat/new/render_products" / name).glob("version*/segment*"))
            assert len(segments) == 1, (work, name, segments)
            return segments[0]
        a, b = segment(full), segment(resumed)
        checks = []
        for step in ([0, 2, 4] if raw else ([1, 2, 3, 4] if source == "reynolds" else list(range(5)))):
            stem = f"step{step:012d}"
            frame = a / ("frame."+stem+".jpeg")
            with Image.open(frame) as image:
                pixels = np.asarray(image.convert("RGB"))
            assert pixels.shape == (600, 800, 3) and pixels.std() > 1., frame
            receipts = [json.loads(p.read_text()) for p in a.glob(f"frame.{stem}.rank*.json")]
            assert len(receipts) == (ranks if raw else 1), frame
            assert all(r["step"] == step and r["time"] == step*DT and
                       r["rendering_pipeline"] == pipeline for r in receipts)
            ranges = [r["product"].get("actual_range") for r in receipts]
            if raw:
                assert all(r["field_d2h_bytes"] == r["geometry_host_bytes"] == r["geometry_d2h_bytes"] == 0
                           for r in receipts)
                assert all((value is not None) == (r["product"]["local_points"] > 0)
                           for value, r in zip(ranges, receipts))
                assert not any(r["product"].get("outside_fixed_color_range", False) for r in receipts)
                with h5py.File(archive_frame(full, "slice_products", step, name+"-data")) as data:
                    values = data[field][...]
            else:
                product = field if source == "instantaneous" else field+"_mean"
                with wall_frames(full, product, step).open() as stream:
                    values = np.array([float(row["value"]) for row in
                                       csv.DictReader(line for line in stream if not line.startswith("#"))])
                assert len(values) == receipts[0]["product"]["points"] == 17
                assert not receipts[0]["nondimensional"] and receipts[0]["chart_backend"] == "existing-host-matplotlib"
                if source == "reynolds":
                    assert receipts[0]["statistics"]["provider"] == "wall-time"
            active_ranges = [value for value in ranges if value is not None]
            np.testing.assert_array_equal([min(r[0] for r in active_ranges), max(r[1] for r in active_ranges)],
                                          [values.min(), values.max()])
            if step > 2:
                with Image.open(b / frame.name) as image:
                    np.testing.assert_array_equal(pixels, np.asarray(image.convert("RGB")))
            checks.append(step)
        assert not list(a.glob("*.eps")) and not list(b.glob("*.eps"))
        images[name] = checks
    log = (full / "run.log").read_text()
    usage = re.findall(r"ASTR_INSITU_CPP_WALL_STATS rank=\d+ staging_d2h_bytes=(\d+) staging_h2d_bytes=(\d+) "
                       r"status_d2h_bytes=\d+ field_download_bytes=0 face_exchange_bytes=(\d+)", log)
    assert len(usage) == ranks*5, usage
    d2h, h2d, exchanged = np.array(usage, dtype=int).sum(axis=0)
    if transport == "device-aware":
        assert d2h == h2d == 0
    if topology[2] > 1:
        assert exchanged > 0
    return dict(frames=images, wall_face_exchange_bytes=int(exchanged),
                wall_face_staged_d2h_bytes=int(d2h), wall_face_staged_h2d_bytes=int(h2d))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "reference", "mpiexec", "input", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--topology", nargs=3, type=int, default=(2, 1, 1))
    parser.add_argument("--mpi-btl", default="self,vader,tcp")
    parser.add_argument("--mpi-osc", default="pt2pt")
    parser.add_argument("--reference-mpiexec", type=Path)
    parser.add_argument("--separation", action="store_true")
    parser.add_argument("--volume-statistics", action="store_true")
    parser.add_argument("--no-volume-reduction", action="store_true")
    parser.add_argument("--images-implementation", type=Path)
    parser.add_argument("--pipeline", choices=("standard-device", "direct-device"), default="standard-device")
    parser.add_argument("--postprocess-transport", choices=("pinned", "device-aware"), default="pinned")
    args = parser.parse_args()
    if args.no_volume_reduction and not args.volume_statistics:
        parser.error("--no-volume-reduction requires --volume-statistics")
    root = args.output.resolve()
    root.mkdir()
    lengths = list(map(float, (args.input / "air5_hbl_domain.dat").read_text().splitlines()[1].split()))
    bounds = [edge for length in lengths for edge in (0., 1.01*length)]
    images = render_products(lengths) if args.images_implementation else []
    nranks = prod(args.topology)
    env = os.environ | dict(OMPI_MCA_pml="ob1", OMPI_MCA_btl=args.mpi_btl, OMPI_MCA_osc=args.mpi_osc,
                            OMPI_MCA_coll="^cuda,hcoll,ucc", OMPI_MCA_opal_cuda_support="0",
                            ASTR_GPU_HALO_TRANSPORT="pinned", ASTR_GPU_SYNC_MODE="explicit",
                            ASTR_FORCE_MPI_TOPOLOGY=",".join(map(str, args.topology)),
                            ASTR_AIR5_C4_CONSERVATION="f")
    if args.postprocess_transport == "device-aware":
        env |= dict(OMPI_MCA_pml="ucx", OMPI_MCA_opal_cuda_support="1", UCX_MEMTYPE_CACHE="n",
                    UCX_CUDA_COPY_ENABLE_FABRIC="no", UCX_CUDA_COPY_DMABUF="no",
                    UCX_CUDA_IPC_ENABLE_MNNVL="no", UCX_TLS="self,sm,cuda_copy,cuda_ipc")

    def run(name, steps, mode, binary=args.binary, resume=None):
        work = root / name
        prepare(work, args.input.resolve(), steps)
        (work / "outdat/new").mkdir(parents=True)
        archive = mode == "oracle" and args.volume_statistics and binary == args.binary
        (work / "datin/input.output").write_text(f"""&output
 directory='outdat/new', restore_directory='{resume or ''}', exact_restart=t,
 host_budget_bytes=-1, device_budget_bytes=-1, buffer_bytes=16384
/
&checkpoint
 enabled=t, mode='steps', interval_steps=2, keep=2, initial_frame=f
/
&volume
 enabled={'.true.' if archive else '.false.'}, mode='steps', interval_steps=1, initial_frame=t
/
""")
        catalog = [dict(catalog_schema="astr-insitu-products-v1")]
        flags = {}
        if mode == "oracle":
            flags = dict(ASTR_INSITU_SAMPLE_PREFIX="outdat/sample", ASTR_INSITU_TEST_INITIAL="1")
        if mode == "statistics":
            (work / "outdat/render").mkdir()
            rendering = "render=f"
            if images:
                script = Path(__file__).resolve().parents[2] / "scripts/insitu/device_render_pipeline.py"
                rendering = ("render=t, processing_backend='device', derivative_backend='gpu',\n"
                             f" postprocess_transport='{args.postprocess_transport}', rendering_pipeline='{args.pipeline}',\n"
                             f" implementation_path='{args.images_implementation.resolve()}', pipeline_file='{script}'")
            (work / "datin/insitu.nml").write_text(f"""&insitu_run
 enabled=t, statistics=t, {rendering}, products='air5_walls',
 wall_separation={'.true.' if args.separation else '.false.'},
 air5_volume_statistics={'.true.' if args.volume_statistics else '.false.'},
 air5_volume_reduction={'.true.' if args.volume_statistics and not args.no_volume_reduction else '.false.'},
 statistics_window={WINDOW[0]:.17e},{WINDOW[1]:.17e}, output_directory='outdat/render',
 schedule_mode='steps', step_interval=1, initial_frame=f, final_frame=t,
 host_budget_bytes=9223372036854775807, device_budget_bytes=9223372036854775807, device_reserve_bytes=0
/
""")
            flags = {"ASTR_INSITU_CONFIG": "datin/insitu.nml"}
            for field in WALL_FIELDS:
                product = dict(id=field, kind="wall_curve", enable_request=1, field=field, source="instantaneous",
                               sampling=dict(mode="wall", axis="j", side="lower", spanwise_index=4,
                                             bounds=bounds),
                               output=dict(channel="data", format="csv"), schedule=dict(mode="steps", interval=2))
                if images:
                    product["schedule"]["interval"] = 1
                catalog.extend((product, product | dict(id=field+"_mean", source="reynolds",
                                                       statistics_source="wall-time",
                                                       schedule=dict(mode="steps", interval=1))))
            catalog.extend(images)
        (work / "input.products.jsonl").write_text("".join(json.dumps(item)+"\n" for item in catalog))
        launcher = args.reference_mpiexec if binary == args.reference and args.reference_mpiexec else args.mpiexec
        launch = [launcher.resolve(), "--bind-to", "none", "-np", nranks, binary.resolve(), "run", "datin/input.case"]
        command(launch, work, "run.log", env | flags)
        assert "The job is done!" in (work / "run.log").read_text(errors="replace"), work
        return work

    oracle = run("cpp_wall_fields", 4, "oracle")
    old = run("old_gpu_wall_fields", 4, "oracle", binary=args.reference)
    full = run("continuous", 4, "statistics")
    seed = run("seed", 2, "statistics")
    resumed = run("resumed", 4, "statistics", resume=checkpoint(seed, 2).parent)
    bare = run("without_observers", 4, "")
    for other in (resumed, oracle, bare):
        compare_fields(checkpoint(full, 4), checkpoint(other, 4))
    state_paths = compare_fields(statistics(full, 4), statistics(resumed, 4))
    errors = dict(old_wall_fields=0., independent_mean=0., independent_variance=0.)
    owned_nodes = 0
    k4 = np.empty((17, 18))
    k4_mean = np.empty_like(k4)
    line_coverage = np.zeros(17, dtype=int)
    for rank in range(nranks):
        frames = []
        for step in range(5):
            header, xyz, fields, owned = wall_oracle(oracle, rank, step)
            old_header, old_xyz, old_fields, old_owned = wall_oracle(old, rank, step, legacy=True)
            np.testing.assert_array_equal(header[:7], old_header)
            np.testing.assert_array_equal(xyz, old_xyz)
            np.testing.assert_array_equal(owned, old_owned)
            error = np.max(np.abs(fields-old_fields)/FIELD_SCALES, initial=0)
            errors["old_wall_fields"] = max(errors["old_wall_fields"], float(error))
            np.testing.assert_array_equal(fields[..., 15], 0)
            np.testing.assert_array_equal(fields[..., 16], fields[..., 13]+fields[..., 14])
            frames.append(fields)
        mean, variance, duration = oracle_moments(np.stack(frames), np.arange(5)*DT)
        rel = f"outdat/render/sample.wall_statistics.step00000004.rank{rank:08d}.bin"
        _, meta, count, stat_xyz, values, stat_owned = read_statistics(full / rel)
        assert count == 5
        np.testing.assert_array_equal(meta, [4*DT, *WINDOW, duration])
        np.testing.assert_array_equal(xyz, stat_xyz)
        np.testing.assert_array_equal(owned, stat_owned)
        assert (full / rel).read_bytes() == (resumed / rel).read_bytes(), rank
        errors["independent_mean"] = max(errors["independent_mean"], float(
            np.max(np.abs(values[..., :18]-mean)/FIELD_SCALES, initial=0)))
        errors["independent_variance"] = max(errors["independent_variance"], float(
            np.max(np.abs(values[..., 18:36]-variance)/FIELD_SCALES**2, initial=0)))
        owned_nodes += int(owned.sum())
        for i, k, w in np.argwhere(owned):
            if k+header[8] == 4:
                gi = i+header[6]
                k4[gi], k4_mean[gi] = fields[i, k, w], values[i, k, w, :18]
                line_coverage[gi] += 1
    assert owned_nodes == 272 and np.all(line_coverage == 1)
    assert max(errors.values()) <= ATOL, errors
    for field, column in WALL_FIELDS.items():
        for name, reference in ((field, k4), (field+"_mean", k4_mean)):
            path = wall_frames(full, name, 4)
            assert path.read_bytes() == wall_frames(resumed, name, 4).read_bytes()
            with path.open() as stream:
                line = np.array([float(row["value"]) for row in csv.DictReader(line for line in stream if not line.startswith("#"))])
            np.testing.assert_array_equal(line, reference[:, column])
    report = dict(topology=args.topology, errors=errors, wall_fields=18, owned_nodes=owned_nodes,
                  statistics_datasets=len(state_paths), exact_continuation=True, observer_isolation=True)
    if args.separation:
        report["separation"] = check_separation(oracle, full, resumed, nranks, lengths[2])
    if args.volume_statistics:
        report["volume"] = check_volume(oracle, full, resumed, nranks, lengths, not args.no_volume_reduction)
    if images:
        report["rendering"] = check_rendering(full, resumed, images, args.pipeline,
                                              args.topology, args.postprocess_transport)
    (root / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(root / "report.json")


if __name__ == "__main__":
    main()
