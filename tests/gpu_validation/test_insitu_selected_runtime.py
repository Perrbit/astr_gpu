"""Real completed-step catalog sampling; deliberately no checkpoint or rendering yet."""
import os
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest

from run_output_restart_validation import run_case, compare_fields
from test_index_slice_runtime import ROOT, arguments, product, segments, frames
from prepare_m12_local_case import prepare


EXE = os.environ.get("ASTR_SELECTED_RUNTIME_EXE")
pytestmark = pytest.mark.skipif(not EXE, reason="Select a root-CMake-built selected-statistics executable")


def assert_primitive_corner_padding(path):
    with h5py.File(path) as state:
        assert int(state["identity"][6]) == 11
        global_shape = state["identity"][3:6]
        offset = 0
        for partition in state["partitions"][...].reshape(-1, 8):
            origin, cells, halo, count = partition[:3], partition[3:6], int(partition[6]), int(partition[7])
            owned = cells + (origin + cells == global_shape - 1)
            full = cells + 2*halo + 1
            k, j, i = np.indices(tuple(map(int, full[::-1])))
            i, j, k = i-halo, j-halo, k-halo
            off_axis = (i < 0) | (i > cells[0])
            off_axis = off_axis.astype(np.int8) + ((j < 0) | (j > cells[1])) + ((k < 0) | (k > cells[2]))
            external = ~((i >= 0) & (i < owned[0]) & (j >= 0) & (j < owned[1]) & (k >= 0) & (k < owned[2]))
            padding = (off_axis >= 2)[external]
            extra = state["rank_extras"][offset:offset+count].reshape(11, -1)
            assert np.all(extra[5:11, padding].copy().view(np.uint64) == 0), "primitive corner storage must be positive zero"
            offset += count


def configuration(window=(.0005, .0115)):
    return f"""&insitu_run
 enabled=t, statistics=t, render=f, products='catalog_stats',
 statistics_window={window[0]},{window[1]}, output_directory='outdat',
 host_budget_bytes=67108864,device_budget_bytes=67108864,device_reserve_bytes=1073741824
/
"""


def definitions():
    raw = product("raw", ["density", "velocity_x", "pressure", "temperature"])
    raw["schedule"]["interval"] = 1
    records = [raw]
    for source in ("reynolds", "favre"):
        mean = deepcopy(raw)
        mean.update(id=source, source=source, statistics_source="selected-time")
        if source == "favre":
            mean["fields"].remove("density")
        records.append(mean)
    return records


def check_completed_means(case, raw, steps, dt, window):
    paths = {path.parent.parent.name: path for path in segments(case)}
    assert set(paths) == {"raw", "reynolds", "favre"}, (case / "run.log").read_text()
    raw_frames = frames(paths["raw"])
    assert len(raw_frames) == steps + 1
    samples = {}
    for path in raw_frames:
        with h5py.File(path) as data:
            step = int(np.asarray(data["step"]).item())
            samples[step] = {field: np.asarray(data[field]) for field in raw["fields"]}
    # Integrate the endpoint interpolants, not an arithmetic mean of saved frames.
    weights = np.zeros(steps + 1)
    for right in range(1, steps + 1):
        t0, t1 = (right - 1) * dt, right * dt
        a, b = max(t0, window[0]), min(t1, window[1])
        if b <= a:
            continue
        fraction = .5 * ((a - t0) / dt) + .5 * ((b - t0) / dt)
        weights[right - 1] += (b - a) * (1 - fraction)
        weights[right] += (b - a) * fraction
    rho = np.stack([samples[s]["density"] for s in range(steps + 1)])
    mass = np.tensordot(weights, rho, axes=1)
    worst = 0.
    absolute = 0.
    for source in ("reynolds", "favre"):
        assert len(frames(paths[source])) == steps
        with h5py.File(frames(paths[source])[-1]) as actual:
            assert "statistics_source=selected-time" in str(actual.attrs["source_measure"])
            for field in raw["fields"]:
                if source == "favre" and field == "density":
                    continue
                values = np.stack([samples[s][field] for s in range(steps + 1)])
                expected = (np.tensordot(weights, values, axes=1) / weights.sum() if source == "reynolds"
                            else np.tensordot(weights, rho * values, axes=1) / mass)
                observed = np.asarray(actual[field])
                scaled = np.max(np.abs(observed - expected) / np.maximum(1., np.maximum(np.abs(observed), np.abs(expected))))
                worst = max(worst, scaled)
                absolute = max(absolute, float(np.max(np.abs(observed - expected))))
                assert scaled <= 2e-10, (source, field, scaled)
    assert not list((case / "outdat/new/checkpoints").glob("step*"))
    return dict(max_scaled_error=worst, max_absolute_error=absolute)


@pytest.mark.parametrize("backend", ("cpu", "gpu"))
def test_completed_state_pressure_temperature_means(tmp_path, backend, record_property):
    args = arguments(tmp_path)
    args.executable = Path(EXE).resolve()
    records = definitions()
    case, _ = run_case(args, ROOT, backend, 1, "means", 12, checkpoint_enabled=False,
                       insitu_config=configuration(), product_catalog_records=records, insitu_timing=True)
    for key, value in check_completed_means(case, records[0], 12, .001, (.0005, .0115)).items():
        record_property(key, value)


M12_SOURCE = os.environ.get("ASTR_SELECTED_RUNTIME_M12_SOURCE")


def run_flatplate(case, backend, topology, with_statistics=True):
    prepare(Path(M12_SOURCE), case, backend)
    records = definitions()
    if not with_statistics:
        records = records[:1]
    for item in records:
        item["sampling"]["index"] = 12
        item["sampling"]["index_ranges"] = dict(i=[0, 64], j=[0, 32])
    (case / "input.products.jsonl").write_text("\n".join(json.dumps(row) for row in
        [{"catalog_schema": "astr-insitu-products-v1"}, *records]) + "\n")
    if with_statistics:
        (case / "datin/input.insitu").write_text(configuration((.01, .19)))
    env = {key: value for key, value in os.environ.items() if not key.startswith("ASTR_")}
    env.update(ASTR_FORCE_MPI_TOPOLOGY=",".join(map(str, topology)),
               ASTR_WALL_BLOWING_MODE="legacy_random", ASTR_GPU_HALO_TRANSPORT="pinned",
               ASTR_GPU_SYNC_MODE="explicit", ASTR_INSITU_TIMING="1")
    if with_statistics:
        env["ASTR_INSITU_CONFIG"] = "datin/input.insitu"
    args = arguments(case.parent)
    with (case / "run.log").open("w") as log:
        result = subprocess.run([str(args.mpiexec), "--mca", "coll_hcoll_enable", "0", "--oversubscribe",
                                 "-np", str(np.prod(topology)), str(Path(EXE).resolve()), "run", "datin/input.dat"],
                                cwd=case, env=env, stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode == 0, (case / "run.log").read_text()
    assert "The job is done!" in (case / "run.log").read_text()
    return records


@pytest.mark.skipif(not M12_SOURCE, reason="Select the existing read-only M12 short-fixture resources")
@pytest.mark.parametrize("backend", ("cpu", "gpu"))
@pytest.mark.parametrize("topology", ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2),
                                     (2, 2, 1), (2, 1, 2), (1, 2, 2)))
def test_real_flatplate_pressure_temperature_sampling(tmp_path, backend, topology, record_property):
    case = tmp_path / backend
    records = run_flatplate(case, backend, topology)
    for key, value in check_completed_means(case, records[0], 10, .02, (.01, .19)).items():
        record_property(key, value)
    record_property("case_directory", str(case.resolve()))
    record_property("backend", backend)
    record_property("topology", ",".join(map(str, topology)))


@pytest.mark.skipif(not M12_SOURCE, reason="Select the existing read-only M12 short-fixture resources")
def test_shared_face_raw_difference_already_exists_without_statistics(tmp_path, record_property):
    # Diagnostic, not a relaxed cross-topology acceptance gate.
    off = []
    for topology in ((1, 1, 1), (1, 2, 1)):
        name = "x".join(map(str, topology))
        baseline, observed = tmp_path / ("off-" + name), tmp_path / ("on-" + name)
        run_flatplate(baseline, "cpu", topology, with_statistics=False)
        run_flatplate(observed, "cpu", topology)
        a = next(path for path in segments(baseline) if path.parent.parent.name == "raw")
        b = next(path for path in segments(observed) if path.parent.parent.name == "raw")
        for left, right in zip(frames(a), frames(b), strict=True):
            with h5py.File(left) as x, h5py.File(right) as y:
                for field in ("density", "velocity_x", "pressure", "temperature"):
                    np.testing.assert_array_equal(x[field], y[field], err_msg=f"{name}/{left.name}/{field}")
        off.append(a)
    with h5py.File(off[0] / "step000000000003.h5") as a, h5py.File(off[1] / "step000000000003.h5") as b:
        error = np.abs(a["velocity_x"][...] - b["velocity_x"][...])
        record_property("existing_raw_topology_max_abs", float(error.max()))
        record_property("statistics_on_off_raw_max_abs", 0.)


@pytest.mark.parametrize("backend", ("cpu", "gpu"))
@pytest.mark.parametrize("topology", ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2),
                                     (2, 2, 1), (2, 1, 2), (1, 2, 2)))
def test_selected_statistics_exact_checkpoint_continuation(tmp_path, backend, topology):
    ranks = int(np.prod(topology))
    args = arguments(tmp_path)
    args.executable = Path(EXE).resolve()
    args.statistics = True
    records = definitions()
    for record in records:
        record["schedule"]["interval"] = 4
    common = dict(insitu_config=configuration(), product_catalog_records=records,
                  checkpoint_enabled=True, checkpoint_interval=5, exact_restart=True, topology=topology)
    full, _ = run_case(args, ROOT, backend, ranks, "full", 12, **common)
    first, _ = run_case(args, ROOT, backend, ranks, "prefix", 5, **common)
    tail, _ = run_case(args, ROOT, backend, ranks, "tail", 12,
                       restore=first / "outdat/new/checkpoints/step000000000005", **common)
    final = "outdat/new/checkpoints/step000000000012"
    assert_primitive_corner_padding(full / final / "state.h5")
    assert_primitive_corner_padding(tail / final / "state.h5")
    for filename in ("state.h5", "statistics.h5"):
        compare_fields(full / final / filename, tail / final / filename)
    expected = {path.parent.parent.name: path for path in segments(full)}
    for segment in segments(tail):
        product_name = segment.parent.parent.name
        for frame in frames(segment):
            compare_fields(expected[product_name] / frame.name, frame)
