"""Completed-step station profiles against independent archived endpoint samples."""
import os
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

from run_output_restart_validation import run_case, compare_fields
from test_index_slice_runtime import ROOT, arguments, product, segments, frames
from test_insitu_selected_runtime import configuration
from prepare_m12_local_case import prepare

EXE = os.environ.get("ASTR_SELECTED_RUNTIME_EXE")
M12_SOURCE = os.environ.get("ASTR_SELECTED_RUNTIME_M12_SOURCE")
pytestmark = pytest.mark.skipif(not EXE, reason="Select the root-CMake-built runtime executable")
FIELDS = ["density", "velocity_x", "velocity_y", "velocity_z", "pressure", "temperature"]
PAIRS = ((0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2))


def station(name, source="reynolds", mode="station_span", krange=(0, 16)):
    return dict(id=name, kind="station_profile", enable_request=1, source=source,
                statistics_source="selected-time", fields=FIELDS[1:],
                sampling=dict(mode=mode, axis="i", index=8,
                              index_ranges=dict(j=[0, 16], k=list(krange))),
                output=dict(channel="data", format="csv"),
                schedule=dict(mode="steps", interval=1))


def definitions():
    raw = product("raw", FIELDS)
    raw["sampling"] = dict(mode="grid_index", axis="i", index=8,
                           index_ranges=dict(j=[0, 16], k=[0, 16]))
    raw["schedule"]["interval"] = 1
    records = [raw]
    for source in ("reynolds", "favre"):
        records.extend([station(source, source),
                        station(source + "-partial", source, krange=(5, 16)),
                        station(source + "-line", source, "station_line", (16, 16))])
    return records


def read_profile(path):
    lines = path.read_text().splitlines()
    data = np.genfromtxt([line for line in lines if not line.startswith("#")], delimiter=",", names=True)
    return data, "\n".join(line for line in lines if line.startswith("#"))


def endpoint_weights(steps, dt, window):
    weights = np.zeros(steps + 1)
    for right in range(1, steps + 1):
        t0, t1 = (right-1)*dt, right*dt
        a, b = max(t0, window[0]), min(t1, window[1])
        if b <= a:
            continue
        fraction = .5*((a-t0)/dt + (b-t0)/dt)
        weights[right-1] += (b-a)*(1-fraction)
        weights[right] += (b-a)*fraction
    return weights


def check_profiles(case, records, record_property, steps=12, dt=.001, window=(.0005, .0115), ny=17, nk=16):
    raw = next(path for path in segments(case) if path.parent.parent.name == "raw")
    samples = []
    for path in frames(raw):
        with h5py.File(path) as data:
            samples.append(np.stack([data[name][...] for name in FIELDS]))
    values = np.stack(samples)
    assert values.shape == (steps+1, 6, nk+1, ny)
    with h5py.File(raw / "geometry.h5") as data:
        xyz = data["coordinates"][...]
    # Native HDF5 ordering for an i-station is (k,j), with physical coordinates last.
    assert np.all(np.diff(xyz[0, :, 1]) > 0) and np.all(np.diff(xyz[:, 0, 2]) > 0)
    time = endpoint_weights(steps, dt, window)
    rho = values[:, 0]
    primitive = values[:, 1:]
    mass = np.einsum("t,tkj->kj", time, rho)
    mean_r = np.einsum("t,tfkj->fkj", time, primitive)/time.sum()
    mean_f = np.einsum("t,tkj,tfkj->fkj", time, rho, primitive)/mass
    cov_r, cov_f = [], []
    for a, b in PAIRS:
        dr_a, dr_b = primitive[:, a]-mean_r[a], primitive[:, b]-mean_r[b]
        df_a, df_b = primitive[:, a]-mean_f[a], primitive[:, b]-mean_f[b]
        cov_r.append(np.einsum("t,tkj,tkj->kj", time, dr_a, dr_b)/time.sum())
        cov_f.append(np.einsum("t,tkj,tkj,tkj->kj", time, rho, df_a, df_b)/mass)
    cov_r, cov_f = np.asarray(cov_r), np.asarray(cov_f)
    density = mass/time.sum()
    worst = absolute = raw_rms = 0.
    for item in records[1:]:
        paths = list((case / "outdat/new/profile_products" / item["id"]).glob("*/segment*"))
        assert len(paths) == 1, (case / "run.log").read_text()
        files = sorted(paths[0].glob("frame.step*.csv"))
        assert files, (case / "run.log").read_text()
        data, metadata = read_profile(files[-1])
        assert f"step={steps};" in metadata and "phase=completed_step" in metadata
        assert "spatial_measure=" in metadata and "sampling_segments=1" in metadata
        zweight = np.zeros(nk)
        lo, hi = item["sampling"]["index_ranges"]["k"]
        if item["sampling"]["mode"] == "station_line":
            zweight[lo % nk] = 1
        else:
            for k in range(lo, hi+1):
                left = max(lo, k-1)
                right = min(hi, k+1)
                zweight[k % nk] += .5*abs(xyz[right, 0, 2]-xyz[left, 0, 2])
        measure = zweight.sum()
        rho_z = np.einsum("k,kj->j", zweight, density[:nk])/measure
        expected = {"y": xyz[0, :, 1], "rho_time_mean": rho_z}
        requested = set(item['fields'])-{'density'}
        covariance_requested = bool(requested & set(FIELDS[1:4]))
        if covariance_requested:
            requested.update(FIELDS[1:4])
        favre = item["source"] == "favre"
        for f, name in enumerate(FIELDS[1:]):
            if name not in requested:
                continue
            expected[name] = (np.einsum("k,kj,kj->j", zweight, density[:nk], mean_f[f, :nk]) /
                              (rho_z*measure) if favre else
                              np.einsum("k,kj->j", zweight, mean_r[f, :nk])/measure)
        for pair, (a, b) in enumerate(PAIRS):
            if not covariance_requested:
                break
            suffix = "xyz"[a]+"xyz"[b]
            stress = np.einsum("k,kj,kj->j", zweight, density[:nk], cov_f[pair, :nk])/measure
            covariance = stress/rho_z if favre else np.einsum("k,kj->j", zweight, cov_r[pair, :nk])/measure
            expected["covariance_"+suffix] = covariance
            expected["rho_covariance_f_"+suffix] = stress
            if a == b:
                name = "rms_velocity_"+"xyz"[a]
                observed = data[name]
                raw_rms = max(raw_rms, float(np.max(np.abs(observed-np.sqrt(covariance)))))
                difference = np.abs(observed**2-covariance)
                assert np.max(difference/np.maximum(1, np.maximum(observed**2, np.abs(covariance)))) <= 2e-10
        for name, target in expected.items():
            observed = data[name]
            error = np.abs(observed-target)
            scaled = error/np.maximum(1, np.maximum(np.abs(observed), np.abs(target)))
            worst, absolute = max(worst, float(scaled.max())), max(absolute, float(error.max()))
            assert scaled.max() <= 2e-10, (item["id"], name, scaled.max())
        np.testing.assert_array_equal(data["global_j"], np.arange(ny))
        assert set(data.dtype.names) == {'global_j', *expected, *(
            ('rms_velocity_x', 'rms_velocity_y', 'rms_velocity_z') if covariance_requested else ())}
    log = (case / "run.log").read_text()
    assert "ASTR_STATION_PROFILE" in log and "ASTR_STATION_PROFILE_DISABLED" not in log
    for key, value in dict(max_scaled_error=worst, max_absolute_error=absolute, rms_raw_maxabs=raw_rms).items():
        record_property(key, value)
    record_property("case_directory", str(case))


@pytest.mark.parametrize("backend", ("cpu", "gpu"))
@pytest.mark.parametrize("topology", ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2),
                                     (2, 2, 1), (2, 1, 2), (1, 2, 2)))
def test_real_station_temporal_profiles(tmp_path, backend, topology, record_property):
    args = arguments(tmp_path)
    args.executable = Path(EXE).resolve()
    records = definitions()
    case, _ = run_case(args, ROOT, backend, int(np.prod(topology)), "profile", 12,
                       topology=topology, checkpoint_enabled=False, insitu_config=configuration(),
                       product_catalog_records=records, insitu_timing=True)
    check_profiles(case, records, record_property)


@pytest.mark.skipif(not M12_SOURCE, reason="Select the existing read-only M12 fixture resources")
@pytest.mark.parametrize("backend", ("cpu", "gpu"))
@pytest.mark.parametrize("topology", ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2),
                                     (2, 2, 1), (2, 1, 2), (1, 2, 2)))
def test_flatplate_station_profiles(tmp_path, backend, topology, record_property):
    case = tmp_path / backend
    prepare(Path(M12_SOURCE), case, backend)
    records = definitions()
    for item in records:
        item["sampling"]["index"] = 32
        item["sampling"]["index_ranges"]["j"] = [0, 32]
        krange = item["sampling"]["index_ranges"]["k"]
        item["sampling"]["index_ranges"]["k"] = [12 if krange[0] == 5 else (24 if krange[0] == 16 else 0), 24]
    (case / "input.products.jsonl").write_text("\n".join(json.dumps(row) for row in
        [{"catalog_schema": "astr-insitu-products-v1"}, *records])+"\n")
    (case / "datin/input.insitu").write_text(configuration((.01, .19)))
    env = {key: value for key, value in os.environ.items() if not key.startswith("ASTR_")}
    env.update(ASTR_FORCE_MPI_TOPOLOGY=",".join(map(str, topology)), ASTR_WALL_BLOWING_MODE="legacy_random",
               ASTR_GPU_HALO_TRANSPORT="pinned", ASTR_GPU_SYNC_MODE="explicit",
               ASTR_INSITU_CONFIG="datin/input.insitu", ASTR_INSITU_TIMING="1")
    args = arguments(case.parent)
    with (case / "run.log").open("w") as log:
        result = subprocess.run([str(args.mpiexec), "--mca", "coll_hcoll_enable", "0", "--oversubscribe",
                                 "-np", str(np.prod(topology)), str(Path(EXE).resolve()), "run", "datin/input.dat"],
                                cwd=case, env=env, stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode == 0, (case / "run.log").read_text()
    assert "The job is done!" in (case / "run.log").read_text()
    check_profiles(case, records, record_property, steps=10, dt=.02, window=(.01, .19), ny=33, nk=24)


@pytest.mark.parametrize("backend", ("cpu", "gpu"))
def test_station_exact_continuation(tmp_path, backend):
    args = arguments(tmp_path)
    args.executable = Path(EXE).resolve()
    args.statistics = True
    records = definitions()[1:]
    common = dict(insitu_config=configuration(), product_catalog_records=records, checkpoint_enabled=True,
                  checkpoint_interval=5, exact_restart=True, topology=(1, 1, 2))
    full, _ = run_case(args, ROOT, backend, 2, "full", 12, **common)
    first, _ = run_case(args, ROOT, backend, 2, "prefix", 5, **common)
    tail, _ = run_case(args, ROOT, backend, 2, "tail", 12,
                       restore=first / "outdat/new/checkpoints/step000000000005", **common)
    for filename in ("state.h5", "statistics.h5"):
        final = Path("outdat/new/checkpoints/step000000000012")
        compare_fields(full / final / filename, tail / final / filename)
    for item in records:
        original = next((full / "outdat/new/profile_products" / item["id"]).glob("*/segment*"))
        resumed = next((tail / "outdat/new/profile_products" / item["id"]).glob("*/segment*"))
        for path in resumed.glob("frame.step*.csv"):
            assert path.read_bytes() == (original / path.name).read_bytes(), path


@pytest.mark.parametrize('backend', ('cpu', 'gpu'))
def test_cartesian_read_grid_station_exact_continuation(tmp_path, backend):
    grid_source=os.environ.get('ASTR_SELECTED_RUNTIME_TGV_READ_GRID')
    if not grid_source:
        pytest.skip('Select an immutable uniform Cartesian 16^3 TGV mesh')
    args=arguments(tmp_path);args.executable=Path(EXE).resolve();args.statistics=True
    records=[station('read-reynolds'),station('read-favre','favre')]
    common=dict(insitu_config=configuration(),product_catalog_records=records,checkpoint_enabled=True,
                checkpoint_interval=5,checkpoint_keep=1,exact_restart=True,topology=(1,1,1),
                tgv_cartesian_grid=grid_source)
    full,_=run_case(args,ROOT,backend,1,'read-full',12,**common)
    first,_=run_case(args,ROOT,backend,1,'read-prefix',5,**common)
    tail,_=run_case(args,ROOT,backend,1,'read-tail',12,
                    restore=first/'outdat/new/checkpoints/step000000000005',**common)
    final=Path('outdat/new/checkpoints/step000000000012')
    for filename in ('state.h5','statistics.h5'):
        compare_fields(full/final/filename,tail/final/filename)
    for case in (full,first,tail):
        text=(case/'run.log').read_text()
        assert text.count('ASTR_SELECTED_GEOMETRY rank=0 cartesian=T')==1
        assert len(list((case/'outdat/new/checkpoints').glob('step*')))==1
    for item in records:
        before=next((full/'outdat/new/profile_products'/item['id']).glob('*/segment*'))
        after=next((tail/'outdat/new/profile_products'/item['id']).glob('*/segment*'))
        for frame in after.glob('frame.step*.csv'):
            assert frame.read_bytes()==(before/frame.name).read_bytes()


@pytest.mark.parametrize("backend", ("cpu", "gpu"))
def test_station_scalar_dependencies(tmp_path, backend, record_property):
    args = arguments(tmp_path)
    args.executable = Path(EXE).resolve()
    raw = definitions()[0]
    records = [raw, station('density'), station('scalars'), station('favre-scalars', 'favre')]
    records[1]['fields'] = ['density']
    for item in records[2:]:
        item['fields'] = ['pressure', 'temperature']
    case, _ = run_case(args, ROOT, backend, 2, "scalar", 12,
                       topology=(1, 2, 1), checkpoint_enabled=False, insitu_config=configuration(),
                       product_catalog_records=records)
    check_profiles(case, records, record_property)


@pytest.mark.parametrize("backend", ("cpu", "gpu"))
def test_station_failure_isolated_and_authority_unchanged(tmp_path, backend):
    args = arguments(tmp_path)
    args.executable = Path(EXE).resolve()
    records = definitions()
    bad = station('x'*256)
    bad['sampling']['index'] = 9
    common = dict(topology=(2, 1, 1), checkpoint_enabled=True, checkpoint_interval=12,
                  insitu_config=configuration(), exact_restart=True)
    baseline, _ = run_case(args, ROOT, backend, 2, 'baseline', 12, product_catalog_records=records[:1], **common)
    enabled, _ = run_case(args, ROOT, backend, 2, 'isolated', 12, product_catalog_records=[*records, bad], **common)
    state = Path('outdat/new/checkpoints/step000000000012/state.h5')
    compare_fields(baseline / state, enabled / state)
    status = json.loads((enabled / 'outdat/new/products.status.json').read_text())['products']
    by_id = {item['id']: item for item in status}
    assert by_id[bad['id']]['state'] == 'resource-disabled'
    for item in records:
        assert by_id[item['id']]['state'] == 'active'
