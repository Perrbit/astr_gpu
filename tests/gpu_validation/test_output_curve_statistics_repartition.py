"""Approved OR4 static-CURVE mean44/compact-history migration, bounded to 16 cubed."""
import hashlib
import json

import h5py
import numpy as np
import pytest

from run_output_restart_validation import run_case, compare_fields, archive_schedule_payload, controlled_checkpoint_buffers
from test_output_archive_segments import fingerprints
from test_output_device_reserve import reserve_records
from test_output_repartition_runtime import (
    ROOT, FINAL, ATOL, curve_arguments, clocks, compare_global_state,
    compare_curve_target_halos,
)


def compact_total(path):
    with h5py.File(path) as state:
        assert int(state["identity"][12]) == 5
        metadata = state["metadata"][:]
        assert metadata[0] == 2 and metadata[11] == 17
        assert state["inherited/metadata"][:].tobytes() == metadata.tobytes()
        assert state["inherited/identity"][8:12].tobytes() == state["identity"][8:12].tobytes()
        result = {}
        for component in range(1, 18):
            name = f"q{component:04d}"
            local, inherited = state[name][:], state["inherited/"+name][:]
            assert np.isfinite(local).all() and np.isfinite(inherited).all()
            assert np.all(local[-1] == 0) and np.all(inherited[-1] == 0)
            # HDF5 dimension order is z,j,i. The terminal plane is padding.
            result[name] = np.sum(local[:-1], axis=0)
            if component not in (1, 14):
                result[name] += inherited[0]
        assert np.any(result["q0002"]), "the migration gate requires nonzero history"
        return metadata, result


def compare_statistics(reference, actual, backend):
    if backend == "gpu":
        am, a = compact_total(reference)
        bm, b = compact_total(actual)
        assert am[np.r_[0:4, 7:12]].tobytes() == bm[np.r_[0:4, 7:12]].tobytes()
    else:
        with h5py.File(reference) as left, h5py.File(actual) as right:
            assert int(left["identity"][12]) == int(right["identity"][12]) == 6
            assert left["metadata"][:].tobytes() == right["metadata"][:].tobytes()
            a = {name: left[name][:] for name in left if name.startswith("q")}
            b = {name: right[name][:] for name in right if name.startswith("q")}
            assert len(a) == len(b) == 44 and np.any(a["q0002"])
    errors = {}
    for name, field in a.items():
        assert field.shape == b[name].shape
        assert np.isfinite(field).all() and np.isfinite(b[name]).all()
        errors[name] = float(np.max(np.abs(field-b[name])))
        assert errors[name] <= ATOL, (name, errors[name])
    return max(errors.values())


def execute_gate(backend, source_np, target_np, source_axis, target_axis, tmp_path,
                 record_property, exact=False, repeat=False, memcheck=False, case_name="curve"):
    args = curve_arguments(tmp_path, backend, target_axis)
    args.case = case_name
    if case_name == "dynamic":
        args.inflow_count = 12
    args.legacy_statistics = True
    digest = hashlib.sha256(args.executable.read_bytes()).hexdigest()
    interval = 2 if repeat else 5
    archive = """&volume
 enabled=.true., mode='steps', interval_steps=99, initial_frame=.true., final_frame=.true.
/
&slices
 enabled=.false.
/
"""
    common = dict(buffer_bytes=4096, checkpoint_interval=interval,
        device_reserve_bytes=1073741824, archive_groups=archive)
    reference, _ = run_case(args, ROOT, backend, target_np, "continuous", 12, **common)
    args.axis = source_axis
    seed, _ = run_case(args, ROOT, backend, source_np, "seed", 5, **common)
    source_tree = seed / "outdat/new"
    original = fingerprints(source_tree)
    args.axis = target_axis
    restored, _ = run_case(args, ROOT, backend, target_np, "restored", 12,
        restore=source_tree / "checkpoints/step000000000005", memcheck=memcheck, **common)
    cases = [(reference, target_np), (seed, source_np), (restored, target_np)]
    assert clocks(seed)+clocks(restored) == clocks(reference)
    assert ("ASTR_OUTPUT_REPARTITION bl" in (restored / "run.log").read_text()) != exact
    fields = compare_global_state(reference / FINAL / "state.h5", restored / FINAL / "state.h5")
    halos = compare_curve_target_halos(reference / FINAL / "state.h5", restored / FINAL / "state.h5")
    statistics = compare_statistics(reference / FINAL / "statistics.h5", restored / FINAL / "statistics.h5", backend)
    inflow_error = 0.0
    if case_name == "dynamic":
        inflow_error = compare_inflow(reference / FINAL / "inflow.h5", restored / FINAL / "inflow.h5")
        with h5py.File(source_tree / "checkpoints/step000000000005/inflow.h5") as initial, \
                h5py.File(restored / FINAL / "inflow.h5") as final:
            assert final["metadata"][2] > initial["metadata"][2] >= 3, "the gate must cross a window rotation"
            assert initial["metadata"][6:10].tobytes() != final["metadata"][6:10].tobytes()
        assert not (restored / "inflow").exists()
        assert not (restored / "datin/grid.flatplate.h5").exists()
        assert not (restored / "datin/inlet.prof").exists()
    if exact:
        for name in ("state.h5", "statistics.h5") + (("inflow.h5",) if case_name == "dynamic" else ()):
            compare_fields(reference / FINAL / name, restored / FINAL / name)
    if repeat:
        migrated_tree = restored / "outdat/new"
        migrated_before = fingerprints(migrated_tree)
        resumed, _ = run_case(args, ROOT, backend, target_np, "second_restart", 12,
            restore=migrated_tree / "checkpoints/step000000000010", **common)
        cases.append((resumed, target_np))
        for name in ("state.h5", "statistics.h5") + (("inflow.h5",) if case_name == "dynamic" else ()):
            compare_fields(restored / FINAL / name, resumed / FINAL / name)
        assert migrated_before == fingerprints(migrated_tree)
    for name in ("control.bin", "archives.bin", "insitu_control.bin"):
        read = archive_schedule_payload if name == "archives.bin" else lambda p: p.read_bytes()
        assert read(reference / FINAL / name) == read(restored / FINAL / name)
    assert original == fingerprints(source_tree)
    assert digest == hashlib.sha256(args.executable.read_bytes()).hexdigest()
    size = sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file())
    assert size <= 64*1024**2
    host_bounds, device_bounds, reserves = [], [], []
    for case, ranks in cases:
        checkpoint = case / ("outdat/new/checkpoints/step000000000005" if case == seed else FINAL)
        with h5py.File(checkpoint / "state.h5") as state:
            partitions = state["partitions"][:].reshape(-1, 8)
        # Conservative explicit-allocation bound: local staging, saved global
        # history, global geometry staging and worst old two-z-partition payload.
        baseline = 17*17*2*17*8 if backend == "gpu" else 0
        migration_staging = baseline*3 + 17*17*3*17*8 if backend == "gpu" else 0
        device = max((int((r[3]+1)*(r[4]+1))*13+int(r[3]+1)*4)*8 for r in partitions) if backend == "gpu" else 0
        inflow_host = 26*17*17*8 if case_name == "dynamic" else 0
        inflow_device = 20*17*17*8 if case_name == "dynamic" and backend == "gpu" else 0
        host_bounds += controlled_checkpoint_buffers(checkpoint / "state.h5", baseline+4096+inflow_host, backend)
        host_bounds += [v+migration_staging for v in controlled_checkpoint_buffers(
            checkpoint / "statistics.h5", baseline+4096+inflow_host, backend)]
        if case_name == "dynamic":
            host_bounds += controlled_checkpoint_buffers(checkpoint / "inflow.h5", baseline+4096+inflow_host, backend)
        device_bounds.append(device+4096+inflow_device if backend == "gpu" else 0)
        if backend == "gpu":
            reserves.append(reserve_records(case, ranks))
    assert max(host_bounds+device_bounds) <= 64*1024**2
    report = dict(case=case_name, backend=backend, source_np=source_np, target_np=target_np,
        source_axis=source_axis, target_axis=target_axis, exact=exact, second_exact_restart=repeat,
        statistics_max_abs=statistics, state_max_abs=max(fields.values()), target_halos=halos,
        inflow_max_abs=inflow_error,
        directory_bytes=size, executable_sha256=digest, controlled_host_bytes=max(host_bounds),
        controlled_device_bytes=max(device_bounds), reserve_measurements=reserves)
    (tmp_path / "statistics_migration.json").write_text(json.dumps(report, indent=2)+"\n")
    record_property("result", json.dumps(report))


def compare_inflow(reference, actual):
    with h5py.File(reference) as a, h5py.File(actual) as b:
        assert a["metadata"][:].tobytes() == b["metadata"][:].tobytes()
        assert a["identity"][8:12].tobytes() == b["identity"][8:12].tobytes()
        assert int(a["identity"][12]) == int(b["identity"][12]) == 8
        error = 0.0
        for m in range(1, 27):
            left, right = a[f"q{m:04d}"][:], b[f"q{m:04d}"][:]
            assert left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all()
            error = max(error, float(np.max(np.abs(left-right))))
        assert error <= ATOL, error
        assert np.ptp(a["q0002"][0]) > 0, "the inlet cache must contain spatially varying data"
        return error


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("axis", ["x", "z"])
def test_same_topology_exact(backend, axis, tmp_path, record_property):
    execute_gate(backend, 2, 2, axis, axis, tmp_path, record_property, exact=True)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("source_np,target_np,source_axis,target_axis", [
    (1, 2, "x", "x"), (2, 1, "x", "x"), (1, 2, "z", "z"),
    (2, 1, "z", "z"), (2, 2, "x", "z"), (2, 2, "z", "x"),
])
def test_statistics_repartition(backend, source_np, target_np, source_axis, target_axis,
                                tmp_path, record_property):
    execute_gate(backend, source_np, target_np, source_axis, target_axis, tmp_path,
                 record_property, repeat=backend == "gpu" and source_axis != target_axis)


@pytest.mark.parametrize("source_axis,target_axis", [("x", "z"), ("z", "x")])
def test_statistics_repartition_memcheck(source_axis, target_axis, tmp_path, record_property):
    execute_gate("gpu", 2, 2, source_axis, target_axis, tmp_path, record_property, memcheck=True)
