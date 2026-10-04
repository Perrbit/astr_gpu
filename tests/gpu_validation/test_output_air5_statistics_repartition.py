"""Approved OR4 dimensional AIR5 HBL mean44/conservation migration gate."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from run_output_air5_restart_validation import DOMAIN, launch, state_sanity, compare_air5_resources
from run_output_restart_validation import archive_schedule_payload, compare_fields, controlled_checkpoint_buffers
from test_output_archive_segments import fingerprints
from test_output_device_reserve import reserve_records
from test_output_repartition_runtime import (
    FINAL, ATOL, air5_arguments, air5_scales, clocks, compare_air5_scaled,
    compare_curve_target_halos, frozen_geometry_scales,
)

BUDGET = 64 * 1024**2
ARCHIVE = """&volume
 enabled=t,mode='steps',interval_steps=99,initial_frame=t,final_frame=t
/
&slices
 enabled=f
/
"""


def mean_scales(physical):
    rho, sound, pressure, temperature = (physical[name] for name in
        ("rho", "sound", "pressure", "temperature"))
    stress = rho * sound**2
    length = DOMAIN[1]
    result = np.array([rho, *([rho*sound]*3), pressure, rho*temperature,
        *([rho*sound**2]*6), pressure**2, rho*temperature**2,
        *([rho*sound*temperature]*3), *([rho*sound**3]*10), *([sound]*3),
        *([pressure*sound]*3), *([stress]*6), stress*sound/length,
        pressure*sound/length, *([stress*sound]*3)])
    assert result.shape == (44,) and np.isfinite(result).all() and np.all(result > 0)
    return result


def compare_mean(reference, actual, scales, expected_samples):
    dimensional, normalized = {}, {}
    with h5py.File(reference) as a, h5py.File(actual) as b:
        assert a["metadata"][:].tobytes() == b["metadata"][:].tobytes()
        metadata = a["metadata"][:]
        assert int(metadata[0]) == 2 and int(metadata[1]) == expected_samples
        assert int(a["identity"][12]) == int(b["identity"][12]) == 6
        assert a["identity"][8:12].tobytes() == b["identity"][8:12].tobytes()
        for m, scale in enumerate(scales, 1):
            name = f"q{m:04d}"
            left, right = a[name][:], b[name][:]
            assert left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all()
            error = float(np.max(np.abs(left-right)))
            dimensional[name] = error
            if expected_samples:
                normalized[name] = error/(expected_samples*float(scale))
                assert normalized[name] <= ATOL, (name, error, normalized[name])
            else:
                assert np.all(left == 0) and np.all(right == 0)
                normalized[name] = 0.0
        if expected_samples:
            assert np.any(a["q0001"][:]) and np.any(a["q0035"][:]), "exercise nonzero viscous history"
    return dict(max_normalized=max(normalized.values()), dimensional=dimensional,
                normalized=normalized, samples=expected_samples)


def conservation_state(path):
    payload = path.read_bytes()
    assert len(payload) == 168 and payload[:8] == b"ASTRA5C1"
    step = int(np.frombuffer(payload[8:16], dtype="<i8")[0])
    clock = np.frombuffer(payload[16:40], dtype="<f8")
    metadata = np.frombuffer(payload[40:80], dtype="<i8")
    baseline = np.frombuffer(payload[80:], dtype="<f8")
    assert metadata.tolist() == [1, 1, 0, step, step]
    assert np.isfinite(clock).all() and np.isfinite(baseline).all()
    return payload, metadata, baseline


def conservation_rows(case):
    files = list((case / "outdat/new").glob("air5_conservation_from_step*.dat"))
    assert len(files) == 1
    rows = np.loadtxt(files[0], comments="#", ndmin=2)
    assert rows.shape[1] == 21 and np.isfinite(rows).all()
    samples = rows[rows[:, 1] == 1]
    assert len(samples) and np.all(np.diff(samples[:, 0]) == 1)
    return rows, samples


def compare_conservation(reference, actual, source, conserved_scales, start_step, exact=False):
    _, am, av = conservation_state(reference / FINAL / "air5_conservation.bin")
    _, bm, bv = conservation_state(actual / FINAL / "air5_conservation.bin")
    _, _, initial = conservation_state(source / "air5_conservation.bin")
    assert am.tobytes() == bm.tobytes()
    assert initial.tobytes() == bv.tobytes(), "persisted baseline must not be recomputed or multiplied"
    scale = conserved_scales * np.prod(DOMAIN)
    baseline_error = float(np.max(np.abs(av-bv)/scale))
    assert baseline_error <= ATOL
    _, full = conservation_rows(reference)
    rows, tail = conservation_rows(actual)
    echo = rows[rows[:, 1] == 0]
    assert echo.shape[0] == 1 and int(echo[0, 0]) == 0
    assert echo[0, 2:13].tobytes() == initial.tobytes()
    matched = full[full[:, 0] > start_step]
    assert tail[:, :2].tobytes() == matched[:, :2].tobytes()
    error = float(np.max(np.abs(tail[:, 2:13]-matched[:, 2:13])/scale))
    assert error <= ATOL
    if exact:
        assert (reference / FINAL / "air5_conservation.bin").read_bytes() == (actual / FINAL / "air5_conservation.bin").read_bytes()
        assert matched.tobytes() == tail.tobytes()
    return dict(baseline_max_normalized=baseline_error, totals_max_normalized=error,
                persisted_baseline="bitwise", samples=len(tail))


def options(backend):
    return dict(buffer_bytes=4096, conservation=backend == "gpu",
                archive_groups=ARCHIVE, device_reserve_bytes=1073741824)


def arguments(tmp_path, backend, case_name="hbl"):
    args = air5_arguments(tmp_path, backend)
    args.mean_statistics = True
    args.case = case_name
    if case_name == "sbli":
        args.axis = "x"
        args.reconstruction = 3
    return args


@pytest.fixture(scope="module")
def reference_cache(tmp_path_factory):
    cache = {}

    def get(backend, ranks, case_name="hbl"):
        key = backend, ranks, case_name
        if key not in cache:
            root = tmp_path_factory.mktemp(f"air5_{case_name}_history_reference_{backend}_{ranks}")
            args = arguments(root, backend, case_name)
            digest = hashlib.sha256(args.executable.read_bytes()).hexdigest()
            common = dict(options(backend), interval=2)
            continuous, _ = launch(args, backend, ranks, "continuous", 12, **common)
            seed, _ = launch(args, backend, ranks, "seed", 5, **common)
            size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
            assert size <= BUDGET
            cache[key] = continuous, seed, digest, size
        return cache[key]

    return get


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_partition_independence(backend, tmp_path, record_property):
    partition_gate(backend, tmp_path, record_property)


def partition_gate(backend, tmp_path, record_property, case_name="hbl"):
    args = arguments(tmp_path, backend, case_name)
    scales, physical = air5_scales()
    # First sample is step two; three advances exercise one real accumulated sample.
    one, _ = launch(args, backend, 1, "one", 3, **options(backend))
    two, _ = launch(args, backend, 2, "two", 3, **options(backend))
    final = Path("outdat/new/checkpoints/step000000000003")
    state = compare_air5_scaled(one / final / "state.h5", two / final / "state.h5", scales)
    mean = compare_mean(one / final / "statistics.h5", two / final / "statistics.h5", mean_scales(physical), 1)
    conservation = None
    if backend == "gpu":
        _, am, a = conservation_state(one / final / "air5_conservation.bin")
        _, bm, b = conservation_state(two / final / "air5_conservation.bin")
        assert am.tobytes() == bm.tobytes()
        _, left = conservation_rows(one)
        _, right = conservation_rows(two)
        scale = scales[:11] * np.prod(DOMAIN)
        assert left[:, :2].tobytes() == right[:, :2].tobytes()
        conservation = max(float(np.max(np.abs(a-b)/scale)),
                           float(np.max(np.abs(left[:, 2:13]-right[:, 2:13])/scale)))
        assert conservation <= ATOL
    for case in (one, two):
        state_sanity(case / final / "state.h5")
    size = sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file())
    assert size <= BUDGET
    record_property("result", json.dumps(dict(case=case_name, backend=backend, state=state, mean=mean,
        conservation_max_normalized=conservation, test_root_bytes=size,
        executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest())))


def execute_gate(backend, source_ranks, target_ranks, tmp_path, record_property, reference_cache,
                 exact=False, memcheck=False, repeat=False, case_name="hbl"):
    args = arguments(tmp_path, backend, case_name)
    scales, physical = air5_scales()
    digest = hashlib.sha256(args.executable.read_bytes()).hexdigest()
    common = options(backend)
    common["interval"] = 2
    reference, _, reference_digest, reference_size = reference_cache(backend, target_ranks, case_name)
    _, seed, seed_digest, seed_size = reference_cache(backend, source_ranks, case_name)
    assert digest == reference_digest == seed_digest, "freeze executable while reusing references"
    source_tree = seed / "outdat/new"
    original = fingerprints(source_tree)
    source = source_tree / "checkpoints/step000000000005"
    pure, _ = launch(args, backend, target_ranks, "pure", 5, restore=source, restore_probe=True, **common)
    assert clocks(pure) == [] and not list((pure / "outdat/new").rglob("COMPLETE"))
    with h5py.File(source / "state.h5") as a, h5py.File(pure / "outdat/restore_probe.h5") as b:
        for m in range(1, 35):
            name = f"q{m:04d}"
            assert a[name][:].tobytes() == b[name][:].tobytes(), name
        assert any(np.any(a[f"q{m:04d}"][:]) for m in range(12, 23))
    with h5py.File(source / "statistics.h5") as a, h5py.File(pure / "outdat/restore_statistics_probe.h5") as b:
        assert a["metadata"][:].tobytes() == b["metadata"][:].tobytes()
        for m in range(1, 45):
            name = f"q{m:04d}"
            assert a[name][:].tobytes() == b[name][:].tobytes(), name
    if backend == "gpu":
        assert (source / "air5_conservation.bin").read_bytes() == (pure / "outdat/restore_conservation_probe.bin").read_bytes()
    restored, _ = launch(args, backend, target_ranks, "restored", 12, restore=source,
                         restore_probe=True, memcheck=memcheck, **common)
    for name in ("restore_probe.h5", "restore_statistics_probe.h5"):
        compare_fields(pure / "outdat" / name, restored / "outdat" / name)
    if backend == "gpu":
        assert (pure / "outdat/restore_conservation_probe.bin").read_bytes() == (restored / "outdat/restore_conservation_probe.bin").read_bytes()
    assert (f"ASTR_OUTPUT_REPARTITION air5{case_name}" in (restored / "run.log").read_text()) != exact
    assert clocks(seed)+clocks(restored) == clocks(reference)
    fields = compare_air5_scaled(reference / FINAL / "state.h5", restored / FINAL / "state.h5", scales, True)
    halos = compare_curve_target_halos(reference / FINAL / "state.h5", restored / FINAL / "state.h5", scales)
    mean = compare_mean(reference / FINAL / "statistics.h5", restored / FINAL / "statistics.h5", mean_scales(physical), 5)
    geometry = Path("outdat/new/resources/geometry.h5")
    compare_air5_scaled(reference / geometry, restored / geometry, frozen_geometry_scales(reference / geometry), True)
    resources = compare_air5_resources(reference, restored, case_name == "sbli")
    conservation = compare_conservation(reference, restored, source, scales[:11], 5, exact) if backend == "gpu" else None
    if exact:
        for name in ("state.h5", "statistics.h5"):
            compare_fields(reference / FINAL / name, restored / FINAL / name)
    cases = [(reference, FINAL), (seed, Path("outdat/new/checkpoints/step000000000005")), (restored, FINAL)]
    if repeat:
        before = fingerprints(restored / "outdat/new")
        resumed, _ = launch(args, backend, target_ranks, "second_restart", 12,
            restore=restored / "outdat/new/checkpoints/step000000000010", **common)
        for name in ("state.h5", "statistics.h5"):
            compare_fields(restored / FINAL / name, resumed / FINAL / name)
        if backend == "gpu":
            assert (restored / FINAL / "air5_conservation.bin").read_bytes() == (resumed / FINAL / "air5_conservation.bin").read_bytes()
            compare_conservation(restored, resumed,
                restored / "outdat/new/checkpoints/step000000000010", scales[:11], 10, True)
        assert before == fingerprints(restored / "outdat/new")
        cases.append((resumed, FINAL))
    for name in ("control.bin", "air5_config.bin", "archives.bin", "insitu_control.bin"):
        read = archive_schedule_payload if name == "archives.bin" else Path.read_bytes
        assert read(reference / FINAL / name) == read(restored / FINAL / name), name
    assert original == fingerprints(source_tree)
    assert digest == hashlib.sha256(args.executable.read_bytes()).hexdigest()
    host, device, reserves = [], [], []
    for case, final in cases:
        checkpoint = case / final
        state_sanity(checkpoint / "state.h5")
        host += controlled_checkpoint_buffers(checkpoint / "state.h5", 512+4096, backend)
        host += controlled_checkpoint_buffers(checkpoint / "statistics.h5", 512+4096, backend)
        with h5py.File(checkpoint / "state.h5") as state:
            rows = state["partitions"][:].reshape(-1, 8)
            for row in rows:
                cells, halo = row[3:6], int(row[6])
                face = max(int((cells[a]+1)*(cells[b]+1)) for a, b in ((0, 1), (0, 2), (1, 2)))
                host.append(int(np.prod(cells+2*halo+1))*34*8 +
                            int(np.prod(cells+1))*44*8 + 6*face*halo*34*8 + 512+4096)
                device.append((int(np.prod(cells+1))*(44+11)*8+4096+8) if backend == "gpu" else 0)
        if backend == "gpu":
            reserves.append(reserve_records(case, len(rows)))
    size = sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file())
    assert size <= BUDGET and max(host+device) <= BUDGET
    report = dict(case=case_name, backend=backend, source_np=source_ranks, target_np=target_ranks,
        exact=exact, memcheck=memcheck, second_restart=repeat, state=fields, halos=halos,
        mean=mean, conservation=conservation, resources=resources,
        pure_restore="bitwise state34,mean44,conservation", physical_scales=physical,
        mean_scales=mean_scales(physical).tolist(), controlled_host_bytes=max(host),
        controlled_device_bytes=max(device), test_root_bytes=size,
        shared_reference_root_bytes=[reference_size, seed_size],
        reserve_measurements=reserves, executable_sha256=digest)
    (tmp_path / "air5_statistics_migration.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    record_property("result", json.dumps(report))


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_exact(backend, tmp_path, record_property, reference_cache):
    execute_gate(backend, 2, 2, tmp_path, record_property, reference_cache, exact=True)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("source_ranks,target_ranks", [(1, 2), (2, 1)])
def test_migration(backend, source_ranks, target_ranks, tmp_path, record_property, reference_cache):
    execute_gate(backend, source_ranks, target_ranks, tmp_path, record_property, reference_cache, repeat=True)


def test_memcheck(tmp_path, record_property, reference_cache):
    execute_gate("gpu", 1, 2, tmp_path, record_property, reference_cache, memcheck=True)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("axis", ["x", "y"])
def test_nonperiodic_history_migration_rejected(backend, axis, tmp_path):
    args = arguments(tmp_path, backend)
    seed, _ = launch(args, backend, 1, "seed", 5, **options(backend))
    source_tree = seed / "outdat/new"
    before = fingerprints(source_tree)
    args.axis = axis
    launch(args, backend, 2, "rejected", 12,
        restore=source_tree / "checkpoints/step000000000005",
        reject="exact restore rank count", **options(backend))
    assert before == fingerprints(source_tree)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("ranks", [1, 2])
def test_history_output_does_not_change_advancement(backend, ranks, tmp_path, record_property):
    args = arguments(tmp_path, backend)
    full, _ = launch(args, backend, ranks, "full", 12, **options(backend))
    off, _ = launch(args, backend, ranks, "off", 12, statistics=False, buffer_bytes=4096)
    compare_fields(full / FINAL / "state.h5", off / FINAL / "state.h5")
    state_sanity(full / FINAL / "state.h5")
    assert not (off / FINAL / "statistics.h5").exists()
    size = sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file())
    assert size <= BUDGET
    if backend == "gpu":
        reserve_records(full, ranks)
    record_property("result", json.dumps(dict(backend=backend, np=ranks,
        output_invariance="bitwise", test_root_bytes=size,
        executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest())))
