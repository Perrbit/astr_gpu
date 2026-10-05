"""Bounded bc41 wall-only extraction at the real completed-step hook."""
from pathlib import Path
import re
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from run_output_restart_validation import run_case, compare_fields
from test_insitu_is4 import compare_numerical
from test_output_insitu_restart import ROOT, MPIEXEC

TOPOLOGIES = ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2))
FINAL = "outdat/new/checkpoints/step000000000004"


def arguments(output, backend, samples=True):
    return SimpleNamespace(output=output, executable=ROOT / f"build_insitu_{'gpu' if backend == 'gpu' else 'check'}/bin/astr",
        mpiexec=MPIEXEC, case="channel", mode="steps", restart_step=3, initial_dimension=3,
        legacy_statistics=False, statistics=False, initial_restart=False, filter_workspace="scalar",
        force="fixed", axis="x", no_samples=True, wall_samples=samples,
        directory_budget_bytes=256 * 1024**2)


def read_wall(path):
    with path.open("rb") as stream:
        assert stream.read(8) == b"ASTRIW01"
        header = np.fromfile(stream, "<i4", 9)
        clock = np.fromfile(stream, "<f8", 2)
        shape = tuple(header[3:6])
        nodes = int(np.prod(shape))
        xyz = np.fromfile(stream, "<f8", nodes * 3).reshape((*shape, 3), order="F")
        values = np.fromfile(stream, "<f8", nodes * 4).reshape((*shape, 4), order="F")
        owned = np.fromfile(stream, "<i4", nodes).reshape(shape, order="F")
        assert not stream.read(1)
    assert header[0] == 1 and np.isfinite(xyz).all() and np.isfinite(values).all()
    assert np.isfinite(clock).all() and clock[1] > 0
    np.testing.assert_allclose(clock, [header[1] * .001, .001], rtol=0, atol=1e-15)
    assert np.isin(values[..., 3], [-1., 1.]).all()
    expected = np.zeros(shape, dtype=int)
    expected[:-1, :-1, :] = 1
    np.testing.assert_array_equal(owned, expected)
    return header, xyz, values, owned


def global_component(path, name):
    with h5py.File(path) as state:
        return state[name][...].transpose(2, 1, 0)


def check_wall(case, ranks, step):
    state_file = case / f"outdat/new/checkpoints/step{step:012d}/state.h5"
    geometry_file = case / "outdat/new/resources/geometry.h5"
    q = np.stack([global_component(state_file, f"q{c:04d}") for c in range(1, 6)], axis=-1)
    metric = global_component(geometry_file, "q0009")
    coordinates = np.stack([global_component(geometry_file, f"q{c:04d}") for c in range(1, 4)], axis=-1)
    coverage = np.zeros((16, 16, 2), dtype=int)
    result = np.empty((16, 16, 2, 4))
    worst = 0.
    for rank in range(ranks):
        path = case / f"outdat/sample.wall.step{step:08d}.rank{rank:08d}.bin"
        header, xyz, fields, owned = read_wall(path)
        for i, k, wall in np.ndindex(owned.shape):
            gi, gk = int(header[6] + i), int(header[8] + k)
            side = int(fields[i, k, wall, 3])
            gj = 0 if side == 1 else 16
            np.testing.assert_array_equal(xyz[i, k, wall], coordinates[gi, gj, gk])
            line = q[gi % 16, [gj + side * n for n in range(3)], gk % 16]
            rho = line[:, 0]
            velocity = line[:, 1:4] / rho[:, None]
            pressure = (line[:, 4] - .5 * rho * np.sum(velocity**2, axis=1)) / 2.5
            temperature = pressure / rho * (1.4 * .3**2)
            assert (rho > 0).all() and (temperature > 0).all()
            t = temperature[0]
            sutherland = 110.4 / 273.15
            mu = t * np.sqrt(t) * (1 + sutherland) / (t + sutherland) / 3000
            du = (-.5 * velocity[2, 0] + 2 * velocity[1, 0] - 1.5 * velocity[0, 0]) * metric[gi, gj, gk]
            dt = (-.5 * temperature[2] + 2 * temperature[1] - 1.5 * temperature[0]) * metric[gi, gj, gk]
            expected = [pressure[0], mu * du, -((mu / .72) / ((1.4 - 1) * .3**2)) * dt, side]
            error = float(np.max(abs(fields[i, k, wall] - expected)))
            assert error <= 2e-10, (rank, i, k, wall, error)
            worst = max(worst, error)
            if owned[i, k, wall]:
                wi = 0 if side == 1 else 1
                coverage[gi, gk, wi] += 1
                result[gi, gk, wi] = fields[i, k, wall]
    np.testing.assert_array_equal(coverage, 1)
    return result, worst


@pytest.fixture(scope="module", params=TOPOLOGIES, ids=("single", "x", "y", "z"))
def reference(request, tmp_path_factory):
    topology = request.param
    ranks = int(np.prod(topology))
    cases = {}
    for backend in ("cpu", "gpu"):
        args = arguments(tmp_path_factory.mktemp("channel_wall_" + backend), backend)
        cases[backend], _ = run_case(args, ROOT, backend, ranks, "reference", 4,
                                    checkpoint_interval=1, topology=topology)
    return topology, ranks, cases


def test_complete_step_wall_fields(reference, record_property):
    topology, ranks, cases = reference
    cpu, a = check_wall(cases["cpu"], ranks, 4)
    gpu, b = check_wall(cases["gpu"], ranks, 4)
    error = float(np.max(abs(cpu - gpu)))
    assert error <= 2e-10
    state_error = compare_numerical(cases["cpu"] / FINAL / "state.h5", cases["gpu"] / FINAL / "state.h5")
    record_property("topology", topology)
    record_property("wall_oracle_maxabs", max(a, b))
    record_property("cpu_gpu_wall_maxabs", error)
    record_property("cpu_gpu_state_maxabs", state_error)
    transfers = re.findall(r"ASTR_INSITU_WALL rank=(\d+) step=(\d+) owned_nodes=(\d+) field_download_bytes=(\d+)",
                          (cases["gpu"] / "run.log").read_text())
    assert len(transfers) == 4 * ranks
    for rank, step, owned, download in transfers:
        header, _, _, mask = read_wall(cases["gpu"] / f"outdat/sample.wall.step{int(step):08d}.rank{int(rank):08d}.bin")
        assert int(owned) == int(mask.sum())
        assert int(download) == int(np.prod(header[3:6])) * 15 * 8


@pytest.mark.parametrize("backend", ("cpu", "gpu"))
def test_wall_output_isolation_and_exact_restart(reference, backend, tmp_path):
    topology, ranks, cases = reference
    baseline = cases[backend]
    source = baseline / "outdat/new/checkpoints/step000000000003"
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    args = arguments(tmp_path / "restart", backend)
    restarted, _ = run_case(args, ROOT, backend, ranks, "restart", 4, restore=source,
                            checkpoint_interval=1, topology=topology)
    for name in ("state.h5", "control.bin", "insitu_control.bin"):
        if name.endswith(".h5"):
            compare_fields(baseline / FINAL / name, restarted / FINAL / name)
        else:
            assert (baseline / FINAL / name).read_bytes() == (restarted / FINAL / name).read_bytes()
    for rank in range(ranks):
        name = f"sample.wall.step00000004.rank{rank:08d}.bin"
        assert (baseline / "outdat" / name).read_bytes() == (restarted / "outdat" / name).read_bytes()
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    args = arguments(tmp_path / "off", backend, samples=False)
    plain, _ = run_case(args, ROOT, backend, ranks, "off", 4, checkpoint_interval=1, topology=topology)
    compare_fields(baseline / FINAL / "state.h5", plain / FINAL / "state.h5")
    assert not list((plain / "outdat").glob("sample.wall.*"))


def test_wall_pack_memory_safety(tmp_path):
    args = arguments(tmp_path, "gpu")
    case, _ = run_case(args, ROOT, "gpu", 2, "memcheck", 2, topology=(1, 2, 1),
                       checkpoint_interval=1, memcheck=True)
    assert len(list(case.glob("memcheck.*.log"))) == 2
    for rank in range(2):
        read_wall(case / f"outdat/sample.wall.step00000002.rank{rank:08d}.bin")
