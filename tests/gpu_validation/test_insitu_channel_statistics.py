"""Completed-step velocity statistics on a stretched, nonperiodic channel."""
import numpy as np
import pytest
import re

from run_output_restart_validation import run_case, compare_fields
from test_insitu_channel_walls import arguments, TOPOLOGIES, FINAL, global_component
from test_output_insitu_restart import ROOT, configuration
from test_insitu_is4 import compare_numerical
from run_insitu_gpu_derivatives import check_resources


def config(render=False):
    return configuration(render=render, statistics=True, interval=2).replace(
        "statistics_window=0.0005,0.0115", "statistics_window=0.003,0.0115").replace(
        "render=" + (".true." if render else ".false.") + ",",
        "products='channel_walls',render=" + (".true." if render else ".false.") + ",")


@pytest.fixture(scope="module", params=TOPOLOGIES, ids=("single", "x", "y", "z"))
def reference(request, tmp_path_factory):
    topology = request.param
    ranks = int(np.prod(topology))
    cases = {}
    for backend in ("cpu", "gpu"):
        args = arguments(tmp_path_factory.mktemp("channel_statistics_" + backend), backend, samples=False)
        args.statistics = True
        cases[backend], _ = run_case(args, ROOT, backend, ranks, "reference", 4,
            checkpoint_interval=1, topology=topology, insitu_config=config())
    return topology, ranks, cases


def read_statistics(path):
    with path.open("rb") as stream:
        assert stream.read(8) == b"ASTRST01"
        header = np.fromfile(stream, "<i4", 9)
        meta = np.fromfile(stream, "<f8", 10)
        values = np.fromfile(stream, "<f8").reshape((*header[3:6], 41), order="F")
    assert np.isfinite(meta).all() and np.isfinite(values).all()
    return header, meta, values


def test_complete_step_statistics(reference, record_property):
    topology, ranks, cases = reference
    worst, rms_error = 0., 0.
    for rank in range(ranks):
        name = f"outdat/render/sample.statistics.step00000004.rank{rank:08d}.bin"
        ch, cm, cpu = read_statistics(cases["cpu"] / name)
        gh, gm, gpu = read_statistics(cases["gpu"] / name)
        np.testing.assert_array_equal(ch, gh)
        # RMS itself amplifies differences near zero; its square uses the approved gate.
        rms_error = max(rms_error, float(np.max(abs(cpu[..., 8:14]-gpu[..., 8:14]))))
        cpu[..., 8:14] **= 2; gpu[..., 8:14] **= 2
        worst = max(worst, float(np.max(abs(cpu-gpu))))
        np.testing.assert_allclose(cpu, gpu, rtol=0, atol=2e-10)
        cm[4:] **= 2; gm[4:] **= 2
        np.testing.assert_allclose(cm, gm, rtol=0, atol=2e-10)
        assert abs(cm[3]-4*np.pi**2) <= 2e-10  # (2*pi) * 2 * pi, not uniform-y volume.
        assert np.all(cpu[..., 0] == .001)
    state_error = compare_numerical(cases["cpu"] / FINAL / "state.h5", cases["gpu"] / FINAL / "state.h5")
    record_property("statistics_maxabs", worst)
    record_property("rms_raw_maxabs", rms_error)
    record_property("state_maxabs", state_error)
    record_property("topology", topology)


def test_independent_endpoint_moments(reference):
    _, ranks, cases = reference
    for backend, case in cases.items():
        samples = []
        for step in (3, 4):
            path = case / f"outdat/new/checkpoints/step{step:012d}/state.h5"
            q = np.stack([global_component(path, f"q{c:04d}") for c in range(1, 5)], axis=-1)
            q[-1] = q[0]; q[:, :, -1] = q[:, :, 0]
            samples.append((q[..., 0], q[..., 1:4] / q[..., 0, None]))
        ra, ua = samples[0]; rb, ub = samples[1]
        delta = ub-ua
        covariance = .25 * delta[..., :, None] * delta[..., None, :]
        favre = (ra[..., None]*ua + rb[..., None]*ub) / (ra+rb)[..., None]
        cov_f = (ra*rb/(ra+rb)**2)[..., None, None] * delta[..., :, None] * delta[..., None, :]
        for rank in range(ranks):
            header, _, result = read_statistics(case / f"outdat/render/sample.statistics.step00000004.rank{rank:08d}.bin")
            i, j, k = map(int, header[6:9]); nx, ny, nz = map(int, header[3:6])
            sl = np.s_[i:i+nx, j:j+ny, k:k+nz]
            np.testing.assert_allclose(result[..., 1], (.5*(ra+rb))[sl], rtol=0, atol=2e-10)
            np.testing.assert_allclose(result[..., 2:5], (.5*(ua+ub))[sl], rtol=0, atol=2e-10)
            np.testing.assert_allclose(result[..., 5:8], favre[sl], rtol=0, atol=2e-10)
            for offset, expected in ((14, covariance), (23, cov_f), (32, .5*(ra+rb)[..., None, None]*cov_f)):
                np.testing.assert_allclose(result[..., offset:offset+9].reshape((*result.shape[:3], 3, 3), order="F"),
                    expected[sl], rtol=0, atol=2e-10)


@pytest.mark.parametrize("backend", ("cpu", "gpu"))
def test_exact_statistics_continuation(reference, backend, tmp_path):
    topology, ranks, cases = reference
    source = cases[backend] / "outdat/new/checkpoints/step000000000003"
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    args = arguments(tmp_path, backend, samples=False)
    args.statistics = True
    resumed, _ = run_case(args, ROOT, backend, ranks, "restart", 4, restore=source,
        checkpoint_interval=1, topology=topology, insitu_config=config())
    for name in ("state.h5", "statistics.h5"):
        compare_fields(cases[backend] / FINAL / name, resumed / FINAL / name)
    assert (cases[backend] / FINAL / "control.bin").read_bytes() == (resumed / FINAL / "control.bin").read_bytes()
    for rank in range(ranks):
        name = f"outdat/render/sample.statistics.step00000004.rank{rank:08d}.bin"
        assert (cases[backend] / name).read_bytes() == (resumed / name).read_bytes()
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}


def test_statistics_does_not_change_flow(reference, tmp_path):
    topology, ranks, cases = reference
    for backend in ("cpu", "gpu"):
        args = arguments(tmp_path / backend, backend, samples=False)
        plain, _ = run_case(args, ROOT, backend, ranks, "plain", 4,
            checkpoint_interval=1, topology=topology)
        compare_fields(plain / FINAL / "state.h5", cases[backend] / FINAL / "state.h5")


def test_device_statistics_residency(reference):
    _, ranks, cases = reference
    log = (cases["gpu"] / "run.log").read_text()
    records = re.findall(r"ASTR_INSITU_GPU_STATS rank=(\d+) samples=(\d+) full_output_downloads=(\d+)", log)
    assert sorted(records) == [(str(rank), "5", "1") for rank in range(ranks)]
    assert len(re.findall(r"ASTR_INSITU_GPU_FLOW rank=\d+ frame_downloads=0", log)) == ranks


def test_wall_render_with_statistics(reference, tmp_path):
    topology, ranks, cases = reference
    if topology != (1, 2, 1):
        pytest.skip("joint renderer gate uses the normal-direction decomposition")
    args = arguments(tmp_path, "gpu", samples=False)
    args.statistics = True
    rendered, _ = run_case(args, ROOT, "gpu", ranks, "render", 4, checkpoint_interval=1,
        topology=topology, insitu_config=config(render=True))
    check_resources(rendered, ranks, steps=(2, 4), capture=False, statistics=True)
    for name in ("state.h5", "statistics.h5"):
        compare_fields(cases["gpu"] / FINAL / name, rendered / FINAL / name)
    source = rendered / "outdat/new/checkpoints/step000000000003"
    resumed, _ = run_case(args, ROOT, "gpu", ranks, "render_restart", 4, restore=source,
        checkpoint_interval=1, topology=topology, insitu_config=config(render=True))
    check_resources(resumed, ranks, steps=(4,), capture=False, statistics=True)
    for name in ("state.h5", "statistics.h5"):
        compare_fields(rendered / FINAL / name, resumed / FINAL / name)
    for name in ("control.bin", "insitu_control.bin"):
        assert (rendered / FINAL / name).read_bytes() == (resumed / FINAL / name).read_bytes()
    for path in (rendered / "outdat/render").glob("*.step00000004.*"):
        assert path.read_bytes() == (resumed / "outdat/render" / path.name).read_bytes()


def test_channel_statistics_memory_safety(tmp_path):
    args = arguments(tmp_path, "gpu", samples=False)
    args.statistics = True
    case, _ = run_case(args, ROOT, "gpu", 2, "memcheck", 4, checkpoint_interval=3,
        topology=(1, 2, 1), insitu_config=config(), memcheck=True)
    logs = list(case.glob("memcheck.*.log"))
    assert len(logs) == 2
    for path in logs:
        assert "ERROR SUMMARY: 0 errors" in path.read_text()
