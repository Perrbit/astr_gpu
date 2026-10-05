"""Approved noncatalytic AIR5 wall fields at the completed coupled-step hook."""
from types import SimpleNamespace
import re
import h5py
import numpy as np
import pytest

from run_output_air5_restart_validation import launch, DT, reference_scales
from run_output_restart_validation import compare_fields
from test_output_insitu_restart import ROOT, MPIEXEC

TOPOLOGIES = ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2))
FINAL = "outdat/new/checkpoints/step000000000004"
FIELD_NAMES = "rho u v w T Tv p Y_N2 Y_O2 Y_N Y_O Y_NO shear heat_tr heat_v heat_species heat_total normal".split()
REF = reference_scales()
FIELD_SCALES = np.array([REF['density'], *([REF['velocity']]*3),
    *([REF['temperature']]*2), REF['pressure'], *([1.]*5), REF['shear'],
    *([REF['heat']]*4), 1.])
CONSERVATIVE_SCALES = np.array([REF['density'], *([REF['density']*REF['velocity']]*3),
    REF['pressure'], *([REF['density']]*5), REF['pressure']])
CHECKPOINT_SCALES = np.r_[CONSERVATIVE_SCALES, CONSERVATIVE_SCALES,
    REF['density'], *([REF['velocity']]*3), REF['pressure'], *([REF['temperature']]*2), *([1.]*5)]


def arguments(output, topology):
    return SimpleNamespace(output=output, executable=ROOT / "build_insitu_air5_gpu/bin/astr",
        mpiexec=MPIEXEC, case="hbl", reconstruction=5, mean_statistics=False,
        sample_interval=2, mode="steps", initial_restart=False, filter_workspace="scalar",
        axis="xyz"[next((i for i, size in enumerate(topology) if size > 1), 0)], top_mode="characteristic")


def read_wall(path):
    with path.open("rb") as stream:
        assert stream.read(8) == b"ASTRAW01"
        header = np.fromfile(stream, "<i4", 9)
        clock = np.fromfile(stream, "<f8", 2)
        shape = tuple(header[3:6]); nodes = int(np.prod(shape))
        xyz = np.fromfile(stream, "<f8", nodes*3).reshape((*shape, 3), order="F")
        fields = np.fromfile(stream, "<f8", nodes*18).reshape((*shape, 18), order="F")
        owned = np.fromfile(stream, "<i4", nodes).reshape(shape, order="F")
        assert not stream.read(1)
    assert np.isfinite(fields).all() and np.isfinite(xyz).all() and np.isfinite(clock).all()
    np.testing.assert_allclose(clock, [header[1]*DT, DT], atol=1e-25, rtol=0)
    assert np.isin(owned, [0, 1]).all()
    return header, xyz, fields, owned


@pytest.fixture(scope="module", params=TOPOLOGIES, ids=("single", "x", "y", "z"))
def reference(request, tmp_path_factory):
    topology = request.param; ranks = int(np.prod(topology))
    args = arguments(tmp_path_factory.mktemp("air5_wall"), topology)
    cases = {}
    for backend in ("cpu", "gpu"):
        cases[backend], _ = launch(args, backend, ranks, "reference", 4, interval=1, wall_samples=True)
    return args, topology, ranks, cases


def global_fields(case, ranks):
    coverage = np.zeros((17, 16), dtype=int)
    result = np.zeros((17, 16, 18))
    for rank in range(ranks):
        header, xyz, fields, owned = read_wall(case / f"outdat/sample.air5_wall.step00000004.rank{rank:08d}.bin")
        for i, k, w in np.ndindex(owned.shape):
            if not owned[i, k, w]:
                continue
            gi, gk = header[6]+i, header[8]+k
            coverage[gi, gk] += 1; result[gi, gk] = fields[i, k, w]
            np.testing.assert_allclose(xyz[i, k, w], [gi*.08/16, 0, gk*.002/16], rtol=0, atol=2e-17)
    np.testing.assert_array_equal(coverage, 1)
    assert (result[..., 0] > 0).all() and (result[..., 4:7] > 0).all()
    assert (result[..., 7:12] >= 0).all()
    np.testing.assert_allclose(result[..., 7:12].sum(axis=-1), 1, rtol=0, atol=2e-15)
    np.testing.assert_array_equal(result[..., 15], 0)  # Zero species enthalpy flux, not an omitted contribution.
    np.testing.assert_array_equal(result[..., 16], result[..., 13]+result[..., 14]+result[..., 15])
    np.testing.assert_array_equal(result[..., 17], 1)
    return result


def test_air5_complete_step_wall_fields(reference, record_property):
    _, topology, ranks, cases = reference
    cpu = global_fields(cases["cpu"], ranks); gpu = global_fields(cases["gpu"], ranks)
    error = float(np.max(abs(cpu-gpu)))
    record_property("topology", topology)
    record_property("wall_fields_maxabs", error)
    scaled_error = float(np.max(abs(cpu-gpu)/FIELD_SCALES))
    record_property("wall_fields_reference_scaled_maxabs", scaled_error)
    for i, name in enumerate(FIELD_NAMES):
        difference = float(np.max(abs(cpu[..., i]-gpu[..., i])))
        record_property(name + "_reference_scale", float(FIELD_SCALES[i]))
        record_property(name + "_maxabs", difference)
        record_property(name + "_reference_scaled_maxabs", difference/FIELD_SCALES[i])
    # Independently check primitive fields against the authoritative completed-step cache.
    names = [23, 24, 25, 26, 28, 29, 27, 30, 31, 32, 33, 34]
    for backend, case in cases.items():
        with h5py.File(case / FINAL / "state.h5") as state:
            expected = np.stack([state[f"q{n:04d}"][...].transpose(2, 1, 0)[:, 0, :16] for n in names], axis=-1)
        actual = cpu if backend == "cpu" else gpu
        record_property(backend + "_own_cache_maxabs", float(np.max(abs(actual[..., :12]-expected))))
        np.testing.assert_allclose(actual[..., :12], expected, atol=2e-10, rtol=0)
    assert scaled_error <= 2e-10, scaled_error
    # Also inspect the complete global state/cache, not only a potentially quiet wall.
    with h5py.File(cases['cpu'] / FINAL / 'state.h5') as a, h5py.File(cases['gpu'] / FINAL / 'state.h5') as b:
        worst = 0.
        for component, scale in enumerate(CHECKPOINT_SCALES, start=1):
            name = f'q{component:04d}'
            difference = float(np.max(abs(a[name][...]-b[name][...])))
            record_property('checkpoint_'+name+'_maxabs', difference)
            record_property('checkpoint_'+name+'_reference_scale', float(scale))
            worst = max(worst, difference/scale)
        record_property('complete_state_cache_reference_scaled_maxabs', worst)
        assert worst <= 2e-10, worst


@pytest.mark.parametrize("backend", ("cpu", "gpu"))
def test_air5_wall_exact_continuation(reference, backend, tmp_path):
    args, topology, ranks, cases = reference
    source = cases[backend] / "outdat/new/checkpoints/step000000000003"
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    args = arguments(tmp_path, topology)
    resumed, _ = launch(args, backend, ranks, "restart", 4, restore=source, interval=1, wall_samples=True)
    compare_fields(cases[backend] / FINAL / "state.h5", resumed / FINAL / "state.h5")
    assert (cases[backend] / FINAL / "control.bin").read_bytes() == (resumed / FINAL / "control.bin").read_bytes()
    for rank in range(ranks):
        name = f"outdat/sample.air5_wall.step00000004.rank{rank:08d}.bin"
        assert (cases[backend] / name).read_bytes() == (resumed / name).read_bytes()
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}


def test_air5_wall_sampling_isolation(reference, tmp_path):
    _, topology, ranks, cases = reference
    for backend in ("cpu", "gpu"):
        args = arguments(tmp_path / backend, topology)
        plain, _ = launch(args, backend, ranks, "plain", 4, interval=1)
        compare_fields(cases[backend] / FINAL / "state.h5", plain / FINAL / "state.h5")
    if topology == (1, 2, 1):
        header, _, fields, owned = read_wall(cases["gpu"] / "outdat/sample.air5_wall.step00000004.rank00000001.bin")
        assert fields.size == 0 and owned.size == 0 and header[5] == 0
        records = re.findall(r"ASTR_INSITU_AIR5_WALL rank=1 step=\d+ owned_nodes=0 field_download_bytes=0",
                             (cases["gpu"] / "run.log").read_text())
        assert len(records) == 4
