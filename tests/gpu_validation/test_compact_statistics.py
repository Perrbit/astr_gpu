#!/usr/bin/env python3
"""Tests for the compact GPU production-statistics sidecar contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/gpu_statistics/compact_statistics.py"
SPEC = importlib.util.spec_from_file_location("compact_statistics", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def load_assembler():
    module_path = ROOT / "scripts/gpu_statistics/assemble_compact_statistics.py"
    spec = importlib.util.spec_from_file_location("assemble_compact_statistics", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(module_path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def load_host_reference():
    module_path = ROOT / "tests/gpu_validation/compact_statistics_host_reference.py"
    spec = importlib.util.spec_from_file_location("compact_statistics_host_reference", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_state(*, has_wall: bool = True):
    ni, nj, nk = 2, 3, 5
    measure = np.array(
        [[2.0, 2.5, 3.0], [3.5, 4.0, 4.5]], dtype=np.float64, order="F"
    )
    rho_mean = np.array(
        [[1.0, 1.1, 1.2], [1.3, 1.4, 1.5]], dtype=np.float64, order="F"
    )
    sample_count = 4
    rho_sum = sample_count * measure * rho_mean
    u = np.full((ni, nj), 2.0)
    v = np.full((ni, nj), -0.25)
    w = np.full((ni, nj), 0.5)
    temperature = np.full((ni, nj), 1.75)
    pressure = np.full((ni, nj), 0.8)
    variances = (0.4, 0.3, 0.2, -0.05, 0.07, -0.02)

    plane_sum = np.empty((ni, nj, 12), dtype=np.float64, order="F")
    plane_sum[:, :, 0] = rho_sum
    plane_sum[:, :, 1] = rho_sum * u
    plane_sum[:, :, 2] = rho_sum * v
    plane_sum[:, :, 3] = rho_sum * w
    plane_sum[:, :, 4] = rho_sum * temperature
    plane_sum[:, :, 5] = sample_count * measure * pressure
    plane_sum[:, :, 6] = rho_sum * (u * u + variances[0])
    plane_sum[:, :, 7] = rho_sum * (v * v + variances[1])
    plane_sum[:, :, 8] = rho_sum * (w * w + variances[2])
    plane_sum[:, :, 9] = rho_sum * (u * v + variances[3])
    plane_sum[:, :, 10] = rho_sum * (u * w + variances[4])
    plane_sum[:, :, 11] = rho_sum * (v * w + variances[5])

    header = MODULE.CompactHeader(
        global_dims=(9, 7, 17),
        topology=(2, 1, 2),
        rank=0,
        rank_coords=(0, 0, 0),
        offsets=(0, 0, 0),
        local_dims=(ni, nj, nk),
        checkpoint_step=120,
        checkpoint_time=0.06,
        sample_count=sample_count,
        sampling_start_step=90,
        sampling_start_time=0.045,
        has_wall=has_wall,
    )
    wall_measure = np.array([2.0, 2.5], dtype=np.float64) if has_wall else None
    wall_sum = None
    if has_wall:
        wall_sum = np.asfortranarray(
            np.array([[6.4, 0.08, -0.12], [8.0, 0.10, -0.15]], dtype=np.float64)
        )
    return MODULE.CompactRankState(
        header=header,
        plane_measure=measure,
        plane_sum=plane_sum,
        wall_measure=wall_measure,
        wall_sum=wall_sum,
    ), rho_mean, variances


@pytest.mark.parametrize("has_wall", [False, True])
def test_sidecar_round_trip_preserves_header_and_fortran_array_order(tmp_path, has_wall):
    state, _, _ = make_state(has_wall=has_wall)
    path = tmp_path / "compact.bin"

    MODULE.write_rank_file(path, state)
    restored = MODULE.read_rank_file(path)

    assert restored.header == state.header
    np.testing.assert_array_equal(restored.plane_measure, state.plane_measure)
    np.testing.assert_array_equal(restored.plane_sum, state.plane_sum)
    assert restored.plane_measure.flags.f_contiguous
    assert restored.plane_sum.flags.f_contiguous
    if has_wall:
        np.testing.assert_array_equal(restored.wall_measure, state.wall_measure)
        np.testing.assert_array_equal(restored.wall_sum, state.wall_sum)
        assert restored.wall_sum.flags.f_contiguous
    else:
        assert restored.wall_measure is None
        assert restored.wall_sum is None


def test_derive_statistics_reconstructs_favre_means_and_stresses():
    state, rho_mean, variances = make_state()

    derived = MODULE.derive_statistics(state)

    np.testing.assert_allclose(derived["rho_mean"], rho_mean)
    np.testing.assert_allclose(derived["u_favre"], 2.0)
    np.testing.assert_allclose(derived["v_favre"], -0.25)
    np.testing.assert_allclose(derived["w_favre"], 0.5)
    np.testing.assert_allclose(derived["T_favre"], 1.75)
    np.testing.assert_allclose(derived["p_mean"], 0.8)
    for name, expected in zip(("uu", "vv", "ww", "uv", "uw", "vw"), variances):
        np.testing.assert_allclose(derived[f"{name}_favre"], expected, atol=1.0e-15)
    np.testing.assert_allclose(derived["wall_p_mean"], [0.8, 0.8])
    np.testing.assert_allclose(derived["wall_tau_streamwise_mean"], [0.01, 0.01])
    np.testing.assert_allclose(derived["wall_q_normal_mean"], [-0.015, -0.015])


def test_derive_statistics_rejects_zero_measure_and_density():
    state, _, _ = make_state()
    state.plane_measure[0, 0] = 0.0
    with pytest.raises(ValueError, match="positive plane measure"):
        MODULE.derive_statistics(state)

    state, _, _ = make_state()
    state.plane_sum[0, 0, 0] = 0.0
    with pytest.raises(ValueError, match="positive density sum"):
        MODULE.derive_statistics(state)


@pytest.mark.parametrize(
    ("offset", "replacement", "message"),
    [
        (0, b"BADMAGIC", "magic"),
        (8, np.asarray([2], dtype="<i4").tobytes(), "version"),
        (12, np.asarray([0x04030201], dtype="<i4").tobytes(), "endianness"),
        (16, np.asarray([4], dtype="<i4").tobytes(), "real size"),
    ],
)
def test_reader_rejects_incompatible_fixed_header(tmp_path, offset, replacement, message):
    state, _, _ = make_state()
    path = tmp_path / "compact.bin"
    MODULE.write_rank_file(path, state)
    payload = bytearray(path.read_bytes())
    payload[offset : offset + len(replacement)] = replacement
    path.write_bytes(payload)

    with pytest.raises(ValueError, match=message):
        MODULE.read_rank_file(path)


def test_reader_rejects_truncated_and_trailing_payload(tmp_path):
    state, _, _ = make_state()
    path = tmp_path / "compact.bin"
    MODULE.write_rank_file(path, state)
    payload = path.read_bytes()
    path.write_bytes(payload[:-8])
    with pytest.raises(ValueError, match="truncated"):
        MODULE.read_rank_file(path)
    path.write_bytes(payload + b"stale")
    with pytest.raises(ValueError, match="trailing bytes"):
        MODULE.read_rank_file(path)


def test_merge_rejects_checkpoint_generation_mismatch():
    states, _, _ = make_2x2x1_rank_states()
    states[3].header.checkpoint_step += 1
    with pytest.raises(ValueError, match="checkpoint steps"):
        MODULE.merge_rank_states(states)

    states, _, _ = make_2x2x1_rank_states()
    states[3].header.topology = (1, 4, 1)
    with pytest.raises(ValueError, match="missing ranks|topologies|rank coordinates"):
        MODULE.merge_rank_states(states)


def make_2x2x1_rank_states():
    global_measure = np.asfortranarray(
        np.array([[2.0, 2.2, 2.4], [2.5, 2.7, 2.9], [3.0, 3.2, 3.4]])
    )
    global_sum = np.empty((3, 3, 12), dtype=np.float64, order="F")
    for moment in range(12):
        global_sum[:, :, moment] = (moment + 1) * global_measure
    global_sum[:, :, 0] *= 4.0
    states = []
    layout = (
        (0, (0, 0, 0), (0, 0, 0)),
        (1, (1, 0, 0), (1, 0, 0)),
        (2, (0, 1, 0), (0, 1, 0)),
        (3, (1, 1, 0), (1, 1, 0)),
    )
    for rank, coords, offsets in layout:
        i0, j0, _ = offsets
        has_wall = coords[1] == 0
        wall_measure = np.array([1.5 + i0, 2.5 + i0]) if has_wall else None
        wall_sum = None
        if has_wall:
            wall_sum = np.asfortranarray(
                np.column_stack((4.0 * wall_measure, 0.2 * wall_measure, -0.1 * wall_measure))
            )
        states.append(
            MODULE.CompactRankState(
                header=MODULE.CompactHeader(
                    global_dims=(3, 3, 3),
                    topology=(2, 2, 1),
                    rank=rank,
                    rank_coords=coords,
                    offsets=offsets,
                    local_dims=(2, 2, 3),
                    checkpoint_step=40,
                    checkpoint_time=0.02,
                    sample_count=2,
                    sampling_start_step=20,
                    sampling_start_time=0.01,
                    has_wall=has_wall,
                ),
                plane_measure=np.asfortranarray(global_measure[i0 : i0 + 2, j0 : j0 + 2]),
                plane_sum=np.asfortranarray(global_sum[i0 : i0 + 2, j0 : j0 + 2, :]),
                wall_measure=wall_measure,
                wall_sum=wall_sum,
            )
        )
    return states, global_measure, global_sum


def write_rank_states(directory, states):
    directory.mkdir()
    for state in states:
        MODULE.write_rank_file(
            directory / f"compact_stats.rank{state.header.rank:08d}.bin", state
        )


def test_assemble_2x2x1_writes_global_hdf5_and_metadata(tmp_path):
    assembler = load_assembler()
    states, global_measure, global_sum = make_2x2x1_rank_states()
    input_dir = tmp_path / "sidecars"
    write_rank_states(input_dir, states)
    output_h5 = tmp_path / "compact.h5"
    metadata = tmp_path / "compact.txt"

    assembler.assemble(input_dir, output_h5, metadata)

    with h5py.File(output_h5, "r") as handle:
        np.testing.assert_allclose(handle["measure/plane"][...], global_measure)
        np.testing.assert_allclose(handle["raw/rho"][...], global_sum[:, :, 0])
        assert int(handle["metadata/sample_count"][()]) == 2
        assert tuple(handle["metadata/topology"][...]) == (2, 2, 1)
        assert handle["wall/tau_streamwise"].shape == (3,)
    report = metadata.read_text(encoding="ascii")
    assert "topology: 2 2 1" in report
    assert "rank_files: 4" in report
    assert "checkpoint_step: 40" in report


def test_assemble_rejects_overlap_mismatch(tmp_path):
    assembler = load_assembler()
    states, _, _ = make_2x2x1_rank_states()
    states[1].plane_sum[0, 0, 0] += 2.0e-8
    input_dir = tmp_path / "sidecars"
    write_rank_states(input_dir, states)

    with pytest.raises(ValueError, match="overlap mismatch"):
        assembler.assemble(input_dir, tmp_path / "compact.h5", tmp_path / "compact.txt")


def test_assemble_rejects_missing_rank(tmp_path):
    assembler = load_assembler()
    states, _, _ = make_2x2x1_rank_states()
    input_dir = tmp_path / "sidecars"
    write_rank_states(input_dir, states[:-1])

    with pytest.raises(ValueError, match="missing ranks"):
        assembler.assemble(input_dir, tmp_path / "compact.h5", tmp_path / "compact.txt")


def test_independent_host_reference_integrates_all_plane_moments():
    reference = load_host_reference()
    z = np.array([0.0, 0.2, 0.7, 1.5])
    x = np.zeros((1, 1, z.size, 3))
    x[0, 0, :, 0] = 0.1 * z**2
    x[0, 0, :, 1] = 0.05 * z
    x[0, 0, :, 2] = z
    rho = (1.0 + 0.2 * z)[None, None, :]
    velocity = np.empty((1, 1, z.size, 3))
    velocity[0, 0, :, 0] = 2.0 + z
    velocity[0, 0, :, 1] = -0.3 + 0.5 * z
    velocity[0, 0, :, 2] = 0.4 - 0.1 * z
    pressure = (0.8 + 0.05 * z)[None, None, :]
    temperature = (1.2 + 0.3 * z)[None, None, :]

    measure, moments = reference.integrate_plane_moments(
        x, rho, velocity, pressure, temperature
    )

    expected_measure = np.sum(np.linalg.norm(np.diff(x[0, 0, :, :], axis=0), axis=1))
    assert measure[0, 0] == pytest.approx(expected_measure)
    endpoint_rho_u = rho[0, 0, :] * velocity[0, 0, :, 0]
    segment_length = np.linalg.norm(np.diff(x[0, 0, :, :], axis=0), axis=1)
    expected_rho_u = np.sum(0.5 * (endpoint_rho_u[:-1] + endpoint_rho_u[1:]) * segment_length)
    assert moments[0, 0, 1] == pytest.approx(expected_rho_u)
    assert moments.shape == (1, 1, 12)


def test_reader_rejects_truncated_and_trailing_payload(tmp_path):
    state, _, _ = make_state()
    path = tmp_path / "compact.bin"
    MODULE.write_rank_file(path, state)
    payload = path.read_bytes()

    path.write_bytes(payload[:-8])
    with pytest.raises(ValueError, match="truncated"):
        MODULE.read_rank_file(path)

    path.write_bytes(payload + b"trailing")
    with pytest.raises(ValueError, match="trailing"):
        MODULE.read_rank_file(path)


def test_writer_rejects_shape_and_wall_contract_violations(tmp_path):
    state, _, _ = make_state()
    state.plane_sum = state.plane_sum[:, :, :11]
    with pytest.raises(ValueError, match="plane_sum shape"):
        MODULE.write_rank_file(tmp_path / "bad-shape.bin", state)

    state, _, _ = make_state(has_wall=False)
    state.wall_measure = np.ones(2)
    with pytest.raises(ValueError, match="wall arrays"):
        MODULE.write_rank_file(tmp_path / "bad-wall.bin", state)


def write_validation_snapshot(path, *, has_wall=True):
    reference = load_host_reference()
    ni, nj, nk, hm = 3, 2, 4, 1
    fixed = np.asarray(
        (
            reference.VALIDATION_VERSION,
            reference.ENDIAN_MARKER,
            8,
            int(has_wall),
            1,
            ni,
            nj,
            nk,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            ni,
            nj,
            nk,
            hm,
        ),
        dtype="<i4",
    )
    z = np.linspace(0.0, 1.0, nk)
    coordinates = np.zeros((ni, nj, nk, 3), dtype=np.float64, order="F")
    coordinates[:, :, :, 0] = np.arange(ni)[:, None, None]
    coordinates[:, :, :, 1] = np.arange(nj)[None, :, None]
    coordinates[:, :, :, 2] = z[None, None, :]
    density = np.ones((ni, nj, nk), dtype=np.float64, order="F")
    velocity = np.zeros((ni, nj, nk, 3), dtype=np.float64, order="F")
    velocity[:, :, :, 0] = 0.5 + z[None, None, :]
    pressure = np.full((ni, nj, nk), 0.8, dtype=np.float64, order="F")
    temperature = np.ones((ni, nj, nk), dtype=np.float64, order="F")
    velocity_gradient = np.zeros((ni, nj, nk, 3, 3), dtype=np.float64, order="F")
    velocity_gradient[:, 0, :, 0, 1] = 2.0
    temperature_gradient = np.zeros((ni, nj, nk, 3), dtype=np.float64, order="F")
    temperature_gradient[:, 0, :, 1] = 3.0
    wall_coordinates = np.zeros((ni + 2 * hm, nk, 3), dtype=np.float64, order="F")
    wall_coordinates[:, :, 0] = np.arange(-hm, ni + hm)[:, None]
    wall_coordinates[:, :, 2] = z[None, :]
    wall_normal = np.zeros((ni, nk, 3), dtype=np.float64, order="F")
    wall_normal[:, :, 1] = 1.0

    with path.open("wb") as stream:
        stream.write(reference.VALIDATION_MAGIC)
        stream.write(fixed.tobytes())
        stream.write(np.asarray([7], dtype="<i8").tobytes())
        stream.write(np.asarray([0.007], dtype="<f8").tobytes())
        stream.write(np.asarray([100.0, 0.72, 2.5, 3.5, 0.4, 1.4], dtype="<f8").tobytes())
        for array in (
            coordinates,
            density,
            velocity,
            pressure,
            temperature,
            velocity_gradient,
            temperature_gradient,
        ):
            stream.write(np.asarray(array, dtype="<f8").ravel(order="F").tobytes())
        if has_wall:
            stream.write(wall_coordinates.ravel(order="F").tobytes())
            stream.write(wall_normal.ravel(order="F").tobytes())


def test_validation_snapshot_reader_preserves_fortran_arrays(tmp_path):
    reference = load_host_reference()
    path = tmp_path / "reference.bin"
    write_validation_snapshot(path)

    snapshot = reference.read_validation_snapshot(path)

    assert snapshot.header.global_dims == (3, 2, 4)
    assert snapshot.header.local_dims == (3, 2, 4)
    assert snapshot.header.step == 7
    assert snapshot.header.time == pytest.approx(0.007)
    assert snapshot.coordinates.shape == (3, 2, 4, 3)
    assert snapshot.velocity_gradient.shape == (3, 2, 4, 3, 3)
    assert snapshot.wall_coordinates.shape == (5, 4, 3)
    assert snapshot.coordinates.flags.f_contiguous
    assert snapshot.velocity_gradient.flags.f_contiguous


def test_independent_host_reference_projects_wall_stress_and_heat(tmp_path):
    reference = load_host_reference()
    path = tmp_path / "reference.bin"
    write_validation_snapshot(path)
    snapshot = reference.read_validation_snapshot(path)

    measure, moments = reference.integrate_wall_moments(snapshot)

    np.testing.assert_allclose(measure, 1.0, atol=1.0e-15)
    np.testing.assert_allclose(moments[:, 0], 0.8, atol=1.0e-15)
    np.testing.assert_allclose(moments[:, 1], 2.0 / 100.0, atol=1.0e-15)
    conductivity = ((1.0 / 100.0) / 0.72) / 2.5
    np.testing.assert_allclose(moments[:, 2], 3.0 * conductivity, atol=1.0e-15)


def test_validation_snapshot_reader_rejects_trailing_payload(tmp_path):
    reference = load_host_reference()
    path = tmp_path / "reference.bin"
    write_validation_snapshot(path, has_wall=False)
    path.write_bytes(path.read_bytes() + b"stale")

    with pytest.raises(ValueError, match="trailing"):
        reference.read_validation_snapshot(path)


def test_solver_output_comparison_uses_independent_snapshot_oracle(tmp_path):
    reference = load_host_reference()
    snapshot_path = tmp_path / "reference.step00000007.rank00000000.bin"
    write_validation_snapshot(snapshot_path)
    snapshot = reference.read_validation_snapshot(snapshot_path)
    plane_measure, plane_sum = reference.integrate_plane_moments(
        snapshot.coordinates,
        snapshot.density,
        snapshot.velocity,
        snapshot.pressure,
        snapshot.temperature,
    )
    wall_measure, wall_sum = reference.integrate_wall_moments(snapshot)
    sidecar_dir = tmp_path / "sidecars"
    sidecar_dir.mkdir()
    state = MODULE.CompactRankState(
        header=MODULE.CompactHeader(
            global_dims=snapshot.header.global_dims,
            topology=snapshot.header.topology,
            rank=snapshot.header.rank,
            rank_coords=snapshot.header.rank_coords,
            offsets=snapshot.header.offsets,
            local_dims=snapshot.header.local_dims,
            checkpoint_step=snapshot.header.step,
            checkpoint_time=snapshot.header.time,
            sample_count=1,
            sampling_start_step=snapshot.header.step,
            sampling_start_time=snapshot.header.time,
            has_wall=True,
        ),
        plane_measure=plane_measure,
        plane_sum=plane_sum,
        wall_measure=wall_measure,
        wall_sum=wall_sum,
    )
    MODULE.write_rank_file(sidecar_dir / "compact_stats.rank00000000.bin", state)

    result = reference.compare_solver_outputs(
        [snapshot_path], sidecar_dir, atol=1.0e-12, rtol=1.0e-12
    )

    assert result.passed
    assert result.rank_count == 1
    assert result.sample_count == 1
    assert result.max_abs <= 1.0e-12


def test_solver_output_comparison_rejects_raw_moment_error(tmp_path):
    reference = load_host_reference()
    snapshot_path = tmp_path / "reference.step00000007.rank00000000.bin"
    write_validation_snapshot(snapshot_path)
    snapshot = reference.read_validation_snapshot(snapshot_path)
    plane_measure, plane_sum = reference.integrate_plane_moments(
        snapshot.coordinates,
        snapshot.density,
        snapshot.velocity,
        snapshot.pressure,
        snapshot.temperature,
    )
    plane_sum[0, 0, 1] += 1.0e-6
    wall_measure, wall_sum = reference.integrate_wall_moments(snapshot)
    sidecar_dir = tmp_path / "sidecars"
    sidecar_dir.mkdir()
    state = MODULE.CompactRankState(
        header=MODULE.CompactHeader(
            global_dims=snapshot.header.global_dims,
            topology=snapshot.header.topology,
            rank=0,
            rank_coords=(0, 0, 0),
            offsets=(0, 0, 0),
            local_dims=snapshot.header.local_dims,
            checkpoint_step=7,
            checkpoint_time=0.007,
            sample_count=1,
            sampling_start_step=7,
            sampling_start_time=0.007,
            has_wall=True,
        ),
        plane_measure=plane_measure,
        plane_sum=plane_sum,
        wall_measure=wall_measure,
        wall_sum=wall_sum,
    )
    MODULE.write_rank_file(sidecar_dir / "compact_stats.rank00000000.bin", state)

    result = reference.compare_solver_outputs(
        [snapshot_path], sidecar_dir, atol=1.0e-12, rtol=1.0e-12
    )

    assert not result.passed
    assert result.max_abs >= 1.0e-6
