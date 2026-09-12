#!/usr/bin/env python3
"""Binary contract and mathematics for compact GPU production statistics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np


MAGIC = b"ASTRCST1"
VERSION = 1
ENDIAN_MARKER = 0x01020304
REAL_BYTES = 8
MOMENT_NAMES = (
    "rho",
    "rho_u",
    "rho_v",
    "rho_w",
    "rho_T",
    "p",
    "rho_uu",
    "rho_vv",
    "rho_ww",
    "rho_uv",
    "rho_uw",
    "rho_vw",
)


@dataclass(eq=True)
class CompactHeader:
    global_dims: tuple[int, int, int]
    topology: tuple[int, int, int]
    rank: int
    rank_coords: tuple[int, int, int]
    offsets: tuple[int, int, int]
    local_dims: tuple[int, int, int]
    checkpoint_step: int
    checkpoint_time: float
    sample_count: int
    sampling_start_step: int
    sampling_start_time: float
    has_wall: bool


@dataclass(eq=False)
class CompactRankState:
    header: CompactHeader
    plane_measure: np.ndarray
    plane_sum: np.ndarray
    wall_measure: np.ndarray | None = None
    wall_sum: np.ndarray | None = None


def _tuple3(name: str, values: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain three integers")
    return tuple(int(value) for value in values)


def _validate_header(header: CompactHeader) -> None:
    global_dims = _tuple3("global_dims", header.global_dims)
    topology = _tuple3("topology", header.topology)
    coords = _tuple3("rank_coords", header.rank_coords)
    offsets = _tuple3("offsets", header.offsets)
    local_dims = _tuple3("local_dims", header.local_dims)
    if min(global_dims) < 1 or min(topology) < 1 or min(local_dims) < 1:
        raise ValueError("global, topology, and local dimensions must be positive")
    if int(header.rank) < 0 or int(header.rank) >= int(np.prod(topology)):
        raise ValueError("rank is outside the declared topology")
    if any(coord < 0 or coord >= count for coord, count in zip(coords, topology)):
        raise ValueError("rank coordinates are outside the declared topology")
    if any(offset < 0 for offset in offsets):
        raise ValueError("offsets must be non-negative")
    if any(offset + local > global_ for offset, local, global_ in zip(offsets, local_dims, global_dims)):
        raise ValueError("local extent exceeds the global dimensions")
    if header.checkpoint_step < 0 or header.sample_count < 0:
        raise ValueError("checkpoint step and sample count must be non-negative")
    if not np.isfinite(header.checkpoint_time) or not np.isfinite(header.sampling_start_time):
        raise ValueError("statistics times must be finite")


def _validate_state(state: CompactRankState) -> None:
    _validate_header(state.header)
    ni, nj, _ = state.header.local_dims
    plane_measure = np.asarray(state.plane_measure)
    plane_sum = np.asarray(state.plane_sum)
    if plane_measure.shape != (ni, nj):
        raise ValueError(f"plane_measure shape must be {(ni, nj)}, found {plane_measure.shape}")
    if plane_sum.shape != (ni, nj, len(MOMENT_NAMES)):
        raise ValueError(
            f"plane_sum shape must be {(ni, nj, len(MOMENT_NAMES))}, found {plane_sum.shape}"
        )
    if not np.all(np.isfinite(plane_measure)) or not np.all(np.isfinite(plane_sum)):
        raise ValueError("plane arrays must contain finite values")
    if state.header.has_wall:
        if state.wall_measure is None or state.wall_sum is None:
            raise ValueError("wall arrays are required when has_wall is true")
        if np.asarray(state.wall_measure).shape != (ni,):
            raise ValueError(f"wall_measure shape must be {(ni,)}")
        if np.asarray(state.wall_sum).shape != (ni, 3):
            raise ValueError(f"wall_sum shape must be {(ni, 3)}")
        if not np.all(np.isfinite(state.wall_measure)) or not np.all(np.isfinite(state.wall_sum)):
            raise ValueError("wall arrays must contain finite values")
    elif state.wall_measure is not None or state.wall_sum is not None:
        raise ValueError("wall arrays must be absent when has_wall is false")


def _write_array(stream: BinaryIO, values: np.ndarray, dtype: str) -> None:
    array = np.asarray(values, dtype=np.dtype(dtype), order="F")
    stream.write(array.ravel(order="F").tobytes())


def write_rank_file(path: Path | str, state: CompactRankState) -> None:
    """Write one validated rank-local sidecar in the version-1 stream format."""
    _validate_state(state)
    header = state.header
    integer_header = np.asarray(
        (
            VERSION,
            ENDIAN_MARKER,
            REAL_BYTES,
            int(header.has_wall),
            *header.global_dims,
            *header.topology,
            header.rank,
            *header.rank_coords,
            *header.offsets,
            *header.local_dims,
        ),
        dtype="<i4",
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(MAGIC)
        stream.write(integer_header.tobytes())
        _write_array(stream, np.asarray([header.checkpoint_step]), "<i8")
        _write_array(stream, np.asarray([header.checkpoint_time]), "<f8")
        _write_array(
            stream,
            np.asarray([header.sample_count, header.sampling_start_step]),
            "<i8",
        )
        _write_array(stream, np.asarray([header.sampling_start_time]), "<f8")
        _write_array(stream, state.plane_measure, "<f8")
        _write_array(stream, state.plane_sum, "<f8")
        if header.has_wall:
            _write_array(stream, state.wall_measure, "<f8")
            _write_array(stream, state.wall_sum, "<f8")


def _read_exact(stream: BinaryIO, size: int, label: str) -> bytes:
    payload = stream.read(size)
    if len(payload) != size:
        raise ValueError(f"truncated compact statistics {label}")
    return payload


def _read_array(stream: BinaryIO, count: int, dtype: str, label: str) -> np.ndarray:
    itemsize = np.dtype(dtype).itemsize
    return np.frombuffer(_read_exact(stream, count * itemsize, label), dtype=dtype).copy()


def read_rank_file(path: Path | str) -> CompactRankState:
    """Read and validate one rank-local sidecar."""
    path = Path(path)
    with path.open("rb") as stream:
        if _read_exact(stream, len(MAGIC), "magic") != MAGIC:
            raise ValueError("compact statistics magic mismatch")
        fixed = _read_array(stream, 20, "<i4", "fixed header")
        version, endian_marker, real_bytes, has_wall_value = (int(value) for value in fixed[:4])
        if version != VERSION:
            raise ValueError(f"unsupported compact statistics version {version}")
        if endian_marker != ENDIAN_MARKER:
            raise ValueError("compact statistics endianness marker mismatch")
        if real_bytes != REAL_BYTES:
            raise ValueError(f"unsupported compact statistics real size {real_bytes}")
        if has_wall_value not in (0, 1):
            raise ValueError("invalid compact statistics wall flag")
        checkpoint_step = int(_read_array(stream, 1, "<i8", "checkpoint step")[0])
        checkpoint_time = float(_read_array(stream, 1, "<f8", "checkpoint time")[0])
        sample_fields = _read_array(stream, 2, "<i8", "sample metadata")
        sampling_start_time = float(_read_array(stream, 1, "<f8", "sample start time")[0])
        header = CompactHeader(
            global_dims=tuple(int(value) for value in fixed[4:7]),
            topology=tuple(int(value) for value in fixed[7:10]),
            rank=int(fixed[10]),
            rank_coords=tuple(int(value) for value in fixed[11:14]),
            offsets=tuple(int(value) for value in fixed[14:17]),
            local_dims=tuple(int(value) for value in fixed[17:20]),
            checkpoint_step=checkpoint_step,
            checkpoint_time=checkpoint_time,
            sample_count=int(sample_fields[0]),
            sampling_start_step=int(sample_fields[1]),
            sampling_start_time=sampling_start_time,
            has_wall=bool(has_wall_value),
        )
        ni, nj, _ = header.local_dims
        plane_measure = np.asfortranarray(
            _read_array(stream, ni * nj, "<f8", "plane measure").reshape((ni, nj), order="F")
        )
        plane_sum = np.asfortranarray(
            _read_array(stream, ni * nj * 12, "<f8", "plane moments").reshape(
                (ni, nj, 12), order="F"
            )
        )
        wall_measure = None
        wall_sum = None
        if header.has_wall:
            wall_measure = np.asfortranarray(
                _read_array(stream, ni, "<f8", "wall measure")
            )
            wall_sum = np.asfortranarray(
                _read_array(stream, ni * 3, "<f8", "wall moments").reshape(
                    (ni, 3), order="F"
                )
            )
        if stream.read(1):
            raise ValueError("trailing bytes after compact statistics payload")
    state = CompactRankState(header, plane_measure, plane_sum, wall_measure, wall_sum)
    _validate_state(state)
    return state


def derive_statistics(state: CompactRankState) -> dict[str, np.ndarray]:
    """Derive means and Favre stresses without altering accumulated raw sums."""
    _validate_state(state)
    if state.header.sample_count <= 0:
        raise ValueError("sample count must be positive before deriving statistics")
    if np.any(state.plane_measure <= 0.0):
        raise ValueError("positive plane measure is required")
    rho_sum = state.plane_sum[:, :, 0]
    if np.any(rho_sum <= 0.0):
        raise ValueError("positive density sum is required")
    sample_measure = state.header.sample_count * state.plane_measure
    u = state.plane_sum[:, :, 1] / rho_sum
    v = state.plane_sum[:, :, 2] / rho_sum
    w = state.plane_sum[:, :, 3] / rho_sum
    result = {
        f"raw_{name}": state.plane_sum[:, :, index].copy(order="F")
        for index, name in enumerate(MOMENT_NAMES)
    }
    result.update(
        {
            "rho_mean": rho_sum / sample_measure,
            "u_favre": u,
            "v_favre": v,
            "w_favre": w,
            "T_favre": state.plane_sum[:, :, 4] / rho_sum,
            "p_mean": state.plane_sum[:, :, 5] / sample_measure,
            "uu_favre": state.plane_sum[:, :, 6] / rho_sum - u * u,
            "vv_favre": state.plane_sum[:, :, 7] / rho_sum - v * v,
            "ww_favre": state.plane_sum[:, :, 8] / rho_sum - w * w,
            "uv_favre": state.plane_sum[:, :, 9] / rho_sum - u * v,
            "uw_favre": state.plane_sum[:, :, 10] / rho_sum - u * w,
            "vw_favre": state.plane_sum[:, :, 11] / rho_sum - v * w,
        }
    )
    if state.header.has_wall:
        if np.any(state.wall_measure <= 0.0):
            raise ValueError("positive wall measure is required")
        wall_denominator = state.header.sample_count * state.wall_measure
        result["wall_p_mean"] = state.wall_sum[:, 0] / wall_denominator
        result["wall_tau_streamwise_mean"] = state.wall_sum[:, 1] / wall_denominator
        result["wall_q_normal_mean"] = state.wall_sum[:, 2] / wall_denominator
    return result


def _common_generation(states: list[CompactRankState]) -> CompactHeader:
    reference = states[0].header
    rank_count = int(np.prod(reference.topology))
    ranks = [state.header.rank for state in states]
    if len(states) != rank_count or set(ranks) != set(range(rank_count)):
        raise ValueError("missing ranks or duplicate rank files")
    coords = [state.header.rank_coords for state in states]
    if len(set(coords)) != rank_count:
        raise ValueError("duplicate rank coordinates")
    for state in states:
        header = state.header
        if header.global_dims != reference.global_dims:
            raise ValueError("incompatible global dimensions")
        if header.topology != reference.topology:
            raise ValueError("incompatible topologies")
        if header.checkpoint_step != reference.checkpoint_step:
            raise ValueError("incompatible checkpoint steps")
        tolerance = 1.0e-12 * max(1.0, abs(reference.checkpoint_time))
        if abs(header.checkpoint_time - reference.checkpoint_time) > tolerance:
            raise ValueError("incompatible checkpoint times")
        if (
            header.sample_count != reference.sample_count
            or header.sampling_start_step != reference.sampling_start_step
            or header.sampling_start_time != reference.sampling_start_time
        ):
            raise ValueError("incompatible sampling windows")
    return reference


def merge_rank_states(states: Iterable[CompactRankState]) -> CompactRankState:
    """Merge complete rank-local z integrals into one global x-y state."""
    states = list(states)
    if not states:
        raise ValueError("no compact statistics rank files")
    for state in states:
        _validate_state(state)
    reference = _common_generation(states)
    global_ni, global_nj, _ = reference.global_dims
    topology = reference.topology

    xy_blocks: dict[tuple[int, int], list[CompactRankState]] = {}
    for state in states:
        xy_blocks.setdefault(state.header.rank_coords[:2], []).append(state)
    block_values: list[tuple[CompactHeader, np.ndarray, np.ndarray]] = []
    wall_blocks: list[tuple[CompactHeader, np.ndarray, np.ndarray]] = []
    for key, slabs in xy_blocks.items():
        if {slab.header.rank_coords[2] for slab in slabs} != set(range(topology[2])):
            raise ValueError(f"missing z slab for x-y block {key}")
        first = min(slabs, key=lambda state: state.header.rank)
        plane_measure = np.sum([state.plane_measure for state in slabs], axis=0)
        plane_sum = np.sum([state.plane_sum for state in slabs], axis=0)
        block_values.append((first.header, plane_measure, plane_sum))
        wall_slabs = [state for state in slabs if state.header.has_wall]
        if wall_slabs:
            wall_measure = np.sum([state.wall_measure for state in wall_slabs], axis=0)
            wall_sum = np.sum([state.wall_sum for state in wall_slabs], axis=0)
            wall_blocks.append((first.header, wall_measure, wall_sum))

    global_measure = np.full((global_ni, global_nj), np.nan, order="F")
    global_sum = np.full((global_ni, global_nj, 12), np.nan, order="F")
    for header, block_measure, block_sum in sorted(block_values, key=lambda item: item[0].rank):
        i0, j0, _ = header.offsets
        ni, nj, _ = header.local_dims
        for j in range(nj):
            for i in range(ni):
                gi, gj = i0 + i, j0 + j
                if np.isnan(global_measure[gi, gj]):
                    global_measure[gi, gj] = block_measure[i, j]
                    global_sum[gi, gj, :] = block_sum[i, j, :]
                elif not (
                    np.isclose(global_measure[gi, gj], block_measure[i, j], atol=1e-10, rtol=1e-10)
                    and np.allclose(global_sum[gi, gj, :], block_sum[i, j, :], atol=1e-10, rtol=1e-10)
                ):
                    raise ValueError(f"overlap mismatch at global plane node {(gi, gj)}")
    if np.any(np.isnan(global_measure)) or np.any(np.isnan(global_sum)):
        raise ValueError("rank files do not cover the global x-y plane")

    global_wall_measure = None
    global_wall_sum = None
    has_wall = bool(wall_blocks)
    if has_wall:
        global_wall_measure = np.full(global_ni, np.nan)
        global_wall_sum = np.full((global_ni, 3), np.nan, order="F")
        for header, block_measure, block_sum in sorted(wall_blocks, key=lambda item: item[0].rank):
            i0 = header.offsets[0]
            ni = header.local_dims[0]
            for i in range(ni):
                gi = i0 + i
                if np.isnan(global_wall_measure[gi]):
                    global_wall_measure[gi] = block_measure[i]
                    global_wall_sum[gi, :] = block_sum[i, :]
                elif not (
                    np.isclose(global_wall_measure[gi], block_measure[i], atol=1e-10, rtol=1e-10)
                    and np.allclose(global_wall_sum[gi, :], block_sum[i, :], atol=1e-10, rtol=1e-10)
                ):
                    raise ValueError(f"overlap mismatch at global wall node {gi}")
        if np.any(np.isnan(global_wall_measure)) or np.any(np.isnan(global_wall_sum)):
            raise ValueError("wall rank files do not cover the global streamwise line")

    merged_header = CompactHeader(
        global_dims=reference.global_dims,
        topology=reference.topology,
        rank=0,
        rank_coords=(0, 0, 0),
        offsets=(0, 0, 0),
        local_dims=reference.global_dims,
        checkpoint_step=reference.checkpoint_step,
        checkpoint_time=reference.checkpoint_time,
        sample_count=reference.sample_count,
        sampling_start_step=reference.sampling_start_step,
        sampling_start_time=reference.sampling_start_time,
        has_wall=has_wall,
    )
    return CompactRankState(
        merged_header,
        np.asfortranarray(global_measure),
        np.asfortranarray(global_sum),
        None if global_wall_measure is None else np.asfortranarray(global_wall_measure),
        None if global_wall_sum is None else np.asfortranarray(global_wall_sum),
    )
