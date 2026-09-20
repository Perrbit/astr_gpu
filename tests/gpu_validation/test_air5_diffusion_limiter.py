from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LIMITER_RESERVE = 1.0e-8


def derivative(values: dict[int, float], index: int, dim: int, ntype: int) -> float:
    if ntype in (1, 4) and index == 0:
        return -0.5 * values[2] + 2.0 * values[1] - 1.5 * values[0]
    if ntype in (1, 4) and index == 1:
        return 0.5 * (values[2] - values[0])
    if ntype in (1, 4) and index == 2:
        return (2.0 / 3.0) * (values[3] - values[1]) - (1.0 / 12.0) * (
            values[4] - values[0]
        )
    if ntype in (2, 4) and index == dim - 2:
        return (2.0 / 3.0) * (values[dim - 1] - values[dim - 3]) - (
            1.0 / 12.0
        ) * (values[dim] - values[dim - 4])
    if ntype in (2, 4) and index == dim - 1:
        return 0.5 * (values[dim] - values[dim - 2])
    if ntype in (2, 4) and index == dim:
        return 0.5 * values[dim - 2] - 2.0 * values[dim - 1] + 1.5 * values[dim]
    return (
        0.75 * (values[index + 1] - values[index - 1])
        - 0.15 * (values[index + 2] - values[index - 2])
        + (values[index + 3] - values[index - 3]) / 60.0
    )


def centered_face(values: dict[int, float], face: int) -> float:
    return (
        (37.0 / 60.0) * (values[face] + values[face + 1])
        - (2.0 / 15.0) * (values[face - 1] + values[face + 2])
        + (values[face - 2] + values[face + 3]) / 60.0
    )


def face_flux(values: dict[int, float], face: int, dim: int, ntype: int) -> float:
    if ntype in (1, 4) and face <= 1:
        anchor = centered_face(values, 2)
        result = anchor - derivative(values, 2, dim, ntype)
        if face <= 0:
            result -= derivative(values, 1, dim, ntype)
        if face <= -1:
            result -= derivative(values, 0, dim, ntype)
        return result
    if ntype in (2, 4) and face >= dim - 2:
        anchor = centered_face(values, dim - 3)
        result = anchor + derivative(values, dim - 2, dim, ntype)
        if face >= dim - 1:
            result += derivative(values, dim - 1, dim, ntype)
        if face >= dim:
            result += derivative(values, dim, dim, ntype)
        return result
    return centered_face(values, face)


def test_face_flux_difference_recovers_existing_derivative_closures() -> None:
    rng = np.random.default_rng(20260919)
    dim = 15
    values = {index: value for index, value in zip(range(-3, dim + 4), rng.normal(size=dim + 7))}

    for ntype in (1, 2, 3, 4):
        reconstructed = np.array(
            [
                face_flux(values, index, dim, ntype)
                - face_flux(values, index - 1, dim, ntype)
                for index in range(dim + 1)
            ]
        )
        expected = np.array(
            [derivative(values, index, dim, ntype) for index in range(dim + 1)]
        )
        np.testing.assert_allclose(reconstructed, expected, atol=2.0e-15, rtol=0.0)


def test_shared_face_ratio_preserves_positive_species_baseline() -> None:
    rng = np.random.default_rng(20260919)
    cells = 64
    species = 5
    base = 10.0 ** rng.uniform(-24.0, -2.0, size=(cells, species))
    face = rng.normal(size=(cells, species))
    safety = 1.0 - LIMITER_RESERVE

    negative_budget = np.minimum(face[np.arange(cells) - 1], 0.0) + np.minimum(
        -face, 0.0
    )
    ratio = np.ones(cells)
    for cell in range(cells):
        active = negative_budget[cell] < 0.0
        if np.any(active):
            ratio[cell] = min(
                1.0,
                np.min(safety * base[cell, active] / -negative_budget[cell, active]),
            )

    left_theta = np.minimum(ratio, ratio[np.arange(cells) - 1])
    right_theta = np.minimum(ratio, ratio[(np.arange(cells) + 1) % cells])
    correction = left_theta[:, None] * face[np.arange(cells) - 1] - right_theta[:, None] * face

    assert np.min(base + correction) >= -1.0e-30


def test_trace_species_retains_fp64_roundoff_headroom() -> None:
    base = 2.747120157690203e-26
    negative_correction = -2.747120157902086e-26
    adverse_reconstruction_roundoff = 8.0e-11 * base
    safety = 1.0 - LIMITER_RESERVE

    ratio = min(1.0, safety * base / -negative_correction)
    updated = base + ratio * negative_correction - adverse_reconstruction_roundoff

    assert updated > 0.0


def test_cpu_and_gpu_use_one_shared_scalar_face_ratio() -> None:
    cpu = "".join(
        (ROOT / "src/chemistry_solver.F90").read_text(encoding="utf-8").lower().split()
    )
    gpu = "".join(
        (ROOT / "src_gpu/chemistry_solver_gpu.cuf")
        .read_text(encoding="utf-8")
        .lower()
        .split()
    )
    arrays = "".join(
        (ROOT / "src_gpu/commarray_gpu.cuf").read_text(encoding="utf-8").lower().split()
    )

    assert "air5_flux_limiter_safety" in cpu
    assert "air5_flux_limiter_safety" in gpu

    assert "calllimit_air5_diffusive_fluxes(" in cpu
    assert "theta_left=min(diffusion_ratio(i,j,k)," in cpu
    assert "theta_right=min(diffusion_ratio(i,j,k)," in cpu
    assert "diffusion_ratio_d(:,:,:,:)" in arrays
    assert "callexchange_field_halo_gpu(diffusion_ratio_d,1)" in gpu
    assert "theta_left=min(diffusion_ratio_d(i,j,k,1)," in gpu
    assert "theta_right=min(diffusion_ratio_d(i,j,k,1)," in gpu
    assert "callreport_air5_diffusion_limiter(rkstep)" in gpu
