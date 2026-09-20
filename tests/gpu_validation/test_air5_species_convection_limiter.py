from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LIMITER_RESERVE = 1.0e-8


def close_species_flux(flux: np.ndarray, density_flux: np.ndarray) -> np.ndarray:
    closed = flux.copy()
    closed[..., 0] = density_flux - np.sum(closed[..., 1:], axis=-1)
    return closed


def upwind_species_flux(
    density_flux: np.ndarray, partial_density: np.ndarray, density: np.ndarray
) -> np.ndarray:
    cells = density.size
    result = np.empty((cells, partial_density.shape[1]))
    for face in range(cells):
        left = face
        right = (face + 1) % cells
        donor = left if density_flux[face] >= 0.0 else right
        result[face] = density_flux[face] * partial_density[donor] / density[donor]
    return close_species_flux(result, density_flux)


def limited_species_update(
    partial_density: np.ndarray,
    density: np.ndarray,
    density_flux: np.ndarray,
    high_species_flux: np.ndarray,
    dt_dx: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cells = density.size
    high = close_species_flux(high_species_flux, density_flux)
    low = upwind_species_flux(density_flux, partial_density, density)
    antidiffusive = high - low
    low_update = partial_density - dt_dx * (low - np.roll(low, 1, axis=0))
    if np.min(low_update) < -1.0e-30:
        raise ValueError("low-order species baseline is not positive")

    negative_budget = np.minimum(dt_dx * np.roll(antidiffusive, 1, axis=0), 0.0)
    negative_budget += np.minimum(-dt_dx * antidiffusive, 0.0)
    ratio = np.ones(cells)
    safety = 1.0 - LIMITER_RESERVE
    for cell in range(cells):
        active = negative_budget[cell] < 0.0
        if np.any(active):
            ratio[cell] = min(
                1.0,
                np.min(safety * low_update[cell, active] / -negative_budget[cell, active]),
            )

    theta = np.minimum(ratio, np.roll(ratio, -1))
    final_flux = low + theta[:, None] * antidiffusive
    updated = partial_density - dt_dx * (
        final_flux - np.roll(final_flux, 1, axis=0)
    )
    return updated, theta, final_flux


def test_trace_species_is_positive_without_limiting_bulk_equations() -> None:
    cells = 8
    density = np.ones(cells)
    partial_density = np.zeros((cells, 5))
    partial_density[:, 0] = 0.79
    partial_density[:, 1] = 0.21
    partial_density[:, 4] = 1.0e-35
    partial_density[:, 0] -= partial_density[:, 4]
    density_flux = np.full(cells, 0.2)
    high = upwind_species_flux(density_flux, partial_density, density)
    high[3, 4] += 2.0e-33
    high[3, 0] -= 2.0e-33

    unlimited = partial_density - 0.1 * (high - np.roll(high, 1, axis=0))
    assert unlimited[3, 4] < 0.0

    updated, theta, final_flux = limited_species_update(
        partial_density, density, density_flux, high, 0.1
    )

    assert np.min(updated) >= -1.0e-30
    assert theta[3] < 1.0
    np.testing.assert_allclose(np.sum(final_flux, axis=1), density_flux, atol=1.0e-15)
    density_updated = density - 0.1 * (density_flux - np.roll(density_flux, 1))
    np.testing.assert_allclose(np.sum(updated, axis=1), density_updated, atol=2.0e-15)


def test_limiter_is_identity_for_an_admissible_high_order_update() -> None:
    cells = 32
    x = np.arange(cells) / cells
    density = np.ones(cells)
    partial_density = np.empty((cells, 5))
    partial_density[:, 1] = 0.2 + 0.02 * np.sin(2.0 * np.pi * x)
    partial_density[:, 2] = 0.01
    partial_density[:, 3] = 0.005
    partial_density[:, 4] = 0.002
    partial_density[:, 0] = 1.0 - np.sum(partial_density[:, 1:], axis=1)
    density_flux = np.full(cells, 0.1)
    high = upwind_species_flux(density_flux, partial_density, density)

    updated, theta, final_flux = limited_species_update(
        partial_density, density, density_flux, high, 0.01
    )

    np.testing.assert_array_equal(theta, np.ones(cells))
    np.testing.assert_allclose(final_flux, high, atol=0.0, rtol=0.0)
    assert np.min(updated) > 0.0


def test_cpu_and_gpu_expose_consistent_species_flux_correction() -> None:
    cpu = "".join(
        (ROOT / "src/chemistry_solver.F90").read_text(encoding="utf-8").lower().split()
    )
    gpu = "".join(
        (ROOT / "src_gpu/chemistry_solver_gpu.cuf")
        .read_text(encoding="utf-8")
        .lower()
        .split()
    )

    for source in (cpu, gpu):
        assert "air5_low_order_species_face_flux" in source
        assert "air5_limit_species_convection" in source
        assert "species_face_flux(1)=density_flux" in source
    assert "air5_species_convection_ratio_kernel" in gpu
    assert "callexchange_field_halo_gpu(diffusion_ratio_d,1)" in gpu
