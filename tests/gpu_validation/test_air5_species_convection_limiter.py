from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
AIR5_NUM_CONSERVATIVE = 11
AIR5_TEMPERATURE_MIN_K = 300.0


def interior_ratio(value: float) -> float:
    value = min(1.0, max(0.0, value))
    if 0.0 < value < 1.0:
        return float(np.nextafter(value, 0.0))
    return value


def translational_margin(
    state: np.ndarray, cv_species: np.ndarray, formation_energy: np.ndarray
) -> float:
    density = state[0]
    if density <= 0.0:
        return -np.inf
    momentum = state[1:4]
    species = state[5:10]
    thermal_energy = (
        state[4]
        - state[10]
        - species @ formation_energy
        - AIR5_TEMPERATURE_MIN_K * (species @ cv_species)
    )
    return float(thermal_energy - momentum @ momentum / (2.0 * density))


def state_is_admissible(
    state: np.ndarray,
    cv_species: np.ndarray,
    formation_energy: np.ndarray,
    vibrational_floor: np.ndarray,
) -> bool:
    species = state[5:10]
    return bool(
        state[0] > 0.0
        and np.min(species) >= 0.0
        and state[10] - species @ vibrational_floor >= 0.0
        and translational_margin(state, cv_species, formation_energy) >= 0.0
    )


def admissible_face_ratio(
    low_state: np.ndarray,
    face_correction: np.ndarray,
    cv_species: np.ndarray,
    formation_energy: np.ndarray,
    vibrational_floor: np.ndarray,
) -> float:
    # The six face corrections form an equal-weight convex decomposition:
    # q = (1/6) sum_f (q_low + 6 theta_f delta_q_f).
    trial_correction = 6.0 * face_correction
    if state_is_admissible(
        low_state + trial_correction,
        cv_species,
        formation_energy,
        vibrational_floor,
    ):
        return 1.0

    lower = 0.0
    upper = 1.0
    for _ in range(np.finfo(np.float64).nmant + 1):
        middle = 0.5 * (lower + upper)
        if state_is_admissible(
            low_state + middle * trial_correction,
            cv_species,
            formation_energy,
            vibrational_floor,
        ):
            lower = middle
        else:
            upper = middle
    return interior_ratio(lower)


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
    for cell in range(cells):
        active = negative_budget[cell] < 0.0
        if np.any(active):
            ratio[cell] = interior_ratio(
                np.min(low_update[cell, active] / -negative_budget[cell, active])
            )

    theta = np.minimum(ratio, np.roll(ratio, -1))
    final_flux = low + theta[:, None] * antidiffusive
    updated = partial_density - dt_dx * (
        final_flux - np.roll(final_flux, 1, axis=0)
    )
    return updated, theta, final_flux


def limited_species_vibrational_update(
    partial_density: np.ndarray,
    density: np.ndarray,
    density_flux: np.ndarray,
    vibrational_energy: np.ndarray,
    vibrational_floor: np.ndarray,
    high_species_flux: np.ndarray,
    high_vibrational_flux: np.ndarray,
    dt_dx: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cells = density.size
    high_species = close_species_flux(high_species_flux, density_flux)
    low_species = upwind_species_flux(density_flux, partial_density, density)
    low_vibrational = np.empty(cells)
    for face in range(cells):
        donor = face if density_flux[face] >= 0.0 else (face + 1) % cells
        low_vibrational[face] = (
            density_flux[face] * vibrational_energy[donor] / density[donor]
        )

    high_excess = high_vibrational_flux - high_species @ vibrational_floor
    low_excess = low_vibrational - low_species @ vibrational_floor
    cell_excess = vibrational_energy - partial_density @ vibrational_floor
    low_update = cell_excess - dt_dx * (
        low_excess - np.roll(low_excess, 1)
    )
    if np.min(low_update) < -1.0e-30:
        raise ValueError("low-order vibrational baseline is outside the mechanism domain")

    correction = high_excess - low_excess
    negative_budget = np.minimum(dt_dx * np.roll(correction, 1), 0.0)
    negative_budget += np.minimum(-dt_dx * correction, 0.0)
    ratio = np.ones(cells)
    active = negative_budget < 0.0
    ratio[active] = np.array(
        [
            interior_ratio(value)
            for value in low_update[active] / -negative_budget[active]
        ]
    )
    theta = np.minimum(ratio, np.roll(ratio, -1))
    final_species = low_species + theta[:, None] * (high_species - low_species)
    final_vibrational = low_vibrational + theta * (
        high_vibrational_flux - low_vibrational
    )
    updated_species = partial_density - dt_dx * (
        final_species - np.roll(final_species, 1, axis=0)
    )
    updated_vibrational = vibrational_energy - dt_dx * (
        final_vibrational - np.roll(final_vibrational, 1)
    )
    return updated_species, updated_vibrational, theta


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


def test_shared_limiter_keeps_vibrational_temperature_inside_mechanism_domain() -> None:
    cells = 8
    density = np.ones(cells)
    partial_density = np.zeros((cells, 5))
    partial_density[:, 0] = 0.79
    partial_density[:, 1] = 0.21
    vibrational_floor = np.array([20.0, 30.0, 0.0, 0.0, 25.0])
    floor_energy = partial_density @ vibrational_floor
    vibrational_energy = floor_energy + 1.0
    density_flux = np.full(cells, 0.2)
    high_species = upwind_species_flux(density_flux, partial_density, density)
    high_vibrational = np.full(cells, density_flux[0] * vibrational_energy[0])
    high_vibrational[3] += 20.0

    unlimited = vibrational_energy - 0.1 * (
        high_vibrational - np.roll(high_vibrational, 1)
    )
    assert unlimited[3] < floor_energy[3]

    updated_species, updated_vibrational, theta = limited_species_vibrational_update(
        partial_density,
        density,
        density_flux,
        vibrational_energy,
        vibrational_floor,
        high_species,
        high_vibrational,
        0.1,
    )

    updated_floor = updated_species @ vibrational_floor
    assert np.min(updated_vibrational - updated_floor) >= -1.0e-13
    assert theta[3] < 1.0


def test_full_state_face_ratio_enforces_translational_temperature_floor() -> None:
    cv_species = np.array([742.0, 650.0, 890.0, 780.0, 690.0])
    formation_energy = np.array([-3.1e5, -2.7e5, 3.33e7, 1.52e7, 2.72e6])
    vibrational_floor = np.array([2.0e3, 1.5e3, 0.0, 0.0, 1.0e3])
    species = np.array([0.78, 0.21, 0.004, 0.003, 0.003])
    velocity = np.array([2900.0, 430.0, 0.0])
    vibrational_energy = float(species @ vibrational_floor + 500.0)
    low_state = np.zeros(AIR5_NUM_CONSERVATIVE)
    low_state[0] = np.sum(species)
    low_state[1:4] = low_state[0] * velocity
    low_state[5:10] = species
    low_state[10] = vibrational_energy
    low_state[4] = (
        0.5 * low_state[0] * velocity @ velocity
        + vibrational_energy
        + species @ formation_energy
        + 450.0 * (species @ cv_species)
    )
    face_correction = np.zeros(AIR5_NUM_CONSERVATIVE)
    face_correction[4] = -40.0 * (species @ cv_species)

    assert state_is_admissible(
        low_state, cv_species, formation_energy, vibrational_floor
    )
    assert not state_is_admissible(
        low_state + 6.0 * face_correction,
        cv_species,
        formation_energy,
        vibrational_floor,
    )

    theta = admissible_face_ratio(
        low_state,
        face_correction,
        cv_species,
        formation_energy,
        vibrational_floor,
    )
    limited = low_state + 6.0 * theta * face_correction

    assert 0.0 < theta < 1.0
    assert state_is_admissible(
        limited, cv_species, formation_energy, vibrational_floor
    )
    assert translational_margin(limited, cv_species, formation_energy) >= 0.0


def test_shared_full_state_face_flux_is_conservative() -> None:
    rng = np.random.default_rng(20260921)
    cells = 16
    low_flux = rng.normal(size=(cells, AIR5_NUM_CONSERVATIVE))
    high_flux = rng.normal(size=(cells, AIR5_NUM_CONSERVATIVE))
    ratio = rng.uniform(0.0, 1.0, size=cells)
    theta = np.minimum(ratio, np.roll(ratio, -1))
    final_flux = low_flux + theta[:, None] * (high_flux - low_flux)
    update = np.roll(final_flux, 1, axis=0) - final_flux

    np.testing.assert_allclose(
        np.sum(update, axis=0), np.zeros(AIR5_NUM_CONSERVATIVE), atol=2.0e-15
    )


def test_six_face_convex_decomposition_remains_admissible() -> None:
    cv_species = np.array([742.0, 650.0, 890.0, 780.0, 690.0])
    formation_energy = np.array([-3.1e5, -2.7e5, 3.33e7, 1.52e7, 2.72e6])
    vibrational_floor = np.array([2.0e3, 1.5e3, 0.0, 0.0, 1.0e3])
    species = np.array([0.78, 0.21, 0.004, 0.003, 0.003])
    low_state = np.zeros(AIR5_NUM_CONSERVATIVE)
    low_state[0] = np.sum(species)
    low_state[1:4] = np.array([2800.0, 300.0, 0.0])
    low_state[5:10] = species
    low_state[10] = float(species @ vibrational_floor + 1000.0)
    low_state[4] = (
        low_state[1:4] @ low_state[1:4] / (2.0 * low_state[0])
        + low_state[10]
        + species @ formation_energy
        + 500.0 * (species @ cv_species)
    )
    corrections = np.zeros((6, AIR5_NUM_CONSERVATIVE))
    corrections[:, 4] = np.array([-2.0, 1.0, -3.0, 0.5, -4.0, 1.5]) * (
        species @ cv_species
    )
    theta = np.array(
        [
            admissible_face_ratio(
                low_state,
                correction,
                cv_species,
                formation_energy,
                vibrational_floor,
            )
            for correction in corrections
        ]
    )
    final_state = low_state + np.sum(theta[:, None] * corrections, axis=0)

    assert state_is_admissible(
        final_state, cv_species, formation_energy, vibrational_floor
    )


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
        assert "air5_low_order_convective_face_flux" in source
        assert "air5_limit_full_state_convection" in source
        assert "air5_translational_margin" in source
        assert "air5_admissible_face_ratio" in source
        assert "vibrational_excess" in source
        assert "air5_num_conservative" in source
    cpu_diffusion = cpu.split("subroutinelimit_air5_diffusive_fluxes", 1)[1]
    gpu_diffusion = gpu.split("subroutineair5_diffusion_ratio_kernel", 1)[1]
    assert "air5_admissible_face_ratio" in cpu_diffusion
    assert "air5_admissible_face_ratio" in gpu_diffusion
    assert "air5_full_state_convection_ratio_kernel" in gpu
    assert "callexchange_field_halo_gpu(diffusion_ratio_d,1)" in gpu


def test_production_limiters_use_parameter_free_fp64_ratios() -> None:
    core = (ROOT / "src/chemistry_core.F90").read_text(encoding="utf-8").lower()
    cpu = (ROOT / "src/chemistry_solver.F90").read_text(encoding="utf-8").lower()
    gpu = (ROOT / "src_gpu/chemistry_solver_gpu.cuf").read_text(encoding="utf-8").lower()

    combined = core + cpu + gpu
    assert "air5_flux_limiter_safety" not in combined
    assert "air5_filter_roundoff_relative_tolerance" not in combined
    assert "air5_interior_ratio" in cpu
    assert "air5_interior_ratio_gpu" in gpu
    assert "nearest(" in core
    assert "nearest(" in gpu
    assert "digits(1.0_real64)" in core
