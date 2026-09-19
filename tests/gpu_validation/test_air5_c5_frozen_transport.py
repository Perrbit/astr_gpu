from pathlib import Path

import numpy as np

from air5_transport_reference import (
    Air5TransportReference,
    derivative_sixth_order,
)
from check_air5_c5_frozen_transport import derivative_from_halo


ROOT = Path(__file__).resolve().parents[2]


def compact(text: str) -> str:
    return "".join(text.lower().split())


def test_frozen_transport_cases_have_dedicated_initializers() -> None:
    initializer = compact(
        (ROOT / "src/initialisation.F90").read_text(encoding="utf-8")
    )
    grid = compact((ROOT / "src/gridgeneration.F90").read_text(encoding="utf-8"))
    preparer = (
        ROOT / "tests/gpu_validation/prepare_air5_c4_case.py"
    ).read_text(encoding="utf-8")

    for flowtype, routine in (
        ("air5advection", "air5advectionini"),
        ("air5difflayer", "air5difflayerini"),
        ("air5evpulse", "air5evpulseini"),
    ):
        assert f"case('{flowtype}')" in initializer
        assert f"call{routine}" in initializer
        assert f"subroutine{routine}" in initializer
        assert flowtype in grid
    assert '"advection-wave"' in preparer
    assert '"diffusion-layer"' in preparer
    assert '"ev-pulse"' in preparer


def test_frozen_transport_cases_do_not_enable_strang_chemistry() -> None:
    runtime = compact(
        (ROOT / "src/chemistry_runtime.F90").read_text(encoding="utf-8")
    )

    assert "case('air5reactor','air5postshock','air5tgv','air5hbl')" in runtime
    for flowtype in ("air5advection", "air5difflayer", "air5evpulse"):
        assert f"case('{flowtype}')" not in runtime


def test_sixth_order_periodic_derivative_is_accurate_for_smooth_wave() -> None:
    points = 48
    theta = 2.0 * np.pi * np.arange(points) / points
    numerical = derivative_sixth_order(np.sin(theta), 2.0 * np.pi / points)

    np.testing.assert_allclose(numerical, np.cos(theta), atol=4.0e-8, rtol=0.0)


def test_sixth_order_halo_derivative_uses_only_centered_stencil() -> None:
    points = 48
    halo = 3
    spacing = 2.0 * np.pi / points
    active = 12
    indices = np.arange(-halo, active + halo + 1)
    values = np.sin(spacing * indices)

    numerical = derivative_from_halo(values, halo, active, spacing)

    np.testing.assert_allclose(
        numerical,
        np.cos(spacing * np.arange(active + 1)),
        atol=4.0e-8,
        rtol=0.0,
    )


def test_multicomponent_background_produces_nonzero_correction_velocity() -> None:
    reference = Air5TransportReference(ROOT / "chemMech/air5_kimjo12.json")
    mass_fraction = np.array([0.55, 0.40, 0.02, 0.02, 0.01])
    gradient = np.zeros((5, 3))
    gradient[:, 0] = np.array([0.2, -0.2, 0.0, 0.0, 0.0])

    result = reference.diffusive_flux(
        rho=0.08,
        velocity=np.zeros(3),
        temperature=4000.0,
        tv=2000.0,
        pressure=1.0e5,
        grad_velocity=np.zeros((3, 3)),
        grad_temperature=np.zeros(3),
        grad_tv=np.zeros(3),
        mass_fraction=mass_fraction,
        grad_mass_fraction=gradient,
    )

    assert abs(result.correction_velocity[0]) > 1.0e-12
    np.testing.assert_allclose(np.sum(result.species_flux, axis=0), 0.0, atol=1.0e-14)


def test_frozen_transport_driver_covers_all_three_cases() -> None:
    driver = (
        ROOT / "tests/gpu_validation/run_air5_c5_frozen_transport_compare.sh"
    ).read_text(encoding="utf-8")

    assert "advection-wave" in driver
    assert "diffusion-layer" in driver
    assert "ev-pulse" in driver
    assert "check_air5_c5_frozen_transport.py" in driver
    assert 'DELTAT="${DELTAT:-5.d-7}"' in driver
    assert 'MAX_CFL="${MAX_CFL:-1.0}"' in driver
    assert "check_cfl" in driver
