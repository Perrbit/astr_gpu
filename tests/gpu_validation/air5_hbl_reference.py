"""FP64 equation contract for the independent fixed-air5 HBL reference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from tests.gpu_validation.air5_radau_reference import Air5RadauReference
    from tests.gpu_validation.air5_transport_reference import Air5TransportReference
except ModuleNotFoundError:
    from air5_radau_reference import Air5RadauReference
    from air5_transport_reference import Air5TransportReference


NUM_SPECIES = 5
NUM_INDEPENDENT_SPECIES = 4
NUM_PARABOLIC_EQUATIONS = 8
CLOSURE_SPECIES = 0
INDEPENDENT_SPECIES = np.array([1, 2, 3, 4], dtype=int)


@dataclass(frozen=True)
class Air5BoundaryLayerState:
    pressure: float
    velocity: np.ndarray
    temperature: float
    tv: float
    mass_fraction: np.ndarray


class Air5BoundaryLayerReference:
    """Thin-layer flux and source definitions for a future x marcher.

    The equation order is mass, x momentum, four independent species, total
    energy, and vibrational energy. N2 closes the mixture so trace NO is not
    recovered by subtracting two order-one mass fractions.
    Only wall-normal diffusive fluxes belong to this parabolic contract.
    """

    def __init__(self, mechanism_path: Path):
        self.chemistry = Air5RadauReference(mechanism_path)
        self.transport = Air5TransportReference(mechanism_path)

    def validate_state(self, state: Air5BoundaryLayerState) -> None:
        velocity = np.asarray(state.velocity, dtype=np.float64)
        mass_fraction = np.asarray(state.mass_fraction, dtype=np.float64)
        scalars = np.asarray(
            [state.pressure, state.temperature, state.tv], dtype=np.float64
        )
        if velocity.shape != (3,) or mass_fraction.shape != (NUM_SPECIES,):
            raise ValueError("invalid fixed-air5 boundary-layer state shape")
        if not np.all(np.isfinite(np.r_[scalars, velocity, mass_fraction])):
            raise ValueError("boundary-layer state contains non-finite values")
        if np.any(scalars <= 0.0) or np.any(mass_fraction < 0.0):
            raise ValueError("boundary-layer state is outside the physical domain")
        if not np.isclose(np.sum(mass_fraction), 1.0, atol=2.0e-14, rtol=0.0):
            raise ValueError("boundary-layer mass fractions do not close")

    def density(self, state: Air5BoundaryLayerState) -> float:
        self.validate_state(state)
        gas_constant = float(
            np.dot(state.mass_fraction, self.chemistry.gas_constant)
        )
        return state.pressure / (gas_constant * state.temperature)

    def conservative_state(self, state: Air5BoundaryLayerState) -> np.ndarray:
        density = self.density(state)
        velocity = np.asarray(state.velocity, dtype=np.float64)
        rho_species = density * state.mass_fraction
        momentum = density * velocity
        ev = self.chemistry.ev_from_tv(rho_species, state.tv)
        q5 = self.chemistry.q5_from_state(
            density, momentum, rho_species, ev, state.temperature
        )
        return np.r_[density, momentum, q5, rho_species, ev]

    def streamwise_flux(self, state: Air5BoundaryLayerState) -> np.ndarray:
        q = self.conservative_state(state)
        density = q[0]
        u = state.velocity[0]
        return np.r_[
            density * u,
            density * u * u + state.pressure,
            density * u * state.mass_fraction[INDEPENDENT_SPECIES],
            u * (q[4] + state.pressure),
            u * q[10],
        ]

    def wall_normal_flux(
        self,
        state: Air5BoundaryLayerState,
        *,
        grad_velocity_y: np.ndarray,
        grad_temperature_y: float,
        grad_tv_y: float,
        grad_mass_fraction_y: np.ndarray,
    ) -> np.ndarray:
        """Return convective plus physical wall-normal flux in equation order."""
        q = self.conservative_state(state)
        density = q[0]
        velocity = np.asarray(state.velocity, dtype=np.float64)
        grad_velocity_y = np.asarray(grad_velocity_y, dtype=np.float64)
        grad_mass_fraction_y = np.asarray(
            grad_mass_fraction_y, dtype=np.float64
        )
        if grad_velocity_y.shape != (3,) or grad_mass_fraction_y.shape != (5,):
            raise ValueError("invalid wall-normal gradient shape")
        if not np.all(
            np.isfinite(
                np.r_[
                    grad_velocity_y,
                    grad_temperature_y,
                    grad_tv_y,
                    grad_mass_fraction_y,
                ]
            )
        ):
            raise ValueError("wall-normal gradients contain non-finite values")
        if abs(float(np.sum(grad_mass_fraction_y))) > 2.0e-12:
            raise ValueError("wall-normal mass-fraction gradients do not close")

        grad_velocity = np.zeros((3, 3))
        grad_velocity[:, 1] = grad_velocity_y
        grad_temperature = np.array([0.0, grad_temperature_y, 0.0])
        grad_tv = np.array([0.0, grad_tv_y, 0.0])
        grad_mass_fraction = np.zeros((5, 3))
        grad_mass_fraction[:, 1] = grad_mass_fraction_y
        diffusion = self.transport.diffusive_flux(
            rho=density,
            velocity=velocity,
            temperature=state.temperature,
            tv=state.tv,
            pressure=state.pressure,
            grad_velocity=grad_velocity,
            grad_temperature=grad_temperature,
            grad_tv=grad_tv,
            mass_fraction=state.mass_fraction,
            grad_mass_fraction=grad_mass_fraction,
        )

        v = velocity[1]
        return np.r_[
            density * v,
            density * velocity[0] * v - diffusion.momentum_flux[0, 1],
            density * v * state.mass_fraction[INDEPENDENT_SPECIES]
            + diffusion.species_flux[INDEPENDENT_SPECIES, 1],
            v * (q[4] + state.pressure) - diffusion.energy_flux[1],
            v * q[10] - diffusion.ev_flux[1],
        ]

    def source(self, state: Air5BoundaryLayerState, mode: str = "coupled") -> np.ndarray:
        q = self.conservative_state(state)
        local_state = np.r_[q[5:10], q[10]]
        local_source = self.chemistry.rhs(
            0.0,
            local_state,
            q[0],
            q[1:4],
            q[4],
            source_mode=mode,
        )
        return np.r_[
            0.0,
            0.0,
            local_source[INDEPENDENT_SPECIES],
            0.0,
            local_source[5],
        ]
