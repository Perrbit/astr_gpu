"""Independent steady one-dimensional post-shock relaxation reference."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import solve_ivp

from air5_radau_reference import Air5RadauReference


@dataclass(frozen=True)
class Air5FlowPoint:
    density: float
    speed: float
    pressure: float
    temperature: float
    tv: float
    mass_fraction: np.ndarray
    ev_specific: float
    mach: float


@dataclass(frozen=True)
class Air5RelaxationProfile:
    x: np.ndarray
    points: tuple[Air5FlowPoint, ...]

    @property
    def temperature(self) -> np.ndarray:
        return np.asarray([point.temperature for point in self.points])

    @property
    def tv(self) -> np.ndarray:
        return np.asarray([point.tv for point in self.points])

    @property
    def mass_fraction(self) -> np.ndarray:
        return np.asarray([point.mass_fraction for point in self.points])


class Air5PostShockReference:
    """Frozen jump followed by steady inviscid finite-rate air relaxation.

    The relaxation state contains five mass fractions and specific
    vibrational energy. Density, speed and translational temperature are
    reconstructed from constant mass, momentum and total-enthalpy fluxes.
    """

    DEFAULT_MASS_FRACTION = np.array(
        [0.765297, 0.234699, 1.0e-6, 1.0e-6, 2.0e-6], dtype=float
    )
    JACOBIAN_TRIAL_MASS_TOLERANCE = 1.0e-6

    def __init__(
        self,
        chemistry: Air5RadauReference,
        *,
        upstream_temperature: float = 500.0,
        upstream_pressure: float = 5.0e3,
        upstream_mach: float = 8.0,
        upstream_tv: float = 500.0,
        mass_fraction: np.ndarray | None = None,
    ):
        self.chemistry = chemistry
        self._mass_fraction = np.asarray(
            self.DEFAULT_MASS_FRACTION if mass_fraction is None else mass_fraction,
            dtype=float,
        ).copy()
        self._validate_mass_fraction(self._mass_fraction)
        if upstream_mach <= 1.0:
            raise ValueError("upstream Mach number must be supersonic")
        if not (chemistry.temperature_bounds[0] <= upstream_temperature <= chemistry.temperature_bounds[1]):
            raise ValueError("upstream temperature is outside the mechanism domain")
        if not (chemistry.pressure_bounds[0] <= upstream_pressure <= chemistry.pressure_bounds[1]):
            raise ValueError("upstream pressure is outside the mechanism domain")

        gas_constant = self._mixture_gas_constant(self._mass_fraction)
        cv = float(np.dot(self._mass_fraction, chemistry.cv_tr))
        gamma = (cv + gas_constant) / cv
        density = upstream_pressure / (gas_constant * upstream_temperature)
        speed = upstream_mach * math.sqrt(gamma * gas_constant * upstream_temperature)
        rho_species = density * self._mass_fraction
        ev_specific = chemistry.ev_from_tv(rho_species, upstream_tv) / density
        self.upstream = self._make_point(
            density, speed, upstream_pressure, upstream_temperature,
            upstream_tv, self._mass_fraction, ev_specific,
        )

        density_ratio = (
            (gamma + 1.0) * upstream_mach**2
            / ((gamma - 1.0) * upstream_mach**2 + 2.0)
        )
        pressure_ratio = 1.0 + 2.0 * gamma / (gamma + 1.0) * (
            upstream_mach**2 - 1.0
        )
        post_density = density * density_ratio
        post_speed = speed / density_ratio
        post_pressure = upstream_pressure * pressure_ratio
        post_temperature = post_pressure / (post_density * gas_constant)
        self.postshock = self._make_point(
            post_density, post_speed, post_pressure, post_temperature,
            upstream_tv, self._mass_fraction, ev_specific,
        )

        self.mass_flux = post_density * post_speed
        self.momentum_flux = post_density * post_speed**2 + post_pressure
        self.total_enthalpy = self._specific_total_enthalpy(self.postshock)
        self.momentum_per_mass = self.momentum_flux / self.mass_flux
        self.initial_relaxation_state = np.r_[self._mass_fraction, ev_specific]

    def _validate_mass_fraction(self, mass_fraction: np.ndarray) -> None:
        if mass_fraction.shape != (5,) or not np.all(np.isfinite(mass_fraction)):
            raise ValueError("mass fractions must contain five finite values")
        if np.any(mass_fraction < 0.0):
            raise ValueError("mass fractions must be nonnegative")
        if abs(float(np.sum(mass_fraction)) - 1.0) > 1.0e-13:
            raise ValueError("mass fractions must sum to one")

    def _mixture_gas_constant(self, mass_fraction: np.ndarray) -> float:
        return float(np.dot(mass_fraction, self.chemistry.gas_constant))

    def _mixture_cp(self, mass_fraction: np.ndarray) -> float:
        return float(
            np.dot(
                mass_fraction,
                self.chemistry.cv_tr + self.chemistry.gas_constant,
            )
        )

    def _make_point(
        self,
        density: float,
        speed: float,
        pressure: float,
        temperature: float,
        tv: float,
        mass_fraction: np.ndarray,
        ev_specific: float,
    ) -> Air5FlowPoint:
        gas_constant = self._mixture_gas_constant(mass_fraction)
        cv = float(np.dot(mass_fraction, self.chemistry.cv_tr))
        gamma = (cv + gas_constant) / cv
        mach = speed / math.sqrt(gamma * gas_constant * temperature)
        return Air5FlowPoint(
            density=float(density),
            speed=float(speed),
            pressure=float(pressure),
            temperature=float(temperature),
            tv=float(tv),
            mass_fraction=np.asarray(mass_fraction, dtype=float).copy(),
            ev_specific=float(ev_specific),
            mach=float(mach),
        )

    def _specific_total_enthalpy(self, point: Air5FlowPoint) -> float:
        formation = float(
            np.dot(point.mass_fraction, self.chemistry.formation_energy)
        )
        return (
            0.5 * point.speed**2
            + self._mixture_cp(point.mass_fraction) * point.temperature
            + formation
            + point.ev_specific
        )

    def conservative_fluxes(self, point: Air5FlowPoint) -> np.ndarray:
        mass_flux = point.density * point.speed
        momentum_flux = point.density * point.speed**2 + point.pressure
        energy_flux = mass_flux * self._specific_total_enthalpy(point)
        return np.asarray([mass_flux, momentum_flux, energy_flux])

    def conservative_state(self, point: Air5FlowPoint) -> np.ndarray:
        rho_species = point.density * point.mass_fraction
        ev = point.density * point.ev_specific
        momentum = np.asarray([point.density * point.speed, 0.0, 0.0])
        q5 = self.chemistry.q5_from_state(
            point.density,
            momentum,
            rho_species,
            ev,
            point.temperature,
        )
        return np.r_[point.density, momentum, q5, rho_species, ev]

    def reconstruct(self, state: np.ndarray) -> Air5FlowPoint:
        state = np.asarray(state, dtype=float)
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            raise ValueError("relaxation state must contain six finite values")
        mass_fraction = state[:5]
        if np.any(mass_fraction < 0.0) or state[5] < 0.0:
            raise ValueError("relaxation state must be nonnegative")
        # Radau's finite-difference Jacobian perturbs one component at a time.
        # Do not normalize that trial state; final mass closure is checked separately.
        if (
            abs(float(np.sum(mass_fraction)) - 1.0)
            > self.JACOBIAN_TRIAL_MASS_TOLERANCE
        ):
            raise ValueError("relaxation mass fractions do not sum to one")

        gas_constant = self._mixture_gas_constant(mass_fraction)
        cp = self._mixture_cp(mass_fraction)
        cp_over_r = cp / gas_constant
        formation = float(
            np.dot(mass_fraction, self.chemistry.formation_energy)
        )
        available_enthalpy = self.total_enthalpy - formation - state[5]
        coefficient = cp_over_r - 0.5
        discriminant = (
            (cp_over_r * self.momentum_per_mass) ** 2
            - 4.0 * coefficient * available_enthalpy
        )
        if discriminant <= 0.0 or not math.isfinite(discriminant):
            raise ValueError("post-shock subsonic reconstruction has no real root")
        speed = (
            cp_over_r * self.momentum_per_mass - math.sqrt(discriminant)
        ) / (2.0 * coefficient)
        temperature = (
            speed * (self.momentum_per_mass - speed) / gas_constant
        )
        density = self.mass_flux / speed
        pressure = density * gas_constant * temperature
        if not (self.chemistry.temperature_bounds[0] <= temperature <= self.chemistry.temperature_bounds[1]):
            raise ValueError("reconstructed temperature is outside the mechanism domain")
        if not (self.chemistry.pressure_bounds[0] <= pressure <= self.chemistry.pressure_bounds[1]):
            raise ValueError("reconstructed pressure is outside the mechanism domain")
        rho_species = density * mass_fraction
        tv = self.chemistry.tv_from_ev(rho_species, density * state[5])
        return self._make_point(
            density, speed, pressure, temperature, tv, mass_fraction, state[5]
        )

    def spatial_rhs(
        self, _x: float, state: np.ndarray, *, source_mode: str = "coupled"
    ) -> np.ndarray:
        point = self.reconstruct(state)
        rho_species = point.density * point.mass_fraction
        ev = point.density * point.ev_specific
        momentum = np.asarray([self.mass_flux, 0.0, 0.0])
        q5 = self.chemistry.q5_from_state(
            point.density,
            momentum,
            rho_species,
            ev,
            point.temperature,
        )
        source = self.chemistry.rhs(
            0.0,
            np.r_[rho_species, ev],
            point.density,
            momentum,
            q5,
            source_mode=source_mode,
        )
        return source / self.mass_flux

    def integration_atol(self) -> np.ndarray:
        return np.r_[np.full(5, 1.0e-13), 1.0e-5]

    def _admissible_first_step(self, length: float, source_mode: str) -> float:
        trial_step = length / 200.0
        initial_rhs = self.spatial_rhs(
            0.0, self.initial_relaxation_state, source_mode=source_mode
        )
        for _ in range(80):
            try:
                self.reconstruct(
                    self.initial_relaxation_state + trial_step * initial_rhs
                )
            except ValueError:
                trial_step *= 0.5
                continue
            return 0.5 * trial_step
        raise RuntimeError("could not select an admissible post-shock first step")

    def integrate(
        self,
        length: float,
        *,
        source_mode: str = "coupled",
        rtol: float = 1.0e-9,
    ):
        if not math.isfinite(length) or length <= 0.0:
            raise ValueError("relaxation length must be finite and positive")
        if not math.isfinite(rtol) or rtol <= 0.0:
            raise ValueError("relative tolerance must be finite and positive")
        result = solve_ivp(
            lambda x, state: self.spatial_rhs(x, state, source_mode=source_mode),
            (0.0, length),
            self.initial_relaxation_state,
            method="Radau",
            rtol=rtol,
            atol=self.integration_atol(),
            dense_output=True,
            max_step=length / 200.0,
            first_step=self._admissible_first_step(length, source_mode),
        )
        if not result.success:
            raise RuntimeError(f"post-shock Radau integration failed: {result.message}")
        if not np.all(np.isfinite(result.y)) or np.any(result.y < 0.0):
            raise RuntimeError("post-shock Radau integration left the admissible domain")
        return result

    def sample(self, result, x: np.ndarray) -> Air5RelaxationProfile:
        locations = np.asarray(x, dtype=float)
        if locations.ndim != 1 or not np.all(np.isfinite(locations)):
            raise ValueError("sample locations must be a finite one-dimensional array")
        if np.any(locations < result.t[0]) or np.any(locations > result.t[-1]):
            raise ValueError("sample locations are outside the integrated interval")
        states = result.sol(locations)
        points = tuple(self.reconstruct(states[:, index]) for index in range(locations.size))
        return Air5RelaxationProfile(x=locations.copy(), points=points)
