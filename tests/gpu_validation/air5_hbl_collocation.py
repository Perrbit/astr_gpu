"""Global FP64 mass-streamfunction collocation for the air5 HBL reference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, root
from scipy.sparse import lil_matrix

try:
    from tests.gpu_validation.air5_hbl_reference import (
        Air5BoundaryLayerReference,
        Air5BoundaryLayerState,
    )
except ModuleNotFoundError:
    from air5_hbl_reference import Air5BoundaryLayerReference, Air5BoundaryLayerState


@dataclass(frozen=True)
class Air5HblCollocationBoundary:
    pressure: float
    edge_velocity: float
    edge_temperature: float
    wall_velocity: float
    wall_temperature: float
    mass_fraction: np.ndarray


@dataclass(frozen=True)
class Air5HblCollocationResult:
    x: np.ndarray
    y: np.ndarray
    mass_streamfunction: np.ndarray
    primitive: np.ndarray
    scaled_residual_max: float
    function_evaluations: int


@dataclass(frozen=True)
class Air5HblCollocationWall:
    skin_friction: np.ndarray
    shear_stress: np.ndarray
    translational_heat_flux: np.ndarray
    vibrational_heat_flux: np.ndarray
    total_heat_flux: np.ndarray


def _derivative_matrix(
    coordinate: np.ndarray, *, stencil_width: int = 3
) -> np.ndarray:
    coordinate = np.asarray(coordinate, dtype=np.float64)
    if (
        coordinate.ndim != 1
        or not np.all(np.isfinite(coordinate))
        or np.any(np.diff(coordinate) <= 0.0)
    ):
        raise ValueError("collocation coordinate must be finite and increasing")
    if (
        not isinstance(stencil_width, int)
        or stencil_width < 3
        or stencil_width % 2 == 0
        or coordinate.size < stencil_width
    ):
        raise ValueError(
            "derivative stencil width must be an odd integer within the coordinate"
        )
    matrix = np.zeros((coordinate.size, coordinate.size), dtype=np.float64)
    half_width = stencil_width // 2
    for index in range(coordinate.size):
        start = min(
            max(index - half_width, 0), coordinate.size - stencil_width
        )
        selected = np.arange(start, start + stencil_width)
        offset = coordinate[selected] - coordinate[index]
        vandermonde = np.vstack(
            [offset**power for power in range(stencil_width)]
        )
        right_hand_side = np.zeros(stencil_width, dtype=np.float64)
        right_hand_side[1] = 1.0
        matrix[index, selected] = np.linalg.solve(
            vandermonde, right_hand_side
        )
        matrix[index, index] -= np.sum(matrix[index])
    return matrix


def _differentiate(
    derivative: np.ndarray, values: np.ndarray, axis: int
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.tensordot(derivative, values, axes=(1, axis)).swapaxes(0, axis)


def _solve_square_hybrid(
    residual,
    initial: np.ndarray,
    *,
    residual_tolerance: float,
    max_function_evaluations: int,
) -> tuple[np.ndarray, float, int]:
    """Solve a square nonlinear system with an explicit residual gate."""
    current = np.asarray(initial, dtype=np.float64).copy()
    if current.ndim != 1 or not np.all(np.isfinite(current)):
        raise ValueError("hybrid initial state must be a finite vector")
    if residual_tolerance <= 0.0 or max_function_evaluations <= 0:
        raise ValueError("hybrid solver limits must be positive")
    total_evaluations = 0
    last_result = None
    residual_max = np.inf
    for _ in range(3):
        remaining = max_function_evaluations - total_evaluations
        if remaining <= 0:
            break
        last_result = root(
            residual,
            current,
            method="hybr",
            options={"xtol": 1.0e-12, "maxfev": remaining, "factor": 0.1},
        )
        total_evaluations += int(last_result.nfev)
        current = np.asarray(last_result.x, dtype=np.float64)
        residual_max = float(np.max(np.abs(last_result.fun)))
        if np.all(np.isfinite(current)) and residual_max <= residual_tolerance:
            return current, residual_max, total_evaluations
    message = "function-evaluation budget exhausted"
    if last_result is not None:
        message = str(last_result.message)
    raise RuntimeError(
        "square hybrid solve failed: "
        f"residual={residual_max:.3e} evaluations={total_evaluations} "
        f"message={message}"
    )


class Air5HblCollocation:
    """Global x-y thin-layer discretization used before enabling source terms.

    The mass streamfunction enforces ``rho*u = dpsi/dy`` and
    ``rho*v = -dpsi/dx``. The source-off single-temperature system solves the
    streamwise momentum and total-energy equations globally in x and y.
    """

    def __init__(
        self, mechanism_path: Path, boundary: Air5HblCollocationBoundary
    ) -> None:
        self.reference = Air5BoundaryLayerReference(mechanism_path)
        self.boundary = boundary
        mass_fraction = np.asarray(boundary.mass_fraction, dtype=np.float64)
        values = np.r_[
            boundary.pressure,
            boundary.edge_velocity,
            boundary.edge_temperature,
            boundary.wall_velocity,
            boundary.wall_temperature,
            mass_fraction,
        ]
        if (
            mass_fraction.shape != (5,)
            or not np.all(np.isfinite(values))
            or boundary.pressure <= 0.0
            or boundary.edge_velocity <= 0.0
            or boundary.edge_temperature <= 0.0
            or boundary.wall_temperature <= 0.0
            or boundary.wall_velocity < 0.0
            or np.any(mass_fraction < 0.0)
            or not np.isclose(np.sum(mass_fraction), 1.0, atol=2.0e-14, rtol=0.0)
        ):
            raise ValueError("invalid air5 HBL collocation boundary")
        self._mass_fraction = mass_fraction.copy()
        self._gas_constant = float(
            np.dot(mass_fraction, self.reference.chemistry.gas_constant)
        )
        molar_mass = self.reference.transport.molar_mass
        mole_fraction = mass_fraction / molar_mass
        self._mole_fraction = mole_fraction / np.sum(mole_fraction)
        self._cv_tr = float(
            np.dot(mass_fraction, self.reference.chemistry.cv_tr)
        )
        self._formation_energy = float(
            np.dot(mass_fraction, self.reference.chemistry.formation_energy)
        )

    def _validate_grid(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        _derivative_matrix(x, stencil_width=3)
        _derivative_matrix(y, stencil_width=5)
        if x.size < 3 or y.size < 5 or y[0] != 0.0 or x[0] <= 0.0:
            raise ValueError("air5 HBL collocation requires x>0 and y[0]=0")
        return x, y

    @staticmethod
    def _jacobian_sparsity(
        x_points: int, y_points: int, field_count: int = 2
    ):
        field_size = x_points * y_points * field_count
        sparsity = lil_matrix((field_size, field_size), dtype=np.int8)
        for i in range(x_points):
            for j in range(y_points):
                row = field_count * (i * y_points + j)
                if i == 0:
                    x_dependencies = (0,)
                    y_dependencies = (j,)
                else:
                    x_dependencies = range(max(0, i - 2), min(x_points, i + 3))
                    y_dependencies = range(max(0, j - 6), min(y_points, j + 7))
                for ii in x_dependencies:
                    for jj in y_dependencies:
                        column = field_count * (ii * y_points + jj)
                        sparsity[
                            row : row + field_count,
                            column : column + field_count,
                        ] = 1
        return sparsity.tocsr()

    def _uniform_inlet(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        boundary = self.boundary
        if not (
            boundary.wall_velocity == boundary.edge_velocity
            and boundary.wall_temperature == boundary.edge_temperature
        ):
            raise ValueError("a nonuniform collocation problem requires an inlet profile")
        density = boundary.pressure / (
            self._gas_constant * boundary.edge_temperature
        )
        return (
            density * boundary.edge_velocity * y,
            np.full_like(y, boundary.edge_temperature),
        )

    def _primitive(
        self,
        streamfunction: np.ndarray,
        temperature: np.ndarray,
        derivative_x: np.ndarray,
        derivative_y: np.ndarray,
        tv: np.ndarray | None = None,
    ) -> np.ndarray:
        if tv is None:
            tv = temperature
        tv = np.asarray(tv, dtype=np.float64)
        if tv.shape != temperature.shape:
            raise ValueError("air5 HBL vibrational temperature has an invalid shape")
        density = self.boundary.pressure / (self._gas_constant * temperature)
        streamwise_velocity = _differentiate(
            derivative_y, streamfunction, axis=1
        ) / density
        normal_velocity = -_differentiate(
            derivative_x, streamfunction, axis=0
        ) / density
        primitive = np.empty(streamfunction.shape + (8,), dtype=np.float64)
        primitive[:, :, 0] = streamwise_velocity
        primitive[:, :, 1] = normal_velocity
        primitive[:, :, 2] = temperature
        primitive[:, :, 3] = tv
        primitive[:, :, 4:8] = self._mass_fraction[None, None, 1:5]
        return primitive

    def wall_quantities(
        self, result: Air5HblCollocationResult
    ) -> Air5HblCollocationWall:
        """Evaluate positive-y wall shear and conductive energy fluxes."""
        if (
            result.primitive.shape != (result.x.size, result.y.size, 8)
            or result.mass_streamfunction.shape != (result.x.size, result.y.size)
        ):
            raise ValueError("air5 HBL collocation result has an invalid shape")
        derivative_y = _derivative_matrix(result.y, stencil_width=5)
        velocity_gradient = _differentiate(
            derivative_y, result.primitive[:, :, 0], axis=1
        )[:, 0]
        temperature_gradient = _differentiate(
            derivative_y, result.primitive[:, :, 2], axis=1
        )[:, 0]
        tv_gradient = _differentiate(
            derivative_y, result.primitive[:, :, 3], axis=1
        )[:, 0]
        shear_stress = np.empty(result.x.size, dtype=np.float64)
        translational_heat_flux = np.empty_like(shear_stress)
        vibrational_heat_flux = np.empty_like(shear_stress)
        for index in range(result.x.size):
            viscosity, conductivity_tr, conductivity_v, _ = (
                self.reference.transport.transport_properties(
                    result.primitive[index, 0, 2],
                    result.primitive[index, 0, 3],
                    self.boundary.pressure,
                    self._mass_fraction,
                )
            )
            shear_stress[index] = viscosity * velocity_gradient[index]
            translational_heat_flux[index] = (
                conductivity_tr * temperature_gradient[index]
            )
            vibrational_heat_flux[index] = conductivity_v * tv_gradient[index]
        edge_density = self.boundary.pressure / (
            self._gas_constant * self.boundary.edge_temperature
        )
        skin_friction = 2.0 * shear_stress / (
            edge_density * self.boundary.edge_velocity**2
        )
        return Air5HblCollocationWall(
            skin_friction=skin_friction,
            shear_stress=shear_stress,
            translational_heat_flux=translational_heat_flux,
            vibrational_heat_flux=vibrational_heat_flux,
            total_heat_flux=translational_heat_flux + vibrational_heat_flux,
        )

    def _frozen_two_temperature_fluxes(
        self,
        primitive: np.ndarray,
        velocity_gradient_y: np.ndarray,
        temperature_gradient_y: np.ndarray,
        tv_gradient_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate frozen-composition momentum, total-energy, and Ev fluxes."""
        primitive = np.asarray(primitive, dtype=np.float64)
        velocity_gradient_y = np.asarray(velocity_gradient_y, dtype=np.float64)
        temperature_gradient_y = np.asarray(
            temperature_gradient_y, dtype=np.float64
        )
        tv_gradient_y = np.asarray(tv_gradient_y, dtype=np.float64)
        shape = primitive.shape[:-1]
        if (
            primitive.ndim != 3
            or primitive.shape[-1] != 8
            or velocity_gradient_y.shape != shape + (2,)
            or temperature_gradient_y.shape != shape
            or tv_gradient_y.shape != shape
        ):
            raise ValueError("invalid frozen-air5 collocation flux shape")

        transport = self.reference.transport
        temperature = primitive[:, :, 2]
        tv = primitive[:, :, 3]
        log_temperature = np.log(temperature)
        viscosity_species = 0.1 * np.exp(
            transport.blottner[None, None, :, 0] * log_temperature[:, :, None] ** 2
            + transport.blottner[None, None, :, 1]
            * log_temperature[:, :, None]
            + transport.blottner[None, None, :, 2]
        )
        viscosity_ratio = np.sqrt(
            viscosity_species[:, :, :, None]
            / viscosity_species[:, :, None, :]
        )
        molar_mass = transport.molar_mass
        phi = (
            1.0
            + viscosity_ratio
            * (molar_mass[None, None, None, :] / molar_mass[None, None, :, None])
            ** 0.25
        ) ** 2 / np.sqrt(
            8.0
            * (
                1.0
                + molar_mass[None, None, :, None]
                / molar_mass[None, None, None, :]
            )
        )
        denominators = np.einsum("...ij,j->...i", phi, self._mole_fraction)
        viscosity = np.sum(
            self._mole_fraction[None, None, :]
            * viscosity_species
            / denominators,
            axis=2,
        )
        conductivity_species = (
            np.where(transport.atom_count == 1, 3.75, 4.75)[None, None, :]
            * viscosity_species
            * transport.gas_constant[None, None, :]
        )
        ratio = transport.theta_v[None, None, :] / tv[:, :, None]
        active = transport.theta_v > 0.0
        vibrational_energy_species = np.zeros_like(viscosity_species)
        vibrational_cv_species = np.zeros_like(viscosity_species)
        vibrational_energy_species[:, :, active] = (
            transport.gas_constant[active][None, None, :]
            * transport.theta_v[active][None, None, :]
            / np.expm1(ratio[:, :, active])
        )
        vibrational_cv_species[:, :, active] = (
            transport.gas_constant[active][None, None, :]
            * ratio[:, :, active] ** 2
            * np.exp(ratio[:, :, active])
            / np.expm1(ratio[:, :, active]) ** 2
        )
        conductivity_v_species = viscosity_species * vibrational_cv_species
        conductivity_tr = np.sum(
            self._mole_fraction[None, None, :]
            * conductivity_species
            / denominators,
            axis=2,
        )
        conductivity_v = np.sum(
            self._mole_fraction[None, None, :]
            * conductivity_v_species
            / denominators,
            axis=2,
        )

        density = self.boundary.pressure / (self._gas_constant * temperature)
        u = primitive[:, :, 0]
        v = primitive[:, :, 1]
        vibrational_energy = np.sum(
            self._mass_fraction[None, None, :] * vibrational_energy_species,
            axis=2,
        )
        total_energy = density * (
            self._cv_tr * temperature
            + vibrational_energy
            + self._formation_energy
            + 0.5 * (u**2 + v**2)
        )
        vibrational_energy_density = density * vibrational_energy
        streamwise_flux = np.empty(shape + (3,), dtype=np.float64)
        streamwise_flux[:, :, 0] = density * u**2 + self.boundary.pressure
        streamwise_flux[:, :, 1] = u * (
            total_energy + self.boundary.pressure
        )
        streamwise_flux[:, :, 2] = u * vibrational_energy_density

        shear_stress = viscosity * velocity_gradient_y[:, :, 0]
        normal_stress = (
            (4.0 / 3.0) * viscosity * velocity_gradient_y[:, :, 1]
        )
        normal_flux = np.empty_like(streamwise_flux)
        normal_flux[:, :, 0] = density * u * v - shear_stress
        normal_flux[:, :, 1] = v * (
            total_energy + self.boundary.pressure
        ) - (
            shear_stress * u
            + normal_stress * v
            + conductivity_tr * temperature_gradient_y
            + conductivity_v * tv_gradient_y
        )
        normal_flux[:, :, 2] = (
            v * vibrational_energy_density - conductivity_v * tv_gradient_y
        )
        return streamwise_flux, normal_flux

    def _source_off_single_temperature_fluxes(
        self,
        primitive: np.ndarray,
        velocity_gradient_y: np.ndarray,
        temperature_gradient_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        streamwise, normal = self._frozen_two_temperature_fluxes(
            primitive,
            velocity_gradient_y,
            temperature_gradient_y,
            temperature_gradient_y,
        )
        return streamwise[:, :, :2], normal[:, :, :2]

    def _frozen_vt_source(self, primitive: np.ndarray) -> np.ndarray:
        primitive = np.asarray(primitive, dtype=np.float64)
        if primitive.ndim != 3 or primitive.shape[-1] != 8:
            raise ValueError("invalid frozen-air5 V-T source shape")
        temperature = primitive[:, :, 2]
        tv = primitive[:, :, 3]
        density = self.boundary.pressure / (self._gas_constant * temperature)
        chemistry = self.reference.chemistry
        rho_species = density[:, :, None] * self._mass_fraction[None, None, :]
        concentration = rho_species / chemistry.molar_mass[None, None, :]
        mole_fraction = concentration / np.sum(concentration, axis=2)[:, :, None]
        source = np.zeros(temperature.shape, dtype=np.float64)
        for molecule in range(5):
            theta = chemistry.theta_v[molecule]
            if theta <= 0.0 or self._mass_fraction[molecule] <= 0.0:
                continue
            mw_time = chemistry.ATMOSPHERE_PA / self.boundary.pressure * np.exp(
                chemistry.mw_a[molecule][None, None, :]
                * (
                    temperature[:, :, None] ** (-1.0 / 3.0)
                    - chemistry.mw_b[molecule][None, None, :]
                )
                - 18.42
            )
            park_time = np.zeros_like(mw_time)
            collision = chemistry.sigma0[molecule] > 0.0
            if np.any(collision):
                sigma = chemistry.sigma0[molecule, collision][None, None, :] * (
                    temperature[:, :, None]
                    ** chemistry.sigma_power[molecule, collision][None, None, :]
                )
                reduced_mass = (
                    chemistry.molar_mass[molecule]
                    * chemistry.molar_mass[collision]
                    / (
                        chemistry.molar_mass[molecule]
                        + chemistry.molar_mass[collision]
                    )
                )
                mean_speed = np.sqrt(
                    8.0
                    * chemistry.RU
                    * temperature[:, :, None]
                    / (np.pi * reduced_mass[None, None, :])
                )
                number_density = (
                    concentration[:, :, molecule] * chemistry.AVOGADRO
                )
                park_time[:, :, collision] = 1.0 / (
                    number_density[:, :, None] * sigma * mean_speed
                )
            inverse_time = np.sum(mole_fraction / (mw_time + park_time), axis=2)
            equilibrium = chemistry.gas_constant[molecule] * theta / np.expm1(
                theta / temperature
            )
            current = chemistry.gas_constant[molecule] * theta / np.expm1(
                theta / tv
            )
            source += (
                rho_species[:, :, molecule]
                * (equilibrium - current)
                * inverse_time
            )
        return source

    def source_off_single_temperature_residual(
        self,
        streamfunction: np.ndarray,
        temperature: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        *,
        inlet_streamfunction: np.ndarray | None = None,
        inlet_temperature: np.ndarray | None = None,
    ) -> np.ndarray:
        x, y = self._validate_grid(x, y)
        streamfunction = np.asarray(streamfunction, dtype=np.float64)
        temperature = np.asarray(temperature, dtype=np.float64)
        expected_shape = (x.size, y.size)
        if streamfunction.shape != expected_shape or temperature.shape != expected_shape:
            raise ValueError("air5 HBL collocation field has an invalid shape")
        temperature_minimum, temperature_maximum = (
            self.reference.chemistry.temperature_bounds
        )
        if (
            not np.all(np.isfinite(streamfunction))
            or not np.all(np.isfinite(temperature))
            or np.any(temperature < temperature_minimum)
            or np.any(temperature > temperature_maximum)
        ):
            return np.full(expected_shape + (2,), 1.0e6)
        if inlet_streamfunction is None or inlet_temperature is None:
            inlet_streamfunction, inlet_temperature = self._uniform_inlet(y)
        inlet_streamfunction = np.asarray(inlet_streamfunction, dtype=np.float64)
        inlet_temperature = np.asarray(inlet_temperature, dtype=np.float64)
        if inlet_streamfunction.shape != y.shape or inlet_temperature.shape != y.shape:
            raise ValueError("air5 HBL collocation inlet has an invalid shape")

        derivative_x = _derivative_matrix(x, stencil_width=3)
        derivative_y = _derivative_matrix(y, stencil_width=5)
        primitive = self._primitive(
            streamfunction, temperature, derivative_x, derivative_y
        )
        velocity_gradient_y = _differentiate(
            derivative_y, primitive[:, :, 0:2], axis=1
        )
        temperature_gradient_y = _differentiate(
            derivative_y, temperature, axis=1
        )
        streamwise_flux, normal_flux = (
            self._source_off_single_temperature_fluxes(
                primitive, velocity_gradient_y, temperature_gradient_y
            )
        )

        edge = Air5BoundaryLayerState(
            pressure=self.boundary.pressure,
            velocity=np.array([self.boundary.edge_velocity, 0.0, 0.0]),
            temperature=self.boundary.edge_temperature,
            tv=self.boundary.edge_temperature,
            mass_fraction=self._mass_fraction,
        )
        edge_density = self.reference.density(edge)
        x_scale = x[-1] - x[0]
        flux_scale = np.maximum(
            np.abs(self.reference.streamwise_flux(edge)[[1, 6]]) / x_scale,
            np.finfo(np.float64).tiny,
        )
        streamfunction_scale = max(
            edge_density * self.boundary.edge_velocity * y[-1],
            np.finfo(np.float64).tiny,
        )
        residual = (
            _differentiate(derivative_x, streamwise_flux, axis=0)
            + _differentiate(derivative_y, normal_flux, axis=1)
        ) / flux_scale

        residual[0, :, 0] = (
            streamfunction[0] - inlet_streamfunction
        ) / streamfunction_scale
        residual[0, :, 1] = (
            temperature[0] - inlet_temperature
        ) / self.boundary.edge_temperature
        for i in range(1, x.size):
            residual[i, 0, 0] = streamfunction[i, 0] / streamfunction_scale
            residual[i, 0, 1] = (
                temperature[i, 0] - self.boundary.wall_temperature
            ) / self.boundary.edge_temperature
            residual[i, 1, 0] = (
                primitive[i, 0, 0] - self.boundary.wall_velocity
            ) / self.boundary.edge_velocity
            residual[i, -1, 0] = (
                primitive[i, -1, 0] - self.boundary.edge_velocity
            ) / self.boundary.edge_velocity
            residual[i, -1, 1] = (
                temperature[i, -1] - self.boundary.edge_temperature
            ) / self.boundary.edge_temperature
        return residual

    def frozen_vt_residual(
        self,
        streamfunction: np.ndarray,
        temperature: np.ndarray,
        tv: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        *,
        inlet_streamfunction: np.ndarray | None = None,
        inlet_temperature: np.ndarray | None = None,
        inlet_tv: np.ndarray | None = None,
        source_multiplier: float = 1.0,
    ) -> np.ndarray:
        """Return the frozen-composition momentum/energy/Ev V-T residual."""
        x, y = self._validate_grid(x, y)
        source_multiplier = float(source_multiplier)
        if not np.isfinite(source_multiplier) or not 0.0 <= source_multiplier <= 1.0:
            raise ValueError("V-T source multiplier must be between zero and one")
        streamfunction = np.asarray(streamfunction, dtype=np.float64)
        temperature = np.asarray(temperature, dtype=np.float64)
        tv = np.asarray(tv, dtype=np.float64)
        expected_shape = (x.size, y.size)
        if (
            streamfunction.shape != expected_shape
            or temperature.shape != expected_shape
            or tv.shape != expected_shape
        ):
            raise ValueError("air5 HBL V-T field has an invalid shape")
        temperature_minimum, temperature_maximum = (
            self.reference.chemistry.temperature_bounds
        )
        if (
            not np.all(np.isfinite(streamfunction))
            or not np.all(np.isfinite(temperature))
            or not np.all(np.isfinite(tv))
            or np.any(temperature < temperature_minimum)
            or np.any(temperature > temperature_maximum)
            or np.any(tv < temperature_minimum)
            or np.any(tv > temperature_maximum)
        ):
            return np.full(expected_shape + (3,), 1.0e6)
        if (
            inlet_streamfunction is None
            or inlet_temperature is None
            or inlet_tv is None
        ):
            inlet_streamfunction, inlet_temperature = self._uniform_inlet(y)
            inlet_tv = inlet_temperature.copy()
        inlet_streamfunction = np.asarray(inlet_streamfunction, dtype=np.float64)
        inlet_temperature = np.asarray(inlet_temperature, dtype=np.float64)
        inlet_tv = np.asarray(inlet_tv, dtype=np.float64)
        if (
            inlet_streamfunction.shape != y.shape
            or inlet_temperature.shape != y.shape
            or inlet_tv.shape != y.shape
        ):
            raise ValueError("air5 HBL V-T inlet has an invalid shape")

        derivative_x = _derivative_matrix(x, stencil_width=3)
        derivative_y = _derivative_matrix(y, stencil_width=5)
        primitive = self._primitive(
            streamfunction,
            temperature,
            derivative_x,
            derivative_y,
            tv=tv,
        )
        velocity_gradient_y = _differentiate(
            derivative_y, primitive[:, :, 0:2], axis=1
        )
        temperature_gradient_y = _differentiate(
            derivative_y, temperature, axis=1
        )
        tv_gradient_y = _differentiate(derivative_y, tv, axis=1)
        streamwise_flux, normal_flux = self._frozen_two_temperature_fluxes(
            primitive,
            velocity_gradient_y,
            temperature_gradient_y,
            tv_gradient_y,
        )
        source = np.zeros(expected_shape + (3,), dtype=np.float64)
        if source_multiplier > 0.0:
            source[:, :, 2] = source_multiplier * self._frozen_vt_source(primitive)

        edge = Air5BoundaryLayerState(
            pressure=self.boundary.pressure,
            velocity=np.array([self.boundary.edge_velocity, 0.0, 0.0]),
            temperature=self.boundary.edge_temperature,
            tv=self.boundary.edge_temperature,
            mass_fraction=self._mass_fraction,
        )
        edge_density = self.reference.density(edge)
        x_scale = x[-1] - x[0]
        flux_scale = np.maximum(
            np.abs(self.reference.streamwise_flux(edge)[[1, 6, 7]]) / x_scale,
            np.finfo(np.float64).tiny,
        )
        streamfunction_scale = max(
            edge_density * self.boundary.edge_velocity * y[-1],
            np.finfo(np.float64).tiny,
        )
        residual = (
            _differentiate(derivative_x, streamwise_flux, axis=0)
            + _differentiate(derivative_y, normal_flux, axis=1)
            - source
        ) / flux_scale

        residual[0, :, 0] = (
            streamfunction[0] - inlet_streamfunction
        ) / streamfunction_scale
        residual[0, :, 1] = (
            temperature[0] - inlet_temperature
        ) / self.boundary.edge_temperature
        residual[0, :, 2] = (
            tv[0] - inlet_tv
        ) / self.boundary.edge_temperature
        for i in range(1, x.size):
            residual[i, 0, 0] = streamfunction[i, 0] / streamfunction_scale
            residual[i, 0, 1] = (
                temperature[i, 0] - self.boundary.wall_temperature
            ) / self.boundary.edge_temperature
            residual[i, 0, 2] = (
                tv[i, 0] - self.boundary.wall_temperature
            ) / self.boundary.edge_temperature
            residual[i, 1, 0] = (
                primitive[i, 0, 0] - self.boundary.wall_velocity
            ) / self.boundary.edge_velocity
            residual[i, -1, 0] = (
                primitive[i, -1, 0] - self.boundary.edge_velocity
            ) / self.boundary.edge_velocity
            residual[i, -1, 1] = (
                temperature[i, -1] - self.boundary.edge_temperature
            ) / self.boundary.edge_temperature
            residual[i, -1, 2] = (
                tv[i, -1] - self.boundary.edge_temperature
            ) / self.boundary.edge_temperature
        return residual

    def solve_frozen_vt(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        initial_streamfunction: np.ndarray,
        initial_temperature: np.ndarray,
        initial_tv: np.ndarray,
        residual_tolerance: float = 1.0e-10,
        max_function_evaluations: int = 500,
        source_multiplier: float = 1.0,
    ) -> Air5HblCollocationResult:
        """Solve the frozen-composition HBL with V-T relaxation enabled."""
        x, y = self._validate_grid(x, y)
        initial_streamfunction = np.asarray(
            initial_streamfunction, dtype=np.float64
        )
        initial_temperature = np.asarray(initial_temperature, dtype=np.float64)
        initial_tv = np.asarray(initial_tv, dtype=np.float64)
        shape = (x.size, y.size)
        if (
            initial_streamfunction.shape != shape
            or initial_temperature.shape != shape
            or initial_tv.shape != shape
        ):
            raise ValueError("air5 HBL V-T initial field has an invalid shape")
        inlet_streamfunction = initial_streamfunction[0].copy()
        inlet_temperature = initial_temperature[0].copy()
        inlet_tv = initial_tv[0].copy()
        edge_density = self.boundary.pressure / (
            self._gas_constant * self.boundary.edge_temperature
        )
        variable_scale = np.array(
            [
                max(
                    np.max(np.abs(initial_streamfunction)),
                    edge_density * self.boundary.edge_velocity * y[-1],
                ),
                self.boundary.edge_temperature,
                self.boundary.edge_temperature,
            ]
        )
        initial = np.stack(
            (initial_streamfunction, initial_temperature, initial_tv), axis=-1
        )
        initial[1:, 0, 0] = 0.0
        initial[1:, 0, 1:3] = self.boundary.wall_temperature
        initial[1:, -1, 1:3] = self.boundary.edge_temperature
        unknown = np.ones(shape + (3,), dtype=bool)
        unknown[0, :, :] = False
        unknown[1:, 0, :] = False
        unknown[1:, -1, 1:3] = False
        normalized_template = initial / variable_scale

        def residual(packed: np.ndarray) -> np.ndarray:
            normalized = normalized_template.copy()
            normalized[unknown] = packed
            fields = normalized * variable_scale
            full_residual = self.frozen_vt_residual(
                fields[:, :, 0],
                fields[:, :, 1],
                fields[:, :, 2],
                x,
                y,
                inlet_streamfunction=inlet_streamfunction,
                inlet_temperature=inlet_temperature,
                inlet_tv=inlet_tv,
                source_multiplier=source_multiplier,
            )
            return full_residual[unknown]

        normalized_initial = normalized_template[unknown]
        initial_residual = residual(normalized_initial)
        if np.max(np.abs(initial_residual)) <= residual_tolerance:
            solved = initial
            residual_max = float(np.max(np.abs(initial_residual)))
            function_evaluations = 1
        else:
            normalized_solution, residual_max, function_evaluations = (
                _solve_square_hybrid(
                    residual,
                    normalized_initial,
                    residual_tolerance=residual_tolerance,
                    max_function_evaluations=max_function_evaluations,
                )
            )
            reconstructed = normalized_template.copy()
            reconstructed[unknown] = normalized_solution
            solved = reconstructed * variable_scale
            final_residual = residual(normalized_solution)
            residual_max = float(np.max(np.abs(final_residual)))
            if residual_max > residual_tolerance:
                raise RuntimeError(
                    "global frozen V-T air5 HBL collocation failed final gate: "
                    f"residual={residual_max:.3e}"
                )

        derivative_x = _derivative_matrix(x, stencil_width=3)
        derivative_y = _derivative_matrix(y, stencil_width=5)
        primitive = self._primitive(
            solved[:, :, 0],
            solved[:, :, 1],
            derivative_x,
            derivative_y,
            tv=solved[:, :, 2],
        )
        return Air5HblCollocationResult(
            x=x.copy(),
            y=y.copy(),
            mass_streamfunction=solved[:, :, 0].copy(),
            primitive=primitive,
            scaled_residual_max=residual_max,
            function_evaluations=function_evaluations,
        )

    def _predict_frozen_tv_with_evaluations(
        self,
        streamfunction: np.ndarray,
        temperature: np.ndarray,
        initial_tv: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        *,
        residual_tolerance: float = 1.0e-10,
        max_function_evaluations: int = 500,
    ) -> tuple[np.ndarray, int]:
        """Solve the source-free Ev equation with psi and T held fixed."""
        x, y = self._validate_grid(x, y)
        streamfunction = np.asarray(streamfunction, dtype=np.float64)
        temperature = np.asarray(temperature, dtype=np.float64)
        initial_tv = np.asarray(initial_tv, dtype=np.float64)
        shape = (x.size, y.size)
        if (
            streamfunction.shape != shape
            or temperature.shape != shape
            or initial_tv.shape != shape
        ):
            raise ValueError("air5 HBL Ev predictor field has an invalid shape")
        inlet_streamfunction = streamfunction[0].copy()
        inlet_temperature = temperature[0].copy()
        inlet_tv = initial_tv[0].copy()
        template = initial_tv.copy()
        template[1:, 0] = self.boundary.wall_temperature
        template[1:, -1] = self.boundary.edge_temperature
        unknown = np.ones(shape, dtype=bool)
        unknown[0, :] = False
        unknown[1:, 0] = False
        unknown[1:, -1] = False
        scale = self.boundary.edge_temperature

        def residual(packed: np.ndarray) -> np.ndarray:
            trial = template.copy()
            trial[unknown] = packed * scale
            full_residual = self.frozen_vt_residual(
                streamfunction,
                temperature,
                trial,
                x,
                y,
                inlet_streamfunction=inlet_streamfunction,
                inlet_temperature=inlet_temperature,
                inlet_tv=inlet_tv,
                source_multiplier=0.0,
            )
            return full_residual[:, :, 2][unknown]

        normalized_initial = template[unknown] / scale
        initial_residual = residual(normalized_initial)
        if np.max(np.abs(initial_residual)) <= residual_tolerance:
            return template, 1
        temperature_minimum, temperature_maximum = (
            self.reference.chemistry.temperature_bounds
        )
        result = least_squares(
            residual,
            normalized_initial,
            bounds=(temperature_minimum / scale, temperature_maximum / scale),
            x_scale="jac",
            xtol=1.0e-12,
            ftol=1.0e-12,
            gtol=1.0e-12,
            max_nfev=max_function_evaluations,
            tr_solver="exact",
        )
        residual_max = float(np.max(np.abs(result.fun)))
        if not result.success or residual_max > residual_tolerance:
            raise RuntimeError(
                "frozen air5 HBL Ev predictor failed: "
                f"success={result.success} residual={residual_max:.3e} "
                f"message={result.message}"
            )
        predicted = template.copy()
        predicted[unknown] = result.x * scale
        return predicted, result.nfev

    def _predict_frozen_tv(
        self,
        streamfunction: np.ndarray,
        temperature: np.ndarray,
        initial_tv: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        *,
        residual_tolerance: float = 1.0e-10,
        max_function_evaluations: int = 500,
    ) -> np.ndarray:
        predicted, _ = self._predict_frozen_tv_with_evaluations(
            streamfunction,
            temperature,
            initial_tv,
            x,
            y,
            residual_tolerance=residual_tolerance,
            max_function_evaluations=max_function_evaluations,
        )
        return predicted

    def solve_frozen_vt_continuation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        initial_streamfunction: np.ndarray,
        initial_temperature: np.ndarray,
        initial_tv: np.ndarray,
        residual_tolerance: float = 1.0e-10,
        predictor_max_function_evaluations: int = 500,
        source_free_max_function_evaluations: int = 5000,
        source_stage_max_function_evaluations: int = 2000,
    ) -> Air5HblCollocationResult:
        """Solve V-T HBL through an Ev predictor and source-strength homotopy."""
        predicted_tv, predictor_evaluations = (
            self._predict_frozen_tv_with_evaluations(
                initial_streamfunction,
                initial_temperature,
                initial_tv,
                x,
                y,
                residual_tolerance=residual_tolerance,
                max_function_evaluations=predictor_max_function_evaluations,
            )
        )
        streamfunction = np.asarray(initial_streamfunction, dtype=np.float64)
        temperature = np.asarray(initial_temperature, dtype=np.float64)
        tv = predicted_tv
        total_evaluations = predictor_evaluations
        final_result = None
        for source_multiplier in (0.0, 0.1, 0.3, 0.6, 1.0):
            maximum = (
                source_free_max_function_evaluations
                if source_multiplier == 0.0
                else source_stage_max_function_evaluations
            )
            final_result = self.solve_frozen_vt(
                x,
                y,
                initial_streamfunction=streamfunction,
                initial_temperature=temperature,
                initial_tv=tv,
                residual_tolerance=residual_tolerance,
                max_function_evaluations=maximum,
                source_multiplier=source_multiplier,
            )
            total_evaluations += final_result.function_evaluations
            streamfunction = final_result.mass_streamfunction
            temperature = final_result.primitive[:, :, 2]
            tv = final_result.primitive[:, :, 3]
        assert final_result is not None
        return Air5HblCollocationResult(
            x=final_result.x,
            y=final_result.y,
            mass_streamfunction=final_result.mass_streamfunction,
            primitive=final_result.primitive,
            scaled_residual_max=final_result.scaled_residual_max,
            function_evaluations=total_evaluations,
        )

    def solve_source_off_single_temperature(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        initial_streamfunction: np.ndarray,
        initial_temperature: np.ndarray,
        residual_tolerance: float = 1.0e-10,
        max_function_evaluations: int = 500,
    ) -> Air5HblCollocationResult:
        x, y = self._validate_grid(x, y)
        initial_streamfunction = np.asarray(
            initial_streamfunction, dtype=np.float64
        )
        initial_temperature = np.asarray(initial_temperature, dtype=np.float64)
        shape = (x.size, y.size)
        if initial_streamfunction.shape != shape or initial_temperature.shape != shape:
            raise ValueError("air5 HBL collocation initial field has an invalid shape")
        inlet_streamfunction = initial_streamfunction[0].copy()
        inlet_temperature = initial_temperature[0].copy()
        edge_density = self.boundary.pressure / (
            self._gas_constant * self.boundary.edge_temperature
        )
        variable_scale = np.array(
            [
                max(
                    np.max(np.abs(initial_streamfunction)),
                    edge_density * self.boundary.edge_velocity * y[-1],
                ),
                self.boundary.edge_temperature,
            ]
        )
        initial = np.stack(
            (initial_streamfunction, initial_temperature), axis=-1
        )
        initial[1:, 0, 0] = 0.0
        initial[1:, 0, 1] = self.boundary.wall_temperature
        initial[1:, -1, 1] = self.boundary.edge_temperature
        unknown = np.ones(shape + (2,), dtype=bool)
        unknown[0, :, :] = False
        unknown[1:, 0, :] = False
        unknown[1:, -1, 1] = False
        normalized_template = initial / variable_scale

        def residual(packed: np.ndarray) -> np.ndarray:
            normalized = normalized_template.copy()
            normalized[unknown] = packed
            fields = normalized * variable_scale
            full_residual = self.source_off_single_temperature_residual(
                fields[:, :, 0],
                fields[:, :, 1],
                x,
                y,
                inlet_streamfunction=inlet_streamfunction,
                inlet_temperature=inlet_temperature,
            )
            return full_residual[unknown]

        normalized_initial = normalized_template[unknown]
        initial_residual = residual(normalized_initial)
        if np.max(np.abs(initial_residual)) <= residual_tolerance:
            solved = initial
            residual_max = float(np.max(np.abs(initial_residual)))
            function_evaluations = 1
        else:
            full_lower = np.tile(
                np.array(
                    [-np.inf, self.reference.chemistry.temperature_bounds[0]]
                )
                / variable_scale,
                x.size * y.size,
            ).reshape(shape + (2,))
            full_upper = np.tile(
                np.array(
                    [np.inf, self.reference.chemistry.temperature_bounds[1]]
                )
                / variable_scale,
                x.size * y.size,
            ).reshape(shape + (2,))
            lower = full_lower[unknown]
            upper = full_upper[unknown]
            linear_solver = {}
            if normalized_initial.size <= 128:
                linear_solver["tr_solver"] = "exact"
            else:
                full_sparsity = self._jacobian_sparsity(x.size, y.size)
                active = unknown.ravel()
                linear_solver["jac_sparsity"] = full_sparsity[active][:, active]
            result = least_squares(
                residual,
                normalized_initial,
                bounds=(lower, upper),
                x_scale="jac",
                xtol=1.0e-12,
                ftol=1.0e-12,
                gtol=1.0e-12,
                max_nfev=max_function_evaluations,
                **linear_solver,
            )
            residual_max = float(np.max(np.abs(result.fun)))
            if not result.success or residual_max > residual_tolerance:
                raise RuntimeError(
                    "global source-off air5 HBL collocation failed: "
                    f"success={result.success} residual={residual_max:.3e} "
                    f"message={result.message}"
                )
            normalized_solution = normalized_template.copy()
            normalized_solution[unknown] = result.x
            solved = normalized_solution * variable_scale
            function_evaluations = result.nfev

        derivative_x = _derivative_matrix(x, stencil_width=3)
        derivative_y = _derivative_matrix(y, stencil_width=5)
        primitive = self._primitive(
            solved[:, :, 0], solved[:, :, 1], derivative_x, derivative_y
        )
        return Air5HblCollocationResult(
            x=x.copy(),
            y=y.copy(),
            mass_streamfunction=solved[:, :, 0].copy(),
            primitive=primitive,
            scaled_residual_max=residual_max,
            function_evaluations=function_evaluations,
        )
