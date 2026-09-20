"""FP64 frozen-air5 compressible flat-plate similarity reference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_bvp
from scipy.interpolate import PchipInterpolator

try:
    from tests.gpu_validation.air5_hbl_reference import Air5BoundaryLayerReference
except ModuleNotFoundError:
    from air5_hbl_reference import Air5BoundaryLayerReference


@dataclass(frozen=True)
class Air5FrozenSimilarityProfile:
    y: np.ndarray
    primitive: np.ndarray
    mass_streamfunction: np.ndarray


@dataclass(frozen=True)
class Air5FrozenSimilarityWall:
    skin_friction: float
    shear_stress: float
    conductive_energy_flux: float


@dataclass(frozen=True)
class Air5FrozenSimilarityDiagnostics:
    status: int
    iterations: int
    nodes: int
    maximum_boundary_residual: float
    maximum_rms_residual: float


class Air5FrozenSimilarity:
    """Dorodnitsyn similarity solution for the A1-R1 frozen-air5 limit."""

    def __init__(
        self,
        mechanism_path: Path,
        *,
        pressure: float,
        edge_velocity: float,
        edge_temperature: float,
        wall_temperature: float,
        mass_fraction: np.ndarray,
        eta_max: float = 16.0,
        initial_points: int = 241,
        property_table_points: int = 1025,
        tolerance: float = 1.0e-8,
    ):
        self.reference = Air5BoundaryLayerReference(mechanism_path)
        self.pressure = float(pressure)
        self.edge_velocity = float(edge_velocity)
        self.edge_temperature = float(edge_temperature)
        self.wall_temperature = float(wall_temperature)
        self.mass_fraction = np.asarray(mass_fraction, dtype=np.float64).copy()
        self.eta_max = float(eta_max)
        values = np.r_[
            self.pressure,
            self.edge_velocity,
            self.edge_temperature,
            self.wall_temperature,
            self.mass_fraction,
            self.eta_max,
            tolerance,
        ]
        if (
            self.mass_fraction.shape != (5,)
            or not np.all(np.isfinite(values))
            or self.pressure <= 0.0
            or self.edge_velocity <= 0.0
            or self.edge_temperature <= 0.0
            or self.wall_temperature <= 0.0
            or self.eta_max <= 0.0
            or tolerance <= 0.0
            or np.any(self.mass_fraction < 0.0)
            or not np.isclose(np.sum(self.mass_fraction), 1.0, atol=2.0e-14)
            or initial_points < 21
            or property_table_points < 65
        ):
            raise ValueError("invalid frozen-air5 similarity contract")

        chemistry = self.reference.chemistry
        gas_constant = float(np.dot(self.mass_fraction, chemistry.gas_constant))
        temperature_ratio_wall = self.wall_temperature / self.edge_temperature
        temperature_ratio_min = min(1.0, temperature_ratio_wall)
        temperature_ratio_max = max(1.0, temperature_ratio_wall)
        ratio_grid = np.linspace(
            temperature_ratio_min,
            temperature_ratio_max,
            property_table_points,
        )
        properties = np.empty((property_table_points, 4), dtype=np.float64)
        for index, ratio in enumerate(ratio_grid):
            temperature = ratio * self.edge_temperature
            viscosity, conductivity_tr, conductivity_v, _ = (
                self.reference.transport.transport_properties(
                    temperature,
                    temperature,
                    self.pressure,
                    self.mass_fraction,
                )
            )
            vibrational_cv = self.reference.transport.species_vibrational_cv(
                temperature
            )
            specific_heat = float(
                np.dot(
                    self.mass_fraction,
                    chemistry.cv_tr + chemistry.gas_constant + vibrational_cv,
                )
            )
            density = self.pressure / (gas_constant * temperature)
            properties[index] = (
                density,
                viscosity,
                conductivity_tr + conductivity_v,
                specific_heat,
            )

        edge_index = int(np.argmin(np.abs(ratio_grid - 1.0)))
        self.edge_density = float(properties[edge_index, 0])
        self.edge_viscosity = float(properties[edge_index, 1])
        self.edge_specific_heat = float(properties[edge_index, 3])
        chapman_rubesin = (
            properties[:, 0]
            * properties[:, 1]
            / (self.edge_density * self.edge_viscosity)
        )
        conductivity_coefficient = (
            properties[:, 0]
            * properties[:, 2]
            / (
                self.edge_density
                * self.edge_viscosity
                * self.edge_specific_heat
            )
        )
        self._log_c = PchipInterpolator(ratio_grid, np.log(chapman_rubesin))
        self._log_k = PchipInterpolator(
            ratio_grid, np.log(conductivity_coefficient)
        )
        self._cp_ratio = PchipInterpolator(
            ratio_grid, properties[:, 3] / self.edge_specific_heat
        )
        self._log_c_derivative = self._log_c.derivative()
        self._log_k_derivative = self._log_k.derivative()
        kinetic_to_thermal = self.edge_velocity**2 / (
            self.edge_specific_heat * self.edge_temperature
        )

        eta = np.linspace(0.0, self.eta_max, initial_points)
        velocity = 1.0 - np.exp(-eta)
        temperature_ratio = 1.0 + (temperature_ratio_wall - 1.0) * np.exp(
            -eta
        )
        initial = np.vstack(
            (
                eta - 1.0 + np.exp(-eta),
                velocity,
                np.exp(-eta),
                temperature_ratio,
                -(temperature_ratio_wall - 1.0) * np.exp(-eta),
            )
        )

        def rhs(_: np.ndarray, state: np.ndarray) -> np.ndarray:
            stream_function, velocity_ratio, velocity_gradient, ratio, ratio_gradient = (
                state
            )
            coefficient_c = np.exp(self._log_c(ratio))
            coefficient_k = np.exp(self._log_k(ratio))
            return np.vstack(
                (
                    velocity_ratio,
                    velocity_gradient,
                    -self._log_c_derivative(ratio)
                    * ratio_gradient
                    * velocity_gradient
                    - stream_function * velocity_gradient / coefficient_c,
                    ratio_gradient,
                    -self._log_k_derivative(ratio) * ratio_gradient**2
                    - stream_function
                    * self._cp_ratio(ratio)
                    * ratio_gradient
                    / coefficient_k
                    - kinetic_to_thermal
                    * coefficient_c
                    * velocity_gradient**2
                    / coefficient_k,
                )
            )

        def boundary(left: np.ndarray, right: np.ndarray) -> np.ndarray:
            return np.array(
                (
                    left[0],
                    left[1],
                    left[3] - temperature_ratio_wall,
                    right[1] - 1.0,
                    right[3] - 1.0,
                )
            )

        solution = solve_bvp(
            rhs,
            boundary,
            eta,
            initial,
            tol=tolerance,
            max_nodes=100000,
        )
        if solution.status != 0:
            raise RuntimeError(
                f"frozen-air5 similarity solve failed: {solution.message}"
            )
        boundary_residual = boundary(solution.y[:, 0], solution.y[:, -1])
        if np.max(np.abs(boundary_residual)) > 10.0 * tolerance:
            raise RuntimeError("frozen-air5 similarity boundary residual is too large")
        if (
            np.min(solution.y[1]) < -10.0 * tolerance
            or np.max(solution.y[1]) > 1.0 + 10.0 * tolerance
            or np.min(solution.y[3]) <= 0.0
        ):
            raise RuntimeError("frozen-air5 similarity solution is nonphysical")
        self._solution = solution
        self.diagnostics = Air5FrozenSimilarityDiagnostics(
            status=int(solution.status),
            iterations=int(solution.niter),
            nodes=int(solution.x.size),
            maximum_boundary_residual=float(
                np.max(np.abs(boundary_residual))
            ),
            maximum_rms_residual=float(np.max(solution.rms_residuals)),
        )

    def profile(self, streamwise_coordinate: float, y: np.ndarray) -> Air5FrozenSimilarityProfile:
        streamwise_coordinate = float(streamwise_coordinate)
        y = np.asarray(y, dtype=np.float64)
        if (
            not np.isfinite(streamwise_coordinate)
            or streamwise_coordinate <= 0.0
            or y.ndim != 1
            or y.size < 3
            or not np.all(np.isfinite(y))
            or y[0] != 0.0
            or np.any(np.diff(y) <= 0.0)
        ):
            raise ValueError("invalid frozen-air5 similarity profile coordinates")
        eta_table = np.linspace(0.0, self.eta_max, 4097)
        state_table = self._solution.sol(eta_table)
        density_ratio = state_table[3]
        transformed_height = cumulative_trapezoid(
            density_ratio, eta_table, initial=0.0
        )
        scale = np.sqrt(
            2.0
            * (self.edge_viscosity / self.edge_density)
            * streamwise_coordinate
            / self.edge_velocity
        )
        transformed_target = y / scale
        eta = np.interp(
            transformed_target,
            transformed_height,
            eta_table,
        )
        outside = transformed_target > transformed_height[-1]
        eta[outside] = self.eta_max + (
            transformed_target[outside] - transformed_height[-1]
        )
        evaluation_eta = np.minimum(eta, self.eta_max)
        stream_function, velocity_ratio, _, temperature_ratio, _ = (
            self._solution.sol(evaluation_eta)
        )
        local_transformed_height = np.interp(
            evaluation_eta, eta_table, transformed_height
        )
        if np.any(outside):
            extension = eta[outside] - self.eta_max
            stream_function[outside] += extension
            velocity_ratio[outside] = 1.0
            temperature_ratio[outside] = 1.0
            local_transformed_height[outside] = transformed_target[outside]
        normal_velocity = np.sqrt(
            (self.edge_viscosity / self.edge_density)
            * self.edge_velocity
            / (2.0 * streamwise_coordinate)
        ) * (
            velocity_ratio * local_transformed_height
            - temperature_ratio * stream_function
        )
        primitive = np.empty((y.size, 8), dtype=np.float64)
        primitive[:, 0] = self.edge_velocity * velocity_ratio
        primitive[:, 1] = normal_velocity
        primitive[:, 2] = self.edge_temperature * temperature_ratio
        primitive[:, 3] = primitive[:, 2]
        primitive[:, 4:8] = self.mass_fraction[None, 1:5]
        primitive[0, 0:2] = 0.0
        primitive[0, 2:4] = self.wall_temperature
        mass_streamfunction = (
            self.edge_density
            * self.edge_velocity
            * scale
            * stream_function
        )
        mass_streamfunction[0] = 0.0
        return Air5FrozenSimilarityProfile(
            y=y.copy(),
            primitive=primitive,
            mass_streamfunction=mass_streamfunction,
        )

    def wall_quantities(self, streamwise_coordinate: float) -> Air5FrozenSimilarityWall:
        if not np.isfinite(streamwise_coordinate) or streamwise_coordinate <= 0.0:
            raise ValueError("streamwise coordinate must be positive and finite")
        wall = self._solution.sol(0.0)
        wall_ratio = self.wall_temperature / self.edge_temperature
        scale = np.sqrt(
            2.0
            * (self.edge_viscosity / self.edge_density)
            * streamwise_coordinate
            / self.edge_velocity
        )
        wall_viscosity, conductivity_tr, conductivity_v, _ = (
            self.reference.transport.transport_properties(
                self.wall_temperature,
                self.wall_temperature,
                self.pressure,
                self.mass_fraction,
            )
        )
        velocity_gradient = self.edge_velocity * wall[2] / (scale * wall_ratio)
        temperature_gradient = (
            self.edge_temperature * wall[4] / (scale * wall_ratio)
        )
        shear_stress = wall_viscosity * velocity_gradient
        skin_friction = 2.0 * shear_stress / (
            self.edge_density * self.edge_velocity**2
        )
        conductive_energy_flux = (
            conductivity_tr + conductivity_v
        ) * temperature_gradient
        return Air5FrozenSimilarityWall(
            skin_friction=float(skin_friction),
            shear_stress=float(shear_stress),
            conductive_energy_flux=float(conductive_energy_flux),
        )
