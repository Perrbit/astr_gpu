"""Rejected station-marcher candidates retained for A1-R1 diagnostics only.

The authoritative frozen-air5 A1-R1 reference is ``air5_hbl_similarity.py``.
Neither station formulation in this module qualifies R2/R3 validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

try:
    from tests.gpu_validation.air5_hbl_reference import (
        Air5BoundaryLayerReference,
        Air5BoundaryLayerState,
        NUM_PARABOLIC_EQUATIONS,
    )
except ModuleNotFoundError:
    from air5_hbl_reference import (
        Air5BoundaryLayerReference,
        Air5BoundaryLayerState,
        NUM_PARABOLIC_EQUATIONS,
    )


NUM_PRIMITIVES = 8
DYNAMIC_COLUMNS = np.array([0, 2, 3, 4, 5, 6, 7], dtype=int)


@dataclass(frozen=True)
class Air5HblBoundaryConditions:
    pressure: float
    edge_velocity: np.ndarray
    edge_temperature: float
    edge_tv: float
    edge_mass_fraction: np.ndarray
    wall_velocity: float
    wall_temperature: float
    wall_tv: float


@dataclass(frozen=True)
class Air5HblStationResult:
    primitive: np.ndarray
    scaled_residual_max: float
    function_evaluations: int


@dataclass(frozen=True)
class Air5HblStreamfunctionStationResult:
    mass_streamfunction: np.ndarray
    primitive: np.ndarray
    scaled_residual_max: float
    function_evaluations: int


@dataclass(frozen=True)
class Air5HblDerivativeResult:
    dynamic_derivative: np.ndarray
    normal_velocity: np.ndarray
    scaled_linear_residual_max: float


@dataclass(frozen=True)
class Air5HblMarchResult:
    x: np.ndarray
    primitive: np.ndarray
    scaled_linear_residual_max: float


def _derivative_matrix(coordinate: np.ndarray) -> np.ndarray:
    coordinate = np.asarray(coordinate, dtype=np.float64)
    if (
        coordinate.ndim != 1
        or coordinate.size < 3
        or not np.all(np.isfinite(coordinate))
        or np.any(np.diff(coordinate) <= 0.0)
    ):
        raise ValueError("HBL y coordinate must be finite and strictly increasing")
    matrix = np.zeros((coordinate.size, coordinate.size), dtype=np.float64)
    for index in range(coordinate.size):
        start = min(max(index - 1, 0), coordinate.size - 3)
        selection = np.arange(start, start + 3)
        offset = coordinate[selection] - coordinate[index]
        vandermonde = np.vstack([offset**power for power in range(3)])
        right_hand_side = np.array([0.0, 1.0, 0.0])
        matrix[index, selection] = np.linalg.solve(vandermonde, right_hand_side)
        matrix[index, index] -= np.sum(matrix[index])
    return matrix


def _differentiate(matrix: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Apply a first derivative through differences so constants are exact."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        differences = values[None, :] - values[:, None]
        return np.einsum("ij,ij->i", matrix, differences)
    differences = values[None, :, :] - values[:, None, :]
    return np.einsum("ij,ijk->ik", matrix, differences)


class Air5HblMarcher:
    """Diagnostic backward-Euler candidate, not an authoritative reference."""

    def __init__(self, mechanism_path: Path):
        self.reference = Air5BoundaryLayerReference(mechanism_path)

    @staticmethod
    def _validate_boundary(boundary: Air5HblBoundaryConditions) -> None:
        edge_velocity = np.asarray(boundary.edge_velocity, dtype=np.float64)
        edge_mass_fraction = np.asarray(
            boundary.edge_mass_fraction, dtype=np.float64
        )
        if edge_velocity.shape != (3,) or edge_mass_fraction.shape != (5,):
            raise ValueError("invalid HBL boundary state shape")
        values = np.r_[
            boundary.pressure,
            edge_velocity,
            boundary.edge_temperature,
            boundary.edge_tv,
            edge_mass_fraction,
            boundary.wall_velocity,
            boundary.wall_temperature,
            boundary.wall_tv,
        ]
        if not np.all(np.isfinite(values)):
            raise ValueError("HBL boundary state contains non-finite values")
        if (
            boundary.pressure <= 0.0
            or boundary.edge_temperature <= 0.0
            or boundary.edge_tv <= 0.0
            or boundary.wall_temperature <= 0.0
            or boundary.wall_tv <= 0.0
            or edge_velocity[0] <= 0.0
            or np.any(edge_mass_fraction < 0.0)
            or not np.isclose(
                np.sum(edge_mass_fraction), 1.0, atol=2.0e-14, rtol=0.0
            )
        ):
            raise ValueError("HBL boundary state is outside the physical domain")

    @staticmethod
    def _mass_fraction(primitive: np.ndarray) -> np.ndarray:
        independent = primitive[:, 4:8]
        closure = 1.0 - np.sum(independent, axis=1)
        return np.column_stack((closure, independent))

    def _states(
        self, primitive: np.ndarray, pressure: float
    ) -> tuple[Air5BoundaryLayerState, ...]:
        mass_fraction = self._mass_fraction(primitive)
        if np.any(mass_fraction < 0.0) or np.any(mass_fraction > 1.0):
            raise ValueError("HBL primitive profile contains invalid mass fractions")
        return tuple(
            Air5BoundaryLayerState(
                pressure=pressure,
                velocity=np.array([row[0], row[1], 0.0]),
                temperature=row[2],
                tv=row[3],
                mass_fraction=mass_fraction[index],
            )
            for index, row in enumerate(primitive)
        )

    def _flux_scale(
        self, boundary: Air5HblBoundaryConditions, streamwise_step: float
    ) -> np.ndarray:
        edge = Air5BoundaryLayerState(
            pressure=boundary.pressure,
            velocity=np.asarray(boundary.edge_velocity, dtype=np.float64),
            temperature=boundary.edge_temperature,
            tv=boundary.edge_tv,
            mass_fraction=np.asarray(boundary.edge_mass_fraction, dtype=np.float64),
        )
        flux = np.abs(self.reference.streamwise_flux(edge))
        mass_flux = max(flux[0], np.finfo(np.float64).tiny)
        flux[2:6] = mass_flux
        flux[7] = max(flux[7], 1.0e-8 * flux[6])
        return np.maximum(flux / streamwise_step, np.finfo(np.float64).tiny)

    @staticmethod
    def _dynamic_scale(
        primitive: np.ndarray, boundary: Air5HblBoundaryConditions
    ) -> np.ndarray:
        return np.array(
            [
                boundary.edge_velocity[0],
                boundary.edge_temperature,
                boundary.edge_tv,
                *[
                    max(np.max(np.abs(primitive[:, column])), 1.0e-14)
                    for column in range(4, 8)
                ],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _primitive_from_dynamic(
        dynamic: np.ndarray, normal_velocity: np.ndarray
    ) -> np.ndarray:
        dynamic = np.asarray(dynamic, dtype=np.float64)
        normal_velocity = np.asarray(normal_velocity, dtype=np.float64)
        if dynamic.ndim != 2 or dynamic.shape[1] != 7:
            raise ValueError("HBL dynamic profile must have seven columns")
        if normal_velocity.shape != (dynamic.shape[0],):
            raise ValueError("HBL normal velocity has an invalid shape")
        primitive = np.empty((dynamic.shape[0], NUM_PRIMITIVES), dtype=np.float64)
        primitive[:, DYNAMIC_COLUMNS] = dynamic
        primitive[:, 1] = normal_velocity
        return primitive

    def _streamwise_flux_jacobian(
        self,
        dynamic: np.ndarray,
        pressure: float,
        variable_scale: np.ndarray,
    ) -> np.ndarray:
        """Analytic local x-flux Jacobian in scaled primitive coordinates."""
        u, temperature, tv = dynamic[:3]
        mass_fraction = np.r_[1.0 - np.sum(dynamic[3:7]), dynamic[3:7]]
        chemistry = self.reference.chemistry
        gas_constant = float(np.dot(mass_fraction, chemistry.gas_constant))
        density = pressure / (gas_constant * temperature)
        cv_tr = float(np.dot(mass_fraction, chemistry.cv_tr))
        formation = float(np.dot(mass_fraction, chemistry.formation_energy))
        species_ev = self.reference.transport.species_vibrational_energy(tv)
        species_cv_v = self.reference.transport.species_vibrational_cv(tv)
        ev = float(np.dot(mass_fraction, species_ev))
        cv_v = float(np.dot(mass_fraction, species_cv_v))
        specific_energy = cv_tr * temperature + formation + ev + 0.5 * u**2
        energy_density = density * specific_energy

        physical = np.zeros((NUM_PARABOLIC_EQUATIONS, 7), dtype=np.float64)
        physical[:, 0] = np.r_[
            density,
            2.0 * density * u,
            density * mass_fraction[1:5],
            energy_density + pressure + density * u**2,
            density * ev,
        ]

        density_t = -density / temperature
        energy_density_t = density_t * specific_energy + density * cv_tr
        physical[:, 1] = np.r_[
            density_t * u,
            density_t * u**2,
            density_t * u * mass_fraction[1:5],
            u * energy_density_t,
            density_t * u * ev,
        ]
        physical[6, 2] = u * density * cv_v
        physical[7, 2] = u * density * cv_v

        for independent, species in enumerate(range(1, 5)):
            column = 3 + independent
            gas_constant_y = (
                chemistry.gas_constant[species] - chemistry.gas_constant[0]
            )
            density_y = -density * gas_constant_y / gas_constant
            cv_y = chemistry.cv_tr[species] - chemistry.cv_tr[0]
            formation_y = (
                chemistry.formation_energy[species]
                - chemistry.formation_energy[0]
            )
            ev_y = species_ev[species] - species_ev[0]
            specific_energy_y = cv_y * temperature + formation_y + ev_y
            energy_density_y = (
                density_y * specific_energy + density * specific_energy_y
            )
            species_flux_y = density_y * u * mass_fraction[1:5]
            species_flux_y[independent] += density * u
            physical[:, column] = np.r_[
                density_y * u,
                density_y * u**2,
                species_flux_y,
                u * energy_density_y,
                u * (density_y * ev + density * ev_y),
            ]
        return physical * variable_scale[None, :]

    def streamwise_derivative(
        self,
        primitive: np.ndarray,
        y: np.ndarray,
        boundary: Air5HblBoundaryConditions,
        *,
        source_mode: str = "off",
        single_temperature: bool = False,
        frozen_species: bool = False,
    ) -> Air5HblDerivativeResult:
        """Solve the parabolic DAE for primitive x derivatives and normal speed."""
        self._validate_boundary(boundary)
        primitive = np.asarray(primitive, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if primitive.shape != (y.size, NUM_PRIMITIVES):
            raise ValueError("HBL primitive profile has an invalid shape")
        if source_mode not in {"off", "vt", "chemical", "coupled"}:
            raise ValueError("unsupported HBL source mode")
        mass_fraction = self._mass_fraction(primitive)
        if (
            not np.all(np.isfinite(primitive))
            or np.any(primitive[:, 0] < 0.0)
            or np.any(primitive[:, 2:4] <= 0.0)
            or np.any(mass_fraction < 0.0)
        ):
            raise ValueError("HBL primitive profile is outside the physical domain")

        derivative = _derivative_matrix(y)
        dynamic = primitive[:, DYNAMIC_COLUMNS]
        dynamic_scale = self._dynamic_scale(primitive, boundary)
        u_scale = boundary.edge_velocity[0]

        velocity_gradient = _differentiate(derivative, primitive[:, 0])
        temperature_gradient = _differentiate(derivative, primitive[:, 2])
        tv_gradient = _differentiate(derivative, primitive[:, 3])
        independent_gradient = _differentiate(derivative, primitive[:, 4:8])
        species_gradient = np.column_stack(
            (-np.sum(independent_gradient, axis=1), independent_gradient)
        )

        states: list[Air5BoundaryLayerState] = []
        diffusive_flux = np.empty((y.size, NUM_PARABOLIC_EQUATIONS))
        normal_coefficient = np.empty_like(diffusive_flux)
        source = np.zeros_like(diffusive_flux)
        jacobian = np.empty((y.size, NUM_PARABOLIC_EQUATIONS, 7))
        for index in range(y.size):
            state = Air5BoundaryLayerState(
                pressure=boundary.pressure,
                velocity=np.array([primitive[index, 0], 0.0, 0.0]),
                temperature=primitive[index, 2],
                tv=primitive[index, 3],
                mass_fraction=mass_fraction[index],
            )
            states.append(state)
            diffusive_flux[index] = self.reference.wall_normal_flux(
                state,
                grad_velocity_y=np.array([velocity_gradient[index], 0.0, 0.0]),
                grad_temperature_y=temperature_gradient[index],
                grad_tv_y=tv_gradient[index],
                grad_mass_fraction_y=species_gradient[index],
            )
            q = self.reference.conservative_state(state)
            normal_coefficient[index] = np.r_[
                q[0],
                q[1],
                q[6:10],
                q[4] + boundary.pressure,
                q[10],
            ]
            jacobian[index] = self._streamwise_flux_jacobian(
                dynamic[index], boundary.pressure, dynamic_scale
            )
            if source_mode != "off":
                source[index] = self.reference.source(state, source_mode)

        diffusive_divergence = _differentiate(derivative, diffusive_flux)
        rhs = source - diffusive_divergence
        edge_state = states[-1]
        row_scale = np.abs(self.reference.streamwise_flux(edge_state))
        row_scale[2:6] = max(row_scale[0], np.finfo(np.float64).tiny)
        row_scale[7] = max(row_scale[7], 1.0e-8 * row_scale[6])
        row_scale = np.maximum(row_scale, np.finfo(np.float64).tiny)

        points = y.size
        size = points * NUM_PARABOLIC_EQUATIONS
        matrix = lil_matrix((size, size), dtype=np.float64)
        right_hand_side = np.zeros(size, dtype=np.float64)
        for station in range(points):
            row = station * NUM_PARABOLIC_EQUATIONS
            matrix[row : row + 8, row : row + 7] = (
                jacobian[station] / row_scale[:, None]
            )
            for normal_station in np.flatnonzero(derivative[station]):
                normal_column = normal_station * NUM_PARABOLIC_EQUATIONS + 7
                matrix[row : row + 8, normal_column] = (
                    derivative[station, normal_station]
                    * normal_coefficient[normal_station]
                    * u_scale
                    / row_scale
                )[:, None]
            right_hand_side[row : row + 8] = rhs[station] / row_scale

        # Wall: v=0; fixed u/T/Tv; noncatalytic species remain equal to row 1.
        matrix[0:8, :] = 0.0
        matrix[0, 7] = 1.0
        matrix[1, 0] = 1.0
        matrix[2, 1] = 1.0
        matrix[3, 2] = 1.0
        for species in range(4):
            matrix[4 + species, 3 + species] = 1.0
            matrix[4 + species, 8 + 3 + species] = -1.0
        right_hand_side[0:8] = 0.0

        # Edge: retain mass continuity for v; hold all seven dynamic variables.
        edge_row = (points - 1) * 8
        for equation in range(1, 8):
            matrix[edge_row + equation, :] = 0.0
            matrix[edge_row + equation, edge_row + equation - 1] = 1.0
            right_hand_side[edge_row + equation] = 0.0

        if single_temperature:
            ratio = dynamic_scale[2] / dynamic_scale[1]
            for station in range(1, points - 1):
                row = station * 8 + 7
                matrix[row, :] = 0.0
                matrix[row, station * 8 + 1] = -1.0
                matrix[row, station * 8 + 2] = ratio
                right_hand_side[row] = 0.0

        if frozen_species:
            for station in range(1, points - 1):
                for species in range(4):
                    row = station * 8 + 2 + species
                    matrix[row, :] = 0.0
                    matrix[row, station * 8 + 3 + species] = 1.0
                    right_hand_side[row] = 0.0

        system = matrix.tocsr()
        solution = spsolve(system, right_hand_side)
        if not np.all(np.isfinite(solution)):
            raise RuntimeError("HBL parabolic derivative solve is singular")
        linear_residual = system @ solution - right_hand_side
        packed = solution.reshape((points, 8))
        dynamic_derivative = packed[:, :7] * dynamic_scale[None, :]
        normal_velocity = packed[:, 7] * u_scale
        return Air5HblDerivativeResult(
            dynamic_derivative=dynamic_derivative,
            normal_velocity=normal_velocity,
            scaled_linear_residual_max=float(np.max(np.abs(linear_residual))),
        )

    @staticmethod
    def _apply_boundary_dynamic(
        dynamic: np.ndarray,
        boundary: Air5HblBoundaryConditions,
        *,
        single_temperature: bool,
    ) -> None:
        dynamic[0, 0] = boundary.wall_velocity
        dynamic[0, 1] = boundary.wall_temperature
        dynamic[0, 2] = boundary.wall_tv
        dynamic[0, 3:7] = dynamic[1, 3:7]
        dynamic[-1] = np.r_[
            boundary.edge_velocity[0],
            boundary.edge_temperature,
            boundary.edge_tv,
            boundary.edge_mass_fraction[1:5],
        ]
        if single_temperature:
            dynamic[:, 2] = dynamic[:, 1]

    def march(
        self,
        initial: np.ndarray,
        y: np.ndarray,
        x: np.ndarray,
        boundary: Air5HblBoundaryConditions,
        *,
        source_mode: str = "off",
        single_temperature: bool = False,
        frozen_species: bool = False,
    ) -> Air5HblMarchResult:
        """March a profile with explicit midpoint integration in x."""
        initial = np.asarray(initial, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        if initial.shape != (y.size, NUM_PRIMITIVES):
            raise ValueError("HBL initial primitive profile has an invalid shape")
        if (
            x.ndim != 1
            or x.size < 2
            or not np.all(np.isfinite(x))
            or np.any(np.diff(x) <= 0.0)
        ):
            raise ValueError("HBL x coordinate must be finite and strictly increasing")

        history = np.empty((x.size, y.size, NUM_PRIMITIVES), dtype=np.float64)
        dynamic = initial[:, DYNAMIC_COLUMNS].copy()
        self._apply_boundary_dynamic(
            dynamic, boundary, single_temperature=single_temperature
        )
        maximum_linear_residual = 0.0
        normal_velocity = np.zeros(y.size)
        history[0] = self._primitive_from_dynamic(dynamic, normal_velocity)

        for station in range(x.size - 1):
            step = x[station + 1] - x[station]
            current = self._primitive_from_dynamic(dynamic, normal_velocity)
            first = self.streamwise_derivative(
                current,
                y,
                boundary,
                source_mode=source_mode,
                single_temperature=single_temperature,
                frozen_species=frozen_species,
            )
            midpoint_dynamic = dynamic + 0.5 * step * first.dynamic_derivative
            self._apply_boundary_dynamic(
                midpoint_dynamic, boundary, single_temperature=single_temperature
            )
            midpoint = self._primitive_from_dynamic(
                midpoint_dynamic, first.normal_velocity
            )
            middle = self.streamwise_derivative(
                midpoint,
                y,
                boundary,
                source_mode=source_mode,
                single_temperature=single_temperature,
                frozen_species=frozen_species,
            )
            dynamic = dynamic + step * middle.dynamic_derivative
            self._apply_boundary_dynamic(
                dynamic, boundary, single_temperature=single_temperature
            )
            candidate = self._primitive_from_dynamic(dynamic, middle.normal_velocity)
            mass_fraction = self._mass_fraction(candidate)
            if (
                not np.all(np.isfinite(candidate))
                or np.any(candidate[:, 0] < 0.0)
                or np.any(candidate[:, 2:4] <= 0.0)
                or np.any(mass_fraction < 0.0)
            ):
                raise RuntimeError(
                    f"HBL reference left the physical domain at x={x[station + 1]:.16e}"
                )
            final = self.streamwise_derivative(
                candidate,
                y,
                boundary,
                source_mode=source_mode,
                single_temperature=single_temperature,
                frozen_species=frozen_species,
            )
            normal_velocity = final.normal_velocity
            history[station + 1] = self._primitive_from_dynamic(
                dynamic, normal_velocity
            )
            maximum_linear_residual = max(
                maximum_linear_residual,
                first.scaled_linear_residual_max,
                middle.scaled_linear_residual_max,
                final.scaled_linear_residual_max,
            )

        return Air5HblMarchResult(
            x=x.copy(),
            primitive=history,
            scaled_linear_residual_max=maximum_linear_residual,
        )

    def march_radau(
        self,
        initial: np.ndarray,
        y: np.ndarray,
        x: np.ndarray,
        boundary: Air5HblBoundaryConditions,
        *,
        source_mode: str = "off",
        single_temperature: bool = False,
        frozen_species: bool = False,
        relative_tolerance: float = 1.0e-7,
        absolute_tolerance: float = 1.0e-10,
        maximum_step: float | None = None,
    ) -> Air5HblMarchResult:
        """March the stiff parabolic system with an FP64 Radau integrator."""
        initial = np.asarray(initial, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        if initial.shape != (y.size, NUM_PRIMITIVES):
            raise ValueError("HBL initial primitive profile has an invalid shape")
        if (
            x.ndim != 1
            or x.size < 2
            or not np.all(np.isfinite(x))
            or np.any(np.diff(x) <= 0.0)
        ):
            raise ValueError("HBL x coordinate must be finite and strictly increasing")
        if relative_tolerance <= 0.0 or absolute_tolerance <= 0.0:
            raise ValueError("HBL Radau tolerances must be positive")
        if maximum_step is None:
            maximum_step = x[-1] - x[0]
        if maximum_step <= 0.0 or not np.isfinite(maximum_step):
            raise ValueError("HBL Radau maximum step must be positive and finite")

        dynamic = initial[:, DYNAMIC_COLUMNS].copy()
        self._apply_boundary_dynamic(
            dynamic, boundary, single_temperature=single_temperature
        )
        if frozen_species:
            dynamic[:, 3:7] = boundary.edge_mass_fraction[1:5]
        scale = self._dynamic_scale(
            self._primitive_from_dynamic(dynamic, np.zeros(y.size)), boundary
        )
        maximum_linear_residual = 0.0

        def right_hand_side(_: float, normalized_flat: np.ndarray) -> np.ndarray:
            nonlocal maximum_linear_residual
            local_dynamic = normalized_flat.reshape((-1, 7)) * scale
            self._apply_boundary_dynamic(
                local_dynamic, boundary, single_temperature=single_temperature
            )
            if frozen_species:
                local_dynamic[:, 3:7] = boundary.edge_mass_fraction[1:5]
            local_primitive = self._primitive_from_dynamic(
                local_dynamic, np.zeros(y.size)
            )
            derivative = self.streamwise_derivative(
                local_primitive,
                y,
                boundary,
                source_mode=source_mode,
                single_temperature=single_temperature,
                frozen_species=frozen_species,
            )
            maximum_linear_residual = max(
                maximum_linear_residual, derivative.scaled_linear_residual_max
            )
            return (derivative.dynamic_derivative / scale).ravel()

        solution = solve_ivp(
            right_hand_side,
            (x[0], x[-1]),
            (dynamic / scale).ravel(),
            method="Radau",
            t_eval=x,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            max_step=maximum_step,
        )
        if not solution.success:
            raise RuntimeError(f"HBL Radau march failed: {solution.message}")

        history = np.empty((x.size, y.size, NUM_PRIMITIVES), dtype=np.float64)
        for station in range(x.size):
            local_dynamic = solution.y[:, station].reshape((-1, 7)) * scale
            self._apply_boundary_dynamic(
                local_dynamic, boundary, single_temperature=single_temperature
            )
            if frozen_species:
                local_dynamic[:, 3:7] = boundary.edge_mass_fraction[1:5]
            local_primitive = self._primitive_from_dynamic(
                local_dynamic, np.zeros(y.size)
            )
            derivative = self.streamwise_derivative(
                local_primitive,
                y,
                boundary,
                source_mode=source_mode,
                single_temperature=single_temperature,
                frozen_species=frozen_species,
            )
            local_primitive[:, 1] = derivative.normal_velocity
            mass_fraction = self._mass_fraction(local_primitive)
            if (
                not np.all(np.isfinite(local_primitive))
                or np.any(local_primitive[:, 0] < 0.0)
                or np.any(local_primitive[:, 2:4] <= 0.0)
                or np.any(mass_fraction < 0.0)
            ):
                raise RuntimeError(
                    f"HBL Radau reference left the physical domain at x={x[station]:.16e}"
                )
            history[station] = local_primitive
            maximum_linear_residual = max(
                maximum_linear_residual, derivative.scaled_linear_residual_max
            )

        return Air5HblMarchResult(
            x=x.copy(),
            primitive=history,
            scaled_linear_residual_max=maximum_linear_residual,
        )

    @staticmethod
    def _jacobian_sparsity(points: int):
        """Return the conservative five-station coupling of the y operator."""
        size = NUM_PRIMITIVES * points
        sparsity = lil_matrix((size, size), dtype=np.int8)
        for station in range(points):
            row = slice(
                station * NUM_PRIMITIVES, (station + 1) * NUM_PRIMITIVES
            )
            if station == 0:
                dependencies = range(0, min(points, 2))
            elif station == points - 1:
                dependencies = (points - 1,)
            else:
                dependencies = range(max(0, station - 2), min(points, station + 3))
            for dependency in dependencies:
                column = slice(
                    dependency * NUM_PRIMITIVES,
                    (dependency + 1) * NUM_PRIMITIVES,
                )
                sparsity[row, column] = 1
        return sparsity.tocsr()

    @staticmethod
    def _three_equation_sparsity(points: int):
        size = 3 * points
        sparsity = lil_matrix((size, size), dtype=np.int8)
        for station in range(points):
            row = slice(3 * station, 3 * station + 3)
            if station == 0:
                dependencies = (0,)
            elif station == points - 1:
                dependencies = range(max(0, station - 2), points)
            else:
                dependencies = range(max(0, station - 2), min(points, station + 3))
            for dependency in dependencies:
                sparsity[row, 3 * dependency : 3 * dependency + 3] = 1
        return sparsity.tocsr()

    @staticmethod
    def _streamfunction_sparsity(points: int):
        size = 2 * points
        sparsity = lil_matrix((size, size), dtype=np.int8)
        for station in range(points):
            row = slice(2 * station, 2 * station + 2)
            for dependency in range(
                max(0, station - 3), min(points, station + 4)
            ):
                sparsity[row, 2 * dependency : 2 * dependency + 2] = 1
        return sparsity.tocsr()

    def _single_temperature_streamfunction_residual(
        self,
        normalized_flat: np.ndarray,
        previous_primitive: np.ndarray,
        previous_streamfunction: np.ndarray,
        y: np.ndarray,
        streamwise_step: float,
        boundary: Air5HblBoundaryConditions,
        variable_scale: np.ndarray,
        derivative: np.ndarray,
        previous_flux: np.ndarray,
        flux_scale: np.ndarray,
    ) -> np.ndarray:
        unknown = normalized_flat.reshape((-1, 2)) * variable_scale
        streamfunction = unknown[:, 0]
        temperature = unknown[:, 1]
        temperature_minimum, temperature_maximum = (
            self.reference.chemistry.temperature_bounds
        )
        if (
            not np.all(np.isfinite(unknown))
            or np.any(temperature < temperature_minimum)
            or np.any(temperature > temperature_maximum)
        ):
            return np.full(2 * y.size, 1.0e6)

        composition = np.asarray(boundary.edge_mass_fraction, dtype=np.float64)
        gas_constant = float(
            np.dot(composition, self.reference.chemistry.gas_constant)
        )
        density = boundary.pressure / (gas_constant * temperature)
        mass_flux = _differentiate(derivative, streamfunction)
        streamwise_velocity = mass_flux / density
        normal_velocity = -(
            streamfunction - previous_streamfunction
        ) / (streamwise_step * density)
        primitive = np.column_stack(
            (
                streamwise_velocity,
                normal_velocity,
                temperature,
                temperature,
                np.tile(composition[1:5, None], (1, y.size)).T,
            )
        )
        velocity_gradient = _differentiate(derivative, primitive[:, 0:2])
        temperature_gradient = _differentiate(derivative, temperature)
        streamwise_flux = np.empty((y.size, 2), dtype=np.float64)
        normal_flux = np.empty_like(streamwise_flux)
        for index in range(y.size):
            state = Air5BoundaryLayerState(
                pressure=boundary.pressure,
                velocity=np.array(
                    [streamwise_velocity[index], normal_velocity[index], 0.0]
                ),
                temperature=temperature[index],
                tv=temperature[index],
                mass_fraction=composition,
            )
            streamwise_flux[index] = self.reference.streamwise_flux(state)[[1, 6]]
            normal_flux[index] = self.reference.wall_normal_flux(
                state,
                grad_velocity_y=np.array(
                    [velocity_gradient[index, 0], velocity_gradient[index, 1], 0.0]
                ),
                grad_temperature_y=temperature_gradient[index],
                grad_tv_y=temperature_gradient[index],
                grad_mass_fraction_y=np.zeros(5),
            )[[1, 6]]

        residual = (
            (streamwise_flux - previous_flux) / streamwise_step
            + _differentiate(derivative, normal_flux)
        ) / flux_scale
        residual[0] = np.array(
            [
                streamfunction[0] / variable_scale[0],
                (temperature[0] - boundary.wall_temperature)
                / boundary.edge_temperature,
            ]
        )
        residual[1, 0] = (
            streamwise_velocity[0] - boundary.wall_velocity
        ) / boundary.edge_velocity[0]
        residual[-1] = np.array(
            [
                (streamwise_velocity[-1] - boundary.edge_velocity[0])
                / boundary.edge_velocity[0],
                (temperature[-1] - boundary.edge_temperature)
                / boundary.edge_temperature,
            ]
        )
        return residual.ravel()

    def march_single_temperature_streamfunction_station(
        self,
        previous_primitive: np.ndarray,
        previous_streamfunction: np.ndarray,
        y: np.ndarray,
        streamwise_step: float,
        boundary: Air5HblBoundaryConditions,
        *,
        initial_primitive: np.ndarray | None = None,
        initial_streamfunction: np.ndarray | None = None,
        residual_tolerance: float = 1.0e-10,
        max_function_evaluations: int = 300,
    ) -> Air5HblStreamfunctionStationResult:
        """Advance the A1-R1 limit with discrete mass continuity built in."""
        self._validate_boundary(boundary)
        previous_primitive = np.asarray(previous_primitive, dtype=np.float64)
        previous_streamfunction = np.asarray(
            previous_streamfunction, dtype=np.float64
        )
        y = np.asarray(y, dtype=np.float64)
        if previous_primitive.shape != (y.size, NUM_PRIMITIVES):
            raise ValueError("HBL previous station has an invalid shape")
        if previous_streamfunction.shape != (y.size,):
            raise ValueError("HBL previous streamfunction has an invalid shape")
        if streamwise_step <= 0.0 or not np.isfinite(streamwise_step):
            raise ValueError("HBL streamwise step must be positive and finite")
        if initial_primitive is None:
            initial_primitive = previous_primitive
        else:
            initial_primitive = np.asarray(initial_primitive, dtype=np.float64)
            if initial_primitive.shape != previous_primitive.shape:
                raise ValueError("HBL streamfunction initial primitive has an invalid shape")
        if initial_streamfunction is None:
            initial_streamfunction = previous_streamfunction
        else:
            initial_streamfunction = np.asarray(
                initial_streamfunction, dtype=np.float64
            )
            if initial_streamfunction.shape != previous_streamfunction.shape:
                raise ValueError("HBL streamfunction initial guess has an invalid shape")

        derivative = _derivative_matrix(y)
        composition = np.asarray(boundary.edge_mass_fraction, dtype=np.float64)
        gas_constant = float(
            np.dot(composition, self.reference.chemistry.gas_constant)
        )
        previous_density = boundary.pressure / (
            gas_constant * previous_primitive[:, 2]
        )
        previous_velocity = _differentiate(
            derivative, previous_streamfunction
        ) / previous_density
        previous_flux = np.empty((y.size, 2), dtype=np.float64)
        for index in range(y.size):
            previous_state = Air5BoundaryLayerState(
                pressure=boundary.pressure,
                velocity=np.array(
                    [
                        previous_velocity[index],
                        previous_primitive[index, 1],
                        0.0,
                    ]
                ),
                temperature=previous_primitive[index, 2],
                tv=previous_primitive[index, 2],
                mass_fraction=composition,
            )
            previous_flux[index] = self.reference.streamwise_flux(previous_state)[
                [1, 6]
            ]
        edge = Air5BoundaryLayerState(
            pressure=boundary.pressure,
            velocity=np.asarray(boundary.edge_velocity, dtype=np.float64),
            temperature=boundary.edge_temperature,
            tv=boundary.edge_temperature,
            mass_fraction=composition,
        )
        flux_scale = np.maximum(
            np.abs(self.reference.streamwise_flux(edge)[[1, 6]])
            / streamwise_step,
            np.finfo(np.float64).tiny,
        )
        variable_scale = np.array(
            [
                max(
                    np.max(np.abs(initial_streamfunction)),
                    self.reference.density(edge)
                    * boundary.edge_velocity[0]
                    * y[-1],
                ),
                boundary.edge_temperature,
            ]
        )
        initial = np.column_stack(
            (initial_streamfunction, initial_primitive[:, 2])
        )
        lower = np.tile(
            np.array([-np.inf, self.reference.chemistry.temperature_bounds[0]])
            / variable_scale,
            y.size,
        )
        upper = np.tile(
            np.array([np.inf, self.reference.chemistry.temperature_bounds[1]])
            / variable_scale,
            y.size,
        )
        result = least_squares(
            lambda trial: self._single_temperature_streamfunction_residual(
                trial,
                previous_primitive,
                previous_streamfunction,
                y,
                streamwise_step,
                boundary,
                variable_scale,
                derivative,
                previous_flux,
                flux_scale,
            ),
            initial.ravel() / np.tile(variable_scale, y.size),
            bounds=(lower, upper),
            jac_sparsity=self._streamfunction_sparsity(y.size),
            x_scale="jac",
            diff_step=1.0e-5,
            xtol=1.0e-12,
            ftol=1.0e-12,
            gtol=1.0e-12,
            max_nfev=max_function_evaluations,
        )
        residual_max = float(np.max(np.abs(result.fun)))
        if not result.success or residual_max > residual_tolerance:
            raise RuntimeError(
                "single-temperature streamfunction HBL station solve failed: "
                f"success={result.success} residual={residual_max:.3e} "
                f"message={result.message}"
            )
        solved = result.x.reshape((-1, 2)) * variable_scale
        streamfunction = solved[:, 0]
        temperature = solved[:, 1]
        density = boundary.pressure / (gas_constant * temperature)
        streamwise_velocity = _differentiate(
            derivative, streamfunction
        ) / density
        normal_velocity = -(
            streamfunction - previous_streamfunction
        ) / (streamwise_step * density)
        primitive = np.column_stack(
            (
                streamwise_velocity,
                normal_velocity,
                temperature,
                temperature,
                np.tile(composition[1:5, None], (1, y.size)).T,
            )
        )
        if not np.all(np.isfinite(primitive)) or np.any(temperature <= 0.0):
            raise RuntimeError(
                "single-temperature streamfunction HBL solve produced a nonphysical state"
            )
        return Air5HblStreamfunctionStationResult(
            mass_streamfunction=streamfunction,
            primitive=primitive,
            scaled_residual_max=residual_max,
            function_evaluations=result.nfev,
        )

    def _single_temperature_frozen_residual(
        self,
        normalized_flat: np.ndarray,
        y: np.ndarray,
        streamwise_step: float,
        boundary: Air5HblBoundaryConditions,
        variable_scale: np.ndarray,
        derivative: np.ndarray,
        previous_flux: np.ndarray,
        flux_scale: np.ndarray,
    ) -> np.ndarray:
        unknown = normalized_flat.reshape((-1, 3)) * variable_scale
        composition = np.asarray(boundary.edge_mass_fraction, dtype=np.float64)
        current = np.column_stack(
            (
                unknown[:, 0],
                unknown[:, 1],
                unknown[:, 2],
                unknown[:, 2],
                np.tile(composition[1:5, None], (1, y.size)).T,
            )
        )
        temperature_minimum, temperature_maximum = (
            self.reference.chemistry.temperature_bounds
        )
        if (
            not np.all(np.isfinite(current))
            or np.any(current[:, 2] < temperature_minimum)
            or np.any(current[:, 2] > temperature_maximum)
        ):
            return np.full(current.shape[0] * 3, 1.0e6)

        velocity_gradient = _differentiate(derivative, current[:, 0:2])
        temperature_gradient = _differentiate(derivative, current[:, 2])
        flux = np.empty((y.size, 3), dtype=np.float64)
        normal_flux = np.empty_like(flux)
        for index in range(y.size):
            state = Air5BoundaryLayerState(
                pressure=boundary.pressure,
                velocity=np.array([current[index, 0], current[index, 1], 0.0]),
                temperature=current[index, 2],
                tv=current[index, 2],
                mass_fraction=composition,
            )
            flux[index] = self.reference.streamwise_flux(state)[[0, 1, 6]]
            normal_flux[index] = self.reference.wall_normal_flux(
                state,
                grad_velocity_y=np.array(
                    [velocity_gradient[index, 0], velocity_gradient[index, 1], 0.0]
                ),
                grad_temperature_y=temperature_gradient[index],
                grad_tv_y=temperature_gradient[index],
                grad_mass_fraction_y=np.zeros(5),
            )[[0, 1, 6]]

        residual = (
            (flux - previous_flux) / streamwise_step
            + _differentiate(derivative, normal_flux)
        ) / flux_scale
        residual[0] = np.array(
            [
                (current[0, 0] - boundary.wall_velocity)
                / boundary.edge_velocity[0],
                current[0, 1] / boundary.edge_velocity[0],
                (current[0, 2] - boundary.wall_temperature)
                / boundary.edge_temperature,
            ]
        )
        residual[-1, 1] = (
            current[-1, 0] - boundary.edge_velocity[0]
        ) / boundary.edge_velocity[0]
        residual[-1, 2] = (
            current[-1, 2] - boundary.edge_temperature
        ) / boundary.edge_temperature
        return residual.ravel()

    def march_single_temperature_frozen_station(
        self,
        previous: np.ndarray,
        y: np.ndarray,
        streamwise_step: float,
        boundary: Air5HblBoundaryConditions,
        *,
        initial_guess: np.ndarray | None = None,
        residual_tolerance: float = 1.0e-10,
        max_function_evaluations: int = 300,
    ) -> Air5HblStationResult:
        """Advance the A1-R1 single-temperature uniform-composition limit."""
        self._validate_boundary(boundary)
        previous = np.asarray(previous, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if previous.shape != (y.size, NUM_PRIMITIVES):
            raise ValueError("HBL previous station has an invalid shape")
        if streamwise_step <= 0.0 or not np.isfinite(streamwise_step):
            raise ValueError("HBL streamwise step must be positive and finite")
        if initial_guess is None:
            initial = previous[:, [0, 1, 2]].copy()
        else:
            initial_guess = np.asarray(initial_guess, dtype=np.float64)
            if initial_guess.shape != previous.shape:
                raise ValueError("HBL station initial guess has an invalid shape")
            initial = initial_guess[:, [0, 1, 2]].copy()
        initial[0] = np.array(
            [boundary.wall_velocity, 0.0, boundary.wall_temperature]
        )
        initial[-1, [0, 2]] = np.array(
            [boundary.edge_velocity[0], boundary.edge_temperature]
        )
        variable_scale = np.array(
            [
                boundary.edge_velocity[0],
                max(np.max(np.abs(initial[:, 1])), 1.0e-2 * boundary.edge_velocity[0]),
                boundary.edge_temperature,
            ]
        )
        derivative = _derivative_matrix(y)
        composition = np.asarray(boundary.edge_mass_fraction, dtype=np.float64)
        previous_flux = np.empty((y.size, 3), dtype=np.float64)
        for index in range(y.size):
            previous_state = Air5BoundaryLayerState(
                pressure=boundary.pressure,
                velocity=np.array(
                    [previous[index, 0], previous[index, 1], 0.0]
                ),
                temperature=previous[index, 2],
                tv=previous[index, 2],
                mass_fraction=composition,
            )
            previous_flux[index] = self.reference.streamwise_flux(previous_state)[
                [0, 1, 6]
            ]
        edge = Air5BoundaryLayerState(
            pressure=boundary.pressure,
            velocity=np.asarray(boundary.edge_velocity, dtype=np.float64),
            temperature=boundary.edge_temperature,
            tv=boundary.edge_temperature,
            mass_fraction=composition,
        )
        flux_scale = np.abs(self.reference.streamwise_flux(edge)[[0, 1, 6]])
        flux_scale = np.maximum(
            flux_scale / streamwise_step, np.finfo(np.float64).tiny
        )
        lower = np.tile(
            np.array([-np.inf, -np.inf, self.reference.chemistry.temperature_bounds[0]])
            / variable_scale,
            y.size,
        )
        upper = np.tile(
            np.array([np.inf, np.inf, self.reference.chemistry.temperature_bounds[1]])
            / variable_scale,
            y.size,
        )
        result = least_squares(
            lambda trial: self._single_temperature_frozen_residual(
                trial,
                y,
                streamwise_step,
                boundary,
                variable_scale,
                derivative,
                previous_flux,
                flux_scale,
            ),
            initial.ravel() / np.tile(variable_scale, y.size),
            bounds=(lower, upper),
            jac_sparsity=self._three_equation_sparsity(y.size),
            x_scale="jac",
            diff_step=1.0e-5,
            xtol=1.0e-12,
            ftol=1.0e-12,
            gtol=1.0e-12,
            max_nfev=max_function_evaluations,
        )
        residual_max = float(np.max(np.abs(result.fun)))
        if not result.success or residual_max > residual_tolerance:
            raise RuntimeError(
                "single-temperature HBL station solve failed: "
                f"success={result.success} residual={residual_max:.3e} "
                f"message={result.message}"
            )
        solved = result.x.reshape((-1, 3)) * variable_scale
        if not np.all(np.isfinite(solved)) or np.any(solved[:, 2] <= 0.0):
            raise RuntimeError(
                "single-temperature HBL station solve produced a nonphysical state"
            )
        primitive = np.column_stack(
            (
                solved[:, 0],
                solved[:, 1],
                solved[:, 2],
                solved[:, 2],
                np.tile(composition[1:5, None], (1, y.size)).T,
            )
        )
        return Air5HblStationResult(
            primitive=primitive,
            scaled_residual_max=residual_max,
            function_evaluations=result.nfev,
        )

    def march_single_temperature_frozen(
        self,
        initial: np.ndarray,
        y: np.ndarray,
        x: np.ndarray,
        boundary: Air5HblBoundaryConditions,
        *,
        residual_tolerance: float = 1.0e-10,
    ) -> Air5HblMarchResult:
        initial = np.asarray(initial, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        if initial.shape != (y.size, NUM_PRIMITIVES):
            raise ValueError("HBL initial station has an invalid shape")
        if x.ndim != 1 or x.size < 2 or np.any(np.diff(x) <= 0.0):
            raise ValueError("HBL x coordinate must be strictly increasing")
        composition = np.asarray(boundary.edge_mass_fraction, dtype=np.float64)
        current = initial.copy()
        current[:, 3] = current[:, 2]
        current[:, 4:8] = composition[1:5]
        current[0, [0, 1, 2, 3]] = np.array(
            [
                boundary.wall_velocity,
                0.0,
                boundary.wall_temperature,
                boundary.wall_temperature,
            ]
        )
        current[-1, [0, 2, 3]] = np.array(
            [
                boundary.edge_velocity[0],
                boundary.edge_temperature,
                boundary.edge_temperature,
            ]
        )
        history = np.empty((x.size, y.size, NUM_PRIMITIVES), dtype=np.float64)
        history[0] = current
        maximum_residual = 0.0
        for station, step in enumerate(np.diff(x), start=1):
            result = self.march_single_temperature_frozen_station(
                current,
                y,
                step,
                boundary,
                residual_tolerance=residual_tolerance,
            )
            current = result.primitive
            history[station] = current
            maximum_residual = max(maximum_residual, result.scaled_residual_max)
        return Air5HblMarchResult(
            x=x.copy(),
            primitive=history,
            scaled_linear_residual_max=maximum_residual,
        )

    def scaled_residual(
        self,
        current_flat: np.ndarray,
        previous: np.ndarray,
        y: np.ndarray,
        streamwise_step: float,
        boundary: Air5HblBoundaryConditions,
        *,
        source_mode: str = "off",
        single_temperature: bool = False,
    ) -> np.ndarray:
        self._validate_boundary(boundary)
        y = np.asarray(y, dtype=np.float64)
        previous = np.asarray(previous, dtype=np.float64)
        current = np.asarray(current_flat, dtype=np.float64).reshape((-1, 8))
        if previous.shape != current.shape or current.shape != (y.size, 8):
            raise ValueError("HBL primitive profile has an invalid shape")
        if not np.isfinite(streamwise_step) or streamwise_step <= 0.0:
            raise ValueError("HBL streamwise step must be positive and finite")
        if source_mode not in {"off", "vt", "chemical", "coupled"}:
            raise ValueError("unsupported HBL source mode")
        if not np.all(np.isfinite(current)) or not np.all(np.isfinite(previous)):
            raise ValueError("HBL primitive profile contains non-finite values")

        current_mass_fraction = self._mass_fraction(current)
        invalid = (
            np.any(current[:, :1] < 0.0)
            or np.any(current[:, 2:4] <= 0.0)
            or np.any(current_mass_fraction < 0.0)
            or np.any(current_mass_fraction > 1.0)
        )
        if invalid:
            return np.full(current.size, 1.0e6, dtype=np.float64)

        derivative = _derivative_matrix(y)
        current_states = self._states(current, boundary.pressure)
        previous_states = self._states(previous, boundary.pressure)
        streamwise_flux = np.asarray(
            [self.reference.streamwise_flux(state) for state in current_states]
        )
        previous_flux = np.asarray(
            [self.reference.streamwise_flux(state) for state in previous_states]
        )

        velocity_gradient = _differentiate(derivative, current[:, 0:2])
        temperature_gradient = _differentiate(derivative, current[:, 2])
        tv_gradient = _differentiate(derivative, current[:, 3])
        independent_species_gradient = _differentiate(derivative, current[:, 4:8])
        species_gradient = np.column_stack(
            (-np.sum(independent_species_gradient, axis=1), independent_species_gradient)
        )
        normal_flux = np.asarray(
            [
                self.reference.wall_normal_flux(
                    state,
                    grad_velocity_y=np.array(
                        [velocity_gradient[index, 0], velocity_gradient[index, 1], 0.0]
                    ),
                    grad_temperature_y=temperature_gradient[index],
                    grad_tv_y=tv_gradient[index],
                    grad_mass_fraction_y=species_gradient[index],
                )
                for index, state in enumerate(current_states)
            ]
        )
        normal_divergence = _differentiate(derivative, normal_flux)
        if source_mode == "off":
            source = np.zeros_like(streamwise_flux)
        else:
            source = np.asarray(
                [self.reference.source(state, source_mode) for state in current_states]
            )

        scale = self._flux_scale(boundary, streamwise_step)
        residual = (
            (streamwise_flux - previous_flux) / streamwise_step
            + normal_divergence
            - source
        ) / scale[None, :]
        if single_temperature:
            residual[1:-1, 7] = (
                current[1:-1, 3] - current[1:-1, 2]
            ) / boundary.edge_temperature

        u_scale = boundary.edge_velocity[0]
        t_scale = boundary.edge_temperature
        residual[0] = np.r_[
            (current[0, 0] - boundary.wall_velocity) / u_scale,
            current[0, 1] / u_scale,
            (current[0, 2] - boundary.wall_temperature) / t_scale,
            (current[0, 3] - boundary.wall_tv) / t_scale,
            current[0, 4:8] - current[1, 4:8],
        ]
        residual[-1] = np.r_[
            (current[-1, 0] - boundary.edge_velocity[0]) / u_scale,
            (current[-1, 1] - boundary.edge_velocity[1]) / u_scale,
            (current[-1, 2] - boundary.edge_temperature) / t_scale,
            (current[-1, 3] - boundary.edge_tv) / t_scale,
            current[-1, 4:8] - boundary.edge_mass_fraction[1:5],
        ]
        return residual.ravel()

    def march_station(
        self,
        previous: np.ndarray,
        y: np.ndarray,
        streamwise_step: float,
        boundary: Air5HblBoundaryConditions,
        *,
        source_mode: str = "off",
        single_temperature: bool = False,
        residual_tolerance: float = 1.0e-10,
        max_function_evaluations: int = 1000,
    ) -> Air5HblStationResult:
        previous = np.asarray(previous, dtype=np.float64)
        if previous.ndim != 2 or previous.shape[1] != NUM_PRIMITIVES:
            raise ValueError("HBL primitive profile must have eight columns")
        variable_scale = np.array(
            [
                boundary.edge_velocity[0],
                max(np.max(np.abs(previous[:, 1])), 1.0e-4 * boundary.edge_velocity[0]),
                boundary.edge_temperature,
                boundary.edge_tv,
                *[
                    max(np.max(np.abs(previous[:, column])), 1.0e-14)
                    for column in range(4, 8)
                ],
            ],
            dtype=np.float64,
        )
        tiled_scale = np.tile(variable_scale, previous.shape[0])
        lower = np.tile(
            np.array(
                [
                    0.0,
                    -0.25 * boundary.edge_velocity[0],
                    self.reference.chemistry.temperature_bounds[0],
                    self.reference.chemistry.temperature_bounds[0],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            previous.shape[0],
        ) / tiled_scale
        upper = np.tile(
            np.array(
                [
                    1.5 * boundary.edge_velocity[0],
                    0.25 * boundary.edge_velocity[0],
                    self.reference.chemistry.temperature_bounds[1],
                    self.reference.chemistry.temperature_bounds[1],
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ]
            ),
            previous.shape[0],
        ) / tiled_scale
        result = least_squares(
            lambda trial: self.scaled_residual(
                trial * tiled_scale,
                previous,
                y,
                streamwise_step,
                boundary,
                source_mode=source_mode,
                single_temperature=single_temperature,
            ),
            previous.ravel() / tiled_scale,
            bounds=(lower, upper),
            xtol=1.0e-12,
            ftol=1.0e-12,
            gtol=1.0e-12,
            jac_sparsity=self._jacobian_sparsity(previous.shape[0]),
            max_nfev=max_function_evaluations,
        )
        residual_max = float(np.max(np.abs(result.fun)))
        if not result.success or residual_max > residual_tolerance:
            raise RuntimeError(
                "HBL station solve failed: "
                f"success={result.success} residual={residual_max:.3e} "
                f"message={result.message}"
            )
        primitive = (result.x * tiled_scale).reshape(previous.shape)
        if np.any(self._mass_fraction(primitive) < 0.0):
            raise RuntimeError("HBL station solve produced a negative species")
        return Air5HblStationResult(
            primitive=primitive,
            scaled_residual_max=residual_max,
            function_evaluations=result.nfev,
        )
