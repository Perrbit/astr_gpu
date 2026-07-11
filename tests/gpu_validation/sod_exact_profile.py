#!/usr/bin/env python3
"""Exact ideal-gas Riemann solution and diagnostics for ASTR Sod output."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class PrimitiveState:
    rho: float
    velocity: float
    pressure: float


@dataclass(frozen=True)
class CaseProfile:
    x: np.ndarray
    time: float
    fields: dict[str, np.ndarray]


@dataclass(frozen=True)
class ErrorNorms:
    l1: float
    l2: float
    linf: float


@dataclass(frozen=True)
class BoundViolation:
    lower_violation: float
    upper_violation: float
    normalized_violation: float


@dataclass(frozen=True)
class TransitionDiagnostic:
    resolved: bool
    thickness_cells: float
    position_error_cells: float


@dataclass(frozen=True)
class SodDiagnostics:
    grid_spacing: float
    smooth_point_count: int
    smooth_errors: dict[str, ErrorNorms]
    bounds: dict[str, BoundViolation]
    transitions: dict[str, TransitionDiagnostic]


@dataclass(frozen=True)
class ValidationLimits:
    max_smooth_l1: float = float("inf")
    max_smooth_l2: float = float("inf")
    max_smooth_linf: float = float("inf")
    max_bound_violation: float = float("inf")
    max_contact_thickness_cells: float = float("inf")
    max_shock_thickness_cells: float = float("inf")
    max_position_error_cells: float = float("inf")


@dataclass(frozen=True)
class RiemannSolution:
    left: PrimitiveState
    right: PrimitiveState
    gamma: float
    star_pressure: float
    star_velocity: float
    left_star_density: float
    right_star_density: float

    def sample(self, x: np.ndarray, time: float, x0: float = 0.0) -> dict[str, np.ndarray]:
        if time <= 0.0:
            raise ValueError("time must be positive")

        x = np.asarray(x, dtype=np.float64)
        similarity = (x - x0) / time
        rho = np.empty_like(similarity)
        velocity = np.empty_like(similarity)
        pressure = np.empty_like(similarity)

        for index, xi in np.ndenumerate(similarity):
            state = self._sample_similarity(float(xi))
            rho[index] = state.rho
            velocity[index] = state.velocity
            pressure[index] = state.pressure

        return {"ro": rho, "u1": velocity, "p": pressure}

    def _sample_similarity(self, xi: float) -> PrimitiveState:
        if xi <= self.star_velocity:
            return self._sample_left(xi)
        return self._sample_right(xi)

    def _sample_left(self, xi: float) -> PrimitiveState:
        state = self.left
        gamma = self.gamma
        sound = _sound_speed(state, gamma)
        star = PrimitiveState(
            self.left_star_density,
            self.star_velocity,
            self.star_pressure,
        )

        if self.star_pressure > state.pressure:
            shock_speed = state.velocity - sound * np.sqrt(
                (gamma + 1.0) * self.star_pressure / (2.0 * gamma * state.pressure)
                + (gamma - 1.0) / (2.0 * gamma)
            )
            return state if xi <= shock_speed else star

        star_sound = sound * (self.star_pressure / state.pressure) ** (
            (gamma - 1.0) / (2.0 * gamma)
        )
        head_speed = state.velocity - sound
        tail_speed = self.star_velocity - star_sound
        if xi <= head_speed:
            return state
        if xi >= tail_speed:
            return star

        fan_velocity = 2.0 / (gamma + 1.0) * (
            sound + 0.5 * (gamma - 1.0) * state.velocity + xi
        )
        fan_sound = 2.0 / (gamma + 1.0) * (
            sound + 0.5 * (gamma - 1.0) * (state.velocity - xi)
        )
        return PrimitiveState(
            state.rho * (fan_sound / sound) ** (2.0 / (gamma - 1.0)),
            fan_velocity,
            state.pressure * (fan_sound / sound) ** (2.0 * gamma / (gamma - 1.0)),
        )

    def _sample_right(self, xi: float) -> PrimitiveState:
        state = self.right
        gamma = self.gamma
        sound = _sound_speed(state, gamma)
        star = PrimitiveState(
            self.right_star_density,
            self.star_velocity,
            self.star_pressure,
        )

        if self.star_pressure > state.pressure:
            shock_speed = state.velocity + sound * np.sqrt(
                (gamma + 1.0) * self.star_pressure / (2.0 * gamma * state.pressure)
                + (gamma - 1.0) / (2.0 * gamma)
            )
            return state if xi >= shock_speed else star

        star_sound = sound * (self.star_pressure / state.pressure) ** (
            (gamma - 1.0) / (2.0 * gamma)
        )
        head_speed = state.velocity + sound
        tail_speed = self.star_velocity + star_sound
        if xi >= head_speed:
            return state
        if xi <= tail_speed:
            return star

        fan_velocity = 2.0 / (gamma + 1.0) * (
            -sound + 0.5 * (gamma - 1.0) * state.velocity + xi
        )
        fan_sound = 2.0 / (gamma + 1.0) * (
            sound - 0.5 * (gamma - 1.0) * (state.velocity - xi)
        )
        return PrimitiveState(
            state.rho * (fan_sound / sound) ** (2.0 / (gamma - 1.0)),
            fan_velocity,
            state.pressure * (fan_sound / sound) ** (2.0 * gamma / (gamma - 1.0)),
        )


def _sound_speed(state: PrimitiveState, gamma: float) -> float:
    return float(np.sqrt(gamma * state.pressure / state.rho))


def _pressure_function(
    pressure: float,
    state: PrimitiveState,
    gamma: float,
) -> tuple[float, float]:
    if pressure > state.pressure:
        coefficient_a = 2.0 / ((gamma + 1.0) * state.rho)
        coefficient_b = (gamma - 1.0) / (gamma + 1.0) * state.pressure
        root = np.sqrt(coefficient_a / (pressure + coefficient_b))
        value = (pressure - state.pressure) * root
        derivative = root * (
            1.0 - 0.5 * (pressure - state.pressure) / (pressure + coefficient_b)
        )
        return float(value), float(derivative)

    sound = _sound_speed(state, gamma)
    ratio = pressure / state.pressure
    exponent = (gamma - 1.0) / (2.0 * gamma)
    value = 2.0 * sound / (gamma - 1.0) * (ratio**exponent - 1.0)
    derivative = ratio ** (-(gamma + 1.0) / (2.0 * gamma)) / (state.rho * sound)
    return float(value), float(derivative)


def _star_density(state: PrimitiveState, star_pressure: float, gamma: float) -> float:
    ratio = star_pressure / state.pressure
    if star_pressure > state.pressure:
        coefficient = (gamma - 1.0) / (gamma + 1.0)
        return state.rho * (ratio + coefficient) / (coefficient * ratio + 1.0)
    return state.rho * ratio ** (1.0 / gamma)


def solve_riemann(
    left: PrimitiveState,
    right: PrimitiveState,
    gamma: float = 1.4,
) -> RiemannSolution:
    if gamma <= 1.0:
        raise ValueError("gamma must be greater than one")
    for state in (left, right):
        if state.rho <= 0.0 or state.pressure <= 0.0:
            raise ValueError("density and pressure must be positive")

    left_sound = _sound_speed(left, gamma)
    right_sound = _sound_speed(right, gamma)
    pressure = max(
        1.0e-14,
        0.5 * (left.pressure + right.pressure)
        - 0.125
        * (right.velocity - left.velocity)
        * (left.rho + right.rho)
        * (left_sound + right_sound),
    )

    for _ in range(100):
        left_value, left_derivative = _pressure_function(pressure, left, gamma)
        right_value, right_derivative = _pressure_function(pressure, right, gamma)
        next_pressure = pressure - (
            left_value + right_value + right.velocity - left.velocity
        ) / (left_derivative + right_derivative)
        next_pressure = max(1.0e-14, next_pressure)
        if abs(next_pressure - pressure) <= 1.0e-13 * max(1.0, next_pressure):
            pressure = next_pressure
            break
        pressure = next_pressure
    else:
        raise RuntimeError("exact Riemann pressure solve did not converge")

    left_value, _ = _pressure_function(pressure, left, gamma)
    right_value, _ = _pressure_function(pressure, right, gamma)
    velocity = 0.5 * (left.velocity + right.velocity + right_value - left_value)

    return RiemannSolution(
        left=left,
        right=right,
        gamma=gamma,
        star_pressure=pressure,
        star_velocity=velocity,
        left_star_density=_star_density(left, pressure, gamma),
        right_star_density=_star_density(right, pressure, gamma),
    )


def _profile_along_axis(values: np.ndarray, axis: int) -> np.ndarray:
    ordered = np.moveaxis(values, axis, 0)
    if ordered.ndim == 1:
        return np.asarray(ordered, dtype=np.float64)
    transverse_axes = tuple(range(1, ordered.ndim))
    return np.asarray(np.mean(ordered, axis=transverse_axes), dtype=np.float64)


def _detect_x_axis(x: np.ndarray) -> int:
    if x.ndim == 0:
        raise ValueError("grid x dataset must have at least one dimension")
    variation = []
    for axis in range(x.ndim):
        line = _profile_along_axis(x, axis)
        variation.append(float(np.ptp(line)))
    axis = int(np.argmax(variation))
    if variation[axis] <= 0.0:
        raise ValueError("grid x dataset has no varying coordinate axis")
    return axis


def read_case_profile(case: Path) -> CaseProfile:
    case = Path(case)
    grid_path = case / "datin" / "grid.h5"
    flow_path = case / "outdat" / "flowfield.h5"
    if not grid_path.exists():
        raise FileNotFoundError(grid_path)
    if not flow_path.exists():
        raise FileNotFoundError(flow_path)

    with h5py.File(grid_path, "r") as h5:
        if "x" not in h5:
            raise KeyError(f"{grid_path}: missing dataset x")
        grid_x = np.asarray(h5["x"][...], dtype=np.float64)

    x_axis = _detect_x_axis(grid_x)
    x = _profile_along_axis(grid_x, x_axis)

    fields: dict[str, np.ndarray] = {}
    names = ("ro", "u1", "u2", "u3", "p")
    with h5py.File(flow_path, "r") as h5:
        missing = [name for name in names if name not in h5]
        if missing:
            raise KeyError(f"{flow_path}: missing datasets {missing}")
        if "time" not in h5:
            raise KeyError(f"{flow_path}: missing dataset time")
        time_values = np.asarray(h5["time"][...], dtype=np.float64).reshape(-1)
        if time_values.size != 1:
            raise ValueError(f"{flow_path}: expected scalar time dataset")
        time = float(time_values[0])
        for name in names:
            values = np.asarray(h5[name][...], dtype=np.float64)
            if values.shape != grid_x.shape:
                raise ValueError(
                    f"{flow_path}: dataset {name} shape {values.shape} "
                    f"does not match grid shape {grid_x.shape}"
                )
            fields[name] = _profile_along_axis(values, x_axis)

    order = np.argsort(x)
    return CaseProfile(
        x=x[order],
        time=time,
        fields={name: values[order] for name, values in fields.items()},
    )


def _reconstruct_q(fields: dict[str, np.ndarray], gamma: float) -> dict[str, np.ndarray]:
    rho = fields["ro"]
    u1 = fields["u1"]
    u2 = fields["u2"]
    u3 = fields["u3"]
    pressure = fields["p"]
    return {
        "q1": rho,
        "q2": rho * u1,
        "q3": rho * u2,
        "q4": rho * u3,
        "q5": pressure / (gamma - 1.0)
        + 0.5 * rho * (u1 * u1 + u2 * u2 + u3 * u3),
    }


def _standard_wave_positions(
    solution: RiemannSolution,
    time: float,
    x0: float,
) -> dict[str, float]:
    if solution.star_pressure >= solution.left.pressure:
        raise ValueError("Sod diagnostics require a left rarefaction")
    if solution.star_pressure <= solution.right.pressure:
        raise ValueError("Sod diagnostics require a right shock")

    gamma = solution.gamma
    left_sound = _sound_speed(solution.left, gamma)
    left_star_sound = left_sound * (
        solution.star_pressure / solution.left.pressure
    ) ** ((gamma - 1.0) / (2.0 * gamma))
    right_sound = _sound_speed(solution.right, gamma)
    right_shock_speed = solution.right.velocity + right_sound * np.sqrt(
        (gamma + 1.0)
        * solution.star_pressure
        / (2.0 * gamma * solution.right.pressure)
        + (gamma - 1.0) / (2.0 * gamma)
    )
    speeds = {
        "rarefaction_head": solution.left.velocity - left_sound,
        "rarefaction_tail": solution.star_velocity - left_star_sound,
        "contact": solution.star_velocity,
        "shock": float(right_shock_speed),
    }
    return {name: x0 + time * speed for name, speed in speeds.items()}


def _normalized_error(
    numerical: np.ndarray,
    exact: np.ndarray,
    error_mask: np.ndarray,
    scale: float,
) -> ErrorNorms:
    difference = numerical[error_mask] - exact[error_mask]
    return ErrorNorms(
        l1=float(np.mean(np.abs(difference)) / scale),
        l2=float(np.sqrt(np.mean(difference * difference)) / scale),
        linf=float(np.max(np.abs(difference)) / scale),
    )


def _bound_violation(
    numerical: np.ndarray,
    exact: np.ndarray,
    mask: np.ndarray,
) -> BoundViolation:
    exact_values = exact[mask]
    numerical_values = numerical[mask]
    exact_minimum = float(np.min(exact_values))
    exact_maximum = float(np.max(exact_values))
    lower = max(0.0, exact_minimum - float(np.min(numerical_values)))
    upper = max(0.0, float(np.max(numerical_values)) - exact_maximum)
    scale = max(abs(exact_minimum), abs(exact_maximum), exact_maximum - exact_minimum, 1.0e-14)
    return BoundViolation(lower, upper, max(lower, upper) / scale)


def _crossing_position(
    x: np.ndarray,
    values: np.ndarray,
    left_value: float,
    right_value: float,
    target: float,
    expected_position: float,
    radius: float,
) -> float:
    normalized = (values - right_value) / (left_value - right_value)
    candidates: list[float] = []
    for index in range(x.size - 1):
        if x[index + 1] < expected_position - radius:
            continue
        if x[index] > expected_position + radius:
            break
        first = float(normalized[index] - target)
        second = float(normalized[index + 1] - target)
        if first == 0.0:
            candidates.append(float(x[index]))
        if first * second < 0.0 or second == 0.0:
            denominator = float(normalized[index + 1] - normalized[index])
            if denominator != 0.0:
                fraction = (target - float(normalized[index])) / denominator
                candidates.append(float(x[index] + fraction * (x[index + 1] - x[index])))
    if not candidates:
        raise ValueError(
            f"no density crossing at level {target:.1f} near x={expected_position:.8e}"
        )
    return min(candidates, key=lambda candidate: abs(candidate - expected_position))


def _transition_diagnostic(
    x: np.ndarray,
    density: np.ndarray,
    left_density: float,
    right_density: float,
    expected_position: float,
    radius: float,
    grid_spacing: float,
) -> TransitionDiagnostic:
    try:
        position_90 = _crossing_position(
            x, density, left_density, right_density, 0.9, expected_position, radius
        )
        position_50 = _crossing_position(
            x, density, left_density, right_density, 0.5, expected_position, radius
        )
        position_10 = _crossing_position(
            x, density, left_density, right_density, 0.1, expected_position, radius
        )
    except ValueError:
        return TransitionDiagnostic(
            resolved=False,
            thickness_cells=float("inf"),
            position_error_cells=float("inf"),
        )
    return TransitionDiagnostic(
        resolved=True,
        thickness_cells=abs(position_10 - position_90) / grid_spacing,
        position_error_cells=abs(position_50 - expected_position) / grid_spacing,
    )


def compute_diagnostics(
    profile: CaseProfile,
    solution: RiemannSolution,
    analysis_half_width: float = 2.5,
    exclude_cells: float = 3.0,
    x0: float = 0.0,
) -> SodDiagnostics:
    x = np.asarray(profile.x, dtype=np.float64)
    if x.ndim != 1 or x.size < 3 or np.any(np.diff(x) <= 0.0):
        raise ValueError("profile x coordinates must be a strictly increasing one-dimensional array")
    grid_spacing = float(np.median(np.diff(x)))
    if profile.time <= 0.0:
        raise ValueError("profile time must be positive")

    required = ("ro", "u1", "u2", "u3", "p")
    missing = [name for name in required if name not in profile.fields]
    if missing:
        raise KeyError(f"profile missing fields {missing}")
    for name in required:
        if np.asarray(profile.fields[name]).shape != x.shape:
            raise ValueError(f"profile field {name} does not match x coordinates")

    exact_primitive = solution.sample(x=x, time=profile.time, x0=x0)
    exact_primitive["u2"] = np.zeros_like(x)
    exact_primitive["u3"] = np.zeros_like(x)
    numerical_primitive = {
        name: np.asarray(profile.fields[name], dtype=np.float64) for name in required
    }
    exact_q = _reconstruct_q(exact_primitive, solution.gamma)
    numerical_q = _reconstruct_q(numerical_primitive, solution.gamma)
    exact_fields = {**exact_primitive, **exact_q}
    numerical_fields = {**numerical_primitive, **numerical_q}

    wave_positions = _standard_wave_positions(solution, profile.time, x0)
    analysis_mask = np.abs(x - x0) <= analysis_half_width
    smooth_mask = analysis_mask.copy()
    for position in wave_positions.values():
        smooth_mask &= np.abs(x - position) > exclude_cells * grid_spacing
    if not np.any(smooth_mask):
        raise ValueError("smooth-region mask contains no grid points")

    error_names = ("ro", "u1", "p", "q1", "q2", "q3", "q4", "q5")
    normalization_scales = {
        name: max(
            float(np.max(np.abs(exact_fields[name][analysis_mask]))),
            float(np.ptp(exact_fields[name][analysis_mask])),
            1.0e-14,
        )
        for name in error_names
    }
    momentum_scale = normalization_scales["q2"]
    normalization_scales["q3"] = momentum_scale
    normalization_scales["q4"] = momentum_scale
    smooth_errors = {
        name: _normalized_error(
            numerical_fields[name],
            exact_fields[name],
            smooth_mask,
            normalization_scales[name],
        )
        for name in error_names
    }
    bounds = {
        name: _bound_violation(
            numerical_primitive[name], exact_primitive[name], analysis_mask
        )
        for name in ("ro", "u1", "p")
    }

    tail = wave_positions["rarefaction_tail"]
    contact = wave_positions["contact"]
    shock = wave_positions["shock"]
    contact_radius = 0.45 * min(contact - tail, shock - contact)
    shock_radius = 0.45 * (shock - contact)
    transitions = {
        "contact": _transition_diagnostic(
            x,
            numerical_primitive["ro"],
            solution.left_star_density,
            solution.right_star_density,
            contact,
            contact_radius,
            grid_spacing,
        ),
        "shock": _transition_diagnostic(
            x,
            numerical_primitive["ro"],
            solution.right_star_density,
            solution.right.rho,
            shock,
            shock_radius,
            grid_spacing,
        ),
    }
    return SodDiagnostics(
        grid_spacing=grid_spacing,
        smooth_point_count=int(np.count_nonzero(smooth_mask)),
        smooth_errors=smooth_errors,
        bounds=bounds,
        transitions=transitions,
    )


def evaluate_diagnostics(
    diagnostics: SodDiagnostics,
    limits: ValidationLimits,
) -> list[str]:
    failures: list[str] = []
    for name, error in diagnostics.smooth_errors.items():
        for norm_name, value, limit in (
            ("L1", error.l1, limits.max_smooth_l1),
            ("L2", error.l2, limits.max_smooth_l2),
            ("Linf", error.linf, limits.max_smooth_linf),
        ):
            if value > limit:
                failures.append(
                    f"{name} smooth {norm_name} {value:.8e} exceeds {limit:.8e}"
                )
    for name, bounds in diagnostics.bounds.items():
        if bounds.normalized_violation > limits.max_bound_violation:
            failures.append(
                f"{name} bound violation {bounds.normalized_violation:.8e} "
                f"exceeds {limits.max_bound_violation:.8e}"
            )

    contact = diagnostics.transitions["contact"]
    shock = diagnostics.transitions["shock"]
    if not contact.resolved:
        failures.append("contact transition unresolved")
    elif contact.thickness_cells > limits.max_contact_thickness_cells:
        failures.append(
            f"contact thickness {contact.thickness_cells:.8e} cells exceeds "
            f"{limits.max_contact_thickness_cells:.8e}"
        )
    if not shock.resolved:
        failures.append("shock transition unresolved")
    elif shock.thickness_cells > limits.max_shock_thickness_cells:
        failures.append(
            f"shock thickness {shock.thickness_cells:.8e} cells exceeds "
            f"{limits.max_shock_thickness_cells:.8e}"
        )
    for name, transition in diagnostics.transitions.items():
        if transition.resolved and transition.position_error_cells > limits.max_position_error_cells:
            failures.append(
                f"{name} position error {transition.position_error_cells:.8e} cells exceeds "
                f"{limits.max_position_error_cells:.8e}"
            )
    return failures


def format_report(
    profile: CaseProfile,
    solution: RiemannSolution,
    diagnostics: SodDiagnostics,
    failures: list[str],
) -> str:
    lines = [
        f"status: {'fail' if failures else 'pass'}",
        f"time: {profile.time:.16e}",
        f"grid_spacing: {diagnostics.grid_spacing:.16e}",
        f"smooth_point_count: {diagnostics.smooth_point_count}",
        f"star_pressure: {solution.star_pressure:.16e}",
        f"star_velocity: {solution.star_velocity:.16e}",
        f"left_star_density: {solution.left_star_density:.16e}",
        f"right_star_density: {solution.right_star_density:.16e}",
        "",
        "[smooth_errors]",
    ]
    for name, error in diagnostics.smooth_errors.items():
        lines.append(
            f"{name} l1={error.l1:.16e} l2={error.l2:.16e} linf={error.linf:.16e}"
        )

    lines.extend(["", "[bounds]"])
    for name, bounds in diagnostics.bounds.items():
        lines.append(
            f"{name} lower={bounds.lower_violation:.16e} "
            f"upper={bounds.upper_violation:.16e} "
            f"normalized={bounds.normalized_violation:.16e}"
        )

    lines.extend(["", "[transitions]"])
    for name, transition in diagnostics.transitions.items():
        lines.append(
            f"{name} thickness_cells={transition.thickness_cells:.16e} "
            f"position_error_cells={transition.position_error_cells:.16e} "
            f"resolved={'t' if transition.resolved else 'f'}"
        )

    lines.extend(["", "[failures]"])
    lines.extend(failures if failures else ["none"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare an ASTR Sod flowfield against the exact ideal-gas Riemann solution."
    )
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--x0", type=float, default=0.0)
    parser.add_argument("--analysis-half-width", type=float, default=2.5)
    parser.add_argument("--exclude-cells", type=float, default=3.0)
    parser.add_argument("--left-rho", type=float, default=1.0)
    parser.add_argument("--left-velocity", type=float, default=0.0)
    parser.add_argument("--left-pressure", type=float, default=1.0)
    parser.add_argument("--right-rho", type=float, default=0.125)
    parser.add_argument("--right-velocity", type=float, default=0.0)
    parser.add_argument("--right-pressure", type=float, default=0.1)
    parser.add_argument("--max-smooth-l1", type=float, default=float("inf"))
    parser.add_argument("--max-smooth-l2", type=float, default=float("inf"))
    parser.add_argument("--max-smooth-linf", type=float, default=float("inf"))
    parser.add_argument("--max-bound-violation", type=float, default=float("inf"))
    parser.add_argument(
        "--max-contact-thickness-cells", type=float, default=float("inf")
    )
    parser.add_argument(
        "--max-shock-thickness-cells", type=float, default=float("inf")
    )
    parser.add_argument(
        "--max-position-error-cells", type=float, default=float("inf")
    )
    args = parser.parse_args()

    profile = read_case_profile(args.case)
    solution = solve_riemann(
        PrimitiveState(args.left_rho, args.left_velocity, args.left_pressure),
        PrimitiveState(args.right_rho, args.right_velocity, args.right_pressure),
        gamma=args.gamma,
    )
    diagnostics = compute_diagnostics(
        profile,
        solution,
        analysis_half_width=args.analysis_half_width,
        exclude_cells=args.exclude_cells,
        x0=args.x0,
    )
    limits = ValidationLimits(
        max_smooth_l1=args.max_smooth_l1,
        max_smooth_l2=args.max_smooth_l2,
        max_smooth_linf=args.max_smooth_linf,
        max_bound_violation=args.max_bound_violation,
        max_contact_thickness_cells=args.max_contact_thickness_cells,
        max_shock_thickness_cells=args.max_shock_thickness_cells,
        max_position_error_cells=args.max_position_error_cells,
    )
    failures = evaluate_diagnostics(diagnostics, limits)
    report = format_report(profile, solution, diagnostics, failures)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report, end="")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
