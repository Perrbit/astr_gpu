"""Independent wall and profile diagnostics for Cartesian fixed-air5 HBL fields."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from tests.gpu_validation.air5_radau_reference import Air5RadauReference
    from tests.gpu_validation.air5_transport_reference import Air5TransportReference
except ModuleNotFoundError:
    from air5_radau_reference import Air5RadauReference
    from air5_transport_reference import Air5TransportReference

try:
    from tests.gpu_validation.compare_q_validation_snapshots import read_q_snapshot
except ModuleNotFoundError:
    from compare_q_validation_snapshots import read_q_snapshot


@dataclass(frozen=True)
class Air5HblDiagnostics:
    wall_x: np.ndarray
    skin_friction: np.ndarray
    heat_flux_total: np.ndarray
    heat_flux_translational: np.ndarray
    heat_flux_vibrational: np.ndarray
    heat_flux_species: np.ndarray
    profile_x: np.ndarray
    profile_y: np.ndarray
    profile_velocity: np.ndarray
    profile_temperature: np.ndarray
    profile_tv: np.ndarray
    profile_mass_fraction: np.ndarray


def load_active_q_snapshot(path: Path) -> np.ndarray:
    snapshot = read_q_snapshot(path)
    im, jm, km, hm, numq = snapshot.header
    if numq != 11:
        raise ValueError(f"air5 HBL diagnostics require numq=11, found {numq}")
    full = snapshot.values.reshape(
        (im + 2 * hm + 1, jm + 2 * hm + 1, km + 2 * hm + 1, numq),
        order="F",
    )
    return full[
        hm : hm + im + 1,
        hm : hm + jm + 1,
        hm : hm + km + 1,
        :,
    ].copy(order="F")


def save_diagnostics(
    result: Air5HblDiagnostics, archive: Path, report: Path
) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    np.savez(archive, **result.__dict__)
    decomposition_residual = result.heat_flux_total - (
        result.heat_flux_translational
        + result.heat_flux_vibrational
        + result.heat_flux_species
    )
    lines = [
        "status: pass",
        "skin_friction_definition: 2*tau_xy/(rho_inf*u_inf^2)",
        "heat_flux_definition: positive from lower wall into fluid",
        f"min_skin_friction: {np.min(result.skin_friction):.16e}",
        f"max_skin_friction: {np.max(result.skin_friction):.16e}",
        f"min_heat_flux_total: {np.min(result.heat_flux_total):.16e}",
        f"max_heat_flux_total: {np.max(result.heat_flux_total):.16e}",
        "max_heat_decomposition_residual: "
        f"{np.max(np.abs(decomposition_residual)):.16e}",
        f"minimum_species_mass_fraction: {np.min(result.profile_mass_fraction):.16e}",
        "maximum_species_sum_residual: "
        f"{np.max(np.abs(np.sum(result.profile_mass_fraction, axis=2)-1.0)):.16e}",
        f"minimum_temperature: {np.min(result.profile_temperature):.16e}",
        f"maximum_temperature: {np.max(result.profile_temperature):.16e}",
        f"minimum_vibrational_temperature: {np.min(result.profile_tv):.16e}",
        f"maximum_vibrational_temperature: {np.max(result.profile_tv):.16e}",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="ascii")


def finite_difference_weights(coordinate: np.ndarray, target: float) -> np.ndarray:
    """Return first-derivative weights exact through degree ``n - 1``."""
    nodes = np.asarray(coordinate, dtype=np.float64)
    offset = nodes - float(target)
    if nodes.ndim != 1 or nodes.size < 2 or np.unique(nodes).size != nodes.size:
        raise ValueError("finite-difference coordinates must be distinct")
    vandermonde = np.vstack([offset**power for power in range(nodes.size)])
    right_hand_side = np.zeros(nodes.size)
    right_hand_side[1] = 1.0
    return np.linalg.solve(vandermonde, right_hand_side)


def _derivative_matrix(coordinate: np.ndarray, stencil: int = 7) -> np.ndarray:
    coordinate = np.asarray(coordinate, dtype=np.float64)
    if coordinate.ndim != 1 or coordinate.size < 2:
        raise ValueError("derivative coordinate requires at least two points")
    width = min(stencil, coordinate.size)
    matrix = np.zeros((coordinate.size, coordinate.size))
    radius = width // 2
    for index in range(coordinate.size):
        start = min(max(index - radius, 0), coordinate.size - width)
        selection = np.arange(start, start + width)
        matrix[index, selection] = finite_difference_weights(
            coordinate[selection], coordinate[index]
        )
    return matrix


def _validate_coordinate(values: np.ndarray, name: str) -> np.ndarray:
    coordinate = np.asarray(values, dtype=np.float64)
    if (
        coordinate.ndim != 1
        or coordinate.size < 2
        or not np.all(np.isfinite(coordinate))
        or np.any(np.diff(coordinate) <= 0.0)
    ):
        raise ValueError(f"{name} coordinate must be finite and strictly increasing")
    return coordinate


def _recover_primitives(
    q: np.ndarray, model: Air5RadauReference
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = q.shape[:-1]
    flat = q.reshape((-1, 11))
    density = flat[:, 0]
    rho_species = flat[:, 5:10]
    if not np.all(np.isfinite(flat)):
        raise ValueError("air5 HBL field contains non-finite values")
    if np.any(density <= 0.0) or np.any(rho_species < 0.0):
        raise ValueError("air5 HBL field contains a non-positive physical state")
    if not np.allclose(
        np.sum(rho_species, axis=1), density, atol=2.0e-12, rtol=2.0e-12
    ):
        raise ValueError("air5 HBL species densities do not close to total density")

    momentum = flat[:, 1:4]
    velocity = momentum / density[:, None]
    temperature = np.asarray(
        [
            model.temperature_from_q5(
                density[index],
                momentum[index],
                rho_species[index],
                flat[index, 10],
                flat[index, 4],
            )
            for index in range(flat.shape[0])
        ]
    )
    tv = np.asarray(
        [
            model.tv_from_ev(rho_species[index], flat[index, 10])
            for index in range(flat.shape[0])
        ]
    )
    pressure = np.sum(
        rho_species * model.gas_constant[None, :], axis=1
    ) * temperature
    mass_fraction = rho_species / density[:, None]
    return (
        velocity.reshape((*shape, 3)),
        temperature.reshape(shape),
        tv.reshape(shape),
        pressure.reshape(shape),
        mass_fraction.reshape((*shape, 5)),
    )


def _spanwise_average(values: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.trapezoid(values, z, axis=-1) / (z[-1] - z[0])


def _interpolate_station(values: np.ndarray, x: np.ndarray, station: float) -> np.ndarray:
    if station < x[0] or station > x[-1]:
        raise ValueError(f"profile station {station:g} is outside the x domain")
    upper = int(np.searchsorted(x, station, side="left"))
    if upper == 0:
        return values[0]
    if upper == x.size:
        return values[-1]
    if x[upper] == station:
        return values[upper]
    lower = upper - 1
    weight = (station - x[lower]) / (x[upper] - x[lower])
    return (1.0 - weight) * values[lower] + weight * values[upper]


def analyze_cartesian_hbl(
    q: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    model: Air5RadauReference,
    transport: Air5TransportReference,
    reference_density: float,
    reference_velocity: float,
    profile_stations: tuple[float, ...],
    spanwise_tolerance: float = 2.0e-12,
) -> Air5HblDiagnostics:
    """Diagnose a spanwise-uniform Cartesian lower-wall air5 boundary layer.

    Heat flux is positive from the lower wall into the fluid. The three reported
    components are translational conduction, vibrational conduction, and species
    enthalpy diffusion under the same sign convention.
    """
    q = np.asarray(q, dtype=np.float64)
    x = _validate_coordinate(x, "x")
    y = _validate_coordinate(y, "y")
    z = _validate_coordinate(z, "z")
    if q.shape != (x.size, y.size, z.size, 11):
        raise ValueError("air5 HBL field and coordinate dimensions do not match")
    if y.size < 7:
        raise ValueError("wall diagnostics require at least seven y points")
    if reference_density <= 0.0 or reference_velocity <= 0.0:
        raise ValueError("reference density and velocity must be positive")
    scale = np.maximum(np.max(np.abs(q), axis=2, keepdims=True), 1.0)
    spanwise_delta = np.max(np.abs(q - q[:, :, :1, :]) / scale)
    if spanwise_delta > spanwise_tolerance:
        raise ValueError(
            f"spanwise variation {spanwise_delta:.3e} exceeds the laminar A1 gate"
        )

    velocity, temperature, tv, pressure, mass_fraction = _recover_primitives(q, model)
    wall_weights = finite_difference_weights(y[:7], y[0])
    x_derivative = _derivative_matrix(x)

    wall_velocity = velocity[:, 0, :, :]
    velocity_scale = max(reference_velocity, 1.0)
    if np.max(np.abs(wall_velocity)) > 2.0e-12 * velocity_scale:
        raise ValueError("air5 HBL wall diagnostics require a no-slip wall state")

    dvelocity_dy = np.einsum("j,ijzc->izc", wall_weights, velocity[:, :7, :, :])
    dtemperature_dy = np.einsum("j,ijz->iz", wall_weights, temperature[:, :7, :])
    dtv_dy = np.einsum("j,ijz->iz", wall_weights, tv[:, :7, :])
    dvelocity_dx = np.einsum("li,ijzc->ljzc", x_derivative, velocity)[:, 0, :, :]

    skin_friction = np.empty((x.size, z.size))
    heat_total = np.empty((x.size, z.size))
    heat_tr = np.empty((x.size, z.size))
    heat_v = np.empty((x.size, z.size))
    heat_species = np.empty((x.size, z.size))
    dynamic_pressure_twice = reference_density * reference_velocity**2
    for i in range(x.size):
        for k in range(z.size):
            gradients = np.zeros((3, 3))
            gradients[:, 0] = dvelocity_dx[i, k]
            gradients[:, 1] = dvelocity_dy[i, k]
            gradient_temperature = np.array([0.0, dtemperature_dy[i, k], 0.0])
            gradient_tv = np.array([0.0, dtv_dy[i, k], 0.0])
            # The fixed-air5 HBL lower wall is noncatalytic.  Match the solver's
            # imposed Js,n=0 boundary flux instead of differentiating the nearby
            # cell-center composition profile.
            gradient_mass_fraction = np.zeros((5, 3))
            wall_y = mass_fraction[i, 0, k]
            flux = transport.diffusive_flux(
                rho=q[i, 0, k, 0],
                velocity=wall_velocity[i, k],
                temperature=temperature[i, 0, k],
                tv=tv[i, 0, k],
                pressure=pressure[i, 0, k],
                grad_velocity=gradients,
                grad_temperature=gradient_temperature,
                grad_tv=gradient_tv,
                mass_fraction=wall_y,
                grad_mass_fraction=gradient_mass_fraction,
            )
            _, conductivity_tr, conductivity_v, _ = transport.transport_properties(
                temperature[i, 0, k], tv[i, 0, k], pressure[i, 0, k], wall_y
            )
            vibrational_energy = transport.species_vibrational_energy(tv[i, 0, k])
            enthalpy = (
                transport.cv_tr * temperature[i, 0, k]
                + vibrational_energy
                + transport.formation_energy
                + transport.gas_constant * temperature[i, 0, k]
            )
            skin_friction[i, k] = 2.0 * flux.momentum_flux[0, 1] / dynamic_pressure_twice
            heat_tr[i, k] = -conductivity_tr * dtemperature_dy[i, k]
            heat_v[i, k] = -conductivity_v * dtv_dy[i, k]
            heat_species[i, k] = float(enthalpy @ flux.species_flux[:, 1])
            heat_total[i, k] = heat_tr[i, k] + heat_v[i, k] + heat_species[i, k]

    profile_x = np.asarray(profile_stations, dtype=np.float64)
    if profile_x.ndim != 1 or profile_x.size == 0 or not np.all(np.isfinite(profile_x)):
        raise ValueError("at least one finite profile station is required")
    profile_velocity = []
    profile_temperature = []
    profile_tv = []
    profile_mass_fraction = []
    for station in profile_x:
        profile_velocity.append(
            _spanwise_average(_interpolate_station(velocity, x, station).transpose(0, 2, 1), z)
        )
        profile_temperature.append(
            _spanwise_average(_interpolate_station(temperature, x, station), z)
        )
        profile_tv.append(_spanwise_average(_interpolate_station(tv, x, station), z))
        profile_mass_fraction.append(
            _spanwise_average(
                _interpolate_station(mass_fraction, x, station).transpose(0, 2, 1), z
            )
        )

    return Air5HblDiagnostics(
        wall_x=x.copy(),
        skin_friction=_spanwise_average(skin_friction, z),
        heat_flux_total=_spanwise_average(heat_total, z),
        heat_flux_translational=_spanwise_average(heat_tr, z),
        heat_flux_vibrational=_spanwise_average(heat_v, z),
        heat_flux_species=_spanwise_average(heat_species, z),
        profile_x=profile_x,
        profile_y=y.copy(),
        profile_velocity=np.asarray(profile_velocity),
        profile_temperature=np.asarray(profile_temperature),
        profile_tv=np.asarray(profile_tv),
        profile_mass_fraction=np.asarray(profile_mass_fraction),
    )


def _comma_separated_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(","))
    if not values or not all(np.isfinite(values)):
        raise argparse.ArgumentTypeError("expected comma-separated finite values")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--mechanism", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--lengths", required=True, type=_comma_separated_floats)
    parser.add_argument("--stations", required=True, type=_comma_separated_floats)
    parser.add_argument("--reference-density", required=True, type=float)
    parser.add_argument("--reference-velocity", required=True, type=float)
    args = parser.parse_args()
    if len(args.lengths) != 3 or any(value <= 0.0 for value in args.lengths):
        parser.error("--lengths requires three positive values")

    q = load_active_q_snapshot(args.snapshot)
    x = np.linspace(0.0, args.lengths[0], q.shape[0])
    y = np.linspace(0.0, args.lengths[1], q.shape[1])
    z = np.linspace(0.0, args.lengths[2], q.shape[2])
    result = analyze_cartesian_hbl(
        q,
        x,
        y,
        z,
        model=Air5RadauReference(args.mechanism),
        transport=Air5TransportReference(args.mechanism),
        reference_density=args.reference_density,
        reference_velocity=args.reference_velocity,
        profile_stations=args.stations,
    )
    save_diagnostics(result, args.archive, args.report)
    print(args.report.read_text(encoding="ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
