#!/usr/bin/env python3
"""Convert absolute ASTR precursor slices into a validated time-series inlet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


GAMMA = 1.4
PRANDTL = 0.72
SUTHERLAND_TEMPERATURE_K = 110.4
FIELDS = ("ro", "u1", "u2", "u3", "t")


def derivative_weights(nodes: np.ndarray, evaluation_point: float) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.ndim != 1 or nodes.size < 2 or not np.all(np.diff(nodes) > 0.0):
        raise ValueError("derivative nodes must be finite and strictly increasing")
    offsets = nodes - float(evaluation_point)
    matrix = np.vstack([offsets**degree for degree in range(nodes.size)])
    rhs = np.zeros(nodes.size, dtype=np.float64)
    rhs[1] = 1.0
    return np.linalg.solve(matrix, rhs)


def trapezoidal(values: np.ndarray, coordinates: np.ndarray) -> float:
    integrator = getattr(np, "trapezoid", None)
    if integrator is None:
        integrator = np.trapz
    return float(integrator(values, coordinates))


def sutherland_ratio(temperature: np.ndarray | float, reference_temperature: float) -> np.ndarray:
    temperature = np.asarray(temperature, dtype=np.float64)
    constant = SUTHERLAND_TEMPERATURE_K / reference_temperature
    return temperature**1.5 * (1.0 + constant) / (temperature + constant)


def delta_99(y: np.ndarray, velocity: np.ndarray, freestream_velocity: float) -> float:
    ratio = velocity / freestream_velocity
    indices = np.flatnonzero(ratio >= 0.99)
    if indices.size == 0:
        raise ValueError("mean velocity does not reach 99 percent of the freestream value")
    right = int(indices[0])
    if right == 0:
        return float(y[0])
    left = right - 1
    fraction = (0.99 - ratio[left]) / (ratio[right] - ratio[left])
    return float(y[left] + fraction * (y[right] - y[left]))


def normalized_autocorrelation(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[0] < 2:
        raise ValueError("autocorrelation requires a time-by-spanwise array")
    result = np.empty(signal.shape[0], dtype=np.float64)
    for lag in range(signal.shape[0]):
        left = signal[: signal.shape[0] - lag]
        right = signal[lag:]
        denominator = np.sqrt(np.sum(left * left) * np.sum(right * right))
        result[lag] = np.sum(left * right) / denominator if denominator > 0.0 else 0.0
    return result


def _read_slice(path: Path, mach: float) -> tuple[float, dict[str, np.ndarray], str]:
    with h5py.File(path, "r") as handle:
        required = ("time", "u1", "u2", "u3", "t")
        missing = [name for name in required if name not in handle]
        if missing:
            raise KeyError(f"{path}: missing datasets {missing}")
        time = float(np.asarray(handle["time"]).reshape(-1)[0])
        values = {name: np.asarray(handle[name], dtype=np.float64) for name in required[1:]}
        if "ro" in handle:
            values["ro"] = np.asarray(handle["ro"], dtype=np.float64)
            density_source = "dataset"
        elif "p" in handle:
            pressure = np.asarray(handle["p"], dtype=np.float64)
            values["ro"] = GAMMA * mach**2 * pressure / values["t"]
            density_source = "reconstructed_from_pressure_temperature"
        else:
            raise KeyError(f"{path}: need either ro or p to obtain density")
    shape = values["t"].shape
    if len(shape) != 2 or any(array.shape != shape for array in values.values()):
        raise ValueError(f"{path}: precursor fields must share one two-dimensional shape")
    if not np.isfinite(time) or any(not np.all(np.isfinite(array)) for array in values.values()):
        raise FloatingPointError(f"{path}: precursor slice contains non-finite data")
    if np.min(values["ro"]) <= 0.0 or np.min(values["t"]) <= 0.0:
        raise FloatingPointError(f"{path}: precursor density and temperature must be positive")
    return time, values, density_source


def _load_series(source_dir: Path, mach: float, y_size: int, periodic_atol: float):
    paths = sorted(source_dir.glob("islice*.h5"))
    if len(paths) < 4:
        raise ValueError("at least four precursor slices are required")
    times: list[float] = []
    records: list[dict[str, np.ndarray]] = []
    density_sources: set[str] = set()
    seam_linf = 0.0
    expected_shape = None
    for path in paths:
        time, values, density_source = _read_slice(path, mach)
        shape = values["t"].shape
        if shape[1] != y_size:
            raise ValueError(f"{path}: y extent {shape[1]} does not match grid extent {y_size}")
        if shape[0] < 3:
            raise ValueError(f"{path}: spanwise extent must include at least two intervals and an endpoint")
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError("all precursor slices must share the same shape")
        seam_linf = max(
            seam_linf,
            max(float(np.max(np.abs(values[name][0] - values[name][-1]))) for name in FIELDS),
        )
        times.append(time)
        records.append(values)
        density_sources.add(density_source)
    if seam_linf > periodic_atol:
        raise ValueError(
            f"precursor periodic seam error {seam_linf:.16e} exceeds {periodic_atol:.16e}"
        )
    times_array = np.asarray(times, dtype=np.float64)
    intervals = np.diff(times_array)
    if np.any(intervals <= 0.0):
        raise ValueError("precursor times must be strictly increasing")
    scale = max(1.0, float(np.max(np.abs(times_array))))
    if np.max(np.abs(intervals - intervals[0])) > 1.0e-12 * scale:
        raise ValueError("precursor times must be uniformly spaced for cubic interpolation")
    density_source = "mixed" if len(density_sources) > 1 else density_sources.pop()
    return paths, times_array, records, density_source, seam_linf


def _profile_metrics(
    y: np.ndarray,
    mean: dict[str, np.ndarray],
    reynolds: float,
    reference_temperature: float,
) -> dict[str, float]:
    rho_inf = float(mean["ro"][-1])
    u_inf = float(mean["u1"][-1])
    temperature_inf = float(mean["t"][-1])
    if rho_inf <= 0.0 or u_inf <= 0.0 or temperature_inf <= 0.0:
        raise ValueError("freestream density, velocity, and temperature must be positive")
    delta = delta_99(y, mean["u1"], u_inf)
    mass_velocity = mean["ro"] * mean["u1"] / (rho_inf * u_inf)
    displacement = trapezoidal(1.0 - mass_velocity, y)
    momentum = trapezoidal(mass_velocity * (1.0 - mean["u1"] / u_inf), y)
    stencil = min(7, y.size)
    dudy_wall = float(derivative_weights(y[:stencil], y[0]) @ mean["u1"][:stencil])
    if dudy_wall <= 0.0:
        raise ValueError("mean wall-normal velocity gradient must be positive")
    mu_wall = float(sutherland_ratio(mean["t"][0], reference_temperature))
    mu_inf = float(sutherland_ratio(temperature_inf, reference_temperature))
    u_tau = np.sqrt((mu_wall / reynolds) * dudy_wall / mean["ro"][0])
    re_theta = reynolds * rho_inf * u_inf * momentum / mu_inf
    re_tau = reynolds * mean["ro"][0] * u_tau * delta / mu_wall
    return {
        "delta_99": delta,
        "displacement_thickness": displacement,
        "momentum_thickness": momentum,
        "friction_velocity": float(u_tau),
        "re_theta": float(re_theta),
        "re_tau": float(re_tau),
    }


def _write_profile(path: Path, mean: dict[str, np.ndarray], metrics: dict[str, float], mach: float) -> None:
    pressure = mean["ro"] * mean["t"] / (GAMMA * mach**2)
    with path.open("w", encoding="ascii") as handle:
        handle.write("Turbulent precursor mean profile density=provided pressure=provided\n")
        handle.write("delta delta_star theta u_tau\n")
        handle.write(
            f"{metrics['delta_99']:.16e} {metrics['displacement_thickness']:.16e} "
            f"{metrics['momentum_thickness']:.16e} {metrics['friction_velocity']:.16e}\n"
        )
        handle.write("rho u v temperature pressure\n")
        for values in zip(mean["ro"], mean["u1"], mean["u2"], mean["t"], pressure):
            handle.write(" ".join(f"{float(value):.16e}" for value in values) + "\n")


def prepare(
    *,
    source_dir: Path,
    output_dir: Path,
    y: np.ndarray,
    mach: float,
    reynolds: float,
    reference_temperature: float,
    periodic_atol: float,
    target_re_theta: float | None = None,
    target_re_tau: float | None = None,
    relative_re_tolerance: float | None = None,
    mass_flow_relative_drift_max: float | None = None,
    correlation_lag: float | None = None,
    correlation_abs_max: float | None = None,
) -> dict:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    y = np.asarray(y, dtype=np.float64)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    if y.ndim != 1 or y.size < 6 or not np.all(np.isfinite(y)) or not np.all(np.diff(y) > 0.0):
        raise ValueError("y must be a finite, strictly increasing line with at least six points")
    if mach <= 0.0 or reynolds <= 0.0 or reference_temperature <= 0.0 or periodic_atol < 0.0:
        raise ValueError("Mach, Reynolds, reference temperature, and periodic tolerance are invalid")
    paths, times, records, density_source, seam_linf = _load_series(
        source_dir, mach, y.size, periodic_atol
    )
    stack = {name: np.stack([record[name][:-1] for record in records], axis=0) for name in FIELDS}
    mean = {name: np.mean(values, axis=(0, 1)) for name, values in stack.items()}
    # inlet.prof has no spanwise-mean w column; remove finite-window drift explicitly.
    fluctuations = {
        name: values - mean[name][None, None, :] for name, values in stack.items()
    }
    metrics = _profile_metrics(y, mean, reynolds, reference_temperature)

    rho = stack["ro"]
    favre_denominator = np.sum(rho, axis=(0, 1))
    favre_velocity = {
        name: np.sum(rho * stack[name], axis=(0, 1)) / favre_denominator
        for name in ("u1", "u2", "u3")
    }
    favre_stress = {}
    for left, right in (("u1", "u1"), ("u2", "u2"), ("u3", "u3"), ("u1", "u2")):
        key = f"{left}_{right}_favre"
        favre_stress[key] = (
            np.sum(rho * stack[left] * stack[right], axis=(0, 1)) / favre_denominator
            - favre_velocity[left] * favre_velocity[right]
        )

    mass_flow = np.asarray(
        [trapezoidal(np.mean(r * u, axis=0), y) for r, u in zip(stack["ro"], stack["u1"])],
        dtype=np.float64,
    )
    mass_flow_mean = float(np.mean(mass_flow))
    mass_flow_drift = float(np.max(np.abs(mass_flow - mass_flow_mean)) / abs(mass_flow_mean))
    probe_index = int(np.argmax(np.sqrt(np.mean(fluctuations["u1"] ** 2, axis=(0, 1)))))
    autocorrelation = normalized_autocorrelation(fluctuations["u1"][:, :, probe_index])

    gate_results: dict[str, dict[str, float | bool]] = {}
    if target_re_theta is not None or target_re_tau is not None or relative_re_tolerance is not None:
        if target_re_theta is None or target_re_tau is None or relative_re_tolerance is None:
            raise ValueError("Reynolds-number gating requires both targets and one relative tolerance")
        for name, target in (("re_theta", target_re_theta), ("re_tau", target_re_tau)):
            relative_error = abs(metrics[name] - target) / target
            gate_results[name] = {
                "value": metrics[name],
                "target": target,
                "relative_error": relative_error,
                "passed": relative_error <= relative_re_tolerance,
            }
    if mass_flow_relative_drift_max is not None:
        gate_results["mass_flow"] = {
            "relative_drift": mass_flow_drift,
            "limit": mass_flow_relative_drift_max,
            "passed": mass_flow_drift <= mass_flow_relative_drift_max,
        }
    correlation_lag_index = None
    if correlation_lag is not None or correlation_abs_max is not None:
        if correlation_lag is None or correlation_abs_max is None:
            raise ValueError("correlation gating requires both lag and absolute limit")
        correlation_lag_index = int(round(correlation_lag / (times[1] - times[0])))
        if correlation_lag_index < 1 or correlation_lag_index >= times.size:
            raise ValueError("requested correlation lag lies outside the sampled time series")
        correlation_value = float(autocorrelation[correlation_lag_index])
        gate_results["autocorrelation"] = {
            "lag": correlation_lag,
            "value": correlation_value,
            "absolute_limit": correlation_abs_max,
            "passed": abs(correlation_value) <= correlation_abs_max,
        }

    output_dir.mkdir(parents=True)
    inflow_dir = output_dir / "inflow"
    inflow_dir.mkdir()
    for index, record in enumerate(records):
        with h5py.File(inflow_dir / f"islice{index:05d}.h5", "w") as handle:
            handle.create_dataset("time", data=np.float64(times[index] - times[0]))
            for name in FIELDS:
                unique = fluctuations[name][index]
                periodic = np.concatenate((unique, unique[:1]), axis=0)
                handle.create_dataset(name, data=np.asarray(periodic, dtype=np.float64))
    _write_profile(output_dir / "inlet.prof", mean, metrics, mach)
    np.savez_compressed(
        output_dir / "profile_statistics.npz",
        y=y,
        times=times - times[0],
        mass_flow=mass_flow,
        autocorrelation=autocorrelation,
        rho_mean=mean["ro"],
        u_mean=mean["u1"],
        v_mean=mean["u2"],
        w_mean=mean["u3"],
        T_mean=mean["t"],
        u_favre=favre_velocity["u1"],
        v_favre=favre_velocity["u2"],
        w_favre=favre_velocity["u3"],
        **favre_stress,
    )
    status = "diagnostic_only"
    if gate_results:
        status = "pass" if all(bool(result["passed"]) for result in gate_results.values()) else "fail"
    report = {
        "status": status,
        "source_contract": "absolute primitive ASTR islice fields",
        "output_contract": "primitive fluctuations relative to inlet.prof",
        "slice_count": len(paths),
        "slice_interval": float(times[1] - times[0]),
        "density_source": density_source,
        "periodic_seam_linf": seam_linf,
        "minimum_density": float(np.min(stack["ro"])),
        "minimum_temperature": float(np.min(stack["t"])),
        "mass_flow_relative_drift": mass_flow_drift,
        "autocorrelation_probe_y_index": probe_index,
        "autocorrelation_gate_lag_index": correlation_lag_index,
        "profile_metrics": metrics,
        "gate_results": gate_results,
    }
    (output_dir / "turbulent_inflow_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="ascii"
    )
    return report


def read_y_line(grid_path: Path) -> np.ndarray:
    with h5py.File(grid_path, "r") as handle:
        if "y" not in handle:
            raise KeyError(f"{grid_path}: missing y dataset")
        values = np.asarray(handle["y"], dtype=np.float64)
    if values.ndim == 3:
        return values[0, :, 0]
    if values.ndim == 1:
        return values
    raise ValueError("grid y dataset must be one-dimensional or use ASTR (z,y,x) layout")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--mach", required=True, type=float)
    parser.add_argument("--reynolds", required=True, type=float)
    parser.add_argument("--reference-temperature", required=True, type=float)
    parser.add_argument("--periodic-atol", type=float, default=1.0e-10)
    parser.add_argument("--target-re-theta", type=float)
    parser.add_argument("--target-re-tau", type=float)
    parser.add_argument("--relative-re-tolerance", type=float)
    parser.add_argument("--mass-flow-relative-drift-max", type=float)
    parser.add_argument("--correlation-lag", type=float)
    parser.add_argument("--correlation-abs-max", type=float)
    args = parser.parse_args()
    report = prepare(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        y=read_y_line(args.grid),
        mach=args.mach,
        reynolds=args.reynolds,
        reference_temperature=args.reference_temperature,
        periodic_atol=args.periodic_atol,
        target_re_theta=args.target_re_theta,
        target_re_tau=args.target_re_tau,
        relative_re_tolerance=args.relative_re_tolerance,
        mass_flow_relative_drift_max=args.mass_flow_relative_drift_max,
        correlation_lag=args.correlation_lag,
        correlation_abs_max=args.correlation_abs_max,
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
