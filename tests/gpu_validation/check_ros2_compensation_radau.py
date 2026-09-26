"""Check emitted ASTR CPU/GPU endpoints against the existing 0D Radau reference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from air5_radau_reference import Air5RadauReference

ROOT = Path(__file__).resolve().parents[2]


def check(log: Path) -> dict:
    rows = [line.split() for line in log.read_text().splitlines()]
    inputs, endpoints = {}, {}
    for row in rows:
        if not row:
            continue
        if row[0] == "RADAU_INPUT":
            case = int(row[1])
            if case in inputs or len(row) != 14:
                raise ValueError("duplicate or malformed input")
            inputs[case] = np.array(list(map(float, row[2:])))
        elif row[0] in ("RADAU_CPU", "RADAU_GPU"):
            key = (int(row[1]), row[0][6:], int(row[2]))
            if key in endpoints or len(row) != 9:
                raise ValueError("duplicate or malformed endpoint")
            endpoints[key] = np.array(list(map(float, row[3:])))
    expected = {(case, backend, mode) for case in (1, 2)
                for backend in ("CPU", "GPU") for mode in (0, 1)}
    if set(inputs) != {1, 2} or set(endpoints) != expected:
        raise ValueError("missing or extra comparison cases")
    mechanism = ROOT / "chemMech/air5_kimjo12.json"
    reference = Air5RadauReference(mechanism)
    results = []
    for case, values in sorted(inputs.items()):
        if not np.all(np.isfinite(values)):
            raise ValueError("nonfinite initial fixture")
        rho, momentum, q5 = values[0], values[1:4], values[4]
        initial, duration = values[5:11], values[11]
        atol = np.r_[np.full(5, 1e-13 * rho), 1e-13 * max(abs(initial[5]), 1)]
        trajectory = reference.integrate_radau(
            rho, momentum, q5, initial, duration, rtol=1e-11, atol=atol,
        )
        if not trajectory.success or trajectory.t[-1] != duration:
            raise ValueError("reference did not reach requested time")
        target = trajectory.y[:, -1]
        target_t = np.array(reference.temperatures(rho, momentum, q5, target[:, None])).ravel()
        for backend in ("CPU", "GPU"):
            for mode in (0, 1):
                state = endpoints[(case, backend, mode)]
                if not np.all(np.isfinite(state)) or np.any(state < 0):
                    raise ValueError("nonfinite or negative endpoint")
                species_ratio = np.max(abs(state[:5] / rho - target[:5] / rho) /
                                       (1e-10 + 1e-6 * abs(target[:5] / rho)))
                temperatures = np.array(reference.temperatures(rho, momentum, q5, state[:, None])).ravel()
                temperature_ratio = np.max(abs(temperatures - target_t) / (1e-7 * abs(target_t)))
                invariant = reference.invariant_drifts(initial, state)
                closure = abs(sum(state[:5] / rho) - 1.0)
                energy = reference.q5_from_state(rho, momentum, state[:5], state[5], temperatures[0])
                energy_error = abs(energy - q5) / max(abs(q5), 1.0)
                passed = (species_ratio <= 1 and temperature_ratio <= 1 and
                          max(invariant) <= 1e-14 and closure <= 128 * np.finfo(float).eps and
                          energy_error <= 32 * np.finfo(float).eps)
                results.append(dict(case=case, backend=backend, compensation=bool(mode),
                                    species_ratio=float(species_ratio), temperature_ratio=float(temperature_ratio),
                                    invariant_drifts=list(invariant), closure=float(closure),
                                    q5_eos_roundtrip=float(energy_error), passed=bool(passed)))
    return dict(status="passed" if all(row["passed"] for row in results) else "failed",
                scope="zero-dimensional chemistry only; no full-flow qualification",
                log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                mechanism_sha256=hashlib.sha256(mechanism.read_bytes()).hexdigest(), comparisons=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = check(args.log)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)
