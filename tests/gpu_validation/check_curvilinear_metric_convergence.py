#!/usr/bin/env python3
"""Require curvilinear metric errors to decrease under grid refinement."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def errors_decrease(values: list[float]) -> bool:
    return len(values) >= 3 and all(math.isfinite(value) for value in values) and all(
        fine < coarse for coarse, fine in zip(values[:-1], values[1:], strict=True)
    )


def report_value(path: Path, key: str) -> float:
    prefix = f"{key}="
    for line in path.read_text().splitlines():
        if line.startswith(prefix):
            return float(line[len(prefix) :])
    raise KeyError(f"{path}: missing {key}")


def report_norm_linf(path: Path, name: str) -> float:
    prefix = f"{name} linf="
    for line in path.read_text().splitlines():
        if line.startswith(prefix):
            return float(line[len(prefix) :].split()[0])
    raise KeyError(f"{path}: missing {name} linf")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    jacobian_errors = [report_value(path, "jacob_error_max") for path in args.reports]
    inverse_errors = [report_value(path, "dxi_error_max") for path in args.reports]
    identity_residuals = [report_value(path, "metric_identity_max") for path in args.reports]
    component_errors = {
        f"dxi{i}{j}": [report_norm_linf(path, f"dxi{i}{j}") for path in args.reports]
        for i in range(1, 4)
        for j in range(1, 4)
    }
    passed = errors_decrease(jacobian_errors) and errors_decrease(inverse_errors)
    passed = passed and all(errors_decrease(values) for values in component_errors.values())
    passed = passed and all(math.isfinite(value) for value in identity_residuals)

    lines = [f"status: {'pass' if passed else 'fail'}"]
    for index, path in enumerate(args.reports):
        lines.append(
            f"report={path} jacob_error_max={jacobian_errors[index]:.16e} "
            f"dxi_error_max={inverse_errors[index]:.16e} "
            f"metric_identity_max={identity_residuals[index]:.16e}"
        )
    for name, values in component_errors.items():
        lines.append(f"{name}_errors=" + ",".join(f"{value:.16e}" for value in values))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
