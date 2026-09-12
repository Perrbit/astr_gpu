#!/usr/bin/env python3
"""Compare CPU and GPU characteristic-policy probe records."""

from pathlib import Path
import argparse

import numpy as np


def records(path: Path, prefix: str) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    parsed = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or fields[0] != prefix:
            continue
        if len(fields) != 17:
            raise ValueError(f"malformed {prefix} record: {line}")
        parsed[fields[1]] = (
            np.asarray(fields[2:7], dtype=float),
            np.asarray(fields[7:12], dtype=int),
            np.asarray(fields[12:17], dtype=float),
        )
    if len(parsed) != 6:
        raise ValueError(f"expected six {prefix} records in {path}, got {len(parsed)}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    args = parser.parse_args()

    cpu = records(args.cpu, "NSCBC_CPU_POLICY")
    gpu = records(args.gpu, "NSCBC_GPU_POLICY")
    if cpu.keys() != gpu.keys():
        raise SystemExit(f"case mismatch: CPU={sorted(cpu)} GPU={sorted(gpu)}")

    worst = 0.0
    for label in cpu:
        cpu_lambda, cpu_mask, cpu_lodi = cpu[label]
        gpu_lambda, gpu_mask, gpu_lodi = gpu[label]
        if not np.array_equal(cpu_mask, gpu_mask):
            raise SystemExit(f"mask mismatch for {label}: {cpu_mask} != {gpu_mask}")
        error = max(
            float(np.max(np.abs(cpu_lambda - gpu_lambda))),
            float(np.max(np.abs(cpu_lodi - gpu_lodi))),
        )
        worst = max(worst, error)
        if error > args.atol:
            raise SystemExit(f"value mismatch for {label}: {error:.16e} > {args.atol:.16e}")
    print(f"NSCBC_POLICY_COMPARE_PASS max_abs={worst:.16e} atol={args.atol:.16e}")


if __name__ == "__main__":
    main()
