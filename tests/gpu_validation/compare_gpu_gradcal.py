#!/usr/bin/env python3
"""Check GPU gradcal dvel comparison summary."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_summary(path: Path) -> dict[str, str]:
    fpath = path / "gpu_gradcal_dvel_compare.dat" if path.is_dir() else path
    if not fpath.exists():
        raise FileNotFoundError(fpath)
    values: dict[str, str] = {}
    with fpath.open() as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                values[parts[0]] = parts[1]
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--atol", type=float, default=1.0e-10)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    args = parser.parse_args()

    values = parse_summary(args.gpu)
    max_abs = float(values["max_abs"])
    max_rel = float(values["max_rel"])
    status = 0 if max_abs <= args.atol or max_rel <= args.rtol else 1

    lines = [
        f"status: {'pass' if status == 0 else 'fail'}",
        f"atol: {args.atol:.16e}",
        f"rtol: {args.rtol:.16e}",
        f"max_abs: {max_abs:.16e}",
        f"max_rel: {max_rel:.16e}",
        f"source: {args.gpu}",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
