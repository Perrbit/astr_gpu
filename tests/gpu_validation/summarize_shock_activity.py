#!/usr/bin/env python3
"""Summarize active shock nodes and adjacent interfaces from a sensor dump."""

from __future__ import annotations

import argparse
from pathlib import Path

from compare_shock_sensor import read_sensor_dump_set, summarize_shock_activity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    dump = read_sensor_dump_set(args.sensor)
    activity = summarize_shock_activity(dump.mask)
    lines = [
        "# ASTR Shock Activity",
        "",
        f"- shape: `{dump.shape[0]},{dump.shape[1]},{dump.shape[2]}`",
        f"- active nodes: `{activity.active_nodes}/{activity.total_nodes}`",
        f"- active-node fraction: `{activity.node_fraction:.12e}`",
    ]
    for axis, active, total, fraction in zip(
        "xyz",
        activity.active_interfaces,
        activity.total_interfaces,
        activity.interface_fractions,
        strict=True,
    ):
        lines.append(f"- active {axis}-interfaces: `{active}/{total}` (`{fraction:.12e}`)")
    report = "\n".join(lines) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="ascii")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
