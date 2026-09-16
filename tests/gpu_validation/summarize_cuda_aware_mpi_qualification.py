#!/usr/bin/env python3
"""Summarize fail-closed device-aware MPI qualification records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED_SHAPES = {"513x257", "513x513"}
HALO_WIDTHS = (5, 6)
COMPONENT_COUNTS = (1, 3, 5, 6, 9)
TRANSPORTS = {
    "ucx_ipc": "UCX self,sm,cuda_copy,cuda_ipc with verified cuda_ipc selection",
    "ucx_no_ipc": "UCX self,sm,cuda_copy with cuda_ipc excluded",
}


def expected_message_bytes() -> dict[str, list[int]]:
    sizes: dict[str, list[int]] = {}
    for shape in EXPECTED_SHAPES:
        ny, nz = (int(value) for value in shape.split("x"))
        sizes[shape] = sorted(
            {
                width * ny * nz * nvar * 8
                for width in HALO_WIDTHS
                for nvar in COMPONENT_COUNTS
            }
        )
    return sizes


def summarize_qualification(
    records: list[dict[str, str]], mpi_stack: str = "unknown"
) -> dict[str, object]:
    configs = sorted({record["config"] for record in records})
    details: dict[str, dict[str, object]] = {}
    payload_status: dict[str, str] = {}
    protocol_status: dict[str, str] = {}
    sanitizer_status: dict[str, str] = {}
    qualified_configs = []
    for config in configs:
        selected = [record for record in records if record["config"] == config]
        payloads = [record for record in selected if record["kind"] == "payload"]
        protocols = [record for record in selected if record["kind"] == "protocol"]
        sanitizers = [record for record in selected if record["kind"] == "sanitizer"]
        payload_shapes = {record["shape"] for record in payloads}
        payload_pass = (
            payload_shapes == EXPECTED_SHAPES
            and len(payloads) == len(EXPECTED_SHAPES)
            and all(record["status"] == "PASS" for record in payloads)
        )
        protocol_pass = len(protocols) == 1 and protocols[0]["status"] == "PASS"
        sanitizer_pass = (
            len(sanitizers) == 1 and sanitizers[0]["status"] == "PASS"
        )
        qualified = payload_pass and protocol_pass and sanitizer_pass
        payload_status[config] = "PASS" if payload_pass else "FAIL"
        protocol_status[config] = "PASS" if protocol_pass else "FAIL"
        sanitizer_status[config] = "PASS" if sanitizer_pass else "FAIL"
        details[config] = {
            "payload_pass": payload_pass,
            "protocol_pass": protocol_pass,
            "sanitizer_pass": sanitizer_pass,
            "qualified": qualified,
            "records": selected,
        }
        if qualified:
            qualified_configs.append(config)
    return {
        "qualified": bool(qualified_configs),
        "production_qualified": "ucx_ipc" in qualified_configs,
        "qualified_configs": qualified_configs,
        "mpi_stack": mpi_stack,
        "transport": {config: TRANSPORTS.get(config, "unknown") for config in configs},
        "message_bytes": expected_message_bytes(),
        "payload_status": payload_status,
        "protocol_status": protocol_status,
        "sanitizer_status": sanitizer_status,
        "expected_payload_shapes": sorted(EXPECTED_SHAPES),
        "configs": details,
    }


def read_records(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = ["config", "shape", "kind", "status", "exit_code", "log"]
        if reader.fieldnames != expected:
            raise ValueError(f"unexpected result columns: {reader.fieldnames}")
        records = list(reader)
    if not records:
        raise ValueError("qualification result table is empty")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mpi-stack", default="unknown")
    args = parser.parse_args()

    summary = summarize_qualification(read_records(args.results), args.mpi_stack)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["production_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
