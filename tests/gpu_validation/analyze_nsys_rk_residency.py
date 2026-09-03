#!/usr/bin/env python3
"""Audit host/device transfers after a selected RK kernel in an Nsight SQLite export."""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path


H2D_KINDS = (1, 11)
D2H_KINDS = (2, 12)


@dataclass(frozen=True)
class TransferSummary:
    count: int
    total_bytes: int
    max_bytes: int


def summarize_transfers(
    connection: sqlite3.Connection, start: int, kinds: tuple[int, ...]
) -> TransferSummary:
    placeholders = ",".join("?" for _ in kinds)
    row = connection.execute(
        f"""
        SELECT COUNT(*), COALESCE(SUM(bytes), 0), COALESCE(MAX(bytes), 0)
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE start >= ? AND copyKind IN ({placeholders})
        """,
        (start, *kinds),
    ).fetchone()
    assert row is not None
    return TransferSummary(*(int(value) for value in row))


def analyze(
    input_path: Path,
    start_kernel: str,
    large_transfer_bytes: int,
) -> tuple[bool, list[str]]:
    if large_transfer_bytes <= 0:
        raise ValueError("large-transfer-bytes must be positive")
    with sqlite3.connect(input_path) as connection:
        matches = connection.execute(
            """
            SELECT MIN(kernel.start), COUNT(*), MIN(strings.value)
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel
            JOIN StringIds AS strings ON strings.id = kernel.demangledName
            WHERE strings.value LIKE ?
            """,
            (f"%{start_kernel}%",),
        ).fetchone()
        if matches is None or matches[0] is None:
            raise ValueError(f"start kernel not found: {start_kernel}")
        start, kernel_matches, matched_name = int(matches[0]), int(matches[1]), str(matches[2])
        h2d = summarize_transfers(connection, start, H2D_KINDS)
        d2h = summarize_transfers(connection, start, D2H_KINDS)
        kernel_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start >= ?", (start,)
            ).fetchone()[0]
        )
        large_count = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY
                WHERE start >= ? AND copyKind IN (1, 2, 11, 12) AND bytes >= ?
                """,
                (start, large_transfer_bytes),
            ).fetchone()[0]
        )

    passed = kernel_count > 0 and large_count == 0
    lines = [
        f"status: {'pass' if passed else 'fail'}",
        f"start_kernel_query: {start_kernel}",
        f"matched_kernel: {matched_name}",
        f"matched_kernel_occurrences: {kernel_matches}",
        f"start_timestamp_ns: {start}",
        f"kernels_after_start: {kernel_count}",
        f"large_transfer_threshold_bytes: {large_transfer_bytes}",
        f"large_h2d_d2h_count: {large_count}",
        f"h2d_count: {h2d.count}",
        f"h2d_total_bytes: {h2d.total_bytes}",
        f"h2d_max_bytes: {h2d.max_bytes}",
        f"d2h_count: {d2h.count}",
        f"d2h_total_bytes: {d2h.total_bytes}",
        f"d2h_max_bytes: {d2h.max_bytes}",
    ]
    return passed, lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--start-kernel", required=True)
    parser.add_argument("--large-transfer-bytes", type=int, default=65536)
    args = parser.parse_args()

    passed, lines = analyze(args.input, args.start_kernel, args.large_transfer_bytes)
    report = "\n".join(lines) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="ascii")
    print(report, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
