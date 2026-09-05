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
    allowed_d2h_kernel_queries: tuple[str, ...] = (),
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
        large_transfers = connection.execute(
            """
            SELECT memcpy.start, memcpy.bytes, memcpy.copyKind,
                   (
                     SELECT strings.value
                     FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel
                     JOIN StringIds AS strings ON strings.id = kernel.demangledName
                     WHERE kernel.end <= memcpy.start
                     ORDER BY kernel.end DESC
                     LIMIT 1
                   ) AS previous_kernel
            FROM CUPTI_ACTIVITY_KIND_MEMCPY AS memcpy
            WHERE memcpy.start >= ?
              AND memcpy.copyKind IN (1, 2, 11, 12)
              AND memcpy.bytes >= ?
            ORDER BY memcpy.start
            """,
            (start, large_transfer_bytes),
        ).fetchall()

    allowed_large_d2h = []
    forbidden_large = []
    for transfer_start, transfer_bytes, copy_kind, previous_kernel in large_transfers:
        previous_name = "" if previous_kernel is None else str(previous_kernel)
        allowed = copy_kind in D2H_KINDS and any(
            query in previous_name for query in allowed_d2h_kernel_queries
        )
        record = (int(transfer_start), int(transfer_bytes), int(copy_kind), previous_name)
        if allowed:
            allowed_large_d2h.append(record)
        else:
            forbidden_large.append(record)

    passed = kernel_count > 0 and not forbidden_large
    lines = [
        f"status: {'pass' if passed else 'fail'}",
        f"start_kernel_query: {start_kernel}",
        f"matched_kernel: {matched_name}",
        f"matched_kernel_occurrences: {kernel_matches}",
        f"start_timestamp_ns: {start}",
        f"kernels_after_start: {kernel_count}",
        f"large_transfer_threshold_bytes: {large_transfer_bytes}",
        f"large_h2d_d2h_count: {len(large_transfers)}",
        f"allowed_large_d2h_count: {len(allowed_large_d2h)}",
        f"forbidden_large_h2d_d2h_count: {len(forbidden_large)}",
        "allowed_d2h_after_kernel_queries: "
        + (",".join(allowed_d2h_kernel_queries) if allowed_d2h_kernel_queries else "none"),
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
    parser.add_argument(
        "--allow-d2h-after-kernel",
        action="append",
        default=[],
        help="Allow a large D2H transfer only when the immediately preceding kernel name contains this value",
    )
    args = parser.parse_args()

    passed, lines = analyze(
        args.input,
        args.start_kernel,
        args.large_transfer_bytes,
        tuple(args.allow_d2h_after_kernel),
    )
    report = "\n".join(lines) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="ascii")
    print(report, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
