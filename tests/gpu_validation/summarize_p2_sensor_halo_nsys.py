#!/usr/bin/env python3
"""Summarize the shock-sensor halo interval from an Nsight Systems export."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import re
import sqlite3
import statistics


RAW_KERNEL = "shock_sensor_gpu_raw_shock_sensor_kernel_"
EXPANDED_KERNEL = "shock_sensor_gpu_expand_shock_sensor_kernel_"
RK_PATTERN = re.compile(
    r"^ASTR_GPU_RK_TIMING\s+\d+\s+\S+\s+\S+\s+(\S+)", re.MULTILINE
)


def format_ns(value: int) -> str:
    return f"{value} ns"


def sensor_intervals(
    connection: sqlite3.Connection,
) -> tuple[dict[int, list[int]], dict[int, list[tuple[int, int]]]]:
    rows = connection.execute(
        """
        SELECT k.deviceId, k.start, k.end, k.globalPid, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        JOIN StringIds AS s ON s.id = k.shortName
        WHERE s.value IN (?, ?)
        ORDER BY k.deviceId, k.start
        """,
        (RAW_KERNEL, EXPANDED_KERNEL),
    ).fetchall()
    by_device: dict[int, list[tuple[int, int, int, str]]] = defaultdict(list)
    for device, start, end, global_pid, name in rows:
        by_device[int(device)].append(
            (int(start), int(end), int(global_pid) >> 24, str(name))
        )

    intervals: dict[int, list[int]] = {}
    windows_by_process: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for device, events in by_device.items():
        pending_end: int | None = None
        process_id: int | None = None
        values: list[int] = []
        for start, end, event_process, name in events:
            if name == RAW_KERNEL:
                if pending_end is not None:
                    raise ValueError(f"device {device}: raw sensor kernels are not paired")
                pending_end = end
                process_id = event_process
            elif pending_end is not None:
                if event_process != process_id:
                    raise ValueError(f"device {device}: kernel process changed within pair")
                if start < pending_end:
                    raise ValueError(f"device {device}: negative sensor halo interval")
                values.append(start - pending_end)
                windows_by_process[event_process].append((pending_end, start))
                pending_end = None
        if pending_end is not None:
            raise ValueError(f"device {device}: missing expanded sensor kernel")
        if values:
            intervals[device] = values
    if not intervals:
        raise ValueError("no paired shock-sensor kernels found")
    counts = {len(values) for values in intervals.values()}
    if len(counts) != 1:
        raise ValueError("devices have different shock-sensor interval counts")
    return intervals, windows_by_process


def matching_mpi_events(
    connection: sqlite3.Connection,
    windows_by_process: dict[int, list[tuple[int, int]]],
    message_bytes: int,
) -> tuple[int, int]:
    events = connection.execute(
        "SELECT start, end, globalTid FROM MPI_P2P_EVENTS WHERE size = ?",
        (message_bytes,),
    ).fetchall()
    matched = []
    for start, end, global_tid in events:
        process_id = int(global_tid) >> 24
        if any(
            int(start) >= window_start and int(end) <= window_end
            for window_start, window_end in windows_by_process.get(process_id, [])
        ):
            matched.append(int(end) - int(start))
    return len(matched), sum(matched)


def parse_rk_total_ns(path: Path) -> int:
    values = [float(value) for value in RK_PATTERN.findall(path.read_text(encoding="utf-8"))]
    if not values:
        raise ValueError("runtime log contains no ASTR_GPU_RK_TIMING records")
    return round(sum(values) * 1.0e9)


def write_report(
    path: Path,
    intervals: dict[int, list[int]],
    rk_total_ns: int,
    sensor_message_bytes: int,
    matching_mpi_ns: int,
    matching_mpi_instances: int,
) -> None:
    count = len(next(iter(intervals.values())))
    critical_total = sum(max(values[index] for values in intervals.values()) for index in range(count))
    fraction = 100.0 * critical_total / rk_total_ns
    lines = [
        "# P2 Shock-Sensor Halo Timeline",
        "",
        f"- devices: `{len(intervals)}`",
        f"- sensor exchanges per device: `{count}`",
        f"- sensor message size: `{sensor_message_bytes} bytes`",
        f"- critical-path halo total: `{format_ns(critical_total)}`",
        f"- profiled complete-RK total: `{format_ns(rk_total_ns)}`",
        f"- complete-RK upper-bound fraction: `{fraction:.3f}%`",
        f"- matching MPI instances: `{matching_mpi_instances}`",
        f"- matching MPI time: `{format_ns(matching_mpi_ns)}`",
        "",
        "| Device | Count | Median interval (ns) | Maximum interval (ns) | Total interval (ns) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for device, values in sorted(intervals.items()):
        lines.append(
            f"| {device} | {len(values)} | {statistics.median(values):.0f} | "
            f"{max(values)} | {sum(values)} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--runtime-log", required=True, type=Path)
    parser.add_argument("--sensor-message-bytes", required=True, type=int)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    with sqlite3.connect(args.input) as connection:
        intervals, windows_by_process = sensor_intervals(connection)
        instances, mpi_ns = matching_mpi_events(
            connection, windows_by_process, args.sensor_message_bytes
        )
    write_report(
        args.report,
        intervals,
        parse_rk_total_ns(args.runtime_log),
        args.sensor_message_bytes,
        int(mpi_ns),
        int(instances),
    )
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
