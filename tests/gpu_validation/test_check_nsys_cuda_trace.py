#!/usr/bin/env python3
"""Tests for strict Nsight Systems CUDA-event admission."""

from pathlib import Path
import sqlite3
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "tests/gpu_validation/check_nsys_cuda_trace.py"
PROBE_JOB = ROOT / "tests/gpu_validation/run_zhongke_a800_profiler_probe.sbatch"


def run_checker(database: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), str(database)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_rejects_export_without_cuda_kernel_table(tmp_path: Path) -> None:
    database = tmp_path / "missing.sqlite"
    sqlite3.connect(database).close()

    completed = run_checker(database)

    assert completed.returncode != 0
    assert "CUPTI_ACTIVITY_KIND_KERNEL" in completed.stderr


def test_rejects_empty_cuda_kernel_table(tmp_path: Path) -> None:
    database = tmp_path / "empty.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER)")

    completed = run_checker(database)

    assert completed.returncode != 0
    assert "zero CUDA kernel events" in completed.stderr


def test_accepts_nonempty_cuda_kernel_table(tmp_path: Path) -> None:
    database = tmp_path / "valid.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER)")
        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1)")

    completed = run_checker(database)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "cuda_kernel_events=1"


def test_platform_probe_uses_strict_cuda_event_checker() -> None:
    source = PROBE_JOB.read_text(encoding="ascii")

    assert "check_nsys_cuda_trace.py" in source
    assert '"$OUT/${label}.sqlite"' in source
