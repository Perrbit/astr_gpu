from __future__ import annotations

import sqlite3
from pathlib import Path

from tests.gpu_validation.analyze_nsys_rk_residency import analyze


def write_profile(path: Path, transfer_bytes: int) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, demangledName INTEGER)"
        )
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER, bytes INTEGER, copyKind INTEGER)"
        )
        connection.execute("INSERT INTO StringIds VALUES(1, 'target_rhs_kernel_')")
        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(100, 1)")
        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(90, 1048576, 1)")
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(110, ?, 2)", (transfer_bytes,)
        )


def test_analyze_ignores_large_startup_transfer(tmp_path: Path) -> None:
    profile = tmp_path / "profile.sqlite"
    write_profile(profile, transfer_bytes=6144)
    passed, lines = analyze(profile, "target_rhs_kernel", 65536)
    assert passed
    assert "large_h2d_d2h_count: 0" in lines
    assert "d2h_max_bytes: 6144" in lines


def test_analyze_rejects_large_rk_transfer(tmp_path: Path) -> None:
    profile = tmp_path / "profile.sqlite"
    write_profile(profile, transfer_bytes=65536)
    passed, lines = analyze(profile, "target_rhs_kernel", 65536)
    assert not passed
    assert "large_h2d_d2h_count: 1" in lines
