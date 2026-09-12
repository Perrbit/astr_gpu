from __future__ import annotations

import sqlite3
from pathlib import Path

from tests.gpu_validation.analyze_curvilinear_nscbc52_nsys import analyze


def write_profile(path: Path, *, rhs_count: int, forbidden: str | None = None) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(demangledName INTEGER)"
        )
        connection.execute(
            "INSERT INTO StringIds VALUES(1, 'boundary_gpu_nscbc_farfield_y_upper_nonreflecting_rhs_kernel_')"
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(1)",
            [()] * rhs_count,
        )
        if forbidden is not None:
            connection.execute("INSERT INTO StringIds VALUES(2, ?)", (forbidden,))
            connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(2)")


def test_analyze_accepts_one_rhs_launch_per_rk_substage(tmp_path: Path) -> None:
    profile = tmp_path / "profile.sqlite"
    write_profile(profile, rhs_count=6)
    passed, lines = analyze(profile, rk_steps=2)
    assert passed
    assert "observed_nonreflecting_rhs_launches: 6" in lines


def test_analyze_rejects_legacy_filter_work(tmp_path: Path) -> None:
    profile = tmp_path / "profile.sqlite"
    write_profile(
        profile,
        rhs_count=6,
        forbidden="boundary_gpu_nscbc_farfield_y_upper_filter_x_kernel_",
    )
    passed, lines = analyze(profile, rk_steps=2)
    assert not passed
    assert "forbidden_nscbc_farfield_y_upper_filter_x_kernel: 1" in lines


def test_analyze_rejects_missing_rhs_launch(tmp_path: Path) -> None:
    profile = tmp_path / "profile.sqlite"
    write_profile(profile, rhs_count=5)
    passed, lines = analyze(profile, rk_steps=2)
    assert not passed
    assert "expected_nonreflecting_rhs_launches: 6" in lines
