from __future__ import annotations

import sqlite3
from pathlib import Path

from tests.gpu_validation.analyze_nsys_rk_residency import analyze


def write_profile(
    path: Path,
    transfer_bytes: int,
    *,
    copy_kind: int = 2,
    kernel_name: str = "target_rhs_kernel_",
) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, demangledName INTEGER)"
        )
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER, bytes INTEGER, copyKind INTEGER)"
        )
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, nameId INTEGER)"
        )
        connection.execute("INSERT INTO StringIds VALUES(1, ?)", (kernel_name,))
        connection.execute("INSERT INTO StringIds VALUES(2, 'cudaDeviceSynchronize')")
        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(100, 105, 1)")
        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(106, 116, 2)")
        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(90, 1048576, 1)")
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(110, ?, ?)",
            (transfer_bytes, copy_kind),
        )


def test_analyze_ignores_large_startup_transfer(tmp_path: Path) -> None:
    profile = tmp_path / "profile.sqlite"
    write_profile(profile, transfer_bytes=6144)
    passed, lines = analyze(profile, "target_rhs_kernel", 65536)
    assert passed
    assert "large_h2d_d2h_count: 0" in lines
    assert "forbidden_large_h2d_d2h_count: 0" in lines
    assert "d2h_max_bytes: 6144" in lines
    assert "cuda_device_synchronize_count: 1" in lines
    assert "cuda_device_synchronize_total_ns: 10" in lines


def test_analyze_rejects_large_rk_transfer(tmp_path: Path) -> None:
    profile = tmp_path / "profile.sqlite"
    write_profile(profile, transfer_bytes=65536)
    passed, lines = analyze(profile, "target_rhs_kernel", 65536)
    assert not passed
    assert "large_h2d_d2h_count: 1" in lines
    assert "forbidden_large_h2d_d2h_count: 1" in lines


def test_analyze_allows_named_diagnostic_d2h(tmp_path: Path) -> None:
    profile = tmp_path / "profile.sqlite"
    write_profile(
        profile,
        transfer_bytes=524288,
        kernel_name="statistic_gpu_kenergy_partial_kernel_",
    )
    passed, lines = analyze(profile, "statistic_gpu", 65536, ("statistic_gpu_",))
    assert passed
    assert "large_h2d_d2h_count: 1" in lines
    assert "allowed_large_d2h_count: 1" in lines
    assert "forbidden_large_h2d_d2h_count: 0" in lines


def test_analyze_never_allows_h2d(tmp_path: Path) -> None:
    profile = tmp_path / "profile.sqlite"
    write_profile(
        profile,
        transfer_bytes=524288,
        copy_kind=1,
        kernel_name="statistic_gpu_kenergy_partial_kernel_",
    )
    passed, lines = analyze(profile, "statistic_gpu", 65536, ("statistic_gpu_",))
    assert not passed
    assert "allowed_large_d2h_count: 0" in lines
    assert "forbidden_large_h2d_d2h_count: 1" in lines
