#!/usr/bin/env python3
"""Fail unless an Nsight Systems SQLite export contains CUDA kernels."""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3


KERNEL_TABLE = "CUPTI_ACTIVITY_KIND_KERNEL"


def count_cuda_kernel_events(database: Path) -> int:
    if not database.is_file():
        raise ValueError(f"Nsight SQLite export does not exist: {database}")

    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if KERNEL_TABLE not in tables:
            raise ValueError(f"Nsight export lacks {KERNEL_TABLE}")
        count = connection.execute(
            f'SELECT COUNT(*) FROM "{KERNEL_TABLE}"'
        ).fetchone()[0]

    if count <= 0:
        raise ValueError("Nsight export contains zero CUDA kernel events")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    args = parser.parse_args()

    try:
        count = count_cuda_kernel_events(args.database)
    except (OSError, sqlite3.Error, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"cuda_kernel_events={count}")


if __name__ == "__main__":
    main()
