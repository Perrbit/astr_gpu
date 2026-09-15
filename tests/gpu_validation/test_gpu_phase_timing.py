#!/usr/bin/env python3
"""Contracts for strict per-rank GPU RK phase timing summaries."""

from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys

import pytest

from summarize_gpu_phase_timing import PHASE_ORDER, parse_phase_log, summarize_logs


PHASE_COUNTS = {
    "prepare": 1,
    "filter": 3,
    "solution_halo": 3,
    "convection": 3,
    "diffusion_flux": 3,
    "diffusion_halo": 3,
    "diffusion_rhs": 3,
    "rk_update": 3,
}


def valid_log(ranks: int = 2, steps: int = 1, scale: float = 1.0) -> str:
    lines = ["unrelated solver output is ignored"]
    for step in range(steps):
        for rank in range(ranks):
            lines.append(
                f"ASTR_GPU_PHASE_TIMING {rank} {step} 0 prepare "
                f"{scale * (rank + 1):.6f}"
            )
            for rkstep in range(1, 4):
                for index, label in enumerate(tuple(PHASE_COUNTS)[1:], start=1):
                    value = scale * (100 * rank + 10 * rkstep + index)
                    lines.append(
                        f"ASTR_GPU_PHASE_TIMING {rank} {step} {rkstep} "
                        f"{label} {value:.6f}"
                    )
    return "\n".join(lines)


def timing_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("ASTR_GPU_PHASE_TIMING")]


def test_parser_uses_slowest_rank_for_each_exact_occurrence() -> None:
    samples = parse_phase_log(valid_log(steps=2), ranks=2, steps=2)

    assert samples["prepare"] == [2.0, 2.0]
    assert samples["filter"] == [111.0, 121.0, 131.0] * 2
    assert all(
        len(samples[label]) == 2 * count for label, count in PHASE_COUNTS.items()
    )


@pytest.mark.parametrize(
    "replacement",
    [
        "ASTR_GPU_PHASE_TIMING 0 0 0 prepare",
        "ASTR_GPU_PHASE_TIMING 0 0 0 prepare 1.0 extra",
        "ASTR_GPU_PHASE_TIMING rank 0 0 prepare 1.0",
        "ASTR_GPU_PHASE_TIMING 0 0 0 prepare seconds",
    ],
)
def test_parser_rejects_malformed_records(replacement: str) -> None:
    lines = timing_lines(valid_log())
    lines[0] = replacement

    with pytest.raises(ValueError, match="invalid phase timing record"):
        parse_phase_log("\n".join(lines), ranks=2, steps=1)


def test_parser_rejects_duplicate_records() -> None:
    lines = timing_lines(valid_log())
    lines.append(lines[0])

    with pytest.raises(ValueError, match="duplicate phase record"):
        parse_phase_log("\n".join(lines), ranks=2, steps=1)


def test_parser_rejects_missing_records() -> None:
    lines = timing_lines(valid_log())
    lines.pop()

    with pytest.raises(ValueError, match="missing phase record"):
        parse_phase_log("\n".join(lines), ranks=2, steps=1)


def test_parser_rejects_unknown_labels() -> None:
    lines = timing_lines(valid_log())
    lines[0] = lines[0].replace("prepare", "other")

    with pytest.raises(ValueError, match="unknown phase label"):
        parse_phase_log("\n".join(lines), ranks=2, steps=1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (1, "-1"),
        (1, "2"),
        (2, "-1"),
        (2, "1"),
    ],
)
def test_parser_rejects_rank_or_step_out_of_range(field: int, value: str) -> None:
    lines = timing_lines(valid_log())
    fields = lines[0].split()
    fields[field] = value
    lines[0] = " ".join(fields)

    with pytest.raises(ValueError, match="outside expected range"):
        parse_phase_log("\n".join(lines), ranks=2, steps=1)


@pytest.mark.parametrize(
    ("line_index", "rkstep", "label"),
    [
        (0, "1", "prepare"),
        (1, "0", "filter"),
        (1, "4", "filter"),
        (1, "-1", "filter"),
    ],
)
def test_parser_rejects_invalid_phase_occurrences(
    line_index: int, rkstep: str, label: str
) -> None:
    lines = timing_lines(valid_log())
    fields = lines[line_index].split()
    fields[3] = rkstep
    fields[4] = label
    lines[line_index] = " ".join(fields)

    with pytest.raises(ValueError, match="invalid phase occurrence"):
        parse_phase_log("\n".join(lines), ranks=2, steps=1)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-0.001"])
def test_parser_rejects_non_finite_or_negative_times(value: str) -> None:
    lines = timing_lines(valid_log())
    lines[0] = lines[0].rsplit(" ", 1)[0] + f" {value}"

    with pytest.raises(ValueError, match="non-finite or negative phase time"):
        parse_phase_log("\n".join(lines), ranks=2, steps=1)


@pytest.mark.parametrize(("ranks", "steps"), [(0, 1), (-1, 1), (1, 0), (1, -1)])
def test_parser_rejects_non_positive_dimensions(ranks: int, steps: int) -> None:
    with pytest.raises(ValueError, match="ranks and steps must be positive"):
        parse_phase_log(valid_log(), ranks=ranks, steps=steps)


def test_summary_rejects_empty_log_list() -> None:
    with pytest.raises(ValueError, match="at least one phase timing log"):
        summarize_logs([], ranks=1, steps=1)


def test_summary_keeps_independent_logs_separate(tmp_path: Path) -> None:
    first = tmp_path / "first.log"
    second = tmp_path / "second.log"
    first.write_text(valid_log(scale=1.0), encoding="ascii")
    second.write_text(valid_log(scale=2.0), encoding="ascii")

    rows = summarize_logs([first, second], ranks=2, steps=1)

    assert [row[0] for row in rows] == list(PHASE_ORDER)
    prepare = next(row for row in rows if row[0] == "prepare")
    assert prepare[1] == 2
    assert math.isclose(prepare[2], 3.0)
    assert prepare[3:] == (2.0, 4.0)


def test_cli_writes_ascii_tsv_in_stable_phase_order(tmp_path: Path) -> None:
    script = Path(__file__).with_name("summarize_gpu_phase_timing.py")
    log = tmp_path / "run.log"
    output = tmp_path / "nested" / "phases.tsv"
    log.write_text(valid_log(ranks=1), encoding="ascii")

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--log",
            str(log),
            "--ranks",
            "1",
            "--steps",
            "1",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    raw = output.read_bytes()
    raw.decode("ascii")
    lines = raw.decode("ascii").splitlines()
    assert lines[0] == "phase\tsamples\tmedian_seconds\tmin_seconds\tmax_seconds"
    assert [line.split("\t", 1)[0] for line in lines[1:]] == list(PHASE_ORDER)
