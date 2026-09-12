import pathlib
import sys


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from summarize_flux_pair_benchmark import summarize


def _write_timings(path: pathlib.Path, label: str, values: list[float]) -> None:
    lines = [
        "label\trepeat\trk_samples\tmedian_rk_seconds\tmin_rk_seconds\t"
        "max_rk_seconds\twall_seconds\tmax_memory_mib\tmax_utilization_percent"
    ]
    for repeat, value in enumerate(values, start=1):
        lines.append(
            f"{label}\t{repeat}\t5\t{value}\t{value}\t{value}\t1.0\t0\t0"
        )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def test_summary_uses_timing_only_and_accepts_unsampled_memory(tmp_path):
    split = tmp_path / "split.tsv"
    fused = tmp_path / "fused.tsv"
    _write_timings(split, "split", [1.00, 1.01, 0.99, 1.00, 1.00])
    _write_timings(fused, "fused", [0.80, 0.81, 0.79, 0.80, 0.80])

    report = "\n".join(summarize(split, fused, "fp64"))

    assert "20.000%" in report
    assert "1.25000x" in report
