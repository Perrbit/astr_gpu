from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT / "tests/gpu_validation/run_zhongke_a800_cpu_rk_scaling.sbatch"
SUMMARY = ROOT / "tests/gpu_validation/summarize_cpu_rk_scaling.py"


def test_job_is_cpu_only_and_uses_the_requested_512_cube_matrix() -> None:
    text = JOB.read_text(encoding="ascii")
    assert "#SBATCH --ntasks=32" in text
    assert "#SBATCH --hint=nomultithread" in text
    assert "#SBATCH --gres=gpu" not in text
    assert "GRID=512,512,512" in text
    for row in (
        "1 1,1,1",
        "2 2,1,1",
        "4 2,2,1",
        "8 2,2,2",
        "16 4,2,2",
        "32 4,4,2",
    ):
        assert f'"{row}"' in text


def test_job_records_only_complete_rk_timings_without_field_output() -> None:
    text = JOB.read_text(encoding="ascii")
    assert "MAXSTEP=5" in text
    assert "ASTR_GPU_BENCHMARK_NO_FIELD_IO=1" in text
    assert "ASTR_CPU_RK_TIMING=1" in text
    assert "ASTR_GPU_RK_TIMING=0" in text
    assert "export OMPI_MCA_pml=ob1" in text
    assert "export OMPI_MCA_btl=self,vader,tcp" in text
    assert "forbidden field output" in text
    assert "expected six CPU RK timing records" in text
    assert "run_preflight" in text
    assert "CRLF found" in text
    assert "not found" in text


def test_summary_discards_step_zero_and_reports_speedup_and_efficiency(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    for np, seconds in ((1, 100.0), (2, 52.0), (4, 27.0)):
        lines = ["ASTR_CPU_RK_TIMING 0 0.0 999.0 999.0"]
        for step, offset in zip(range(1, 6), (-2.0, -1.0, 0.0, 1.0, 2.0)):
            value = seconds + offset
            lines.append(f"ASTR_CPU_RK_TIMING {step} 0.0 {value} {value}")
        (logs / f"np{np}.log").write_text("\n".join(lines) + "\n", encoding="ascii")

    output = tmp_path / "summary.tsv"
    report = tmp_path / "summary.md"
    subprocess.run(
        [
            sys.executable,
            str(SUMMARY),
            "--logs",
            str(logs),
            "--output-tsv",
            str(output),
            "--summary",
            str(report),
        ],
        check=True,
    )
    text = output.read_text(encoding="ascii")
    assert "1\t5\t100.000000000\t1.000000\t1.000000" in text
    assert "2\t5\t52.000000000\t1.923077\t0.961538" in text
    assert "4\t5\t27.000000000\t3.703704\t0.925926" in text
    assert "999" not in text
