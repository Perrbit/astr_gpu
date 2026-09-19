from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT / "tests/gpu_validation/run_zhongke_a800_cpu_gpu_comparison.sbatch"


def test_resource_and_workspace_contract() -> None:
    text = JOB.read_text(encoding="ascii")
    assert "#SBATCH --ntasks=32" in text
    assert "#SBATCH --hint=nomultithread" in text
    assert "#SBATCH --gres=gpu:4" in text
    assert "WORK_ROOT=/data/user/hd56000/weiph" in text


def test_matched_tgv_and_shock_matrix() -> None:
    text = JOB.read_text(encoding="ascii")
    for row in (
        "tgv_cpu_np1 cpu tgv_256 1 1,1,1 256,256,256",
        "tgv_cpu_np32 cpu tgv_256 32 4,4,2 256,256,256",
        "tgv_gpu_np1 gpu tgv_256 1 1,1,1 256,256,256",
        "tgv_gpu_np4 gpu tgv_256 4 2,2,1 256,256,256",
        "shuosher_cpu_np1 cpu shuosher_400x128x128 1 1,1,1 400,128,128",
        "shuosher_gpu_np1 gpu shuosher_400x128x128 1 1,1,1 400,128,128",
    ):
        assert row in text
    assert "tgv_cpu_np64" not in text


def test_no_field_output_and_preflight_contract() -> None:
    text = JOB.read_text(encoding="ascii")
    assert "ASTR_GPU_BENCHMARK_NO_FIELD_IO=1" in text
    assert "forbidden field output" in text
    assert "run_preflight" in text
    assert "CUDA-aware MPI is not production-qualified" in text
    assert "CRLF found" in text
    assert "prohibited relaxed-math option found" in text
    for label in (
        "preflight_tgv_cpu",
        "preflight_tgv_gpu",
        "preflight_shuosher_cpu",
        "preflight_shuosher_gpu",
    ):
        assert label in text


def test_summary_reports_rank_and_node_speedups(tmp_path: Path) -> None:
    timings = tmp_path / "timings.tsv"
    rows = ["case\tbackend\tnp\ttopology\trepeat\twall_seconds"]
    configs = [
        ("tgv_256", "cpu", 1, "1,1,1", 100.0),
        ("tgv_256", "cpu", 8, "2,2,2", 20.0),
        ("tgv_256", "gpu", 1, "1,1,1", 5.0),
        ("tgv_256", "gpu", 4, "2,2,1", 2.0),
        ("shuosher_400x128x128", "cpu", 1, "1,1,1", 60.0),
        ("shuosher_400x128x128", "gpu", 1, "1,1,1", 3.0),
    ]
    for case, backend, np, topology, median in configs:
        for repeat, offset in enumerate((-1.0, 0.0, 1.0), start=1):
            rows.append(f"{case}\t{backend}\t{np}\t{topology}\t{repeat}\t{median + offset}")
    timings.write_text("\n".join(rows) + "\n", encoding="ascii")
    output = tmp_path / "summary.tsv"
    summary = tmp_path / "summary.md"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tests/gpu_validation/summarize_cpu_gpu_comparison.py"),
            "--timings",
            str(timings),
            "--output-tsv",
            str(output),
            "--summary",
            str(summary),
        ],
        check=True,
    )
    text = summary.read_text(encoding="ascii")
    assert "TGV CPU NP=1 / GPU NP=1 speedup: `20.000000x`" in text
    assert "TGV best CPU configuration: `NP=8` / `2,2,2`" in text
    assert "TGV best CPU / GPU NP=4 speedup: `10.000000x`" in text
