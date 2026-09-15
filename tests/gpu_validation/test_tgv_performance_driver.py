#!/usr/bin/env python3
"""Contracts for single- and multi-rank TGV performance timing."""

import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (ROOT / "tests/gpu_validation/run_tgv_256_performance_benchmark.sh").read_text(
    encoding="utf-8"
)
DRIVER = ROOT / "tests/gpu_validation/run_tgv_256_performance_benchmark.sh"


def function_body(name: str) -> str:
    marker = f"{name}() {{\n"
    start = SCRIPT.index(marker) + len(marker)
    end = SCRIPT.index("\n}\n", start)
    return SCRIPT[start:end]


def write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_for_process_exit(pid: int, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_is_alive(pid):
            return True
        time.sleep(0.02)
    return not process_is_alive(pid)


class TgvPerformanceDriverTests(unittest.TestCase):
    def test_driver_accepts_rank_topology_and_visible_gpu_list(self) -> None:
        self.assertIn('NP="${NP:-1}"', SCRIPT)
        self.assertIn('TOPOLOGY="${TOPOLOGY:-1,1,1}"', SCRIPT)
        self.assertIn('GPU_IDS="${GPU_IDS:-$GPU_ID}"', SCRIPT)
        self.assertIn('CUDA_VISIBLE_DEVICES="$GPU_IDS"', SCRIPT)
        self.assertIn('ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY"', SCRIPT)
        self.assertIn('mpirun -np "$NP"', SCRIPT)

    def test_driver_uses_slowest_rank_for_each_rk_step(self) -> None:
        self.assertIn('ASTR_GPU_RANK_RK_TIMING=1', SCRIPT)
        self.assertIn('summarize_rank_rk_timing.py', SCRIPT)
        self.assertIn('--ranks "$NP"', SCRIPT)

    def test_driver_requires_one_distinct_gpu_per_rank(self) -> None:
        self.assertIn("math.prod(topology) != np", SCRIPT)
        self.assertIn("len(devices) != np", SCRIPT)
        self.assertIn("len(set(devices)) != np", SCRIPT)

    def test_driver_records_halo_transport_backend(self) -> None:
        self.assertIn('HALO_TRANSPORT="${HALO_TRANSPORT:-pageable}"', SCRIPT)
        self.assertIn('ASTR_GPU_HALO_TRANSPORT="$HALO_TRANSPORT"', SCRIPT)
        self.assertIn("halo_transport=%s", SCRIPT)

    def test_driver_accepts_cfl_selected_time_step(self) -> None:
        self.assertIn('DELTAT="${DELTAT:-}"', SCRIPT)
        self.assertIn('args+=(--deltat "$DELTAT")', SCRIPT)

    def test_driver_records_and_forwards_filter_workspace(self) -> None:
        self.assertIn('FILTER_WORKSPACE="${FILTER_WORKSPACE:-full}"', SCRIPT)
        self.assertIn('ASTR_GPU_FILTER_WORKSPACE="$FILTER_WORKSPACE"', SCRIPT)
        self.assertIn("filter_workspace=%s", SCRIPT)

    def test_driver_enables_and_records_no_field_io_mode(self) -> None:
        self.assertIn("ASTR_GPU_BENCHMARK_NO_FIELD_IO=1", SCRIPT)
        self.assertIn("benchmark_no_field_io=1", SCRIPT)
        self.assertIn("ASTR_GPU_BENCHMARK_NO_FIELD_IO enabled", SCRIPT)

    def test_driver_forces_generated_grid_and_removes_copied_hdf5(self) -> None:
        self.assertIn('args+=(--lreadgrid f)', SCRIPT)
        self.assertIn('rm -f "$CASE_DIR/datin/grid.h5"', SCRIPT)

    def test_driver_rejects_generated_grid_or_flowfield_hdf5(self) -> None:
        self.assertIn("assert_no_field_hdf5", SCRIPT)
        self.assertIn("-name 'grid*.h5'", SCRIPT)
        self.assertIn("-name 'flowfield*.h5'", SCRIPT)

    def test_run_once_keeps_admission_and_hdf5_checks_around_each_launch(self) -> None:
        body = function_body("run_once")
        launch = body.index('mpirun -np "$NP"')
        pre_hdf5 = body.index("assert_no_field_hdf5")
        monitor_cleanup = body.index("stop_monitor", launch)
        post_hdf5 = body.index("assert_no_field_hdf5", launch)
        admission = body.index("ASTR_GPU_BENCHMARK_NO_FIELD_IO enabled", launch)

        self.assertEqual(body.count('mpirun -np "$NP"'), 1)
        self.assertEqual(body.count("ASTR_GPU_BENCHMARK_NO_FIELD_IO=1"), 1)
        self.assertEqual(body.count("assert_no_field_hdf5"), 2)
        self.assertLess(pre_hdf5, launch)
        self.assertIn("ASTR_GPU_BENCHMARK_NO_FIELD_IO=1", body[:launch])
        self.assertLess(launch, monitor_cleanup)
        self.assertLess(monitor_cleanup, post_hdf5)
        self.assertLess(post_hdf5, admission)

    def test_driver_runs_one_warmup_then_at_least_five_retained_processes(self) -> None:
        lifecycle = SCRIPT[SCRIPT.index('mkdir -p "$OUT_DIR"') :]
        warmup = lifecycle.index("run_once warmup f")
        retained_loop = lifecycle.index('for repeat in $(seq 1 "$REPEATS"); do')
        retained_run = lifecycle.index('run_once "$repeat" t', retained_loop)
        repeat_gate = SCRIPT.index('if [[ "$REPEATS" -lt 5 ]]')
        warmup_absolute = SCRIPT.index("run_once warmup f")

        self.assertLess(repeat_gate, warmup_absolute)
        self.assertEqual(lifecycle.count("run_once warmup f"), 1)
        self.assertLess(warmup, retained_loop)
        self.assertLess(retained_loop, retained_run)

    def test_driver_installs_exit_and_signal_monitor_cleanup(self) -> None:
        self.assertIn("trap 'handle_exit' EXIT", SCRIPT)
        self.assertIn("trap 'handle_signal 130' INT", SCRIPT)
        self.assertIn("trap 'handle_signal 143' TERM", SCRIPT)
        self.assertIn("trap - EXIT INT TERM", function_body("handle_exit"))
        self.assertIn("trap - EXIT INT TERM", function_body("handle_signal"))

    def test_failure_after_monitor_start_does_not_leave_monitor_alive(self) -> None:
        with tempfile.TemporaryDirectory(prefix="astr-p4-monitor-") as temporary:
            for failure_mode, date_source, expected_status in (
                (
                    "solver",
                    """#!/usr/bin/env bash
exec /usr/bin/date "$@"
""",
                    37,
                ),
                (
                    "shell",
                    """#!/usr/bin/env bash
sleep 0.2
exit 71
""",
                    71,
                ),
            ):
                with self.subTest(failure_mode=failure_mode):
                    root = Path(temporary) / failure_mode
                    fake_bin = root / "bin"
                    fake_bin.mkdir(parents=True)
                    monitor_pid_file = root / "monitor.pid"
                    failing_solver = root / "failing_solver.sh"

                    write_executable(
                        fake_bin / "nvidia-smi",
                        """#!/usr/bin/env bash
printf '%s\n' "$PPID" > "$FAKE_MONITOR_PID_FILE"
printf '1, 1\n'
sleep 0.02
""",
                    )
                    write_executable(
                        fake_bin / "mpirun",
                        """#!/usr/bin/env bash
shift 2
exec "$@"
""",
                    )
                    write_executable(fake_bin / "date", date_source)
                    write_executable(
                        failing_solver,
                        """#!/usr/bin/env bash
sleep 0.2
exit 37
""",
                    )

                    environment = os.environ.copy()
                    environment.update(
                        {
                            "PATH": f"{fake_bin}:{environment['PATH']}",
                            "GPU_EXE": str(failing_solver),
                            "OUT_DIR": str(root / "out"),
                            "LABEL": "monitor_failure",
                            "GRID": "8,8,8",
                            "MAXSTEP": "1",
                            "REPEATS": "5",
                            "DISCARD_STEPS": "0",
                            "FEQCHKPT": "9999",
                            "NP": "1",
                            "TOPOLOGY": "1,1,1",
                            "GPU_IDS": "0",
                            "FAKE_MONITOR_PID_FILE": str(monitor_pid_file),
                        }
                    )
                    stdout_path = root / "driver.stdout"
                    stderr_path = root / "driver.stderr"
                    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                        "w", encoding="utf-8"
                    ) as stderr:
                        completed = subprocess.run(
                            ["bash", str(DRIVER)],
                            cwd=ROOT,
                            env=environment,
                            stdout=stdout,
                            stderr=stderr,
                            text=True,
                            timeout=10,
                            check=False,
                        )

                    self.assertEqual(
                        completed.returncode,
                        expected_status,
                        stderr_path.read_text(encoding="utf-8"),
                    )
                    self.assertTrue(
                        monitor_pid_file.exists(), "fake monitor never started"
                    )
                    monitor_pid = int(
                        monitor_pid_file.read_text(encoding="ascii").strip()
                    )
                    monitor_exited = wait_for_process_exit(monitor_pid)
                    if not monitor_exited:
                        os.kill(monitor_pid, signal.SIGTERM)
                        wait_for_process_exit(monitor_pid)
                    self.assertTrue(
                        monitor_exited,
                        f"monitor process {monitor_pid} survived driver exit",
                    )


if __name__ == "__main__":
    unittest.main()
