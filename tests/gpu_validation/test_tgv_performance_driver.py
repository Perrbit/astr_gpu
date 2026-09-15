#!/usr/bin/env python3
"""Contracts for single- and multi-rank TGV performance timing."""

import csv
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


def wait_for_file(path: Path, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.02)
    return path.exists()


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

    def test_driver_forwards_phase_timing_and_writes_retained_summary(self) -> None:
        self.assertIn('PHASE_TIMING="${PHASE_TIMING:-0}"', SCRIPT)
        self.assertIn('ASTR_GPU_PHASE_TIMING="$PHASE_TIMING"', SCRIPT)
        self.assertIn("summarize_gpu_phase_timing.py", SCRIPT)
        self.assertIn('PHASE_SUMMARY="$OUT_DIR/${LABEL}_phase_summary.tsv"', SCRIPT)
        self.assertIn('phase_timing=%s', SCRIPT)

        lifecycle = SCRIPT[SCRIPT.index('run_once warmup f') :]
        summary = lifecycle.index("summarize_gpu_phase_timing.py")
        self.assertLess(lifecycle.index('run_once "$repeat" t'), summary)
        self.assertNotIn("${LABEL}_run_warmup/run.log", lifecycle[summary:])

    def test_phase_timing_requires_explicit_sync_and_boolean_value(self) -> None:
        self.assertIn('PHASE_TIMING must be 0 or 1', SCRIPT)
        self.assertIn('PHASE_TIMING=1 requires SYNC_MODE=explicit', SCRIPT)

    def test_phase_summary_uses_five_retained_complete_logs(self) -> None:
        solver_source = """#!/usr/bin/env bash
printf '%s\n' 'ASTR_GPU_BENCHMARK_NO_FIELD_IO enabled'
for step in 0 1; do
  printf 'ASTR_GPU_RK_TIMING %s 0.1 0.2 0.3\n' "$step"
  printf 'ASTR_GPU_RANK_RK_TIMING 0 %s 0.1 0.2 0.3\n' "$step"
  if [[ "$ASTR_GPU_PHASE_TIMING" == "1" ]]; then
    printf 'ASTR_GPU_PHASE_TIMING 0 %s 0 prepare 0.1\n' "$step"
    for rkstep in 1 2 3; do
      for phase in filter solution_halo convection diffusion_flux diffusion_halo diffusion_rhs rk_update; do
        printf 'ASTR_GPU_PHASE_TIMING 0 %s %s %s 0.1\n' "$step" "$rkstep" "$phase"
      done
    done
  fi
done
printf '%s\n' 'The job is done!'
"""
        with tempfile.TemporaryDirectory(prefix="astr-p4-phase-summary-") as temporary:
            root = Path(temporary)
            environment, monitor_leader, monitor_child, *_ = self._monitor_environment(
                root, solver_source
            )
            environment["PHASE_TIMING"] = "1"
            completed = self._run_driver(root, environment)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self._assert_processes_exited(monitor_leader, monitor_child)

            summary = root / "out" / "monitor_contract_phase_summary.tsv"
            with summary.open(encoding="ascii", newline="") as stream:
                rows = {
                    row["phase"]: int(row["samples"])
                    for row in csv.DictReader(stream, delimiter="\t")
                }
            rk_phases = {
                "filter",
                "solution_halo",
                "convection",
                "diffusion_flux",
                "diffusion_halo",
                "diffusion_rhs",
                "rk_update",
            }
            self.assertEqual(set(rows), {"prepare"} | rk_phases)
            self.assertEqual(rows["prepare"], 10)
            self.assertEqual({rows[phase] for phase in rk_phases}, {30})

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

    def _monitor_environment(
        self,
        root: Path,
        solver_source: str,
        *,
        date_source: str = '#!/usr/bin/env bash\nexec /usr/bin/date "$@"\n',
        monitor_source: str | None = None,
    ) -> tuple[dict[str, str], Path, Path, Path, Path, Path, Path]:
        fake_bin = root / "bin"
        fake_bin.mkdir(parents=True)
        monitor_leader_file = root / "monitor-leader.pid"
        monitor_child_file = root / "monitor-child.pid"
        solver_leader_file = root / "solver-leader.pid"
        solver_child_file = root / "solver-child.pid"
        solver_started_file = root / "solver.started"
        solver = root / "solver.sh"
        if monitor_source is None:
            monitor_source = """#!/usr/bin/env bash
printf '%s\n' "$$" > "$FAKE_MONITOR_LEADER_FILE"
sleep 30 &
child=$!
printf '%s\n' "$child" > "$FAKE_MONITOR_CHILD_FILE"
trap 'wait "$child" 2>/dev/null || true; exit 0' TERM INT
printf '1, 1\n'
wait "$child"
"""
        write_executable(fake_bin / "nvidia-smi", monitor_source)
        write_executable(
            fake_bin / "mpirun",
            """#!/usr/bin/env bash
shift 2
exec "$@"
""",
        )
        write_executable(fake_bin / "date", date_source)
        write_executable(solver, solver_source)
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{fake_bin}:{environment['PATH']}",
                "GPU_EXE": str(solver),
                "OUT_DIR": str(root / "out"),
                "LABEL": "monitor_contract",
                "GRID": "8,8,8",
                "MAXSTEP": "1",
                "REPEATS": "5",
                "DISCARD_STEPS": "0",
                "FEQCHKPT": "9999",
                "NP": "1",
                "TOPOLOGY": "1,1,1",
                "GPU_IDS": "0",
                "FAKE_MONITOR_LEADER_FILE": str(monitor_leader_file),
                "FAKE_MONITOR_CHILD_FILE": str(monitor_child_file),
                "FAKE_SOLVER_LEADER_FILE": str(solver_leader_file),
                "FAKE_SOLVER_CHILD_FILE": str(solver_child_file),
                "FAKE_SOLVER_STARTED_FILE": str(solver_started_file),
            }
        )
        return (
            environment,
            monitor_leader_file,
            monitor_child_file,
            solver_leader_file,
            solver_child_file,
            solver_started_file,
            solver,
        )

    def _assert_processes_exited(self, *pid_files: Path) -> None:
        for pid_file in pid_files:
            self.assertTrue(pid_file.exists(), f"missing process PID file: {pid_file}")
            pid = int(pid_file.read_text(encoding="ascii").strip())
            exited = wait_for_process_exit(pid)
            if not exited:
                os.kill(pid, signal.SIGKILL)
                wait_for_process_exit(pid)
            self.assertTrue(exited, f"process {pid} survived driver exit")

    def _run_driver(self, root: Path, environment: dict[str, str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(DRIVER)],
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )

    def test_monitor_uses_and_reaps_a_dedicated_process_group(self) -> None:
        self.assertIn("command -v setsid", SCRIPT)
        self.assertIn("setsid nvidia-smi", SCRIPT)
        self.assertIn('--loop-ms=100', SCRIPT)
        cleanup_body = function_body("terminate_process_group")
        self.assertIn('kill -TERM -- "-$pid"', cleanup_body)
        self.assertIn('wait "$pid"', cleanup_body)
        self.assertLess(cleanup_body.index('kill -TERM -- "-$pid"'), cleanup_body.index("for _ in"))
        self.assertLess(cleanup_body.index("for _ in"), cleanup_body.index('kill -KILL -- "-$pid"'))
        self.assertLess(cleanup_body.index('kill -KILL -- "-$pid"'), cleanup_body.index('wait "$pid"'))

    def test_monitor_that_ignores_term_is_killed_without_hanging(self) -> None:
        solver_source = """#!/usr/bin/env bash
printf '%s\n' 'ASTR_GPU_BENCHMARK_NO_FIELD_IO enabled'
printf '%s\n' 'ASTR_GPU_RK_TIMING 0 0 0.1'
printf '%s\n' 'ASTR_GPU_RK_TIMING 0 1 0.1'
printf '%s\n' 'The job is done!'
"""
        monitor_source = """#!/usr/bin/env bash
printf '%s\n' "$$" > "$FAKE_MONITOR_LEADER_FILE"
trap '' TERM INT
sleep 30 &
child=$!
printf '%s\n' "$child" > "$FAKE_MONITOR_CHILD_FILE"
printf '1, 1\n'
wait "$child"
"""
        with tempfile.TemporaryDirectory(prefix="astr-p4-monitor-ignore-term-") as temporary:
            root = Path(temporary)
            environment, leader_file, child_file, *_ = self._monitor_environment(
                root, solver_source, monitor_source=monitor_source
            )
            started = time.monotonic()
            completed = self._run_driver(root, environment)
            elapsed = time.monotonic() - started
            self.assertNotEqual(completed.returncode, 0)
            self.assertLess(elapsed, 5.0, completed.stderr)
            self.assertIn("required SIGKILL", completed.stderr)
            self._assert_processes_exited(leader_file, child_file)

    def test_failures_after_monitor_start_reap_leader_and_child(self) -> None:
        with tempfile.TemporaryDirectory(prefix="astr-p4-monitor-") as temporary:
            for failure_mode, solver_source, date_source, expected_status in (
                (
                    "solver",
                    "#!/usr/bin/env bash\nsleep 0.2\nexit 37\n",
                    '#!/usr/bin/env bash\nexec /usr/bin/date "$@"\n',
                    37,
                ),
                (
                    "shell",
                    "#!/usr/bin/env bash\nsleep 0.2\nexit 0\n",
                    "#!/usr/bin/env bash\nsleep 0.2\nexit 71\n",
                    71,
                ),
            ):
                with self.subTest(failure_mode=failure_mode):
                    root = Path(temporary) / failure_mode
                    environment, leader_file, child_file, *_ = self._monitor_environment(
                        root, solver_source, date_source=date_source
                    )
                    completed = self._run_driver(root, environment)
                    self.assertEqual(completed.returncode, expected_status, completed.stderr)
                    self._assert_processes_exited(leader_file, child_file)

    def test_solver_exit_reaps_orphaned_process_group_members(self) -> None:
        cases = (
            ("failure", 37, 37, ""),
            (
                "success",
                0,
                1,
                "ASTR solver exited successfully but left process-group members",
            ),
        )
        with tempfile.TemporaryDirectory(prefix="astr-p4-solver-orphan-") as temporary:
            for name, solver_status, expected_status, diagnostic in cases:
                with self.subTest(solver_exit=name):
                    root = Path(temporary) / name
                    solver_source = f"""#!/usr/bin/env bash
printf '%s\n' "$$" > "$FAKE_SOLVER_LEADER_FILE"
sleep 30 &
child=$!
printf '%s\n' "$child" > "$FAKE_SOLVER_CHILD_FILE"
printf '%s\n' 'ASTR_GPU_BENCHMARK_NO_FIELD_IO enabled'
printf '%s\n' 'ASTR_GPU_RK_TIMING 0 0 0.1'
printf '%s\n' 'ASTR_GPU_RK_TIMING 0 1 0.1'
printf '%s\n' 'The job is done!'
exit {solver_status}
"""
                    (
                        environment,
                        monitor_leader_file,
                        monitor_child_file,
                        solver_leader_file,
                        solver_child_file,
                        _,
                        _,
                    ) = self._monitor_environment(root, solver_source)
                    completed = self._run_driver(root, environment)
                    self.assertEqual(
                        completed.returncode, expected_status, completed.stderr
                    )
                    if diagnostic:
                        self.assertIn(diagnostic, completed.stderr)
                    self._assert_processes_exited(
                        solver_leader_file,
                        solver_child_file,
                        monitor_leader_file,
                        monitor_child_file,
                    )

    def test_signals_reap_monitor_group_and_preserve_signal_status(self) -> None:
        with tempfile.TemporaryDirectory(prefix="astr-p4-signal-") as temporary:
            for name, sent_signal, expected_status in (
                ("sigterm", signal.SIGTERM, 143),
                ("sigint", signal.SIGINT, 130),
            ):
                with self.subTest(signal=name):
                    root = Path(temporary) / name
                    solver_source = """#!/usr/bin/env bash
printf '%s\n' "$$" > "$FAKE_SOLVER_LEADER_FILE"
trap 'wait "$child" 2>/dev/null || true; exit 0' TERM INT
sleep 30 &
child=$!
printf '%s\n' "$child" > "$FAKE_SOLVER_CHILD_FILE"
printf 'started\n' > "$FAKE_SOLVER_STARTED_FILE"
wait "$child"
"""
                    (
                        environment,
                        monitor_leader_file,
                        monitor_child_file,
                        solver_leader_file,
                        solver_child_file,
                        solver_started_file,
                        _,
                    ) = self._monitor_environment(
                        root, solver_source
                    )
                    process = subprocess.Popen(
                        ["bash", str(DRIVER)],
                        cwd=ROOT,
                        env=environment,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    self.assertTrue(
                        wait_for_file(solver_started_file),
                        f"solver did not start before {name}",
                    )
                    started = time.monotonic()
                    process.send_signal(sent_signal)
                    _, stderr = process.communicate(timeout=5)
                    elapsed = time.monotonic() - started
                    self.assertEqual(process.returncode, expected_status, stderr)
                    self.assertLess(elapsed, 2.5, stderr)
                    self._assert_processes_exited(
                        solver_leader_file,
                        solver_child_file,
                        monitor_leader_file,
                        monitor_child_file,
                    )

    def test_monitor_exit_and_empty_output_fail_closed(self) -> None:
        solver_source = """#!/usr/bin/env bash
printf '%s\n' 'ASTR_GPU_BENCHMARK_NO_FIELD_IO enabled'
printf '%s\n' 'ASTR_GPU_RK_TIMING 0 0 0.1'
printf '%s\n' 'ASTR_GPU_RK_TIMING 0 1 0.1'
printf '%s\n' 'The job is done!'
sleep 0.2
"""
        cases = (
            (
                "crash_with_sample",
                """#!/usr/bin/env bash
printf '%s\n' "$$" > "$FAKE_MONITOR_LEADER_FILE"
sleep 30 &
child=$!
printf '%s\n' "$child" > "$FAKE_MONITOR_CHILD_FILE"
printf '1, 1\n'
sleep 0.05
exit 42
""",
                "GPU monitor exited unexpectedly",
            ),
            (
                "empty_monitor",
                """#!/usr/bin/env bash
printf '%s\n' "$$" > "$FAKE_MONITOR_LEADER_FILE"
sleep 30 &
child=$!
printf '%s\n' "$child" > "$FAKE_MONITOR_CHILD_FILE"
trap 'wait "$child" 2>/dev/null || true; exit 0' TERM INT
wait "$child"
""",
                "GPU monitor produced no valid samples",
            ),
        )
        with tempfile.TemporaryDirectory(prefix="astr-p4-monitor-health-") as temporary:
            for name, monitor_source, diagnostic in cases:
                with self.subTest(monitor_failure=name):
                    root = Path(temporary) / name
                    environment, leader_file, child_file, *_ = self._monitor_environment(
                        root, solver_source, monitor_source=monitor_source
                    )
                    completed = self._run_driver(root, environment)
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn(diagnostic, completed.stderr)
                    self._assert_processes_exited(leader_file, child_file)


if __name__ == "__main__":
    unittest.main()
