#!/usr/bin/env python3
"""Contracts for device-aware MPI qualification before solver integration."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "tests/gpu_validation/halo_cuda_aware_probe.cuf"
DRIVER = ROOT / "tests/gpu_validation/run_cuda_aware_mpi_qualification.sh"
SUMMARY = ROOT / "tests/gpu_validation/summarize_cuda_aware_mpi_qualification.py"
A800_DRIVER = ROOT / "tests/gpu_validation/run_zhongke_a800_cuda_aware_mpi_qualification.sbatch"
A800_SOLVER_DRIVER = ROOT / "tests/gpu_validation/run_zhongke_a800_device_aware_solver_admission.sbatch"
A800_NONREACTING_DRIVER = ROOT / "tests/gpu_validation/run_zhongke_a800_device_aware_nonreacting_admission.sbatch"
NONREACTING_DRIVER = ROOT / "tests/gpu_validation/run_device_aware_nonreacting_matrix.sh"
A800_SCALING_DRIVER = ROOT / "tests/gpu_validation/run_zhongke_a800_device_aware_scaling.sbatch"
PERFORMANCE_DRIVER = ROOT / "tests/gpu_validation/run_tgv_256_performance_benchmark.sh"
SCALING_SUMMARY = ROOT / "tests/gpu_validation/summarize_device_aware_scaling.py"
SANITIZER_SUPPRESSIONS = ROOT / "tests/gpu_validation/compute_sanitizer_ucx_cuda_aware.supp.xml"
DEVICE_TRANSPORT = ROOT / "src_gpu/device_mpi_transport_gpu.cuf"
HOST_TRANSPORT = ROOT / "src_gpu/halo_transport_gpu.cuf"
HALO_EXCHANGE = ROOT / "src_gpu/halo_exchange_gpu.cuf"
CMAKE = ROOT / "src/CMakeLists.txt"
ASTR_MAIN = ROOT / "src/astr.F90"
GPU_RUNTIME = ROOT / "src_gpu/gpu_runtime.cuf"
DEVICE_RUNTIME = ROOT / "src_gpu/device_runtime_gpu.cuf"
ZEROEXTRAP_DRIVER = ROOT / "tests/gpu_validation/run_xextrap_phaseb_compare.sh"
CURVE_DRIVER = ROOT / "tests/gpu_validation/run_curvilinear_tgv_compare.sh"
SHOCK_DRIVER = ROOT / "tests/gpu_validation/run_s2_hbl_oblique_shock_compare.sh"
WALL41_DRIVER = ROOT / "tests/gpu_validation/run_wall41_phased_compare.sh"


class DeviceAwareProbeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.probe = PROBE.read_text(encoding="utf-8").lower().replace(" ", "")

    def test_covers_all_astr_transport_component_counts(self) -> None:
        self.assertIn("variables(5)=[1,3,5,6,9]", self.probe)
        self.assertIn("dowidth=5,6", self.probe)

    def test_covers_blocking_and_nonblocking_pair_exchanges(self) -> None:
        for call in ("mpi_sendrecv", "mpi_irecv", "mpi_isend", "mpi_waitall"):
            self.assertIn(call, self.probe)
        self.assertIn("doexchange_mode=1,2", self.probe)

    def test_keeps_periodic_and_proc_null_neighbor_modes(self) -> None:
        self.assertIn("doneighbor_mode=1,2", self.probe)
        self.assertIn("mpi_proc_null", self.probe)


class DeviceAwareQualificationDriverContractTests(unittest.TestCase):
    def test_driver_uses_actual_tgv_payload_shapes(self) -> None:
        self.assertTrue(DRIVER.is_file(), f"missing qualification driver: {DRIVER}")
        text = DRIVER.read_text(encoding="ascii")
        self.assertIn('"513 257"', text)
        self.assertIn('"513 513"', text)
        self.assertIn("compute-sanitizer", text)
        self.assertIn("--target-processes all", text)
        self.assertIn('"early-bind"', text)
        self.assertIn('"$clean_count" -ge 1', text)
        self.assertIn("QUALIFICATION_CUDA_VISIBLE_DEVICES", text)
        self.assertIn("PAYLOAD_REPEATS", text)
        self.assertIn('UCX_TLS="$tls"', text)
        self.assertIn('tls="self,sm,cuda_copy,cuda_ipc"', text)
        self.assertIn("UCX_PROTO_INFO=y", text)
        self.assertIn('--suppressions "$SUPPRESSIONS"', text)

    def test_driver_uses_one_consistent_mpi_ucx_stack(self) -> None:
        self.assertTrue(DRIVER.is_file(), f"missing qualification driver: {DRIVER}")
        text = DRIVER.read_text(encoding="ascii")
        for marker in (
            "mpifort --showme:libdirs",
            "hpcx-init-ompi.sh",
            "hpcx_load",
            'MPIEXEC_EXE="$mpi_prefix/bin/.bin/mpirun"',
            "command -v ompi_info",
            "ucx_info -v",
            'ldd "$PROBE_EXE"',
        ):
            self.assertIn(marker, text)

    def test_driver_is_fail_closed_and_records_provenance(self) -> None:
        self.assertTrue(DRIVER.is_file(), f"missing qualification driver: {DRIVER}")
        text = DRIVER.read_text(encoding="ascii")
        for marker in (
            "qualification.json",
            "sha256sum",
            "mpirun --version",
            "nvidia-smi topo -m",
            "payload",
            "sanitizer",
        ):
            self.assertIn(marker, text)
        self.assertNotIn("|| true", text)

    def test_summary_requires_all_payloads_and_clean_sanitizer(self) -> None:
        self.assertTrue(SUMMARY.is_file(), f"missing qualification summary: {SUMMARY}")
        text = SUMMARY.read_text(encoding="ascii")
        self.assertIn("def summarize_qualification(", text)
        self.assertIn('record["kind"] == "payload"', text)
        self.assertIn('record["kind"] == "sanitizer"', text)
        self.assertIn('record["kind"] == "protocol"', text)
        self.assertIn('"production_qualified": "ucx_ipc" in qualified_configs', text)
        for field in (
            '"payload_status"',
            '"sanitizer_status"',
            '"protocol_status"',
            '"mpi_stack"',
            '"transport"',
            '"message_bytes"',
        ):
            self.assertIn(field, text)

    def test_a800_driver_is_probe_only_and_uses_slurm_devices(self) -> None:
        self.assertTrue(A800_DRIVER.is_file(), f"missing A800 driver: {A800_DRIVER}")
        text = A800_DRIVER.read_text(encoding="ascii")
        for marker in (
            "#SBATCH --gres=gpu:2",
            "#SBATCH --ntasks=2",
            "CUDA_VISIBLE_DEVICES",
            "QUALIFICATION_CUDA_VISIBLE_DEVICES",
            "run_cuda_aware_mpi_qualification.sh",
            "halo_cuda_aware_probe",
            "refusing to overwrite",
            "venv_tgv_campaign_py311/bin/python",
        ):
            self.assertIn(marker, text)
        self.assertNotIn("bin/astr", text)

    def test_ucx_suppressions_are_narrow_api_stack_matches(self) -> None:
        self.assertTrue(
            SANITIZER_SUPPRESSIONS.is_file(),
            f"missing UCX sanitizer suppressions: {SANITIZER_SUPPRESSIONS}",
        )
        text = SANITIZER_SUPPRESSIONS.read_text(encoding="ascii")
        self.assertEqual(text.count("<record>"), 4)
        self.assertIn("uct_cuda_copy_detect_vmm", text)
        self.assertIn("mca_common_cuda_is_gpu_buffer", text)
        self.assertIn("uct_cuda_copy_mem_reg", text)
        self.assertNotIn("cuMemCreate", text)
        self.assertNotIn(".*", text)

    def test_a800_solver_admission_requires_ipc_statistics_and_sanitizer(self) -> None:
        self.assertTrue(
            A800_SOLVER_DRIVER.is_file(),
            f"missing A800 solver admission driver: {A800_SOLVER_DRIVER}",
        )
        text = A800_SOLVER_DRIVER.read_text(encoding="ascii")
        for marker in (
            "production_qualified",
            "UCX_TLS=self,sm,cuda_copy,cuda_ipc",
            "ASTR_GPU_BENCHMARK_NO_FIELD_IO=1",
            "ASTR_GPU_RK_TIMING=1",
            'export UCX_PROTO_INFO="$protocol_info"',
            'run_case "$candidate" "$topology" device-aware y',
            "cuda_ipc/cuda",
            "compare_tgv_flowstate.py",
            "assert_no_field_hdf5",
            "prohibited relaxed-math compiler option found",
            "flags.make",
            "--grid 128,128,128",
            'local maxstep="${2:-5}"',
            'MPI_NP="${MPI_NP:-2}"',
            'mpirun -np "$MPI_NP"',
            "2,1,1",
            "1,2,1",
            "1,1,2",
            "2,2,1",
            "2,1,2",
            "1,2,2",
            "compute-sanitizer",
            '--suppressions "$SUPPRESSIONS"',
            "ERROR SUMMARY: 0 errors",
            "A800_DEVICE_AWARE_SOLVER_ADMISSION=PASS",
        ):
            self.assertIn(marker, text)
        self.assertNotIn("compare_flowfield_h5.py", text)
        self.assertNotIn("|| true", text)

    def test_nonreacting_driver_covers_four_device_aware_case_families(self) -> None:
        self.assertTrue(
            NONREACTING_DRIVER.is_file(),
            f"missing nonreacting device-aware matrix: {NONREACTING_DRIVER}",
        )
        text = NONREACTING_DRIVER.read_text(encoding="ascii")
        for marker in (
            "cartesian-boundary",
            "curve",
            "shock-sensor",
            "wall-family",
            "run_zeroextrap_phaseb_mpirank_matrix.sh",
            "run_curvilinear_wall41_compare.sh",
            "run_s2_hbl_selective_roe_s2c3_compare.sh",
            "run_wall_family_phaseh_matrix.sh",
            "GPU_HALO_TRANSPORT=device-aware",
            "ASTR_GPU_HALO_TRANSPORT_SELECTED=device-aware",
            "cuda_ipc/cuda",
            "flowstate_compare.txt",
            "flowfield_compare.txt",
            "shock_sensor_compare.txt",
            "NONREACTING_DEVICE_AWARE_MATRIX=PASS",
        ):
            self.assertIn(marker, text)
        self.assertNotIn("|| true", text)

    def test_compare_drivers_scope_transport_to_gpu_run(self) -> None:
        for driver in (ZEROEXTRAP_DRIVER, CURVE_DRIVER, SHOCK_DRIVER, WALL41_DRIVER):
            text = driver.read_text(encoding="ascii")
            self.assertIn('GPU_HALO_TRANSPORT="${GPU_HALO_TRANSPORT:-}"', text)
            self.assertIn('export ASTR_GPU_HALO_TRANSPORT="$GPU_HALO_TRANSPORT"', text)
            cpu_block, gpu_block = text.split('cd "$OUT_DIR/gpu"', maxsplit=1)
            self.assertNotIn("ASTR_GPU_HALO_TRANSPORT", cpu_block.split('cd "$OUT_DIR/cpu"')[-1])
            self.assertIn("ASTR_GPU_HALO_TRANSPORT", gpu_block)

    def test_a800_nonreacting_wrapper_is_fail_closed(self) -> None:
        self.assertTrue(
            A800_NONREACTING_DRIVER.is_file(),
            f"missing A800 nonreacting admission driver: {A800_NONREACTING_DRIVER}",
        )
        text = A800_NONREACTING_DRIVER.read_text(encoding="ascii")
        for marker in (
            "#SBATCH --gres=gpu:2",
            "#SBATCH --ntasks=2",
            "production_qualified",
            "UCX_TLS=self,sm,cuda_copy,cuda_ipc",
            "CUDA_VISIBLE_DEVICES",
            "ldd",
            "prohibited relaxed-math compiler option found",
            "run_device_aware_nonreacting_matrix.sh",
            "refusing to overwrite",
        ):
            self.assertIn(marker, text)
        self.assertNotIn("|| true", text)

    def test_performance_driver_accepts_device_aware_transport(self) -> None:
        text = PERFORMANCE_DRIVER.read_text(encoding="ascii")
        self.assertIn('"$HALO_TRANSPORT" != "device-aware"', text)
        self.assertIn("pinned-pipeline or device-aware", text)

    def test_a800_scaling_job_is_matched_and_fail_closed(self) -> None:
        self.assertTrue(
            A800_SCALING_DRIVER.is_file(),
            f"missing A800 device-aware scaling driver: {A800_SCALING_DRIVER}",
        )
        text = A800_SCALING_DRIVER.read_text(encoding="ascii")
        for marker in (
            "#SBATCH --gres=gpu:4",
            "MAXSTEP=21",
            "DISCARD_STEPS=2",
            "REPEATS=5",
            "FILTER_WORKSPACE=full",
            "ASTR_GPU_SYNC_MODE=explicit",
            "ASTR_GPU_BENCHMARK_NO_FIELD_IO=1",
            "UCX_TLS=self,sm,cuda_copy,cuda_ipc",
            "production_qualified",
            "cuda_ipc/cuda",
            "strong_np2_211_pinned_pipeline",
            "strong_np2_211_device_aware",
            "strong_np4_122_pinned_pipeline",
            "strong_np4_122_device_aware",
            "summarize_device_aware_scaling.py",
            "EXPECTED_CASES=19",
            "EXPECTED_PAIRS=9",
        ):
            self.assertIn(marker, text)
        self.assertNotIn("compare_flowfield_h5.py", text)

    def test_device_aware_scaling_summary_reports_backend_gain(self) -> None:
        text = SCALING_SUMMARY.read_text(encoding="ascii")
        for marker in (
            "pinned_pipeline_seconds",
            "device_aware_seconds",
            "transport_speedup",
            "parallel_efficiency",
            "Backend comparison",
        ):
            self.assertIn(marker, text)


class DeviceAwareSolverIntegrationContractTests(unittest.TestCase):
    def test_device_transport_posts_receives_before_sends_and_waits(self) -> None:
        self.assertTrue(DEVICE_TRANSPORT.is_file(), f"missing device transport: {DEVICE_TRANSPORT}")
        text = DEVICE_TRANSPORT.read_text(encoding="ascii").lower().replace(" ", "")
        self.assertIn("subroutineexchange_device_pair(", text)
        self.assertIn("real(8),device,contiguous,intent(in)::send_low", text)
        first_recv = text.index("callmpi_irecv(")
        second_recv = text.index("callmpi_irecv(", first_recv + 1)
        first_send = text.index("callmpi_isend(")
        self.assertLess(second_recv, first_send)
        self.assertIn("callmpi_waitall(4,requests,statuses,ierr)", text)

    def test_selector_is_collective_opt_in_and_preserves_host_backends(self) -> None:
        text = HOST_TRANSPORT.read_text(encoding="ascii").lower().replace(" ", "")
        for backend in (
            "pageable",
            "pinned",
            "pinned-overlap",
            "pinned-pipeline",
            "device-aware",
        ):
            self.assertIn(f"trim(value)=='{backend}'", text)
        self.assertIn("device_aware_transport_enabled", text)
        self.assertIn("callmpi_allreduce(choice,lowest", text)
        self.assertIn("callmpi_allreduce(choice,highest", text)
        self.assertIn("mpix_query_cuda_support", text)
        self.assertIn("requiresmpicudasupport", text)
        self.assertIn("astr_mpi_library_version=", text)

    def test_solution_halos_dispatch_device_buffers_without_host_copies(self) -> None:
        text = HALO_EXCHANGE.read_text(encoding="ascii").lower().replace(" ", "")
        expected_calls = (
            "callexchange_device_pair(x_send_left_d,x_send_right_d,x_recv_right_d,x_recv_left_d",
            "callexchange_device_pair(y_send_down_d,y_send_up_d,y_recv_up_d,y_recv_down_d",
            "callexchange_device_pair(z_send_back_d,z_send_front_d,z_recv_front_d,z_recv_back_d",
        )
        for call in expected_calls:
            self.assertIn(call, text)
        self.assertGreaterEqual(text.count("if(device_aware_transport_enabled)then"), 3)

    def test_filter_halos_dispatch_all_device_buffers(self) -> None:
        text = HALO_EXCHANGE.read_text(encoding="ascii").lower().replace(" ", "")
        expected_calls = (
            "callexchange_device_pair(x_halo_send_left_d,x_halo_send_right_d,",
            "callexchange_device_pair(y_halo_send_down_d,y_halo_send_up_d,",
            "callexchange_device_pair(z_halo_send_back_d,z_halo_send_front_d,",
            "callexchange_device_pair(field_send_down_d(:,:,:,1:1),field_send_up_d(:,:,:,1:1),",
            "callexchange_device_pair(field_send_back_d(:,:,:,1:1),field_send_front_d(:,:,:,1:1),",
        )
        for call in expected_calls:
            self.assertIn(call, text)

    def test_fp64_and_mixed_diffusion_fields_dispatch_device_buffers(self) -> None:
        text = HALO_EXCHANGE.read_text(encoding="ascii").lower().replace(" ", "")
        for call in (
            "callexchange_device_pair(field_send_left_d(:,:,:,1:nvar),field_send_right_d(:,:,:,1:nvar),",
            "callexchange_device_pair(field_send_down_d(:,:,:,1:nvar),field_send_up_d(:,:,:,1:nvar),",
            "callexchange_device_pair(field_send_back_d(:,:,:,1:nvar),field_send_front_d(:,:,:,1:nvar),",
        ):
            self.assertGreaterEqual(text.count(call), 2)

    def test_device_transport_preserves_independent_work_callback(self) -> None:
        text = DEVICE_TRANSPORT.read_text(encoding="ascii").lower().replace(" ", "")
        self.assertIn("procedure(independent_host_pair_work),optional::work", text)
        self.assertIn("callbegin_external_pair_work()", text)
        self.assertIn("callwork(requests,0)", text)
        self.assertIn("callend_external_pair_work()", text)

    def test_cmake_builds_device_transport_before_halo_exchange(self) -> None:
        text = CMAKE.read_text(encoding="ascii")
        device = text.index("../src_gpu/device_mpi_transport_gpu.cuf")
        exchange = text.index("../src_gpu/halo_exchange_gpu.cuf")
        self.assertLess(device, exchange)
        self.assertIn("ASTR_MPI_CUDA_AWARE_QUERY", text)

    def test_device_aware_mode_prebinds_cuda_before_mpi_init(self) -> None:
        main = ASTR_MAIN.read_text(encoding="ascii").lower().replace(" ", "")
        prebind = main.index("callgpu_pre_mpi_bind_device()")
        mpi_init = main.index("callmpiinitial")
        self.assertLess(prebind, mpi_init)
        runtime = GPU_RUNTIME.read_text(encoding="ascii").lower().replace(" ", "")
        self.assertIn("subroutinegpu_pre_mpi_bind_device()", runtime)
        device = DEVICE_RUNTIME.read_text(encoding="ascii").lower().replace(" ", "")
        self.assertIn("subroutineprebind_gpu_device_from_environment()", device)
        self.assertIn("ompi_comm_world_local_rank", device)
        self.assertIn("slurm_localid", device)
        self.assertIn("astr_gpu_halo_transport", device)


if __name__ == "__main__":
    unittest.main()
