import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
PRECISION_PATH = ROOT / "src_gpu" / "precision_mode_gpu.cuf"
PRECISION = PRECISION_PATH.read_text() if PRECISION_PATH.exists() else ""
CMAKE = (ROOT / "src" / "CMakeLists.txt").read_text()
COMMARRAY = (ROOT / "src_gpu" / "commarray_gpu.cuf").read_text()
SOLVER = (ROOT / "src_gpu" / "solver_gpu.cuf").read_text()
MAINLOOP = (ROOT / "src_gpu" / "mainloop_gpu.cuf").read_text()
CAPABILITY = (ROOT / "src_gpu" / "case_capability_gpu.cuf").read_text()
COMPARE_DRIVER_PATH = ROOT / "tests" / "gpu_validation" / "run_tgv_upwind_mixed_precision_compare.sh"
COMPARE_DRIVER = COMPARE_DRIVER_PATH.read_text() if COMPARE_DRIVER_PATH.exists() else ""
MEMCHECK_DRIVER_PATH = ROOT / "tests" / "gpu_validation" / "run_tgv_upwind_mixed_precision_memcheck.sh"
MEMCHECK_DRIVER = MEMCHECK_DRIVER_PATH.read_text() if MEMCHECK_DRIVER_PATH.exists() else ""
BENCHMARK_DRIVER_PATH = ROOT / "tests" / "gpu_validation" / "run_tgv_upwind_mixed_precision_benchmark.sh"
BENCHMARK_DRIVER = BENCHMARK_DRIVER_PATH.read_text() if BENCHMARK_DRIVER_PATH.exists() else ""


class MixedPrecisionWorkspaceContract(unittest.TestCase):
    def test_runtime_mode_has_fp64_default_and_mixed_workspace_option(self):
        self.assertIn("module precision_mode_gpu", PRECISION)
        self.assertIn("GPU_PRECISION_FP64=0", PRECISION.replace(" ", ""))
        self.assertIn("GPU_PRECISION_MIXED_WORKSPACE=1", PRECISION.replace(" ", ""))
        self.assertIn("ASTR_GPU_PRECISION_MODE", PRECISION)
        self.assertIn("case('fp64')", PRECISION.replace(" ", ""))
        self.assertIn("case('mixed_workspace')", PRECISION.replace(" ", ""))

    def test_runtime_mode_is_collectively_validated(self):
        lower = PRECISION.lower().replace(" ", "")
        self.assertGreaterEqual(lower.count("mpi_allreduce"), 2)
        self.assertIn("mpi_min", lower)
        self.assertIn("mpi_max", lower)
        self.assertIn("lowest/=highest", lower)
        self.assertIn("mpi_abort", lower)
        self.assertIn("invalidorinconsistentastr_gpu_precision_mode", lower)

    def test_runtime_mode_exports_query_and_name(self):
        lower = PRECISION.lower()
        self.assertIn("subroutine configure_gpu_precision_mode", lower)
        self.assertIn("logical function gpu_mixed_workspace_requested", lower)
        self.assertIn("function gpu_precision_mode_name", lower)

    def test_cmake_orders_precision_module_before_device_arrays(self):
        precision = CMAKE.find("../src_gpu/precision_mode_gpu.cuf")
        commarray = CMAKE.find("../src_gpu/commarray_gpu.cuf")
        self.assertGreaterEqual(precision, 0)
        self.assertGreater(commarray, precision)
        self.assertIn("precision_mode_setup_test", CMAKE)

    def test_flux_workspace_has_mutually_exclusive_fp64_and_fp32_storage(self):
        compact = COMMARRAY.replace(" ", "").lower()
        self.assertRegex(
            compact,
            r"real\(4\),allocatable,device::flux_work_sp_d\(:,:,:,?\)",
        )
        self.assertIn("logicalfunctiongpu_mixed_flux_workspace_enabled", compact)
        self.assertIn("gpu_mixed_workspace_requested()", compact)
        self.assertIn("trim(conschm)=='543e'", compact)
        self.assertIn(".not.lchardecomp", compact)
        self.assertIn("lihomo.and.ljhomo.and.lkhomo", compact)
        self.assertIn("allocate(flux_work_sp_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm))", compact)
        self.assertIn("allocate(flux_work_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm))", compact)
        self.assertIn("if(gpu_mixed_flux_workspace_enabled())then", compact)
        self.assertIn("else", compact)

    def test_flux_workspace_reports_mode_candidate_and_exact_bytes(self):
        compact = COMMARRAY.replace(" ", "").lower()
        self.assertIn("astr_gpu_precision_mode=", compact)
        self.assertIn("astr_gpu_active_mixed_workspace=", compact)
        self.assertIn("astr_gpu_flux_workspace_bytes=", compact)
        self.assertIn("*4_8", compact)
        self.assertIn("*8_8", compact)

    def test_periodic_fp32_kernels_round_only_at_workspace_boundary(self):
        for axis in "xyz":
            self.assertIn(f"explicit_upwind_flux_{axis}_global_sp_kernel", SOLVER)
            self.assertIn(f"explicit_upwind_rhs_{axis}_global_sp_kernel", SOLVER)
        compact = SOLVER.replace(" ", "").lower()
        self.assertGreaterEqual(compact.count("real(fh,4)"), 3)
        self.assertGreaterEqual(compact.count("real(flux_work_sp_d"), 6)
        self.assertNotIn("flux_work_conversion_kernel", compact)

    def test_mainloop_dispatches_fp32_periodic_kernels_with_explicit_sync(self):
        compact = MAINLOOP.replace(" ", "").lower()
        self.assertIn("gpu_mixed_flux_workspace_enabled", compact)
        for axis in "xyz":
            flux = f"explicit_upwind_flux_{axis}_global_sp_kernel"
            rhs = f"explicit_upwind_rhs_{axis}_global_sp_kernel"
            self.assertIn(f"call{flux}<<<", compact)
            self.assertIn(f"sync_after_kernel('{flux}", compact)
            self.assertIn(f"call{rhs}<<<", compact)
            self.assertIn(f"sync_after_kernel('{rhs}", compact)
        self.assertIn("explicit_upwind_flux_x_global_kernel<<<", compact)

    def test_smooth_periodic_tgv_upwind_has_a_named_capability_gate(self):
        compact = CAPABILITY.replace(" ", "").lower()
        self.assertIn("logicalfunctiongpu_tgv_explicit_upwind_supported", compact)
        self.assertIn("trim(flowtype)=='tgv'", compact)
        self.assertIn("trim(conschm)=='543e'", compact)
        self.assertIn("recon_schem==1.or.recon_schem==3", compact)
        self.assertIn(".not.lchardecomp", compact)
        self.assertIn(".not.lfilter", compact)
        self.assertIn(".not.diffterm", compact)
        self.assertIn("lihomo.and.ljhomo.and.lkhomo", compact)
        self.assertIn("gpu_tgv_explicit_upwind_supported()", MAINLOOP)

    def test_compare_driver_runs_cpu_fp64_gpu_fp64_and_gpu_mixed(self):
        self.assertIn("cpu_fp64", COMPARE_DRIVER)
        self.assertIn("gpu_fp64", COMPARE_DRIVER)
        self.assertIn("gpu_mixed", COMPARE_DRIVER)
        self.assertIn("--flowtype tgv", COMPARE_DRIVER)
        self.assertIn("--conschm 543e", COMPARE_DRIVER)
        self.assertIn("--lchardecomp f", COMPARE_DRIVER)
        self.assertIn('ASTR_GPU_PRECISION_MODE="fp64"', COMPARE_DRIVER)
        self.assertIn('ASTR_GPU_PRECISION_MODE="mixed_workspace"', COMPARE_DRIVER)
        self.assertIn("compare_flowfield_h5.py", COMPARE_DRIVER)
        self.assertIn("compare_flowstate.py", COMPARE_DRIVER)
        self.assertIn("gpu_fp64_vs_mixed", COMPARE_DRIVER)
        self.assertIn("cpu_vs_gpu_fp64", COMPARE_DRIVER)

    def test_compare_driver_separates_field_and_statistics_tolerances(self):
        self.assertIn('MIXED_FIELD_ATOL="${MIXED_FIELD_ATOL:-2e-6}"', COMPARE_DRIVER)
        self.assertIn('MIXED_FIELD_RTOL="${MIXED_FIELD_RTOL:-0}"', COMPARE_DRIVER)
        self.assertIn('MIXED_STATS_ATOL="${MIXED_STATS_ATOL:-5e-11}"', COMPARE_DRIVER)
        self.assertIn('MIXED_STATS_RTOL="${MIXED_STATS_RTOL:-0}"', COMPARE_DRIVER)

    def test_memcheck_driver_exercises_the_mixed_upwind_path(self):
        self.assertIn("compute-sanitizer --tool memcheck", MEMCHECK_DRIVER)
        self.assertIn("--conschm 543e", MEMCHECK_DRIVER)
        self.assertIn("--lchardecomp f", MEMCHECK_DRIVER)
        self.assertIn("ASTR_GPU_PRECISION_MODE=mixed_workspace", MEMCHECK_DRIVER)
        self.assertIn("ASTR_GPU_ACTIVE_MIXED_WORKSPACE=flux_work", MEMCHECK_DRIVER)
        self.assertIn("ERROR SUMMARY: 0 errors", MEMCHECK_DRIVER)

    def test_benchmark_driver_uses_paired_repeats_without_speedup_gate(self):
        self.assertIn('REPEATS="${REPEATS:-5}"', BENCHMARK_DRIVER)
        self.assertIn("ASTR_GPU_RK_TIMING=1", BENCHMARK_DRIVER)
        self.assertIn("ASTR_GPU_PRECISION_MODE", BENCHMARK_DRIVER)
        self.assertIn("fp64", BENCHMARK_DRIVER)
        self.assertIn("mixed_workspace", BENCHMARK_DRIVER)
        self.assertIn("summarize_mixed_precision_benchmark.py", BENCHMARK_DRIVER)


if __name__ == "__main__":
    unittest.main()
