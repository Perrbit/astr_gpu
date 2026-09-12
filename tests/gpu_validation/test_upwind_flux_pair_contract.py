import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODE_PATH = ROOT / "src_gpu" / "upwind_flux_mode_gpu.cuf"
MODE = MODE_PATH.read_text() if MODE_PATH.exists() else ""
CMAKE = (ROOT / "src" / "CMakeLists.txt").read_text()
SOLVER = (ROOT / "src_gpu" / "solver_gpu.cuf").read_text()
MAINLOOP = (ROOT / "src_gpu" / "mainloop_gpu.cuf").read_text()
COMPARE_PATH = ROOT / "tests" / "gpu_validation" / "run_tgv_upwind_flux_pair_compare.sh"
COMPARE = COMPARE_PATH.read_text() if COMPARE_PATH.exists() else ""
BENCHMARK_PATH = ROOT / "tests" / "gpu_validation" / "run_tgv_upwind_flux_pair_benchmark.sh"
BENCHMARK = BENCHMARK_PATH.read_text() if BENCHMARK_PATH.exists() else ""


class UpwindFluxPairContract(unittest.TestCase):
    def test_runtime_mode_defaults_to_split_and_accepts_fused(self):
        compact = MODE.replace(" ", "").lower()
        self.assertIn("moduleupwind_flux_mode_gpu", compact)
        self.assertIn("upwind_flux_split=0", compact)
        self.assertIn("upwind_flux_fused=1", compact)
        self.assertIn("astr_gpu_flux_pair_mode", compact)
        self.assertIn("case('split')", compact)
        self.assertIn("case('fused')", compact)
        self.assertIn("upwind_flux_mode=upwind_flux_split", compact)

    def test_runtime_mode_is_collectively_validated(self):
        compact = MODE.replace(" ", "").lower()
        self.assertGreaterEqual(compact.count("mpi_allreduce"), 2)
        self.assertIn("mpi_min", compact)
        self.assertIn("mpi_max", compact)
        self.assertIn("lowest/=highest", compact)
        self.assertIn("mpi_abort", compact)
        self.assertIn("invalidorinconsistentastr_gpu_flux_pair_mode", compact)

    def test_runtime_mode_exports_query_and_name(self):
        lower = MODE.lower()
        self.assertIn("subroutine configure_gpu_upwind_flux_mode", lower)
        self.assertIn("logical function gpu_fused_flux_pair_requested", lower)
        self.assertIn("function gpu_upwind_flux_mode_name", lower)

    def test_cmake_orders_mode_before_mainloop_and_builds_probe(self):
        mode = CMAKE.find("../src_gpu/upwind_flux_mode_gpu.cuf")
        mainloop = CMAKE.find("../src_gpu/mainloop_gpu.cuf")
        self.assertGreaterEqual(mode, 0)
        self.assertGreater(mainloop, mode)
        self.assertIn("upwind_flux_mode_setup_test", CMAKE)

    def test_pair_reconstruction_uses_eight_unique_cells(self):
        compact = SOLVER.replace(" ", "").lower()
        self.assertIn("subroutinesteger_warming_split_component_pair_at", compact)
        self.assertIn("subroutineexplicit_reconstruction_interface_flux_pair", compact)
        self.assertIn("dooffset=-3,4", compact)
        self.assertIn("plus_index=offset+4", compact)
        self.assertIn("minus_index=5-offset", compact)
        self.assertIn("offset>=-3.and.offset<=3", compact)
        self.assertIn("offset>=-2.and.offset<=4", compact)

    def test_fused_fp64_and_fp32_kernels_exist_for_each_axis(self):
        compact = SOLVER.replace(" ", "").lower()
        for axis in "xyz":
            fp64 = f"explicit_upwind_flux_{axis}_global_fused_kernel"
            fp32 = f"explicit_upwind_flux_{axis}_global_fused_sp_kernel"
            self.assertIn(f"subroutine{fp64}", compact)
            self.assertIn(f"subroutine{fp32}", compact)
        self.assertGreaterEqual(compact.count("real(real(fh_plus,4),8)+fh_minus"), 3)

    def test_mainloop_dispatches_one_fused_flux_kernel_and_sync_per_axis(self):
        compact = MAINLOOP.replace(" ", "").lower()
        self.assertIn("gpu_fused_flux_pair_requested", compact)
        for axis in "xyz":
            for suffix in ("fused_kernel", "fused_sp_kernel"):
                kernel = f"explicit_upwind_flux_{axis}_global_{suffix}"
                self.assertIn(f"call{kernel}<<<", compact)
                self.assertIn(f"sync_after_kernel('{kernel}')", compact)
        self.assertIn("explicit_upwind_flux_x_global_kernel<<<", compact)
        self.assertIn("explicit_upwind_flux_x_global_sp_kernel<<<", compact)

    def test_fused_dispatch_is_limited_to_admitted_periodic_tgv_route(self):
        compact = MAINLOOP.replace(" ", "").replace("&", "").replace("\n", "").lower()
        self.assertIn(
            "fused_flux_pair=gpu_fused_flux_pair_requested().and.gpu_tgv_explicit_upwind_supported()",
            compact,
        )

    def test_compare_driver_covers_both_pair_and_precision_modes(self):
        for label in ("fp64_split", "fp64_fused", "mixed_split", "mixed_fused"):
            self.assertIn(label, COMPARE)
        self.assertIn("ASTR_GPU_FLUX_PAIR_MODE", COMPARE)
        self.assertIn("ASTR_GPU_PRECISION_MODE", COMPARE)
        self.assertIn("compare_flowfield_h5.py", COMPARE)
        self.assertIn("compare_flowstate.py", COMPARE)

    def test_compare_driver_uses_precision_specific_field_thresholds(self):
        self.assertIn('FIELD_ATOL="${FIELD_ATOL:-1e-12}"', COMPARE)
        self.assertIn('MIXED_FIELD_ATOL="${MIXED_FIELD_ATOL:-1e-8}"', COMPARE)
        self.assertIn(
            "compare_pair fp64_split_vs_fused fp64_split fp64_fused \"$FIELD_ATOL\"",
            COMPARE,
        )
        self.assertIn(
            "compare_pair mixed_split_vs_fused mixed_split mixed_fused \"$MIXED_FIELD_ATOL\"",
            COMPARE,
        )

    def test_benchmark_driver_pairs_split_and_fused_without_speedup_gate(self):
        self.assertIn('REPEATS="${REPEATS:-5}"', BENCHMARK)
        self.assertIn("ASTR_GPU_RK_TIMING=1", BENCHMARK)
        self.assertIn("ASTR_GPU_FLUX_PAIR_MODE", BENCHMARK)
        self.assertIn("split", BENCHMARK)
        self.assertIn("fused", BENCHMARK)
        self.assertIn("summarize_flux_pair_benchmark.py", BENCHMARK)

    def test_benchmark_warmup_returns_success_under_errexit(self):
        self.assertNotIn('[[ "$record" == "t" ]] || return', BENCHMARK)
        self.assertIn("return 0", BENCHMARK)


if __name__ == "__main__":
    unittest.main()
