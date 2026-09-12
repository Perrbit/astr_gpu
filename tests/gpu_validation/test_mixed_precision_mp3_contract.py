import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


def read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


CANDIDATE = read(ROOT / "src_gpu" / "mixed_candidate_gpu.cuf")
COMMARRAY = read(ROOT / "src_gpu" / "commarray_gpu.cuf")
SOLVER = read(ROOT / "src_gpu" / "solver_gpu.cuf")
MAINLOOP = read(ROOT / "src_gpu" / "mainloop_gpu.cuf")
PROBE = read(ROOT / "tests" / "gpu_validation" / "mixed_candidate_setup_test.cuf")
COMPARE_DRIVER = read(
    ROOT / "tests" / "gpu_validation" / "run_mp3_characteristic_flux_compare.sh"
)
MPI_DRIVER = read(
    ROOT / "tests" / "gpu_validation" / "run_mp3_characteristic_flux_mpi_matrix.sh"
)
MEMCHECK_DRIVER = read(
    ROOT / "tests" / "gpu_validation" / "run_mp3_characteristic_flux_memcheck.sh"
)
BENCHMARK_DRIVER = read(
    ROOT / "tests" / "gpu_validation" / "run_mp3_characteristic_flux_benchmark.sh"
)
XPHYSICAL_COMPARE_DRIVER = read(
    ROOT
    / "tests"
    / "gpu_validation"
    / "run_mp3_characteristic_flux_xphysical_compare.sh"
)
XPHYSICAL_MATRIX_DRIVER = read(
    ROOT
    / "tests"
    / "gpu_validation"
    / "run_mp3_characteristic_flux_xphysical_matrix.sh"
)
XPHYSICAL_MEMCHECK_DRIVER = read(
    ROOT
    / "tests"
    / "gpu_validation"
    / "run_mp3_characteristic_flux_xphysical_memcheck.sh"
)
BENCHMARK_SUMMARY = read(
    ROOT / "tests" / "gpu_validation" / "summarize_mp2_benchmark.py"
)
VALIDATION_README = read(ROOT / "tests" / "gpu_validation" / "README.md")
VALIDATION_MATRIX = read(ROOT / "documents" / "GPU_VALIDATION_MATRIX.md")
CURRENT_STATUS = read(
    ROOT / "documents" / "ASTR_GPU_CURRENT_STATUS_AND_NEXT_TARGETS.md"
)
ARCHITECTURE_PLAN = read(ROOT / "documents" / "ASTR_FULL_GPU_ARCHITECTURE_PLAN.md")


class MixedPrecisionMp3Contract(unittest.TestCase):
    def test_selector_accepts_characteristic_flux_as_fourth_candidate(self):
        compact = CANDIDATE.replace(" ", "").lower()
        self.assertIn("gpu_mixed_characteristic_flux=3", compact)
        self.assertIn("case('characteristic_flux')", compact)
        self.assertIn("gpu_mixed_candidate_name='characteristic_flux'", compact)

    def test_characteristic_flux_has_a_strict_eligibility_gate(self):
        compact = COMMARRAY.replace(" ", "").lower()
        self.assertIn(
            "logicalfunctiongpu_mixed_characteristic_flux_workspace_enabled", compact
        )
        self.assertIn("gpu_mixed_characteristic_flux", compact)
        for term in (
            "trim(conschm)=='543e'",
            "recon_schem==3",
            "lchardecomp",
            "lihomo.and.ljhomo.and.lkhomo",
            ".not.diffterm",
            ".not.lfilter",
            "numq==5",
            "num_species==0",
            "num_modequ==0",
            "trim(rkscheme)=='rk3'",
        ):
            self.assertIn(term, compact)
        self.assertIn("requestedmixedcandidateisineligible", compact)
        self.assertIn("mpi_abort", compact)

    def test_characteristic_flux_admits_only_the_existing_xphysical_case_gate(self):
        compact = COMMARRAY.replace(" ", "").lower()
        self.assertIn(
            "usecase_capability_gpu,only:gpu_shock_characteristic_s0b0_xphysical_supported",
            compact,
        )
        self.assertIn(
            "gpu_shock_characteristic_s0b0_xphysical_supported()", compact
        )

    def test_probe_reports_characteristic_flux_query(self):
        compact = PROBE.replace(" ", "").lower()
        self.assertIn("gpu_mixed_characteristic_flux", compact)
        self.assertIn("characteristic_flux=", compact)

    def test_probe_configures_precision_before_candidate(self):
        compact = PROBE.replace(" ", "").lower()
        precision = compact.find("callconfigure_gpu_precision_mode()")
        candidate = compact.find("callconfigure_gpu_mixed_candidate()")
        self.assertGreaterEqual(precision, 0)
        self.assertGreater(candidate, precision)

    def test_characteristic_workspace_allocations_are_mutually_exclusive(self):
        compact = COMMARRAY.replace(" ", "").lower()
        self.assertIn(
            "real(4),allocatable,device::flux_characteristic_work_sp_d(:,:,:,:)",
            compact,
        )
        self.assertIn("if(characteristic_active)then", compact)
        self.assertIn(
            "allocate(flux_characteristic_work_sp_d(-hm:im+hm,-hm:jm+hm,"
            "-hm:km+hm,1:5))",
            compact,
        )
        self.assertIn(
            "allocate(flux_characteristic_work_d(-hm:im+hm,-hm:jm+hm,"
            "-hm:km+hm,1:5))",
            compact,
        )
        self.assertIn("halo_points*5_8*4_8", compact)
        self.assertIn("halo_points*5_8*8_8", compact)

    def test_release_deallocates_both_characteristic_workspace_types(self):
        lower = COMMARRAY.lower()
        self.assertIn(
            "if(allocated(flux_characteristic_work_sp_d)) "
            "deallocate(flux_characteristic_work_sp_d)",
            lower,
        )
        self.assertIn(
            "if(allocated(flux_characteristic_work_d)) "
            "deallocate(flux_characteristic_work_d)",
            lower,
        )

    def test_periodic_characteristic_fp32_kernels_exist(self):
        lower = SOLVER.lower()
        for axis in "xyz":
            self.assertIn(
                f"subroutine characteristic_upwind_flux_{axis}_global_sp_kernel",
                lower,
            )
            self.assertIn(
                f"subroutine characteristic_upwind_rhs_{axis}_global_sp_kernel",
                lower,
            )
        compact = SOLVER.replace(" ", "").lower()
        self.assertGreaterEqual(compact.count("real(8)::fh(5)"), 3)
        self.assertGreaterEqual(
            compact.count(
                "flux_characteristic_work_sp_d(i,j,k,m)=real(fh(m),4)"
            ),
            3,
        )
        self.assertGreaterEqual(
            compact.count("real(flux_characteristic_work_sp_d("), 6
        )

    def test_xphysical_characteristic_fp32_kernels_preserve_fp64_algebra(self):
        lower = SOLVER.lower()
        self.assertIn(
            "subroutine characteristic_upwind_flux_x_physical_global_sp_kernel",
            lower,
        )
        self.assertIn(
            "subroutine characteristic_upwind_rhs_x_physical_global_sp_kernel",
            lower,
        )
        compact = SOLVER.replace(" ", "").lower()
        self.assertIn(
            "gamma,hm,npdci,1,fh)", compact
        )
        self.assertIn(
            "flux_characteristic_work_sp_d(i,j,k,m)=real(fh(m),4)", compact
        )
        self.assertIn(
            "real(flux_characteristic_work_sp_d(i-1,j,k,m),8)", compact
        )

    def test_mainloop_dispatches_periodic_and_xphysical_characteristic_sp_kernels(self):
        compact = MAINLOOP.replace(" ", "").lower()
        self.assertIn("mixed_characteristic_flux_workspace", compact)
        for axis in "xyz":
            for kind in ("flux", "rhs"):
                name = f"characteristic_upwind_{kind}_{axis}_global_sp_kernel"
                self.assertIn(f"call{name}<<<", compact)
                self.assertIn(f"sync_after_kernel('{name}')", compact)
        for kind in ("flux", "rhs"):
            name = f"characteristic_upwind_{kind}_x_physical_global_sp_kernel"
            self.assertIn(f"call{name}<<<", compact)
            self.assertIn(f"sync_after_kernel('{name}')", compact)
        self.assertNotIn(
            "characteristic_upwind_flux_x_xyphysical_global_sp_kernel", compact
        )

    def test_compare_driver_has_three_references_and_exact_candidate_sensor_gate(
        self,
    ):
        text = COMPARE_DRIVER
        for target in ("cpu", "gpu_fp64", "gpu_characteristic_flux"):
            self.assertIn(target, text)
        self.assertIn('ASTR_GPU_PRECISION_MODE="$precision"', text)
        self.assertIn('ASTR_GPU_MIXED_CANDIDATE="$candidate"', text)
        self.assertIn("--atol 0 --rtol 0", text)
        self.assertIn("ASTR_GPU_ACTIVE_MIXED_WORKSPACE=characteristic_flux", text)
        self.assertIn("refusing to overwrite", text)
        self.assertIn("CALIBRATE", text)

    def test_compare_driver_normalizes_output_before_case_directory_changes(self):
        compact = COMPARE_DRIVER.replace(" ", "")
        normalize = compact.find('if[["$OUT_DIR"!=/*]];then')
        run_case = compact.find("run_case(){")
        self.assertGreaterEqual(normalize, 0)
        self.assertGreater(run_case, normalize)

    def test_compare_driver_enables_local_multi_rank_mpi_transports(self):
        self.assertIn("OMPI_MCA_btl=self,vader,tcp", COMPARE_DRIVER)
        self.assertIn("OMPI_MCA_osc=pt2pt", COMPARE_DRIVER)

    def test_mpi_matrix_locks_all_characteristic_topologies(self):
        for line in (
            "NP=2 TOPOLOGY=2,1,1 GRID=400,8,8 SHOCK_X=0.d0 RANKWISE=f",
            "NP=2 TOPOLOGY=1,2,1 GRID=400,16,8 RANKWISE=t",
            "NP=2 TOPOLOGY=1,1,2 GRID=400,8,16 RANKWISE=t",
            "NP=8 TOPOLOGY=2,2,2 GRID=400,16,16 RANKWISE=t",
        ):
            self.assertIn(line, MPI_DRIVER)
        self.assertIn("TOLERANCE_FILE", MPI_DRIVER)
        self.assertNotIn("CALIBRATE=t", MPI_DRIVER)

    def test_memcheck_requires_zero_errors(self):
        self.assertIn("compute-sanitizer --tool memcheck", MEMCHECK_DRIVER)
        self.assertIn("--error-exitcode", MEMCHECK_DRIVER)
        self.assertIn("ERROR SUMMARY: 0 errors", MEMCHECK_DRIVER)
        self.assertIn("ASTR_GPU_MIXED_CANDIDATE=characteristic_flux", MEMCHECK_DRIVER)
        self.assertIn("OMPI_MCA_btl=self", MEMCHECK_DRIVER)
        self.assertIn("OMPI_MCA_osc=pt2pt", MEMCHECK_DRIVER)

    def test_xphysical_compare_driver_is_three_way_and_untrimmed(self):
        text = XPHYSICAL_COMPARE_DRIVER
        for target in ("cpu", "gpu_fp64", "gpu_characteristic_flux"):
            self.assertIn(target, text)
        self.assertIn("--homogeneous f,t,t", text)
        self.assertIn("--bctype 50,50,1,1,1,1", text)
        self.assertIn("--lchardecomp t", text)
        self.assertIn("ASTR_GPU_ACTIVE_MIXED_WORKSPACE=characteristic_flux", text)
        self.assertIn("--atol 0 --rtol 0", text)
        self.assertNotIn("--trim-boundaries", text)

    def test_xphysical_matrix_locks_np1_and_x_slab(self):
        text = XPHYSICAL_MATRIX_DRIVER
        self.assertIn("TOLERANCE_FILE", text)
        self.assertIn("NP=1 TOPOLOGY=1,1,1", text)
        self.assertIn("NP=2 TOPOLOGY=2,1,1", text)
        self.assertNotIn("CALIBRATE=t", text)

    def test_xphysical_memcheck_requires_two_clean_ranks(self):
        text = XPHYSICAL_MEMCHECK_DRIVER
        self.assertIn("mpirun -np 2", text)
        self.assertIn("ASTR_FORCE_MPI_TOPOLOGY=2,1,1", text)
        self.assertIn("compute-sanitizer --tool memcheck", text)
        self.assertIn("--error-exitcode 99", text)
        self.assertIn("ERROR SUMMARY: 0 errors", text)
        self.assertIn("ASTR_GPU_ACTIVE_MIXED_WORKSPACE=characteristic_flux", text)
        self.assertIn("OMPI_MCA_opal_cuda_support=0", text)
        self.assertIn("UCX_MEMTYPE_CACHE=n", text)
        self.assertIn("OMPI_MCA_btl=self,tcp", text)

    def test_benchmark_is_interleaved_and_has_exact_five_component_bytes(self):
        self.assertIn('REPEATS="${REPEATS:-5}"', BENCHMARK_DRIVER)
        self.assertIn("repeat % 2", BENCHMARK_DRIVER)
        self.assertIn("ASTR_GPU_RK_TIMING=1", BENCHMARK_DRIVER)
        self.assertIn("*5)", BENCHMARK_DRIVER)
        self.assertIn("--phase MP3", BENCHMARK_DRIVER)
        self.assertNotIn("MIN_SPEEDUP", BENCHMARK_DRIVER)

    def test_benchmark_summary_accepts_mp3_without_changing_mp2_default(self):
        self.assertIn('parser.add_argument("--phase", default="MP2"', BENCHMARK_SUMMARY)
        self.assertIn('f"# {phase} {candidate_name} Workspace Benchmark"', BENCHMARK_SUMMARY)
        self.assertIn('{"MP2", "MP3"}', BENCHMARK_SUMMARY)

    def test_mp3_local_classification_is_synchronized_across_documents(self):
        for text in (
            VALIDATION_README,
            VALIDATION_MATRIX,
            CURRENT_STATUS,
            ARCHITECTURE_PLAN,
        ):
            self.assertIn("characteristic_flux", text)
            self.assertIn("local-pass-not-promoted", text)
            self.assertIn("S0-A6", text)
            self.assertIn("S0-A10", text)
        for driver in (
            "run_mp3_characteristic_flux_compare.sh",
            "run_mp3_characteristic_flux_mpi_matrix.sh",
            "run_mp3_characteristic_flux_memcheck.sh",
            "run_mp3_characteristic_flux_benchmark.sh",
        ):
            self.assertIn(driver, VALIDATION_README)

    def test_mp3_xphysical_evidence_is_synchronized_across_documents(self):
        for text in (
            VALIDATION_README,
            VALIDATION_MATRIX,
            CURRENT_STATUS,
            ARCHITECTURE_PLAN,
        ):
            self.assertIn("MP3-XP2", text)
            self.assertIn("x-physical-local-pass-not-promoted", text)
            self.assertIn("1.5973888878306752e-7", text)
            self.assertIn("1.0126266403176487e-7", text)
        for driver in (
            "run_mp3_characteristic_flux_xphysical_compare.sh",
            "run_mp3_characteristic_flux_xphysical_matrix.sh",
            "run_mp3_characteristic_flux_xphysical_memcheck.sh",
        ):
            self.assertIn(driver, VALIDATION_README)


if __name__ == "__main__":
    unittest.main()
