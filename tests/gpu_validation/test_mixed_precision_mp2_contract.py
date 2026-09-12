import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
CANDIDATE_PATH = ROOT / "src_gpu" / "mixed_candidate_gpu.cuf"
CANDIDATE = CANDIDATE_PATH.read_text(encoding="utf-8") if CANDIDATE_PATH.exists() else ""
CMAKE = (ROOT / "src" / "CMakeLists.txt").read_text(encoding="utf-8")
PROBE_PATH = ROOT / "tests" / "gpu_validation" / "mixed_candidate_setup_test.cuf"
PROBE = PROBE_PATH.read_text(encoding="utf-8") if PROBE_PATH.exists() else ""
COMMARRAY = (ROOT / "src_gpu" / "commarray_gpu.cuf").read_text(encoding="utf-8")
GRADCAL = (ROOT / "src_gpu" / "gradcal_gpu.cuf").read_text(encoding="utf-8")
SOLVER = (ROOT / "src_gpu" / "solver_gpu.cuf").read_text(encoding="utf-8")
STATISTIC = (ROOT / "src_gpu" / "statistic_gpu.cuf").read_text(encoding="utf-8")
PRODUCTION_STATS = (
    ROOT / "src_gpu" / "production_statistics_gpu.cuf"
).read_text(encoding="utf-8")
HALO = (ROOT / "src_gpu" / "halo_exchange_gpu.cuf").read_text(encoding="utf-8")
DERIVATIVE_TGV_DRIVER_PATH = (
    ROOT / "tests" / "gpu_validation" / "run_mp2_derivative_tgv_compare.sh"
)
DERIVATIVE_TGV_DRIVER = (
    DERIVATIVE_TGV_DRIVER_PATH.read_text(encoding="utf-8")
    if DERIVATIVE_TGV_DRIVER_PATH.exists()
    else ""
)
DERIVATIVE_DRIVER_NAMES = (
    "run_mp2_derivative_hbl_compare.sh",
    "run_mp2_derivative_curve_compare.sh",
    "run_mp2_derivative_mpi_matrix.sh",
    "run_mp2_derivative_memcheck.sh",
    "run_mp2_derivative_benchmark.sh",
)
DERIVATIVE_DRIVERS = {
    name: (ROOT / "tests" / "gpu_validation" / name).read_text(encoding="utf-8")
    if (ROOT / "tests" / "gpu_validation" / name).exists()
    else ""
    for name in DERIVATIVE_DRIVER_NAMES
}
MP2_SUMMARY_PATH = ROOT / "tests" / "gpu_validation" / "summarize_mp2_benchmark.py"
MP2_SUMMARY = (
    MP2_SUMMARY_PATH.read_text(encoding="utf-8") if MP2_SUMMARY_PATH.exists() else ""
)


class MixedPrecisionMp2Contract(unittest.TestCase):
    def test_candidate_selector_accepts_exactly_the_named_values(self):
        compact = CANDIDATE.replace(" ", "").lower()
        self.assertIn("modulemixed_candidate_gpu", compact)
        self.assertIn("astr_gpu_mixed_candidate", compact)
        self.assertIn("gpu_mixed_flux=0", compact)
        self.assertIn("gpu_mixed_derivative=1", compact)
        self.assertIn("gpu_mixed_viscous_flux=2", compact)
        self.assertIn("case('flux')", compact)
        self.assertIn("case('derivative')", compact)
        self.assertIn("case('viscous_flux')", compact)

    def test_unset_candidate_defaults_to_flux(self):
        compact = CANDIDATE.replace(" ", "").lower()
        self.assertIn("choice=gpu_mixed_flux", compact)
        self.assertIn("status==1", compact)
        self.assertIn("len_trim(value)==0", compact)

    def test_candidate_is_collectively_validated(self):
        compact = CANDIDATE.replace(" ", "").lower()
        self.assertGreaterEqual(compact.count("mpi_allreduce"), 2)
        self.assertIn("mpi_min", compact)
        self.assertIn("mpi_max", compact)
        self.assertIn("lowest/=highest", compact)
        self.assertIn("mpi_abort", compact)
        self.assertIn("invalidorinconsistentastr_gpu_mixed_candidate", compact)

    def test_fp64_rejects_explicit_nondefault_candidate(self):
        compact = CANDIDATE.replace(" ", "").lower()
        self.assertIn("gpu_mixed_workspace_requested()", compact)
        self.assertIn("candidate_was_explicit", compact)
        self.assertIn("choice/=gpu_mixed_flux", compact)
        self.assertIn("requiresastr_gpu_precision_mode=mixed_workspace", compact)

    def test_selector_exports_queries_and_rank_zero_provenance(self):
        lower = CANDIDATE.lower()
        self.assertIn("subroutine configure_gpu_mixed_candidate", lower)
        self.assertIn("integer function gpu_mixed_candidate_id", lower)
        self.assertIn("logical function gpu_mixed_candidate_requested", lower)
        self.assertIn("function gpu_mixed_candidate_name", lower)
        self.assertIn("mpi_comm_rank", lower)
        self.assertIn("astr_gpu_mixed_candidate=", lower)

    def test_cmake_orders_selector_before_device_arrays_and_builds_probe(self):
        precision = CMAKE.find("../src_gpu/precision_mode_gpu.cuf")
        candidate = CMAKE.find("../src_gpu/mixed_candidate_gpu.cuf")
        commarray = CMAKE.find("../src_gpu/commarray_gpu.cuf")
        self.assertGreaterEqual(precision, 0)
        self.assertGreater(candidate, precision)
        self.assertGreater(commarray, candidate)
        self.assertIn("mixed_candidate_setup_test", CMAKE)
        self.assertIn("MPI::MPI_Fortran", CMAKE)

    def test_probe_reports_candidate_queries(self):
        lower = PROBE.lower()
        self.assertIn("call configure_gpu_mixed_candidate()", lower)
        self.assertIn("gpu_mixed_candidate_id()", lower)
        self.assertIn("gpu_mixed_candidate_name()", lower)
        self.assertIn("gpu_mixed_candidate_requested", lower)

    def test_candidate_specific_arrays_are_declared(self):
        compact = COMMARRAY.replace(" ", "").lower()
        self.assertIn(
            "real(4),allocatable,device::dvel_sp_d(:,:,:,:,:),dtmp_sp_d(:,:,:,:)",
            compact,
        )
        self.assertIn(
            "real(4),allocatable,device::sigma_sp_d(:,:,:,:),qflux_sp_d(:,:,:,:)",
            compact,
        )

    def test_each_workspace_query_requires_its_selected_candidate(self):
        compact = COMMARRAY.replace(" ", "").lower()
        self.assertIn("logicalfunctiongpu_mixed_derivative_workspace_enabled", compact)
        self.assertIn("logicalfunctiongpu_mixed_viscous_flux_workspace_enabled", compact)
        self.assertGreaterEqual(compact.count("gpu_mixed_candidate_requested("), 3)
        self.assertIn("gpu_mixed_flux", compact)
        self.assertIn("gpu_mixed_derivative", compact)
        self.assertIn("gpu_mixed_viscous_flux", compact)

    def test_mp2_eligibility_is_a_hard_runtime_gate(self):
        compact = COMMARRAY.replace(" ", "").lower()
        self.assertIn("subroutinevalidate_gpu_mixed_candidate_eligibility", compact)
        self.assertIn(".not.diffterm", compact)
        self.assertIn("trim(difschm)/='643e'", compact)
        self.assertIn("numq/=5", compact)
        self.assertIn("num_species/=0", compact)
        self.assertIn("num_modequ/=0", compact)
        self.assertIn("trim(rkscheme)/='rk3'", compact)
        self.assertIn("shock_sensor_validation_enabled().or.lchardecomp", compact)
        self.assertIn("mpi_abort", compact)

    def test_derivative_and_viscous_allocations_are_mutually_exclusive(self):
        compact = COMMARRAY.replace(" ", "").lower()
        self.assertIn(
            "allocate(dvel_sp_d(0:im,0:jm,0:km,1:3,1:3))", compact
        )
        self.assertIn("allocate(dtmp_sp_d(0:im,0:jm,0:km,1:3))", compact)
        self.assertIn("allocate(dvel_d(0:im,0:jm,0:km,1:3,1:3))", compact)
        self.assertIn("allocate(dtmp_d(0:im,0:jm,0:km,1:3))", compact)
        self.assertIn(
            "allocate(sigma_sp_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,1:6))",
            compact,
        )
        self.assertIn(
            "allocate(qflux_sp_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,1:3))",
            compact,
        )
        self.assertIn(
            "allocate(sigma_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,1:6))", compact
        )
        self.assertIn(
            "allocate(qflux_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,1:3))", compact
        )
        self.assertIn("if(derivative_active)then", compact)
        self.assertIn("if(viscous_active)then", compact)

    def test_workspace_logs_report_exact_candidate_and_reference_bytes(self):
        compact = COMMARRAY.replace(" ", "").lower()
        self.assertIn("astr_gpu_active_mixed_workspace=", compact)
        self.assertIn("astr_gpu_mixed_workspace_bytes=", compact)
        self.assertIn("astr_gpu_fp64_workspace_bytes=", compact)
        self.assertIn("12_8*4_8", compact)
        self.assertIn("12_8*8_8", compact)
        self.assertIn("9_8*4_8", compact)
        self.assertIn("9_8*8_8", compact)

    def test_release_covers_all_optional_mp2_arrays(self):
        lower = COMMARRAY.lower()
        for name in ("dvel_sp_d", "dtmp_sp_d", "sigma_sp_d", "qflux_sp_d"):
            self.assertIn(f"if(allocated({name})) deallocate({name})", lower)

    def test_all_gradcal_variants_have_fp32_final_store_kernels(self):
        lower = GRADCAL.lower()
        for suffix in ("", "_xphysical", "_yphysical", "_xyphysical", "_zphysical"):
            self.assertIn(f"subroutine gradcal_dvel{suffix}_sp_kernel", lower)
        compact = GRADCAL.replace(" ", "").lower()
        self.assertGreaterEqual(compact.count("dvel_sp_d(i,j,k,a,b)=real(value,4)"), 5)
        self.assertGreaterEqual(compact.count("dtmp_sp_d(i,j,k,b)=real(value,4)"), 5)

    def test_gradcal_dispatch_selects_one_precision_and_syncs_each_sp_launch(self):
        compact = GRADCAL.replace(" ", "").lower()
        self.assertIn("gpu_mixed_derivative_workspace_enabled()", compact)
        for suffix in ("", "_xphysical", "_yphysical", "_xyphysical", "_zphysical"):
            kernel = f"gradcal_dvel{suffix}_sp_kernel"
            self.assertIn(f"call{kernel}<<<", compact)
            self.assertIn(f"sync_after_kernel('{kernel}')", compact)
        self.assertIn("gradcal_dvel_kernel<<<", compact)

    def test_solver_keeps_direct_fp64_diffusion_gradient_recomputation(self):
        compact = SOLVER.replace(" ", "").lower()
        self.assertNotIn("dvel_sp_d", compact)
        self.assertNotIn("dtmp_sp_d", compact)
        for suffix in ("", "_xphysical", "_yphysical", "_zphysical", "_xyphysical"):
            self.assertIn(f"callgradient_at{suffix}(", compact)

    def test_all_derivative_statistics_have_fp32_reader_kernels(self):
        compact = STATISTIC.replace(" ", "").lower()
        for name in ("enstophy", "dissipation", "channel", "s1_bl"):
            kernel = f"{name}_partial_sp_kernel"
            self.assertIn(f"subroutine{kernel}", compact)
            self.assertIn(f"call{kernel}<<<", compact)
            self.assertIn(f"sync_after_kernel('{kernel}'", compact)
        self.assertGreaterEqual(compact.count("real(dvel("), 4)
        self.assertIn("real(dtmp(", compact)
        self.assertIn("gpu_mixed_derivative_workspace_enabled()", compact)

    def test_compact_wall_statistics_have_an_fp32_reader(self):
        compact = PRODUCTION_STATS.replace(" ", "").lower()
        self.assertIn("subroutinecompact_wall_point_values_sp", compact)
        self.assertIn("subroutinecompact_wall_moment_sp_kernel", compact)
        self.assertIn("real(dvel(", compact)
        self.assertIn("real(dtmp(", compact)
        self.assertIn("gpu_mixed_derivative_workspace_enabled()", compact)
        self.assertIn("callcompact_wall_moment_sp_kernel<<<", compact)
        self.assertIn("sync_after_kernel('compact_wall_moment_sp_kernel')", compact)

    def test_gradcal_comparison_promotes_fp32_values_on_host(self):
        compact = GRADCAL.replace(" ", "").lower()
        self.assertIn("real(4),allocatable::dvel_sp_h", compact)
        self.assertIn("dvel_sp_h=dvel_sp_d", compact)
        self.assertIn("real(dvel_sp_h(i,j,k,a,b),8)", compact)

    def test_derivative_tgv_driver_uses_diffusive_central_case_and_exact_fields(self):
        text = DERIVATIVE_TGV_DRIVER
        self.assertIn("--conschm 643e", text)
        self.assertIn("--diffterm t", text)
        self.assertIn("--difschm 643e", text)
        self.assertIn('ASTR_GPU_MIXED_CANDIDATE="$candidate"', text)
        self.assertIn('CANDIDATE="${MP2_CANDIDATE:-derivative}"', text)
        self.assertIn('run_case "$CANDIDATE_TARGET" mixed_workspace "$CANDIDATE"', text)
        self.assertIn('FIELD_ATOL="${FIELD_ATOL:-0}"', text)
        self.assertIn('FIELD_RTOL="${FIELD_RTOL:-0}"', text)
        self.assertIn("compare_flowfield_h5.py", text)
        self.assertIn("compare_flowstate.py", text)
        self.assertIn("ASTR_GPU_ACTIVE_MIXED_WORKSPACE=${CANDIDATE}", text)

    def test_derivative_admission_drivers_are_present_and_non_overwriting(self):
        for name, text in DERIVATIVE_DRIVERS.items():
            self.assertTrue(text, name)
            self.assertIn("set -euo pipefail", text, name)
            self.assertIn("refusing to overwrite", text, name)

    def test_derivative_hbl_driver_checks_runtime_wall_and_field_diagnostics(self):
        text = DERIVATIVE_DRIVERS["run_mp2_derivative_hbl_compare.sh"]
        self.assertIn("prepare_s1_flatplate_case.py", text)
        self.assertIn("generate_compressible_blasius_profile.py", text)
        self.assertIn("run_case gpu_fp64 fp64 flux", text)
        self.assertIn('run_case "$CANDIDATE_TARGET" mixed_workspace "$CANDIDATE"', text)
        self.assertIn("compare_flowstate.py", text)
        self.assertIn("compare_flowfield_h5.py", text)
        self.assertIn("analyze_curvilinear_hbl_physics.py", text)
        self.assertIn("compare_hbl_diagnostics.py", text)
        self.assertIn("fbcx", text)
        self.assertIn("wallheatflux", text)

    def test_derivative_curve_driver_covers_uniform_acoustic_and_viscous_hbl(self):
        text = DERIVATIVE_DRIVERS["run_mp2_derivative_curve_compare.sh"]
        for case in ("uniform", "acoustic", "viscous_hbl"):
            self.assertIn(case, text)
        self.assertIn("generate_curvilinear_tgv_grid.py", text)
        self.assertIn("generate_curvilinear_acoustic_pulse.py", text)
        self.assertIn("run_mp2_derivative_hbl_compare.sh", text)
        self.assertIn("compare_flowfield_h5.py", text)
        self.assertIn("compare_flowstate.py", text)
        self.assertIn("ASTR_GPU_ACTIVE_MIXED_WORKSPACE=${CANDIDATE}", text)

    def test_derivative_mpi_matrix_locks_np2_and_np4_topologies(self):
        text = DERIVATIVE_DRIVERS["run_mp2_derivative_mpi_matrix.sh"]
        self.assertIn("NP=2 TOPOLOGY=2,1,1", text)
        self.assertIn("NP=4 TOPOLOGY=2,2,1", text)
        self.assertIn("run_mp2_derivative_tgv_compare.sh", text)

    def test_derivative_memcheck_requires_zero_errors(self):
        text = DERIVATIVE_DRIVERS["run_mp2_derivative_memcheck.sh"]
        self.assertIn("compute-sanitizer --tool memcheck", text)
        self.assertIn("--error-exitcode", text)
        self.assertIn("ASTR_GPU_PRECISION_MODE=mixed_workspace", text)
        self.assertIn('CANDIDATE="${MP2_CANDIDATE:-derivative}"', text)
        self.assertIn('ASTR_GPU_MIXED_CANDIDATE="$CANDIDATE"', text)
        self.assertIn("ERROR SUMMARY: 0 errors", text)

    def test_derivative_benchmark_is_interleaved_and_has_no_speedup_gate(self):
        text = DERIVATIVE_DRIVERS["run_mp2_derivative_benchmark.sh"]
        self.assertIn('REPEATS="${REPEATS:-5}"', text)
        self.assertIn("REPEATS must be at least 5", text)
        self.assertIn("repeat % 2", text)
        self.assertIn("ASTR_GPU_RK_TIMING=1", text)
        self.assertIn("summarize_mp2_benchmark.py", text)
        self.assertNotIn("MIN_SPEEDUP", text)

    def test_mp2_summary_reports_spread_and_exact_workspace_byte_accounting(self):
        self.assertIn("run-to-run spread", MP2_SUMMARY.lower())
        self.assertIn("expected_fp64_workspace_bytes", MP2_SUMMARY)
        self.assertIn("expected_mixed_workspace_bytes", MP2_SUMMARY)
        self.assertIn("no speedup acceptance threshold", MP2_SUMMARY.lower())
        self.assertIn("candidate_name=candidate_rows[0].label", MP2_SUMMARY.replace(" ", ""))
        self.assertNotIn("# MP2 Derivative Workspace Benchmark", MP2_SUMMARY)
        self.assertIn("raise ValueError", MP2_SUMMARY)

    def test_viscous_flux_has_all_fp32_writer_variants(self):
        lower = SOLVER.lower()
        for suffix in ("", "_xphysical", "_yphysical", "_zphysical", "_xyphysical"):
            name = f"diffusion_flux{suffix}_sp_global_kernel"
            self.assertIn(f"subroutine {name}", lower)
            self.assertIn(f"call {name}<<<", DERIVATIVE_DRIVERS.get("mainloop", "") + (ROOT / "src_gpu" / "mainloop_gpu.cuf").read_text(encoding="utf-8").lower())
        compact = SOLVER.replace(" ", "").lower()
        self.assertGreaterEqual(compact.count("sigma_sp_d(i,j,k,1)=real(tau11,4)"), 5)
        self.assertGreaterEqual(compact.count("qflux_sp_d(i,j,k,1)=real("), 5)

    def test_viscous_flux_has_all_fp32_stored_flux_helpers(self):
        compact = SOLVER.replace(" ", "").lower()
        for suffix in ("", "_xphysical", "_yphysical", "_zphysical", "_xyphysical"):
            self.assertIn(f"subroutinestored_diff_flux_at{suffix}_sp", compact)
        self.assertGreaterEqual(compact.count("real(sigma_sp_d("), 20)
        self.assertGreaterEqual(compact.count("real(qflux_sp_d("), 5)

    def test_viscous_flux_has_all_thirteen_fp32_rhs_kernels(self):
        lower = SOLVER.lower()
        names = (
            "diffusion_rhs_x_stored_sp_global_kernel",
            "diffusion_rhs_y_stored_sp_global_kernel",
            "diffusion_rhs_z_stored_sp_global_kernel",
            "diffusion_rhs_x_xphysical_stored_sp_global_kernel",
            "diffusion_rhs_x_yphysical_stored_sp_global_kernel",
            "diffusion_rhs_y_yphysical_stored_sp_global_kernel",
            "diffusion_rhs_z_yphysical_stored_sp_global_kernel",
            "diffusion_rhs_x_zphysical_stored_sp_global_kernel",
            "diffusion_rhs_y_zphysical_stored_sp_global_kernel",
            "diffusion_rhs_z_zphysical_stored_sp_global_kernel",
            "diffusion_rhs_x_xyphysical_stored_sp_global_kernel",
            "diffusion_rhs_y_xyphysical_stored_sp_global_kernel",
            "diffusion_rhs_z_xyphysical_stored_sp_global_kernel",
        )
        for name in names:
            self.assertIn(f"subroutine {name}", lower)

    def test_mainloop_dispatches_viscous_flux_precision_and_syncs_sp_kernels(self):
        mainloop = (ROOT / "src_gpu" / "mainloop_gpu.cuf").read_text(encoding="utf-8").lower()
        compact = mainloop.replace(" ", "")
        self.assertIn("gpu_mixed_viscous_flux_workspace_enabled", compact)
        for name in (
            "diffusion_flux_sp_global_kernel",
            "diffusion_flux_xphysical_sp_global_kernel",
            "diffusion_flux_yphysical_sp_global_kernel",
            "diffusion_flux_zphysical_sp_global_kernel",
            "diffusion_flux_xyphysical_sp_global_kernel",
            "diffusion_rhs_x_stored_sp_global_kernel",
            "diffusion_rhs_y_stored_sp_global_kernel",
            "diffusion_rhs_z_stored_sp_global_kernel",
            "diffusion_rhs_x_xphysical_stored_sp_global_kernel",
            "diffusion_rhs_x_yphysical_stored_sp_global_kernel",
            "diffusion_rhs_y_yphysical_stored_sp_global_kernel",
            "diffusion_rhs_z_yphysical_stored_sp_global_kernel",
            "diffusion_rhs_x_zphysical_stored_sp_global_kernel",
            "diffusion_rhs_y_zphysical_stored_sp_global_kernel",
            "diffusion_rhs_z_zphysical_stored_sp_global_kernel",
            "diffusion_rhs_x_xyphysical_stored_sp_global_kernel",
            "diffusion_rhs_y_xyphysical_stored_sp_global_kernel",
            "diffusion_rhs_z_xyphysical_stored_sp_global_kernel",
        ):
            self.assertIn(f"call{name}<<<", compact)
            self.assertIn(f"sync_after_kernel('{name}'", compact)

    def test_viscous_flux_fp32_halo_converts_through_fp64_transport_buffers(self):
        compact = HALO.replace(" ", "").lower()
        kernels = []
        for axis, sides in (
            ("x", ("left", "right")),
            ("y", ("down", "up")),
            ("z", ("back", "front")),
        ):
            for side in sides:
                kernels.append(f"pack_field_sp_{axis}_{side}_kernel")
            kernels.append(f"unpack_field_sp_{axis}_kernel")
            kernels.append(f"field_sp_qswap_{axis}_kernel")
        for kernel in kernels:
            self.assertIn(f"subroutine{kernel}", compact)
            self.assertIn(f"call{kernel}<<<", compact)
            self.assertIn(f"sync_after_kernel('{kernel}'", compact)
        self.assertIn("real(4),device::field(", compact)
        self.assertIn("real(8),device::sendbuf(", compact)
        self.assertGreaterEqual(compact.count("=real(field("), 6)
        self.assertGreaterEqual(compact.count("=real(recv_"), 6)

    def test_viscous_flux_fp32_halo_preserves_transport_contract(self):
        compact = HALO.replace(" ", "").lower()
        self.assertIn("subroutineexchange_diffusion_flux_sp_halo_gpu(work)", compact)
        self.assertIn("subroutineexchange_field_sp_halo_gpu(field,nvar,work)", compact)
        for axis in "xyz":
            self.assertIn(f"subroutineexchange_field_sp_{axis}_halo_gpu", compact)
        self.assertIn("callexchange_host_pair(", compact)
        self.assertIn("tag_x_first,tag_x_second", compact)
        self.assertIn("tag_y_first,tag_y_second", compact)
        self.assertIn("tag_z_first,tag_z_second", compact)
        self.assertIn("astr_gpu_viscous_flux_halo_payload=fp64", compact)

    def test_mainloop_selects_fp32_viscous_flux_halo(self):
        compact = (ROOT / "src_gpu" / "mainloop_gpu.cuf").read_text(
            encoding="utf-8"
        ).replace(" ", "").lower()
        self.assertIn("exchange_diffusion_flux_sp_halo_gpu", compact)
        self.assertIn("if(mixed_viscous_flux_workspace)then", compact)
        self.assertIn("callexchange_diffusion_flux_sp_halo_gpu(diffusion_interior_work)", compact)
        self.assertIn("callexchange_diffusion_flux_sp_halo_gpu()", compact)

    def test_viscous_flux_validation_drivers_exist(self):
        for name in (
            "run_mp2_viscous_flux_tgv_compare.sh",
            "run_mp2_viscous_flux_hbl_compare.sh",
            "run_mp2_viscous_flux_curve_compare.sh",
            "run_mp2_viscous_flux_mpi_matrix.sh",
            "run_mp2_viscous_flux_memcheck.sh",
            "run_mp2_viscous_flux_benchmark.sh",
        ):
            path = ROOT / "tests" / "gpu_validation" / name
            self.assertTrue(path.is_file(), name)
            self.assertIn("viscous_flux", path.read_text(encoding="utf-8").lower())


if __name__ == "__main__":
    unittest.main()
