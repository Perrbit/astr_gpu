import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
COMMARRAY = (ROOT / "src_gpu" / "commarray_gpu.cuf").read_text()
SOLVER = (ROOT / "src_gpu" / "solver_gpu.cuf").read_text()
HALO = (ROOT / "src_gpu" / "halo_exchange_gpu.cuf").read_text()
MAINLOOP = (ROOT / "src_gpu" / "mainloop_gpu.cuf").read_text()
STATS_DRIVER = (ROOT / "tests" / "gpu_validation" / "run_tgv_stats_compare.sh").read_text()
FIELD_DRIVER = (ROOT / "tests" / "gpu_validation" / "run_tgv_field_compare.sh").read_text()
MPI_STATS_DRIVER = (ROOT / "tests" / "gpu_validation" / "run_tgv_mpirank2_stats_compare.sh").read_text()
MPI_FIELD_DRIVER = (ROOT / "tests" / "gpu_validation" / "run_tgv_mpirank2_field_compare.sh").read_text()
CHANNEL_DRIVER = (ROOT / "tests" / "gpu_validation" / "run_channel_phased_compare.sh").read_text()
LDC_DRIVER = (ROOT / "tests" / "gpu_validation" / "run_ldcavity_phaseia_compare.sh").read_text()
CURVE_DRIVER = (ROOT / "tests" / "gpu_validation" / "run_curvilinear_tgv_compare.sh").read_text()
MEMCHECK_DRIVER = (ROOT / "tests" / "gpu_validation" / "run_tgv_gpu_memcheck.sh").read_text()


class ScalarFilterWorkspaceContract(unittest.TestCase):
    def test_runtime_mode_defaults_to_full_and_is_rank_consistent(self):
        self.assertIn("ASTR_GPU_FILTER_WORKSPACE", COMMARRAY)
        self.assertIn("GPU_FILTER_WORKSPACE_FULL", COMMARRAY)
        self.assertIn("GPU_FILTER_WORKSPACE_SCALAR", COMMARRAY)
        self.assertIn("mpi_allreduce", COMMARRAY.lower())
        self.assertIn("lowest/=highest", COMMARRAY.lower())

    def test_scalar_storage_is_one_haloed_three_dimensional_field(self):
        self.assertRegex(
            COMMARRAY,
            r"real\(8\),allocatable,device\s*::\s*filter_work_d\s*\(:,:,:,?\)",
        )
        self.assertIn("gpu_scalar_filter_workspace_enabled", COMMARRAY)
        self.assertIn("allocate(filter_work_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm))", COMMARRAY)
        self.assertIn(
            "if(gpu_scalar_filter_workspace_enabled().and.lfilter)",
            COMMARRAY.replace(" ", ""),
        )

    def test_scalar_kernels_are_component_scoped(self):
        for name in (
            "filter_x_scalar_global_kernel",
            "filter_y_scalar_to_q_halo_global_kernel",
            "filter_z_scalar_global_kernel",
            "copy_filter_scalar_to_q_global_kernel",
        ):
            self.assertIn(name, SOLVER)
        self.assertIn("integer,value :: component", SOLVER)

    def test_scalar_halo_entrypoints_are_axis_and_component_scoped(self):
        self.assertIn("exchange_filter_scalar_y_halo_gpu", HALO)
        self.assertIn("refresh_filter_scalar_y_local_halo_gpu", HALO)
        self.assertIn("exchange_filter_scalar_z_halo_gpu", HALO)
        self.assertIn("refresh_filter_scalar_z_local_halo_gpu", HALO)

    def test_mainloop_dispatches_without_removing_full_backend(self):
        self.assertIn("apply_explicit_filter_gpu", MAINLOOP)
        self.assertIn("gpu_scalar_filter_workspace_enabled", MAINLOOP)
        self.assertIn("filter_x_global_kernel", MAINLOOP)
        self.assertIn("filter_x_scalar_global_kernel", MAINLOOP)

    def test_tgv_drivers_forward_mode_only_to_gpu_run(self):
        for driver in (STATS_DRIVER, FIELD_DRIVER, MPI_STATS_DRIVER, MPI_FIELD_DRIVER):
            self.assertIn('FILTER_WORKSPACE="${FILTER_WORKSPACE:-full}"', driver)
            self.assertIn('ASTR_GPU_FILTER_WORKSPACE="$FILTER_WORKSPACE"', driver)

    def test_channel_uses_complete_rk_cpu_snapshot_for_field_comparison(self):
        self.assertIn('CPU_SNAPSHOT="outdat/rk_complete_snapshot.h5"', CHANNEL_DRIVER)
        self.assertIn('ASTR_VALIDATION_RK_SNAPSHOT="$CPU_SNAPSHOT"', CHANNEL_DRIVER)
        self.assertIn('--cpu "$OUT_DIR/cpu/$CPU_SNAPSHOT"', CHANNEL_DRIVER)

    def test_representative_boundary_drivers_forward_scalar_mode_to_gpu_only(self):
        for driver in (CHANNEL_DRIVER, LDC_DRIVER, CURVE_DRIVER):
            self.assertIn('FILTER_WORKSPACE="${FILTER_WORKSPACE:-full}"', driver)
            self.assertIn('ASTR_GPU_FILTER_WORKSPACE="$FILTER_WORKSPACE"', driver)

    def test_memcheck_driver_forwards_scalar_mode(self):
        self.assertIn('FILTER_WORKSPACE="${FILTER_WORKSPACE:-full}"', MEMCHECK_DRIVER)
        self.assertIn('ASTR_GPU_FILTER_WORKSPACE="$FILTER_WORKSPACE"', MEMCHECK_DRIVER)


if __name__ == "__main__":
    unittest.main()
