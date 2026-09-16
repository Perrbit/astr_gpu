#!/usr/bin/env python3
"""Static contracts for the A800 TGV scaling diagnostics job."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT / "tests/gpu_validation/run_zhongke_a800_tgv_scaling_diagnostics.sbatch"


class ZhongkeA800TgvScalingDiagnosticsJobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = JOB.read_text(encoding="ascii")

    def test_requests_one_four_gpu_a800_node(self) -> None:
        self.assertIn("#SBATCH --partition=A800-N", self.text)
        self.assertIn("#SBATCH --nodes=1", self.text)
        self.assertIn("#SBATCH --ntasks=4", self.text)
        self.assertIn("#SBATCH --gres=gpu:4", self.text)
        self.assertIn("#SBATCH --time=12:00:00", self.text)

    def test_uses_the_frozen_scaling_executable(self) -> None:
        self.assertIn(
            "EXPECTED_SOURCE_COMMIT=c801eebc8f14a84d045eb60db92e9c022ccce7b7",
            self.text,
        )
        self.assertIn("astr_gpu_p4_c801eeb", self.text)
        self.assertIn("build_gpu_p4_c801eeb", self.text)
        self.assertIn("7ed0e326e907143d9cb5b7abb045ea8cea6d0256c0f63cdd02f91ceed0b06755", self.text)
        self.assertIn("nvhpc/24.11/Linux_x86_64/24.11/compilers/bin/nsys", self.text)
        self.assertIn("nvhpc/25.5/Linux_x86_64/25.5/compilers/bin/ncu", self.text)

    def test_profiles_the_complete_scaling_and_control_matrix(self) -> None:
        expected = [
            "strong_np1_111 1 1,1,1 512,512,512 pinned explicit",
            "strong_np2_211 2 2,1,1 512,512,512 pinned-pipeline explicit",
            "strong_np2_121 2 1,2,1 512,512,512 pinned-pipeline explicit",
            "strong_np2_112 2 1,1,2 512,512,512 pinned-pipeline explicit",
            "strong_np4_411 4 4,1,1 512,512,512 pinned-pipeline explicit",
            "strong_np4_141 4 1,4,1 512,512,512 pinned-pipeline explicit",
            "strong_np4_114 4 1,1,4 512,512,512 pinned-pipeline explicit",
            "strong_np4_221 4 2,2,1 512,512,512 pinned-pipeline explicit",
            "strong_np4_212 4 2,1,2 512,512,512 pinned-pipeline explicit",
            "strong_np4_122 4 1,2,2 512,512,512 pinned-pipeline explicit",
            "weak_np1_111 1 1,1,1 256,256,256 pinned explicit",
            "weak_np2_211 2 2,1,1 512,256,256 pinned-pipeline explicit",
            "weak_np4_221 4 2,2,1 512,512,256 pinned-pipeline explicit",
            "control_np2_overlap 2 2,1,1 512,512,512 pinned-overlap explicit",
            "control_np2_dependency 2 2,1,1 512,512,512 pinned-pipeline dependency",
            "control_np4_overlap 4 1,2,2 512,512,512 pinned-overlap explicit",
            "control_np4_dependency 4 1,2,2 512,512,512 pinned-pipeline dependency",
        ]
        for row in expected:
            self.assertIn(row, self.text)

    def test_collects_cuda_mpi_nvtx_timeline_and_stats(self) -> None:
        self.assertIn("--trace=cuda,mpi,nvtx", self.text)
        self.assertIn("--mpi-impl=openmpi", self.text)
        self.assertIn("trace_rank_%q{OMPI_COMM_WORLD_RANK}", self.text)
        self.assertIn('mpirun -np "$np" "$NSYS_EXE" profile', self.text)
        self.assertIn("run_profiler_preflight", self.text)
        self.assertIn("profiler preflight report lacks CUDA or MPI trace data", self.text)
        self.assertIn("--sample=none", self.text)
        self.assertIn("cuda_gpu_kern_sum", self.text)
        self.assertIn("cuda_api_sum", self.text)
        self.assertIn("cuda_gpu_mem_time_sum", self.text)
        self.assertIn("mpi_event_sum", self.text)
        self.assertIn("mpi_msg_size_sum", self.text)
        self.assertIn("summarize_mpi_kernel_overlap.py", self.text)

    def test_profiles_twelve_compute_kernels_on_one_gpu(self) -> None:
        kernels = [
            "solver_gpu_q_to_primitive_kernel_",
            "solver_gpu_filter_x_periodic_interior_global_kernel_",
            "solver_gpu_filter_y_pong_to_q_halo_global_kernel_",
            "solver_gpu_filter_z_halo_global_kernel_",
            "gradcal_gpu_gradcal_dvel_kernel_",
            "solver_gpu_convective_rhs_x_global_kernel_",
            "solver_gpu_convective_rhs_y_global_kernel_",
            "solver_gpu_convective_rhs_z_global_kernel_",
            "solver_gpu_diffusion_flux_global_kernel_",
            "solver_gpu_diffusion_rhs_x_stored_global_kernel_",
            "solver_gpu_diffusion_rhs_y_stored_global_kernel_",
            "solver_gpu_diffusion_rhs_z_stored_global_kernel_",
        ]
        for kernel in kernels:
            self.assertIn(kernel, self.text)
        self.assertIn("--set full", self.text)
        self.assertIn("--launch-count 1", self.text)
        self.assertIn('mpirun -np 1 "$NCU_EXE"', self.text)

    def test_forbids_field_io_and_isolates_case_failures(self) -> None:
        self.assertIn("ASTR_GPU_BENCHMARK_NO_FIELD_IO=1", self.text)
        self.assertIn("forbidden field HDF5", self.text)
        self.assertIn("case_status.tsv", self.text)
        self.assertIn("run_case", self.text)
        self.assertNotIn("set -euo pipefail", self.text)


if __name__ == "__main__":
    unittest.main()
