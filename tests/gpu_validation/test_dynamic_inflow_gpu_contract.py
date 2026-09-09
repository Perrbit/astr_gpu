#!/usr/bin/env python3
"""Static integration contracts for the GPU-resident time-series inlet."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


class DynamicInflowGpuContractTests(unittest.TestCase):
    def source(self, relative: str) -> str:
        path = ROOT / relative
        self.assertTrue(path.is_file(), f"missing production source: {relative}")
        return path.read_text(encoding="utf-8")

    def test_four_slice_device_ring_and_single_slot_upload(self) -> None:
        source = self.source("src_gpu/inflow_timeseries_gpu.cuf")
        self.assertRegex(source, r"inflow_slices_d\s*\(:,:,:,:\)")
        self.assertIn("0:3", source)
        self.assertRegex(source, r"inflow_slices_d\s*\(:,:,:,slot\)\s*=\s*inflow_slice_h")
        self.assertNotRegex(source, r"inflow_slices_d\s*=\s*inflow_slices_h")

    def test_window_advance_handles_more_than_one_crossing(self) -> None:
        source = self.source("src_gpu/inflow_timeseries_gpu.cuf")
        self.assertRegex(
            source,
            re.compile(r"do\s+while\s*\(.*time.*>.*inflow_times_h", re.DOTALL),
        )
        self.assertIn("ninflowslice=ninflowslice+1", source)

    def test_gpu_runtime_updates_window_once_before_rk_preparation(self) -> None:
        source = self.source("src_gpu/gpu_runtime.cuf")
        body = re.search(
            r"subroutine gpu_prepare_rkfirst_stats\b(.*?)end subroutine gpu_prepare_rkfirst_stats",
            source,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(body)
        text = body.group(1)
        self.assertIn("call prepare_dynamic_inflow_gpu", text)
        self.assertIn("call prepare_rkfirst_stats_gpu", text)
        self.assertLess(
            text.index("call prepare_dynamic_inflow_gpu"),
            text.index("call prepare_rkfirst_stats_gpu"),
        )

    def test_static_profile_and_dynamic_modes_have_separate_kernels(self) -> None:
        source = self.source("src_gpu/boundary_gpu.cuf")
        self.assertIn("s1_bl_profile_inflow_x_kernel<<<", source)
        self.assertIn("s1_bl_dynamic_inflow_x_kernel<<<", source)
        self.assertRegex(source, r"select\s+case\s*\(trim\(turbinf\)\)")

    def test_dynamic_mode_is_enabled_for_the_validated_mpi_mp7_path(self) -> None:
        source = self.source("src_gpu/case_capability_gpu.cuf")
        body = re.search(
            r"logical function gpu_s1_flatplate_s1a1_supported\(\)(.*?)"
            r"end function gpu_s1_flatplate_s1a1_supported",
            source,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(body)
        text = body.group(1)
        self.assertIn("trim(conschm) == '543e'", text)
        self.assertRegex(
            text,
            r"trim\(turbinf\)\s*==\s*'prof'.*?trim\(turbinf\)\s*==\s*'intp'",
        )
        self.assertIn("mpisize == 2", text)
        self.assertRegex(
            text,
            re.compile(
                r"mpisize\s*==\s*4.*?isize\s*==\s*4.*?"
                r"jsize\s*==\s*1.*?ksize\s*==\s*1",
                re.DOTALL,
            ),
        )

    def test_dynamic_checkpoint_writes_complete_rk_state_without_host_boundary_replay(self) -> None:
        source = self.source("src/mainloop.F90")
        self.assertIn("dynamic_inflow_output", source)
        self.assertRegex(
            source,
            re.compile(
                r"dynamic_inflow_output\s*=.*?bctype\(1\)\s*==\s*11.*?"
                r"trim\(turbinf\)\s*==\s*'intp'.*?"
                r"if\s*\(flowtype\(1:2\)/='0d'\s*\.and\.\s*\.not\.conservative_case\s*\.and\..*?"
                r"\.not\.dynamic_inflow_output\)\s*then.*?call boucon",
                re.DOTALL,
            ),
        )
        self.assertNotIn("dynamic_q_xmin", source)

    def test_cpu_dynamic_oracle_writes_the_same_unprepared_rk_phase(self) -> None:
        source = self.source("src/readwrite.F90")
        body = re.search(
            r"subroutine write_validation_rk_snapshot\b(.*?)"
            r"end subroutine write_validation_rk_snapshot",
            source,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(body)
        text = body.group(1)
        self.assertRegex(
            text,
            re.compile(
                r"dynamic_inflow_snapshot\s*=.*?bctype\(1\)\s*==\s*11.*?"
                r"trim\(turbinf\)\s*==\s*'intp'.*?"
                r"if\s*\(\.not\.conservative_boundary%enabled\s*\.and\.\s*"
                r"\.not\.dynamic_inflow_snapshot\)\s*then.*?call boucon",
                re.DOTALL,
            ),
        )


if __name__ == "__main__":
    unittest.main()
