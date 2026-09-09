#!/usr/bin/env python3
"""Static contracts for the Fang 2020 modal wall-forcing path."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CPU_BC = (ROOT / "src/bc.F90").read_text(encoding="utf-8")
GPU_BC = (ROOT / "src_gpu/boundary_gpu.cuf").read_text(encoding="utf-8")


class FangWallForcingContractTests(unittest.TestCase):
    def test_random_forcing_canonicalizes_the_periodic_k_endpoint(self) -> None:
        self.assertIn("if(ka>0) gk=modulo(gk,int(ka,8))", CPU_BC)
        self.assertIn("wall_ka_d=ka", GPU_BC)
        self.assertIn(
            "if(wall_ka_d>0) gk=gk-(gk/int(wall_ka_d,8))*int(wall_ka_d,8)",
            GPU_BC,
        )

    def test_cpu_dispatches_positive_temporal_mode_to_modal_forcing(self) -> None:
        self.assertRegex(
            CPU_BC,
            re.compile(
                r"function\s+wall_blowing_velocity\(\).*?"
                r"if\s*\(wall_blowing_nmod_t>0\)\s*then.*?"
                r"wallbs\(.*?wall_blowing_xa,wall_blowing_xb,.*?"
                r"wall_blowing_nmod_t,wall_blowing_nmod_z\).*?"
                r"else.*?wallbs_rand\(",
                re.DOTALL,
            ),
        )

    def test_modal_phases_are_read_broadcast_and_uploaded(self) -> None:
        self.assertIn("wall_blowing_phase(15)", CPU_BC)
        self.assertIn("datin/wallbs_phase.dat", CPU_BC)
        self.assertIn("call bcast(wall_blowing_phase)", CPU_BC)
        self.assertIn("wall_blowing_phase_d=wall_blowing_phase", GPU_BC)

    def test_gpu_modal_formula_uses_time_and_all_published_modes(self) -> None:
        self.assertIn("wall_blowing_nmod_t_d", GPU_BC)
        self.assertIn("wall_blowing_beta_d", GPU_BC)
        self.assertRegex(GPU_BC, r"do\s+l=1,wall_blowing_nmod_z_d")
        self.assertRegex(GPU_BC, r"do\s+m=1,wall_blowing_nmod_t_d")
        self.assertIn("wall_blowing_phase_d(l)", GPU_BC)
        self.assertIn("wall_blowing_phase_d(m+10)", GPU_BC)
        self.assertRegex(
            GPU_BC,
            r"wall_blowing_velocity_gpu\(i,k,xcoord,zcoord,forcing_time\)",
        )

    def test_gpu_wall_kernels_receive_host_physical_time(self) -> None:
        self.assertRegex(
            GPU_BC,
            re.compile(
                r"subroutine\s+apply_s1_flatplate_boundary_conditions_gpu.*?"
                r"use\s+commvar,\s*only:.*?time.*?"
                r"wall_isothermal_noslip_y_kernel<<<.*?>>>\(.*?time",
                re.DOTALL,
            ),
        )


if __name__ == "__main__":
    unittest.main()
