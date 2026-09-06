#!/usr/bin/env python3
"""Source contract for vectorized physical flux work inside selective Roe."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "src_gpu/solver_gpu.cuf").read_text(encoding="utf-8")


def device_subroutine(name: str) -> str:
    match = re.search(
        rf"attributes\(device\) subroutine {name}\b(.*?)end subroutine {name}",
        SOURCE,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"missing device subroutine {name}")
    return match.group(1)


class P2PhysicalFluxVectorContractTests(unittest.TestCase):
    def test_vector_helper_preserves_physical_sound_speed_definition(self) -> None:
        body = device_subroutine("steger_warming_split_physical_vector_at")
        self.assertIn("use commvar_gpu, only: mach_d", body)
        self.assertIn("css = sqrt(tmp_d(i,j,k))/mach_d", body)
        self.assertIn("f(1) = jro*var4", body)
        self.assertIn("f(5) = jloc*(var1*q5 + rho*", body)

    def test_nonshock_path_computes_one_vector_per_stencil_side(self) -> None:
        body = device_subroutine("characteristic_reconstruction_interface_flux")
        nonshock = body[
            body.index("if(.not.shock_interface_active") :
            body.index("call roe_characteristic_state")
        ]
        self.assertEqual(
            nonshock.count("call steger_warming_split_physical_vector_at"), 2
        )
        self.assertIn("do n=1,7", nonshock)
        self.assertIn("do m=1,5", nonshock)
        self.assertNotIn("explicit_reconstruction_interface_flux(", nonshock)
        self.assertNotIn("explicit_reconstruction_interface_flux_x_physical(", nonshock)
        self.assertNotIn("explicit_reconstruction_interface_flux_y_physical(", nonshock)

    def test_nonshock_physical_sampling_preserves_internal_rank_halos(self) -> None:
        body = device_subroutine("characteristic_reconstruction_interface_flux")
        nonshock = body[
            body.index("if(.not.shock_interface_active") :
            body.index("call roe_characteristic_state")
        ]
        self.assertGreaterEqual(nonshock.count("case(3)"), 2)
        self.assertIn("ibeg = -hm\n          iend = im + hm", nonshock)
        self.assertIn("jbeg = -hm\n          jend = jm + hm", nonshock)

    def test_shock_physical_sampling_preserves_internal_rank_halos(self) -> None:
        body = device_subroutine("characteristic_reconstruction_interface_flux")
        shock = body[
            body.index("call roe_characteristic_state") :
            body.index("do n=1,7", body.index("call roe_characteristic_state"))
        ]
        self.assertEqual(shock.count("case(3)"), 2)
        self.assertIn("ibeg = -hm\n        iend = im + hm", shock)
        self.assertIn("jbeg = -hm\n        jend = jm + hm", shock)


if __name__ == "__main__":
    unittest.main()
