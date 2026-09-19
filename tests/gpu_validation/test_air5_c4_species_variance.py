from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.gpu_validation.check_air5_c4_species_variance import (
    snapshot_species_variance,
)


def write_snapshot(path: Path, amplitude: float) -> None:
    im = jm = km = 8
    hm = 3
    numq = 11
    shape = (im + 2 * hm + 1, jm + 2 * hm + 1, km + 2 * hm + 1, numq)
    q = np.zeros(shape, dtype=np.float64, order="F")
    q[..., 0] = 1.0
    for i in range(-hm, im + hm + 1):
        fraction = 0.75 + amplitude * np.sin(2.0 * np.pi * i / im)
        q[i + hm, :, :, 5] = fraction
        q[i + hm, :, :, 6] = 1.0 - fraction
    with path.open("wb") as stream:
        np.asarray([im, jm, km, hm, numq], dtype=np.int32).tofile(stream)
        q.ravel(order="F").tofile(stream)


class Air5SpeciesVarianceTests(unittest.TestCase):
    def test_density_weighted_variance_tracks_wave_damping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial_path = root / "initial.bin"
            final_path = root / "final.bin"
            write_snapshot(initial_path, 0.1)
            write_snapshot(final_path, 0.08)
            initial = snapshot_species_variance([initial_path])
            final = snapshot_species_variance([final_path])
            self.assertAlmostEqual(initial.mean, final.mean)
            self.assertLess(final.variance, initial.variance)
            self.assertAlmostEqual(final.variance / initial.variance, 0.64)

    def test_empty_snapshot_set_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no q snapshots"):
            snapshot_species_variance([])


class Air5DiffusionSignContractTests(unittest.TestCase):
    def test_cpu_species_divergence_is_subtracted(self) -> None:
        source = Path("src/chemistry_solver.F90").read_text(encoding="utf-8")
        self.assertEqual(source.count("call add_air5_diffusive_divergence"), 3)
        self.assertIn("air5_idx_species_last", source)
        self.assertIn(
            "qrhs(:,air5_idx_species_first:air5_idx_species_last)= &\n"
            "      qrhs(:,air5_idx_species_first:air5_idx_species_last)- &",
            source,
        )

    def test_gpu_species_divergence_is_subtracted(self) -> None:
        source = Path("src_gpu/chemistry_solver_gpu.cuf").read_text(encoding="utf-8")
        self.assertEqual(
            source.count("air5_diffusion_rhs_sign(component)*air5_deriv6"), 3
        )
        self.assertIn(
            "if(component>=air5_idx_species_first .and. &\n"
            "       component<=air5_idx_species_last) air5_diffusion_rhs_sign=-1.0_real64",
            source,
        )


if __name__ == "__main__":
    unittest.main()
