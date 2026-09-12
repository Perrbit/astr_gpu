"""Source contract supplement to the MPI endpoint and CFD runtime tests."""

from pathlib import Path
import re
import unittest


class HaloEndpointUploadContract(unittest.TestCase):
    def test_every_receive_upload_checks_its_neighbor(self):
        source = (
            Path(__file__).resolve().parents[2] / "src_gpu/halo_exchange_gpu.cuf"
        ).read_text()
        neighbors = {
            "right": "mpiright", "left": "mpileft",
            "up": "mpiup", "down": "mpidown",
            "front": "mpifront", "back": "mpiback",
        }
        uploads = []
        for line in source.splitlines():
            match = re.search(
                r"\b(\w+_recv_(right|left|up|down|front|back)_d)"
                r"(?:\([^)]*\))?\s*=\s*\w+_recv_\w+_h", line
            )
            if match:
                uploads.append((line, match))
        self.assertEqual(
            len(uploads),
            28,
            "Audit nine full-field, two scalar-filter, and three FP32 field receive-buffer pairs",
        )
        for line, match in uploads:
            with self.subTest(buffer=match[1]):
                guard = re.sub(r"\s+", "", line[:match.start()]).lower()
                self.assertEqual(
                    guard, f"if({neighbors[match[2]]}/=mpi_proc_null)",
                    "An absent neighbor must not upload its receive buffer",
                )


if __name__ == "__main__":
    unittest.main()
