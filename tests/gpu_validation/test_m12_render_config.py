"""Exercise the real namelist parser for the approved boundary-layer preset."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
import tempfile

from run_m12_insitu_acceptance import configuration
from run_m12_insitu_acceptance import visible_geometry_pixels
import numpy as np


class RenderImageTests(unittest.TestCase):
    def test_memcheck_requires_explicit_sanitizer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('run_m12_insitu_acceptance.py')),
                '--source', directory, '--output', str(path/'never_created'),
                '--executable', '/bin/true', '--mpiexec', '/bin/true', '--library', directory,
                '--memcheck'], capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('--memcheck requires --sanitizer', result.stderr)
            self.assertFalse((path/'never_created').exists())

    def test_legend_is_not_visible_geometry(self):
        pixels = np.full((384, 1536, 3), 255, dtype=np.uint8)
        pixels[:200, 1429:1450] = 0
        self.assertEqual(visible_geometry_pixels(pixels), 0)
        pixels[200, 500] = 0
        self.assertEqual(visible_geometry_pixels(pixels), 1)


@unittest.skipUnless(os.environ.get('ASTR_INSITU_CONFIG_PROBE'), 'requires built Fortran parser probe')
class M12RenderConfigTests(unittest.TestCase):
    def parse(self, content):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'input.insitu'
            path.write_text(content)
            return subprocess.run([os.environ['ASTR_INSITU_CONFIG_PROBE'], str(path)],
                                  capture_output=True, text=True, timeout=10)

    def test_approved_preset_is_accepted(self):
        result = self.parse(configuration(Path('/lib/catalyst')))
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_unapproved_changes_are_rejected(self):
        base = configuration(Path('/lib/catalyst'))
        changes = (('statistics=f', 'statistics=t'),
                   ("rendering_pipeline='standard-device'", "rendering_pipeline='compatible'"),
                   ("postprocess_transport='pinned'", "postprocess_transport='device-aware'"),
                   ("streamline_seeds='bl-layered64'", "streamline_seeds='line16'"),
                   ('step_interval=5', 'step_interval=1'),
                   ('initial_frame=f', 'initial_frame=t'),
                   ('slice_origin=0.,0.,45.', 'slice_origin=0.,0.,40.'))
        for old, new in changes:
            with self.subTest(change=new):
                result = self.parse(base.replace(old, new))
                self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()
