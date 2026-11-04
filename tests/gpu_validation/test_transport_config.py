import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class TransportConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        build = Path(os.environ.get('ASTR_TEST_BUILD_DIR', ROOT/'build_native_fp3_fp64_cpu'))
        subprocess.run(['cmake', '--build', str(build), '--target', 'transport_config_probe', '-j4'],
                       check=True, capture_output=True, text=True)
        cls.exe = build/'bin/transport_config_probe'
        cache = (build/'CMakeCache.txt').read_text().splitlines()
        cls.mpiexec = next(line.split('=',1)[1] for line in cache if line.startswith('MPIEXEC_EXECUTABLE:'))

    def run_config(self, values, retired=None):
        env = dict(os.environ)
        for key in ("ASTR_PERFECT_GAS_PRANDTL", "ASTR_SUTHERLAND_TEMPERATURE_K"):
            env.pop(key, None)
        env.update(retired or {})
        env.update(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp',
                   OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_opal_cuda_support='0')
        path = Path(self.directory.name)/'input.nml'
        path.write_text('&transport_options\n'+''.join(f'{k}={v}\n' for k,v in values.items())+'/\n')
        return subprocess.run([self.mpiexec,'-n','1',str(self.exe),str(path)],
                              env=env, capture_output=True, text=True)

    def payload(self,result):
        return next(line.split()[1:] for line in result.stdout.splitlines()
                    if line.split() and line.split()[0]=='TRANSPORT_CONFIG')

    def test_defaults(self):
        result = self.run_config({})
        self.assertEqual(result.returncode, 0, result.stderr)
        pr, suth, changed = self.payload(result)
        self.assertEqual((float(pr), float(suth), changed), (0.72, 110.3, "F"))

    def test_reference_overrides(self):
        result = self.run_config({"perfect_gas_prandtl": "0.71d0",
                                 "sutherland_temperature_k": "110.4"})
        self.assertEqual(result.returncode, 0, result.stderr)
        pr, suth, changed = self.payload(result)
        self.assertEqual((float(pr), float(suth), changed), (0.71, 110.4, "T"))

    def test_invalid_values_fail(self):
        for key in ("perfect_gas_prandtl", "sutherland_temperature_k"):
            for value in ("0", "-1", "NaN", "Inf", "'bad'", "1" * 300, "1,2"):
                with self.subTest(key=key, value=value):
                    result = self.run_config({key: value})
                    self.assertIn('TRANSPORT_CONFIG_BEGIN', result.stdout)
                    self.assertNotEqual(result.returncode, 0)

    def test_retired_environment_cannot_override_input(self):
        result = self.run_config({}, {'ASTR_PERFECT_GAS_PRANDTL': 'bad',
                                     'ASTR_SUTHERLAND_TEMPERATURE_K': '-1'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.payload(result), self.payload(self.run_config({})))
        self.assertEqual(result.stdout.count('ASTR_CASE ignored retired environment'), 2)


if __name__ == "__main__":
    unittest.main()
