import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MECHANISM = ROOT / "chemMech/air5_kimjo12.json"
GENERATOR = ROOT / "scripts/generate_air5_mechanism.py"
GENERATED = ROOT / "src/chemistry_air5_data.F90"
CANONICAL_SHA256 = "d83ecc5b112f7962b9a234cbabad1cd8842e12136afdbb820c1dbca541976758"
LEGACY_ARCHIVE_SHA256 = "353edc87d665026a6650f3aa06862e1574df74248a2b2864d284fe0ea3692da1"


class Air5MechanismGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("air5_generator", GENERATOR)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import {GENERATOR}")
        cls.generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.generator)
        cls.data = json.loads(MECHANISM.read_text(encoding="ascii"))

    def generate(self, data):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            input_path = directory / "mechanism.json"
            output_path = directory / "chemistry_air5_data.F90"
            input_path.write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="ascii"
            )
            result = subprocess.run(
                [sys.executable, str(GENERATOR), "--input", str(input_path),
                 "--output", str(output_path)],
                cwd=ROOT, capture_output=True, text=True,
            )
            content = output_path.read_bytes() if output_path.exists() else b""
            return result, content

    def assert_rejected(self, data, fragment):
        result, content = self.generate(data)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(content, b"")
        self.assertIn(fragment, result.stderr)

    def test_authoritative_mechanism_is_strict_si_air5_kimjo12(self):
        data = self.data
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["mechanism_id"], "air5_kimjo12")
        self.assertEqual(data["species_order"], ["N2", "O2", "N", "O", "NO"])
        self.assertEqual(data["units"], {
            "amount": "mol",
            "energy": "J",
            "length": "m",
            "mass": "kg",
            "temperature": "K",
            "time": "s",
        })
        self.assertEqual(len(data["species"]), 5)
        self.assertEqual(len(data["reactions"]), 12)
        self.assertEqual(data["validity"]["temperature_k"], [300.0, 8000.0])
        self.assertEqual(data["validity"]["pressure_pa"], [1.0e3, 1.0e6])
        self.assertEqual(data["provenance"]["legacy_archive_sha256"],
                         LEGACY_ARCHIVE_SHA256)
        canonical = json.dumps(
            data, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), CANONICAL_SHA256)

    def test_cgs_rate_and_equilibrium_constants_are_converted_to_si(self):
        reactions = {reaction["id"]: reaction for reaction in self.data["reactions"]}
        expected_cgs_a = [
            1.216e20, 7.0e21, 3.591e20, 3.0e22,
            3.354e15, 1.117e25, 1.0e22, 3.0e21,
            1.45e15, 9.64e14, 6.4e17, 8.4e12,
        ]
        for index, expected in enumerate(expected_cgs_a, start=1):
            actual = reactions[f"R{index}"]["arrhenius"]["A"] * 1.0e6
            self.assertTrue(math.isclose(actual, expected, rel_tol=2.0e-16),
                            f"R{index}: {actual} != {expected}")
        self.assertEqual(
            reactions["R1"]["arrhenius"]["A_units"],
            "m^3 mol^-1 s^-1 K^-temperature_exponent",
        )
        self.assertEqual(reactions["R1"]["equilibrium"]["concentration_basis"],
                         "mol m^-3")
        expected_cgs_equilibrium = {
            "R1": [2.491, 0.7155, 2.091, -11.69, 0.005921],
            "R5": [1.567, 1.217, 1.909, -6.281, 0.005237],
            "R9": [2.093, -0.6229, 2.028, -7.872, 0.005586],
            "R11": [-0.1789, 1.728, -0.2172, -3.733, -0.0002285],
            "R12": [-0.1673, -1.39, -0.1656, -1.551, -0.0001102],
        }
        for reaction_id, expected in expected_cgs_equilibrium.items():
            reaction = reactions[reaction_id]
            actual = reaction["equilibrium"]["ln_kc_coefficients"].copy()
            actual[1] -= reaction["equilibrium"]["delta_nu"] * math.log(1.0e6)
            for coefficient, reference in zip(actual, expected):
                self.assertTrue(math.isclose(coefficient, reference, abs_tol=2.0e-15),
                                f"{reaction_id}: {actual} != {expected}")

    def test_generation_is_byte_stable_and_committed_output_is_current(self):
        first_result, first = self.generate(self.data)
        second_result, second = self.generate(self.data)
        self.assertEqual(first_result.returncode, 0, first_result.stderr)
        self.assertEqual(second_result.returncode, 0, second_result.stderr)
        self.assertEqual(first, second)
        self.assertEqual(first, GENERATED.read_bytes())

        check = subprocess.run(
            [sys.executable, str(GENERATOR), "--input", str(MECHANISM),
             "--output", str(GENERATED), "--check"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(check.returncode, 0, check.stderr)

    def test_unknown_species_is_rejected(self):
        data = copy.deepcopy(self.data)
        data["reactions"][0]["reactants"]["AR"] = 1
        self.assert_rejected(data, "unknown species")

    def test_duplicate_reaction_is_rejected(self):
        data = copy.deepcopy(self.data)
        duplicate = copy.deepcopy(data["reactions"][0])
        duplicate["id"] = "R12"
        data["reactions"][-1] = duplicate
        self.assert_rejected(data, "duplicate reaction")

    def test_duplicate_identifier_is_rejected(self):
        data = copy.deepcopy(self.data)
        data["reactions"][1]["id"] = data["reactions"][0]["id"]
        self.assert_rejected(data, "duplicate reaction id")

    def test_missing_required_field_is_rejected(self):
        mutations = (
            (lambda data: data["species"][0].pop("formation_energy_j_per_kg"),
             "formation_energy_j_per_kg"),
            (lambda data: data["provenance"].pop("legacy_archive_sha256"),
             "legacy_archive_sha256"),
            (lambda data: data["reactions"][0].pop("equation"), "equation"),
        )
        for mutate, fragment in mutations:
            with self.subTest(fragment=fragment):
                data = copy.deepcopy(self.data)
                mutate(data)
                self.assert_rejected(data, fragment)

    def test_fixed_identity_and_audit_metadata_are_rejected_when_invalid(self):
        mutations = (
            (lambda data: data["provenance"].__setitem__(
                "legacy_archive_sha256", "0" * 64), "legacy_archive_sha256"),
            (lambda data: data["reactions"][0].__setitem__("id", ""),
             "reaction id"),
            (lambda data: data["reactions"][0].__setitem__("equation", ""),
             "equation"),
        )
        for mutate, fragment in mutations:
            with self.subTest(fragment=fragment):
                data = copy.deepcopy(self.data)
                mutate(data)
                self.assert_rejected(data, fragment)

    def test_unknown_schema_field_is_rejected(self):
        data = copy.deepcopy(self.data)
        data["reactions"][0]["activation_temprature_k"] = 1.0
        self.assert_rejected(data, "unknown field")

    def test_duplicate_json_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            input_path = directory / "duplicate.json"
            output_path = directory / "output.F90"
            content = MECHANISM.read_text(encoding="ascii").replace(
                '"schema_version": 1,',
                '"schema_version": 1,\n  "schema_version": 1,',
                1,
            )
            input_path.write_text(content, encoding="ascii")
            result = subprocess.run(
                [sys.executable, str(GENERATOR), "--input", str(input_path),
                 "--output", str(output_path)],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate JSON key", result.stderr)
            self.assertFalse(output_path.exists())

    def test_non_si_units_are_rejected(self):
        data = copy.deepcopy(self.data)
        data["units"]["length"] = "cm"
        self.assert_rejected(data, "strict SI")

    def test_out_of_domain_or_nonfinite_values_are_rejected(self):
        mutations = (
            (lambda data: data["validity"].__setitem__("temperature_k", [0.0, 8000.0]),
             "temperature_k"),
            (lambda data: data["reactions"][0]["arrhenius"].__setitem__("A", -1.0),
             "Arrhenius A"),
            (lambda data: data["species"][0].__setitem__("molecular_weight_kg_per_mol",
                                                         float("inf")),
             "finite"),
        )
        for mutate, fragment in mutations:
            with self.subTest(fragment=fragment):
                data = copy.deepcopy(self.data)
                mutate(data)
                self.assert_rejected(data, fragment)


if __name__ == "__main__":
    unittest.main()
