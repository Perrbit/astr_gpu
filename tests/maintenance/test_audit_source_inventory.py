from __future__ import annotations

import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "maintenance" / "audit_source_inventory.py"


def load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_source_inventory", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AuditScriptBootstrapTest(unittest.TestCase):
    def test_audit_script_exists(self):
        self.assertTrue(SCRIPT.is_file(), f"missing implementation: {SCRIPT}")


@unittest.skipUnless(SCRIPT.is_file(), "audit script not implemented yet")
class FortranScannerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = load_audit_module()

    def make_repo(self, files: dict[str, str]) -> Path:
        repo = Path(tempfile.mkdtemp(prefix="astr-inventory-test-"))
        self.addCleanup(shutil.rmtree, repo)
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "config", "user.name", "Test"], check=True
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "config",
                "user.email",
                "test@example.invalid",
            ],
            check=True,
        )
        for relative, content in files.items():
            target = repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "--all"], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True
        )
        return repo

    def facts(
        self,
        path: str,
        *,
        programs: tuple[str, ...] = (),
        modules: tuple[str, ...] = (),
        subroutines: tuple[str, ...] = (),
        functions: tuple[str, ...] = (),
        includes: tuple[str, ...] = (),
    ):
        return self.audit.FortranFacts(
            path=path,
            programs=programs,
            modules=modules,
            subroutines=subroutines,
            functions=functions,
            uses=(),
            calls=(),
            includes=includes,
            sha256="0" * 64,
        )

    def test_logical_fortran_lines_fold_continuations_and_strip_comments(self):
        text = """module demo
          use alpha, only: first, &
               & second ! trailing comment
          call advance(state, &
               & dt)
        end module demo
        """

        self.assertEqual(
            self.audit.logical_fortran_lines(text),
            [
                "module demo",
                "use alpha, only: first, second",
                "call advance(state, dt)",
                "end module demo",
            ],
        )

    def test_logical_fortran_lines_preserve_quoted_comment_and_semicolon(self):
        text = "print *, 'not ! a comment; still text'; call finish ! comment\n"

        self.assertEqual(
            self.audit.logical_fortran_lines(text),
            ["print *, 'not ! a comment; still text'", "call finish"],
        )

    def test_parse_fortran_records_declarations_and_edges(self):
        text = """module demo
          use, intrinsic :: iso_fortran_env, only: real64
          use alpha
          include 'body.inc'
        contains
          attributes(global) subroutine run
            call step
          end subroutine run
          real(real64) function value()
            value = 1.0_real64
          end function value
        end module demo
        """

        facts = self.audit.parse_fortran(Path("src/demo.F90"), text)

        self.assertEqual(facts.path, "src/demo.F90")
        self.assertEqual(facts.modules, ("demo",))
        self.assertEqual(facts.subroutines, ("run",))
        self.assertEqual(facts.functions, ("value",))
        self.assertEqual(facts.uses, ("alpha", "iso_fortran_env"))
        self.assertEqual(facts.calls, ("step",))
        self.assertEqual(facts.includes, ("body.inc",))
        self.assertEqual(len(facts.sha256), 64)

    def test_parse_fortran_records_program_and_preprocessor_include(self):
        text = """program sample
        #include "config.inc"
        end program sample
        """

        facts = self.audit.parse_fortran(Path("src/sample.F90"), text)

        self.assertEqual(facts.programs, ("sample",))
        self.assertEqual(facts.includes, ("config.inc",))

    def test_parse_fortran_excludes_end_module_and_module_procedure(self):
        text = """module interfaces
          interface generic_name
            module procedure concrete_name
          end interface generic_name
        end module interfaces
        """

        facts = self.audit.parse_fortran(Path("src/interfaces.F90"), text)

        self.assertEqual(facts.modules, ("interfaces",))
        self.assertEqual(facts.subroutines, ())
        self.assertEqual(facts.functions, ())

    def test_tracked_core_files_exclude_untracked_and_non_source_files(self):
        repo = self.make_repo(
            {
                "src/a.F90": "module a\nend module a\n",
                "src/body.inc": "call body_step\n",
                "src/CMakeLists.txt": "set(ASTR_SOURCES a.F90)\n",
                "src_gpu/k.cuf": "module k\nend module k\n",
            }
        )
        (repo / "src" / "untracked.F90").write_text(
            "module hidden\nend module hidden\n", encoding="utf-8"
        )

        self.assertEqual(
            [path.as_posix() for path in self.audit.tracked_core_files(repo)],
            ["src/a.F90", "src/body.inc", "src_gpu/k.cuf"],
        )

    def test_parse_cmake_membership_resolves_cpu_and_gpu_lists(self):
        cmake = """set(ASTR_SOURCES a.F90 body.inc)
        set(ASTR_GPU_SOURCES
          ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/k.cuf)
        add_executable(astr ${ASTR_SOURCES} ${ASTR_GPU_SOURCES})
        """

        membership = self.audit.parse_cmake_membership(cmake)

        self.assertEqual(membership["src/a.F90"], "astr CPU")
        self.assertEqual(membership["src_gpu/k.cuf"], "astr CUDA")

    def test_validate_source_anchors_rejects_missing_symbol(self):
        facts = {"src/a.F90": self.facts("src/a.F90", modules=("a",))}

        errors = self.audit.validate_source_anchors(
            "Evidence `src/a.F90::missing`", facts
        )

        self.assertEqual(
            errors, ["source anchor symbol not found: src/a.F90::missing"]
        )

    def test_source_baseline_ignores_documentation_only_commit(self):
        repo = self.make_repo(
            {
                "CMakeLists.txt": "add_subdirectory(src)\n",
                "src/a.F90": "module a\nend module a\n",
                "src/CMakeLists.txt": "set(ASTR_SOURCES a.F90)\n",
            }
        )
        baseline = self.audit.source_baseline(repo)
        doc = repo / "documents" / "note.md"
        doc.parent.mkdir(parents=True)
        doc.write_text("documentation only\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", str(doc)], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-qm", "docs only"], check=True
        )

        self.assertEqual(self.audit.source_baseline(repo), baseline)

    def test_two_inventory_renders_are_byte_identical(self):
        snapshot = self.audit.InventorySnapshot(
            baseline="0123456789abcdef",
            facts=(self.facts("src/a.F90", modules=("a",)),),
            memberships=(("src/a.F90", "astr CPU"),),
            include_owners=(),
            fingerprint="f" * 64,
        )

        first = self.audit.render_inventory(snapshot)
        second = self.audit.render_inventory(snapshot)

        self.assertEqual(first.encode("utf-8"), second.encode("utf-8"))

    def test_build_snapshot_reports_unlisted_source(self):
        repo = self.make_repo(
            {
                "src/a.F90": "module a\nend module a\n",
                "src/b.F90": "module b\nend module b\n",
                "src/CMakeLists.txt": "set(ASTR_SOURCES a.F90)\n",
            }
        )

        _, errors = self.audit.build_snapshot(repo)

        self.assertIn("production source missing from astr target: src/b.F90", errors)

    def test_main_write_and_check_detect_stale_inventory(self):
        repo = self.make_repo(
            {
                "src/a.F90": "module a\nend module a\n",
                "src/CMakeLists.txt": "set(ASTR_SOURCES a.F90)\n",
            }
        )

        self.assertEqual(
            self.audit.main(["--repo", str(repo), "--write"]), 0
        )
        inventory = repo / self.audit.INVENTORY_PATH
        inventory.write_text(
            inventory.read_text(encoding="utf-8") + "stale\n", encoding="utf-8"
        )

        self.assertEqual(
            self.audit.main(["--repo", str(repo), "--check"]), 1
        )


if __name__ == "__main__":
    unittest.main()
