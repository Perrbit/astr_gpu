"""Exercise the native source-range scanner without advancing a solver."""
import ctypes
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def scanner(tmp_path_factory):
    compiler = shutil.which("cc")
    if not compiler:
        pytest.skip("C compiler unavailable")
    output = tmp_path_factory.mktemp("inflow_scanner") / "scanner.so"
    source = Path(__file__).resolve().parents[2] / "src/checkpoint_filesystem.c"
    subprocess.run([compiler, "-std=c99", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                    str(source), "-o", str(output)], check=True, timeout=30)
    function = ctypes.CDLL(str(output)).astr_checkpoint_inflow_count
    function.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
    function.restype = ctypes.c_int

    def discover(path):
        count = ctypes.c_int(-1)
        status = function(str(path).encode(), ctypes.byref(count))
        return status, count.value

    return discover


def source_directory(tmp_path, count=4):
    root = tmp_path / "inflow"
    root.mkdir()
    for number in range(count):
        (root / f"islice{number:05d}.h5").write_bytes(b"source scanner fixture")
    return root


@pytest.mark.parametrize("count", [4, 70])
def test_contiguous_source_range(scanner, tmp_path, count):
    root = source_directory(tmp_path, count)
    (root / "README.txt").write_text("not a source frame\n")
    assert scanner(root) == (0, count)


@pytest.mark.parametrize("fault", ["hole", "short", "symlink", "hardlink", "directory",
                                   "bad_name", "beyond_limit", "missing"])
def test_source_range_rejection(scanner, tmp_path, fault):
    root = source_directory(tmp_path)
    leaf = root / "islice00001.h5"
    if fault == "hole":
        leaf.unlink()
    elif fault == "short":
        (root / "islice00003.h5").unlink()
    elif fault == "symlink":
        leaf.rename(tmp_path / "external.h5")
        leaf.symlink_to(tmp_path / "external.h5")
    elif fault == "hardlink":
        (tmp_path / "linked.h5").hardlink_to(leaf)
    elif fault == "directory":
        leaf.unlink()
        leaf.mkdir()
    elif fault == "bad_name":
        (root / "isliceABC01.h5").write_bytes(b"bad index")
    elif fault == "beyond_limit":
        (root / "islice100000.h5").write_bytes(b"index does not fit legacy naming")
    else:
        root = tmp_path / "missing"
    status, count = scanner(root)
    assert status != 0 and count == 0


def test_source_directory_symlink_rejection(scanner, tmp_path):
    root = source_directory(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    assert scanner(alias) == (1, 0)
