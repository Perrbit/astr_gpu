"""Bounded tests of the production metadata replacement helper, without solver runs."""
import ctypes
from pathlib import Path
import subprocess

import pytest


@pytest.fixture(scope="module")
def filesystem(tmp_path_factory):
    library = tmp_path_factory.mktemp("series_filesystem") / "helper.so"
    source = Path(__file__).resolve().parents[2] / "src/checkpoint_filesystem.c"
    subprocess.run(["cc", "-std=c99", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                    str(source), "-o", str(library)], check=True, timeout=30)
    return ctypes.CDLL(str(library))


@pytest.fixture(scope="module")
def replace(filesystem):
    function = filesystem.astr_checkpoint_replace_plain_file
    function.argtypes = [ctypes.c_char_p] * 3
    function.restype = ctypes.c_int
    return lambda root, temporary="new", target="old": function(str(root).encode(), temporary.encode(), target.encode())


@pytest.mark.parametrize("defect", ["none", "capacity", "symlink", "ancestor_alias", "hardlink", "missing", "same"])
def test_explicit_parent_relative_path(filesystem, tmp_path, defect):
    current = tmp_path / "new/fields/segment00000001"
    parent = tmp_path / "old/fields/segment00000000"
    current.mkdir(parents=True)
    parent.mkdir(parents=True)
    for name in ("SEGMENT", "input.txt"):
        (parent / name).write_bytes(b"provenance")
    capacity = 1200
    if defect == "capacity":
        capacity = 2
    elif defect == "symlink":
        alias = parent.with_name("alias")
        alias.symlink_to(parent, target_is_directory=True)
        parent = alias
    elif defect == "ancestor_alias":
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path / "old", target_is_directory=True)
        parent = alias / "fields/segment00000000"
    elif defect == "hardlink":
        (parent / "extra").hardlink_to(parent / "SEGMENT")
    elif defect == "missing":
        (parent / "input.txt").unlink()
    elif defect == "same":
        current = parent
    function = filesystem.astr_output_archive_parent
    function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    result = ctypes.create_string_buffer(1200)
    status = function(str(current).encode(), str(parent).encode(), result, capacity)
    assert (status == 0) == (defect == "none")
    if status == 0:
        relative = Path(result.value.decode())
        assert not relative.is_absolute() and (current / relative).resolve() == parent


@pytest.mark.parametrize("latest", ["absent", "regular", "symlink", "hardlink", "directory"])
@pytest.mark.parametrize("stale", [False, True])
def test_publication_preflight(filesystem, tmp_path, latest, stale):
    victim = tmp_path / "victim"
    victim.write_bytes(b"external")
    pointer = tmp_path / "LATEST"
    if latest == "regular":
        pointer.write_text("batch1\n")
    elif latest == "symlink":
        pointer.symlink_to(victim)
    elif latest == "hardlink":
        pointer.hardlink_to(victim)
    elif latest == "directory":
        pointer.mkdir()
    if stale:
        (tmp_path / ".LATEST.tmp").write_bytes(b"unfinished")
    function = filesystem.astr_checkpoint_publication_root
    function.argtypes = [ctypes.c_char_p]
    function.restype = ctypes.c_int
    assert (function(str(tmp_path).encode()) == 0) == (not stale and latest in ("absent", "regular"))
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    assert function(str(alias).encode()) != 0
    assert victim.read_bytes() == b"external"


@pytest.mark.parametrize("existing", [False, True])
def test_replace_regular(replace, tmp_path, existing):
    (tmp_path / "new").write_bytes(b"complete candidate")
    if existing:
        (tmp_path / "old").write_bytes(b"previous index")
    assert replace(tmp_path) == 0
    assert (tmp_path / "old").read_bytes() == b"complete candidate"
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("member", ["new", "old"])
@pytest.mark.parametrize("fault", ["symlink", "hardlink", "directory"])
def test_reject_unsafe_member(replace, tmp_path, member, fault):
    (tmp_path / "new").write_bytes(b"complete candidate")
    (tmp_path / "old").write_bytes(b"previous index")
    victim = tmp_path / "victim"
    victim.write_bytes(b"external content")
    leaf = tmp_path / member
    leaf.unlink()
    if fault == "symlink":
        leaf.symlink_to(victim)
    elif fault == "hardlink":
        leaf.hardlink_to(victim)
    else:
        leaf.mkdir()
    other = tmp_path / ("old" if member == "new" else "new")
    before = other.read_bytes()
    assert replace(tmp_path) != 0
    assert other.read_bytes() == before and victim.read_bytes() == b"external content"
    assert leaf.exists()


@pytest.mark.parametrize("temporary,target", [("", "old"), ("new", ""), ("../new", "old"),
                                               ("new", "../old"), ("new", "new"), (".", "old"),
                                               ("new", "..")])
def test_reject_invalid_names(replace, tmp_path, temporary, target):
    (tmp_path / "new").write_bytes(b"candidate")
    assert replace(tmp_path, temporary, target) != 0
    assert (tmp_path / "new").read_bytes() == b"candidate"


def test_reject_root_alias_and_missing_source(replace, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    (root / "new").write_bytes(b"candidate")
    assert replace(alias) != 0
    assert replace(root, "missing") != 0
    assert (root / "new").read_bytes() == b"candidate"


@pytest.mark.parametrize("mode", ["fresh", "same", "other", "missing", "symlink"])
def test_run_reuse_requires_own_restore(filesystem, tmp_path, mode):
    root = tmp_path / "run"
    root.mkdir()
    restore = tmp_path / "external/checkpoints/source"
    restore.mkdir(parents=True)
    if mode != "fresh":
        (root / "resources").mkdir()
        (root / "checkpoints").mkdir()
        source = root / "checkpoints/source"
        source.mkdir()
        if mode == "same":
            restore = source
        elif mode == "missing":
            restore = root / "checkpoints/missing"
        elif mode == "symlink":
            source.rmdir()
            source.symlink_to(restore, target_is_directory=True)
            restore = source
    function = filesystem.astr_checkpoint_reuse_run
    function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
    reuse = ctypes.c_int(-1)
    result = function(str(root).encode(), str(restore).encode(), ctypes.byref(reuse))
    assert (result == 0) == (mode in ("fresh", "same"))
    if result == 0:
        assert reuse.value == (1 if mode == "same" else 0)


@pytest.mark.parametrize("defect", ["none", "alias", "bad_name", "hardlink", "no_restore", "incomplete"])
def test_choose_new_archive_segment(filesystem, tmp_path, defect):
    root = tmp_path / "fields"
    root.mkdir()
    (root / "resources").mkdir()
    (root / "resources/data.h5").write_bytes(b"coordinate placeholder")
    segment = root / "segment00000000"
    segment.mkdir()
    for name in ("SEGMENT", "input.txt", "series.frames"):
        (segment / name).write_bytes(b"sealed metadata placeholder")
    if defect == "alias":
        alias = tmp_path / "alias"
        alias.symlink_to(root, target_is_directory=True)
        root = alias
    elif defect == "bad_name":
        (root / "segment_bad").mkdir()
    elif defect == "hardlink":
        (root / "resources/alias.h5").hardlink_to(root / "resources/data.h5")
    elif defect == "incomplete":
        (segment / "SEGMENT").unlink()
    function = filesystem.astr_output_archive_segment
    function.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    generation, fresh = ctypes.c_int(-1), ctypes.c_int(-1)
    result = function(str(root).encode(), defect != "no_restore", ctypes.byref(generation), ctypes.byref(fresh))
    assert (result == 0) == (defect == "none")
    if result == 0:
        assert generation.value == 1 and fresh.value == 0
        assert not (root / "segment00000001").exists()


def test_choose_initial_archive_segment(filesystem, tmp_path):
    root = tmp_path / "slices"
    function = filesystem.astr_output_archive_segment
    function.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    generation, fresh = ctypes.c_int(-1), ctypes.c_int(-1)
    assert function(str(root).encode(), 0, ctypes.byref(generation), ctypes.byref(fresh)) == 0
    assert generation.value == 0 and fresh.value == 1 and root.is_dir()
