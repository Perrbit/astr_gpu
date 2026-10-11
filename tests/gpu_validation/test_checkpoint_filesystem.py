"""Direct tests of the native-unavailable directory reservation implementation."""
import ctypes
import errno
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import threading

import pytest


@pytest.fixture(scope="module")
def fallback(tmp_path_factory):
    library = tmp_path_factory.mktemp("checkpoint_fallback") / "filesystem.so"
    source = Path(__file__).resolve().parents[2] / "src/checkpoint_filesystem.c"
    subprocess.run(["cc", "-std=c99", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                    str(source), "-o", str(library)], check=True, timeout=30)
    handle = ctypes.CDLL(str(library))
    function = handle.astr_checkpoint_publish_directory
    function.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    function.restype = ctypes.c_int
    return function


def test_competing_publishers_never_replace_the_winner(tmp_path, fallback):
    for number in range(20):
        root = tmp_path / str(number)
        root.mkdir(mode=0o700)
        candidate = root / "batch.tmp"
        candidate.mkdir()
        (candidate / "payload").write_bytes(b"must be preserved\n")
        ready = threading.Barrier(2)

        def publish():
            ready.wait(timeout=10)
            return fallback(str(root).encode(), b"batch")

        with ThreadPoolExecutor(max_workers=2) as pool:
            requests = [pool.submit(publish) for _ in range(2)]
            results = [request.result(timeout=10) for request in requests]
        assert results.count(0) == 1
        assert next(result for result in results if result) in (errno.EEXIST, errno.ENOENT)
        assert (root / "batch/payload").read_bytes() == b"must be preserved\n"
        assert not candidate.exists()


def test_fallback_requires_a_private_output_root(tmp_path, fallback):
    candidate = tmp_path / "batch.tmp"
    candidate.mkdir()
    tmp_path.chmod(0o770)
    try:
        assert fallback(str(tmp_path).encode(), b"batch") == errno.EPERM
        assert candidate.is_dir()
        assert not (tmp_path / "batch").exists()
    finally:
        tmp_path.chmod(0o700)
