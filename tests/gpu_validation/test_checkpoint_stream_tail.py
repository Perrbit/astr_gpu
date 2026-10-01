"""A short partial read can return EOF: compare stream position with exact size."""
from pathlib import Path
import subprocess

import pytest

from test_checkpoint_bundle import PROBE, MPIEXEC


@pytest.mark.parametrize("payload", [b"", b"v", b"valid-histor", b"valid-history",
                                     b"valid-historyx", b"valid-historyxxxx", b"valid-historyxxxxxxx",
                                     b"valid-historyxxxxxxxx", b"valid-history" + bytes(17)])
def test_exact_stream_end_rejects_partial_tail(tmp_path, payload):
    assert PROBE.is_file(), "build the current checkpoint_bundle_probe first"
    path = tmp_path / "history.bin"
    path.write_bytes(payload)
    result = subprocess.run([MPIEXEC, "--mca", "coll_hcoll_enable", "0", "-n", "2",
                             str(PROBE), str(path), "stream"], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=30)
    assert (result.returncode == 0) == (len(payload) == 13), result.stdout
    assert path.read_bytes() == payload
