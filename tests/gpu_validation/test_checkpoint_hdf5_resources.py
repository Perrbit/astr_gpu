"""Check native HDF5 dependency rejection without reading field payloads."""
import ctypes
import os
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np
import pytest


@pytest.fixture(scope="module")
def inspect_resource(tmp_path_factory):
    wrapper = os.environ.get("ASTR_OUTPUT_H5CC", shutil.which("h5cc"))
    if not wrapper:
        pytest.skip("set ASTR_OUTPUT_H5CC to the matching HDF5 C wrapper")
    build = tmp_path_factory.mktemp("hdf5_resource_checker")
    source = Path(__file__).resolve().parents[2] / "src/checkpoint_hdf5_resources.c"
    library = build / "checker.so"
    subprocess.run([wrapper, "-shared", "-fPIC", str(source), "-o", str(library)],
                   cwd=build, check=True, timeout=30)
    function = ctypes.CDLL(str(library)).astr_checkpoint_hdf5_self_contained
    function.argtypes = [ctypes.c_char_p]
    function.restype = ctypes.c_int
    return lambda path: function(str(path).encode())


def test_self_contained_nested_fields(inspect_resource, tmp_path):
    path = tmp_path / "field.h5"
    with h5py.File(path, "w") as state:
        state["nested/time"] = np.float64(1e-5)
        state.create_dataset("nested/field", data=np.ones((4, 3)), compression="gzip")
        state["alias"] = state["nested/field"]
    assert inspect_resource(path) == 0


@pytest.mark.parametrize("kind", ["soft", "external", "virtual", "external_storage"])
def test_hidden_hdf5_dependencies_rejected(inspect_resource, tmp_path, kind):
    external = tmp_path / "external.h5"
    with h5py.File(external, "w") as state:
        state["field"] = np.ones((4, 3))
    path = tmp_path / "field.h5"
    with h5py.File(path, "w") as state:
        if kind == "soft":
            state["local"] = np.ones((4, 3))
            state["nested/field"] = h5py.SoftLink("/local")
        elif kind == "external":
            state["nested/field"] = h5py.ExternalLink(str(external), "field")
        elif kind == "virtual":
            layout = h5py.VirtualLayout(shape=(4, 3), dtype=np.float64)
            layout[:] = h5py.VirtualSource(str(external), "field", shape=(4, 3))
            state.create_virtual_dataset("nested/field", layout)
        else:
            state.create_dataset("nested/field", shape=(4, 3), dtype=np.float64,
                                 external=[(str(tmp_path / "raw.bin"), 0, h5py.h5f.UNLIMITED)])
    assert inspect_resource(path) != 0
