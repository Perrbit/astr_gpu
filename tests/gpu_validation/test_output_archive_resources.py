"""Exercise the production bundle validator on real shared-coordinate archives."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import h5py
import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("ASTR_ARCHIVE_RESOURCE_SOURCE", str(
    ROOT / "tests/gpu_validation/out/or5_shared_geometry_cpu_checked_20261001/cpu_np2_continuous/outdat/new")))
PROBE = Path(os.environ.get("ASTR_CHECKPOINT_BUNDLE_PROBE", str(ROOT / "build_insitu_gpu/bin/checkpoint_bundle_probe")))
MPIEXEC = os.environ.get("MPIEXEC", "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec")


@pytest.mark.parametrize("product", ["fields", "slices"])
@pytest.mark.parametrize("fault", ["none", "missing", "mutated", "symlink"])
def test_real_shared_coordinate_integrity_and_relocation(tmp_path, product, fault):
    source_frame = SOURCE / product / "segment00000000/step000000000012"
    source_geometry = SOURCE / product / "resources/data.h5"
    if not source_frame.is_dir() or not PROBE.is_file():
        pytest.fail("prepare the immutable real shared-coordinate source and build the production bundle probe")
    originals = [*source_frame.iterdir(), source_geometry]
    before = {p: (hashlib.sha256(p.read_bytes()).digest(), p.stat().st_mtime_ns) for p in originals}
    moved = tmp_path / "relocated" / product
    frame = moved / "segment00000000/step000000000012"
    shutil.copytree(source_frame, frame)
    resources = moved / "resources"
    resources.mkdir()
    geometry = resources / "data.h5"
    shutil.copyfile(source_geometry, geometry)
    with h5py.File(geometry) as data:
        if product == "fields":
            assert set(data) == {"coordinates", "metadata"}
            assert data["coordinates"].shape == (17, 17, 17, 3)
        else:
            assert set(data) == {"frame_identity", "i000000000008", "j000000000000", "k000000000016"}
            for label in set(data) - {"frame_identity"}:
                assert set(data[label]) == {"coordinates", "metadata"}
                assert data[label]["coordinates"].shape == (17, 17, 3)
    victim = tmp_path / "external_geometry.h5"
    if fault == "missing":
        geometry.unlink()
    elif fault == "mutated":
        with geometry.open("r+b") as stream:
            stream.seek(geometry.stat().st_size // 2)
            byte = stream.read(1)
            stream.seek(-1, 1)
            stream.write(bytes([byte[0] ^ 1]))
    elif fault == "symlink":
        shutil.copyfile(geometry, victim)
        geometry.unlink()
        geometry.symlink_to(victim)
    result = subprocess.run([MPIEXEC, "--mca", "coll_hcoll_enable", "0", "-np", "2",
                             str(PROBE), str(frame), "validate"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
    assert (result.returncode == 0) == (fault == "none"), result.stdout
    assert ("PASS bundle" if fault == "none" else "REJECT bundle") in result.stdout
    assert {p: (hashlib.sha256(p.read_bytes()).digest(), p.stat().st_mtime_ns) for p in originals} == before
    if victim.exists():
        assert victim.read_bytes() == source_geometry.read_bytes()
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 2 * 1024**2
