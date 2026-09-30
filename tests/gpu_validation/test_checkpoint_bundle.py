"""Bounded integrity tests for the Fortran checkpoint bundle primitive."""
import os
from pathlib import Path
import subprocess
import shutil

import pytest

from test_checkpoint_state import MPIEXEC, run as state_run

PROBE = Path(os.environ.get("ASTR_CHECKPOINT_BUNDLE_PROBE", "build_insitu_gpu/bin/checkpoint_bundle_probe")).resolve()


def bundle(path, mode, ranks=2, success=True, name=None):
    if not PROBE.is_file():
        pytest.skip("build checkpoint_bundle_probe first")
    result = subprocess.run(
        [MPIEXEC, "--oversubscribe", "-n", str(ranks), str(PROBE), str(path), mode]
        + ([] if name is None else [name]),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30,
    )
    assert (result.returncode == 0) == success, result.stdout
    assert ("PASS bundle" if success else "REJECT bundle") in result.stdout
    assert sum(p.stat().st_size for p in path.iterdir() if p.is_file()) < 4 * 1024**2


@pytest.mark.parametrize("defect", ["none", "data", "manifest", "marker", "empty_marker",
                                    "missing_marker", "missing_data", "trailing_marker"])
def test_sealed_state_integrity(tmp_path, defect):
    state_run(tmp_path / "state.h5", "write", ranks=2, components=11)
    bundle(tmp_path, "validate", success=False)
    bundle(tmp_path, "seal")
    bundle(tmp_path, "validate", ranks=1)
    if defect == "none":
        before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
        bundle(tmp_path, "seal", success=False)
        assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
        bundle(tmp_path, "validate")
        state_run(tmp_path / "state.h5", "exact", ranks=2, components=11)
        return
    if defect in {"data", "manifest", "marker"}:
        name = {"data": "state.h5", "manifest": "MANIFEST", "marker": "COMPLETE"}[defect]
        path = tmp_path / name
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 1
        path.write_bytes(data)
    elif defect == "empty_marker":
        (tmp_path / "COMPLETE").write_bytes(b"")
    elif defect == "trailing_marker":
        with (tmp_path / "COMPLETE").open("ab") as f:
            f.write(b"unexpected\n")
    else:
        (tmp_path / ("COMPLETE" if defect == "missing_marker" else "state.h5")).unlink()
    bundle(tmp_path, "validate", success=False)


def test_no_seal_when_payload_missing(tmp_path):
    bundle(tmp_path, "seal", success=False)
    assert not (tmp_path / "COMPLETE").exists()
    assert not (tmp_path / "MANIFEST").exists()


@pytest.mark.parametrize("fault", ["none", "incomplete", "destination", "destination_symlink",
                                   "latest_temp", "latest_directory"])
def test_publish_keeps_old_batch(tmp_path, fault):
    def prepare(name):
        directory = tmp_path / (name + ".tmp")
        directory.mkdir()
        state_run(directory / "state.h5", "write", ranks=2)
        bundle(directory, "seal")
        return directory

    prepare("batch1")
    bundle(tmp_path, "publish", name="batch1")
    assert (tmp_path / "LATEST").read_text() == "batch1\n"
    old = {p.name: p.read_bytes() for p in (tmp_path / "batch1").iterdir()}
    candidate = prepare("batch2")
    if fault == "incomplete":
        (candidate / "COMPLETE").unlink()
    elif fault == "destination":
        (tmp_path / "batch2").mkdir()
    elif fault == "destination_symlink":
        (tmp_path / "batch2").symlink_to("absent_destination", target_is_directory=True)
    elif fault == "latest_temp":
        (tmp_path / ".LATEST.tmp").write_text("incomplete previous update\n")
    elif fault == "latest_directory":
        (tmp_path / "LATEST").unlink()
        (tmp_path / "LATEST").mkdir()
    bundle(tmp_path, "publish", name="batch2", success=fault == "none")
    assert old == {p.name: p.read_bytes() for p in (tmp_path / "batch1").iterdir()}
    if fault != "latest_directory":
        assert (tmp_path / "LATEST").read_text() == ("batch2\n" if fault == "none" else "batch1\n")
    if fault in {"none", "latest_temp", "latest_directory"}:
        bundle(tmp_path / "batch2", "validate")
        assert not candidate.exists()
    else:
        assert candidate.exists()
    if fault == "destination_symlink":
        assert (tmp_path / "batch2").is_symlink()
        assert not (tmp_path / "absent_destination").exists()
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 4 * 1024**2


@pytest.mark.parametrize("keep", [1, 2])
@pytest.mark.parametrize("protection", ["none", "protected", "unknown", "symlink", "hardlink"])
def test_owned_retention(tmp_path, keep, protection):
    source = tmp_path / "restore_source"
    source.mkdir()
    state_run(source / "state.h5", "write", ranks=2)
    bundle(source, "seal")
    old = {p.name: p.read_bytes() for p in source.iterdir()}
    (tmp_path / "LATEST").write_text("restore_source\n")
    for i in range(1, 5):
        shutil.copytree(source, tmp_path / f"batch{i}.tmp")
    first = tmp_path / "batch1.tmp"
    if protection == "protected":
        (first / "PROTECT").write_text("retained for archival reference\n")
    elif protection == "unknown":
        (first / "unregistered.txt").write_text("must not delete\n")
    elif protection in {"symlink", "hardlink"}:
        (first / "state.h5").unlink()
        if protection == "symlink":
            (first / "state.h5").symlink_to(source / "state.h5")
        else:
            os.link(source / "state.h5", first / "state.h5")
    success = protection in {"none", "protected"}
    bundle(tmp_path, "retain", name=str(keep), success=success)
    assert old == {p.name: p.read_bytes() for p in source.iterdir()}
    if success:
        assert (tmp_path / "LATEST").read_text() == "batch4\n"
        expected = set(range(5 - keep, 5))
        if protection == "protected":
            expected.add(1)
        assert {i for i in range(1, 5) if (tmp_path / f"batch{i}").exists()} == expected
        for i in expected:
            bundle(tmp_path / f"batch{i}", "validate")
    else:
        assert (tmp_path / "LATEST").read_text() == f"batch{keep + 1}\n"
        bundle(tmp_path / "batch1", "validate")
        assert (tmp_path / "batch1" / "COMPLETE").is_file()
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 4 * 1024**2


@pytest.mark.parametrize("fault", ["none", "missing", "mutated", "symlink", "resource_dir_symlink"])
def test_shared_resources_and_relocation(tmp_path, fault):
    run_root = tmp_path / "original"
    resources = run_root / "resources"
    resources.mkdir(parents=True)
    checkpoints = run_root / "checkpoints"
    batch = checkpoints / "batch1.tmp"
    batch.mkdir(parents=True)
    state_run(batch / "state.h5", "write", ranks=2)
    shutil.copyfile(batch / "state.h5", resources / "mesh.h5")
    (resources / "model.nml").write_text("&synthetic model=1 /\n")
    bundle(batch, "resources")
    bundle(batch, "seal", success=False)
    bundle(batch, "seal_resources")
    bundle(checkpoints, "publish", name="batch1")
    # Moving the whole run removes every original absolute path.
    moved = tmp_path / "moved"
    run_root.rename(moved)
    batch = moved / "checkpoints" / "batch1"
    resources = moved / "resources"
    bundle(batch, "validate", ranks=1)
    if fault == "missing":
        (resources / "model.nml").unlink()
    elif fault == "mutated":
        (resources / "model.nml").write_text("&synthetic model=2 /\n")
    elif fault == "symlink":
        (resources / "model.nml").rename(tmp_path / "external.nml")
        (resources / "model.nml").symlink_to(tmp_path / "external.nml")
    elif fault == "resource_dir_symlink":
        resources.rename(tmp_path / "external_resources")
        resources.symlink_to(tmp_path / "external_resources", target_is_directory=True)
    bundle(batch, "validate", success=fault == "none")
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 4 * 1024**2
