"""Opt-in tests of the real Fortran/Catalyst admission path, without flow advance."""

import os
from pathlib import Path
import shlex
import subprocess

import pytest


BINARY = os.environ.get("ASTR_INSITU_TEST_BINARY")
BACKEND = os.environ.get("ASTR_INSITU_TEST_BACKEND")
MPI = shlex.split(os.environ.get("ASTR_INSITU_TEST_MPI", "mpirun"))
pytestmark = pytest.mark.skipif(not BINARY or not BACKEND,
                              reason="Set ASTR_INSITU_TEST_BINARY and ASTR_INSITU_TEST_BACKEND")


def run(tmp_path, contents, np=2, extra_env=None):
    config = tmp_path / "insitu.nml"
    config.write_bytes(contents)
    env = os.environ.copy()
    env.pop("CATALYST_IMPLEMENTATION_PREFER_ENV", None)
    env.update(extra_env or {})
    result = subprocess.run(
        [*MPI, "-np", str(np), str(Path(BINARY).resolve()), "insitu-check", str(config)],
        cwd=tmp_path, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=60,
    )
    # Admission must not create solver or visualization products.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["insitu.nml"]
    return result


@pytest.mark.parametrize("np", [1, 2])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_backend_lifecycle(tmp_path, np, newline):
    text = newline.join(["&insitu", f" implementation_path='{BACKEND}'", "/", ""])
    result = run(tmp_path, text.encode(), np)
    assert result.returncode == 0, result.stdout
    assert "ParaView backend lifecycle PASS; flow was not advanced" in result.stdout


@pytest.mark.parametrize("contents, message", [
    (b"&insitu /\n", "implementation_path is required"),
    (b"&insitu unknown=1 /\n", "invalid &insitu namelist"),
    (b"&insitu implementation_path='/nonexistent/astr/backend' /\n", "library is missing"),
    (b"!" + b" " * 5000 + b"bad\n", "line limit"),
    (b"&insitu implementation_path='" + b"a" * 1100 + b"' /\n", "exceeds 1024"),
])
def test_invalid_config(tmp_path, contents, message):
    result = run(tmp_path, contents)
    assert result.returncode != 0, result.stdout
    assert message in result.stdout


def test_env_override_rejected(tmp_path):
    result = run(tmp_path, f"&insitu implementation_path='{BACKEND}' /\n".encode(),
                 extra_env={"CATALYST_IMPLEMENTATION_PREFER_ENV": "1",
                            "CATALYST_IMPLEMENTATION_NAME": "stub"})
    assert result.returncode != 0, result.stdout
    assert "unset CATALYST_IMPLEMENTATION_PREFER_ENV" in result.stdout


@pytest.mark.parametrize("mismatch", ["filename", "value", "missing_file"])
def test_collective_config_rejection(tmp_path, mismatch):
    directories = [tmp_path / "rank0", tmp_path / "rank1"]
    command = list(MPI)
    for rank, directory in enumerate(directories):
        directory.mkdir()
        name = "other.nml" if mismatch == "filename" and rank == 1 else "insitu.nml"
        backend = BACKEND + "/." if mismatch == "value" and rank == 1 else BACKEND
        if mismatch != "missing_file" or rank == 0:
            (directory / name).write_text(f"&insitu implementation_path='{backend}' /\n")
        if rank:
            command.append(":")
        command.extend(["-np", "1", "-wdir", str(directory), str(Path(BINARY).resolve()),
                        "insitu-check", name])
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=60)
    assert result.returncode != 0, result.stdout
    message = {"filename": "configuration paths differ", "value": "backend configuration differs",
               "missing_file": "cannot open configuration on every rank"}[mismatch]
    assert message in result.stdout


def test_disabled_build(tmp_path):
    disabled = os.environ.get("ASTR_INSITU_TEST_DISABLED_BINARY")
    if not disabled:
        pytest.skip("Set ASTR_INSITU_TEST_DISABLED_BINARY for the build-off gate")
    result = subprocess.run([*MPI, "-np", "2", str(Path(disabled).resolve()),
                             "insitu-check", "not-needed.nml"], cwd=tmp_path,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    assert result.returncode != 0, result.stdout
    assert "requires ASTR_WITH_CATALYST=ON" in result.stdout
    assert not list(tmp_path.iterdir())
