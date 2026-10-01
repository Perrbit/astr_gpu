"""Link a tiny sampler against the root-CMake production objects, not a solver model."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BUILD = Path(os.environ.get("ASTR_MEANFLOW_BUILD", ROOT / "build_cpu_probe")).resolve()
MPIEXEC = os.environ.get("ASTR_OUTPUT_MPIEXEC", shutil.which("mpiexec") or "mpiexec")


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    link = BUILD / "src/CMakeFiles/astr.dir/link.txt"
    if not link.exists():
        pytest.skip("build ASTR with AIR5 through root CMake first")
    command = shlex.split(link.read_text())
    flags = (link.parent / "flags.make").read_text().splitlines()
    includes = next(line.split("=", 1)[1] for line in flags if line.startswith("Fortran_INCLUDES ="))
    executable = tmp_path_factory.mktemp("meanflow_transport") / "probe"
    result = []
    skip = False
    for token in command:
        if skip:
            skip = False
            continue
        if token == "-o":
            skip = True
        elif token.endswith("/astr.F90.o") or token.startswith("-Wl,--dependency-file="):
            continue
        else:
            result.append(token)
    result.extend(shlex.split(includes))
    if os.environ.get("ASTR_MEANFLOW_GPU") == "1":
        result.append("-DTEST_MEANFLOW_GPU")
    result.extend(["-I", str(BUILD / "src"),
                   str(ROOT / "tests/gpu_validation/meanflow_air5_transport_probe.F90"),
                   "-o", str(executable)])
    compiled = subprocess.run(result, cwd=BUILD / "src", capture_output=True, text=True, timeout=120)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return executable


@pytest.mark.parametrize("ranks", [1, 2])
@pytest.mark.parametrize("mode", ["air5", "perfect_dim", "perfect_nondim"])
def test_production_meanflow_transport(probe, ranks, mode, tmp_path):
    prefix = tmp_path / "same_state"
    run = subprocess.run([MPIEXEC, "--oversubscribe", "-n", str(ranks), str(probe), mode, str(prefix)],
                         capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.count("MEANFLOW_TRANSPORT_PASS") == ranks
    if os.environ.get("ASTR_MEANFLOW_GPU") == "1" and mode == "air5":
        with h5py.File(str(prefix) + "_cpu.h5") as cpu, h5py.File(str(prefix) + "_gpu.h5") as gpu:
            assert set(cpu) == set(gpu)
            for name in cpu:
                a, b = cpu[name][:], gpu[name][:]
                assert a.shape == b.shape and a.dtype == b.dtype
                if name.startswith("q") or name == "rank_extras":
                    assert np.isfinite(b).all()
                    assert np.max(np.abs(a-b), initial=0) <= 1e-14 * max(1, np.max(np.abs(a), initial=0))
                else:
                    assert a.tobytes() == b.tobytes(), name
        assert sum(p.stat().st_size for p in tmp_path.iterdir()) < 4 * 1024**2


@pytest.mark.parametrize("mode", ["air5_invalid", "air5_nondim"])
def test_invalid_air5_statistic_rejected(probe, mode):
    run = subprocess.run([MPIEXEC, "--oversubscribe", "-n", "2", str(probe), mode],
                         capture_output=True, text=True, timeout=30)
    assert run.returncode != 0, run.stdout + run.stderr
    assert "AIR5 meanflow" in run.stdout + run.stderr
