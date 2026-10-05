"""Device interpolation and bounded synthetic distributed-trajectory probes."""
import math
import os
from pathlib import Path
import re
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]
BIN = Path(os.environ.get("ASTR_INSITU_DEVICE_PROBE_BIN", ROOT / "build_insitu_device_probes/bin"))


def test_borrowed_device_grid_rk45():
    prefix = os.environ.get("ASTR_INSITU_DEVICE_MPI_PREFIX")
    assert prefix, "Set ASTR_INSITU_DEVICE_MPI_PREFIX to the matching Open MPI installation"
    launcher = Path(prefix) / "bin/mpirun"
    # This component uses MPI only for initialization, not CUDA-aware halo/RMA.
    result = subprocess.run([str(launcher), "--prefix", prefix,
                             "--mca", "pml", "ob1", "--mca", "btl", "self,tcp",
                             "--mca", "osc", "pt2pt", "-np", "1",
                             "--mca", "coll_hcoll_enable", "0", "--mca", "coll_ucc_enable", "0",
                             "--mca", "opal_cuda_support", "0",
                             str(BIN / "insitu_viskores_rk45_probe")],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    match = re.search(r"cases=32 same_arclength_max_error=(\S+) input_host_mirror=0", result.stdout)
    assert match, result.stdout
    error = float(match.group(1))
    assert math.isfinite(error) and error <= 2e-10


@pytest.mark.parametrize("ranks", [1, 2])
def test_device_mesh_particle_continuation(ranks):
    prefix = os.environ.get("ASTR_INSITU_DEVICE_MPI_PREFIX")
    assert prefix, "Set ASTR_INSITU_DEVICE_MPI_PREFIX to the matching Open MPI installation"
    result = subprocess.run([str(Path(prefix) / "bin/mpirun"), "--prefix", prefix,
                             "--mca", "pml", "ob1", "--mca", "btl", "self,tcp",
                             "--mca", "osc", "pt2pt", "-np", str(ranks),
                             "--mca", "coll_hcoll_enable", "0", "--mca", "coll_ucc_enable", "0",
                             "--mca", "opal_cuda_support", "0",
                             str(BIN / "insitu_viskores_trace_probe")],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    match = re.search(r"particles=16 max_error=(\S+) seam_error=(\S+) transfers=(\d+)", result.stdout)
    assert match, result.stdout
    for value in match.group(1, 2):
        assert math.isfinite(float(value)) and float(value) <= 2e-10
    assert int(match.group(3)) == (16 if ranks == 2 else 0)
    assert "input_host_mirror=0" in result.stdout


@pytest.mark.parametrize('ranks,axis', [(1,0),(2,0),(2,1),(2,2)])
def test_device_bidirectional_particle_continuation(ranks, axis):
    prefix = os.environ.get("ASTR_INSITU_DEVICE_MPI_PREFIX")
    assert prefix
    result = subprocess.run([str(Path(prefix) / "bin/mpirun"), "--prefix", prefix,
        "--mca", "pml", "ob1", "--mca", "btl", "self,tcp", "--mca", "osc", "pt2pt",
        "--mca", "coll_hcoll_enable", "0", "--mca", "coll_ucc_enable", "0",
        "--mca", "opal_cuda_support", "0", "-np", str(ranks),
        str(BIN / "insitu_viskores_trace_probe"), str(axis), 'bidirectional'],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    match = re.search(r'particles=32 max_error=(\S+) seam_error=(\S+) transfers=(\d+)', result.stdout)
    assert match, result.stdout
    assert all(math.isfinite(float(match.group(i))) and float(match.group(i)) <= 2e-10 for i in (1,2))
    assert int(match.group(3)) == (32 if ranks == 2 else 0)
