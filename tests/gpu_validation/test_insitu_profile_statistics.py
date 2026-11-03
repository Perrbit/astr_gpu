"""Exercise the compiled local-temporal profile reducer with independent moments."""
import os
from pathlib import Path
import re
import subprocess

import pytest


PROBE = os.environ.get("ASTR_INSITU_PROFILE_PROBE")
MPIEXEC = os.environ.get("ASTR_INSITU_MPIEXEC", "mpirun")
MOMENT_PROBE = os.environ.get("ASTR_INSITU_MOMENT_PROBE")
SELECTED_PROBE = os.environ.get("ASTR_INSITU_SELECTED_PROBE")


@pytest.mark.skipif(not PROBE, reason="Set ASTR_INSITU_PROFILE_PROBE to the CMake-built spatial probe")
@pytest.mark.parametrize("ranks", (1, 2, 4))
def test_local_temporal_profile_summary(tmp_path, ranks):
    result = subprocess.run(
        [MPIEXEC, "--mca", "coll_hcoll_enable", "0", "--oversubscribe", "-np", str(ranks),
         str(Path(PROBE).resolve())],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: local temporal profile, physical lengths, Favre weights, empty ranks" in result.stdout
    assert "local RMS=1, regional-signal RMS=0" in result.stdout


@pytest.mark.skipif(not MOMENT_PROBE, reason="Set ASTR_INSITU_MOMENT_PROBE to the CMake-built compact probe")
@pytest.mark.parametrize("ranks", (1, 2, 4))
def test_selected_primitive_moments(tmp_path, ranks, record_property):
    result = subprocess.run(
        [MPIEXEC, "--mca", "coll_hcoll_enable", "0", "--oversubscribe", "-np", str(ranks),
         str(Path(MOMENT_PROBE).resolve())],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for rank in range(ranks):
        assert f"PASS compact primitive moments rank={rank} " in result.stdout
    references = re.findall(r"reference_error=\s*([0-9.eE+-]+)", result.stdout)
    assert len(references) == ranks
    record_property("reference_error_max", max(map(float, references)))
    rms_errors = re.findall(r"raw_rms_error=\s*([0-9.eE+-]+)", result.stdout)
    assert len(rms_errors) == ranks
    record_property("rms_raw_maxabs", max(map(float, rms_errors)))
    devices = re.findall(r"PASS device compact moments rank=(\d+) device=(\d+) max_scaled_error=\s*([0-9.eE+-]+)", result.stdout)
    if devices:
        assert len(devices) == ranks
        record_property("device_error_max", max(float(error) for _, _, error in devices))
        record_property("rank_devices", ",".join(f"{rank}:{device}" for rank, device, _ in sorted(devices)))


@pytest.mark.skipif(not SELECTED_PROBE, reason="Set ASTR_INSITU_SELECTED_PROBE to the selected-statistics probe")
@pytest.mark.parametrize("topology", ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2),
                                     (2, 2, 1), (2, 1, 2), (1, 2, 2)))
def test_selected_statistics_lifecycle(tmp_path, topology, record_property):
    ranks = topology[0] * topology[1] * topology[2]
    result = subprocess.run(
        [MPIEXEC, "--mca", "coll_hcoll_enable", "0", "--oversubscribe", "-np", str(ranks),
         str(Path(SELECTED_PROBE).resolve()), ",".join(map(str, topology))],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    errors = re.findall(r"PASS selected statistics rank=\d+ max_scaled_error=\s*([0-9.eE+-]+)", result.stdout)
    assert len(errors) == ranks
    record_property("max_scaled_error", max(map(float, errors)))
    record_property("topology", ",".join(map(str, topology)))
