#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest


MODULE_PATH = Path(__file__).with_name("prepare_turbulent_inflow_from_slices.py")
SPEC = importlib.util.spec_from_file_location("prepare_turbulent_inflow_from_slices", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_raw_series(directory: Path, *, broken_seam: bool = False, nonuniform_time: bool = False):
    directory.mkdir()
    y = np.array([0.0, 0.02, 0.06, 0.14, 0.30, 0.60, 1.0])
    z = np.linspace(0.0, 2.0 * np.pi, 5)
    mean_rho = 0.8 + 0.2 * y
    mean_u = 1.0 - np.exp(-8.0 * y)
    mean_v = 0.01 * y
    mean_t = 1.2 - 0.2 * y
    absolute = []
    times = np.arange(8, dtype=np.float64) * 0.25
    if nonuniform_time:
        times[4:] += 0.01
    for index, time in enumerate(times):
        phase = 2.0 * np.pi * index / times.size
        spatial = np.sin(z)[..., None] * np.sin(np.pi * y)[None, :]
        temporal = np.cos(phase)
        rho = mean_rho[None, :] + 0.01 * temporal * spatial
        u1 = mean_u[None, :] + 0.03 * temporal * spatial
        u2 = mean_v[None, :] + 0.005 * np.sin(phase) * spatial
        u3 = 0.02 * np.sin(phase) * spatial
        temperature = mean_t[None, :] + 0.02 * temporal * spatial
        pressure = rho * temperature / (1.4 * 5.0**2)
        if broken_seam and index == 3:
            u1[-1, 3] += 1.0e-3
        with h5py.File(directory / f"islice{index:05d}.h5", "w") as handle:
            handle.create_dataset("time", data=time)
            for name, values in (
                ("u1", u1),
                ("u2", u2),
                ("u3", u3),
                ("p", pressure),
                ("t", temperature),
            ):
                handle.create_dataset(name, data=values)
        absolute.append({"ro": rho, "u1": u1, "u2": u2, "u3": u3, "t": temperature})
    return y, absolute


def test_conversion_reconstructs_absolute_series_and_writes_profile(tmp_path):
    raw = tmp_path / "raw"
    output = tmp_path / "prepared"
    y, absolute = write_raw_series(raw)

    report = MODULE.prepare(
        source_dir=raw,
        output_dir=output,
        y=y,
        mach=5.0,
        reynolds=1000.0,
        reference_temperature=300.0,
        periodic_atol=1.0e-12,
    )

    assert report["status"] == "diagnostic_only"
    assert report["slice_count"] == 8
    assert report["density_source"] == "reconstructed_from_pressure_temperature"
    profile = np.loadtxt(output / "inlet.prof", skiprows=4)
    assert profile.shape == (y.size, 5)
    for index, original in enumerate(absolute):
        with h5py.File(output / "inflow" / f"islice{index:05d}.h5", "r") as handle:
            assert float(handle["time"][()]) == pytest.approx(0.25 * index)
            for column, name in enumerate(("ro", "u1", "u2", "t")):
                reconstructed = np.asarray(handle[name]) + profile[None, :, column]
                np.testing.assert_allclose(reconstructed, original[name], atol=2.0e-15, rtol=0.0)
            np.testing.assert_allclose(np.asarray(handle["u3"]), original["u3"], atol=2.0e-15, rtol=0.0)

    manifest = json.loads((output / "turbulent_inflow_report.json").read_text(encoding="ascii"))
    assert manifest["periodic_seam_linf"] <= 1.0e-12
    assert manifest["minimum_density"] > 0.0
    assert manifest["minimum_temperature"] > 0.0
    assert (output / "profile_statistics.npz").is_file()


def test_conversion_rejects_nonperiodic_spanwise_endpoint(tmp_path):
    raw = tmp_path / "raw"
    write_raw_series(raw, broken_seam=True)
    with pytest.raises(ValueError, match="periodic seam"):
        MODULE.prepare(
            source_dir=raw,
            output_dir=tmp_path / "prepared",
            y=np.array([0.0, 0.02, 0.06, 0.14, 0.30, 0.60, 1.0]),
            mach=5.0,
            reynolds=1000.0,
            reference_temperature=300.0,
            periodic_atol=1.0e-12,
        )


def test_conversion_rejects_nonuniform_slice_times(tmp_path):
    raw = tmp_path / "raw"
    write_raw_series(raw, nonuniform_time=True)
    with pytest.raises(ValueError, match="uniformly spaced"):
        MODULE.prepare(
            source_dir=raw,
            output_dir=tmp_path / "prepared",
            y=np.array([0.0, 0.02, 0.06, 0.14, 0.30, 0.60, 1.0]),
            mach=5.0,
            reynolds=1000.0,
            reference_temperature=300.0,
            periodic_atol=1.0e-12,
        )
