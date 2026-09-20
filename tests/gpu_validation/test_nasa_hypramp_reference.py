from pathlib import Path

import numpy as np

from tests.gpu_validation.nasa_hypramp_reference import (
    SOURCE_URL,
    load_nasa_hypramp_run,
)


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "documents/reference_data/nasa_hypramp_mach7"


def test_pinned_nasa_mach7_run_e_contract() -> None:
    run = load_nasa_hypramp_run(REFERENCE)

    assert SOURCE_URL.startswith("https://www.grc.nasa.gov/")
    assert run.mach == 7.0
    assert run.pressure_psia == 14.7
    assert run.temperature_rankine == 520.0
    assert run.cfl == 0.5
    assert run.surface_x_ft.shape == (78,)
    assert run.exit_y_ft.shape == (11,)


def test_nasa_mach7_reference_retains_reported_shock_response() -> None:
    run = load_nasa_hypramp_run(REFERENCE)

    np.testing.assert_allclose(run.surface_x_ft[[0, -1]], [-0.5, 1.0])
    np.testing.assert_allclose(run.surface_pressure_psf[0] / 144.0, 14.6978888889)
    assert np.max(run.surface_pressure_psf) / run.surface_pressure_psf[0] > 7.0
    assert np.max(run.surface_temperature_rankine) > 4800.0
    assert run.exit_velocity_fps[0] == 0.0
    assert run.exit_velocity_fps[-1] > 7000.0
