#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODULE_PATH = Path(__file__).with_name("compare_opensbli_katzer_statistics.py")
SPEC = importlib.util.spec_from_file_location("compare_opensbli_katzer_statistics", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_nonuniform_six_point_derivative_is_exact_for_degree_five():
    nodes = np.array([0.0, 0.03, 0.09, 0.2, 0.45, 0.9])
    values = 2.0 + 3.0 * nodes - 4.0 * nodes**2 + nodes**5
    weights = MODULE.derivative_weights(nodes, 0.0)
    assert abs(float(weights @ values) - 3.0) < 1.0e-11


def test_directed_zero_crossings_identify_separation_and_reattachment():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    cf = np.array([1.0, -1.0, -2.0, 2.0])
    separation, reattachment = MODULE.directed_zero_crossings(x, cf)
    assert separation == [0.5]
    assert reattachment == [2.5]
    metrics = MODULE.separation_metrics(x, cf)
    assert metrics["x_separation"] == 0.5
    assert metrics["x_reattachment"] == 2.5
    assert metrics["separation_length"] == 2.0


def test_error_metrics_use_reference_peak_scale():
    reference = np.array([-2.0, 1.0])
    candidate = np.array([-1.0, 1.5])
    metrics = MODULE.error_metrics(candidate, reference)
    assert metrics["linf"] == 1.0
    assert metrics["reference_scale"] == 2.0
    assert metrics["relative_linf_by_reference_peak"] == 0.5


def test_expected_time_allows_only_sub_microsecond_accumulation_error():
    assert MODULE.time_reached(13000.000000098078, 13000.0)
    assert not MODULE.time_reached(13000.00001, 13000.0)


def test_write_plots_does_not_require_external_latex(tmp_path):
    x = np.array([0.0, 1.0, 2.0])
    astr = {
        "cf": np.array([1.0e-3, -2.0e-4, 8.0e-4]),
        "pressure_ratio": np.array([1.0, 1.1, 1.2]),
    }
    reference = {
        "cf": np.array([1.1e-3, -1.0e-4, 7.5e-4]),
        "pressure_ratio": np.array([1.0, 1.09, 1.19]),
    }

    MODULE.write_plots(tmp_path, x, astr, reference)

    assert plt.rcParams["text.usetex"] is False
    for stem in ("skin_friction", "wall_pressure"):
        assert (tmp_path / f"{stem}.eps").stat().st_size > 0
        assert (tmp_path / f"{stem}.jpeg").stat().st_size > 0


def test_numerical_report_survives_plot_failure(tmp_path, monkeypatch):
    report = {"status": "time_reached", "astr_nstep": 325000}
    x = np.array([0.0, 1.0])
    astr = {
        "cf": np.array([1.0e-3, 2.0e-3]),
        "pressure_ratio": np.array([1.0, 1.2]),
    }
    reference = {
        "cf": np.array([1.1e-3, 2.1e-3]),
        "pressure_ratio": np.array([1.0, 1.19]),
    }

    def fail_plot(*_args, **_kwargs):
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(MODULE, "write_plots", fail_plot)
    try:
        MODULE.write_outputs(tmp_path, x, astr, reference, report)
    except RuntimeError as error:
        assert str(error) == "renderer unavailable"
    else:
        raise AssertionError("the injected plotting failure was not raised")

    assert json.loads((tmp_path / "summary.json").read_text(encoding="ascii")) == report
    assert (tmp_path / "wall_comparison.csv").is_file()
