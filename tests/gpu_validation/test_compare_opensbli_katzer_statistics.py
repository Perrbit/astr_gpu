#!/usr/bin/env python3

import importlib.util
from pathlib import Path

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
