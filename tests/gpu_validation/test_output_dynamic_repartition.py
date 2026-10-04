"""Approved OR4 four-frame inlet/mean-history migration on bounded CURVE grids."""
import pytest

from run_output_restart_validation import run_case
from test_output_archive_segments import fingerprints
from test_output_curve_statistics_repartition import execute_gate
from test_output_repartition_runtime import curve_arguments, ROOT


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("axis", ["x", "z"])
def test_dynamic_exact(backend, axis, tmp_path, record_property):
    execute_gate(backend, 2, 2, axis, axis, tmp_path, record_property,
                 exact=True, case_name="dynamic")


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("source_np,target_np,source_axis,target_axis", [
    (1, 2, "x", "x"), (2, 1, "x", "x"), (1, 2, "z", "z"),
    (2, 1, "z", "z"), (2, 2, "x", "z"), (2, 2, "z", "x"),
])
def test_dynamic_repartition(backend, source_np, target_np, source_axis, target_axis,
                             tmp_path, record_property):
    execute_gate(backend, source_np, target_np, source_axis, target_axis, tmp_path,
                 record_property, case_name="dynamic", repeat=source_axis != target_axis)


@pytest.mark.parametrize("source_axis,target_axis", [("x", "z"), ("z", "x")])
def test_dynamic_repartition_memcheck(source_axis, target_axis, tmp_path, record_property):
    execute_gate("gpu", 2, 2, source_axis, target_axis, tmp_path, record_property,
                 case_name="dynamic", memcheck=True)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_dynamic_y_history_migration_is_rejected(backend, tmp_path):
    args = curve_arguments(tmp_path, backend, "z")
    args.case = "dynamic"; args.inflow_count = 12; args.legacy_statistics = True
    seed, _ = run_case(args, ROOT, backend, 2, "seed", 5, buffer_bytes=4096)
    source = seed / "outdat/new"
    before = fingerprints(source)
    args.axis = "y"
    rejected, _ = run_case(args, ROOT, backend, 2, "rejected", 12,
        restore=source / "checkpoints/step000000000005", buffer_bytes=4096,
        reject="exact restore partition mismatch")
    assert not list((rejected / "outdat/new").rglob("COMPLETE"))
    assert before == fingerprints(source)
