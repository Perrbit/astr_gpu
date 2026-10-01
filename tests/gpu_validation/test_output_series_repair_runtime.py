"""Rebuild copied indexes from immutable, previously accepted native artifacts."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

from run_output_archive_validation import check_series, check_series_reader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/output"))
spec = importlib.util.spec_from_file_location("series_repair_runtime", ROOT / "scripts/output/repair_series.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def fingerprint_tree(root):
    return {str(path.relative_to(root)): (repair.fingerprint(path), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("reference,backend,expected_fields", [
    ("or6_archive_context_cpu_20261001", "cpu", 7),
    ("or6_archive_context_air5_20261001", "gpu", 7),
])
@pytest.mark.parametrize("product", ["fields", "slices"])
def test_native_archive_index_recovery(tmp_path, reference, backend, expected_fields, product):
    references = Path(os.environ.get("ASTR_OUTPUT_REPAIR_REFERENCE_ROOT", ROOT / "tests/gpu_validation/out"))
    report_path = references / reference / "summary.json"
    if not report_path.is_file():
        pytest.skip(f"immutable accepted reference required: {report_path}")
    reference_report = json.loads(report_path.read_text())
    assert reference_report["status"] == "passed"
    source = references / reference / f"{backend}_np2_continuous/outdat/new" / product
    original = fingerprint_tree(source)
    target = tmp_path / "case/outdat/new" / product
    shutil.copytree(source, target)
    before = fingerprint_tree(target)
    segment = target / "segment00000000"
    # A complete frame can exist even when neither index records its clock.
    (segment / "series.frames").write_text("ASTR_FRAME_SERIES_1\n")
    (segment / "series.xdmf").write_text("interrupted index generation\n")
    repaired = repair.repair_segment(segment, publish=True, catalog_bytes=4096)
    expected = expected_fields if product == "fields" else 5
    assert repaired["frames"] == expected and repaired["field_array_read_bytes"] == 0
    assert len(check_series(tmp_path / "case", product)) == expected
    actual_reader = check_series_reader(tmp_path / "case", product)
    assert actual_reader["status"] == "passed" and actual_reader["frames"] == expected
    after = fingerprint_tree(target)
    for name, value in before.items():
        if Path(name).name not in repair.INDEX_NAMES:
            assert after[name] == value
    assert fingerprint_tree(source) == original
    report = {**repaired, "reader": actual_reader, "source": str(source),
              "directory_bytes": sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file())}
    (tmp_path / "repair_report.json").write_text(json.dumps(report, indent=2) + "\n")
    assert sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file()) < 64 * 1024**2
