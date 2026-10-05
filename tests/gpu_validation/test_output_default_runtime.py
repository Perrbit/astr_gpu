"""Mandatory completed-step output entry; no legacy restart fallback."""
import subprocess
import sys

import pytest

from run_output_restart_validation import compare_fields, run_case
from run_output_archive_validation import groups
from test_output_repartition_runtime import ROOT, arguments

CPU = ROOT / "build_release_restart_cpu/bin/astr"
GPU = ROOT / "build_gpu_probe/bin/astr"
FINAL = "outdat/new/checkpoints/step000000000004"


def options(path, backend):
    args = arguments(path)
    args.executable = CPU if backend == "cpu" else GPU
    return args


def assert_no_legacy_files(case):
    for pattern in ("flowfield*.h5", "auxiliary*.txt", "restart_q*.bin", "compact_stats*.bin"):
        assert not list((case / "outdat").glob(pattern)), pattern
    for name in ("islice", "jslice", "kslice", "bakup"):
        assert not [p for p in (case / name).rglob("*") if p.is_file()]


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("ranks", [1, 2])
def test_default_entry_exact_restart(tmp_path, backend, ranks):
    args = options(tmp_path, backend)
    reference, _ = run_case(args, ROOT, backend, ranks, "continuous", 4, checkpoint_interval=2,
        buffer_bytes=4096, device_reserve_bytes=1073741824)
    assert "ASTR_OUTPUT_CONFIG_FILE datin/input.output" in (reference / "run.log").read_text()
    resumed, _ = run_case(args, ROOT, backend, ranks, "resumed", 4,
        restore=reference / "outdat/new/checkpoints/step000000000002",
        checkpoint_interval=2, buffer_bytes=4096, device_reserve_bytes=1073741824)
    for name in ("state.h5", "statistics.h5"):
        compare_fields(reference / FINAL / name, resumed / FINAL / name)
    for name in ("control.bin", "archives.bin", "insitu_control.bin"):
        assert (reference / FINAL / name).read_bytes() == (resumed / FINAL / name).read_bytes()
    assert_no_legacy_files(reference)
    assert_no_legacy_files(resumed)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("override", [None, "datin/custom.output"])
def test_missing_default_is_fatal(tmp_path, backend, override):
    path = override or "datin/input.output"
    case, _ = run_case(options(tmp_path, backend), ROOT, backend, 2, "missing", 1,
        omit_output_config=True, output_config_override=override,
        reject=f"cannot load required output configuration {path}")
    assert "ASTR_OUTPUT complete_step=" not in (case / "run.log").read_text()
    assert_no_legacy_files(case)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_explicit_products_off_do_not_enable_legacy_output(tmp_path, backend):
    args = options(tmp_path, backend)
    args.statistics = False
    case, _ = run_case(args, ROOT, backend, 1, "disabled", 2, enabled=False,
        legacy_output=True, buffer_bytes=4096)
    assert not list((case / "outdat/new").rglob("COMPLETE"))
    assert "The job is done!" in (case / "run.log").read_text()
    assert_no_legacy_files(case)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_no_field_io_overrides_due_native_products(tmp_path, backend):
    args = options(tmp_path, backend)
    args.statistics = False
    case, _ = run_case(args, ROOT, backend, 2, "no_field_io", 2,
        checkpoint_interval=1, archive_groups=groups("steps"), no_field_io=True,
        buffer_bytes=4096)
    assert "The job is done!" in (case / "run.log").read_text()
    assert not list((case / "outdat").rglob("*.h5"))
    assert not list((case / "outdat").rglob("COMPLETE"))
    assert_no_legacy_files(case)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_legacy_restart_is_rejected(tmp_path, backend):
    case, _ = run_case(options(tmp_path, backend), ROOT, backend, 2, "legacy_restart", 1,
        legacy_restart=True, reject="legacy checkpoint restart is disabled")
    assert_no_legacy_files(case)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_legacy_output_flags_are_ignored(tmp_path, backend):
    args = options(tmp_path, backend)
    reference, _ = run_case(args, ROOT, backend, 2, "reference", 4, checkpoint_interval=2, buffer_bytes=4096)
    legacy, _ = run_case(args, ROOT, backend, 2, "legacy_flags", 4, legacy_output=True,
        checkpoint_interval=2, buffer_bytes=4096)
    compare_fields(reference / FINAL / "state.h5", legacy / FINAL / "state.h5")
    compare_fields(reference / FINAL / "statistics.h5", legacy / FINAL / "statistics.h5")
    assert "legacy lwsequ/lwslic are ignored" in (legacy / "run.log").read_text()
    assert_no_legacy_files(legacy)


@pytest.mark.parametrize("override", ["datin/custom.output", ""])
def test_optional_path_override(tmp_path, override):
    case, _ = run_case(options(tmp_path, "cpu"), ROOT, "cpu", 1, "override", 1,
        output_config_override=override, buffer_bytes=4096)
    expected = override or "datin/input.output"
    assert f"ASTR_OUTPUT_CONFIG_FILE {expected}" in (case / "run.log").read_text()
    assert (case / "outdat/new/checkpoints/step000000000001/COMPLETE").is_file()


@pytest.mark.parametrize("name", ["tgv", "channel", "flatplate"])
def test_quickstart_supplies_default_config(tmp_path, name):
    destination = tmp_path / name
    subprocess.run([sys.executable, str(ROOT / "examples/GPU_Quickstart/prepare.py"), name,
        "--destination", str(destination), "--mode", "cpu"], check=True, timeout=30)
    template = ROOT / "examples/GPU_Quickstart/input.output"
    assert (destination / "datin/input.output").read_bytes() == template.read_bytes()
    assert (destination / "outdat/output").is_dir()
    assert sum(p.stat().st_size for p in destination.rglob("*") if p.is_file()) < 4 * 1024**2
