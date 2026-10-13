"""Source evaluator selection is part of the AIR5 exact-restart contract."""
from types import SimpleNamespace
import shutil

import pytest

from run_output_air5_restart_validation import launch
from run_output_restart_validation import compare_fields
from test_output_air5_conservation import arguments, FINAL
from test_output_series_repair import seal_file_records, repair, snapshot


@pytest.fixture(scope="module", params=[(n, mode) for n in (1, 2) for mode in ("joint", "source-only")])
def reference(request, tmp_path_factory):
    ranks, mode = request.param
    args = arguments(tmp_path_factory.mktemp(f"source_{ranks}_{mode}"), case="sbli")
    args.mean_statistics = False
    args.source_evaluation = mode
    full, _ = launch(args, "gpu", ranks, "continuous", 12)
    first, _ = launch(args, "gpu", ranks, "first", 5)
    seed = first / "outdat/new/checkpoints/step000000000005"
    return args, ranks, full, seed


def modified_batch(seed, root, payload):
    shutil.copytree(seed.parent.parent / "resources", root / "resources")
    batch = root / "checkpoints" / seed.name
    shutil.copytree(seed, batch)
    (batch / "air5_config.bin").write_bytes(payload)
    names = [row.split()[0] for row in (batch / "MANIFEST").read_text().splitlines()[2:]]
    (batch / "MANIFEST").write_text(seal_file_records(batch, names, "ASTR_CHECKPOINT_BUNDLE 1"))
    size, crc = repair.fingerprint(batch / "MANIFEST")
    (batch / "COMPLETE").write_text(f"ASTR_COMPLETE_1 {size} {crc:016X}\n")
    return batch


def test_matching_mode_continues_exactly(reference, tmp_path):
    args, ranks, full, seed = reference
    args = SimpleNamespace(**vars(args))
    args.output = tmp_path
    before = snapshot(seed.parent.parent)
    resumed, _ = launch(args, "gpu", ranks, "resumed", 12, restore=seed)
    compare_fields(full / FINAL / "state.h5", resumed / FINAL / "state.h5")
    config = (resumed / FINAL / "air5_config.bin").read_bytes()
    assert len(config) == 1768 and config[:8] == b"ASTRA502"
    assert config[1640:] == args.source_evaluation.encode().ljust(128, b" ")
    assert config == (full / FINAL / "air5_config.bin").read_bytes()
    assert before == snapshot(seed.parent.parent)


def test_changing_mode_is_rejected(reference, tmp_path):
    args, ranks, _, seed = reference
    args = SimpleNamespace(**vars(args))
    args.output = tmp_path
    args.source_evaluation = "source-only" if args.source_evaluation == "joint" else "joint"
    launch(args, "gpu", ranks, "changed", 12, restore=seed,
           reject="AIR5 source evaluation contract mismatch")


def test_pre_extension_batch_means_joint_only(reference, tmp_path):
    args, ranks, full, seed = reference
    if args.source_evaluation != "joint":
        pytest.skip("pre-extension batches were joint-only")
    args = SimpleNamespace(**vars(args))
    args.output = tmp_path
    batch = modified_batch(seed, tmp_path / "pre_extension", (seed / "air5_config.bin").read_bytes()[:1640])
    before = snapshot(batch.parent.parent)
    resumed, _ = launch(args, "gpu", ranks, "legacy_joint", 12, restore=batch)
    compare_fields(full / FINAL / "state.h5", resumed / FINAL / "state.h5")
    args.source_evaluation = "source-only"
    launch(args, "gpu", ranks, "legacy_source_only", 12, restore=batch,
           reject="AIR5 source evaluation contract mismatch")
    assert before == snapshot(batch.parent.parent)


@pytest.mark.parametrize("tail", [b"x", b"x" * 128, b"x" * 129])
def test_invalid_appended_field_is_rejected(tmp_path, tail):
    args = arguments(tmp_path, case="sbli")
    args.mean_statistics = False
    first, _ = launch(args, "gpu", 1, "first", 1)
    seed = first / "outdat/new/checkpoints/step000000000001"
    batch = modified_batch(seed, tmp_path / "modified", (seed / "air5_config.bin").read_bytes()[:1640] + tail)
    message = "AIR5 source evaluation contract mismatch" if len(tail) == 128 else "AIR5 configuration size/tail"
    launch(args, "gpu", 1, "invalid", 2, restore=batch, reject=message)


def test_cpu_does_not_silently_ignore_source_only(tmp_path):
    args = arguments(tmp_path)
    args.mean_statistics = False
    args.source_evaluation = "source-only"
    launch(args, "cpu", 1, "unsupported", 1, reject="AIR5 source-only evaluation is GPU-only")


def test_cpu_joint_three_resource_metadata_continues_exactly(tmp_path):
    args = arguments(tmp_path)
    args.mean_statistics = False
    full, _ = launch(args, "cpu", 1, "continuous", 2)
    first, _ = launch(args, "cpu", 1, "first", 1)
    seed = first / "outdat/new/checkpoints/step000000000001"
    config = (seed / "air5_config.bin").read_bytes()
    assert len(config) == 1760 and config[:8] == b"ASTRA501"
    assert config[1632:] == b"joint".ljust(128, b" ")
    batch = modified_batch(seed, tmp_path / "pre_extension", config[:1632])
    relative = "outdat/new/checkpoints/step000000000002/state.h5"
    for name, source in (("new", seed), ("old", batch)):
        resumed, _ = launch(args, "cpu", 1, name, 2, restore=source)
        compare_fields(full / relative, resumed / relative)
