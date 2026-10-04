"""Bounded completed-step checkpoint/archive/statistics/render joint gates."""
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from run_output_archive_validation import groups, frames, check_series, check_frame, check_accounting
from run_output_restart_validation import run_case, compare_fields, archive_schedule_payload, controlled_checkpoint_buffers
from test_checkpoint_bundle import fault_library
from test_output_archive_segments import fingerprints
from test_output_insitu_restart import arguments, configuration, check_render, ROOT, FINAL

BUDGET = 64 * 1024**2


def disk_bytes(root):
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    assert size <= BUDGET, (root, size)
    return size


def verify_products(case, backend, ranks, expected, render):
    for product, steps in expected.items():
        inventory = frames(case, product)
        assert sorted(inventory) == steps
        assert len(check_series(case, product)) == len(steps)
        check_frame(inventory[12], product, case / FINAL,
                    case / "outdat/new/resources/geometry.h5", backend)
    check_accounting(case, backend, ranks)
    if render:
        check_render(case, ranks, [8, 12])
    else:
        assert not list((case / "outdat/render").glob("*.jpeg"))
    flow = max(controlled_checkpoint_buffers(case / FINAL / "state.h5", 512 + 4096))
    # Conservative simultaneous ASTR statistics, packing and rendering arrays;
    # the separate resource sampler measures third-party process memory.
    owned_nodes = 16**3 // ranks
    physical_nodes = (16 // ranks + 1) * 17**2
    host_bound = flow + physical_nodes * 256 * 8 + 4096
    device_bound = owned_nodes * (71 * 8 + 4) + physical_nodes * 25 * 8 + 1024 + 4096
    assert host_bound <= BUDGET and device_bound <= BUDGET
    return disk_bytes(case)


@pytest.fixture(scope="module", params=[("cpu", 1, "steps"), ("cpu", 2, "time"),
                                       ("gpu", 1, "steps"), ("gpu", 2, "time")])
def joint_reference(request, tmp_path_factory, fault_library):
    backend, ranks, mode = request.param
    args = arguments(tmp_path_factory.mktemp("joint_products"))
    args.mode = mode
    if backend == "cpu":
        args.executable = ROOT / "build_release_restart_cpu/bin/astr"
    config = configuration(render=backend == "gpu", mode=mode)
    baseline = None
    if backend == "gpu":
        base_args = SimpleNamespace(**vars(args))
        base_args.output = tmp_path_factory.mktemp("joint_baseline")
        plain, _ = run_case(base_args, ROOT, backend, ranks, "plain", 12,
            insitu_config=configuration(render=False, mode=mode), buffer_bytes=4096,
            checkpoint_interval=99 if mode == "steps" else 0.099,
            monitor_resources=True, device_reserve_bytes=1073741824)
        baseline = json.loads((plain / "resources.sampled.json").read_text())
        disk_bytes(base_args.output)
    continuous, _ = run_case(args, ROOT, backend, ranks, "continuous", 12,
        insitu_config=config, archive_groups=groups(mode), buffer_bytes=4096,
        checkpoint_interval=5 if mode == "steps" else 0.005,
        device_reserve_bytes=1073741824,
        monitor_resources=backend == "gpu", resource_baseline=baseline,
        publication_fault=(fault_library, "protect_batch", 5, 0))
    size = verify_products(continuous, backend, ranks,
        dict(fields=[0, 2, 4, 6, 8, 10, 12], slices=[0, 3, 6, 9, 12]), backend == "gpu")
    source = continuous / "outdat/new/checkpoints/step000000000005"
    assert (source / "PROTECT").is_file()
    return args, backend, ranks, config, continuous, source, size, baseline


def test_joint_exact_continuation(joint_reference, tmp_path, record_property):
    args, backend, ranks, config, continuous, source, reference_size, baseline = joint_reference
    args = SimpleNamespace(**vars(args)); args.output = tmp_path
    before = fingerprints(source.parent.parent)
    resumed, _ = run_case(args, ROOT, backend, ranks, "resumed", 12, restore=source,
        insitu_config=config, archive_groups=groups(args.mode), buffer_bytes=4096,
        checkpoint_interval=5 if args.mode == "steps" else 0.005,
        device_reserve_bytes=1073741824,
        monitor_resources=backend == "gpu", resource_baseline=baseline)
    for name in ("state.h5", "statistics.h5"):
        compare_fields(continuous / FINAL / name, resumed / FINAL / name)
    for name in ("control.bin", "insitu_control.bin"):
        assert (continuous / FINAL / name).read_bytes() == (resumed / FINAL / name).read_bytes()
    assert archive_schedule_payload(continuous / FINAL / "archives.bin") == archive_schedule_payload(
        resumed / FINAL / "archives.bin")
    size = verify_products(resumed, backend, ranks,
        dict(fields=[6, 8, 10, 12], slices=[6, 9, 12]), backend == "gpu")
    for product in ("fields", "slices"):
        reference = frames(continuous, product)
        for step, path in frames(resumed, product).items():
            compare_fields(reference[step] / "data.h5", path / "data.h5")
    for rank in range(ranks):
        name = f"sample.statistics.step00000012.rank{rank:08d}.bin"
        assert (continuous / "outdat/render" / name).read_bytes() == (resumed / "outdat/render" / name).read_bytes()
    if backend == "gpu":
        for path in (resumed / "outdat/render").glob("*.jpeg"):
            with Image.open(path) as actual, Image.open(continuous / "outdat/render" / path.name) as reference:
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))
        geometry = list((resumed / "outdat/render").rglob("*.vtp"))
        assert geometry
        for path in geometry:
            assert path.read_bytes() == (continuous / "outdat/render" / path.relative_to(resumed / "outdat/render")).read_bytes()
    assert before == fingerprints(source.parent.parent)
    assert not list((resumed / "outdat").glob("restart_q.*"))
    assert not list((resumed / "outdat").glob("insitu_cpu_q.*"))
    record_property("directory_bytes", [reference_size, size, disk_bytes(tmp_path)])
    if backend == "gpu":
        record_property("sampled_resources", json.loads((resumed / "resources.sampled.json").read_text()))


def test_joint_products_disabled_preserve_state(joint_reference, tmp_path, record_property):
    args, backend, ranks, _, continuous, _, reference_size, _ = joint_reference
    args = SimpleNamespace(**vars(args)); args.output = tmp_path
    plain, size = run_case(args, ROOT, backend, ranks, "plain", 12,
        insitu_config=configuration(render=False, mode=args.mode),
        archive_groups=groups(args.mode, volume=False, slices=False), buffer_bytes=4096,
        checkpoint_interval=99 if args.mode == "steps" else 0.099,
        device_reserve_bytes=1073741824)
    for name in ("state.h5", "statistics.h5"):
        compare_fields(continuous / FINAL / name, plain / FINAL / name)
    for rank in range(ranks):
        name = f"sample.statistics.step00000012.rank{rank:08d}.bin"
        assert (continuous / "outdat/render" / name).read_bytes() == (plain / "outdat/render" / name).read_bytes()
    assert not frames(plain, "fields") and not frames(plain, "slices")
    assert not list((plain / "outdat/render").glob("*.jpeg"))
    record_property("directory_bytes", [reference_size, size, disk_bytes(tmp_path)])
