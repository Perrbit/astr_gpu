"""Approved OR4 fixed AIR5 SBLI x migration; no production physics claim."""
import pytest

from run_output_air5_restart_validation import launch
from test_output_archive_segments import fingerprints
from test_output_air5_statistics_repartition import (
    arguments, execute_gate, options, partition_gate, reference_cache,
)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_partition_independence(backend, tmp_path, record_property):
    partition_gate(backend, tmp_path, record_property, case_name="sbli")


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_exact(backend, tmp_path, record_property, reference_cache):
    execute_gate(backend, 2, 2, tmp_path, record_property, reference_cache,
                 exact=True, case_name="sbli")


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("source_ranks,target_ranks", [(1, 2), (2, 1)])
def test_migration(backend, source_ranks, target_ranks, tmp_path, record_property, reference_cache):
    execute_gate(backend, source_ranks, target_ranks, tmp_path, record_property, reference_cache,
                 repeat=True, case_name="sbli")


def test_memcheck(tmp_path, record_property, reference_cache):
    execute_gate("gpu", 1, 2, tmp_path, record_property, reference_cache,
                 memcheck=True, case_name="sbli")


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("axis", ["y", "z"])
def test_unregistered_direction_rejected(backend, axis, tmp_path):
    args = arguments(tmp_path, backend, "sbli")
    seed, _ = launch(args, backend, 1, "seed", 5, **options(backend))
    tree = seed / "outdat/new"
    before = fingerprints(tree)
    args.axis = axis
    launch(args, backend, 2, "rejected", 12,
           restore=tree / "checkpoints/step000000000005",
           reject="exact restore rank count", **options(backend))
    assert fingerprints(tree) == before
