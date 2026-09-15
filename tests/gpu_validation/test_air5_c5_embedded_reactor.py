from pathlib import Path

import numpy as np

from tests.gpu_validation.check_air5_c5_embedded_reactor import analyze


ROOT = Path(__file__).resolve().parents[2]


def write_snapshot(path: Path, state: np.ndarray) -> None:
    q = np.empty((2, 1, 1, 11), dtype=np.float64, order="F")
    q[:, 0, 0, :] = state
    with path.open("wb") as stream:
        np.asarray([1, 0, 0, 0, 11], dtype=np.int32).tofile(stream)
        q.ravel(order="F").tofile(stream)


def test_uniform_strang_step_satisfies_reactor_contract(tmp_path: Path) -> None:
    prefix = tmp_path / "air5"
    pre = np.asarray(
        [0.05, 2.0, -0.25, 0.1, 1.0e5, 0.0275, 0.0075, 0.005, 0.006, 0.004, 100.0]
    )
    reaction_delta = np.asarray(
        [0.0, 0.0, 0.0, 0.0, 0.0, -2.8e-6, -3.2e-6, 0.0, 0.0, 6.0e-6, 1.0]
    )
    post_first = pre + reaction_delta
    post_second = post_first + reaction_delta
    snapshots = {
        "pre_chemistry.step00000000.rk01.rank00000000.bin": pre,
        "post_chemistry.step00000000.rk01.rank00000000.bin": post_first,
        "post_transport.step00000000.rk01.rank00000000.bin": post_first,
        "post_chemistry.step00000000.rk02.rank00000000.bin": post_second,
    }
    for suffix, state in snapshots.items():
        write_snapshot(tmp_path / f"air5.{suffix}", state)

    metrics = analyze(prefix, ROOT / "chemMech/air5_kimjo12.json")

    assert metrics.max_uniformity_error <= 1.0e-10
    assert metrics.max_constraint_change <= 1.0e-10
    assert metrics.max_transport_change <= 1.0e-10
    assert metrics.minimum_chemistry_change >= 1.0e-12
    assert metrics.max_species_mass_closure <= 1.0e-10
    assert metrics.max_element_relative_change <= 5.0e-12
