from pathlib import Path

import numpy as np

from air5_radau_reference import Air5RadauReference
from check_air5_c5_reacting_tgv import analyze


ROOT = Path(__file__).resolve().parents[2]
MECHANISM = ROOT / "chemMech/air5_kimjo12.json"


def write_snapshot(path: Path, q: np.ndarray) -> None:
    im, jm, km = (size - 1 for size in q.shape[:3])
    with path.open("wb") as stream:
        np.asarray([im, jm, km, 0, 11], dtype=np.int32).tofile(stream)
        np.asarray(q, order="F").ravel(order="F").tofile(stream)


def make_tgv_state() -> np.ndarray:
    model = Air5RadauReference(MECHANISM)
    points = np.linspace(0.0, 2.0 * np.pi, 5)
    q = np.empty((5, 5, 5, 11), dtype=np.float64, order="F")
    rho = 0.05
    mass_fraction = np.asarray([0.55, 0.15, 0.10, 0.12, 0.08])
    rho_species = rho * mass_fraction
    ev = model.ev_from_tv(rho_species, 1000.0)
    for i, x in enumerate(points):
        for j, y in enumerate(points):
            for k, z in enumerate(points):
                velocity = np.asarray(
                    [
                        100.0 * np.sin(x) * np.cos(y) * np.cos(z),
                        -100.0 * np.cos(x) * np.sin(y) * np.cos(z),
                        0.0,
                    ]
                )
                momentum = rho * velocity
                q[i, j, k, 0] = rho
                q[i, j, k, 1:4] = momentum
                q[i, j, k, 4] = model.q5_from_state(
                    rho, momentum, rho_species, ev, 6000.0
                )
                q[i, j, k, 5:10] = rho_species
                q[i, j, k, 10] = ev
    return q


def write_case(prefix: Path) -> None:
    pre = make_tgv_state()
    post_first = pre.copy()
    post_first[..., 10] += 1.0
    post_transport = post_first.copy()
    velocity = post_transport[..., 1] / post_transport[..., 0]
    new_velocity = velocity + 1.0e-3 * np.sin(
        np.linspace(0.0, 2.0 * np.pi, 5)
    )[:, None, None]
    post_transport[..., 4] += (
        0.5 * post_transport[..., 0] * (new_velocity**2 - velocity**2)
    )
    post_transport[..., 1] = post_transport[..., 0] * new_velocity
    post_second = post_transport.copy()
    post_second[..., 10] += 1.0
    snapshots = {
        "pre_chemistry.step00000000.rk01.rank00000000.bin": pre,
        "post_chemistry.step00000000.rk01.rank00000000.bin": post_first,
        "post_transport.step00000000.rk01.rank00000000.bin": post_transport,
        "post_chemistry.step00000000.rk02.rank00000000.bin": post_second,
    }
    for suffix, values in snapshots.items():
        write_snapshot(prefix.parent / f"{prefix.name}.{suffix}", values)
    datin = prefix.parent.parent / "datin"
    datin.mkdir()
    (datin / "parallel.info").write_text(
        "    isize     jsize     ksize\n"
        "        1         1         1\n"
        "     Rank       Irk       Jrk       Krk        IM        JM        KM"
        "        I0        J0        K0\n"
        "        0         0         0         0         4         4         4"
        "         0         0         0\n",
        encoding="ascii",
    )


def test_reacting_tgv_checker_accepts_constrained_nonuniform_step(tmp_path: Path) -> None:
    validation = tmp_path / "validation"
    validation.mkdir()
    prefix = validation / "air5"
    write_case(prefix)

    metrics = analyze(prefix, MECHANISM)

    assert metrics.max_chemistry_constraint_change == 0.0
    assert metrics.minimum_chemistry_change > 0.0
    assert metrics.max_species_mass_closure < 1.0e-14
    assert metrics.max_element_relative_change < 1.0e-14
    assert metrics.minimum_species_density > 0.0
    assert metrics.minimum_temperature > 5000.0
    assert metrics.minimum_vibrational_temperature >= 300.0
    assert metrics.transport_change > 0.0
    assert metrics.x_velocity_range > 1.0
    assert metrics.y_velocity_range > 1.0
    assert metrics.z_modulation > 1.0


def test_reacting_tgv_checker_rejects_chemistry_energy_drift(tmp_path: Path) -> None:
    validation = tmp_path / "validation"
    validation.mkdir()
    prefix = validation / "air5"
    write_case(prefix)
    path = validation / "air5.post_chemistry.step00000000.rk01.rank00000000.bin"
    with path.open("r+b") as stream:
        value_index = 4 * 5 * 5 * 5
        stream.seek(
            5 * np.dtype(np.int32).itemsize
            + value_index * np.dtype(np.float64).itemsize
        )
        value = np.fromfile(stream, dtype=np.float64, count=1)[0]
        stream.seek(-np.dtype(np.float64).itemsize, 1)
        np.asarray([value + 1.0], dtype=np.float64).tofile(stream)

    metrics = analyze(prefix, MECHANISM)

    assert metrics.max_chemistry_constraint_change >= 1.0


def test_reacting_tgv_runner_locks_same_phase_and_conservation_gates() -> None:
    runner = (
        ROOT / "tests/gpu_validation/run_air5_c5_reacting_tgv_compare.sh"
    ).read_text(encoding="utf-8")
    compact = "".join(runner.lower().split())

    assert "--initial-conditionhigh-temperature-tgv" in compact
    assert "conservation=t" in compact
    assert 'astr_air5_c4_conservation="$conservation"' in compact
    assert "check_air5_c5_reacting_tgv.py" in compact
    assert "check_air5_c4_conservation.py" in compact
    assert "pre_chemistry,post_chemistry,pre_rhs,post_update,post_transport" in compact
    assert 'astr_air5_source_mode="coupled"' in compact
