from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.gpu_validation.air5_radau_reference import Air5RadauReference
from tests.gpu_validation.check_air5_c5_hbl import analyze, _similarity_farfield_q


ROOT = Path(__file__).resolve().parents[2]
MECHANISM = ROOT / "chemMech/air5_kimjo12.json"


def primitive_q(
    model: Air5RadauReference,
    density: float,
    velocity: tuple[float, float, float],
    temperature: float,
    tv: float,
    mass_fraction: np.ndarray,
) -> np.ndarray:
    rho_species = density * mass_fraction
    momentum = density * np.asarray(velocity)
    ev = model.ev_from_tv(rho_species, tv)
    q = np.empty(11)
    q[0] = density
    q[1:4] = momentum
    q[4] = model.q5_from_state(density, momentum, rho_species, ev, temperature)
    q[5:10] = rho_species
    q[10] = ev
    return q


def write_snapshot(path: Path, q: np.ndarray) -> None:
    im, jm, km = (size - 1 for size in q.shape[:3])
    with path.open("wb") as stream:
        np.asarray([im, jm, km, 0, 11], dtype=np.int32).tofile(stream)
        np.asarray(q, order="F").ravel(order="F").tofile(stream)


def apply_boundaries(
    q: np.ndarray,
    profile_q: np.ndarray,
    model: Air5RadauReference,
    wall_temperature: float,
) -> None:
    im, jm, km = (size - 1 for size in q.shape[:3])
    q[im, :, :, :] = (4.0 * q[im - 1, :, :, :] - q[im - 2, :, :, :]) / 3.0
    for i in range(im + 1):
        for k in range(km + 1):
            inner_one = q[i, 1, k]
            inner_two = q[i, 2, k]
            rho_one = inner_one[0]
            rho_two = inner_two[0]
            y_one = inner_one[5:10] / rho_one
            t_one = model.temperature_from_q5(
                rho_one, inner_one[1:4], inner_one[5:10], inner_one[10], inner_one[4]
            )
            t_two = model.temperature_from_q5(
                rho_two, inner_two[1:4], inner_two[5:10], inner_two[10], inner_two[4]
            )
            p_one = model.pressure(inner_one[5:10], t_one)
            p_two = model.pressure(inner_two[5:10], t_two)
            p_wall = (4.0 * p_one - p_two) / 3.0
            gas_constant = float(np.dot(y_one, model.gas_constant))
            q[i, 0, k] = primitive_q(
                model,
                p_wall / (gas_constant * wall_temperature),
                (0.0, 0.0, 0.0),
                wall_temperature,
                wall_temperature,
                y_one,
            )
    x = 20.0 * 1.5e-5 * np.arange(im + 1, dtype=float) / im
    farfield = _similarity_farfield_q(profile_q[jm], x, 1.0e-3)
    q[:, jm, :, :] = farfield[:, None, :]
    q[0, :, :, :] = profile_q[:, None, :]


def write_case(
    directory: Path, *, negative_trace_species: bool = False, step: int = 0
) -> tuple[Path, Path]:
    model = Air5RadauReference(MECHANISM)
    validation = directory / "validation"
    datin = directory / "datin"
    validation.mkdir(parents=True)
    datin.mkdir(parents=True)
    prefix = validation / "air5"
    profile_path = datin / "air5_hbl_profile.dat"

    y = np.asarray([0.0, 4.0e-5, 8.0e-5, 1.2e-4])
    mass_fractions = np.asarray(
        [
            [0.76, 0.23, 0.002, 0.005, 0.003],
            [0.765, 0.229, 0.0015, 0.003, 0.0015],
            [0.767, 0.230, 0.001, 0.001, 0.001],
            [0.768, 0.230, 0.0005, 0.0005, 0.001],
        ]
    )
    temperature = np.asarray([2925.0, 2400.0, 1200.0, 450.0])
    velocity = np.asarray([0.0, 900.0, 2200.0, 2900.0])
    pressure = 101325.0
    profile_q = np.empty((4, 11))
    rows: list[str] = []
    for index in range(4):
        gas_constant = float(np.dot(mass_fractions[index], model.gas_constant))
        density = pressure / (gas_constant * temperature[index])
        profile_q[index] = primitive_q(
            model,
            density,
            (velocity[index], 0.0, 0.0),
            temperature[index],
            temperature[index],
            mass_fractions[index],
        )
        columns = (
            y[index], density, velocity[index], 0.0, 0.0, pressure,
            temperature[index], temperature[index], *mass_fractions[index]
        )
        rows.append(" ".join(f"{value:.17e}" for value in columns))
    profile_path.write_text(
        "# x_origin=1.00000000000000002e-03\n" + "\n".join(rows) + "\n",
        encoding="ascii",
    )

    base = np.empty((4, 4, 3, 11), order="F")
    for j in range(4):
        base[:, j, :, :] = profile_q[j]
    pre = base.copy()
    post_first = base.copy()
    delta = 1.0e-8
    post_first[1:3, 1:3, :, 5] -= delta
    post_first[1:3, 1:3, :, 7] += delta
    apply_boundaries(post_first, profile_q, model, temperature[0])
    post_transport = post_first.copy()
    post_second = post_transport.copy()
    post_second[1:3, 1:3, :, 5] -= delta
    post_second[1:3, 1:3, :, 7] += delta
    apply_boundaries(post_second, profile_q, model, temperature[0])
    pre_rhs = post_first.copy()
    if negative_trace_species:
        pre_rhs[1, 1, 0, 9] = -1.0e-20

    snapshots = {
        f"pre_chemistry.step{step:08d}.rk01.rank00000000.bin": pre,
        f"post_chemistry.step{step:08d}.rk01.rank00000000.bin": post_first,
        f"post_transport.step{step:08d}.rk01.rank00000000.bin": post_transport,
        f"post_chemistry.step{step:08d}.rk02.rank00000000.bin": post_second,
        f"pre_rhs.step{step:08d}.rk01.rank00000000.bin": pre_rhs,
        f"pre_rhs.step{step:08d}.rk02.rank00000000.bin": pre_rhs,
        f"pre_rhs.step{step:08d}.rk03.rank00000000.bin": pre_rhs,
    }
    for suffix, values in snapshots.items():
        write_snapshot(validation / f"air5.{suffix}", values)
    (datin / "parallel.info").write_text(
        "    isize     jsize     ksize\n"
        "        1         1         1\n"
        "     Rank       Irk       Jrk       Krk        IM        JM        KM"
        "        I0        J0        K0\n"
        "        0         0         0         0         3         3         2"
        "         0         0         0\n",
        encoding="ascii",
    )
    return prefix, profile_path


def test_hbl_checker_accepts_positive_boundary_contract(tmp_path: Path) -> None:
    prefix, profile = write_case(tmp_path)

    metrics = analyze(prefix, profile, MECHANISM, ref_len=1.5e-5)

    assert metrics.minimum_density > 0.0
    assert metrics.minimum_species_density > 0.0
    assert metrics.minimum_total_energy > 0.0
    assert metrics.minimum_vibrational_energy > 0.0
    assert metrics.max_species_mass_closure < 1.0e-12
    assert metrics.max_chemistry_constraint_change == 0.0
    assert metrics.minimum_chemistry_change > 0.0
    assert metrics.max_element_relative_change < 1.0e-12
    assert metrics.inlet_scaled_error < 1.0e-8
    assert metrics.farfield_scaled_error < 1.0e-8
    assert metrics.wall_scaled_error < 1.0e-8
    assert metrics.outflow_scaled_error < 1.0e-8
    assert metrics.z_extrusion_scaled_error < 1.0e-8
    assert metrics.z_primary_extrusion_scaled_error < 1.0e-8
    assert metrics.z_momentum_reference_scaled_error == 0.0
    assert metrics.z_mean_momentum_reference_scaled_error == 0.0
    assert metrics.z_momentum_reference_rms == 0.0
    assert metrics.z_momentum_max_absolute == 0.0


def test_hbl_checker_selects_requested_long_time_step(tmp_path: Path) -> None:
    prefix, profile = write_case(tmp_path, step=37)

    metrics = analyze(prefix, profile, MECHANISM, ref_len=1.5e-5, step=37)

    assert metrics.minimum_density > 0.0


def test_hbl_checker_rejects_negative_trace_species(tmp_path: Path) -> None:
    prefix, profile = write_case(tmp_path, negative_trace_species=True)

    try:
        analyze(prefix, profile, MECHANISM, ref_len=1.5e-5)
    except ValueError as error:
        assert "negative species" in str(error)
    else:
        raise AssertionError("negative trace species must fail closed")


def test_hbl_runner_locks_a0_phase_and_open_boundary_contracts() -> None:
    runner = (
        ROOT / "tests/gpu_validation/run_air5_c5_hbl_compare.sh"
    ).read_text(encoding="utf-8")
    compact = "".join(runner.lower().split())

    assert 'grid="${grid:-31,127,7}"' in compact
    assert 'validation_step="${validation_step:-$maxstep}"' in compact
    assert 'validation_step_secondary="${validation_step_secondary:-}"' in compact
    assert 'list_frequency="${list_frequency:-100}"' in compact
    assert 'deltat="${deltat:-1.d-10}"' in compact
    assert 'extrusion_scaled_tol="${extrusion_scaled_tol:-2.0e-10}"' in compact
    assert 'extrusion_gate="${extrusion_gate:-raw}"' in compact
    assert (
        'spanwise_momentum_relative_tol="${spanwise_momentum_relative_tol:-1.0e-10}"'
        in compact
    )
    assert '--extrusion-gate"$extrusion_gate"' in compact
    assert (
        '--spanwise-momentum-relative-tol"$spanwise_momentum_relative_tol"'
        in compact
    )
    assert 'same_phase_scaled_tol="${same_phase_scaled_tol:-}"' in compact
    assert "--initial-conditionhigh-enthalpy-boundary-layer" in compact
    assert '--list-frequency"$list_frequency"' in compact
    assert "--difftermt" in compact
    assert 'astr_air5_source_mode="coupled"' in compact
    assert 'astr_air5_c4_conservation=f' in compact
    assert "ompi_mca_coll='^hcoll,ucc'" in compact
    assert "ompi_mca_pml=ob1" in compact
    assert "ompi_mca_btl=self,vader,tcp" in compact
    assert 'astr_validation_rhs_step="$validation_step"' in compact
    assert 'astr_validation_rhs_step_secondary="$validation_step_secondary"' in compact
    assert 'timeout--kill-after=10s"${timeout_seconds}s"mpirun--oversubscribe' in compact
    assert "minimum_local_extent" in compact
    assert "smallerthanhm" in compact
    assert "--labelspost_chemistry,pre_rhs,post_update,post_transport" in compact
    assert '--step"$validation_step"' in compact
    assert '--extrusion-scaled-tol"$extrusion_scaled_tol"' in compact
    assert '--scaled-tol"$same_phase_scaled_tol"' in compact
    assert "check_air5_c5_hbl.py" in compact
    assert "check_air5_c4_conservation.py" not in compact


def test_hbl_memcheck_runner_isolatedly_covers_the_gpu_a0_path() -> None:
    runner = (
        ROOT / "tests/gpu_validation/run_air5_c5_hbl_memcheck.sh"
    ).read_text(encoding="utf-8")
    compact = "".join(runner.lower().split())

    assert "--initial-conditionhigh-enthalpy-boundary-layer" in compact
    assert 'deltat="${deltat:-1.d-10}"' in compact
    assert 'if[["$out_dir"!=/*]]' in compact
    assert 'out_dir="$root_dir/$out_dir"' in compact
    assert "ompi_mca_coll='^hcoll,ucc'" in compact
    assert "ompi_mca_opal_cuda_support=0" in compact
    assert "compute-sanitizer--toolmemcheck--leak-checkfull--error-exitcode99" in compact
    assert "errorsummary:0errors" in compact
    assert "leaksummary:0bytesleaked" in compact
