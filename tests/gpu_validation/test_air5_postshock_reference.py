from pathlib import Path
import subprocess
import sys

import numpy as np

from air5_postshock_reference import Air5PostShockReference
from air5_radau_reference import Air5RadauReference
from check_air5_c5_postshock import (
    analyze_profile_line,
    load_global_x_line,
    snapshot_extrusion_max_abs,
)


ROOT = Path(__file__).resolve().parents[2]


def make_reference() -> Air5PostShockReference:
    chemistry = Air5RadauReference(ROOT / "chemMech/air5_kimjo12.json")
    return Air5PostShockReference(chemistry)


def test_frozen_normal_shock_conserves_fluxes_and_stays_in_mechanism_domain():
    reference = make_reference()
    upstream = reference.upstream
    postshock = reference.postshock

    np.testing.assert_allclose(
        reference.conservative_fluxes(postshock),
        reference.conservative_fluxes(upstream),
        rtol=2.0e-14,
        atol=1.0e-10,
    )
    assert upstream.mach > 1.0
    assert postshock.mach < 1.0
    assert 300.0 <= postshock.temperature <= 8000.0
    assert 1.0e3 <= postshock.pressure <= 1.0e6
    assert abs(postshock.temperature - 6693.368459884081) < 1.0e-9


def test_spatial_rhs_decomposes_into_chemical_and_vt_sources():
    reference = make_reference()
    state = reference.initial_relaxation_state
    coupled = reference.spatial_rhs(0.0, state, source_mode="coupled")
    chemical = reference.spatial_rhs(0.0, state, source_mode="chemical")
    vt = reference.spatial_rhs(0.0, state, source_mode="vt")

    np.testing.assert_allclose(coupled, chemical + vt, rtol=3.0e-14, atol=1.0e-14)
    np.testing.assert_array_equal(vt[:5], np.zeros(5))
    assert np.linalg.norm(chemical[:5]) > 0.0
    assert abs(vt[5]) > 0.0


def test_coupled_relaxation_preserves_fluxes_and_changes_the_hot_air_state():
    reference = make_reference()
    result = reference.integrate(0.02, source_mode="coupled", rtol=1.0e-9)
    sample_x = np.array([0.0, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 2.0e-2])
    profile = reference.sample(result, sample_x)
    initial_fluxes = reference.conservative_fluxes(reference.postshock)

    assert result.success
    assert np.all(profile.mass_fraction >= 0.0)
    np.testing.assert_allclose(
        np.sum(profile.mass_fraction, axis=1), np.ones(sample_x.size),
        rtol=0.0, atol=2.0e-12,
    )
    for point in profile.points:
        np.testing.assert_allclose(
            reference.conservative_fluxes(point), initial_fluxes,
            rtol=3.0e-13, atol=2.0e-8,
        )
    assert abs(profile.temperature[-1] - profile.tv[-1]) < 1.0
    assert profile.temperature[-1] < profile.temperature[0]
    assert profile.tv[-1] > profile.tv[0]
    assert profile.mass_fraction[-1, 3] > 0.05
    assert profile.mass_fraction[-1, 4] > 0.05


def test_tighter_radau_tolerance_reduces_terminal_profile_error():
    reference = make_reference()
    loose = reference.integrate(0.002, source_mode="coupled", rtol=1.0e-6)
    medium = reference.integrate(0.002, source_mode="coupled", rtol=1.0e-8)
    tight = reference.integrate(0.002, source_mode="coupled", rtol=1.0e-10)

    scale = np.maximum(np.abs(tight.y[:, -1]), reference.integration_atol())
    loose_error = np.linalg.norm((loose.y[:, -1] - tight.y[:, -1]) / scale)
    medium_error = np.linalg.norm((medium.y[:, -1] - tight.y[:, -1]) / scale)
    assert medium_error < loose_error


def test_chemical_only_radau_selects_an_admissible_initial_step():
    reference = make_reference()
    result = reference.integrate(0.002, source_mode="chemical", rtol=1.0e-10)

    assert result.success
    assert np.all(result.y >= 0.0)


def test_profile_generator_writes_complete_air5_conservative_state(tmp_path):
    output = tmp_path / "air5_postshock_profile.dat"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tests/gpu_validation/generate_air5_postshock_profile.py"),
            "--mechanism",
            str(ROOT / "chemMech/air5_kimjo12.json"),
            "--output",
            str(output),
            "--points",
            "17",
            "--length",
            "0.002",
        ],
        check=True,
    )
    values = np.loadtxt(output)
    assert values.shape == (17, 13)
    np.testing.assert_allclose(values[[0, -1], 0], [0.0, 0.002], atol=1.0e-16)
    density = values[:, 1]
    speed = values[:, 2]
    mass_fraction = values[:, 6:11]
    ev = values[:, 11]
    q5 = values[:, 12]
    np.testing.assert_allclose(np.sum(mass_fraction, axis=1), 1.0, atol=2.0e-12)
    assert np.all(density > 0.0)
    assert np.all(speed > 0.0)
    assert np.all(ev >= 0.0)
    assert np.all(q5 > 0.0)

    reference = make_reference()
    state = reference.conservative_state(reference.postshock)
    np.testing.assert_allclose(
        state,
        np.r_[
            values[0, 1],
            values[0, 1] * values[0, 2],
            0.0,
            0.0,
            values[0, 12],
            values[0, 1] * values[0, 6:11],
            values[0, 11],
        ],
        rtol=5.0e-15,
        atol=1.0e-12,
    )


def test_profile_line_checker_is_exact_for_reference_conservative_state():
    reference = make_reference()
    result = reference.integrate(0.02, source_mode="coupled", rtol=1.0e-10)
    profile = reference.sample(result, np.linspace(0.0, 0.02, 17))
    q = np.asarray([reference.conservative_state(point) for point in profile.points])

    metrics = analyze_profile_line(q, q, profile, reference.chemistry, boundary_cut=3)

    assert metrics.initial_q_max_abs == 0.0
    assert metrics.rho_max_abs == 0.0
    assert metrics.u_max_abs == 0.0
    assert metrics.temperature_max_abs < 1.0e-11
    assert metrics.tv_max_abs < 1.0e-10
    assert metrics.mass_fraction_max_abs < 1.0e-15
    assert metrics.species_mass_closure_max_abs < 1.0e-15
    assert metrics.minimum_species_density > 0.0


def test_profile_line_checker_detects_interior_species_error():
    reference = make_reference()
    result = reference.integrate(0.02, source_mode="coupled", rtol=1.0e-10)
    profile = reference.sample(result, np.linspace(0.0, 0.02, 17))
    q = np.asarray([reference.conservative_state(point) for point in profile.points])
    perturbed = q.copy()
    perturbed[8, 7] += 1.0e-5

    metrics = analyze_profile_line(
        q, perturbed, profile, reference.chemistry, boundary_cut=3
    )

    assert metrics.mass_fraction_max_abs > 1.0e-5
    assert metrics.species_mass_closure_max_abs > 1.0e-6


def test_profile_line_checker_assembles_owned_x_nodes_from_multirank_snapshots(
    tmp_path,
):
    prefix = tmp_path / "air5"
    topology = (2, 2, 1)
    hm = 3
    im = 2
    jm = 2
    km = 1
    for rank in range(4):
        irk = rank % topology[0]
        shape = (im + 2 * hm + 1, jm + 2 * hm + 1, km + 2 * hm + 1, 11)
        q = np.zeros(shape, dtype=np.float64, order="F")
        for i in range(im + 1):
            q[hm + i, hm, hm, :] = 2 * irk + i
        path = tmp_path / (
            f"air5.post_chemistry.step00000000.rk02.rank{rank:08d}.bin"
        )
        with path.open("wb") as stream:
            np.asarray([im, jm, km, hm, 11], dtype=np.int32).tofile(stream)
            q.ravel(order="F").tofile(stream)

    line = load_global_x_line(
        prefix,
        label="post_chemistry",
        stage=2,
        topology=topology,
        point_count=5,
    )

    np.testing.assert_array_equal(line[:, 0], np.arange(5, dtype=np.float64))
    assert line.shape == (5, 11)


def test_profile_line_checker_detects_transverse_or_interface_mismatch(tmp_path):
    prefix = tmp_path / "air5"
    topology = (2, 2, 1)
    hm = 3
    im = 2
    jm = 2
    km = 1
    reference_line = np.repeat(np.arange(5, dtype=np.float64)[:, None], 11, axis=1)
    for rank in range(4):
        irk = rank % topology[0]
        shape = (im + 2 * hm + 1, jm + 2 * hm + 1, km + 2 * hm + 1, 11)
        q = np.zeros(shape, dtype=np.float64, order="F")
        for i in range(im + 1):
            q[hm + i, hm : hm + jm + 1, hm : hm + km + 1, :] = 2 * irk + i
        if rank == 3:
            q[hm + 1, hm + 1, hm, 4] += 2.5e-6
        path = tmp_path / (
            f"air5.post_chemistry.step00000000.rk02.rank{rank:08d}.bin"
        )
        with path.open("wb") as stream:
            np.asarray([im, jm, km, hm, 11], dtype=np.int32).tofile(stream)
            q.ravel(order="F").tofile(stream)

    mismatch = snapshot_extrusion_max_abs(
        prefix,
        label="post_chemistry",
        stage=2,
        topology=topology,
        reference_line=reference_line,
    )

    assert abs(mismatch - 2.5e-6) < 1.0e-15


def test_profile_line_checker_cli_accepts_exact_generated_profile(tmp_path):
    profile_path = tmp_path / "air5_postshock_profile.dat"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tests/gpu_validation/generate_air5_postshock_profile.py"),
            "--mechanism",
            str(ROOT / "chemMech/air5_kimjo12.json"),
            "--output",
            str(profile_path),
            "--points",
            "17",
            "--length",
            "0.002",
        ],
        check=True,
    )
    profile = np.loadtxt(profile_path)
    q_line = np.column_stack(
        [
            profile[:, 1],
            profile[:, 1] * profile[:, 2],
            np.zeros((17, 2)),
            profile[:, 12],
            profile[:, 1, None] * profile[:, 6:11],
            profile[:, 11],
        ]
    )
    hm = 3
    shape = (17 + 2 * hm, 1 + 2 * hm, 1 + 2 * hm, 11)
    q = np.zeros(shape, dtype=np.float64, order="F")
    q[hm : hm + 17, hm, hm, :] = q_line
    prefix = tmp_path / "air5"
    for label, stage in (("pre_chemistry", 1), ("post_chemistry", 2)):
        path = tmp_path / f"air5.{label}.step00000000.rk{stage:02d}.rank00000000.bin"
        with path.open("wb") as stream:
            np.asarray([16, 0, 0, hm, 11], dtype=np.int32).tofile(stream)
            q.ravel(order="F").tofile(stream)

    report = tmp_path / "report.txt"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tests/gpu_validation/check_air5_c5_postshock.py"),
            "--prefix",
            str(prefix),
            "--profile",
            str(profile_path),
            "--mechanism",
            str(ROOT / "chemMech/air5_kimjo12.json"),
            "--topology",
            "1,1,1",
            "--boundary-cut",
            "3",
            "--report",
            str(report),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "status: pass" in report.read_text(encoding="utf-8")
