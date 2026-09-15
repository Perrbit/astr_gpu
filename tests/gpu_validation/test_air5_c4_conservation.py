from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "tests/gpu_validation/check_air5_c4_conservation.py"


def write_record(
    path: Path, relative_energy_drift: float, species_absolute_drift: float = 0.0
) -> None:
    q = [1.0, 0.0, 0.0, 0.0, 2.0, 0.7653, 0.2347, 0.0, 0.0, 0.0, 0.1]
    element = [54.66428571428571, 14.66875]
    initial = [0.0, 0.0, 0.0, 0.0, 0.0]
    final = [0.0, 0.0, relative_energy_drift, 0.0, species_absolute_drift]
    lines = [
        "# step phase q1 q2 q3 q4 q5 q6 q7 q8 q9 q10 q11 "
        "element_n element_o species_mass_closure rel_mass rel_energy "
        "rel_element_n rel_element_o max_species_abs_drift",
        " ".join(str(value) for value in [0, 0, *q, *element, 0.0, *initial]),
        " ".join(str(value) for value in [1, 1, *q, *element, 0.0, *final]),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_conservation_checker_accepts_roundoff_drift(tmp_path: Path) -> None:
    data = tmp_path / "air5_c4_conservation.dat"
    write_record(data, 2.0e-15)

    result = subprocess.run(
        [sys.executable, str(CHECKER), "--input", str(data)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "status: pass" in result.stdout


def test_conservation_checker_rejects_energy_drift(tmp_path: Path) -> None:
    data = tmp_path / "air5_c4_conservation.dat"
    write_record(data, 1.0e-6)

    result = subprocess.run(
        [sys.executable, str(CHECKER), "--input", str(data)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "status: fail" in result.stdout


def test_reacting_mode_allows_species_evolution_but_keeps_invariant_gates(
    tmp_path: Path,
) -> None:
    data = tmp_path / "air5_c4_conservation.dat"
    write_record(data, 2.0e-15, species_absolute_drift=1.0e-3)

    frozen = subprocess.run(
        [sys.executable, str(CHECKER), "--input", str(data)],
        check=False,
        capture_output=True,
        text=True,
    )
    reacting = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--input",
            str(data),
            "--allow-species-change",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert frozen.returncode != 0
    assert reacting.returncode == 0, reacting.stdout + reacting.stderr
    assert "species_change_gate: disabled" in reacting.stdout
