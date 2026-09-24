import json
from pathlib import Path

import numpy as np
import pytest

from prepare_air5_sbli_case import prepare


def test_runtime_data_uses_profile_edge_and_frozen_jump(tmp_path):
    case = tmp_path / "case"
    input_file = prepare(case, "31,31,7", 2, "1.d-11", "t")
    meta = json.loads((case / "incident_shock_metadata.json").read_text())
    lines = (case / "datin/air5_incident_shock.dat").read_text().splitlines()
    assert lines[0] == "air5_incident_shock_v1"
    geometry = np.fromstring(lines[1], sep=" ")
    assert geometry.shape == (5,)
    np.testing.assert_array_equal(np.fromstring(lines[2], sep=" "), meta["upstream_q"])
    np.testing.assert_array_equal(np.fromstring(lines[3], sep=" "), meta["downstream_q"])
    assert meta["temperature"][0] == pytest.approx(450.)
    assert meta["temperature"][1] > 450.
    assert 0 < meta["top_x"] < meta["geometric_wall_intersection_x"] < meta["domain"][0]
    assert meta["max_scaled_normal_flux_residual"] < 2e-12
    text = input_file.read_text()
    assert "air5sbli\n" in text
    assert "3,f,0.3d0,0.05d0" in text
    assert "f,t,f,f,f,f,t,t,t" in text
    assert not (case / "datin/air5_hbl_initial_field.dat").exists()


def test_prepare_does_not_overwrite_evidence(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    evidence = case / "evidence"
    evidence.write_text("preserve")
    with pytest.raises(FileExistsError):
        prepare(case, "31,31,7", 2, "1.d-11", "t")
    assert evidence.read_text() == "preserve"
