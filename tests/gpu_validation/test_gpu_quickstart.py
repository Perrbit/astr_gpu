import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples/GPU_Quickstart"
SPEC = importlib.util.spec_from_file_location("quickstart", EXAMPLES / "prepare.py")
QUICK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUICK)


@pytest.mark.parametrize("case", QUICK.CASES)
@pytest.mark.parametrize("mode", ["cpu", "gpu"])
def test_complete_input_and_topology(case, mode, tmp_path):
    dst = QUICK.prepare(case, tmp_path / case, mode, 2, "2,1,1", 3)
    text = (dst / "datin/input.dat").read_text()
    assert "\r" not in text
    lines = text.splitlines()
    assert lines[5] in ("tgv", "channel", "bl", "air5tgv", "air5hbl")
    assert lines[14].split(",")[-1] == ("t" if mode == "gpu" else "f")
    assert (dst / "datin/controller").read_text().splitlines()[8].startswith("2,2,")
    if case == "flatplate":
        import h5py
        with h5py.File(dst / "datin/grid.h5") as handle:
            assert handle["x"].shape == (17, 65, 33)
        assert np.loadtxt(dst / "datin/inlet.prof", skiprows=4).shape == (65, 4)
    if case == "air5_flatplate":
        rows = np.loadtxt(dst / "datin/air5_hbl_profile.dat")
        assert rows.shape == (4098, 13)
        assert np.isfinite(rows).all()
        assert (np.diff(rows[:, 0]) > 0).all()
        assert (rows[:, 8:] >= 0).all()
        np.testing.assert_allclose(rows[:, 8:].sum(axis=1), 1., rtol=0., atol=1e-15)
        gas = 8.31446261815324030 * (.767 / .028 + .233 / .032)
        np.testing.assert_allclose(rows[:, 1] * gas * rows[:, 6], rows[:, 5], rtol=1e-14)
    with pytest.raises(FileExistsError):
        QUICK.prepare(case, dst)


@pytest.mark.parametrize("kwargs", [
    dict(np=2), dict(np=2, topology="0,2,1"),
    dict(np=32, topology="32,1,1"), dict(steps=0),
    dict(deltat="nan"), dict(deltat="-1.d-3"),
])
def test_invalid_options_leave_no_output(tmp_path, kwargs):
    dst = tmp_path / "bad"
    with pytest.raises(ValueError):
        QUICK.prepare("tgv", dst, **kwargs)
    assert not dst.exists()


def test_scripts_parse_and_do_not_import_validation_helpers():
    for path in EXAMPLES.rglob("*.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)
    assert "tests/gpu_validation" not in (EXAMPLES / "prepare.py").read_text()


@pytest.mark.parametrize('air5,expected', [(None, 'OFF'), ('', 'OFF'), ('OFF', 'OFF'), ('ON', 'ON')])
def test_build_forwards_explicit_air5_choice(tmp_path, air5, expected):
    cmake = tmp_path / 'cmake'
    cmake.write_text(
        '#!/usr/bin/env python3\n'
        'import json, os, sys\n'
        'with open(os.environ["CMAKE_CALL_LOG"], "a") as f:\n'
        '    f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    cmake.chmod(0o755)
    log = tmp_path / 'calls.jsonl'
    env = dict(os.environ, PATH=str(tmp_path)+os.pathsep+os.environ['PATH'],
               CMAKE_CALL_LOG=str(log), BUILD_DIR=str(tmp_path / 'build'), CUDA='ON')
    env.pop('AIR5', None)
    if air5 is not None:
        env['AIR5'] = air5
    subprocess.run(['bash', str(EXAMPLES / 'build.sh')], env=env, check=True,
                   capture_output=True, text=True)
    configure, build = [json.loads(row) for row in log.read_text().splitlines()]
    assert f'-DASTR_WITH_AIR5_CHEMISTRY={expected}' in configure
    assert '-DBUILD_TESTING=OFF' in configure
    assert '-DCHEMISTRY=OFF' in configure
    assert configure[:2] == ['-S', str(ROOT)]
    assert build[:2] == ['--build', str(tmp_path / 'build')]
