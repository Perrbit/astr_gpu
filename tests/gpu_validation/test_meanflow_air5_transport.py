"""Link a tiny sampler against the root-CMake production objects, not a solver model."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BUILD = Path(os.environ.get("ASTR_MEANFLOW_BUILD", ROOT / "build_cpu_probe")).resolve()
MPIEXEC = os.environ.get("ASTR_OUTPUT_MPIEXEC", shutil.which("mpiexec") or "mpiexec")


def build_probe(tmp_path_factory, source):
    link = BUILD / "src/CMakeFiles/astr.dir/link.txt"
    if not link.exists():
        pytest.skip("build ASTR with AIR5 through root CMake first")
    command = shlex.split(link.read_text())
    flags = (link.parent / "flags.make").read_text().splitlines()
    includes = next(line.split("=", 1)[1] for line in flags if line.startswith("Fortran_INCLUDES ="))
    definitions = next(line.split("=", 1)[1] for line in flags if line.startswith("Fortran_DEFINES ="))
    executable = tmp_path_factory.mktemp("meanflow_transport") / "probe"
    result = []
    skip = False
    for token in command:
        if skip:
            skip = False
            continue
        if token == "-o":
            skip = True
        elif token.endswith("/astr.F90.o") or token.startswith("-Wl,--dependency-file="):
            continue
        else:
            result.append(token)
    result.extend(shlex.split(includes))
    result.extend(shlex.split(definitions))
    if os.environ.get("ASTR_MEANFLOW_GPU") == "1":
        result.append("-DTEST_MEANFLOW_GPU")
    result.extend(["-I", str(BUILD / "src"),
                   str(ROOT / "tests/gpu_validation" / source),
                   "-o", str(executable)])
    compiled = subprocess.run(result, cwd=BUILD / "src", capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return executable


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    return build_probe(tmp_path_factory, "meanflow_air5_transport_probe.F90")


@pytest.fixture(scope="module")
def compact_time_probe(tmp_path_factory):
    if os.environ.get("ASTR_MEANFLOW_GPU") != "1":
        pytest.skip("compact statistics require the CUDA build")
    return build_probe(tmp_path_factory, "compact_statistics_time_probe.cuf")


@pytest.fixture(scope="module")
def gradcal_columns_probe(tmp_path_factory):
    return build_probe(tmp_path_factory, "gradcal_columns_probe.F90")


@pytest.fixture(scope="module")
def timestep_counter_probe(tmp_path_factory):
    return build_probe(tmp_path_factory, "timestep_counter_probe.F90")


@pytest.fixture(scope="module")
def repair_lifecycle_probe(tmp_path_factory):
    if os.environ.get("ASTR_MEANFLOW_GPU") != "1":
        pytest.skip("repair device lifecycle requires the CUDA build")
    return build_probe(tmp_path_factory, "repair_lifecycle_probe.cuf")


@pytest.fixture(scope="module")
def timestep_diagnostics_probe(tmp_path_factory):
    if os.environ.get("ASTR_MEANFLOW_GPU") != "1":
        pytest.skip("CPU/GPU timestep diagnostics require the CUDA build")
    return build_probe(tmp_path_factory,"timestep_diagnostics_probe.cuf")


def test_timestep_failure_reports_distinguish_state_and_geometry(timestep_diagnostics_probe):
    result=subprocess.run([MPIEXEC,'-n','1',str(timestep_diagnostics_probe)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'TIMESTEP_DIAGNOSTICS_PASS' in result.stdout
    for reason in ('nonfinite velocity','invalid sound speed','nonfinite metric','wave-rate arithmetic'):
        assert result.stdout.count('reason='+reason)==2,result.stdout
    thermo='TIMESTEP_DIAGNOSTICS_THERMO' in result.stdout
    if thermo:
        assert result.stdout.count('reason=thermodynamic state recovery')==2,result.stdout
    assert result.stdout.count('global_ijk= 1 1 1')==2*(4+thermo),result.stdout


@pytest.mark.parametrize("ranks", [1,2])
def test_repair_device_history_survives_disable_and_restart(repair_lifecycle_probe,ranks,tmp_path):
    result=subprocess.run([MPIEXEC,'-n',str(ranks),str(repair_lifecycle_probe),str(tmp_path/'mask.h5')],
        capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert result.stdout.count('REPAIR_LIFECYCLE_PASS')==ranks


@pytest.mark.parametrize("ranks", [1, 2])
@pytest.mark.parametrize("mode", ["air5", "perfect_dim", "perfect_nondim"])
def test_production_meanflow_transport(probe, ranks, mode, tmp_path):
    prefix = tmp_path / "same_state"
    run = subprocess.run([MPIEXEC, "--oversubscribe", "-n", str(ranks), str(probe), mode, str(prefix)],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.count("MEANFLOW_TRANSPORT_PASS") == ranks
    if os.environ.get("ASTR_MEANFLOW_GPU") == "1" and mode == "air5":
        for suffix in ('','_resumed','_empty','_seed','_reset_seed','_reset_resumed'):
            with h5py.File(str(prefix) + '_cpu'+suffix+'.h5') as cpu, h5py.File(str(prefix) + '_gpu'+suffix+'.h5') as gpu:
                assert set(cpu) == set(gpu)
                for name in cpu:
                    a, b = cpu[name][:], gpu[name][:]
                    assert a.shape == b.shape and a.dtype == b.dtype
                    if name.startswith('q') or name == 'rank_extras':
                        assert np.isfinite(b).all()
                        assert np.max(np.abs(a-b), initial=0) <= 1e-14 * max(1, np.max(np.abs(a), initial=0))
                    else:
                        assert a.tobytes() == b.tobytes(), name


@pytest.mark.parametrize("mode", ["air5_invalid", "air5_nondim"])
def test_invalid_air5_statistic_rejected(probe, mode):
    run = subprocess.run([MPIEXEC, "--oversubscribe", "-n", "2", str(probe), mode],
                         capture_output=True, text=True)
    assert run.returncode != 0, run.stdout + run.stderr
    assert "AIR5 meanflow" in run.stdout + run.stderr


@pytest.mark.parametrize("ranks", [1, 2])
@pytest.mark.parametrize("action", ["write", "read"])
def test_legacy_meanflow_tu3_binding(probe, ranks, action, tmp_path):
    (tmp_path / "outdat").mkdir()
    command = [MPIEXEC, "--oversubscribe", "-n", str(ranks), str(probe),
               "perfect_nondim", "", "write"]
    written = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert written.returncode == 0, written.stdout + written.stderr
    for name in ("meanflow", "2ndsta", "3rdsta", "budget"):
        with h5py.File(tmp_path / f"outdat/{name}.h5") as data:
            assert np.all(data["mean_time_weight"][:] == 1)
    with h5py.File(tmp_path / "outdat/2ndsta.h5", "r+") as data:
        assert np.all(data["tu2"][:] == 17)
        if action == "write":
            assert np.all(data["tu3"][:] == -23)
            return
        # Exercise the reader independently of a possibly incorrect writer.
        data["tu3"][:] = -23
    command[-1] = "read"
    restored = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert restored.returncode == 0, restored.stdout + restored.stderr
    assert restored.stdout.count("MEANFLOW_LEGACY_READ_PASS") == ranks


@pytest.mark.parametrize("ranks", [1, 2])
def test_compact_nonuniform_time_restart(compact_time_probe, ranks, tmp_path):
    (tmp_path / "outdat").mkdir()
    run = subprocess.run([MPIEXEC, "--oversubscribe", "-n", str(ranks), str(compact_time_probe),
                          str(tmp_path / "compact")], cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.count("COMPACT_STATISTICS_TIME_RESTART_PASS") == ranks


def test_gradcal_only_differentiates_filled_columns(gradcal_columns_probe):
    run = subprocess.run([MPIEXEC, "-n", "1", str(gradcal_columns_probe)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "GRADCAL_COLUMNS_PASS" in run.stdout


def test_64bit_counter_mpi_hdf_and_stop(timestep_counter_probe, tmp_path):
    for value in (5,1000000000000007,np.iinfo(np.int64).max):
        (tmp_path/"counter_frames"/f'step{value:012d}').mkdir(parents=True)
    run = subprocess.run([MPIEXEC,"-n","2",str(timestep_counter_probe),str(tmp_path/"counter")],
                         capture_output=True,text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.count("TIMESTEP_COUNTER_PASS") == 2
    with h5py.File(tmp_path/"counter_serial.h5") as source:
        assert source["nstep"].dtype==np.dtype('int64')
        assert source["nstep"][0]==1000000000000007
