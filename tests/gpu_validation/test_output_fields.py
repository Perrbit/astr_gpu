"""Root-production CPU/GPU tiled volume/plane packing; not flow integration."""
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
BUILD = Path(os.environ.get("ASTR_FIELDS_BUILD", ROOT / "build_cpu_probe")).resolve()
MPIEXEC = os.environ.get("ASTR_OUTPUT_MPIEXEC", shutil.which("mpiexec") or "mpiexec")
GPU = os.environ.get("ASTR_FIELDS_GPU") == "1"
NAMES = ["density", "velocity_x", "velocity_y", "velocity_z", "pressure", "temperature",
         "vibrational_temperature", "mass_fraction_N2", "mass_fraction_O2", "mass_fraction_N",
         "mass_fraction_O", "mass_fraction_NO"]
DERIVED_NAMES = ["velocity_gradient_xx", "velocity_gradient_yx", "velocity_gradient_zx",
                 "velocity_gradient_xy", "velocity_gradient_yy", "velocity_gradient_zy",
                 "velocity_gradient_xz", "velocity_gradient_yz", "velocity_gradient_zz",
                 "Q_rs", "velocity_divergence", "vorticity_x", "vorticity_y", "vorticity_z"]


def run_probe(command, *, env=None, timeout=45):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True, env=env)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    link = BUILD / "src/CMakeFiles/astr.dir/link.txt"
    if not link.exists():
        pytest.skip("complete a root CMake AIR5 build first")
    command = shlex.split(link.read_text())
    includes = next(line.split("=", 1)[1] for line in
                    (link.parent / "flags.make").read_text().splitlines()
                    if line.startswith("Fortran_INCLUDES ="))
    executable = tmp_path_factory.mktemp("output_fields") / "probe"
    result = []
    skip = False
    for token in command:
        if skip:
            skip = False
        elif token == "-o":
            skip = True
        elif not (token.endswith("/astr.F90.o") or token.startswith("-Wl,--dependency-file=")):
            result.append(token)
    result.extend(shlex.split(includes))
    result.append("-DASTR_AIR5_CHEMISTRY")
    if GPU:
        result.append("-DTEST_FIELDS_GPU")
    result.extend(["-I", str(BUILD / "src"), str(ROOT / "tests/gpu_validation/output_fields_probe.F90"),
                   "-o", str(executable)])
    compiled = subprocess.run(result, cwd=BUILD / "src", capture_output=True, text=True, timeout=120)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return executable


def manufactured_derivatives():
    k, j, i = np.indices((11, 11, 11), dtype=np.float64)
    theta = np.stack((i, j, k), axis=-1)*(2*np.pi/10)
    x, y, z = np.moveaxis(theta, -1, 0)
    h = 2*np.pi/10
    discrete = lambda m: 1.5*np.sin(m*h)-0.3*np.sin(2*m*h)+np.sin(3*m*h)/30
    metric = 10/(np.pi*np.array([2, 3, 4]))
    d1, d2 = discrete(1), discrete(2)
    gradient = np.empty((11, 11, 11, 3, 3))
    gradient[..., 0, 0] = np.cos(x)*d1*metric[0]
    gradient[..., 1, 0] = -0.5*np.sin(x)*d1*metric[0]
    gradient[..., 2, 0] = 0.125*np.cos(2*x)*d2*metric[0]
    gradient[..., 0, 1] = -0.25*np.sin(y)*d1*metric[1]
    gradient[..., 1, 1] = np.cos(2*y)*d2*metric[1]
    gradient[..., 2, 1] = -0.5*np.sin(2*y)*d2*metric[1]
    gradient[..., 0, 2] = 0.125*np.cos(2*z)*d2*metric[2]
    gradient[..., 1, 2] = -0.25*np.sin(z)*d1*metric[2]
    gradient[..., 2, 2] = np.cos(z)*d1*metric[2]
    fields = [gradient[..., c, d] for d in range(3) for c in range(3)]
    fields += [-0.5*np.einsum("...ij,...ji->...", gradient, gradient),
               np.trace(gradient, axis1=-2, axis2=-1),
               gradient[..., 2, 1]-gradient[..., 1, 2],
               gradient[..., 0, 2]-gradient[..., 2, 0],
               gradient[..., 1, 0]-gradient[..., 0, 1]]
    velocity = np.stack((np.sin(x)+0.25*np.cos(y)+0.125*np.sin(2*z),
                         0.5*np.cos(x)+np.sin(2*y)+0.25*np.cos(z),
                         0.125*np.sin(2*x)+0.5*np.cos(2*y)+np.sin(z)), axis=-1)
    return fields, velocity


@pytest.mark.parametrize("ranks,decomposition,mode", [(1, 1, "all"), (2, 1, "all"),
    (2, 2, "all"), (2, 3, "all"), (1, 1, "gradient"), (1, 1, "curl"),
    (1, 1, "q"), (2, 1, "slices"), (2, 2, "slices"), (2, 3, "slices")])
def test_private_manufactured_derivatives(probe, ranks, decomposition, mode, tmp_path):
    """Manufactured snapshots only: no RK, physics gate, restart or runtime admission."""
    axes = range(1, 4) if mode == "slices" else range(4)
    for backend in range(2 if GPU else 1):
        for axis in axes:
            (tmp_path / f"{backend}_{axis}").mkdir()
    run = run_probe([MPIEXEC, "--oversubscribe", "--mca", "coll_hcoll_enable", "0", "-n", str(ranks),
                     str(probe), str(tmp_path), "6", str(decomposition), "numeric_"+mode])
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.count("OUTPUT_PRIVATE_DERIVATIVES_PASS") == ranks
    fields, velocity = manufactured_derivatives()
    selected = {"gradient": range(9), "curl": range(11, 14), "q": range(9, 11)}.get(mode, range(14))
    errors, transfers = [], []
    for axis in axes:
        selection = [slice(None)]*3
        if axis:
            selection[3-axis] = 5
        selection = tuple(selection)
        expected_nodes = velocity[selection].shape[:-1]
        for backend in range(2 if GPU else 1):
            root = tmp_path / f"{backend}_{axis}"
            with h5py.File(root / "data.h5") as saved:
                np.testing.assert_allclose(saved["velocity"][:], velocity[selection], rtol=0, atol=2e-10)
                for field in selected:
                    actual = saved[DERIVED_NAMES[field]][:]
                    assert actual.shape == expected_nodes
                    error = float(np.max(np.abs(actual-fields[field][selection])))
                    assert error <= 2e-10, (backend, axis, DERIVED_NAMES[field], error)
                    errors.append(error)
                if backend:
                    with h5py.File(tmp_path / f"0_{axis}/data.h5") as cpu:
                        for field in selected:
                            np.testing.assert_allclose(saved[DERIVED_NAMES[field]][:], cpu[DERIVED_NAMES[field]][:],
                                                       rtol=0, atol=2e-10)
            rank_reports = [list(map(int, path.read_text().split())) for path in sorted(root.glob("budget_rank*.txt"))]
            assert len(rank_reports) == ranks
            expected_bytes = int(np.prod(expected_nodes))*(6+len(selected))*8
            assert sum(row[1] for row in rank_reports) == expected_bytes
            assert sum(row[2] for row in rank_reports) == backend*expected_bytes
            for peak, written, download, host, device in rank_reports:
                assert peak <= 2160 and host+peak <= 2*1024**2 and device <= 2*1024**2
                assert host >= 0 and written >= 0 and download >= 0
            transfers.append((backend, axis, expected_bytes))
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) <= 4*1024**2
    (tmp_path / "numerical_summary.txt").write_text(f"max_absolute_error {max(errors):.17e}\n"
                                                   f"selected_product_bytes {transfers!r}\n")


@pytest.mark.parametrize("fault,message", [("budget", "private CPU derivative budget"),
    ("boundary", "private derivatives require periodic"), ("scheme", "private derivatives require periodic")])
def test_private_derivative_provider_rejected(probe, fault, message, tmp_path):
    (tmp_path / "0_0").mkdir()
    run = run_probe([MPIEXEC, "--oversubscribe", "--mca", "coll_hcoll_enable", "0", "-n", "2",
                     str(probe), str(tmp_path), "6", "1", "numeric_"+fault])
    assert run.returncode != 0
    assert message in run.stdout + run.stderr
    assert not list(tmp_path.rglob("*.h5"))


@pytest.mark.skipif(not GPU, reason="requires CUDA root build and two devices")
@pytest.mark.parametrize("fault,message", [("budget", "private GPU derivative budget"),
    ("boundary", "private GPU derivatives require periodic"),
    ("scheme", "private GPU derivatives require periodic"),
    ("busy", "private GPU output requires completed halo transport")])
def test_private_gpu_derivative_provider_rejected(probe, fault, message, tmp_path):
    (tmp_path / "1_0").mkdir()
    run = run_probe([MPIEXEC, "--oversubscribe", "--mca", "coll_hcoll_enable", "0", "-n", "2",
                     str(probe), str(tmp_path), "6", "1", "numeric_gpu_"+fault])
    assert run.returncode != 0
    assert message in run.stdout + run.stderr
    assert not list(tmp_path.rglob("*.h5"))


@pytest.mark.skipif(not GPU or os.environ.get("ASTR_FIELDS_MEMCHECK") != "1", reason="opt-in CUDA memcheck")
@pytest.mark.parametrize("decomposition", [1, 2, 3])
def test_private_derivatives_memcheck(probe, decomposition, tmp_path):
    tool = shutil.which("compute-sanitizer")
    assert tool, "Compute Sanitizer is required for this gate"
    for backend in range(2):
        for axis in range(4):
            (tmp_path / f"{backend}_{axis}").mkdir()
    environment = dict(os.environ, OMPI_MCA_pml="ob1", OMPI_MCA_btl="self,vader,tcp", OMPI_MCA_osc="pt2pt",
                       OMPI_MCA_coll="^hcoll,ucc,cuda", OMPI_MCA_opal_cuda_support="0", UCX_MEMTYPE_CACHE="n")
    run = run_probe([MPIEXEC, "--oversubscribe", "--mca", "coll_hcoll_enable", "0", "-n", "2", tool,
                     "--tool", "memcheck", "--error-exitcode", "99", "--log-file", str(tmp_path / "memcheck.%p.log"),
                     str(probe), str(tmp_path), "6", str(decomposition), "numeric_all"], env=environment, timeout=90)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.count("OUTPUT_PRIVATE_DERIVATIVES_PASS") == 2
    logs = list(tmp_path.glob("memcheck.*.log"))
    assert len(logs) == 2
    for log in logs:
        assert "ERROR SUMMARY: 0 errors" in log.read_text(), log.read_text()
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) <= 4*1024**2


@pytest.mark.parametrize("product", ["volume", "planes"])
@pytest.mark.parametrize("components", [6, 12])
def test_time_series_metadata(probe, product, components, tmp_path):
    (tmp_path / "series.frames").write_text("ASTR_FRAME_SERIES_1\n0 0\n2 0.002\n12 0.012\n")
    run = run_probe([MPIEXEC, "--oversubscribe", "--mca", "coll_hcoll_enable", "0", "-n", "2",
                     str(probe), str(tmp_path), str(components), "1", "series_" + product])
    assert run.returncode == 0, run.stdout + run.stderr
    grids = ET.parse(tmp_path / "series.xdmf").findall("./Domain/Grid/Grid")
    assert [float(grid.find("Time").attrib["Value"]) for grid in grids] == [0, 0.002, 0.012]
    for grid, step in zip(grids, (0, 2, 12)):
        assert grid.attrib["Name"] == f"step{step:012d}"
        for item in grid.findall(".//DataItem"):
            filename = item.text.strip().split(":")[0]
            assert filename in ("../resources/data.h5", f"step{step:012d}/data.h5")


@pytest.mark.parametrize("records", ["0\n", "0 0\n2\n", "0 0\n0 0.1\n", "0 0\n2 0\n",
                                     "0 0\n2 NaN\n", "-1 0\n", "0 -0.5\n", "0 0 extra\n", "\n"])
def test_incomplete_or_invalid_series_rejected(probe, records, tmp_path):
    (tmp_path / "series.frames").write_text("ASTR_FRAME_SERIES_1\n" + records)
    run = run_probe([MPIEXEC, "--oversubscribe", "--mca", "coll_hcoll_enable", "0", "-n", "2",
                     str(probe), str(tmp_path), "6", "1", "series_volume"])
    assert run.returncode != 0
    assert "write completed-frame time series" in run.stdout + run.stderr


@pytest.mark.parametrize("unit_parity", [1, 2])
def test_repeated_series_generation(probe, tmp_path, unit_parity):
    (tmp_path / "series.frames").write_text("ASTR_FRAME_SERIES_1\n0 0\n3 0.003\n5 0.005\n")
    run = run_probe([MPIEXEC, "--oversubscribe", "--mca", "coll_hcoll_enable", "0", "-n", "1",
                     str(probe), str(tmp_path), "6", str(unit_parity), "series_repeat"])
    assert run.returncode == 0, run.stdout + run.stderr
    for path in sorted(tmp_path.glob("series*.xdmf")):
        grids = ET.parse(path).findall("./Domain/Grid/Grid")
        assert len(grids) == 3, f"repeated series lost frames: {path.name}\n{run.stdout}"
    assert len(list(tmp_path.glob("series*.xdmf"))) == 50


@pytest.mark.parametrize("ranks", [1, 2])
@pytest.mark.parametrize("decomposition", [1, 2, 3])
@pytest.mark.parametrize("components", [6, 12])
def test_bounded_production_fields(probe, ranks, decomposition, components, tmp_path):
    for backend in range(2 if GPU else 1):
        for axis in range(4):
            (tmp_path / f"{backend}_{axis}").mkdir()
    run = run_probe([MPIEXEC, "--oversubscribe", "--mca", "coll_hcoll_enable", "0",
                     "-n", str(ranks), str(probe), str(tmp_path), str(components), str(decomposition)])
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.count("OUTPUT_FIELDS_PASS") == ranks
    k, j, i = np.indices((5, 7, 9))
    owner = np.zeros_like(i)
    if ranks == 2:
        indices = (i, j, k)[decomposition-1]
        owner = indices >= (8, 6, 4)[decomposition-1]//2
    encoding = 100*i + 10*j + k + owner/1024
    coordinates = np.stack((i+j/32, j+i/16, k+i*j/64), axis=-1)
    for axis in range(4):
        selected = [slice(None)]*3
        if axis:
            selected[3-axis] = (8, 6, 4)[axis-1]//2
        selected = tuple(selected)
        expected_coordinates = coordinates[selected]
        for backend in range(2 if GPU else 1):
            frame = tmp_path / f"{backend}_{axis}"
            with h5py.File(frame / "data.h5") as fields:
                assert fields.attrs["units"].decode() == "si"
                assert fields.attrs["center"].decode() == "Node"
                assert fields.attrs["phase"].decode() == "completed_step"
                assert np.array_equal(fields["coordinates"][:], expected_coordinates)
                for component, name in enumerate(NAMES[:components], start=1):
                    expected = (encoding+component/16)[selected]
                    assert fields[name].dtype == np.float64
                    assert fields[name][:].tobytes() == expected.astype(np.float64).tobytes()
                expected_velocity = np.stack([(encoding+m/16)[selected] for m in (2, 3, 4)], axis=-1)
                assert fields["velocity"][:].tobytes() == expected_velocity.tobytes()
                meta = fields["metadata"][:]
                assert meta[[0, 1, 5, 6]].tolist() == [1, 12, axis, 0 if axis == 0 else (8, 6, 4)[axis-1]//2]
                assert meta[2:5].view("<f8").tolist() == [0.012, 0.001, 0.001]
            peak, field_bytes, gpu = map(int, (frame / "budget.txt").read_text().split())
            assert peak <= 2160 and gpu == backend
            assert field_bytes == expected_coordinates[..., 0].size*components*8
            xml = ET.parse(frame / "data.xdmf")
            assert xml.find(".//Topology").attrib["Dimensions"] == " ".join(map(str, expected_coordinates.shape[:-1]))
            assert float(xml.find(".//Time").attrib["Value"]) == 0.012
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 4*1024**2


@pytest.mark.parametrize("fault,message", [
    ("budget", "host budget cannot hold one basic node"),
    ("index", "slice index"),
    ("nan", "nonfinite basic fields"),
    ("units", "units differ between ranks"),
    ("components", "clock or slice differs between ranks"),
])
def test_invalid_field_output_rejected(probe, fault, message, tmp_path):
    (tmp_path / "0_0").mkdir()
    run = run_probe([MPIEXEC, "--oversubscribe", "--mca", "coll_hcoll_enable", "0", "-n", "2",
                     str(probe), str(tmp_path), "12", "1", fault])
    assert run.returncode != 0
    assert "field output: " + message in run.stdout + run.stderr
    assert not (tmp_path / "0_0/data.xdmf").exists()


@pytest.mark.parametrize("decomposition", [1, 2, 3])
@pytest.mark.parametrize("components", [6, 12])
def test_grouped_planes(probe, decomposition, components, tmp_path):
    for backend in range(2 if GPU else 1):
        (tmp_path / f"{backend}_grouped").mkdir()
    run = run_probe([MPIEXEC, "--oversubscribe", "--mca", "coll_hcoll_enable", "0", "-n", "2",
                     str(probe), str(tmp_path), str(components), str(decomposition), "grouped"])
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.count("OUTPUT_GROUPED_FIELDS_PASS") == 2
    k, j, i = np.indices((5, 7, 9))
    owner = (i, j, k)[decomposition-1] >= (8, 6, 4)[decomposition-1]//2
    encoding = 100*i + 10*j + k + owner/1024
    coordinates = np.stack((i+j/32, j+i/16, k+i*j/64), axis=-1)
    labels = []
    for backend in range(2 if GPU else 1):
        frame = tmp_path / f"{backend}_grouped"
        with h5py.File(frame / "data.h5") as fields:
            for axis, tag in enumerate("ijk", 1):
                index = (8, 6, 4)[axis-1]//2
                label = f"{tag}{index:012d}"
                if backend == 0:
                    labels.append(label)
                selected = [slice(None)]*3
                selected[3-axis] = index
                selected = tuple(selected)
                assert np.array_equal(fields[label]["coordinates"][:], coordinates[selected])
                for component, name in enumerate(NAMES[:components], 1):
                    assert fields[label][name][:].tobytes() == (encoding+component/16)[selected].tobytes()
            assert set(fields) == {"frame_identity", *labels}
        xml = ET.parse(frame / "data.xdmf")
        assert [g.attrib["Name"] for g in xml.findall(".//Grid[@CollectionType='Spatial']/Grid")] == labels
        assert len(list(frame.glob("*.h5"))) == 1


def test_grouped_clock_mismatch_rejected(probe, tmp_path):
    (tmp_path / "0_grouped").mkdir()
    run = run_probe([MPIEXEC, "--oversubscribe", "--mca", "coll_hcoll_enable", "0", "-n", "2",
                     str(probe), str(tmp_path), "12", "1", "grouped_clock"])
    assert run.returncode != 0
    assert "grouped frame clock/layout/unit mismatch" in run.stdout + run.stderr
    assert not (tmp_path / "0_grouped/data.xdmf").exists()


def geometry_probe(probe, path, components, decomposition, mode):
    return run_probe([MPIEXEC, "--oversubscribe", "--mca", "coll_hcoll_enable", "0", "-n", "2",
                      str(probe), str(path), str(components), str(decomposition), mode])


@pytest.mark.parametrize("components", [6, 12])
@pytest.mark.parametrize("decomposition", [1, 2, 3])
def test_bounded_geometry_readonly_verification(probe, tmp_path, components, decomposition):
    for axis in range(4):
        (tmp_path / f"geometry_{axis}").mkdir()
    result = geometry_probe(probe, tmp_path, components, decomposition, "geometry_write")
    assert result.returncode == 0, result.stdout + result.stderr
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.h5")}
    result = geometry_probe(probe, tmp_path, components, decomposition, "geometry_verify")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("OUTPUT_GEOMETRY_PASS") == 2
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*.h5")}
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 4 * 1024**2


@pytest.mark.parametrize("defect", ["coordinate", "fp32", "shape", "metadata", "units", "attribute_array"])
def test_geometry_reuse_rejects_mismatch(probe, tmp_path, defect):
    for axis in range(4):
        (tmp_path / f"geometry_{axis}").mkdir()
    result = geometry_probe(probe, tmp_path, 12, 1, "geometry_write")
    assert result.returncode == 0, result.stdout + result.stderr
    target = tmp_path / "geometry_0/data.h5"
    with h5py.File(target, "r+") as data:
        if defect == "coordinate":
            data["coordinates"][0, 0, 0, 0] += 1
        elif defect in ("fp32", "shape"):
            values = data["coordinates"][:]
            del data["coordinates"]
            data["coordinates"] = values.astype(np.float32) if defect == "fp32" else values[:-1]
        elif defect == "metadata":
            data["metadata"][7] += 1
        elif defect == "units":
            data.attrs.modify("units", b"xx")
        else:
            del data.attrs["center"]
            data.attrs.create("center", np.array([b"Node", b"Node"], dtype="S4"))
    before = target.read_bytes()
    result = geometry_probe(probe, tmp_path, 12, 1, "geometry_verify")
    assert result.returncode != 0
    assert "field output:" in result.stdout + result.stderr
    assert target.read_bytes() == before


def derived_probe(probe, path, mode, *, ranks=2, decomposition=1, components=6, units="si"):
    return run_probe([MPIEXEC, "--oversubscribe", "--mca", "coll_hcoll_enable", "0", "-n", str(ranks),
                      str(probe), str(path), str(components), str(decomposition), "derived_"+mode, units])


def derived_reference(ranks, decomposition):
    k, j, i = np.indices((5, 7, 9))
    owner = np.zeros_like(i)
    if ranks == 2:
        owner = (i, j, k)[decomposition-1] >= (8, 6, 4)[decomposition-1]//2
    encoding = 100*i + 10*j + k + owner/1024
    return encoding, np.stack((i+j/32, j+i/16, k+i*j/64), axis=-1)


def check_derived_group(group, selected, selection, encoding, coordinates, components, units):
    expected_names = NAMES[:components]+[DERIVED_NAMES[n-1] for n in selection]
    assert set(group) == {"metadata", "velocity", "coordinates", *expected_names}
    assert group["coordinates"][:].tobytes() == coordinates[selected].tobytes()
    for c, name in enumerate(NAMES[:components], 1):
        assert group[name][:].tobytes() == (encoding+c/16)[selected].tobytes()
    for n in selection:
        dataset = group[DERIVED_NAMES[n-1]]
        assert dataset.dtype == np.dtype("<f8")
        assert dataset[:].tobytes() == (encoding+n/32)[selected].tobytes()
        quantity_units = "dimensionless" if units == "dimensionless" else ("s^-2" if n == 10 else "s^-1")
        assert dataset.attrs["quantity_units"].decode() == quantity_units
        assert dataset.attrs["coordinate_space"].decode() == "physical"
        assert dataset.attrs["time_dimension_exponent"].decode() == ("-2" if n == 10 else "-1")
        if n == 10:
            assert dataset.attrs["definition"].decode() == "-0.5*tr(A*A); A(i,j)=du_i/dx_j; full strain"
    assert group["velocity"][:].tobytes() == np.stack([(encoding+c/16)[selected] for c in (2, 3, 4)], axis=-1).tobytes()
    return expected_names


@pytest.mark.parametrize("decomposition", [1, 2, 3])
@pytest.mark.parametrize("mode,selection", [("all", list(range(1, 15))), ("gradient", list(range(1, 10))),
    ("curl", [12, 13, 14]), ("q", [10, 11]), ("empty", [])])
def test_derived_field_layout(probe, tmp_path, decomposition, mode, selection):
    for axis in range(4):
        (tmp_path / f"0_{axis}").mkdir()
    result = derived_probe(probe, tmp_path, mode, decomposition=decomposition)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("OUTPUT_DERIVED_LAYOUT_PASS") == 2
    encoding, coordinates = derived_reference(2, decomposition)
    for axis in range(4):
        selected = [slice(None)]*3
        if axis:
            selected[3-axis] = (8, 6, 4)[axis-1]//2
        selected = tuple(selected)
        frame = tmp_path / f"0_{axis}"
        with h5py.File(frame / "data.h5") as data:
            names = check_derived_group(data, selected, selection, encoding, coordinates, 6, "si")
            assert data["metadata"][10] == 6
            if selection:
                assert list(map(int, data.attrs["derived_layout"].decode().split())) == selection+[0]*(14-len(selection))
            else:
                assert "derived_layout" not in data.attrs
        xml = ET.parse(frame / "data.xdmf")
        assert [a.attrib["Name"] for a in xml.findall(".//Attribute")] == names+["velocity"]
        assert all(a.attrib["Center"] == "Node" for a in xml.findall(".//Attribute"))
        assert all(a.find("DataItem").attrib["Precision"] == "8" for a in xml.findall(".//Attribute"))
        peak, size = map(int, (frame / "budget.txt").read_text().split())
        assert peak <= 2160
        assert size == coordinates[selected][..., 0].size*(6+len(selection))*8
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 4*1024**2


@pytest.mark.parametrize("components,units,ranks", [(12, "si", 2), (6, "dimensionless", 1)])
def test_derived_layout_additional_components_units(probe, tmp_path, components, units, ranks):
    for axis in range(4):
        (tmp_path / f"0_{axis}").mkdir()
    result = derived_probe(probe, tmp_path, "all", components=components, units=units, ranks=ranks)
    assert result.returncode == 0, result.stdout + result.stderr
    encoding, coordinates = derived_reference(ranks, 1)
    with h5py.File(tmp_path / "0_0/data.h5") as data:
        check_derived_group(data, (slice(None),)*3, list(range(1, 15)), encoding, coordinates, components, units)
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 4*1024**2


@pytest.mark.parametrize("fault,message", [("invalid", "invalid derived field selection"),
    ("duplicate", "invalid derived field selection"), ("unsorted", "invalid derived field selection"),
    ("rank_mismatch", "derived fields differ between ranks"), ("geometry", "geometry cannot contain derived fields"),
    ("budget", "host budget cannot hold one basic node"), ("nan", "nonfinite basic fields")])
def test_derived_layout_rejection(probe, tmp_path, fault, message):
    (tmp_path / "0_0").mkdir()
    result = derived_probe(probe, tmp_path, fault)
    assert result.returncode != 0
    assert "field output: "+message in result.stdout + result.stderr
    assert not (tmp_path / "0_0/data.xdmf").exists()


@pytest.mark.parametrize("decomposition", [1, 2, 3])
def test_derived_grouped_layout(probe, tmp_path, decomposition):
    (tmp_path / "grouped").mkdir()
    result = derived_probe(probe, tmp_path, "grouped", decomposition=decomposition)
    assert result.returncode == 0, result.stdout + result.stderr
    encoding, coordinates = derived_reference(2, decomposition)
    with h5py.File(tmp_path / "grouped/data.h5") as data:
        assert set(data) == {"frame_identity", "i000000000004", "j000000000003", "k000000000002"}
        for axis, index in enumerate((4, 3, 2), 1):
            selected = [slice(None)]*3; selected[3-axis] = index
            check_derived_group(data[f'{"ijk"[axis-1]}{index:012d}'], tuple(selected), list(range(1, 15)),
                                encoding, coordinates, 6, "si")
    xml = ET.parse(tmp_path / "grouped/data.xdmf")
    for grid in xml.findall(".//Grid[@CollectionType='Spatial']/Grid"):
        assert [a.attrib["Name"] for a in grid.findall("Attribute")] == NAMES[:6]+DERIVED_NAMES+["velocity"]


def test_derived_grouped_mismatch_rejected(probe, tmp_path):
    (tmp_path / "grouped").mkdir()
    result = derived_probe(probe, tmp_path, "group_mismatch")
    assert result.returncode != 0
    assert "field output: geometry attribute length" in result.stdout + result.stderr
    assert not (tmp_path / "grouped/data.xdmf").exists()
    with h5py.File(tmp_path / "grouped/data.h5") as data:
        assert "i000000000004" in data and "j000000000003" not in data


@pytest.mark.parametrize("product", ["volume", "planes"])
def test_derived_time_series_xml(probe, tmp_path, product):
    (tmp_path / "series.frames").write_text("ASTR_FRAME_SERIES_1\n0 0\n2 0.002\n12 0.012\n")
    result = derived_probe(probe, tmp_path, "series_"+product)
    assert result.returncode == 0, result.stdout + result.stderr
    xml = ET.parse(tmp_path / "series.xdmf")
    for grid in xml.findall(".//Grid[@GridType='Uniform']"):
        assert [a.attrib["Name"] for a in grid.findall("Attribute")] == NAMES[:6]+DERIVED_NAMES+["velocity"]


@pytest.mark.parametrize("product", ["volume", "planes"])
def test_derived_writer_native_index_and_paraview(probe, tmp_path, product, record_property):
    from test_output_series_repair import repair, reseal, seal_file_records, snapshot
    grouped = product == "planes"
    if grouped:
        for name in ("grouped", "geometry"):
            (tmp_path / name).mkdir()
    else:
        for axis in range(4):
            (tmp_path / f"0_{axis}").mkdir()
            (tmp_path / f"geometry_{axis}").mkdir()
    result = derived_probe(probe, tmp_path, "shared_grouped" if grouped else "shared")
    assert result.returncode == 0, result.stdout + result.stderr
    raw = tmp_path / ("grouped" if grouped else "0_0")
    native = tmp_path / "native" / ("slices" if grouped else "fields")
    segment = native / "segment00000000"
    frame = segment / "step000000000012"
    resources = native / "resources"
    frame.mkdir(parents=True); resources.mkdir()
    shutil.copyfile(tmp_path / ("geometry" if grouped else "geometry_0") / "data.h5", resources / "data.h5")
    for name in ("data.h5", "data.xdmf"):
        shutil.copyfile(raw / name, frame / name)
    labels = ["i000000000004", "j000000000003", "k000000000002"] if grouped else [""]
    with h5py.File(frame / "data.h5") as data:
        field_bytes = sum((data[label] if label else data)["density"].size*20*8 for label in labels)
    (segment / "SEGMENT").write_text("ASTR_OUTPUT_SEGMENT_1\n0 0\n0 0000000000000000\n")
    (segment / "input.txt").write_text("synthetic\n8,6,4\nt,t,t\nf,f,f,f,f,f,f,f\n")
    (frame / "FRAME").write_text(f"ASTR_DERIVED_FRAME_1\n12 0.012\n6 si\n{field_bytes} 0\n"+
                                  " ".join(map(str, range(1, 15)))+"\n")
    (frame / "RESOURCES").write_text(seal_file_records(resources, ["data.h5"], "ASTR_SHARED_RESOURCES 1"))
    reseal(frame)
    (segment / "series.frames").write_text("ASTR_FRAME_SERIES_1\n")
    (segment / "series.xdmf").write_text("not yet published\n")
    report = repair.repair_segment(segment, publish=True)
    assert report["frames"] == 1 and report["field_array_read_bytes"] == 0
    before = snapshot(native)
    readback = tmp_path / "readback"
    readback.mkdir()
    xml = ET.parse(segment / "series.xdmf")
    for item in xml.findall(".//DataItem"):
        path, dataset = item.text.strip().split(":", 1)
        item.text = str((segment / path).resolve())+":"+dataset
    xml.write(readback / "data.xdmf", encoding="utf-8", xml_declaration=True)
    reference = {"time":np.array(.012), "field_names":np.array(NAMES[:6]+DERIVED_NAMES)}
    if grouped:
        reference["labels"] = np.array(labels)
    with h5py.File(frame / "data.h5") as data, h5py.File(resources / "data.h5") as geometry:
        for label in labels:
            leaf, coordinates = (data[label], geometry[label]) if label else (data, geometry)
            prefix = label+"_" if label else ""
            reference[prefix+"coordinates"] = coordinates["coordinates"][:].reshape(-1, 3)
            for name in NAMES[:6]+DERIVED_NAMES:
                reference[prefix+name] = leaf[name][:].ravel()
            reference[prefix+"velocity"] = leaf["velocity"][:].reshape(-1, 3)
    reference_bytes = sum(v.nbytes for v in reference.values())
    assert reference_bytes <= 2*1024**2
    np.savez(readback / "reader_reference.npz", **reference)
    result = run_probe(["/usr/bin/pvpython", "--no-mpi", str(ROOT / "tests/gpu_validation/run_checkpoint_export_validation.py"),
                        "--native-frame-check", str(readback.resolve())])
    (readback / "paraview.log").write_text(result.stdout+result.stderr)
    assert result.returncode == 0, result.stdout+result.stderr
    assert '"status": "passed"' in result.stdout
    assert snapshot(native) == before
    disk_bytes = sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file())
    assert disk_bytes < 4*1024**2
    record_property("reference_array_bytes", reference_bytes)
    record_property("test_root_bytes", disk_bytes)
    record_property("field_bytes", field_bytes)
