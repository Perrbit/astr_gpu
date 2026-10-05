"""Native two-wall Catalyst products, solver isolation and exact continuation."""
import json
import re

import numpy as np
from PIL import Image
import pytest
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import vtkXMLPPolyDataReader

from run_output_restart_validation import run_case, compare_fields
from run_insitu_gpu_derivatives import check_resources
from test_insitu_channel_walls import arguments, read_wall, TOPOLOGIES, FINAL
from test_output_insitu_restart import ROOT, configuration
from test_insitu_products import compare_geometry

FIELDS = ("wall_pressure", "wall_shear_x", "wall_heat_into_gas", "wall_normal_y")


def config():
    return configuration(statistics=False, interval=2).replace("render=.true.,",
        "render=.true.,products='channel_walls',")


@pytest.fixture(scope="module", params=TOPOLOGIES, ids=("single", "x", "y", "z"))
def reference(request, tmp_path_factory):
    topology = request.param
    ranks = int(np.prod(topology))
    args = arguments(tmp_path_factory.mktemp("channel_wall_render"), "gpu")
    plain, _ = run_case(args, ROOT, "gpu", ranks, "plain", 4, topology=topology, checkpoint_interval=1)
    render, size = run_case(args, ROOT, "gpu", ranks, "render", 4, topology=topology,
                           checkpoint_interval=1, insitu_config=config())
    compare_fields(plain / FINAL / "state.h5", render / FINAL / "state.h5")
    check_resources(render, ranks, steps=(2, 4), capture=False)
    return args, topology, ranks, plain, render, size


def test_wall_surface_products(reference, record_property):
    _, topology, ranks, plain, render, size = reference
    lookups = {}
    for step in (2, 4):
        expected = {}
        for rank in range(ranks):
            _, xyz, fields, _ = read_wall(plain / f"outdat/sample.wall.step{step:08d}.rank{rank:08d}.bin")
            for index in np.ndindex(xyz.shape[:3]):
                key = tuple(xyz[index])
                if key in expected:
                    np.testing.assert_array_equal(fields[index], expected[key])
                expected[key] = fields[index]
        lookups[step] = expected
    products = sorted((render / "outdat/render").glob("*.pvtp"))
    assert len(products) == 6
    worst = 0.
    for path in products:
        step = int(path.name.split(".step")[1].split(".")[0])
        reader = vtkXMLPPolyDataReader()
        reader.SetFileName(str(path)); reader.Update()
        data = reader.GetOutput()
        assert reader.GetErrorCode() == 0 and data.GetNumberOfCells() == 512
        points = vtk_to_numpy(data.GetPoints().GetData())
        assert points.dtype == np.float64
        fields = np.stack([vtk_to_numpy(data.GetPointData().GetArray(name)) for name in FIELDS], axis=-1)
        assert np.isfinite(fields).all()
        for xyz, values in zip(points, fields):
            expected = lookups[step][tuple(xyz)]
            error = float(np.max(abs(values - expected)))
            assert error <= 2e-10
            worst = max(worst, error)
        area = 0.
        for index in range(data.GetNumberOfCells()):
            cell = data.GetCell(index)
            ids = [cell.GetPointId(n) for n in range(cell.GetNumberOfPoints())]
            assert len(ids) == 4
            assert np.unique(fields[ids, 3]).size == 1  # Never connect the two walls.
            a, b, c, d = points[ids]
            area += .5 * (np.linalg.norm(np.cross(b-a, c-a)) + np.linalg.norm(np.cross(c-a, d-a)))
        bounds = data.GetBounds()
        expected_area = 2 * (bounds[1]-bounds[0]) * (bounds[5]-bounds[4])
        assert abs(area-expected_area) <= 2e-10
        assert data.GetFieldData().GetArray("complete_step").GetValue(0) == step
        assert data.GetFieldData().GetArray("simulation_time").GetValue(0) == step * .001
        image_path = path.with_suffix(".jpeg")
        assert image_path.with_suffix(".eps").is_file()
        with Image.open(image_path) as image:
            pixels = np.asarray(image.convert("RGB"))
            assert pixels.shape == (600, 800, 3) and np.any(pixels < 245)
    transfers = re.findall(r"ASTR_INSITU_WALL rank=(\d+) step=(\d+) owned_nodes=(\d+) field_download_bytes=(\d+)",
                          (render / "run.log").read_text())
    assert len(transfers) == 2 * ranks
    assert "ASTR_INSITU_TRANSFER_TIMING" not in (render / "run.log").read_text()
    for rank in range(ranks):
        for step in (2, 4):
            receipt = json.loads((render / f"outdat/render/mesh_step{step:08d}_rank{rank}.json").read_text())
            assert receipt["profile"] == "channel_walls"
            assert set(receipt["available_fields"]) == set(FIELDS)
            assert set(receipt["products"]) == set(FIELDS[:-1])
    record_property("topology", topology)
    record_property("geometry_field_maxabs", worst)
    record_property("directory_bytes", size)


def test_exact_wall_render_continuation(reference, tmp_path):
    args, topology, ranks, _, render, _ = reference
    source = render / "outdat/new/checkpoints/step000000000003"
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    args = arguments(tmp_path, "gpu")
    resumed, _ = run_case(args, ROOT, "gpu", ranks, "restart", 4, restore=source,
                         topology=topology, checkpoint_interval=1, insitu_config=config())
    check_resources(resumed, ranks, steps=(4,), capture=False)
    compare_fields(render / FINAL / "state.h5", resumed / FINAL / "state.h5")
    for name in ("control.bin", "insitu_control.bin"):
        assert (render / FINAL / name).read_bytes() == (resumed / FINAL / name).read_bytes()
    for name in FIELDS[:-1]:
        basename = f"{name}.step00000004"
        compare_geometry(render / f"outdat/render/{basename}.pvtp",
                         resumed / f"outdat/render/{basename}.pvtp", FIELDS)
        for extension in ("jpeg", "eps"):
            assert (render / f"outdat/render/{basename}.{extension}").read_bytes() == (
                resumed / f"outdat/render/{basename}.{extension}").read_bytes()
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
