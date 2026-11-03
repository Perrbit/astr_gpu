"""Coordinate definitions and display transforms, independent of a CFD run."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_insitu_catalog import HEADER, run
from test_insitu_catalog_products import visual_product

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/insitu"))
from render_coordinates import coordinate_model, transform_camera, fixed_physical_axis,static_coordinate_frame


@pytest.mark.skipif(not os.environ.get('ASTR_COORDINATE_PVPYTHON'),reason='Select the existing ParaView pvpython')
@pytest.mark.parametrize('render',[False,True])
def test_coordinate_plane_layout_with_real_vtk(tmp_path,render):
    command=[os.environ.get('ASTR_INSITU_MPIEXEC','/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpirun'),
        '--mca','pml','ob1','--mca','osc','pt2pt','--mca','coll_hcoll_enable','0','-np','1',
        os.environ['ASTR_COORDINATE_PVPYTHON'],'--force-offscreen-rendering',
        str(ROOT/'tests/gpu_validation/insitu_coordinates_vtk_probe.py'),'--include-wall-planes','--require-plane-layout']
    command+=['--out',str(tmp_path/'images')] if render else ['--metadata-only']
    env=dict(os.environ,VTK_DEFAULT_OPENGL_WINDOW='vtkEGLRenderWindow',PYTHONDONTWRITEBYTECODE='1')
    env.pop('DISPLAY',None)
    result=subprocess.run(command,cwd=tmp_path,env=env,capture_output=True,text=True,timeout=180)
    (tmp_path/'probe.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
    if render:
        models={p.stem:json.loads(p.read_text()) for p in (tmp_path/'images').glob('*.json')}
        assert len(models)==10
        for name,model in models.items():
            assert model['input_points_unchanged']
            if name.startswith(('xz-','yz-')):
                assert not model['vtk_xy_only_2d_mode']
        assert len(list((tmp_path/'images').glob('*.jpeg')))==10


def test_static_coordinate_plane_uses_its_physical_position_without_changing_window():
    window=[0.,1100.,0.,150.,0.,90.]
    original=list(window)
    bounds,fixed=static_coordinate_frame(window,planar=True,
        plane=dict(origin=[0.,0.,45.],normal=[0.,0.,1.]))
    assert bounds==[0.,1100.,0.,150.,45.,45.] and fixed==2
    assert window==original


def test_static_coordinate_frame_does_not_guess_a_local_oblique_basis():
    bounds=[0.,3.,0.,3.,0.,2.]
    frame,fixed=static_coordinate_frame(bounds,planar=True,
        plane=dict(origin=[1.,1.,0.],normal=[2**-.5,2**-.5,0.]))
    assert frame==bounds and fixed is None
    frame,fixed=static_coordinate_frame(bounds,planar=True)
    assert frame==bounds and fixed is None


def test_static_coordinate_axis_mapping_is_not_a_product_name_guess():
    bounds=[0.,3.,1.,1.,0.,2.]
    frame,fixed=static_coordinate_frame(bounds,planar=True)
    assert frame==bounds and fixed==1
    frame,fixed=static_coordinate_frame(bounds,planar=False)
    assert frame==bounds and fixed is None


def test_native_coordinates_preserve_geometry_and_label_units():
    bounds = [10., 30., 0., 2., 5., 5.]
    model = coordinate_model([1, 0], [0., 0., 0., 1., 1., 1., 1.], bounds, "m", fixed_axis=2)
    assert model["physical_bounds"] == model["display_bounds"] == bounds
    assert model["axis_ranges"] == [[10., 30.], [0., 2.], [5., 5.]]
    assert model["axis_visible"] == [True, True, False]
    assert model["axis_titles"] == ["x [m]", "y [m]", "z [m]"]
    assert not model["gridlines"] and model["display_note"] == ""


def test_normalization_changes_ticks_not_geometry():
    bounds = [10., 30., 0., 2., 5., 5.]
    model = coordinate_model([1, 1], [10., 0., 0., 2., 1., 1., 1.], bounds, "m", fixed_axis=2)
    assert model["display_bounds"] == bounds
    assert model["axis_ranges"] == [[0., 10.], [0., 1.], [2.5, 2.5]]
    assert model["axis_titles"][0] == "(x-10)/2 [1]"


def test_display_exaggeration_keeps_physical_ticks_and_camera_identity():
    bounds = [10., 30., 0., 2., 5., 5.]
    model = coordinate_model([0, 0], [0., 0., 0., 1., 1., 4., 1.], bounds, "1", fixed_axis=2)
    assert not model["visible"]
    assert model["display_bounds"] == [10., 30., 0., 8., 5., 5.]
    assert model["axis_ranges"][1] == [0., 2.]
    assert model["display_note"] == "y display x4"
    camera = transform_camera([20., 1., 20.], [20., 1., 5.], [0., 1., 0.], 3., model)
    assert camera == dict(position=[20., 4., 20.], focal_point=[20., 4., 5.], view_up=[0., 1., 0.], parallel_scale=3.)
    assert bounds == [10., 30., 0., 2., 5., 5.]


def test_logical_index_is_not_mislabelled_as_a_fixed_physical_axis():
    sampling = dict(mode="grid_index", axis="j")
    assert fixed_physical_axis(sampling, [0., 0., 0.], [0., 3., 1., 1., 0., 2.]) == 1
    assert fixed_physical_axis(sampling, [0., 0., 0.], [0., 3., .8, 1.2, 0., 2.]) is None
    assert fixed_physical_axis(dict(mode="physical_plane"), [0., 0., 1.], [0., 3., 0., 3., 1., 1.]) == 2
    assert fixed_physical_axis(dict(mode="physical_plane"), [1., 1., 0.], [0., 3., 0., 3., 0., 2.]) is None


@pytest.mark.skipif(not os.environ.get("ASTR_CATALOG_PROBE"), reason="Select the root CMake catalog probe")
@pytest.mark.parametrize("coordinates,valid", [
    ({}, True), ({"visible": False}, True),
    ({"reference_length": 2., "origin": [10., 0., 0.], "display_scale": [1., 4., 1.]}, True),
    ({"origin": [0., 0., 0.]}, False), ({"reference_length": 0.}, False),
    ({"display_scale": [1., -1., 1.]}, False), ({"display_scale": [1., 2.]}, False),
    ({"visible": 1}, False), ({"automatic_yplus": True}, False),
])
def test_native_catalog_coordinate_validation(tmp_path, coordinates, valid):
    item = visual_product("q_surface")
    item["display"]["coordinates"] = coordinates
    healthy = visual_product("q_surface", "healthy")
    result = run(tmp_path, [HEADER, item, healthy])
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"PRODUCT 1 valid={'T' if valid else 'F'}" in result.stdout
    assert "PRODUCT 2 valid=T" in result.stdout


@pytest.mark.skipif(not os.environ.get("ASTR_CATALOG_STATE_PROBE"), reason="Select the root control probe")
def test_coordinates_round_trip_in_existing_control_format(tmp_path):
    item = visual_product("q_surface")
    item["display"]["coordinates"] = dict(visible=False, reference_length=2., origin=[10., 0., 0.], display_scale=[1., 4., 1.])
    path = tmp_path / "coordinates.jsonl"
    path.write_text(json.dumps(HEADER) + "\n" + json.dumps(item) + "\n")
    result = subprocess.run([os.environ["ASTR_CATALOG_STATE_PROBE"], str(path), "visual-checkpoint"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not os.environ.get("ASTR_CATALOG_MPI_PROBE"), reason="Select the root MPI catalog probe")
@pytest.mark.parametrize("ranks", (1, 2, 4))
def test_coordinate_metadata_broadcast(tmp_path, ranks):
    item = visual_product("q_surface")
    item["display"]["coordinates"] = dict(visible=False, reference_length=2., origin=[10., 0., 0.], display_scale=[1., 4., 1.])
    path = tmp_path / "coordinates.jsonl"
    path.write_text(json.dumps(HEADER) + "\n" + json.dumps(item) + "\n")
    result = subprocess.run([os.environ.get("ASTR_INSITU_MPIEXEC", "mpirun"), "--mca", "coll_hcoll_enable", "0",
                             "--oversubscribe", "-np", str(ranks), os.environ["ASTR_CATALOG_MPI_PROBE"],
                             str(path), str(path), "coordinates"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("VISUAL_OK") == ranks
