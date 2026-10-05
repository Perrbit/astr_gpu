"""Bounded AIR5 wall images/geometry, including a genuinely empty wall rank."""
import json
import re

import numpy as np
from PIL import Image
import pytest
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import vtkXMLPPolyDataReader

from run_output_air5_restart_validation import launch, DT
from run_output_restart_validation import compare_fields
from run_insitu_gpu_derivatives import check_resources
from test_insitu_air5_walls import arguments, read_wall, TOPOLOGIES, FINAL
from test_output_insitu_restart import configuration
from test_insitu_products import compare_geometry

FIELDS = ('rho','u','v','w','temperature','vibrational_temperature','pressure',
    'Y_N2','Y_O2','Y_N','Y_O','Y_NO','wall_shear_x','wall_heat_tr_into_gas',
    'wall_heat_v_into_gas','wall_heat_species_into_gas','wall_heat_into_gas','wall_normal_y')
PRODUCTS = ('pressure','temperature','vibrational_temperature','Y_N2','Y_O2','Y_N','Y_O','Y_NO',
            'wall_shear_x','wall_heat_into_gas')


def config():
    return configuration(statistics=False, interval=2).replace('render=.true.,',
        "render=.true.,products='air5_walls',")


@pytest.fixture(scope='module', params=TOPOLOGIES, ids=('single','x','y','z'))
def reference(request, tmp_path_factory):
    topology = request.param; ranks = int(np.prod(topology))
    args = arguments(tmp_path_factory.mktemp('air5_wall_render'), topology)
    plain, _ = launch(args,'gpu',ranks,'plain',4,interval=1,wall_samples=True)
    render, size = launch(args,'gpu',ranks,'render',4,interval=1,insitu_config=config(),
                          directory_budget_bytes=256*1024**2)
    compare_fields(plain / FINAL / 'state.h5',render / FINAL / 'state.h5')
    check_resources(render,ranks,steps=(2,4),capture=False,dt=DT)
    return args, topology, ranks, plain, render, size


def test_air5_wall_surface_products(reference, record_property):
    _, topology, ranks, plain, render, size = reference
    lookups = {}
    for step in (2,4):
        expected = {}
        for rank in range(ranks):
            _, xyz, fields, _ = read_wall(plain / f'outdat/sample.air5_wall.step{step:08d}.rank{rank:08d}.bin')
            for index in np.ndindex(xyz.shape[:3]):
                key = tuple(xyz[index])
                if key in expected:
                    np.testing.assert_array_equal(fields[index],expected[key])
                expected[key] = fields[index]
        lookups[step] = expected
    paths = sorted((render / 'outdat/render').glob('*.pvtp'))
    assert len(paths) == 2*len(PRODUCTS)
    worst = 0.
    for path in paths:
        step = int(path.name.split('.step')[1].split('.')[0])
        reader = vtkXMLPPolyDataReader(); reader.SetFileName(str(path)); reader.Update()
        data = reader.GetOutput()
        assert reader.GetErrorCode() == 0 and data.GetNumberOfCells() == 256
        points = vtk_to_numpy(data.GetPoints().GetData())
        assert points.dtype == np.float64
        fields = np.stack([vtk_to_numpy(data.GetPointData().GetArray(name)) for name in FIELDS],axis=-1)
        assert np.isfinite(fields).all()
        for xyz, values in zip(points,fields):
            error = float(np.max(abs(values-lookups[step][tuple(xyz)])))
            assert error <= 2e-10
            worst = max(worst,error)
        area = 0.
        for index in range(data.GetNumberOfCells()):
            cell = data.GetCell(index)
            ids = [cell.GetPointId(n) for n in range(cell.GetNumberOfPoints())]
            assert len(ids) == 4
            a,b,c,d = points[ids]
            area += .5*(np.linalg.norm(np.cross(b-a,c-a))+np.linalg.norm(np.cross(c-a,d-a)))
        np.testing.assert_allclose(area,.08*.002,atol=2e-17,rtol=0)
        np.testing.assert_array_equal(points[:,1],0.)
        units = data.GetFieldData().GetAbstractArray('field_units')
        assert units and units.GetNumberOfValues() == len(FIELDS)
        assert 'wall_heat_into_gas=W/m^2' in [units.GetValue(n) for n in range(units.GetNumberOfValues())]
        assert data.GetFieldData().GetArray('complete_step').GetValue(0) == step
        assert data.GetFieldData().GetArray('simulation_time').GetValue(0) == step*1e-10
        assert path.with_suffix('.eps').is_file()
        with Image.open(path.with_suffix('.jpeg')) as image:
            pixels = np.asarray(image.convert('RGB'))
            assert pixels.shape == (600,800,3) and np.count_nonzero(np.any(pixels<220,axis=-1))>500
    for rank in range(ranks):
        for step in (2,4):
            receipt = json.loads((render / f'outdat/render/mesh_step{step:08d}_rank{rank}.json').read_text())
            assert receipt['profile'] == 'air5_walls' and set(receipt['products']) == set(PRODUCTS)
            if topology == (1,2,1) and rank == 1:
                assert not receipt['available_fields']
            else:
                assert set(receipt['available_fields']) == set(FIELDS)
    if topology == (1,2,1):
        text = (render / 'run.log').read_text()
        assert len(re.findall(r'ASTR_INSITU_AIR5_WALL rank=1 step=\d+ owned_nodes=0 field_download_bytes=0',text)) == 2
    record_property('geometry_field_maxabs',worst)
    record_property('topology',topology)
    record_property('directory_bytes',size)


def test_air5_wall_render_exact_continuation(reference,tmp_path):
    _, topology, ranks, _, render, _ = reference
    source = render / 'outdat/new/checkpoints/step000000000003'
    before = {p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
    resumed, _ = launch(arguments(tmp_path,topology),'gpu',ranks,'restart',4,
                        restore=source,interval=1,insitu_config=config(),directory_budget_bytes=256*1024**2)
    compare_fields(render / FINAL / 'state.h5',resumed / FINAL / 'state.h5')
    for name in ('control.bin','insitu_control.bin'):
        assert (render / FINAL / name).read_bytes() == (resumed / FINAL / name).read_bytes()
    check_resources(resumed,ranks,steps=(4,),capture=False,dt=DT)
    for name in PRODUCTS:
        base = f'outdat/render/{name}.step00000004'
        compare_geometry(render / (base+'.pvtp'),resumed / (base+'.pvtp'),FIELDS)
        for extension in ('jpeg','eps'):
            assert (render / (base+'.'+extension)).read_bytes() == (resumed / (base+'.'+extension)).read_bytes()
    assert before == {p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}


def test_air5_wall_pack_memory_safety(tmp_path):
    case, _ = launch(arguments(tmp_path,(1,2,1)),'gpu',2,'memcheck',2,
                     interval=1,wall_samples=True,memcheck=True)
    assert len(list(case.glob('memcheck.*.log'))) == 2


@pytest.mark.parametrize('backend,change,message',[
    ('cpu',None,'native EGL rendering requires GPU solver binding'),
    ('gpu',("products='air5_walls'","products='air5_walls',derivative_backend='gpu'"),
        'wall products require CPU wall diagnostics'),
])
def test_air5_wall_unsupported_config_rejected(tmp_path,backend,change,message):
    text = config()
    if change:
        text = text.replace(*change)
    launch(arguments(tmp_path,(1,1,1)),backend,1,'rejected',1,interval=1,
           insitu_config=text,reject=message)
