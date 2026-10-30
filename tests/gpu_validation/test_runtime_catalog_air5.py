"""Existing AIR5 wall-plane fields through the dynamic device-image consumer."""
import json
import os
from pathlib import Path
from copy import deepcopy

import h5py
import numpy as np
import pytest

from test_insitu_air5_walls import arguments
from run_output_air5_restart_validation import launch
from run_output_restart_validation import compare_fields
from test_runtime_catalog_images import image_product,image_segments
from test_index_slice_runtime import product,segments,frames
from test_insitu_device_products import resident_configuration

EXE=os.environ.get('ASTR_RUNTIME_AIR5_EXE')
pytestmark=pytest.mark.skipif(not EXE,reason='Select the root AIR5 device-rendering executable')
FIELDS=dict(pressure=27,temperature=28,vibrational_temperature=29,
    mass_fraction_N2=30,mass_fraction_O2=31,mass_fraction_N=32,mass_fraction_O=33,mass_fraction_NO=34)


def definitions():
    items=[]
    for field in FIELDS:
        item=image_product(field,field=field)
        item['sampling']=dict(mode='grid_index',axis='j',index=0,index_ranges=dict(i=[0,16],k=[0,16]))
        item['display'].update(camera=[.04,-.12,.001],focal_point=[.04,0.,.001],view_up=[0.,0.,1.],
            parallel_scale=.04,color_range=[0.,1e6 if field=='pressure' else 15000. if 'temperature' in field else 1.])
        items.append(item)
    data=product('wall-plane-values',fields=list(FIELDS))
    data['sampling']=deepcopy(items[0]['sampling']);data['schedule']=deepcopy(items[0]['schedule'])
    return items+[data]


def verify_case(off,case):
    final='outdat/new/checkpoints/step000000000002/state.h5';compare_fields(off/final,case/final)
    state=json.loads((case/'outdat/new/products.status.json').read_text())
    assert all(row['state']=='active' for row in state['products']),state
    log=(case/'run.log').read_text()
    assert log.count('ASTR_RUNTIME_INPUT ')==4 and log.count(' fields=8')==4
    images={segment.parents[1].name:segment for segment in image_segments(case)}
    assert set(images)==set(FIELDS)
    numerical=frames(segments(case)[0]);assert len(numerical)==2
    for index,step in enumerate((0,2)):
        with h5py.File(numerical[index]) as data:
            for field in FIELDS:
                values=data[field][...];assert np.isfinite(values).all()
                receipts=[json.loads(path.read_text()) for path in images[field].glob(f'frame.step{step:012d}.rank*.json')]
                assert len(receipts)==2
                nonempty=[row for row in receipts if row['product']['local_points']>0]
                assert len(nonempty)==1
                np.testing.assert_allclose(nonempty[0]['product']['actual_range'],[values.min(),values.max()],rtol=0,atol=2e-10)
                assert all(row['geometry_host_bytes']==row['field_d2h_bytes']==0 for row in receipts)
                if step==2:
                    with h5py.File(case/final) as checkpoint:
                        expected=checkpoint[f'q{FIELDS[field]:04d}'][...][:,0,:]
                    np.testing.assert_allclose(values,expected,rtol=0,atol=2e-10)
    assert len(list((case/'outdat').rglob('*.jpeg')))==16
    assert not list((case/'outdat/render').glob('*.jpeg'))


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
def test_air5_wall_plane_temperature_species_and_pressure_keep_native_values(tmp_path,pipeline):
    args=arguments(tmp_path,(1,2,1));args.executable=Path(EXE).resolve()
    options=resident_configuration(pipeline,statistics=False,interval=1,profile='air5_walls')
    off,_=launch(args,'gpu',2,'off',2,interval=2,insitu_config=options,
        product_catalog_records=[],directory_budget_bytes=256*1024**2)
    case,_=launch(args,'gpu',2,'images',2,interval=2,insitu_config=options,
        product_catalog_records=definitions(),directory_budget_bytes=256*1024**2)
    verify_case(off,case)


def test_air5_geometry_has_native_species_and_si_metadata(tmp_path):
    from vtkmodules.vtkIOXML import vtkXMLPPolyDataReader
    from vtkmodules.util.numpy_support import vtk_to_numpy
    args=arguments(tmp_path,(1,2,1));args.executable=Path(EXE).resolve()
    options=resident_configuration('compatible',statistics=False,interval=1,profile='air5_walls')
    item=definitions()[-1];item['id']='native-wall'
    item['output']=dict(channel='geometry',format='vtk-xml')
    case,_=launch(args,'gpu',2,'geometry',2,interval=2,insitu_config=options,
        product_catalog_records=[item],directory_budget_bytes=256*1024**2)
    status=json.loads((case/'outdat/new/products.status.json').read_text())
    assert status['products'][0]['state']=='active',status
    path=next((case/'outdat/new/geometry_products').glob('native-wall/*/segment*/frame.step000000000002.pvtp'))
    reader=vtkXMLPPolyDataReader();reader.SetFileName(str(path));reader.Update()
    assert reader.GetErrorCode()==0
    poly=reader.GetOutput();xyz=vtk_to_numpy(poly.GetPoints().GetData())
    assert xyz.dtype==np.float64 and poly.GetNumberOfPoints()==17*17
    assert poly.GetFieldData().GetAbstractArray('unit_system').GetValue(0)=='si'
    units=json.loads(poly.GetFieldData().GetAbstractArray('field_units').GetValue(0))
    assert units['pressure']=='Pa' and units['temperature']=='K' and units['mass_fraction_N2']=='1'
    with h5py.File(case/'outdat/new/checkpoints/step000000000002/state.h5') as state:
        for field,slot in FIELDS.items():
            values=vtk_to_numpy(poly.GetPointData().GetArray(field))
            assert values.dtype==np.float64 and np.isfinite(values).all()
            np.testing.assert_allclose(values,state[f'q{slot:04d}'][...][:,0,:].ravel(),rtol=0,atol=2e-10)
    assert not list((case/'outdat').rglob('*.jpeg'))


def test_air5_optional_wall_allocation_refusal_preserves_a_healthy_plane(tmp_path):
    from test_runtime_catalog_walls import wall_product,wall_frames
    args=arguments(tmp_path,(1,2,1));args.executable=Path(EXE).resolve()
    options=resident_configuration('standard-device',statistics=False,interval=1,profile='air5_walls')
    wall=wall_product('wall')
    wall['sampling']['bounds']=[0.,.08,0.,.012,0.,.002]
    wall['schedule']['interval']=2
    plane=definitions()[0]
    off,_=launch(args,'gpu',2,'off',2,interval=2,insitu_config=options,
        product_catalog_records=[],directory_budget_bytes=256*1024**2)
    case,_=launch(args,'gpu',2,'source-refusal',2,interval=2,insitu_config=options,
        product_catalog_records=[wall,plane],wall_capture_refusal=(2,1),
        directory_budget_bytes=256*1024**2,
        memcheck=os.environ.get('ASTR_RUNTIME_AIR5_WALL_SOURCE_MEMCHECK')=='1')
    final='outdat/new/checkpoints/step000000000002/state.h5'
    compare_fields(off/final,case/final)
    status={row['id']:row for row in json.loads((case/'outdat/new/products.status.json').read_text())['products']}
    assert status['wall']['state']=='resource-disabled' and status['pressure']['state']=='active',status
    assert len(wall_frames(case,'wall'))==1
    assert len(list((case/'outdat').rglob('*.jpeg')))==2
    assert 'optional wall source allocation' in (case/'run.log').read_text()
