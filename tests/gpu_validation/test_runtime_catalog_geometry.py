"""Native VTK XML products from the existing frozen GPU extractions."""
import json
import os
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest
from vtkmodules.vtkIOXML import vtkXMLPPolyDataReader
from vtkmodules.util.numpy_support import vtk_to_numpy

from run_output_restart_validation import run_case,compare_fields
from test_runtime_catalog_images import arguments,dynamic_image,source_velocity,sample_velocity,catalog_replay_library
from test_insitu_device_products import resident_configuration
from test_index_slice_runtime import ROOT

EXE=os.environ.get('ASTR_RUNTIME_GEOMETRY_EXE')
pytestmark=pytest.mark.skipif(not EXE,reason='Select the root-built native geometry executable')


def params(path):
    result=arguments(path);result.executable=Path(EXE).resolve();return result


def definition(kind,name=None):
    item=dynamic_image(kind,name or kind)
    item.pop('field');item.pop('display')
    item['fields']=['velocity_x','velocity_y','velocity_z']
    if kind!='streamlines': item['fields'].append('Q_rs')
    item['output']=dict(channel='geometry',format='vtk-xml')
    return item


def segments(case):
    return sorted((case/'outdat/new/geometry_products').glob('*/*/segment*'))


def read(path):
    reader=vtkXMLPPolyDataReader();reader.SetFileName(str(path));reader.Update()
    assert reader.GetErrorCode()==0
    result=reader.GetOutput()
    xyz=vtk_to_numpy(result.GetPoints().GetData()).copy()
    data={result.GetPointData().GetArrayName(i):vtk_to_numpy(result.GetPointData().GetArray(i)).copy()
          for i in range(result.GetPointData().GetNumberOfArrays())}
    cell={result.GetCellData().GetArrayName(i):vtk_to_numpy(result.GetCellData().GetArray(i)).copy()
          for i in range(result.GetCellData().GetNumberOfArrays())}
    return result,xyz,data,cell


def scalar_reference(checkpoint):
    velocity=source_velocity(checkpoint,'instantaneous')
    grad=np.zeros((32,32,32,3,3));h=2*np.pi/32
    for c in range(3):
        values=velocity[:32,:32,:32,c]
        for d,axis in enumerate((2,1,0)):
            grad[...,c,d]=(.75*(np.roll(values,-1,axis)-np.roll(values,1,axis))
                -.15*(np.roll(values,-2,axis)-np.roll(values,2,axis))
                +(np.roll(values,-3,axis)-np.roll(values,3,axis))/60)/h
    q=-.5*np.einsum('...ij,...ji->...',grad,grad)
    return velocity,np.pad(q,((0,1),)*3,mode='wrap')


def sample_scalar(values,xyz):
    return sample_velocity(np.stack([values]*3,axis=-1),xyz)[:,0]


def assert_native(case,kind,name=None):
    segment=next(path for path in segments(case) if path.parents[1].name==(name or kind))
    path=segment/'frame.step000000000002.pvtp';poly,xyz,data,cell=read(path)
    assert xyz.dtype==np.float64 and len(xyz)>0
    assert all(value.dtype==np.float64 and np.isfinite(value).all() for value in data.values())
    reference,q=scalar_reference(case/'outdat/new/checkpoints/step000000000002')
    expected=sample_velocity(reference,xyz)
    for component,field in enumerate(('velocity_x','velocity_y','velocity_z')):
        if field in data: np.testing.assert_allclose(data[field],expected[:,component],rtol=0,atol=2e-10)
    if 'Q_rs' in data: np.testing.assert_allclose(data['Q_rs'],sample_scalar(q,xyz),rtol=0,atol=2e-10)
    if kind=='q_surface': np.testing.assert_allclose(data['Q_rs'],.25,rtol=0,atol=2e-10)
    assert poly.GetFieldData().GetArray('complete_step').GetValue(0)==2
    assert poly.GetFieldData().GetArray('simulation_time').GetValue(0)==.002
    assert poly.GetFieldData().GetAbstractArray('unit_system').GetValue(0)=='dimensionless'
    assert poly.GetFieldData().GetArray('reference_state').GetNumberOfTuples()==12
    assert poly.GetFieldData().GetAbstractArray('product_id').GetValue(0)==(name or kind)
    if kind=='streamlines':
        assert set(cell)=={'SeedIds','IntegrationDirection'}
        assert set(np.unique(cell['SeedIds']))==set(range(16))
        assert set(np.unique(cell['IntegrationDirection']))=={-1,1}
        assert all(value.dtype==np.int64 for value in cell.values())
        assert set(np.unique(data['termination_status']))<=set((1.,3.,10.,11.))
        assert np.min(data['accepted_length'])>=0 and np.max(data['accepted_length'])<=2*np.pi+2e-10
        assert np.min(data['untravelled_arc_length'])>=0
    return poly,xyz,data,cell


@pytest.mark.parametrize('ranks',[1,2])
def test_geometry_only_all_existing_kinds_have_native_fields_and_no_views(tmp_path,ranks):
    args=params(tmp_path);options=resident_configuration('compatible',statistics=False,interval=1,profile='velocity_slice')
    items=[definition(kind) for kind in ('slice','physical_plane','q_surface','streamlines')]
    subset=definition('slice','one-field');subset['fields']=['velocity_x'];items.append(subset)
    off,_=run_case(args,ROOT,'gpu',ranks,'off',2,grid='32,32,32',insitu_config=options,product_catalog_records=[])
    case,_=run_case(args,ROOT,'gpu',ranks,'geometry',2,grid='32,32,32',insitu_config=options,product_catalog_records=items)
    compare_fields(off/'outdat/new/checkpoints/step000000000002/state.h5',case/'outdat/new/checkpoints/step000000000002/state.h5')
    status=json.loads((case/'outdat/new/products.status.json').read_text())
    assert all(row['state']=='active' for row in status['products']),status
    assert len(segments(case))==5 and not list((case/'outdat').rglob('*.jpeg'))
    for kind in ('slice','physical_plane','q_surface','streamlines'): assert_native(case,kind)
    _,_,fields,_=assert_native(case,'slice','one-field');assert set(fields)=={'velocity_x'}
    assert len(list((case/'outdat').rglob('*.pvtp')))==10


def test_installed_geometry_example_uses_installed_consumer(tmp_path):
    prefix=os.environ.get('ASTR_RUNTIME_GEOMETRY_INSTALL')
    if not prefix: pytest.skip('Select the staged OutputTools/InsituTools prefix')
    prefix=Path(prefix)
    script=prefix/'share/astr/insitu/device_render_pipeline.py'
    assert (script.parent/'runtime_geometry.py').is_file()
    example=prefix/'share/astr/output/input.products.tgv.geometry.example.jsonl'
    items=[json.loads(line) for line in example.read_text().splitlines()][1:]
    for item in items:
        item['schedule']['interval']=.002 if item['schedule']['mode']=='time' else 2
    args=params(tmp_path)
    options=resident_configuration('compatible',statistics=False,interval=1,profile='velocity_slice')
    options=options.replace(str(ROOT/'scripts/insitu/device_render_pipeline.py'),str(script))
    case,_=run_case(args,ROOT,'gpu',2,'installed',2,grid='32,32,32',insitu_config=options,
        product_catalog_records=items)
    status=json.loads((case/'outdat/new/products.status.json').read_text())
    assert all(row['state']=='active' for row in status['products']),status
    assert len(list((case/'outdat').rglob('*.pvtp')))==8
    assert not list((case/'outdat').rglob('*.jpeg'))
    for kind,name in zip(('slice','physical_plane','q_surface','streamlines'),
                         ('node-surface','physical-surface','Q-quarter','accepted-lines')):
        assert_native(case,kind,name)


def test_later_geometry_request_does_not_reintegrate_an_image_trajectory(tmp_path):
    args=params(tmp_path);options=resident_configuration('compatible',statistics=False,interval=1,profile='velocity_slice')
    picture=dynamic_image('streamlines','picture')
    geometry=definition('streamlines','archive')
    baseline,_=run_case(args,ROOT,'gpu',2,'image-only',2,grid='32,32,32',insitu_config=options,
        product_catalog_records=[picture])
    case,_=run_case(args,ROOT,'gpu',2,'shared',2,grid='32,32,32',insitu_config=options,
        product_catalog_records=[picture,geometry])
    log=(case/'run.log').read_text()
    assert log.count('integrations=1 clipped_geometries=1 scalar_builds=2')==4
    assert_native(case,'streamlines','archive')
    assert len(list((case/'outdat').rglob('*.jpeg')))==2
    from PIL import Image
    reference=next((baseline/'outdat/new/render_products').glob('picture/*/segment*'))
    actual=next((case/'outdat/new/render_products').glob('picture/*/segment*'))
    for path in actual.glob('*.jpeg'):
        with Image.open(path) as a,Image.open(reference/path.name) as b: np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
def test_resident_pipelines_refuse_geometry_without_disabling_the_picture(tmp_path,pipeline):
    args=params(tmp_path);options=resident_configuration(pipeline,statistics=False,interval=1,profile='velocity_slice')
    item=definition('q_surface','archive');picture=dynamic_image('slice','picture')
    case,_=run_case(args,ROOT,'gpu',2,'resident',2,grid='32,32,32',insitu_config=options,product_catalog_records=[item,picture])
    status={row['id']:row for row in json.loads((case/'outdat/new/products.status.json').read_text())['products']}
    assert status['archive']['state']=='invalid-configuration' and status['picture']['state']=='active'
    assert 'explicit compatible' in json.dumps(status['archive'])
    assert not segments(case) and len(list((case/'outdat').rglob('*.jpeg')))==2


def compare_geometry(left,right):
    a,xa,pa,ca=read(left);b,xb,pb,cb=read(right)
    np.testing.assert_array_equal(xa,xb)
    assert set(pa)==set(pb) and set(ca)==set(cb)
    for fields,other in ((pa,pb),(ca,cb)):
        for name in fields: np.testing.assert_array_equal(fields[name],other[name])
    for name in ('Lines','Polys'):
        first=getattr(a,'Get'+name)();second=getattr(b,'Get'+name)()
        np.testing.assert_array_equal(vtk_to_numpy(first.GetConnectivityArray()),vtk_to_numpy(second.GetConnectivityArray()))
        np.testing.assert_array_equal(vtk_to_numpy(first.GetOffsetsArray()),vtk_to_numpy(second.GetOffsetsArray()))
    for name in ('complete_step','TimeValue','reference_state','definition_version'):
        np.testing.assert_array_equal(vtk_to_numpy(a.GetFieldData().GetArray(name)),vtk_to_numpy(b.GetFieldData().GetArray(name)))


def test_geometry_exact_resume_retains_fields_clocks_and_existing_mean(tmp_path):
    args=params(tmp_path);args.statistics=True
    options=resident_configuration('compatible',statistics=True,interval=1,profile='velocity_slice')
    items=[definition(kind) for kind in ('slice','physical_plane','q_surface','streamlines')]
    for item in items: item['schedule']['interval']=4
    items[1]['schedule']=dict(mode='time',interval=.004)
    mean=definition('streamlines','mean');mean.update(source='favre',statistics_source='velocity-time')
    mean['schedule']['interval']=4;items.append(mean)
    common=dict(grid='32,32,32',insitu_config=options,product_catalog_records=items,checkpoint_interval=5)
    full,_=run_case(args,ROOT,'gpu',2,'full',12,**common)
    first,_=run_case(args,ROOT,'gpu',2,'prefix',5,**common)
    tail,_=run_case(args,ROOT,'gpu',2,'tail',12,**dict(common,product_catalog_records=[]),
        restore=first/'outdat/new/checkpoints/step000000000005')
    final='outdat/new/checkpoints/step000000000012'
    for name in ('state.h5','statistics.h5'): compare_fields(full/final/name,tail/final/name)
    expected={path.parents[1].name:path for path in segments(full)}
    assert len(segments(tail))==5
    for path in segments(tail):
        files=sorted(path.glob('*.pvtp'));wanted=[9] if path.parents[1].name=='mean' else [8,12]
        assert [int(file.name[10:22]) for file in files]==wanted
        for file in files: compare_geometry(expected[path.parents[1].name]/file.name,file)


@pytest.mark.parametrize('kind,rank',[('piece',1),('manifest',0)])
def test_geometry_publication_failure_cleans_own_parts_and_needs_a_new_request(tmp_path,kind,rank,catalog_replay_library):
    args=params(tmp_path);options=resident_configuration('compatible',statistics=False,interval=1,profile='velocity_slice')
    failed=definition('q_surface','failed');failed['schedule']['interval']=4
    picture=dynamic_image('slice','healthy')
    frequent=deepcopy(failed);frequent['schedule']['interval']=1
    retry=deepcopy(frequent);retry['enable_request']=2
    off,_=run_case(args,ROOT,'gpu',2,'off',8,grid='32,32,32',insitu_config=options,product_catalog_records=[])
    case,_=run_case(args,ROOT,'gpu',2,'failed',8,grid='32,32,32',insitu_config=options,
        product_catalog_records=[failed,picture],product_poll_steps=1,
        catalog_replay=(catalog_replay_library,{6:[frequent,picture],8:[retry,picture]}),
        geometry_publication_failure=('failed',4,kind,rank))
    final='outdat/new/checkpoints/step000000000008/state.h5';compare_fields(off/final,case/final)
    paths=[file for path in segments(case) for file in path.glob('*.pvtp')]
    assert sorted(int(path.name[10:22]) for path in paths)==[0,7,8]
    assert not list((case/'outdat/new/geometry_products').rglob('*step000000000004*'))
    assert not list((case/'outdat/new/geometry_products').rglob('*.partial'))
    assert len(list((case/'outdat').rglob('*.jpeg')))==5
    for path in paths: assert read(path)[0].GetNumberOfCells()>0


def test_geometry_hot_field_selection_and_bad_record_are_local(tmp_path,catalog_replay_library):
    args=params(tmp_path);options=resident_configuration('compatible',statistics=False,interval=1,profile='velocity_slice')
    original=definition('slice','changing');original['schedule']['interval']=4
    changed=deepcopy(original);changed['fields']=['velocity_x']
    frequent=deepcopy(changed);frequent['schedule']['interval']=1
    invalid=deepcopy(frequent);invalid['output']['format']='csv'
    healthy=definition('q_surface','healthy')
    case,_=run_case(args,ROOT,'gpu',2,'hot',8,grid='32,32,32',insitu_config=options,
        product_catalog_records=[original,healthy],product_poll_steps=1,
        catalog_replay=(catalog_replay_library,{2:[changed,healthy],3:[frequent,healthy],4:[invalid,healthy],5:[frequent,healthy]}))
    histories={int(path.parent.name.removeprefix('version')):sorted(int(file.name[10:22]) for file in path.glob('*.pvtp'))
        for path in segments(case) if path.parents[1].name=='changing'}
    assert histories=={1:[0],2:[1,2,4,5,6,7,8]}
    for path in segments(case):
        if path.parents[1].name!='changing' or path.parent.name.endswith('1'): continue
        for file in path.glob('*.pvtp'): assert set(read(file)[2])=={'velocity_x'}


def test_geometry_metadata_and_retirement_under_two_rank_memcheck(tmp_path):
    args=params(tmp_path);options=resident_configuration('compatible',statistics=False,interval=1,profile='velocity_slice')
    items=[definition(kind) for kind in ('slice','physical_plane','q_surface','streamlines')]
    case,_=run_case(args,ROOT,'gpu',2,'memcheck',2,grid='32,32,32',insitu_config=options,
        product_catalog_records=items,memcheck=True)
    for kind in ('slice','physical_plane','q_surface','streamlines'): assert_native(case,kind)


def test_existing_curve_geometry_keeps_node_coordinates_and_all_selected_fields(tmp_path):
    args=params(tmp_path);args.scale_timestep=2e-5;args.maximum_cfl=.5
    options=resident_configuration('compatible',statistics=False,interval=1,profile='channel_walls')
    item=definition('slice','curve')
    numerical=deepcopy(item);numerical['id']='reference';numerical['output']=dict(channel='data',format='hdf5-xdmf')
    common=dict(grid='32,32,32',tgv_mapping='y-wavy',insitu_config=options)
    off,_=run_case(args,ROOT,'gpu',2,'off',2,**common,product_catalog_records=[numerical])
    case,_=run_case(args,ROOT,'gpu',2,'curve',2,**common,product_catalog_records=[numerical,item])
    final='outdat/new/checkpoints/step000000000002/state.h5';compare_fields(off/final,case/final)
    assert json.loads((case/'cfl_gate.json').read_text())['maximum']<=.5
    segment=next((case/'outdat/new/slice_products').glob('reference/*/segment*'))
    with h5py.File(segment/'geometry.h5') as grid:
        expected=grid['coordinates'][...].reshape(-1,3)
    index={tuple(point):i for i,point in enumerate(expected)}
    with h5py.File(segment/'step000000000002.h5') as fields:
        reference={name:fields[name][...].ravel() for name in item['fields']}
    path=next((case/'outdat/new/geometry_products').glob('curve/*/segment*/frame.step000000000002.pvtp'))
    _,xyz,values,_=read(path);assert xyz.dtype==np.float64
    selected=np.array([index[tuple(point)] for point in xyz])
    for name in item['fields']:
        np.testing.assert_allclose(values[name],reference[name][selected],rtol=0,atol=2e-10)
