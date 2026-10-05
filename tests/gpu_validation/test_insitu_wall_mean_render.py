"""Native wall mean images use the saved FP64 moments, including seam copies."""
import json

import numpy as np
import pytest
from PIL import Image
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import vtkXMLPPolyDataReader

from run_output_air5_restart_validation import launch,DT
from run_output_restart_validation import run_case,compare_fields
from run_insitu_gpu_derivatives import check_resources
from test_insitu_air5_walls import arguments,TOPOLOGIES,FINAL
from test_insitu_air5_wall_render import FIELDS,PRODUCTS
from test_insitu_wall_scalar_statistics import config,read_statistics
from test_insitu_channel_statistics import config as channel_config
from test_insitu_channel_walls import arguments as channel_arguments
from test_output_insitu_restart import ROOT
from test_insitu_products import compare_geometry


def settings(channel=False):
    text=channel_config(True) if channel else config(True)
    return text.replace('step_interval=2','step_interval=4,wall_mean_render=t')


def coordinate_key(point,channel):
    result=np.asarray(point).copy()
    periods={0:2*np.pi,2:np.pi} if channel else {2:.002}
    for axis,period in periods.items():
        if abs(result[axis]-period)<1e-14:
            result[axis]=0.
    return tuple(np.round(result,14))


@pytest.fixture(scope='module',params=TOPOLOGIES,ids=('single','x','y','z'))
def reference(request,tmp_path_factory):
    topology=request.param; ranks=int(np.prod(topology))
    args=arguments(tmp_path_factory.mktemp('air5_wall_means'),topology)
    case,size=launch(args,'gpu',ranks,'mean_render',4,interval=1,insitu_config=settings(),
        directory_budget_bytes=256*1024**2)
    check_resources(case,ranks,steps=(4,),capture=False,dt=DT)
    return args,topology,ranks,case,size


def verify(case,ranks,channel=False):
    products=('wall_pressure','wall_shear_x','wall_heat_into_gas') if channel else PRODUCTS
    fields=products if channel else FIELDS
    expected={}
    for rank in range(ranks):
        _,clock,_,xyz,moments,owned=read_statistics(
            case/f'outdat/render/sample.wall_statistics.step00000004.rank{rank:08d}.bin')
        for index in np.ndindex(owned.shape):
            if owned[index]:
                expected[coordinate_key(xyz[index],channel)]=moments[index]
    for name in products:
        path=case/f'outdat/render/mean_{name}.step00000004.pvtp'
        reader=vtkXMLPPolyDataReader(); reader.SetFileName(str(path)); reader.Update()
        data=reader.GetOutput()
        assert reader.GetErrorCode()==0 and data.GetNumberOfCells()==(512 if channel else 256)
        assert data.GetFieldData().GetArray('complete_step').GetValue(0)==4
        points=vtk_to_numpy(data.GetPoints().GetData())
        actual=np.stack([vtk_to_numpy(data.GetPointData().GetArray(kind+field))
            for kind in ('mean_','variance_','rms_') for field in fields],axis=-1)
        duration=vtk_to_numpy(data.GetPointData().GetArray('statistics_duration'))
        np.testing.assert_array_equal(duration,np.full(len(points),clock[3]))
        for name,value in zip(('statistics_window_start','statistics_window_end','statistics_duration'),clock[1:4]):
            assert data.GetFieldData().GetArray(name).GetValue(0)==value
        for point,value in zip(points,actual):
            np.testing.assert_array_equal(value,expected[coordinate_key(point,channel)])
        assert path.with_suffix('.eps').is_file()
        with Image.open(path.with_suffix('.jpeg')) as image:
            pixels=np.asarray(image.convert('RGB'))
            assert pixels.shape==(600,800,3) and np.count_nonzero(np.any(pixels<220,axis=-1))>500
    for rank in range(ranks):
        receipt=json.loads((case/f'outdat/render/mesh_step00000004_rank{rank}.json').read_text())
        assert set(receipt['products'])==set(products)|{'mean_'+name for name in products}


def test_air5_mean_products(reference):
    _,_,ranks,case,size=reference
    verify(case,ranks)
    assert size<=256*1024**2
    assert not list(case.glob('outdat/render/*air5_statistics*'))


def test_air5_mean_exact_restart(reference,tmp_path):
    _,topology,ranks,case,_=reference
    source=case/'outdat/new/checkpoints/step000000000003'
    resumed,_=launch(arguments(tmp_path,topology),'gpu',ranks,'resumed',4,restore=source,interval=1,
        insitu_config=settings(),directory_budget_bytes=256*1024**2)
    check_resources(resumed,ranks,steps=(4,),capture=False,dt=DT)
    for name in ('state.h5','statistics.h5'):
        compare_fields(case/FINAL/name,resumed/FINAL/name)
    fields=tuple(kind+name for kind in ('mean_','variance_','rms_') for name in FIELDS)+('statistics_duration',)
    for product in PRODUCTS:
        base=f'outdat/render/mean_{product}.step00000004'
        compare_geometry(case/(base+'.pvtp'),resumed/(base+'.pvtp'),fields)
        for ext in ('jpeg','eps'):
            assert (case/(base+'.'+ext)).read_bytes()==(resumed/(base+'.'+ext)).read_bytes()


def test_mean_render_isolation_and_memory_safety(tmp_path):
    topology=(1,2,1); args=arguments(tmp_path,topology)
    plain,_=launch(args,'gpu',2,'plain',4,interval=1,insitu_config=config(),directory_budget_bytes=256*1024**2)
    rendered,_=launch(args,'gpu',2,'memcheck_means',4,interval=1,insitu_config=settings(),memcheck=True,
        directory_budget_bytes=256*1024**2)
    for name in ('state.h5','statistics.h5'):
        compare_fields(plain/FINAL/name,rendered/FINAL/name)
    logs=list(rendered.glob('memcheck.*.log'))
    assert len(logs)==2 and all('ERROR SUMMARY: 0 errors' in path.read_text() for path in logs)
    verify(rendered,2)


def test_channel_mean_products(tmp_path):
    args=channel_arguments(tmp_path,'gpu',samples=False); args.statistics=True
    args.directory_budget_bytes=256*1024**2
    case,_=run_case(args,ROOT,'gpu',2,'channel_means',4,checkpoint_interval=1,topology=(1,2,1),
        insitu_config=settings(True))
    check_resources(case,2,steps=(4,),capture=False,statistics=True)
    verify(case,2,channel=True)


def test_repeated_mean_frames(tmp_path):
    args=arguments(tmp_path,(1,2,1))
    case,_=launch(args,'gpu',2,'two_mean_frames',4,interval=1,
        insitu_config=settings().replace('step_interval=4','step_interval=2'),
        directory_budget_bytes=256*1024**2)
    check_resources(case,2,steps=(2,4),capture=False,dt=DT)
    for rank in range(2):
        first=json.loads((case/f'outdat/render/mesh_step00000002_rank{rank}.json').read_text())
        final=json.loads((case/f'outdat/render/mesh_step00000004_rank{rank}.json').read_text())
        assert set(first['products'])==set(final['products'])
        assert first['statistics']['statistics_duration']<final['statistics']['statistics_duration']
    verify(case,2)


def test_joint_volume_wall_products(tmp_path):
    args=arguments(tmp_path,(1,2,1))
    text=settings().replace('wall_mean_render=t',
        'wall_mean_render=t,air5_volume_statistics=t,air5_volume_reduction=t,wall_separation=t')
    case,_=launch(args,'gpu',2,'joint',4,interval=1,insitu_config=text,directory_budget_bytes=256*1024**2)
    check_resources(case,2,steps=(4,),capture=False,dt=DT)
    verify(case,2)
    source=case/'outdat/new/checkpoints/step000000000003'
    resumed,_=launch(args,'gpu',2,'joint_restart',4,restore=source,interval=1,insitu_config=text,
        directory_budget_bytes=256*1024**2)
    for name in ('state.h5','statistics.h5'):
        compare_fields(case/FINAL/name,resumed/FINAL/name)
    for pattern in ('*.air5_statistics.step00000004.*.bin','*.wall_statistics.step00000004.*.bin',
                    '*.air5_volume_rms.step00000004.csv','*.wall_separation.step00000004.csv'):
        paths=list((case/'outdat/render').glob(pattern)); assert paths
        for path in paths:
            assert path.read_bytes()==(resumed/'outdat/render'/path.name).read_bytes()
