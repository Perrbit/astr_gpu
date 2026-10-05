"""GPU index-plane payload, distributed ownership and native rendering gates."""
import json
import re

import numpy as np
from PIL import Image
import pytest

from run_insitu_gpu_derivatives import check_resources
from run_output_restart_validation import run_case,compare_fields
from test_insitu_products import config,FINAL,full_render
from test_output_insitu_restart import ROOT,arguments


def slice_config(axis='z',index=4,capture=True):
    return config('velocity_slice',capture=capture).replace("products='velocity_slice',",
        f"products='velocity_slice',slice_axis='{axis}',slice_index={index},")


def check_payload(case,ranks,steps,capture=True,statistics=False):
    text=(case/'run.log').read_text()
    records=re.findall(r'ASTR_INSITU_SLICE rank=(\d+) step=(\d+) nodes=(\d+) field_download_bytes=(\d+)',text)
    assert {(int(r),int(s)) for r,s,_,_ in records}=={(r,s) for r in range(ranks) for s in steps}
    assert len(records)==ranks*len(steps)
    assert all(int(b)==int(n)*4*8 for _,_,n,b in records)
    copies=re.findall(r'ASTR_INSITU_BRIDGE_TIMING rank=\d+ step=\d+ .*copy_bytes=(\d+)',text)
    assert sorted(map(int,copies))==sorted(int(n)*6*8 for _,_,n,_ in records)
    for name in ('TRANSFER_TIMING','GPU_DERIVATIVE_TIMING','ASTR_INSITU_SELECTED'):
        assert name not in text
    check_resources(case,ranks,steps=steps,capture=capture,statistics=statistics)


@pytest.fixture(scope='module',params=[(1,'x'),(2,'x'),(2,'y'),(2,'z')])
def reference(request,tmp_path_factory):
    ranks,direction=request.param
    args=arguments(tmp_path_factory.mktemp(f'insitu_slice_np{ranks}_{direction}'),direction)
    args.statistics=False
    full,_=run_case(args,ROOT,'gpu',ranks,'full',4,grid='32,32,32',insitu_config=config('all'))
    return args,ranks,direction,full


def test_slice_same_phase_and_unique_cells(reference):
    args,ranks,direction,full=reference
    planes=[('x',4),('y',4),('z',4)] if ranks==1 else [(direction,0),(direction,4),(direction,16),
        ('xyz'[('xyz'.index(direction)+1)%3],4)]
    for axis,index in planes:
        case,_=run_case(args,ROOT,'gpu',ranks,f'slice_{axis}_{index}',4,grid='32,32,32',
                        insitu_config=slice_config(axis,index))
        compare_fields(full/FINAL/'state.h5',case/FINAL/'state.h5')
        check_payload(case,ranks,[0,2,4])
        for step in (0,2,4):
            cells=0
            for rank in range(ranks):
                name=f'fields.step{step:08d}.rank{rank:08d}.npz'
                with np.load(full/'outdat/render'/name) as a,np.load(case/'outdat/render'/name) as b:
                    assert set(b.files)=={'xyz','u','v','w','cells'}
                    dim='xyz'.index(axis); plane=2*np.pi*index/32
                    xyz=a['xyz']
                    owner=xyz[:,dim].min()<=plane+1e-14 and plane<xyz[:,dim].max()-1e-14
                    mask=np.isclose(xyz[:,dim],plane,rtol=0,atol=1e-14) if owner else np.zeros(len(xyz),bool)
                    np.testing.assert_array_equal(b['xyz'],xyz[mask])
                    for field in ('u','v','w'):
                        np.testing.assert_array_equal(b[field],a[field][mask])
                    assert b['cells'].shape[1]==4
                    if len(b['cells']):
                        assert b['cells'].min()>=0 and b['cells'].max()<len(b['xyz'])
                    cells+=len(b['cells'])
            assert cells==32*32


@pytest.mark.parametrize('direction',['x','z'])
def test_slice_real_render_restart(direction,tmp_path,full_render,record_property):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOXML import vtkXMLPPolyDataReader
    args=arguments(tmp_path,direction); args.statistics=False
    text=slice_config(capture=False)
    if direction=='z':
        text=text.replace("schedule_mode='steps', step_interval=2","schedule_mode='time', time_interval=0.002")
    continuous,_=run_case(args,ROOT,'gpu',2,'continuous',4,grid='32,32,32',checkpoint_interval=1,insitu_config=text)
    resumed,_=run_case(args,ROOT,'gpu',2,'resumed',4,grid='32,32,32',checkpoint_interval=1,
        insitu_config=text,restore=continuous/'outdat/new/checkpoints/step000000000003')
    compare_fields(continuous/FINAL/'state.h5',resumed/FINAL/'state.h5')
    check_payload(continuous,2,[2,4],capture=False); check_payload(resumed,2,[4],capture=False)
    image_errors={}
    for step in (2,4):
        image=continuous/'outdat/render'/f'velocity_slice.step{step:08d}.jpeg'
        assert image.with_suffix('.eps').is_file()
        with Image.open(image) as pixels:
            assert pixels.size==(800,600) and np.any(np.asarray(pixels)<245)
            if direction=='x':
                with Image.open(full_render/'outdat/render'/image.name) as baseline:
                    error=int(np.max(abs(np.asarray(pixels).astype(int)-np.asarray(baseline).astype(int))))
                assert error==0
                image_errors[image.name]=error
        reader=vtkXMLPPolyDataReader(); reader.SetFileName(str(image.with_suffix('.pvtp'))); reader.Update()
        data=reader.GetOutput()
        assert data.GetNumberOfCells()==1024
        points=vtk_to_numpy(data.GetPoints().GetData())
        np.testing.assert_allclose(points[:,2],np.pi/4,rtol=0,atol=2e-10)
        # A complete plane with no overlapping cells has the full box cross-section area.
        area=0.
        for n in range(data.GetNumberOfCells()):
            ids=data.GetCell(n).GetPointIds()
            p=points[[ids.GetId(i) for i in range(ids.GetNumberOfIds())]]
            for i in range(1,len(p)-1):
                area+=.5*np.linalg.norm(np.cross(p[i]-p[0],p[i+1]-p[0]))
        assert abs(area-(2*np.pi)**2)<=2e-10
    for path in (resumed/'outdat/render').rglob('*'):
        if path.suffix in ('.jpeg','.vtp'):
            assert path.read_bytes()==(continuous/'outdat/render'/path.relative_to(resumed/'outdat/render')).read_bytes()
    for rank in range(2):
        receipt=json.loads((continuous/'outdat/render'/f'mesh_step00000004_rank{rank}.json').read_text())
        assert receipt['profile']=='velocity_slice' and set(receipt['products'])=={'velocity_slice'}
        expected=set() if direction=='z' and rank==1 else {'u','v','w'}
        assert set(receipt['available_fields'])==expected
    record_property('same_topology_full_slice_jpeg_difference',json.dumps(image_errors))


def test_slice_memcheck_and_statistics(tmp_path):
    args=arguments(tmp_path,'z'); args.statistics=False
    cases=[]
    for profile in ('all','velocity_slice'):
        text=(config('all') if profile=='all' else slice_config()).replace('statistics=f',
            'statistics=t,statistics_window=0.0005,0.0015').replace('initial_frame=t','initial_frame=f')
        case,_=run_case(args,ROOT,'gpu',2,profile,2,grid='32,32,32',insitu_config=text,
                        memcheck=profile=='velocity_slice')
        cases.append(case)
    for name in ('state.h5','statistics.h5'):
        compare_fields(cases[0]/'outdat/new/checkpoints/step000000000002'/name,
                       cases[1]/'outdat/new/checkpoints/step000000000002'/name)
    for rank in range(2):
        name=f'sample.statistics.step00000002.rank{rank:08d}.bin'
        assert (cases[0]/'outdat/render'/name).read_bytes()==(cases[1]/'outdat/render'/name).read_bytes()
    check_payload(cases[1],2,[2],statistics=True)


def test_slice_identity_and_bounds(tmp_path):
    args=arguments(tmp_path,'z'); args.statistics=False
    initial,_=run_case(args,ROOT,'gpu',2,'seed',3,grid='32,32,32',insitu_config=slice_config(capture=False))
    source=initial/'outdat/new/checkpoints/step000000000003'
    changed=slice_config(index=16,capture=False)
    run_case(args,ROOT,'gpu',2,'reject_plane',4,grid='32,32,32',restore=source,insitu_config=changed,
             reject='native render configuration differs; select explicit override')
    restored,_=run_case(args,ROOT,'gpu',2,'override',4,grid='32,32,32',restore=source,insitu_config=changed,override=True)
    control,_=run_case(args,ROOT,'gpu',2,'control',4,grid='32,32,32',insitu_config=changed)
    compare_fields(restored/FINAL/'state.h5',control/FINAL/'state.h5')
    run_case(args,ROOT,'gpu',2,'invalid_index',2,grid='32,32,32',insitu_config=slice_config(index=32),
             reject='index-plane slice requires a global node index below 32')
