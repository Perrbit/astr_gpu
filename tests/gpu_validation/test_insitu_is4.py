"""Approved 32^3 slab/2x2x1 native products, measure and exact continuation."""
import json
import re
from types import SimpleNamespace

import h5py
import numpy as np
from PIL import Image
import pytest

from run_output_restart_validation import run_case,compare_fields
from test_output_insitu_restart import ROOT,arguments,configuration,check_render
from test_insitu_products import compare_geometry
from insitu_statistics_reference import read_statistics,compare
from run_insitu_gpu_derivatives import check_resources

FINAL='outdat/new/checkpoints/step000000000004'
TOPOLOGIES=((2,1,1),(1,2,1),(1,1,2),(2,2,1))


def config(backend,render=True,profile='all'):
    return configuration(render=render,statistics=True,interval=2).replace(
        "statistics_window=0.0005,0.0115", "statistics_window=0.0005,0.0035").replace(
        "render=.true.,",f"render=.true., derivative_backend='{backend}',products='{profile}',").replace(
        str(ROOT/'scripts/insitu/tgv_pipeline.py'),str(ROOT/'tests/gpu_validation/insitu_is4_pipeline.py'))


def receipts(case,ranks,steps):
    check_render(case,ranks,steps,statistics=True)
    for rank in range(ranks):
        for step in steps:
            record=json.loads((case/'outdat/render'/f'is4.step{step:08d}.rank{rank:08d}.json').read_text())
            assert record['step']==step and abs(record['time']-step*.001)<1e-15
            assert record['unique_nodes']==record['unique_cells']==32**3
            assert abs(record['physical_volume']-(2*np.pi)**3)<=2e-10
            assert abs(record['constant_integral']-(2*np.pi)**3)<=2e-10
            assert max(record['seam_maxabs'].values())<=2e-10
            if rank==0:
                assert set(record['crossing_maxabs'])==set('xyz')
                assert max(record['crossing_maxabs'].values())<=2e-10


def compare_numerical(reference,candidate):
    with h5py.File(reference) as a,h5py.File(candidate) as b:
        names=[]; a.visit(names.append)
        other=[]; b.visit(other.append)
        assert names==other
        worst=0.
        for name in names:
            if isinstance(a[name],h5py.Group):
                continue
            aa,bb=a[name][...],b[name][...]
            assert aa.shape==bb.shape and aa.dtype==bb.dtype
            if aa.dtype.kind=='f':
                assert np.isfinite(aa).all() and np.isfinite(bb).all()
                error=float(np.max(abs(aa-bb))) if aa.size else 0.
                assert error<=2e-10,(name,error)
                worst=max(worst,error)
            else:
                np.testing.assert_array_equal(aa,bb)
    return worst


def compare_statistics(reference,candidate,ranks):
    errors={}
    for rank in range(ranks):
        name=f'sample.statistics.step00000004.rank{rank:08d}.bin'
        a,ma,va=read_statistics(reference/'outdat/render'/name)
        b,mb,vb=read_statistics(candidate/'outdat/render'/name)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(ma[:3],mb[:3])
        np.testing.assert_allclose(ma[3],mb[3],rtol=0,atol=2e-10)
        np.testing.assert_allclose(ma[4:]**2,mb[4:]**2,rtol=0,atol=2e-10)
        np.testing.assert_array_equal(va[...,0],vb[...,0])
        errors[f'statistics_rank{rank}']=compare(va,vb)
    return errors


@pytest.fixture(scope='module',params=TOPOLOGIES,ids=('x','y','z','xy'))
def reference(request,tmp_path_factory):
    topology=request.param; ranks=int(np.prod(topology))
    args=arguments(tmp_path_factory.mktemp('insitu_is4_'+''.join(map(str,topology))))
    args.statistics=False
    args.directory_budget_bytes=256*1024**2
    cases={}
    for backend in ('cpu','gpu'):
        case,_=run_case(args,ROOT,'gpu',ranks,'diagnostics_'+backend,4,grid='32,32,32',
                        topology=topology,checkpoint_interval=1,insitu_config=config(backend))
        receipts(case,ranks,[2,4]); cases[backend]=case
    for name in ('state.h5','statistics.h5'):
        compare_fields(cases['cpu']/FINAL/name,cases['gpu']/FINAL/name)
    return args,topology,ranks,cases


def test_three_axis_crossing_and_real_products(reference,record_property):
    _,topology,ranks,cases=reference
    fields=('u v w du_dx dv_dx dw_dx du_dy dv_dy dw_dy '
            'du_dz dv_dz dw_dz Q_rs divergence omega_x omega_y omega_z').split()
    differences={}
    for path in sorted((cases['cpu']/'outdat/render').glob('*.pvtp')):
        compare_geometry(path,cases['gpu']/'outdat/render'/path.name,fields)
    for picture in sorted((cases['cpu']/'outdat/render').glob('*.jpeg')):
        with Image.open(picture) as a,Image.open(cases['gpu']/'outdat/render'/picture.name) as b:
            aa,bb=np.asarray(a.convert('RGB')),np.asarray(b.convert('RGB'))
            assert aa.shape==bb.shape
            differences[picture.name]=int(np.max(abs(aa.astype(int)-bb.astype(int))))
    assert len(differences)==12
    if ranks==4:
        assert 'GPU oversubscription correctness mode' in (cases['gpu']/'run.log').read_text()
    record_property('topology',list(topology))
    record_property('jpeg_max_channel_difference',json.dumps(differences))


def test_same_topology_state_statistics_and_render_restart(reference,tmp_path):
    args,topology,ranks,cases=reference
    args=SimpleNamespace(**vars(args)); args.output=tmp_path
    source=cases['gpu']/'outdat/new/checkpoints/step000000000003'
    before={p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
    resumed,_=run_case(args,ROOT,'gpu',ranks,'restart',4,grid='32,32,32',topology=topology,
                      checkpoint_interval=1,restore=source,insitu_config=config('gpu'))
    receipts(resumed,ranks,[4])
    for name in ('state.h5','statistics.h5'):
        compare_fields(cases['gpu']/FINAL/name,resumed/FINAL/name)
    for name in ('control.bin','insitu_control.bin'):
        assert (cases['gpu']/FINAL/name).read_bytes()==(resumed/FINAL/name).read_bytes()
    for path in (resumed/'outdat/render').rglob('*'):
        if path.suffix in ('.jpeg','.vtp') or path.name.startswith('sample.statistics'):
            assert path.read_bytes()==(cases['gpu']/'outdat/render'/path.relative_to(resumed/'outdat/render')).read_bytes()
    assert before=={p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}


def test_cpu_solver_same_phase_reference(reference,tmp_path,record_property):
    args,topology,ranks,cases=reference
    args=SimpleNamespace(**vars(args)); args.output=tmp_path
    cpu,_=run_case(args,ROOT,'cpu',ranks,'solver_reference',4,grid='32,32,32',topology=topology,
                  checkpoint_interval=1,insitu_config=config('cpu',render=False))
    errors={'state.h5':compare_numerical(cpu/FINAL/'state.h5',cases['gpu']/FINAL/'state.h5')}
    errors.update(compare_statistics(cpu,cases['gpu'],ranks))
    record_property('cpu_gpu_maxabs',json.dumps(errors))


def test_unapproved_np4_topology_rejected(tmp_path):
    args=arguments(tmp_path); args.statistics=False
    run_case(args,ROOT,'gpu',4,'reject_xz',1,grid='32,32,32',topology=(2,1,2),
             insitu_config=config('gpu'),reject='GPU in-situ derivative candidate requires')


@pytest.fixture(scope='module')
def selected_reference(tmp_path_factory):
    from test_insitu_products import config as product_config
    args=arguments(tmp_path_factory.mktemp('is4_np4_selected'))
    args.statistics=False; args.directory_budget_bytes=256*1024**2
    text=product_config('all').replace('statistics=f',
        'statistics=t,statistics_window=0.0005,0.0035').replace('initial_frame=t','initial_frame=f')
    full,_=run_case(args,ROOT,'gpu',4,'capture_all',4,grid='32,32,32',topology=(2,2,1),
                    checkpoint_interval=1,insitu_config=text)
    check_resources(full,4,steps=(2,4),statistics=True)
    return args,full


@pytest.mark.parametrize('profile',('q_surface','streamlines','q_streamlines'))
def test_np4_selected_product_and_exact_restart(profile,selected_reference,tmp_path,record_property):
    from test_insitu_products import config as product_config
    args,full=selected_reference
    args=SimpleNamespace(**vars(args)); args.output=tmp_path
    capture=product_config(profile).replace('statistics=f',
        'statistics=t,statistics_window=0.0005,0.0035').replace('initial_frame=t','initial_frame=f')
    captured,_=run_case(args,ROOT,'gpu',4,'captured',4,grid='32,32,32',topology=(2,2,1),
                        insitu_config=capture)
    check_resources(captured,4,steps=(2,4),statistics=True)
    for step in (2,4):
        for rank in range(4):
            name=f'fields.step{step:08d}.rank{rank:08d}.npz'
            with np.load(full/'outdat/render'/name) as a,np.load(captured/'outdat/render'/name) as b:
                assert set(b.files)=={'xyz','u','v','w'}|({'Q_rs'} if profile!='streamlines' else set())
                for field in b.files:
                    if field=='Q_rs':
                        np.testing.assert_allclose(a[field],b[field],rtol=0,atol=2e-10)
                    else:
                        np.testing.assert_array_equal(a[field],b[field])
    records=re.findall(r'ASTR_INSITU_SELECTED rank=(\d+) step=(\d+) products=(\w+) '
        r'field_download_bytes=(\d+) velocity_upload_bytes=(\d+)',(captured/'run.log').read_text())
    assert len(records)==8 and {(int(r),int(s)) for r,s,_,_,_ in records}=={(r,s) for r in range(4) for s in (2,4)}
    nodes=17*17*33; needs_q=profile!='streamlines'
    assert all(name==profile and int(download)==nodes*(4+needs_q)*8 and
               int(upload)==nodes*(3 if needs_q else 0)*8 for _,_,name,download,upload in records)
    text=config('gpu',profile=profile)
    continuous,_=run_case(args,ROOT,'gpu',4,'continuous',4,grid='32,32,32',topology=(2,2,1),
                          checkpoint_interval=1,insitu_config=text)
    source=continuous/'outdat/new/checkpoints/step000000000003'
    resumed,_=run_case(args,ROOT,'gpu',4,'restart',4,grid='32,32,32',topology=(2,2,1),
                       checkpoint_interval=1,restore=source,insitu_config=text)
    for name in ('state.h5','statistics.h5'):
        compare_fields(full/FINAL/name,captured/FINAL/name)
        compare_fields(full/FINAL/name,continuous/FINAL/name)
        compare_fields(continuous/FINAL/name,resumed/FINAL/name)
    for case,steps in ((continuous,(2,4)),(resumed,(4,))):
        check_resources(case,4,steps=steps,capture=False,statistics=True)
    for rank in range(4):
        name=f'sample.statistics.step00000004.rank{rank:08d}.bin'
        assert (full/'outdat/render'/name).read_bytes()==(continuous/'outdat/render'/name).read_bytes()
        assert (continuous/'outdat/render'/name).read_bytes()==(resumed/'outdat/render'/name).read_bytes()
    for path in (resumed/'outdat/render').rglob('*'):
        if path.suffix in ('.jpeg','.vtp'):
            assert path.read_bytes()==(continuous/'outdat/render'/path.relative_to(resumed/'outdat/render')).read_bytes()
    if profile=='q_streamlines':
        sanitized,_=run_case(args,ROOT,'gpu',4,'memcheck',4,grid='32,32,32',topology=(2,2,1),
                            insitu_config=capture,memcheck=True)
        check_resources(sanitized,4,steps=(2,4),statistics=True)
        compare_numerical(captured/FINAL/'state.h5',sanitized/FINAL/'state.h5')
        compare_statistics(captured,sanitized,4)
        record_property('memcheck_case',str(sanitized))


@pytest.mark.parametrize('axis,index',(('x',16),('y',16),('z',4)))
def test_np4_index_planes_and_empty_rank_memcheck(axis,index,selected_reference,tmp_path,record_property):
    from test_insitu_slices import slice_config,check_payload
    args,full=selected_reference
    args=SimpleNamespace(**vars(args)); args.output=tmp_path
    text=slice_config(axis,index).replace('statistics=f',
        'statistics=t,statistics_window=0.0005,0.0035').replace('initial_frame=t','initial_frame=f')
    case,_=run_case(args,ROOT,'gpu',4,'plane',4,grid='32,32,32',topology=(2,2,1),
                    insitu_config=text)
    check_payload(case,4,[2,4],statistics=True)
    for name in ('state.h5','statistics.h5'):
        compare_fields(full/FINAL/name,case/FINAL/name)
    for step in (2,4):
        cell_ids=[]; empty=0
        for rank in range(4):
            name=f'fields.step{step:08d}.rank{rank:08d}.npz'
            with np.load(full/'outdat/render'/name) as a,np.load(case/'outdat/render'/name) as b:
                dim='xyz'.index(axis); plane=2*np.pi*index/32
                xyz=a['xyz']
                owner=xyz[:,dim].min()<=plane+1e-14 and plane<xyz[:,dim].max()-1e-14
                mask=np.isclose(xyz[:,dim],plane,rtol=0,atol=1e-14) if owner else np.zeros(len(xyz),bool)
                np.testing.assert_array_equal(b['xyz'],xyz[mask])
                for field in ('u','v','w'):
                    np.testing.assert_array_equal(b[field],a[field][mask])
                empty+=int(len(b['xyz'])==0)
                if len(b['cells']):
                    assert b['cells'].min()>=0 and b['cells'].max()<len(b['xyz'])
                    corners=b['xyz'][b['cells']]
                    lower=np.rint(corners.min(axis=1)/(2*np.pi/32)).astype(int)
                    tangent=[d for d in range(3) if d!=dim]
                    cell_ids.extend(np.ravel_multi_index(lower[:,tangent].T,(32,32)).tolist())
        assert sorted(cell_ids)==list(range(32**2))
        assert empty==(0 if axis=='z' else 2)
    for rank in range(4):
        name=f'sample.statistics.step00000004.rank{rank:08d}.bin'
        assert (full/'outdat/render'/name).read_bytes()==(case/'outdat/render'/name).read_bytes()
    if axis=='x':
        sanitized,_=run_case(args,ROOT,'gpu',4,'memcheck',4,grid='32,32,32',topology=(2,2,1),
                            insitu_config=text,memcheck=True)
        check_payload(sanitized,4,[2,4],statistics=True)
        compare_numerical(case/FINAL/'state.h5',sanitized/FINAL/'state.h5')
        compare_statistics(case,sanitized,4)
        for step in (2,4):
            for rank in range(4):
                name=f'fields.step{step:08d}.rank{rank:08d}.npz'
                with np.load(case/'outdat/render'/name) as a,np.load(sanitized/'outdat/render'/name) as b:
                    for field in a.files:
                        np.testing.assert_array_equal(a[field],b[field])
        record_property('memcheck_case',str(sanitized))


def test_np4_real_index_plane_render_and_restart(selected_reference,tmp_path):
    from test_insitu_slices import slice_config,check_payload
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOXML import vtkXMLPPolyDataReader
    args,full=selected_reference
    args=SimpleNamespace(**vars(args)); args.output=tmp_path
    text=slice_config('x',16,capture=False).replace('statistics=.false.',
        'statistics=.true.').replace('statistics_window=0.0005,0.0115','statistics_window=0.0005,0.0035')
    continuous,_=run_case(args,ROOT,'gpu',4,'continuous',4,grid='32,32,32',topology=(2,2,1),
                          checkpoint_interval=1,insitu_config=text)
    resumed,_=run_case(args,ROOT,'gpu',4,'restart',4,grid='32,32,32',topology=(2,2,1),
                       checkpoint_interval=1,insitu_config=text,
                       restore=continuous/'outdat/new/checkpoints/step000000000003')
    for name in ('state.h5','statistics.h5'):
        compare_fields(full/FINAL/name,continuous/FINAL/name)
        compare_fields(continuous/FINAL/name,resumed/FINAL/name)
    check_payload(continuous,4,[2,4],capture=False,statistics=True)
    check_payload(resumed,4,[4],capture=False,statistics=True)
    for step in (2,4):
        picture=continuous/'outdat/render'/f'velocity_slice.step{step:08d}.jpeg'
        with Image.open(picture) as pixels:
            assert pixels.size==(800,600) and np.any(np.asarray(pixels)<245)
        assert picture.with_suffix('.eps').is_file()
        reader=vtkXMLPPolyDataReader(); reader.SetFileName(str(picture.with_suffix('.pvtp'))); reader.Update()
        data=reader.GetOutput(); points=vtk_to_numpy(data.GetPoints().GetData())
        assert data.GetNumberOfCells()==1024
        np.testing.assert_allclose(points[:,0],np.pi,rtol=0,atol=2e-10)
        cell_ids=[]; area=0.
        for n in range(data.GetNumberOfCells()):
            ids=data.GetCell(n).GetPointIds()
            p=points[[ids.GetId(i) for i in range(ids.GetNumberOfIds())]]
            lower=np.rint(p.min(axis=0)/(2*np.pi/32)).astype(int)
            cell_ids.append(int(np.ravel_multi_index(lower[1:],(32,32))))
            for i in range(1,len(p)-1):
                area+=.5*np.linalg.norm(np.cross(p[i]-p[0],p[i+1]-p[0]))
        assert sorted(cell_ids)==list(range(32**2)) and abs(area-(2*np.pi)**2)<=2e-10
    for path in (resumed/'outdat/render').rglob('*'):
        if path.suffix in ('.jpeg','.vtp') or path.name.startswith('sample.statistics'):
            assert path.read_bytes()==(continuous/'outdat/render'/path.relative_to(resumed/'outdat/render')).read_bytes()
