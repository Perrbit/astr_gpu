"""Native product-specific transfer, same-phase fields and statistics isolation."""
import json
import re

import numpy as np
from PIL import Image
import pytest

from run_insitu_gpu_derivatives import configuration as capture_config, check_resources
from run_output_restart_validation import run_case, compare_fields
from test_output_insitu_restart import ROOT, BACKEND, arguments, configuration

FINAL = 'outdat/new/checkpoints/step000000000004'
PROFILES = ('q_surface','streamlines','q_streamlines')


def config(profile, capture=True):
    if capture:
        text = capture_config('gpu',BACKEND,ROOT/'tests/gpu_validation/insitu_diagnostic_capture.py')
    else:
        text = configuration(statistics=False,interval=2).replace('render=.true.,',
            "render=.true.,derivative_backend='gpu',")
    return text.replace("derivative_backend='gpu',", f"derivative_backend='gpu',products='{profile}',")


def transfer_counts(case, ranks, profile, steps):
    nodes=(32//ranks+1)*33*33
    needs_q=profile!='streamlines'
    text=(case/'run.log').read_text()
    records=re.findall(r'ASTR_INSITU_SELECTED rank=(\d+) step=(\d+) products=(\w+) '
        r'field_download_bytes=(\d+) velocity_upload_bytes=(\d+)',text)
    assert len(records)==ranks*len(steps)
    assert {(int(r),int(s)) for r,s,_,_,_ in records}=={(r,s) for r in range(ranks) for s in steps}
    for rank,step,name,download,upload in records:
        assert name==profile and int(download)==nodes*(4+needs_q)*8
        assert int(upload)==nodes*(3 if needs_q else 0)*8
    if not needs_q:
        assert 'ASTR_INSITU_GPU_DERIVATIVE_TIMING' not in text
    copies=re.findall(r'ASTR_INSITU_BRIDGE_TIMING rank=\d+ step=\d+ .*copy_bytes=(\d+)',text)
    assert len(copies)==len(records) and all(int(n)==nodes*(6+needs_q)*8 for n in copies)
    assert 'ASTR_INSITU_TRANSFER_TIMING' not in text


@pytest.fixture(scope='module',params=[1,2])
def captured(request,tmp_path_factory):
    ranks=request.param
    args=arguments(tmp_path_factory.mktemp(f'insitu_products_np{ranks}')); args.statistics=False
    full,_=run_case(args,ROOT,'gpu',ranks,'full',4,grid='32,32,32',insitu_config=config('all'))
    cases={}
    for profile in PROFILES:
        case,_=run_case(args,ROOT,'gpu',ranks,profile,4,grid='32,32,32',insitu_config=config(profile))
        compare_fields(full/FINAL/'state.h5',case/FINAL/'state.h5')
        transfer_counts(case,ranks,profile,[0,2,4])
        check_resources(case,ranks)
        cases[profile]=case
    return ranks,full,cases


def test_selected_fields_equal_full_same_phase(captured):
    ranks,full,cases=captured
    for profile,case in cases.items():
        for rank in range(ranks):
            out=case/'outdat/render'
            assert json.loads((out/f'capture.rank{rank:08d}.json').read_text())['finalized']
            for step in (0,2,4):
                name=f'fields.step{step:08d}.rank{rank:08d}.npz'
                with np.load(full/'outdat/render'/name) as a,np.load(out/name) as b:
                    assert set(b.files)=={'xyz','u','v','w'}|({'Q_rs'} if profile!='streamlines' else set())
                    for field in b.files:
                        if field=='Q_rs':
                            np.testing.assert_allclose(a[field],b[field],rtol=0,atol=2e-10)
                        else:
                            np.testing.assert_array_equal(a[field],b[field])


def test_products_preserve_device_statistics(tmp_path):
    args=arguments(tmp_path); args.statistics=False
    cases=[]
    for profile in ('all','q_streamlines'):
        text=config(profile).replace('statistics=f','statistics=t,statistics_window=0.0005,0.0015')
        text=text.replace('initial_frame=t','initial_frame=f').replace('step_interval=2','step_interval=99')
        case,_=run_case(args,ROOT,'gpu',2,profile,2,grid='32,32,32',insitu_config=text)
        cases.append(case)
    final='outdat/new/checkpoints/step000000000002'
    for name in ('state.h5','statistics.h5'):
        compare_fields(cases[0]/final/name,cases[1]/final/name)
    for rank in range(2):
        name=f'sample.statistics.step00000002.rank{rank:08d}.bin'
        assert (cases[0]/'outdat/render'/name).read_bytes()==(cases[1]/'outdat/render'/name).read_bytes()


@pytest.fixture(scope='module')
def full_render(tmp_path_factory):
    args=arguments(tmp_path_factory.mktemp('insitu_products_full_render32')); args.statistics=False
    case,_=run_case(args,ROOT,'gpu',2,'full',4,grid='32,32,32',insitu_config=config('all',capture=False))
    return case


def compare_geometry(reference,candidate,fields):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOXML import vtkXMLPPolyDataReader
    results=[]
    for path in (reference,candidate):
        reader=vtkXMLPPolyDataReader(); reader.SetFileName(str(path)); reader.Update()
        data=reader.GetOutput()
        assert data.GetNumberOfPoints()>0 and data.GetNumberOfCells()>0
        results.append(data)
    a,b=results
    np.testing.assert_allclose(vtk_to_numpy(a.GetPoints().GetData()),
                               vtk_to_numpy(b.GetPoints().GetData()),rtol=0,atol=2e-10)
    for getter in ('GetVerts','GetLines','GetPolys','GetStrips'):
        for array in ('GetOffsetsArray','GetConnectivityArray'):
            np.testing.assert_array_equal(vtk_to_numpy(getattr(getattr(a,getter)(),array)()),
                                           vtk_to_numpy(getattr(getattr(b,getter)(),array)()))
    for name in fields:
        aa,bb=a.GetPointData().GetArray(name),b.GetPointData().GetArray(name)
        assert aa is not None and bb is not None
        np.testing.assert_allclose(vtk_to_numpy(aa),vtk_to_numpy(bb),rtol=0,atol=2e-10)


@pytest.mark.parametrize('profile',PROFILES)
def test_real_selected_render_and_restart(profile,tmp_path,full_render,record_property):
    args=arguments(tmp_path); args.statistics=False
    text=config(profile,capture=False)
    continuous,_=run_case(args,ROOT,'gpu',2,'continuous',4,grid='32,32,32',
                          checkpoint_interval=1,insitu_config=text)
    transfer_counts(continuous,2,profile,[2,4])
    source=continuous/'outdat/new/checkpoints/step000000000003'
    restored,_=run_case(args,ROOT,'gpu',2,'restored',4,grid='32,32,32',
                        checkpoint_interval=1,insitu_config=text,restore=source)
    compare_fields(continuous/FINAL/'state.h5',restored/FINAL/'state.h5')
    transfer_counts(restored,2,profile,[4])
    expected={'q_surface'} if profile=='q_surface' else {'instantaneous_streamlines','crossing_streamlines'}
    if profile=='q_streamlines': expected.add('q_surface')
    image_errors={}
    for case,steps in ((continuous,[2,4]),(restored,[4])):
        for rank in range(2):
            out=case/'outdat/render'
            receipt=json.loads((out/f'lifecycle_rank{rank}.json').read_text())
            assert receipt==dict(frames=[[s,s*.001] for s in steps],finalized=True)
            for step in steps:
                record=json.loads((out/f'mesh_step{step:08d}_rank{rank}.json').read_text())
                assert record['profile']==profile and set(record['products'])==expected
                assert set(record['available_fields'])=={'u','v','w'}|({'Q_rs'} if profile!='streamlines' else set())
                if rank==0 and 'crossing_streamlines' in expected:
                    assert record['crossing_maxabs']<=2e-10
        for step in steps:
            for product in expected:
                image=case/'outdat/render'/f'{product}.step{step:08d}.jpeg'
                assert image.with_suffix('.eps').is_file()
                with Image.open(image) as pixels:
                    assert pixels.size==(800,600) and np.any(np.asarray(pixels)<245)
                    reference=full_render/'outdat/render'/image.name
                    with Image.open(reference) as baseline:
                        image_errors[image.name]=int(np.max(abs(np.asarray(pixels).astype(int)-np.asarray(baseline).astype(int))))
                compare_geometry(reference.with_suffix('.pvtp'),image.with_suffix('.pvtp'),
                                 ['u','v','w']+([] if profile=='streamlines' else ['Q_rs']))
    for path in (restored/'outdat/render').rglob('*'):
        if path.suffix in ('.jpeg','.vtp'):
            assert path.read_bytes()==(continuous/'outdat/render'/path.relative_to(restored/'outdat/render')).read_bytes()
    record_property('jpeg_max_channel_difference',json.dumps(image_errors))


def test_selected_q_memcheck(tmp_path):
    args=arguments(tmp_path); args.statistics=False
    case,_=run_case(args,ROOT,'gpu',2,'selected_memcheck',4,grid='32,32,32',
                    insitu_config=config('q_streamlines'),memcheck=True)
    transfer_counts(case,2,'q_streamlines',[0,2,4])


def test_selected_products_reject_unvalidated_grid(tmp_path):
    args=arguments(tmp_path); args.statistics=False
    run_case(args,ROOT,'gpu',2,'rejected_grid',2,grid='16,16,16',insitu_config=config('streamlines'),
             reject='GPU in-situ derivative candidate requires 32^3 periodic 643e TGV NP=1/2')


def test_product_switch_requires_explicit_override(tmp_path):
    args=arguments(tmp_path); args.statistics=False
    text=config('q_surface',capture=False)
    initial,_=run_case(args,ROOT,'gpu',2,'seed',3,grid='32,32,32',
                       insitu_config=config('streamlines',capture=False))
    source=initial/'outdat/new/checkpoints/step000000000003'
    run_case(args,ROOT,'gpu',2,'rejected',4,grid='32,32,32',restore=source,
             insitu_config=text,reject='native render configuration differs; select explicit override')
    restored,_=run_case(args,ROOT,'gpu',2,'override',4,grid='32,32,32',restore=source,
                        override=True,insitu_config=text)
    continuous,_=run_case(args,ROOT,'gpu',2,'control',4,grid='32,32,32',insitu_config=text)
    compare_fields(restored/FINAL/'state.h5',continuous/FINAL/'state.h5')
