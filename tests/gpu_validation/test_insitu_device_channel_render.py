"""Bounded actual bc41 images through both strict device entries.

The separate channel-fields gate supplies the independent CPU field reference.
These runs do not request host geometry or field oracle downloads.
"""
import json
import re

import numpy as np
from PIL import Image
import pytest

from run_output_restart_validation import run_case, compare_fields
from run_insitu_gpu_derivatives import check_resources
from test_insitu_channel_walls import FINAL, ROOT
from test_insitu_device_channel_fields import current_arguments
from test_insitu_device_products import resident_configuration

BACKEND=ROOT.parent/'astr_dependencies/build/paraview-6.1.1-astr-device-gcc13/lib/catalyst'
PRODUCTS={'wall_pressure':[0.,12.], 'wall_shear_x':[-.02,.02], 'wall_heat_into_gas':[-.2,.2]}


def configuration(pipeline,mode):
    config=resident_configuration(pipeline,mode,statistics=False,interval=2,profile='channel_walls')
    config=re.sub(r"implementation_path='[^']+'",f"implementation_path='{BACKEND}'",config)
    return config


def validate(case,ranks,steps,pipeline,mode):
    log=(case/'run.log').read_text()
    assert 'ASTR_INSITU_DEVICE_WALL_CHECK' not in log
    assert 'field_download_bytes=69360' not in log
    frames=re.findall(r'ASTR_INSITU_DEVICE_WALL_FRAME rank=(\d+) step=(\d+) transport=(\S+) '
        r'face_d2h_bytes=(\d+) face_h2d_bytes=(\d+) face_exchanges=(\d+) '
        r'field_download_bytes=0 geometry_host_bytes=0',log)
    assert len(frames)==ranks*len(steps),log
    if mode=='device-aware':
        assert all(int(row[3])==int(row[4])==0 for row in frames)
    geometry=re.findall(r'ASTR_INSITU_DEVICE_WALL_GEOMETRY rank=(\d+) step=(\d+) product=(\S+) '
        r'points=(\d+) triangles=(\d+) local_area=(\S+) global_area=(\S+) '
        r'invalid_triangles=0 geometry_host_bytes=0',log)
    assert len(geometry)==ranks*len(steps)*3,log
    for step in steps:
        for name in PRODUCTS:
            rows=[row for row in geometry if int(row[1])==step and row[2]==name]
            assert len(rows)==ranks and sum(int(row[4]) for row in rows)==1024
            assert all(abs(float(row[6])-4*np.pi**2)<=2e-10 for row in rows)
            path=case/f'outdat/render/{name}.step{step:08d}.jpeg'
            assert path.with_suffix('.eps').is_file()
            with Image.open(path) as image:
                pixels=np.asarray(image.convert('RGB'))
                assert pixels.shape==(600,800,3)
                assert np.count_nonzero(np.any(pixels<220,axis=2))>10000
        for rank in range(ranks):
            receipt=json.loads((case/f'outdat/render/mesh_step{step:08d}_rank{rank}.json').read_text())
            assert receipt['processing_backend']=='device' and receipt['rendering_pipeline']==pipeline
            assert receipt['geometry_host_bytes']==0 and set(receipt['products'])==set(PRODUCTS)
            for name,colors in PRODUCTS.items():
                item=receipt['products'][name]
                assert item['color_field']==name and item['color_range']==colors
                assert item['local_cells']>0 and item['local_points']>0
    assert not list((case/'outdat/render').glob('*.vtp'))
    assert not list((case/'outdat/render').glob('*.pvtp'))
    check_resources(case,ranks,steps=steps,capture=False)


@pytest.mark.parametrize('topology',[(1,1,1),(2,1,1),(1,2,1),(1,1,2)],ids=['single','x','y','z'])
@pytest.mark.parametrize('mode',['pinned','device-aware'])
@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
def test_real_device_wall_render(tmp_path,topology,mode,pipeline,record_property):
    args=current_arguments(tmp_path,samples=False)
    args.runtime_timeout_seconds=300
    ranks=int(np.prod(topology)); config=configuration(pipeline,mode)
    actual,size=run_case(args,ROOT,'gpu',ranks,'render',4,topology=topology,checkpoint_interval=1,
        insitu_config=config,postprocess_transport=mode)
    validate(actual,ranks,(2,4),pipeline,mode)
    record_property('directory_bytes',size)
    off,_=run_case(args,ROOT,'gpu',ranks,'off',4,topology=topology,checkpoint_interval=1)
    compare_fields(actual/FINAL/'state.h5',off/FINAL/'state.h5')
    source=actual/'outdat/new/checkpoints/step000000000003'
    before={p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
    resumed,_=run_case(args,ROOT,'gpu',ranks,'restart',4,topology=topology,checkpoint_interval=1,
        restore=source,insitu_config=config,postprocess_transport=mode)
    validate(resumed,ranks,(4,),pipeline,mode)
    compare_fields(actual/FINAL/'state.h5',resumed/FINAL/'state.h5')
    for name in ('control.bin','insitu_control.bin'):
        assert (actual/FINAL/name).read_bytes()==(resumed/FINAL/name).read_bytes()
    for name in PRODUCTS:
        for suffix in ('jpeg','eps'):
            filename=f'{name}.step00000004.{suffix}'
            assert (actual/'outdat/render'/filename).read_bytes()==(resumed/'outdat/render'/filename).read_bytes()
    assert before=={p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}


def test_wall_separation_not_silently_host_staged(tmp_path):
    args=current_arguments(tmp_path,samples=False)
    config=configuration('standard-device','pinned').replace('statistics=.false.','statistics=.true.,wall_separation=t')
    run_case(args,ROOT,'gpu',1,'reject_stats',1,insitu_config=config,postprocess_transport='pinned',
        reject='resident wall separation reduction is not yet admitted')


def test_wall_device_compatible_not_silently_host_staged(tmp_path):
    args=current_arguments(tmp_path,samples=False)
    config=configuration('standard-device','pinned').replace("rendering_pipeline='standard-device'",
        "rendering_pipeline='compatible'")
    run_case(args,ROOT,'gpu',1,'reject_compatible',1,insitu_config=config,postprocess_transport='pinned',
        reject='device walls require standard-device or direct-device')


@pytest.mark.parametrize('pipeline,mode,topology',[
    ('standard-device','device-aware',(1,1,2)),('direct-device','pinned',(2,1,1))])
def test_wall_device_render_memcheck(tmp_path,pipeline,mode,topology):
    args=current_arguments(tmp_path,samples=False)
    args.runtime_timeout_seconds=300
    case,_=run_case(args,ROOT,'gpu',2,'memcheck',2,topology=topology,checkpoint_interval=1,
        insitu_config=configuration(pipeline,mode),postprocess_transport=mode,memcheck=True)
    validate(case,2,(2,),pipeline,mode)
    reports=list(case.glob('memcheck.*.log'))
    assert len(reports)==2
    for report in reports:
        assert 'ERROR SUMMARY: 0 errors' in report.read_text(),report
