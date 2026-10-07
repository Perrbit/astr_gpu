"""Noncatalytic AIR5 wall images through both resident rendering entries."""
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from run_output_air5_restart_validation import launch, DT
from run_output_restart_validation import compare_fields
from run_insitu_gpu_derivatives import check_resources
from test_insitu_air5_walls import global_fields, TOPOLOGIES, FINAL, FIELD_SCALES
from test_insitu_device_air5_fields import current_arguments
from test_insitu_device_products import resident_configuration
from test_insitu_device_channel_render import BACKEND
from test_output_insitu_restart import ROOT, MPIEXEC

PRODUCTS={
    'pressure':(6,[0.,60000.],'Pa'),
    'temperature':(4,[1500.,3500.],'K'),
    'vibrational_temperature':(5,[1500.,3500.],'K'),
    'Y_N2':(7,[0.,1.],'1'),'Y_O2':(8,[0.,1.],'1'),'Y_N':(9,[0.,1.],'1'),
    'Y_O':(10,[0.,1.],'1'),'Y_NO':(11,[0.,1.],'1'),
    'wall_shear_x':(12,[-1.,1.],'Pa'),'wall_heat_into_gas':(16,[-60000.,60000.],'W/m^2')}


def test_air5_wall_layout_probe(tmp_path):
    env=dict(os.environ,OMPI_MCA_coll_hcoll_enable='0',OMPI_MCA_coll_ucc_enable='0',
        OMPI_MCA_opal_cuda_support='false',OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,tcp',OMPI_MCA_osc='pt2pt')
    result=subprocess.run([str(MPIEXEC),'-np','1',str(ROOT/'build_insitu_air5_device_render/bin/insitu_device_wall_probe')],
        cwd=tmp_path,env=env,capture_output=True,text=True,timeout=60)
    print(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
    rows=re.findall(r'ASTR_DEVICE_AIR5_WALL_LAYOUT component=(\d+) points=25 triangles=32 mismatches=0 '
        r'control_download_bytes=4 geometry_host_bytes=0',result.stdout)
    assert set(map(int,rows))=={values[0] for values in PRODUCTS.values()} and len(rows)==10
    assert 'ASTR_DEVICE_AIR5_WALL_LAYOUT empty=1 normal_product_rejected=1' in result.stdout
    assert result.stdout.count('ASTR_DEVICE_WALL_COLOR component=')==3


def configuration(pipeline,mode):
    config=resident_configuration(pipeline,mode,statistics=False,interval=2,profile='air5_walls')
    return re.sub(r"implementation_path='[^']+'",f"implementation_path='{BACKEND}'",config)


def validate(case,topology,steps,pipeline,mode,reference=None):
    ranks=int(np.prod(topology)); log=(case/'run.log').read_text()
    assert 'ASTR_INSITU_DEVICE_WALL_CHECK' not in log
    assert 'ASTR_INSITU_AIR5_WALL rank=' not in log
    frames=re.findall(r'ASTR_INSITU_DEVICE_WALL_FRAME rank=(\d+) step=(\d+) transport=(\S+) '
        r'face_d2h_bytes=(\d+) face_h2d_bytes=(\d+) face_exchanges=(\d+) '
        r'field_download_bytes=0 geometry_host_bytes=0',log)
    assert len(frames)==ranks*len(steps),log
    if mode=='device-aware':
        assert all(int(row[3])==int(row[4])==0 for row in frames)
    geometry=re.findall(r'ASTR_INSITU_DEVICE_WALL_GEOMETRY rank=(\d+) step=(\d+) product=(\S+) '
        r'points=(\d+) triangles=(\d+) local_area=(\S+) global_area=(\S+) '
        r'invalid_triangles=0 geometry_host_bytes=0',log)
    assert len(geometry)==ranks*len(steps)*len(PRODUCTS),log
    scalars=re.findall(r'ASTR_INSITU_DEVICE_WALL_SCALAR rank=(\d+) step=(\d+) product=(\S+) '
        r'minimum=(\S+) maximum=(\S+) display_minimum=(\S+) display_maximum=(\S+) '
        r'control_values=2 field_download_bytes=0',log)
    assert len(scalars)==len(geometry)
    for step in steps:
        fields=global_fields(reference,ranks,step) if reference else None
        for name,(component,colors,unit) in PRODUCTS.items():
            rows=[row for row in geometry if int(row[1])==step and row[2]==name]
            assert len(rows)==ranks and sum(int(row[4]) for row in rows)==512
            assert all(abs(float(row[6])-.08*.002)<=2e-17 for row in rows)
            if fields is not None:
                active={row[0] for row in rows if int(row[3])>0}
                values=[row for row in scalars if int(row[1])==step and row[2]==name and row[0] in active]
                actual=[min(float(row[3]) for row in values),max(float(row[4]) for row in values)]
                expected=[fields[...,component].min(),fields[...,component].max()]
                assert np.max(abs(np.array(actual)-expected)/FIELD_SCALES[component])<=2e-10,(name,actual,expected)
            path=case/f'outdat/render/{name}.step{step:08d}.jpeg'
            assert path.with_suffix('.eps').is_file()
            with Image.open(path) as image:
                pixels=np.asarray(image.convert('RGB'))
                assert pixels.shape==(600,800,3) and np.count_nonzero(np.any(pixels<220,axis=2))>500
        for rank in range(ranks):
            receipt=json.loads((case/f'outdat/render/mesh_step{step:08d}_rank{rank}.json').read_text())
            assert receipt['processing_backend']=='device' and receipt['rendering_pipeline']==pipeline
            assert receipt['profile']=='air5_walls'
            assert receipt['step']==step and receipt['time']==step*DT
            assert receipt['geometry_host_bytes']==0 and set(receipt['products'])==set(PRODUCTS)
            empty=topology==(1,2,1) and rank==1
            for name,(_,colors,unit) in PRODUCTS.items():
                item=receipt['products'][name]
                assert item['color_field']==name and item['color_range']==colors and item['field_unit']==unit
                np.testing.assert_allclose(item['physical_bounds'],[0.,.08,0.,0.,0.,.002],atol=2e-17,rtol=0)
                assert (item['local_cells']==0)==empty and (item['local_points']==0)==empty
    assert not list((case/'outdat/render').glob('*.vtp')) and not list((case/'outdat/render').glob('*.pvtp'))
    check_resources(case,ranks,steps=steps,capture=False,dt=DT)


@pytest.fixture(scope='module',params=TOPOLOGIES,ids=('single','x','y','z'))
def reference(request,tmp_path_factory):
    topology=request.param; ranks=int(np.prod(topology))
    args=current_arguments(tmp_path_factory.mktemp('air5_device_render_reference'),topology)
    case,_=launch(args,'gpu',ranks,'reference',4,interval=1,wall_samples=True)
    return topology,case


@pytest.mark.parametrize('mode',('pinned','device-aware'))
@pytest.mark.parametrize('pipeline',('standard-device','direct-device'))
def test_real_air5_device_render(reference,tmp_path,pipeline,mode,record_property):
    topology,plain=reference; ranks=int(np.prod(topology))
    args=current_arguments(tmp_path,topology)
    config=configuration(pipeline,mode)
    actual,size=launch(args,'gpu',ranks,'render',4,interval=1,insitu_config=config,
        postprocess_transport=mode,directory_budget_bytes=256*1024**2)
    validate(actual,topology,(2,4),pipeline,mode,plain)
    compare_fields(actual/FINAL/'state.h5',plain/FINAL/'state.h5')
    source=actual/'outdat/new/checkpoints/step000000000003'
    before={p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
    resumed,_=launch(args,'gpu',ranks,'restart',4,restore=source,interval=1,insitu_config=config,
        postprocess_transport=mode,directory_budget_bytes=256*1024**2)
    validate(resumed,topology,(4,),pipeline,mode,plain)
    compare_fields(actual/FINAL/'state.h5',resumed/FINAL/'state.h5')
    for name in ('control.bin','insitu_control.bin','air5_config.bin','air5_conservation.bin'):
        assert (actual/FINAL/name).read_bytes()==(resumed/FINAL/name).read_bytes()
    for name in PRODUCTS:
        for suffix in ('jpeg','eps'):
            path=f'outdat/render/{name}.step00000004.{suffix}'
            assert (actual/path).read_bytes()==(resumed/path).read_bytes()
    assert before=={p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
    record_property('topology',topology)
    record_property('pipeline',pipeline)
    record_property('transport',mode)
    record_property('directory_bytes',size)


@pytest.mark.parametrize('pipeline,mode,topology',[
    ('standard-device','device-aware',(1,2,1)),('direct-device','pinned',(2,1,1))])
def test_real_air5_device_render_memcheck(tmp_path,pipeline,mode,topology):
    case,_=launch(current_arguments(tmp_path,topology),'gpu',2,'memcheck',2,interval=1,
        insitu_config=configuration(pipeline,mode),postprocess_transport=mode,memcheck=True,
        directory_budget_bytes=256*1024**2)
    validate(case,topology,(2,),pipeline,mode)
    assert len(list(case.glob('memcheck.*.log')))==2


@pytest.mark.parametrize('change,message',[
    (('statistics=.false.','statistics=.true.,air5_volume_statistics=t'),
        'resident AIR5 volume statistics are not yet admitted'),
    (("rendering_pipeline='standard-device'","rendering_pipeline='compatible'"),
        'device walls require standard-device or direct-device')])
def test_air5_device_no_silent_host_fallback(tmp_path,change,message):
    launch(current_arguments(tmp_path,(1,1,1)),'gpu',1,'rejected',1,interval=1,
        insitu_config=configuration('standard-device','pinned').replace(*change),
        postprocess_transport='pinned',reject=message)


def test_air5_wall_partition_image_seams(record_property):
    roots=os.environ.get('ASTR_INSITU_AIR5_WALL_IMAGE_ROOTS')
    if not roots:
        pytest.skip('Select the immutable completed AIR5 wall image matrices')
    from scipy.ndimage import distance_transform_edt
    cases=sorted(p for root in roots.split(os.pathsep)
        for p in Path(root).glob('test_real_air5_device_render_*/gpu_np*_render')
        if not p.parent.is_symlink())
    assert len(cases)==16
    identities=[]
    for case in cases:
        log=(case/'run.log').read_text()
        topology=re.search(r'forced mpi topology=\s*(\d+)\s+(\d+)\s+(\d+)',log).groups()
        transport=re.search(r'ASTR_INSITU_DEVICE_WALL_FRAME .*?transport=(\S+)',log).group(1)
        receipt=json.loads((case/'outdat/render/mesh_step00000004_rank0.json').read_text())
        identities.append((topology,transport,receipt['rendering_pipeline']))
    assert len(set(identities))==16
    maxima,entry_difference={},0
    for name in PRODUCTS:
        worst=0.
        for step in (2,4):
            reference,pairs=None,{}
            for case,(topology,transport,_) in zip(cases,identities):
                with Image.open(case/f'outdat/render/{name}.step{step:08d}.jpeg') as image:
                    pixels=np.asarray(image.convert('RGB')).astype(int)[50:550,50:620]
                # Keep neutral shear surfaces; exclude the legend and white background.
                mask=pixels.min(2)<250
                assert mask.sum()>500,(case,name,step)
                if reference is None:
                    reference=mask
                error=max(float(distance_transform_edt(~reference)[mask].max()),
                    float(distance_transform_edt(~mask)[reference].max()))
                assert error<=1.,(case,name,step,error)
                worst=max(worst,error)
                key=topology,transport
                if key in pairs:
                    difference=int(abs(pixels-pairs.pop(key)).max())
                    assert difference==0,(case,name,step,difference)
                    entry_difference=max(entry_difference,difference)
                else:
                    pairs[key]=pixels
            assert not pairs
        maxima[name]=worst
    record_property('partition_seam_max_pixels',json.dumps(maxima,sort_keys=True))
    record_property('cross_entry_pixel_maxabs',entry_difference)
