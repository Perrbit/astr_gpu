"""Explicit direction glyphs of instantaneous or mean complete wall shear."""
import json
import os
import re
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from test_insitu_catalog import HEADER,run
from test_insitu_wall_footprints import wall_image_definitions
from test_insitu_sbli_statistics import launch as launch_sbli,WINDOW,TOPOLOGIES,MPIEXEC
from test_insitu_device_products import resident_configuration
from test_runtime_catalog_images import image_segments


def glyph(source='reynolds'):
    item=deepcopy(wall_image_definitions((source,))[0])
    item.update(id=source+'-wall-vectors',field='wall_shear_vector')
    item['extraction']=dict(stride=[8,4],length=.2)
    item['display']['color_range']=[0.,.0007]
    return item


@pytest.mark.skipif(not os.environ.get('ASTR_CATALOG_PROBE'),reason='Select the root catalog probe')
def test_wall_glyph_definition(tmp_path):
    result=run(tmp_path,[HEADER,glyph()])
    assert 'PRODUCT 1 valid=T' in result.stdout,result.stdout+result.stderr


@pytest.mark.skipif(not os.environ.get('ASTR_CATALOG_PROBE'),reason='Select the root catalog probe')
def test_wall_glyph_shipped_example(tmp_path):
    path=Path(__file__).resolve().parents[2]/'scripts/output/input.products.sbli.wall-vectors.example.jsonl'
    result=run(tmp_path,path.read_text().splitlines())
    assert 'PRODUCT 1 valid=T' in result.stdout,result.stdout+result.stderr


@pytest.mark.skipif(not os.environ.get('ASTR_CATALOG_PROBE'),reason='Select the root catalog probe')
@pytest.mark.parametrize('extraction',[{},dict(length=0,stride=[8,4]),dict(length=.2,stride=[0,4]),
    dict(length=.2,stride=[True,4]),dict(length=.2,stride=[8]),dict(length=.2,stride=[8,4],unknown=1)])
def test_invalid_glyph_is_product_local(tmp_path,extraction):
    item=glyph();item['extraction']=extraction
    result=run(tmp_path,[HEADER,item,dict(glyph(),id='good')])
    assert 'PRODUCT 1 valid=F' in result.stdout and 'PRODUCT 2 valid=T id=good' in result.stdout


@pytest.mark.skipif(not os.environ.get('ASTR_CATALOG_STATE_PROBE'),reason='Select the root control probe')
def test_glyph_definition_checkpoint(tmp_path):
    item=glyph();item['schedule']['interval']=100
    path=tmp_path/'glyph.jsonl';path.write_text('\n'.join(json.dumps(x) for x in [HEADER,item])+'\n')
    result=subprocess.run([os.environ['ASTR_CATALOG_STATE_PROBE'],str(path),'visual-checkpoint'],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.skipif(not os.environ.get('ASTR_WALL_GLYPH_PROBE'),reason='Select the root device-geometry probe')
@pytest.mark.parametrize('topology',TOPOLOGIES)
def test_wall_glyph_analytic_device_geometry(tmp_path,topology,record_property):
    env=dict(os.environ,OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,tcp',OMPI_MCA_opal_cuda_support='false',OMPI_MCA_osc='pt2pt')
    ranks=int(np.prod(topology))
    result=subprocess.run([str(MPIEXEC),'--mca','coll_hcoll_enable','0','--oversubscribe','-np',str(ranks),
        os.environ['ASTR_WALL_GLYPH_PROBE'],'wall',','.join(map(str,topology))],cwd=tmp_path,env=env,
        capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stdout+result.stderr
    (tmp_path/'probe.log').write_text(result.stdout+result.stderr)
    errors=re.findall(r'PASS wall glyph rank=(\d+) error=(\S+) unique_global_arrows=19',result.stdout)
    assert len(errors)==ranks and {int(r) for r,_ in errors}==set(range(ranks))
    record_property('max_absolute_error',max(float(e) for _,e in errors))
    assert max(float(e) for _,e in errors)<=2e-10


@pytest.mark.skipif(not os.environ.get('ASTR_WALL_GLYPH_PROBE'),reason='Select the root device-geometry probe')
def test_wall_glyph_geometry_memcheck(tmp_path):
    import shutil
    sanitizer=shutil.which('compute-sanitizer');assert sanitizer
    env=dict(os.environ,OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,tcp',OMPI_MCA_opal_cuda_support='false',
             OMPI_MCA_coll='^hcoll,ucc,cuda',OMPI_MCA_osc='pt2pt')
    result=subprocess.run([str(MPIEXEC),'--mca','coll_hcoll_enable','0','-np','2',sanitizer,
        '--tool','memcheck','--error-exitcode','99','--log-file',str(tmp_path/'memcheck.%p.log'),
        os.environ['ASTR_WALL_GLYPH_PROBE'],'wall','1,2,1'],cwd=tmp_path,env=env,capture_output=True,text=True,timeout=180)
    (tmp_path/'run.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
    logs=list(tmp_path.glob('memcheck.*.log'));assert len(logs)==2
    assert all('ERROR SUMMARY: 0 errors' in p.read_text() for p in logs)


@pytest.mark.skipif(not os.environ.get('ASTR_WALL_FOOTPRINT_RENDER_EXE'),reason='Select the strict Catalyst/device solver')
@pytest.mark.parametrize('topology,pipeline',[(t,'standard-device') for t in TOPOLOGIES]+[((2,1,1),'direct-device')])
def test_wall_glyph_actual_images(tmp_path,topology,pipeline):
    config=resident_configuration(pipeline,profile='catalog_stats').replace('0.0005,0.0115',f'{WINDOW[0]},{WINDOW[1]}')
    config=config.replace("output_directory='outdat/render'","output_directory='outdat'")
    records=[glyph('instantaneous'),glyph('reynolds')]
    case=launch_sbli(tmp_path/'gpu','gpu',topology,steps=2,records=records,insitu_config=config,
        executable=os.environ['ASTR_WALL_FOOTPRINT_RENDER_EXE'])
    status=json.loads((case/'outdat/new/products.status.json').read_text())
    assert all(p['state']=='active' for p in status['products']),status
    products=image_segments(case);assert len(products)==2
    ranks=int(np.prod(topology))
    counts={}
    for segment in products:
        expected=3 if segment.parent.parent.name.startswith('instantaneous') else 2
        images=list(segment.glob('*.jpeg'));assert len(images)==expected
        for path in images:
            with Image.open(path) as image:
                assert image.size==(800,600) and np.any(np.asarray(image.convert('RGB'))<230)
        for path in segment.glob('frame.*.rank*.json'):
            row=json.loads(path.read_text());g=row['wall_glyph']
            assert row['field_d2h_bytes']==row['geometry_d2h_bytes']==row['geometry_host_bytes']==0
            assert g['stride']==[8,4] and g['length']==.2 and g['metadata_d2h_bytes'] in (0,48)
            key=segment.parent.parent.name,row['step']
            counts.setdefault(key,[]).append(g['count'])
    assert len(counts)==5
    assert all(len(parts)==ranks and sum(parts)==36 for parts in counts.values()),counts


@pytest.mark.skipif(not os.environ.get('ASTR_WALL_FOOTPRINT_RENDER_EXE'),reason='Select the strict Catalyst/device solver')
@pytest.mark.parametrize('topology',TOPOLOGIES)
def test_glyph_real_cpu_gpu_mean_and_magnitude(tmp_path,topology,record_property):
    import h5py
    from test_insitu_selected_runtime import configuration
    from test_index_slice_runtime import segments,frames

    data=deepcopy(glyph());data.update(id='mean-components',fields=['wall_shear_x','wall_shear_z'])
    for key in ('field','display','extraction'):data.pop(key)
    data['output']=dict(channel='data',format='hdf5-xdmf')
    render=resident_configuration('standard-device',profile='catalog_stats').replace('0.0005,0.0115',f'{WINDOW[0]},{WINDOW[1]}')
    render=render.replace("output_directory='outdat/render'","output_directory='outdat'")
    cpu=launch_sbli(tmp_path/'cpu','cpu',topology,steps=2,records=[data],insitu_config=configuration(WINDOW),
        executable=os.environ['ASTR_WALL_FOOTPRINT_RENDER_EXE'])
    gpu=launch_sbli(tmp_path/'gpu','gpu',topology,steps=2,records=[data,glyph()],insitu_config=render,
        executable=os.environ['ASTR_WALL_FOOTPRINT_RENDER_EXE'])
    left,right=frames(segments(cpu)[0]),frames(segments(gpu)[0]);assert len(left)==len(right)==2
    worst=0.
    receipts=[json.loads(p.read_text()) for s in image_segments(gpu) for p in s.glob('frame.*.rank*.json')]
    for a,b in zip(left,right,strict=True):
        with h5py.File(a) as x,h5py.File(b) as y:
            for field in ('wall_shear_x','wall_shear_z'):
                u,v=x[field][...],y[field][...]
                assert np.isfinite(u).all() and np.isfinite(v).all()
                worst=max(worst,float(np.max(np.abs(u-v)/np.maximum(1.,np.maximum(np.abs(u),np.abs(v))))))
            expected=np.hypot(x['wall_shear_x'][...][:-1:4,::8],x['wall_shear_z'][...][:-1:4,::8])
            selected=[r for r in receipts if r['step']==int(x['step'][0]) and r['product']['local_points']>0]
            observed=[min(r['product']['actual_range'][0] for r in selected),max(r['product']['actual_range'][1] for r in selected)]
            worst=max(worst,float(np.max(np.abs(np.array(observed)-[expected.min(),expected.max()]))))
    record_property('maximum_scaled_component_or_absolute_magnitude_error',worst)
    assert worst<=2e-10,worst


@pytest.mark.skipif(not os.environ.get('ASTR_WALL_FOOTPRINT_RENDER_EXE'),reason='Select the strict Catalyst/device solver')
def test_wall_glyph_exact_restart_and_authority(tmp_path):
    from run_output_restart_validation import compare_fields
    config=resident_configuration('standard-device',profile='catalog_stats').replace('0.0005,0.0115',f'{WINDOW[0]},{WINDOW[1]}')
    config=config.replace("output_directory='outdat/render'","output_directory='outdat'")
    common=dict(insitu_config=config,executable=os.environ['ASTR_WALL_FOOTPRINT_RENDER_EXE'],checkpoint=True)
    records=[glyph('instantaneous'),glyph('reynolds')]
    full=launch_sbli(tmp_path/'full','gpu',(2,1,1),records=records,**common)
    first=launch_sbli(tmp_path/'prefix','gpu',(2,1,1),steps=5,records=records,**common)
    tail=launch_sbli(tmp_path/'tail','gpu',(2,1,1),records=records,
        restore=first/'outdat/new/checkpoints/step000000000005',**common)
    off=launch_sbli(tmp_path/'off','gpu',(2,1,1),records=[],**common)
    final=Path('outdat/new/checkpoints/step000000000012')
    compare_fields(full/final/'state.h5',off/final/'state.h5')
    for name in ('state.h5','statistics.h5'):
        compare_fields(full/final/name,tail/final/name)
    reference={p.parent.parent.name:p for p in image_segments(full)}
    for segment in image_segments(tail):
        for path in segment.glob('*.jpeg'):
            assert path.read_bytes()==(reference[segment.parent.parent.name]/path.name).read_bytes(),path


@pytest.mark.skipif(not os.environ.get('ASTR_WALL_FOOTPRINT_RENDER_EXE'),reason='Select the strict Catalyst/device solver')
def test_glyph_roi_terminal_node_on_mpi_interface(tmp_path):
    config=resident_configuration('standard-device',profile='catalog_stats').replace('0.0005,0.0115',f'{WINDOW[0]},{WINDOW[1]}')
    config=config.replace("output_directory='outdat/render'","output_directory='outdat'")
    item=glyph();item['sampling']['index_ranges']['i']=[24,32]
    case=launch_sbli(tmp_path/'gpu','gpu',(2,1,1),steps=2,records=[item],insitu_config=config,
        executable=os.environ['ASTR_WALL_FOOTPRINT_RENDER_EXE'])
    totals={}
    for path in (case/'outdat/new/render_products').rglob('frame.*.rank*.json'):
        row=json.loads(path.read_text());totals.setdefault(row['step'],[]).append(row['wall_glyph']['count'])
    assert set(totals)=={1,2}
    assert all(len(v)==2 and sum(v)==8 for v in totals.values()),totals


@pytest.mark.skipif(not os.environ.get('ASTR_WALL_FOOTPRINT_RENDER_EXE'),reason='Select the strict Catalyst/device solver')
def test_wall_glyph_cost_without_oracles(tmp_path):
    config=resident_configuration('standard-device',profile='catalog_stats').replace('0.0005,0.0115',f'{WINDOW[0]},{WINDOW[1]}')
    config=config.replace("output_directory='outdat/render'","output_directory='outdat'")
    common=dict(steps=2,insitu_config=config,executable=os.environ['ASTR_WALL_FOOTPRINT_RENDER_EXE'],monitor=True)
    off=launch_sbli(tmp_path/'off','gpu',(2,1,1),records=[],**common)
    item=glyph();item['extraction']['length']=.6
    on=launch_sbli(tmp_path/'on','gpu',(2,1,1),records=[item],
        resource_baseline=json.loads((off/'resources.sampled.json').read_text()),**common)
    result=dict(scope='one short NP2 x GRCBC window, one mean-wall glyph picture per step; not a stable performance benchmark',cases={})
    for name,case in [('off',off),('on',on)]:
        assert not list((case/'outdat').rglob('*.h5'))
        times={}
        for line in (case/'run.log').read_text().splitlines():
            match=re.fullmatch(r'ASTR_INSITU_STAGE_TIMING completed_step_inclusive (\d+) (\d+)\s+(\S+)',line)
            if match:times[tuple(map(int,match.group(1,2)))]=float(match[3])
        assert set(times)=={(s,r) for s in (1,2) for r in (0,1)}
        result['cases'][name]=dict(first_step_max_rank_seconds=max(times[1,r] for r in (0,1)),
            second_step_max_rank_seconds=max(times[2,r] for r in (0,1)),
            two_steps_max_rank_sum_seconds=max(sum(times[s,r] for s in (1,2)) for r in (0,1)),
            memory=json.loads((case/'resources.sampled.json').read_text()))
    assert all(p['state']=='active' for p in json.loads((on/'outdat/new/products.status.json').read_text())['products'])
    (tmp_path/'cost.json').write_text(json.dumps(result,indent=2)+'\n')
