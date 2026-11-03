"""TB2.3 rectangular-wall interpolation and strict arc-length RK45."""
import os
import shutil
import subprocess
from pathlib import Path
from copy import deepcopy
import json

import pytest

from test_insitu_wall_vectors import glyph
from test_insitu_catalog import HEADER,run
from test_runtime_catalog_images import catalog_replay_library


def shear_lines(source='reynolds'):
    item=deepcopy(glyph(source))
    item.update(id=source+'-wall-lines',field='wall_shear_lines')
    item['extraction']=dict(seeds=[[8,4],[24,8],[40,12]],integrator='rk45',direction='both',
        initial_step=.01,minimum_step=.001,maximum_step=.05,maximum_error=1e-8,
        maximum_length=2.,maximum_steps=100000)
    return item


@pytest.mark.parametrize('source',['instantaneous','reynolds'])
def test_wall_trace_definition(tmp_path,source):
    if not os.environ.get('ASTR_CATALOG_PROBE'):
        pytest.skip('Select the root catalog probe')
    result=run(tmp_path,[HEADER,shear_lines(source)])
    assert 'PRODUCT 1 valid=T' in result.stdout,result.stdout+result.stderr


def test_wall_trace_shipped_example(tmp_path):
    if not os.environ.get('ASTR_CATALOG_PROBE'):
        pytest.skip('Select the root catalog probe')
    path=Path(__file__).resolve().parents[2]/'scripts/output/input.products.sbli.wall-shear-lines.example.jsonl'
    result=run(tmp_path,path.read_text().splitlines())
    assert 'PRODUCT 1 valid=T' in result.stdout,result.stdout+result.stderr


@pytest.mark.parametrize('change',[dict(seeds=[]),dict(seeds=[[1]]),dict(seeds=[[True,4]]),
    dict(seeds=[[-1,4]]),dict(seeds=[[10,1000]]),dict(seeds=[[8,4,0]]),
    dict(minimum_step=.1),dict(maximum_error=0),dict(integrator='rk4'),dict(unknown=1)])
def test_wall_trace_bad_definition_is_product_local(tmp_path,change):
    if not os.environ.get('ASTR_CATALOG_PROBE'):
        pytest.skip('Select the root catalog probe')
    item=shear_lines();item['extraction'].update(change)
    result=run(tmp_path,[HEADER,item,dict(shear_lines(),id='good')])
    assert 'PRODUCT 1 valid=F' in result.stdout and 'PRODUCT 2 valid=T id=good' in result.stdout


def test_wall_trace_definition_checkpoint(tmp_path):
    executable=os.environ.get('ASTR_CATALOG_STATE_PROBE')
    if not executable:
        pytest.skip('Select the root catalog control probe')
    path=tmp_path/'products.jsonl'
    item=shear_lines();item['schedule']['interval']=100
    path.write_text('\n'.join(json.dumps(row) for row in [HEADER,item])+'\n')
    result=subprocess.run([executable,str(path),'visual-checkpoint'],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize('ranks',[1,2,4])
def test_wall_trace_seed_broadcast_update_and_restore(tmp_path,ranks):
    executable=os.environ.get('ASTR_CATALOG_MPI_PROBE')
    prefix=os.environ.get('ASTR_INSITU_DEVICE_MPI_PREFIX')
    if not executable or not prefix:
        pytest.skip('Select the root collective catalog probe and matching MPI')
    formal=tmp_path/'products.jsonl';candidate=tmp_path/'candidate.jsonl'
    item=shear_lines()
    formal.write_text('\n'.join(json.dumps(row) for row in [HEADER,item])+'\n')
    changed=deepcopy(item);changed['extraction']['seeds'][1][0]=25
    candidate.write_text('\n'.join(json.dumps(row) for row in [HEADER,changed])+'\n')
    result=subprocess.run([str(Path(prefix)/'bin/mpirun'),'--prefix',prefix,'--oversubscribe',
        '--mca','pml','ob1','--mca','btl','self,tcp','--mca','coll_hcoll_enable','0',
        '-np',str(ranks),executable,str(formal),str(candidate),'wall-seeds'],cwd=tmp_path,
        capture_output=True,text=True)
    (tmp_path/'catalog.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
    assert result.stdout.count('WALL_SEEDS_OK')==ranks


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_wall_trace_mathematics(tmp_path,backend):
    executable=os.environ.get(f'ASTR_WALL_TRACE_{backend.upper()}_PROBE')
    if not executable:
        pytest.skip('Select the root wall-trace probe')
    result=subprocess.run([executable],cwd=tmp_path,capture_output=True,text=True)
    (tmp_path/'probe.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'PASS wall trace cases=18' in result.stdout
    assert 'endpoint_zero_retained=1' in result.stdout


def test_wall_trace_mathematics_memcheck(tmp_path):
    executable=os.environ.get('ASTR_WALL_TRACE_GPU_PROBE')
    if not executable:
        pytest.skip('Select the root CUDA wall-trace probe')
    sanitizer=shutil.which('compute-sanitizer')
    assert sanitizer
    log=tmp_path/'memcheck.log'
    result=subprocess.run([sanitizer,'--tool','memcheck','--error-exitcode','99',
        '--log-file',str(log),executable],cwd=tmp_path,capture_output=True,text=True)
    (tmp_path/'probe.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr+log.read_text()
    assert 'ERROR SUMMARY: 0 errors' in log.read_text()


TOPOLOGIES=[(1,1,1),(2,1,1),(1,2,1),(1,1,2),(2,2,1),(2,1,2),(1,2,2)]


def worklet_command(topology):
    executable=os.environ.get('ASTR_WALL_TRACE_WORKLET_PROBE')
    prefix=os.environ.get('ASTR_INSITU_DEVICE_MPI_PREFIX')
    if not executable or not prefix:
        pytest.skip('Select the root wall worklet and matching MPI')
    ranks=topology[0]*topology[1]*topology[2]
    env=dict(os.environ,OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_opal_cuda_support='false',OMPI_MCA_coll='^cuda,hcoll,ucc')
    command=[str(Path(prefix)/'bin/mpirun'),'--prefix',prefix,'--oversubscribe',
        '-np',str(ranks),executable,','.join(map(str,topology))]
    return command,env,ranks


@pytest.mark.parametrize('topology',TOPOLOGIES)
def test_wall_trace_distributed_worklet(tmp_path,topology):
    command,env,ranks=worklet_command(topology)
    result=subprocess.run(command,cwd=tmp_path,env=env,
        capture_output=True,text=True)
    (tmp_path/'worklet.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
    assert f'PASS wall worklet cases=18 ranks={ranks}' in result.stdout
    assert 'source_host_mirror=0 oracle_geometry_readback=1' in result.stdout


def test_wall_trace_worklet_memcheck(tmp_path):
    command,env,ranks=worklet_command((2,2,1))
    sanitizer=shutil.which('compute-sanitizer')
    assert sanitizer
    command[-2:-2]=[sanitizer,'--tool','memcheck','--error-exitcode','99',
        '--log-file',str(tmp_path/'memcheck.%p.log')]
    result=subprocess.run(command,cwd=tmp_path,env=env,capture_output=True,text=True)
    (tmp_path/'worklet.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
    logs=list(tmp_path.glob('memcheck.*.log'))
    assert len(logs)==ranks and all('ERROR SUMMARY: 0 errors' in path.read_text() for path in logs)


def test_wall_trace_worklet_resources(tmp_path):
    from insitu_resource_monitor import run_monitored
    command,env,_=worklet_command((2,2,1))
    with (tmp_path/'worklet.log').open('w') as stream:
        report=run_monitored(command,tmp_path,env,stream,tmp_path/'resources.json',
            device_extra_budget_bytes=None,host_extra_budget_bytes=None,
            device_reserve_bytes=0,timeout_seconds=None)
    assert report['status']=='passed-sampled-observation-not-allocation-proof'
    assert report['limits']==dict(device_extra_bytes=None,host_extra_bytes=None,
        device_reserve_bytes=0,timeout_seconds=None)


@pytest.mark.parametrize('ranks,axis',[(1,0),(2,0),(2,1),(2,2)])
def test_existing_fluid_trace_is_unchanged(tmp_path,ranks,axis):
    executable=os.environ.get('ASTR_WALL_TRACE_FLUID_REGRESSION_PROBE')
    if not executable:
        pytest.skip('Select the rebuilt existing fluid trace probe')
    command,env,_=worklet_command((ranks,1,1))
    command[-2:]=[executable,str(axis),'bidirectional']
    result=subprocess.run(command,cwd=tmp_path,env=env,capture_output=True,text=True)
    (tmp_path/'fluid.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
    assert f'Viskores CUDA trace NP={ranks} particles=32' in result.stdout
    assert f'transfers={32 if ranks==2 else 0}' in result.stdout
    assert 'seam_error=0' in result.stdout and 'input_host_mirror=0' in result.stdout


@pytest.mark.parametrize('ranks,mapping',[(1,''),(2,''),(2,'periodic'),(2,'y-wavy')])
def test_shared_driver_explicit_fluid_parameters(tmp_path,ranks,mapping):
    executable=os.environ.get('ASTR_WALL_TRACE_DRIVER_REGRESSION_PROBE')
    if not executable:
        pytest.skip('Select the rebuilt device streamline driver probe')
    command,env,_=worklet_command((ranks,1,1))
    command[-2:]=[executable,'0','constant','both','resident',mapping,'line16','explicit']
    result=subprocess.run(command,cwd=tmp_path,env=env,capture_output=True,text=True)
    (tmp_path/'driver.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'EXPLICIT_RK45 cases=6 legacy_exact=1 directions=3 step_stop=1' in result.stdout
    assert 'source_errors=0' in result.stdout


@pytest.mark.parametrize('topology',TOPOLOGIES)
def test_wall_shared_production_driver(tmp_path,topology):
    command,env,ranks=worklet_command(topology)
    command.append('shared')
    result=subprocess.run(command,cwd=tmp_path,env=env,capture_output=True,text=True)
    (tmp_path/'shared-driver.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
    assert f'PASS wall shared-driver cases=18 ranks={ranks}' in result.stdout
    assert 'source_host_mirror=0 oracle_geometry_readback=1' in result.stdout


def wall_render_configuration(pipeline='standard-device'):
    from test_insitu_device_products import resident_configuration
    from test_insitu_sbli_statistics import WINDOW
    config=resident_configuration(pipeline,profile='catalog_stats').replace('0.0005,0.0115',f'{WINDOW[0]},{WINDOW[1]}')
    config=config.replace("output_directory='outdat/render'","output_directory='outdat'")
    for old,new in [('host_budget_bytes=4294967296','host_budget_bytes=-1'),
        ('device_budget_bytes=2147483648','device_budget_bytes=-1'),('device_reserve_bytes=1073741824','device_reserve_bytes=0')]:
        assert old in config;config=config.replace(old,new)
    return config


WALL_RENDER_CASES=[(t,'standard-device') for t in TOPOLOGIES]+[((2,1,1),'direct-device')]


@pytest.fixture(scope='module')
def wall_gpu_cases(tmp_path_factory):
    executable=os.environ.get('ASTR_WALL_TRACE_RENDER_EXE')
    if not executable:
        pytest.skip('Select the rebuilt Catalyst/device solver')
    from test_insitu_sbli_statistics import launch
    cases={}
    def obtain(topology,pipeline):
        key=(topology,pipeline)
        if key not in cases:
            root=tmp_path_factory.mktemp('wall-lines-'+''.join(map(str,topology))+'-'+pipeline)
            cases[key]=launch(root/'gpu','gpu',topology,steps=2,records=[shear_lines('instantaneous'),shear_lines('reynolds')],
                insitu_config=wall_render_configuration(pipeline),executable=executable,unrestricted=True)
        return cases[key]
    return obtain


@pytest.mark.parametrize('topology,pipeline',WALL_RENDER_CASES)
def test_wall_trace_real_images(wall_gpu_cases,topology,pipeline):
    from test_runtime_catalog_images import image_segments
    from PIL import Image
    import numpy as np
    case=wall_gpu_cases(topology,pipeline)
    state=json.loads((case/'outdat/new/products.status.json').read_text())
    assert all(p['state']=='active' for p in state['products']),state
    products=image_segments(case);assert len(products)==2
    for segment in products:
        expected=3 if segment.parent.parent.name.startswith('instantaneous') else 2
        images=list(segment.glob('*.jpeg'));assert len(images)==expected
        for path in images:
            with Image.open(path) as image:
                assert image.size==(800,600) and np.any(np.asarray(image.convert('RGB'))<230)
        for path in segment.glob('frame.*.rank*.json'):
            row=json.loads(path.read_text());trace=row['wall_trace']
            assert row['field_d2h_bytes']==row['geometry_d2h_bytes']==row['geometry_host_bytes']==0
            assert trace['seed_count']==3 and trace['particles']==6
            assert sum(trace['terminal_'+reason] for reason in
                ('length','steps','zero','stage_zero','accuracy','physical_exit','region_exit'))==6
            assert trace['terminal_accuracy']==trace['terminal_zero']==trace['terminal_stage_zero']==0
            assert trace['input_metadata_d2h_bytes'] in (0,160)


@pytest.fixture(scope='module')
def cpu_wall_reference(tmp_path_factory):
    executable=os.environ.get('ASTR_WALL_TRACE_CPU_EXE')
    if not executable:
        pytest.skip('Select the root native CPU solver for the real-field oracle')
    from test_insitu_sbli_statistics import launch,WINDOW
    from test_insitu_selected_runtime import configuration
    from test_index_slice_runtime import segments,frames
    records=[]
    for source in ('instantaneous','reynolds'):
        item=shear_lines(source)
        for key in ('field','display','extraction'):item.pop(key)
        item.update(id=source,fields=['wall_shear_x','wall_shear_z'],output=dict(channel='data',format='hdf5-xdmf'))
        records.append(item)
    config=configuration(WINDOW).replace('host_budget_bytes=67108864','host_budget_bytes=-1').replace(
        'device_budget_bytes=67108864','device_budget_bytes=-1').replace('device_reserve_bytes=1073741824','device_reserve_bytes=0')
    root=tmp_path_factory.mktemp('wall-lines-cpu-reference')
    case=launch(root/'cpu','cpu',(1,1,1),steps=2,records=records,insitu_config=config,executable=executable,unrestricted=True)
    return {segment.parent.parent.name:(segment,frames(segment)) for segment in segments(case)}


@pytest.mark.parametrize('topology,pipeline',WALL_RENDER_CASES)
def test_wall_trace_real_cpu_gpu_endpoints(tmp_path,wall_gpu_cases,cpu_wall_reference,topology,pipeline,record_property):
    import ctypes
    import h5py
    import numpy as np
    from test_runtime_catalog_images import image_segments
    library=os.environ.get('ASTR_WALL_TRACE_REFERENCE_LIB')
    if not library:
        pytest.skip('Select the root-built native CPU wall-trace oracle')
    reference=ctypes.CDLL(library).astr_wall_trace_reference
    real=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    integer=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    reference.argtypes=[ctypes.c_int,ctypes.c_int,real,real,real,real,real,ctypes.c_int,real,
        ctypes.c_int,ctypes.c_int,real,integer,real]
    reference.restype=ctypes.c_int
    case=wall_gpu_cases(topology,pipeline);worst=0.
    for segment in image_segments(case):
        source='instantaneous' if segment.parent.parent.name.startswith('instantaneous') else 'reynolds'
        cpu_segment,cpu_frames=cpu_wall_reference[source]
        with h5py.File(cpu_segment/'geometry.h5') as data:xyz=data['coordinates'][...]
        x=np.ascontiguousarray(xyz[0,:,0]);z=np.ascontiguousarray(xyz[:,0,2]);nz,nx=xyz.shape[:2]
        definition=shear_lines(source);options=definition['extraction']
        seeds=np.ascontiguousarray([xyz[k,i] for i,k in options['seeds']])
        parameters=np.array([options[key] for key in
            ('initial_step','minimum_step','maximum_step','maximum_error','maximum_length')],dtype=np.float64)
        by_step={int(path.stem[4:]):path for path in cpu_frames}
        for path in sorted(segment.glob('frame.*.rank00000000.json')):
            frame=json.loads(path.read_text())
            with h5py.File(by_step[frame['step']]) as data:
                tx=np.ascontiguousarray(data['wall_shear_x'][...]);tz=np.ascontiguousarray(data['wall_shear_z'][...])
            # Endpoint aliases use node zero for the existing unique periodic history.
            if source=='reynolds':tx[-1]=tx[0];tz[-1]=tz[0]
            out=np.zeros((6,4),dtype=np.float64);state=np.zeros((6,2),dtype=np.int32);ranges=np.zeros((6,2),dtype=np.float64)
            assert reference(nx,nz,x,z,tx,tz,seeds,3,parameters,options['maximum_steps'],0,out,state,ranges)==0
            actual=frame['wall_trace']['final_particles']
            observed=np.array([[row[key] for key in ('x','y','z','length')] for row in actual])
            scaled=np.abs(observed-out)/np.maximum(1.,np.maximum(np.abs(observed),np.abs(out)))
            worst=max(worst,float(scaled.max()));assert np.isfinite(scaled).all() and scaled.max()<=2e-10
            np.testing.assert_array_equal(np.array([[row['steps'],row['status']] for row in actual]),state)
            np.testing.assert_array_equal([row['direction'] for row in actual],[1,1,1,-1,-1,-1])
            parts=[json.loads(p.read_text()) for p in segment.glob(f'frame.step{frame["step"]:012d}.rank*.json')]
            parts=[row['product']['actual_range'] for row in parts if row['product']['local_points']>0]
            observed_range=np.array([min(row[0] for row in parts),max(row[1] for row in parts)])
            expected_range=np.array([ranges[:,0].min(),ranges[:,1].max()])
            range_error=float(np.abs(observed_range-expected_range).max())
            assert np.isfinite(range_error) and range_error<=2e-10
            worst=max(worst,range_error)
    record_property('maximum_scaled_endpoint_or_length_error',worst)
    (tmp_path/'comparison.json').write_text(json.dumps(dict(maximum_scaled_error=worst,topology=topology,pipeline=pipeline)))


def test_wall_trace_reuses_views_not_seed_or_integrator_changes(tmp_path):
    executable=os.environ.get('ASTR_WALL_TRACE_RENDER_EXE')
    if not executable:
        pytest.skip('Select the rebuilt Catalyst/device solver')
    from test_insitu_sbli_statistics import launch
    from test_runtime_catalog_images import image_segments
    records=[shear_lines()]
    camera=deepcopy(records[0]);camera['id']='camera';camera['display'].update(camera=[4.5,-22.,.125],line_width=3)
    seeds=deepcopy(records[0]);seeds['id']='seeds';seeds['extraction']['seeds'][0][0]=9
    integration=deepcopy(records[0]);integration['id']='integration';integration['extraction']['maximum_length']=1.8
    case=launch(tmp_path/'gpu','gpu',(2,1,1),steps=2,records=[*records,camera,seeds,integration],
        insitu_config=wall_render_configuration(),executable=executable,unrestricted=True)
    state=json.loads((case/'outdat/new/products.status.json').read_text())
    assert all(item['state']=='active' for item in state['products']),state
    expected={'reynolds-wall-lines':1,'camera':0,'seeds':1,'integration':1}
    for segment in image_segments(case):
        name=segment.parent.parent.name
        for path in segment.glob('frame.*.rank*.json'):
            row=json.loads(path.read_text());trace=row['wall_trace']
            assert trace['integration_performed']==expected[name]
            if name=='camera':assert trace['control_read_bytes']==trace['input_metadata_d2h_bytes']==0
    assert (case/'run.log').read_text().count('integrations=3 owner_reuse=1')==4


def test_wall_trace_real_exact_restart_and_authority(tmp_path):
    executable=os.environ.get('ASTR_WALL_TRACE_RENDER_EXE')
    if not executable:
        pytest.skip('Select the rebuilt Catalyst/device solver')
    from test_insitu_sbli_statistics import launch
    from test_runtime_catalog_images import image_segments
    from run_output_restart_validation import compare_fields
    records=[shear_lines('instantaneous'),shear_lines('reynolds')]
    common=dict(insitu_config=wall_render_configuration(),executable=executable,unrestricted=True,checkpoint=True)
    full=launch(tmp_path/'full','gpu',(2,1,1),records=records,**common)
    prefix=launch(tmp_path/'prefix','gpu',(2,1,1),steps=5,records=records,**common)
    tail=launch(tmp_path/'tail','gpu',(2,1,1),records=records,
        restore=prefix/'outdat/new/checkpoints/step000000000005',**common)
    off=launch(tmp_path/'off','gpu',(2,1,1),records=[],**common)
    final=Path('outdat/new/checkpoints/step000000000012')
    for name in ('state.h5','statistics.h5'):compare_fields(full/final/name,tail/final/name)
    compare_fields(full/final/'state.h5',off/final/'state.h5')
    reference={s.parent.parent.name:s for s in image_segments(full)}
    for segment in image_segments(tail):
        original=reference[segment.parent.parent.name]
        for image in segment.glob('*.jpeg'):assert image.read_bytes()==(original/image.name).read_bytes()
        for path in segment.glob('frame.*.rank*.json'):
            actual=json.loads(path.read_text());expected=json.loads((original/path.name).read_text())
            assert actual.get('statistics')==expected.get('statistics')
            assert actual['wall_trace']==expected['wall_trace']


def test_wall_trace_real_memory_safety(tmp_path):
    executable=os.environ.get('ASTR_WALL_TRACE_RENDER_EXE')
    if not executable:
        pytest.skip('Select the rebuilt Catalyst/device solver')
    from test_insitu_sbli_statistics import launch
    case=launch(tmp_path/'gpu','gpu',(2,1,2),steps=2,records=[shear_lines('instantaneous'),shear_lines('reynolds')],
        insitu_config=wall_render_configuration(),executable=executable,unrestricted=True,memcheck=True)
    logs=list(case.glob('memcheck.*.log'))
    assert len(logs)==4 and all('ERROR SUMMARY: 0 errors' in path.read_text() for path in logs)
    state=json.loads((case/'outdat/new/products.status.json').read_text())
    assert all(item['state']=='active' for item in state['products']),state


def test_wall_trace_real_cost_without_oracles(tmp_path):
    import re
    executable=os.environ.get('ASTR_WALL_TRACE_RENDER_EXE')
    if not executable:
        pytest.skip('Select the rebuilt Catalyst/device solver')
    from test_insitu_sbli_statistics import launch
    common=dict(steps=2,insitu_config=wall_render_configuration(),executable=executable,unrestricted=True,monitor=True)
    off=launch(tmp_path/'off','gpu',(2,1,1),records=[],**common)
    on=launch(tmp_path/'on','gpu',(2,1,1),records=[shear_lines('instantaneous'),shear_lines('reynolds')],
        resource_baseline=json.loads((off/'resources.sampled.json').read_text()),**common)
    result=dict(scope='short two-step NP2 x GRCBC, instantaneous and mean wall lines; includes selected statistics, rendering and JPEG; excludes initial-step-zero rendering from the RK window; not a stable performance benchmark',cases={})
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


@pytest.mark.parametrize('keep_history',[False,True])
def test_wall_trace_late_join_and_sampling_gap(tmp_path,keep_history,catalog_replay_library):
    executable=os.environ.get('ASTR_WALL_TRACE_RENDER_EXE')
    if not executable:
        pytest.skip('Select the rebuilt Catalyst/device solver')
    from test_insitu_sbli_statistics import launch,DT,WINDOW
    from test_runtime_catalog_images import image_segments
    item=shear_lines();keeper=deepcopy(item)
    for key in ('field','display','extraction'):keeper.pop(key)
    keeper.update(id='sampling-keeper',fields=['wall_shear_x','wall_shear_z'],output=dict(channel='data',format='hdf5-xdmf'))
    keeper['schedule']['interval']=1000
    keepers=[keeper] if keep_history else []
    publications={4:[*keepers,item],6:keepers,8:[*keepers,item]}
    case=launch(tmp_path/'gpu','gpu',(1,1,2),records=keepers,insitu_config=wall_render_configuration(),
        executable=executable,unrestricted=True,catalog_replay=(catalog_replay_library,publications))
    state=json.loads((case/'outdat/new/products.status.json').read_text())
    assert all(row['state']=='active' for row in state['products']),state
    pictures=image_segments(case);assert pictures
    records=[json.loads(path.read_text()) for segment in pictures for path in segment.glob('frame.*.rank00000000.json')]
    latest=max(records,key=lambda row:row['step']);statistics=latest['statistics']
    expected_start=WINDOW[0] if keep_history else 3*DT
    expected_duration=WINDOW[1]-WINDOW[0] if keep_history else 5.5*DT
    assert abs(statistics['actual_sample_range'][0]-expected_start)<2e-15
    assert abs(statistics['covered_duration']-expected_duration)<2e-15
    assert statistics['gaps_excluded']
    assert min(row['step'] for row in records)>=3 and not any(row['step'] in (5,6) for row in records)
    assert latest['wall_trace']['particles']==6


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
def test_wall_trace_changes_preserve_existing_wall_images(tmp_path,pipeline):
    executable=os.environ.get('ASTR_WALL_TRACE_RENDER_EXE')
    if not executable:
        pytest.skip('Select the rebuilt Catalyst/device solver')
    from test_insitu_sbli_statistics import launch
    from test_insitu_wall_footprints import wall_image_definitions
    from test_runtime_catalog_images import image_segments
    records=[*wall_image_definitions(),glyph('instantaneous'),glyph('reynolds')]
    case=launch(tmp_path/'gpu','gpu',(1,1,2),steps=2,records=records,
        insitu_config=wall_render_configuration(pipeline),executable=executable,unrestricted=True)
    state=json.loads((case/'outdat/new/products.status.json').read_text())
    assert all(row['state']=='active' for row in state['products']),state
    products=image_segments(case)
    assert len(products)==6 and sum(len(list(s.glob('*.jpeg'))) for s in products)==15
    for segment in products:
        for path in segment.glob('frame.*.rank*.json'):
            frame=json.loads(path.read_text())
            assert frame['field_d2h_bytes']==frame['geometry_d2h_bytes']==frame['geometry_host_bytes']==0
