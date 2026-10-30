"""HCW-inspired planar SBLI: completed-step device products, not steady physics."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np
from PIL import Image
import pytest

from prepare_hcw_laminar_sbli_case import prepare_case
from prepare_s1_flatplate_case import write_controller
from run_m12_mp7_nscbc_performance import runtime, mpi_command
from run_output_restart_validation import compare_fields

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'examples/HCW_Laminar_SBLI/case_review'
EXE=os.environ.get('ASTR_HCW_INSITU_EXE')
LIBRARY=os.environ.get('ASTR_OUTPUT_INSITU_BACKEND',
    '/home/dell/workspace/astr_dependencies/install/paraview-6.1.1-hpcx-gcc13/lib/catalyst')
MPIEXEC='/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec'
SBLI_2D_PRODUCTS=('velocity_slice','density_gradient_slice','streamwise_velocity_slice',
                  'pressure_ratio_slice','temperature_slice','wall_pressure_curve',
                  'skin_friction_curve','wall_heat_flux_curve')


def configuration_2d(interval=2):
    config=configuration(interval).replace("products='sbli_diagnostics',streamline_seeds='sbli-layered16'",
                                          "products='sbli_2d'")
    return config


def test_sbli_2d_configuration(tmp_path):
    probe=os.environ.get('ASTR_INSITU_CONFIG_PROBE')
    if not probe:
        pytest.skip('Set ASTR_INSITU_CONFIG_PROBE')
    path=tmp_path/'sbli2d.nml'
    path.write_text(configuration_2d())
    result=subprocess.run([probe,str(path)],capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr


def test_sbli_2d_unlimited_configuration(tmp_path):
    probe=os.environ.get('ASTR_INSITU_CONFIG_PROBE')
    if not probe:
        pytest.skip('Set ASTR_INSITU_CONFIG_PROBE')
    config=configuration_2d().replace('host_budget_bytes=4294967296','host_budget_bytes=-1').replace(
        'device_budget_bytes=2147483648','device_budget_bytes=-1').replace('device_reserve_bytes=1073741824',
        'device_reserve_bytes=0')
    path=tmp_path/'unlimited.nml';path.write_text(config)
    result=subprocess.run([probe,str(path)],capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr


def configuration(interval=2):
    return f"""&insitu_run
 enabled=t,render=t,statistics=f,initial_frame=f,final_frame=f,
 products='sbli_diagnostics',streamline_seeds='sbli-layered16',
 processing_backend='device',derivative_backend='gpu',
 rendering_pipeline='standard-device',postprocess_transport='pinned',
 output_directory='outdat/render',schedule_mode='steps',step_interval={interval},
 host_budget_bytes=4294967296,device_budget_bytes=2147483648,device_reserve_bytes=1073741824,
 implementation_path='{LIBRARY}',pipeline_file='{ROOT}/scripts/insitu/device_render_pipeline.py'
/
"""


def test_sbli_configuration(tmp_path):
    probe=os.environ.get('ASTR_INSITU_CONFIG_PROBE')
    if not probe:
        pytest.skip('Set ASTR_INSITU_CONFIG_PROBE')
    path=tmp_path/'insitu.nml'
    for content in (configuration(),(SOURCE/'datin/input.insitu').read_text()):
        path.write_text(content)
        result=subprocess.run([probe,str(path)],capture_output=True,text=True,timeout=10)
        assert result.returncode==0,result.stdout+result.stderr


def test_review_checkpoint_interval_preserves_controller_and_stop():
    assert 'interval_steps=3000,keep=2' in (SOURCE/'datin/input.output').read_text()
    assert 'host_budget_bytes=-1,device_budget_bytes=-1' in (SOURCE/'datin/input.output').read_text()
    assert 'exact_restart=t,checkpoint_statistics=t' in (SOURCE/'datin/input.output').read_text()
    controller=(SOURCE/'datin/controller').read_text().splitlines()
    settings=next(row for row in controller if row.startswith('29999,'))
    assert settings.split(',')[:2]==['29999','10']


def test_review_requests_two_dimensional_products():
    content=(SOURCE/'datin/input.insitu').read_text()
    assert "products='sbli_2d'" in content
    assert 'step_interval=100' in content
    assert 'q_surface.image' not in content and 'streamlines.image' not in content


def test_sbli_seed_sites_are_physical_and_spanwise_unique(tmp_path):
    source=tmp_path/'seeds.cpp'
    source.write_text('''#include "insitu_streamline_seeds.h"
#include <iostream>
int main() {
  const std::array<astr_insitu::Real,6> b={.6786378346241067,1.,0.,.03,0.,.002};
  const auto seeds=astr_insitu::tgv_streamline_seeds("sbli-layered16",&b);
  std::cout.precision(17);
  for(const auto& s:seeds) std::cout<<s[0]<<" "<<s[1]<<" "<<s[2]<<"\\n";
}''')
    executable=tmp_path/'seeds'
    subprocess.run(['g++','-std=c++17','-I'+str(ROOT/'src'),'-I'+str(ROOT/'src_gpu'),
                    str(source),'-o',str(executable)],check=True,capture_output=True,text=True)
    result=subprocess.run([str(executable)],check=True,capture_output=True,text=True)
    sites=np.array([[float(x) for x in row.split()] for row in result.stdout.splitlines()])
    assert sites.shape==(16,3)
    assert np.all((sites[:,0]>.6786378346241067)&(sites[:,0]<1.))
    assert np.all((sites[:,1]>0)&(sites[:,1]<.03)) and np.all(np.diff(sites[:,1])>0)
    np.testing.assert_array_equal(sites[:,2],np.full(16,.001))


def launch(case,ranks,render,full=False,steps=2,restore=None,memcheck=False,host_budget_bytes=1073741824,
           insitu_config=None,sbli_oracle=False,transport='pinned',output_device_budget_bytes=1073741824,
           case_runtime=False,checkpoint_interval=1,product_catalog_records=None,product_poll_steps=10,
           catalog_replay=None,wall_publication_failure=None,image_publication_failure=None,wall_blowing=False,
           wall_capture_refusal=None):
    if full:
        shutil.copytree(SOURCE/'datin',case/'datin')
        manifest=json.loads((SOURCE/'case_manifest.json').read_text())
        shutil.copyfile(SOURCE/'runtime.env',case/'runtime.env')
        shutil.copyfile(SOURCE/'case_manifest.json',case/'case_manifest.json')
        (case/'outdat/new').mkdir(parents=True)
    else:
        manifest=prepare_case(case,im=64,jm=48,km=5)
    dt=manifest['run_plan']['dt_nondimensional']
    if wall_blowing:
        (case/'datin/wallbs.dat').write_text('wall blowing\nparameters\n0.001 1.0 0.78 0.82 0.86\nmodes\n0 1\n')
    write_controller(case/'datin/controller',steps-1,1,dt)
    (case/'datin/input.output').write_text(f"""&output
 directory='outdat/new',restore_directory='{restore or ''}',exact_restart=t,
 host_budget_bytes={host_budget_bytes},device_budget_bytes={output_device_budget_bytes},buffer_bytes=4194304,
 product_poll_steps={product_poll_steps}
/
&checkpoint
 enabled=t,interval_steps={checkpoint_interval},keep=2,initial_frame=f
/
&volume
 enabled=f
/
&slices
 enabled=f
/
""")
    if product_catalog_records is not None:
        (case/'input.products.jsonl').write_text('\n'.join(json.dumps(record) for record in
            [{'catalog_schema':'astr-insitu-products-v1'},*product_catalog_records])+'\n')
    env=runtime(f'{ranks},1,1',transport)
    for key in tuple(env):
        if key.startswith('ASTR_INSITU_') or key in ('DISPLAY','PYTHONPATH','ASTR_WALL_BLOWING_MODE',
                'CATALYST_IMPLEMENTATION_PREFER_ENV','VTK_EGL_DEVICE_INDEX'):
            env.pop(key,None)
    for line in (case/'runtime.env').read_text().splitlines():
        key,value=line.removeprefix('export ').split('=',1)
        env[key]=value
    if case_runtime:
        for line in (SOURCE/'runtime.insitu.env').read_text().splitlines():
            key,value=line.removeprefix('export ').split('=',1)
            env[key]=value
        env.pop('ASTR_INSITU_CONFIG',None)
    env['ASTR_INSITU_WRITE_EPS']='0'
    if wall_capture_refusal is not None:
        env['ASTR_INSITU_TEST_WALL_CAPTURE_REFUSAL']=' '.join(map(str,wall_capture_refusal))
    env['OMPI_MCA_osc']='sm,pt2pt'
    if catalog_replay is not None:
        library,publications=catalog_replay
        replay=case/'catalog_replay';replay.mkdir()
        for poll,records in publications.items():
            (replay/f'catalog{poll:06d}.jsonl').write_text('\n'.join(json.dumps(record) for record in
                [{'catalog_schema':'astr-insitu-products-v1'},*records])+'\n')
        env.update(LD_PRELOAD=str(Path(library).resolve()),ASTR_TEST_CATALOG_FILE='input.products.jsonl',
                   ASTR_TEST_CATALOG_REPLAY=str(replay.resolve()))
    if wall_publication_failure is not None:
        name,step=wall_publication_failure
        env.update(ASTR_TEST_WALL_FAIL_ID=name,ASTR_TEST_WALL_FAIL_STEP=str(step))
    if image_publication_failure is not None:
        name,step=image_publication_failure
        env.update(ASTR_TEST_IMAGE_FAIL_ID=name,ASTR_TEST_IMAGE_FAIL_STEP=str(step))
    if render:
        (case/'outdat/render').mkdir()
        (case/'datin/input.insitu').write_text(configuration() if insitu_config is None else insitu_config)
        env['ASTR_INSITU_CONFIG']='datin/input.insitu'
        if sbli_oracle:
            env['ASTR_INSITU_TEST_SBLI_PREFIX']=str(case.resolve()/'plane_oracle')
    command=mpi_command(MPIEXEC,ranks,transport)
    if memcheck:
        command+=['compute-sanitizer','--tool','memcheck','--error-exitcode','99']
    with (case/'run.log').open('w') as log:
        result=subprocess.run(command+[str(Path(EXE).resolve()),'run','datin/input.hcw'],
            cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    text=(case/'run.log').read_text()
    assert result.returncode==0 and 'The job is done!' in text,text[-7000:]
    assert 'sbli-grcbc-h1 high-order boundary chain' in text,text[-7000:]
    if memcheck:
        assert text.count('ERROR SUMMARY: 0 errors')==ranks,text[-7000:]
    with h5py.File(case/f'outdat/new/checkpoints/step{steps:012d}/state.h5') as state:
        rho=state['q0001'][...]
        momentum=np.stack([state[f'q{i:04d}'][...] for i in (2,3,4)],axis=-1)
        energy=state['q0005'][...]
        assert np.isfinite(rho).all() and np.isfinite(momentum).all() and np.isfinite(energy).all()
        assert rho.min()>0 and np.min(energy-.5*np.sum(momentum**2,axis=-1)/rho)>0
    control=case/f'outdat/new/checkpoints/step{steps:012d}/control.bin'
    assert b'sbli-grcbc-h1' in control.read_bytes()
    return case


def logical_derivative(values,axis):
    """Independent seven-node polynomial derivative on the logical grid."""
    result=np.empty_like(values)
    count=values.shape[axis]
    for node in range(count):
        start=max(0,min(node-3,count-7))
        distances=np.arange(start,start+7,dtype=float)-node
        moments=np.zeros(7);moments[1]=1.
        weights=np.linalg.solve(np.vander(distances,7,increasing=True).T,moments)
        stencil=np.take(values,np.arange(start,start+7),axis=axis)
        center=np.expand_dims(np.take(values,node,axis=axis),axis)
        derivative=np.tensordot(weights,np.moveaxis(stencil-center,axis,0),axes=1)
        index=[slice(None)]*values.ndim;index[axis]=node
        result[tuple(index)]=derivative
    return result


def compare_sbli_diagnostics(case,ranks,step,manifest_path=None,plane_oracle=True):
    with h5py.File(case/f'outdat/new/checkpoints/step{step:012d}/state.h5') as file:
        q=np.stack([file[f'q{i:04d}'][...] for i in range(1,6)],axis=-1)
    with h5py.File(case/'outdat/new/resources/geometry.h5') as file:
        xyz=np.stack([file[f'q{i:04d}'][...] for i in range(1,4)],axis=-1)
        dxi=[file[f'q{i:04d}'][...] for i in (5,6,8,9)]
    manifest=json.loads((case/'case_manifest.json' if manifest_path is None else manifest_path).read_text())
    flow=manifest['flow'];mach=flow['mach']
    rho=q[...,0];velocity=q[...,1:4]/rho[...,None]
    pressure=(q[...,4]-.5*rho*np.sum(velocity*velocity,axis=-1))/2.5
    temperature=pressure/rho*(1.4*mach*mach)
    pinf=1./(1.4*mach*mach)
    drx=logical_derivative(rho,2);dry=logical_derivative(rho,1)
    gradx=drx*dxi[0]+dry*dxi[1];grady=drx*dxi[2]+dry*dxi[3]
    length=xyz[0,0,-1,0]-xyz[0,0,0,0]
    reference=np.stack([np.linalg.norm(velocity,axis=-1),velocity[...,0],pressure/pinf,temperature,
                        length*np.sqrt(gradx*gradx+grady*grady)],axis=-1)
    z=.5*(xyz[0,0,0,2]+xyz[-1,0,0,2])
    first=np.searchsorted(xyz[:,0,0,2],z,side='right')-1
    weight=(z-xyz[first,0,0,2])/(xyz[first+1,0,0,2]-xyz[first,0,0,2])
    reference=(1-weight)*reference[first]+weight*reference[first+1]
    du=logical_derivative(velocity[...,0],1)*dxi[3]
    dt=logical_derivative(temperature,1)*dxi[3]
    sutherland=flow['sutherland_temperature_K']/flow['temperature_K']
    mu=temperature*np.sqrt(temperature)*(1+sutherland)/((temperature+sutherland)*flow['reynolds_reference'])
    curves=np.stack([xyz[:,0,:,0],pressure[:,0,:]/pinf,2*mu[:,0,:]*du[:,0,:],
                     (mu[:,0,:]/flow['prandtl'])/(.4*mach*mach)*dt[:,0,:]],axis=-1)
    curves=(1-weight)*curves[first]+weight*curves[first+1]
    errors={}
    for rank in range(ranks):
        if not plane_oracle:
            # Production builds omit the test-only plane download. Check the
            # published FP64 ranges instead; wall CSVs remain nodewise checks.
            width=(reference.shape[1]-1)//ranks
            expected=reference[:,rank*width:(rank+1)*width+1]
            record=json.loads((case/f'outdat/render/mesh_step{step:08d}_rank{rank}.json').read_text())
            for name in SBLI_2D_PRODUCTS[:5]:
                # Product order differs from the native compact-field order.
                component={'velocity_slice':0,'density_gradient_slice':4,'streamwise_velocity_slice':1,
                           'pressure_ratio_slice':2,'temperature_slice':3}[name]
                values=expected[...,component]
                if component==4:
                    values=np.log10(1.+values)
                target=np.array([values.min(),values.max()])
                actual=np.array(record['products'][name]['actual_range'])
                error=float(np.max(np.abs(actual-target)))
                errors[f'plane_rank{rank}_{name}_range']=error
                assert error<=2e-10,errors
            continue
        path=case/f'plane_oracle.step{step:08d}.rank{rank}.bin'
        with path.open('rb') as stream:
            nx,ny,offset=np.fromfile(stream,dtype=np.int64,count=3)
            count=(nx+1)*(ny+1)
            points=np.fromfile(stream,dtype=np.float64,count=3*count).reshape(ny+1,nx+1,3)
            actual=np.fromfile(stream,dtype=np.float64,count=5*count).reshape(ny+1,nx+1,5)
            assert not stream.read()
        np.testing.assert_allclose(points[...,2],z,atol=2e-10,rtol=0.)
        expected=reference[:,offset:offset+nx+1]
        for c in range(5):
            error=float(np.max(np.abs(actual[...,c]-expected[...,c])))
            errors[f'plane_rank{rank}_component{c}']=error
            assert error<=2e-10,errors
    for name,column in (('wall_pressure_curve',1),('skin_friction_curve',2),('wall_heat_flux_curve',3)):
        actual=np.loadtxt(case/f'outdat/render/{name}.step{step:08d}.csv',delimiter=',')
        np.testing.assert_allclose(actual[:,0],curves[:,0],atol=2e-10,rtol=0.)
        error=float(np.max(np.abs(actual[:,1]-curves[:,column])))
        errors[name]=error
        assert error<=2e-10,errors
    (case/'diagnostic_errors.json').write_text(json.dumps(errors,indent=2))


@pytest.mark.skipif(not EXE,reason='Set ASTR_HCW_INSITU_EXE')
@pytest.mark.parametrize('ranks',[1,2,4])
def test_hcw_two_dimensional_products(tmp_path,ranks):
    off=launch(tmp_path/'off',ranks,False)
    on=launch(tmp_path/'on',ranks,True,insitu_config=configuration_2d(),sbli_oracle=True)
    compare_fields(off/'outdat/new/checkpoints/step000000000002/state.h5',
                   on/'outdat/new/checkpoints/step000000000002/state.h5')
    for rank in range(ranks):
        record=json.loads((on/f'outdat/render/mesh_step00000002_rank{rank}.json').read_text())
        assert set(record['products'])==set(SBLI_2D_PRODUCTS)
        assert all(item['image']=='published' for item in record['products'].values()),record
        assert record['geometry_host_bytes']==0
    for name in SBLI_2D_PRODUCTS:
        with Image.open(on/f'outdat/render/{name}.step00000002.jpeg') as image:
            assert image.size==((3840,1440) if name.endswith('_curve') else (3840,960))
            assert image.info['dpi']==(384,384)
    assert not list((on/'outdat/render').glob('*.eps'))
    assert len(list((on/'outdat/render').glob('*.jpeg')))==8
    compare_sbli_diagnostics(on,ranks,2)


@pytest.mark.skipif(not EXE,reason='Set ASTR_HCW_INSITU_EXE')
def test_hcw_h1_production_smoke(tmp_path):
    off=launch(tmp_path/'off',2,False)
    on=launch(tmp_path/'on',2,True,insitu_config=configuration_2d())
    compare_fields(off/'outdat/new/checkpoints/step000000000002/state.h5',
                   on/'outdat/new/checkpoints/step000000000002/state.h5')
    for rank in range(2):
        record=json.loads((on/f'outdat/render/mesh_step00000002_rank{rank}.json').read_text())
        assert set(record['products'])==set(SBLI_2D_PRODUCTS)
        assert record['geometry_host_bytes']==0
        assert all(item['image']=='published' for item in record['products'].values()),record
    for name in SBLI_2D_PRODUCTS:
        with Image.open(on/f'outdat/render/{name}.step00000002.jpeg') as image:
            assert image.size==((3840,1440) if name.endswith('_curve') else (3840,960))
            assert image.info['dpi']==(384,384)
    assert not list((on/'outdat/render').glob('*.eps'))
    assert len(list((on/'outdat/render').glob('*.jpeg')))==8
    assert not list(on.glob('plane_oracle*'))
    compare_sbli_diagnostics(on,2,2,plane_oracle=False)


@pytest.mark.skipif(not EXE,reason='Set ASTR_HCW_INSITU_EXE')
def test_hcw_two_dimensional_full_grid(tmp_path):
    off=launch(tmp_path/'off',2,False,full=True)
    on=launch(tmp_path/'on',2,True,full=True,insitu_config=configuration_2d(),sbli_oracle=True)
    compare_fields(off/'outdat/new/checkpoints/step000000000002/state.h5',
                   on/'outdat/new/checkpoints/step000000000002/state.h5')
    for name in SBLI_2D_PRODUCTS:
        assert (on/f'outdat/render/{name}.step00000002.jpeg').is_file()
    compare_sbli_diagnostics(on,2,2)


@pytest.mark.skipif(not EXE,reason='Set ASTR_HCW_INSITU_EXE')
def test_hcw_two_dimensional_memcheck(tmp_path):
    on=launch(tmp_path/'on',2,True,memcheck=True,insitu_config=configuration_2d())
    for rank in range(2):
        record=json.loads((on/f'outdat/render/mesh_step00000002_rank{rank}.json').read_text())
        assert all(item['image']=='published' for item in record['products'].values()),record


@pytest.mark.skipif(not EXE,reason='Set ASTR_HCW_INSITU_EXE')
def test_hcw_two_dimensional_exact_restart(tmp_path):
    config=configuration_2d()
    full=launch(tmp_path/'full',2,True,steps=4,insitu_config=config)
    part=launch(tmp_path/'part',2,True,steps=2,insitu_config=config)
    tail=launch(tmp_path/'tail',2,True,steps=4,insitu_config=config,
                restore=part/'outdat/new/checkpoints/step000000000002')
    final=Path('outdat/new/checkpoints/step000000000004')
    compare_fields(full/final/'state.h5',tail/final/'state.h5')
    assert (full/final/'insitu_control.bin').read_bytes()==(tail/final/'insitu_control.bin').read_bytes()
    for name in SBLI_2D_PRODUCTS:
        frame=f'outdat/render/{name}.step00000004.jpeg'
        with Image.open(full/frame) as a,Image.open(tail/frame) as b:
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
        if name.endswith('_curve'):
            filename=f'outdat/render/{name}.step00000004.csv'
            assert (full/filename).read_bytes()==(tail/filename).read_bytes()


@pytest.mark.skipif(not EXE,reason='Set ASTR_HCW_INSITU_EXE')
def test_hcw_production_settings_aware_hd(tmp_path):
    config=(SOURCE/'datin/input.insitu').read_text().replace('step_interval=100','step_interval=2')
    config=config.replace('../../../../astr_dependencies/install/paraview-6.1.1-hpcx-gcc13/lib/catalyst',LIBRARY)
    config=config.replace('../../../scripts/insitu/device_render_pipeline.py',
                          str(ROOT/'scripts/insitu/device_render_pipeline.py'))
    off=launch(tmp_path/'off',2,False,full=True,transport='device-aware',case_runtime=True,
               host_budget_bytes=-1,output_device_budget_bytes=-1)
    on=launch(tmp_path/'on',2,True,full=True,transport='device-aware',case_runtime=True,
              insitu_config=config,host_budget_bytes=-1,output_device_budget_bytes=-1)
    final=Path('outdat/new/checkpoints/step000000000002/state.h5')
    compare_fields(off/final,on/final)
    with h5py.File(on/final) as checkpoint:
        assert all(f'q{i:04d}' in checkpoint for i in range(1,12))
        assert 'rank_extras' in checkpoint
    with h5py.File(on/'outdat/new/resources/geometry.h5') as geometry:
        assert all(f'q{i:04d}' in geometry for i in range(1,14))
    text=(on/'run.log').read_text()
    assert 'ASTR_GPU_HALO_TRANSPORT_SELECTED=device-aware' in text
    lines=[line for line in text.splitlines() if line.startswith('ASTR_INSITU_SBLI_TRANSPORT')]
    assert len(lines)==2
    assert all('transport=device-aware' in line and 'host_slots=0' in line and
               'face_d2h_bytes=0' in line and 'face_h2d_bytes=0' in line for line in lines)
    for name in SBLI_2D_PRODUCTS:
        with Image.open(on/f'outdat/render/{name}.step00000002.jpeg') as image:
            assert image.size==((3840,1440) if name.endswith('_curve') else (3840,960))
    for rank in range(2):
        record=json.loads((on/f'outdat/render/mesh_step00000002_rank{rank}.json').read_text())
        assert all(item['image']=='published' for item in record['products'].values()),record
    compare_sbli_diagnostics(on,2,2,plane_oracle=False)


@pytest.mark.skipif(not EXE,reason='Set ASTR_HCW_INSITU_EXE')
def test_hcw_h1_production_first_frame(tmp_path):
    config=(SOURCE/'datin/input.insitu').read_text()
    config=config.replace('../../../../astr_dependencies/install/paraview-6.1.1-hpcx-gcc13/lib/catalyst',LIBRARY)
    config=config.replace('../../../scripts/insitu/device_render_pipeline.py',
                          str(ROOT/'scripts/insitu/device_render_pipeline.py'))
    case=launch(tmp_path/'on',2,True,full=True,steps=100,transport='device-aware',case_runtime=True,
                insitu_config=config,host_budget_bytes=-1,output_device_budget_bytes=-1,
                checkpoint_interval=3000)
    frames=sorted((case/'outdat/render').glob('*.jpeg'))
    assert {path.name for path in frames}=={f'{name}.step00000100.jpeg' for name in SBLI_2D_PRODUCTS}
    assert not list((case/'outdat/render').glob('*.eps'))
    for path in frames:
        with Image.open(path) as image:
            assert image.size==((3840,1440) if path.name.startswith(('wall_','skin_')) else (3840,960))
    for rank in range(2):
        record=json.loads((case/f'outdat/render/mesh_step00000100_rank{rank}.json').read_text())
        assert record['geometry_host_bytes']==0
        assert set(record['products'])==set(SBLI_2D_PRODUCTS)
        assert all(item['image']=='published' for item in record['products'].values()),record
    checkpoints=case/'outdat/new/checkpoints'
    assert {path.name for path in checkpoints.iterdir() if path.is_dir()}=={'step000000000100'}
    compare_sbli_diagnostics(case,2,100,plane_oracle=False)


@pytest.mark.skipif(not EXE,reason='Set ASTR_HCW_INSITU_EXE')
def test_hcw_slice_only_does_not_change_advance(tmp_path):
    config=(SOURCE/'datin/input.insitu').read_text().replace('step_interval=100',
        "step_interval=0,product_ids(1)='velocity_slice.image',product_modes(1)='steps',product_steps(1)=2")
    config=config.replace("postprocess_transport='device-aware'","postprocess_transport='pinned'")
    config=config.replace('host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0',
        'host_budget_bytes=4294967296,device_budget_bytes=2147483648,device_reserve_bytes=1073741824')
    config=config.replace('../../../../astr_dependencies/install/paraview-6.1.1-hpcx-gcc13/lib/catalyst',LIBRARY)
    config=config.replace('../../../scripts/insitu/device_render_pipeline.py',
                          str(ROOT/'scripts/insitu/device_render_pipeline.py'))
    off=launch(tmp_path/'off',2,False,steps=4)
    on=launch(tmp_path/'on',2,True,steps=4,insitu_config=config)
    final=Path('outdat/new/checkpoints/step000000000004/state.h5')
    compare_fields(off/final,on/final)
    for step in (2,4):
        for rank in range(2):
            record=json.loads((on/f'outdat/render/mesh_step{step:08d}_rank{rank}.json').read_text())
            assert record['geometry_host_bytes']==0
            assert set(record['products'])=={'velocity_slice'}
            assert record['products']['velocity_slice']['image']=='published'
        with Image.open(on/f'outdat/render/velocity_slice.step{step:08d}.jpeg') as image:
            assert image.size==(3840,960)
    assert len(list((on/'outdat/render').glob('*.jpeg')))==2
    assert not list((on/'outdat/render').glob('*.eps'))


@pytest.mark.skipif(not EXE,reason='Set ASTR_HCW_INSITU_EXE')
def test_zero_area_cleanup_keeps_finite_triangles_and_rejects_nonfinite(tmp_path):
    probe=Path(EXE).resolve().with_name('insitu_device_geometry_probe')
    command=mpi_command(MPIEXEC,1,'pinned')+[str(probe),'0','surface-cleanup']
    result=subprocess.run(command,env=runtime('1,1,1'),capture_output=True,text=True,timeout=90)
    (tmp_path/'cleanup.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'SURFACE_CLEANUP_PASS positive=1 empty=1 nonfinite_rejected=1 tolerance=0' in result.stdout


@pytest.mark.skipif(not EXE,reason='Set ASTR_HCW_INSITU_EXE')
@pytest.mark.parametrize('ranks',[1,2,4])
def test_hcw_render_does_not_change_advance(tmp_path,ranks):
    off=launch(tmp_path/'off',ranks,False,steps=4)
    on=launch(tmp_path/'on',ranks,True,steps=4)
    batch=Path('outdat/new/checkpoints/step000000000004')
    compare_fields(off/batch/'state.h5',on/batch/'state.h5')
    records=[]
    for rank in range(ranks):
        record=json.loads((on/f'outdat/render/mesh_step00000002_rank{rank}.json').read_text())
        records.append(record)
        assert record['geometry_host_bytes']==0
        assert set(record['products'])=={'velocity_slice','q_surface','instantaneous_streamlines'}
        for name,item in record['products'].items():
            assert item['image']=='published',(name,item)
            assert item['physical_bounds'][0]==pytest.approx(.6786378346241067)
            assert item['physical_bounds'][1:]==pytest.approx([1.,0.,.03,0.,.002])
        assert record['products']['instantaneous_streamlines']['seeding']['seeds']==16
    for name in ('velocity_slice','instantaneous_streamlines'):
        assert sum(record['products'][name]['local_cells'] for record in records)>0,name
    for name in ('velocity_slice','q_surface','instantaneous_streamlines'):
        with Image.open(on/f'outdat/render/{name}.step00000002.jpeg') as image:
            assert image.size==(2560,640)
            if name!='q_surface':
                pixels=np.asarray(image.convert('RGB'))[:512,:1920]
                assert np.count_nonzero(np.any(pixels<220,axis=-1))>100,name
        assert not (on/f'outdat/render/{name}.step00000002.eps').exists()


@pytest.mark.skipif(not EXE,reason='Set ASTR_HCW_INSITU_EXE')
def test_hcw_checkpoint_host_budget_np2(tmp_path):
    launch(tmp_path/'checkpoint_budget',2,False,full=True,host_budget_bytes=1073741824)


@pytest.mark.skipif(not EXE,reason='Set ASTR_HCW_INSITU_EXE')
def test_hcw_full_grid_first_frame(tmp_path):
    off=launch(tmp_path/'off',2,False,full=True)
    on=launch(tmp_path/'on',2,True,full=True)
    batch=Path('outdat/new/checkpoints/step000000000002')
    compare_fields(off/batch/'state.h5',on/batch/'state.h5')
    for name in ('velocity_slice','q_surface','instantaneous_streamlines'):
        assert (on/f'outdat/render/{name}.step00000002.jpeg').is_file(),name
    for rank in range(2):
        record=json.loads((on/f'outdat/render/mesh_step00000002_rank{rank}.json').read_text())
        assert all(item['image']=='published' for item in record['products'].values())


@pytest.mark.skipif(not EXE,reason='Set ASTR_HCW_INSITU_EXE')
def test_hcw_render_memcheck(tmp_path):
    launch(tmp_path/'on',2,True,memcheck=True)


@pytest.mark.skipif(not EXE,reason='Set ASTR_HCW_INSITU_EXE')
def test_hcw_render_exact_restart(tmp_path):
    full=launch(tmp_path/'full',2,True,steps=4)
    part=launch(tmp_path/'part',2,True,steps=2)
    source=part/'outdat/new/checkpoints/step000000000002'
    tail=launch(tmp_path/'tail',2,True,steps=4,restore=source)
    final=Path('outdat/new/checkpoints/step000000000004')
    compare_fields(full/final/'state.h5',tail/final/'state.h5')
    for name in ('velocity_slice','q_surface','instantaneous_streamlines'):
        frame=f'outdat/render/{name}.step00000004.jpeg'
        with Image.open(full/frame) as a, Image.open(tail/frame) as b:
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
