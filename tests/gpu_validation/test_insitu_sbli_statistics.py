"""TB1 quantitative products in the existing small GRCBC incident-shock fixture."""
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import subprocess
import shutil

import h5py
import numpy as np
import pytest

from run_m12_mp7_nscbc_performance import runtime,mpi_command
from run_output_restart_validation import compare_fields
from test_sbli_joint_nscbc import prepare,DT,MPIEXEC
from test_insitu_selected_runtime import configuration
from test_insitu_station_profiles import FIELDS,check_profiles,read_profile
from test_insitu_wall_transport_runtime import WALL,check_wall_means,derivative_weights
from test_insitu_wall_unit_profiles import check_wall_unit_profiles,final_profile
from test_insitu_thickness_runtime import integral,metadata_values
from test_index_slice_runtime import product,segments,frames

EXE=os.environ.get('ASTR_SELECTED_RUNTIME_EXE')
pytestmark=pytest.mark.skipif(not EXE,reason='Select the root-CMake FP64 CPU/GPU build')
WINDOW=(.5*DT,11.5*DT)
TOPOLOGIES=((1,1,1),(2,1,1),(1,2,1),(1,1,2),(2,2,1),(2,1,2),(1,2,2))


def definitions():
    raw=product('raw',FIELDS)
    raw['sampling']=dict(mode='grid_index',axis='i',index=24,index_ranges=dict(j=[0,48],k=[0,16]))
    raw['schedule']['interval']=1
    wall=product('raw-wall',['density','temperature',*WALL])
    wall['sampling']=dict(mode='grid_index',axis='j',index=0,index_ranges=dict(i=[0,64],k=[0,16]))
    wall['schedule']['interval']=1
    native_wall=dict(id='mean-wall',kind='wall_profile',enable_request=1,source='reynolds',
                     statistics_source='selected-time',fields=wall['fields'],
                     sampling=dict(mode='wall_span',axis='j',index=0,index_ranges=dict(i=[0,64],k=[0,16])),
                     output=dict(channel='data',format='csv'),schedule=dict(mode='steps',interval=1))
    reference=dict(deepcopy(native_wall),id='mean-wall-reference',wall_reference=dict(recovery_temperature=1.72))
    records=[raw,wall,native_wall,reference]
    for source in ('reynolds','favre'):
        native=dict(id=source,kind='station_profile',enable_request=1,source=source,
                    statistics_source='selected-time',fields=FIELDS[1:],
                    sampling=dict(mode='station_span',axis='i',index=24,index_ranges=dict(j=[0,48],k=[0,16])),
                    output=dict(channel='data',format='csv'),schedule=dict(mode='steps',interval=1))
        records.extend([native,dict(deepcopy(native),id=source+'-plus',wall_units='mean-wall-reference')])
    records.append(dict(deepcopy(records[-2]),id='layer',thickness=dict(edge_band=[40,44],explicit_upper=.6)))
    return records


def launch(case,backend,topology,*,steps=12,restore=None,checkpoint=False,records=None,statistics=True,
           monitor=False,resource_baseline=None,private_transport='device-aware',insitu_config=None,executable=None,memcheck=False,
           unrestricted=False,catalog_replay=None,timing=True):
    if memcheck and (monitor or backend!='gpu'):
        raise ValueError('memory checking requires a separate GPU run')
    assert not case.exists(),'the preparer must never replace an existing evidence directory'
    prepare(case,backend,'743e',steps,restore,checkpoint=checkpoint,unrestricted=unrestricted)
    (case/'input.products.jsonl').write_text('\n'.join(json.dumps(x) for x in
        [{'catalog_schema':'astr-insitu-products-v1'},*(definitions() if records is None else records)])+'\n')
    if catalog_replay is not None:
        path=case/'datin/input.output'
        path.write_text(path.read_text().replace("restart_output='saved',","restart_output='saved',product_poll_steps=1,"))
    # Selected wall statistics need aware private exchanges; solver halos stay pinned.
    env=runtime(','.join(map(str,topology)),transport=private_transport)
    if catalog_replay is not None:
        library,publications=catalog_replay
        directory=case/'catalog_replay';directory.mkdir()
        for poll,items in publications.items():
            (directory/f'catalog{poll:06d}.jsonl').write_text('\n'.join(json.dumps(row) for row in
                [{'catalog_schema':'astr-insitu-products-v1'},*items])+'\n')
        env.update(LD_PRELOAD=str(Path(library).resolve()),ASTR_TEST_CATALOG_FILE='input.products.jsonl',
            ASTR_TEST_CATALOG_REPLAY=str(directory.resolve()))
    env['ASTR_GPU_HALO_TRANSPORT']='pinned'
    if insitu_config is not None:
        env['OMPI_MCA_osc']='sm,pt2pt'
        for key in ('DISPLAY','PYTHONPATH','CATALYST_IMPLEMENTATION_PREFER_ENV','VTK_EGL_DEVICE_INDEX'):
            env.pop(key,None)
    env.pop('ASTR_WALL_BLOWING_MODE',None)
    env.update(ASTR_GPU_MP_POSITIVITY='off',ASTR_GPU_MP_SPLIT_CACHE='off',ASTR_INSITU_TIMING='1' if timing else '0',
               ASTR_NSCBC_FARFIELD_MODE='sbli_grcbc',ASTR_NSCBC_FARFIELD_SHOCK_X='1',
               ASTR_NSCBC_FARFIELD_SHOCK_ANGLE_DEG='32.6',ASTR_NSCBC_FARFIELD_RHO='1',
               ASTR_NSCBC_FARFIELD_U='1',ASTR_NSCBC_FARFIELD_V='0',ASTR_NSCBC_FARFIELD_W='0',
               ASTR_NSCBC_FARFIELD_T='1')
    if statistics:
        config=configuration(WINDOW) if insitu_config is None else insitu_config
        if memcheck and insitu_config is None:
            # Declare the independent device statistics transport so the existing pre-MPI binder sees it.
            config=config.replace('render=f,',"render=f,derivative_backend='gpu',processing_backend='device',postprocess_transport='device-aware',")
        (case/'datin/input.insitu').write_text(config)
        env['ASTR_INSITU_CONFIG']='datin/input.insitu'
    from named_fixture import prepare_named_run
    solver=[str(Path(EXE if executable is None else executable).resolve()),'run',
        prepare_named_run(case,'datin/input.flatplate',env)]
    if memcheck:
        sanitizer=shutil.which('compute-sanitizer')
        assert sanitizer,'Compute Sanitizer is required for this explicit gate'
        solver=[sanitizer,'--tool','memcheck','--target-processes','all','--error-exitcode','99',
                '--log-file',str(case/'memcheck.%p.log'),
                *(['--suppressions',str(Path(__file__).resolve().parent/'compute_sanitizer_ucx_cuda_aware.supp.xml')]
                  if private_transport=='device-aware' else []),*solver]
    command=[*mpi_command(MPIEXEC,int(np.prod(topology)),private_transport),*solver]
    (case/'launch.json').write_text(json.dumps(dict(command=command,environment={k:v for k,v in env.items()
        if k.startswith(('ASTR_','OMPI_MCA_','UCX_'))}),indent=2)+'\n')
    with (case/'run.log').open('w') as log:
        if monitor:
            from insitu_resource_monitor import run_monitored
            limits=dict(device_extra_budget_bytes=None,host_extra_budget_bytes=None,device_reserve_bytes=0,timeout_seconds=None) if unrestricted else {}
            run_monitored(command,case,env,log,case/'resources.sampled.json',baseline=resource_baseline,**limits)
            code=0
        else:
            code=subprocess.run(command,cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=None if unrestricted else 180).returncode
    text=(case/'run.log').read_text()
    assert code==0 and 'The job is done!' in text,text[-7000:]
    assert 'complete-RHS closure, top transverse filter off' in text
    assert 'sbli-grcbc-h1 high-order boundary chain' in text
    return case


def compare(actual,expected):
    a,b=np.asarray(actual),np.asarray(expected)
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    error=np.abs(a-b);scaled=error/np.maximum(1.,np.maximum(np.abs(a),np.abs(b)))
    assert scaled.max()<=2e-10,(scaled.max(),error.max())
    return float(scaled.max()),float(error.max())


def check_native_frame_format(case):
    for name in ('reynolds','favre'):
        paths=list((case/'outdat/new/profile_products'/name).glob('*/segment*/frame.step*.csv'))
        assert paths,'native station frames are required'
        for path in paths:
            data,_=read_profile(path)
            bad=[field for field in data.dtype.names if not np.isfinite(data[field]).all()]
            assert not bad,(path,bad,'every native numeric cell must be a portable finite CSV number')


def check(case,record_property):
    state=json.loads((case/'outdat/new/products.status.json').read_text())
    assert all(p['state']=='active' for p in state['products']),state
    check_native_frame_format(case)
    records=definitions()
    native=[p for p in records if p['id'] in ('raw','reynolds','favre')]
    check_profiles(case,native,record_property,steps=12,dt=DT,window=WINDOW,ny=49,nk=16)
    check_wall_means(case,[records[1],records[2]],record_property,steps=12,dt=DT,window=WINDOW,ni=65,nk=16)
    check_wall_unit_profiles(case,records,record_property,steps=12,expected_histories=2)
    wall,_=final_profile(case,'mean-wall','wall_profile_products',steps=12)
    normalized,_=final_profile(case,'mean-wall-reference','wall_profile_products',steps=12)
    for name,expected in [('Cf',2*wall['wall_shear_x']),('Cq_inf',wall['wall_heat_into_wall']),
                           ('St_stat',wall['wall_heat_into_wall']/((1/(.4*2**2))*(1.72-wall['temperature'])))]:
        compare(normalized[name],expected)
    native,_=final_profile(case,'favre',steps=12)
    reynolds,_=final_profile(case,'reynolds',steps=12)
    layer,metadata=final_profile(case,'layer',steps=12)
    for name in native.dtype.names:
        np.testing.assert_array_equal(layer[name],native[name])
    info=metadata_values(metadata);y=native['y'];rho=native['rho_time_mean'];u=native['velocity_x']
    band=slice(40,45);length=y[44]-y[40]
    rho_e=np.trapezoid(rho[band],y[band])/length
    u_e=np.trapezoid((rho*u)[band],y[band])/(rho_e*length)
    te=np.trapezoid(reynolds['temperature'][band],y[band])/length
    pe=np.trapezoid(reynolds['pressure'][band],y[band])/length
    s=110.3/148.9;mu=te*np.sqrt(te)*(1+s)/(te+s)/296000
    compare(np.array([info[k] for k in ('rho_edge','U_edge','T_edge','p_edge','mu_edge')]),np.array([rho_e,u_e,te,pe,mu]))
    f=rho*u/(rho_e*u_e);d,theta=1-f,f*(1-u/u_e)
    for name,values in [('cumulative_displacement',d),('cumulative_momentum',theta)]:
        compare(layer[name],np.r_[0.,np.cumsum(.5*np.diff(y)*(values[1:]+values[:-1]))])
    delta_star,momentum=integral(y,d,.6),integral(y,theta,.6)
    compare(np.array([info[k] for k in ('displacement_explicit','momentum_explicit','H_explicit','Re_theta_explicit')]),
            np.array([delta_star,momentum,delta_star/momentum,rho_e*u_e*momentum/mu]))
    above=u>=.99*u_e
    assert np.all(above[40:45]),'the test reference band must lie on its external velocity plateau'
    left=40
    while left>0 and above[left-1]:left-=1
    assert left>0
    upper=y[left-1]+(y[left]-y[left-1])*(.99*u_e-u[left-1])/(u[left]-u[left-1])
    compare(np.array([info[k] for k in ('delta99','displacement99','momentum99')]),
            np.array([upper-y[0],integral(y,d,upper),integral(y,theta,upper)]))


def check_wall_oracle(case,record_property):
    final=case/'outdat/new/checkpoints/step000000000012/state.h5'
    with h5py.File(final) as data:
        q=np.stack([data[f'q{c:04d}'][...] for c in range(1,6)])
    rho=q[0];velocity=q[1:4]/rho
    pressure=.4*(q[4]-.5*np.sum(q[1:4]*velocity,axis=0))
    temperature=pressure/rho*5.6
    values=np.concatenate([velocity,temperature[None]])
    values[:,-1]=values[:,0]
    weights=derivative_weights();i=np.arange(65);first=np.clip(i-3,0,58);si=first[:,None]+np.arange(7)
    sk=(np.arange(17)[:,None]-3+np.arange(7))%16
    directional=np.empty((4,3,17,65));metric=np.empty((3,3,17,65))
    with h5py.File(case/'outdat/new/resources/geometry.h5') as geometry:
        for d in range(3):
            for p in range(3):metric[d,p]=geometry[f'q{5+d+3*p:04d}'][:,0,:]
    for c,v in enumerate(values):
        directional[c,0]=np.einsum('kis,is->ki',v[:,0,si],weights[i-first])
        directional[c,1]=np.einsum('kji,j->ki',v[:,:7,:],weights[0])
        directional[c,2]=np.einsum('ksi,s->ki',v[sk,0,:],weights[3])
    gradient=np.einsum('cdki,dpki->cpki',directional,metric)
    t=temperature[:,0,:].copy();t[-1]=t[0]
    s=110.3/148.9;mu=t*np.sqrt(t)*(1+s)/(t+s)/296000
    expected=dict(wall_shear_x=mu*(gradient[0,1]+gradient[1,0]),
                  wall_shear_z=mu*(gradient[2,1]+gradient[1,2]),
                  wall_heat_into_wall=(mu/.72)/1.6*gradient[3,1],dynamic_viscosity=mu)
    raw=next(p for p in segments(case) if p.parent.parent.name=='raw-wall')
    with h5py.File(frames(raw)[-1]) as data:
        worst=absolute=0.
        for name,target in expected.items():
            scaled,error=compare(data[name][...],target)
            worst,absolute=max(worst,scaled),max(absolute,error)
    record_property('wall_oracle_max_scaled_error',worst)
    record_property('wall_oracle_max_absolute_error',absolute)


@pytest.fixture(scope='module')
def continuous_reference(tmp_path_factory):
    cases={}
    def get(backend):
        if backend not in cases:
            saved=os.environ.get('ASTR_SBLI_STATISTICS_REFERENCE_ROOT')
            if saved:
                matches=list(Path(saved).glob('sbli_np2z_'+backend+'[0-9]*/'+backend))
                assert len(matches)==1,'an explicit immutable continuous reference is required'
                cases[backend]=matches[0]
                assert (cases[backend]/'outdat/new/checkpoints/step000000000012/COMPLETE').is_file()
                assert 'The job is done!' in (cases[backend]/'run.log').read_text()
            else:
                root=tmp_path_factory.mktemp('sbli_np2z_'+backend)
                cases[backend]=launch(root/backend,backend,(1,1,2),checkpoint=True)
        return cases[backend]
    return get


@pytest.mark.parametrize('backend',('cpu','gpu'))
@pytest.mark.parametrize('topology',TOPOLOGIES)
def test_real_sbli_quantitative_products(tmp_path,backend,topology,record_property,request):
    case=(request.getfixturevalue('continuous_reference')(backend) if topology==(1,1,2) else
          launch(tmp_path/backend,backend,topology,checkpoint=topology==(1,1,1)))
    check(case,record_property)
    if topology==(1,1,1):check_wall_oracle(case,record_property)


@pytest.mark.parametrize('backend',('cpu','gpu'))
def test_sbli_statistics_exact_continuation(tmp_path,backend,continuous_reference):
    full=continuous_reference(backend)
    first=launch(tmp_path/'prefix',backend,(1,1,2),steps=5,checkpoint=True)
    changed=definitions()
    changed[-1]['thickness']['edge_band']=[1,1]
    tail=launch(tmp_path/'tail',backend,(1,1,2),restore=first/'outdat/new/checkpoints/step000000000005',
                checkpoint=True,records=changed)
    final=Path('outdat/new/checkpoints/step000000000012')
    for name in ('state.h5','statistics.h5'):
        compare_fields(full/final/name,tail/final/name)
    assert (full/final/'control.bin').read_bytes()==(tail/final/'control.bin').read_bytes()
    for family in ('profile_products','wall_profile_products'):
        a={p.parent.parent.parent.name+'/'+p.name:p for p in (full/'outdat/new'/family).glob('*/*/segment*/frame*.csv')}
        for p in (tail/'outdat/new'/family).glob('*/*/segment*/frame*.csv'):
            assert p.read_bytes()==a[p.parent.parent.parent.name+'/'+p.name].read_bytes()


@pytest.mark.parametrize('backend',('cpu','gpu'))
def test_sbli_statistics_preserve_authoritative_state(tmp_path,backend,continuous_reference):
    full=continuous_reference(backend)
    without=launch(tmp_path/'without',backend,(1,1,2),checkpoint=True,records=[],statistics=False)
    final=Path('outdat/new/checkpoints/step000000000012/state.h5')
    compare_fields(full/final,without/final)
    assert not (without/'outdat/new/checkpoints/step000000000012/statistics.h5').exists()


def test_sbli_station_csv_keeps_tiny_finite_covariance(tmp_path):
    case=launch(tmp_path/'cpu','cpu',(1,1,1),checkpoint=False)
    check_native_frame_format(case)
    path=next((case/'outdat/new/profile_products/favre').glob('*/segment*/frame.step000000000001.csv'))
    values,_=read_profile(path)
    tiny=values['covariance_zz']
    assert np.any((np.abs(tiny)>0)&(np.abs(tiny)<1.e-100)), 'retain the actual tiny-covariance regression trigger'


def test_sbli_products_cost_without_oracle_io(tmp_path,continuous_reference):
    off=launch(tmp_path/'off','gpu',(1,1,2),records=[],statistics=False,monitor=True)
    baseline=json.loads((off/'resources.sampled.json').read_text())
    on=launch(tmp_path/'on','gpu',(1,1,2),records=definitions()[2:],monitor=True,resource_baseline=baseline)
    reference=continuous_reference('gpu')
    for family in ('profile_products','wall_profile_products'):
        expected={p.parent.parent.parent.name+'/'+p.name:p for p in (reference/'outdat/new'/family).glob('*/*/segment*/frame*.csv')}
        actual=list((on/'outdat/new'/family).glob('*/*/segment*/frame*.csv'))
        assert actual
        for p in actual:
            assert p.read_bytes()==expected[p.parent.parent.parent.name+'/'+p.name].read_bytes()
    result=dict(scope='one short local GPU NP2 z window; not a stable performance benchmark',cases={})
    for name,case in [('off',off),('on',on)]:
        assert not list((case/'outdat').rglob('*.h5')),'oracle archives and checkpoints must be absent'
        values={}
        for line in (case/'run.log').read_text().splitlines():
            match=re.fullmatch(r'ASTR_INSITU_STAGE_TIMING completed_step_inclusive (\d+) (\d+)\s+(\S+)',line)
            if match:
                key=tuple(map(int,match.group(1,2)))
                assert key not in values
                values[key]=float(match[3])
        assert set(values)=={(step,rank) for step in range(1,13) for rank in range(2)}
        assert all(np.isfinite(v) and v>=0 for v in values.values())
        result['cases'][name]=dict(
            all_12_steps_max_rank_sum_seconds=max(sum(values[s,r] for s in range(1,13)) for r in range(2)),
            after_first_11_steps_max_rank_sum_seconds=max(sum(values[s,r] for s in range(2,13)) for r in range(2)),
            largest_step_rank_seconds=max(values.values()),
            memory=json.loads((case/'resources.sampled.json').read_text()))
    result['all_steps_ratio_on_over_off']=(result['cases']['on']['all_12_steps_max_rank_sum_seconds']/
                                         result['cases']['off']['all_12_steps_max_rank_sum_seconds'])
    (tmp_path/'cost.json').write_text(json.dumps(result,indent=2)+'\n')


def test_sbli_unavailable_private_transport_is_product_local(tmp_path):
    case=launch(tmp_path/'gpu','gpu',(2,1,1),steps=2,private_transport='pinned')
    state=json.loads((case/'outdat/new/products.status.json').read_text())
    products={p['id']:p for p in state['products']}
    assert products['mean-wall']['state']==products['mean-wall-reference']['state']=='resource-disabled'
    for name in ('raw','raw-wall','reynolds','favre','layer'):
        assert products[name]['state']=='active' and products[name]['last_output_step']==2
    assert 'CUDA-aware MPI is unavailable' in (case/'run.log').read_text()
    for name in ('reynolds-plus','favre-plus'):
        path=next((case/'outdat/new/profile_products'/name).glob('*/segment*/frame.step000000000002.csv'))
        values,_=read_profile(path)
        assert np.all(values['wall_scale_available']==0) and np.isnan(values['y_plus']).all()
        assert np.isfinite(values['velocity_x']).all()
