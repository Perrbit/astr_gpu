"""AIR5 wall Reynolds scalars: clipped time weights, resident GPU moments, exact restart."""
import re
import shutil
import h5py
import numpy as np
import pytest

from run_output_air5_restart_validation import launch, DT
from run_output_restart_validation import compare_fields
from test_insitu_air5_walls import arguments, TOPOLOGIES, FINAL, FIELD_SCALES, FIELD_NAMES
from test_output_insitu_restart import configuration
from test_insitu_channel_statistics import reference as channel_reference
from run_insitu_gpu_derivatives import check_resources
from test_insitu_air5_wall_render import PRODUCTS,FIELDS
from test_insitu_products import compare_geometry
from test_output_series_repair import seal_file_records,repair


def config(render=False):
    return configuration(statistics=True,render=render,interval=2).replace(
        'statistics_window=0.0005,0.0115','statistics_window=5.d-11,3.5d-10').replace(
        "schedule_mode='steps'","products='air5_walls',schedule_mode='steps'")


def read_statistics(path):
    with path.open('rb') as stream:
        assert stream.read(8) == b'ASTRWS01'
        header=np.fromfile(stream,'<i4',11)
        meta=np.fromfile(stream,'<f8',4)
        samples=int(np.fromfile(stream,'<i8',1)[0])
        shape=tuple(header[3:6]); count=int(np.prod(shape)); fields=int(header[9])
        xyz=np.fromfile(stream,'<f8',3*count).reshape((*shape,3),order='F')
        values=np.fromfile(stream,'<f8',3*fields*count).reshape((*shape,3*fields),order='F')
        owned=np.fromfile(stream,'<i4',count).reshape(shape,order='F')
        assert not stream.read(1)
    assert np.isfinite(meta).all() and np.isfinite(xyz).all() and np.isfinite(values).all()
    assert np.isin(owned,[0,1]).all()
    assert (values[...,fields:2*fields]>=0).all()
    np.testing.assert_allclose(values[...,2*fields:]**2,values[...,fields:2*fields],rtol=5e-16,atol=0)
    return header,meta,samples,xyz,values,owned


@pytest.fixture(scope='module',params=TOPOLOGIES,ids=('single','x','y','z'))
def reference(request,tmp_path_factory):
    topology=request.param; ranks=int(np.prod(topology))
    args=arguments(tmp_path_factory.mktemp('air5_scalar_statistics'),topology)
    args.initial_restart=True
    cases={}
    for backend in ('cpu','gpu'):
        cases[backend],_=launch(args,backend,ranks,'reference',4,interval=1,
            insitu_config=config(),directory_budget_bytes=256*1024**2)
    return args,topology,ranks,cases


def test_air5_scalar_statistics(reference,record_property):
    _,topology,ranks,cases=reference
    worst_mean=worst_variance=rms_raw=0.
    covered=0
    for rank in range(ranks):
        name=f'outdat/render/sample.wall_statistics.step00000004.rank{rank:08d}.bin'
        ch,cm,cs,cx,c,co=read_statistics(cases['cpu']/name)
        gh,gm,gs,gx,g,go=read_statistics(cases['gpu']/name)
        np.testing.assert_array_equal(ch,gh); np.testing.assert_array_equal(cm,gm)
        np.testing.assert_array_equal(cx,gx); np.testing.assert_array_equal(co,go)
        assert cs==gs==5
        np.testing.assert_allclose(cm,[4*DT,.5*DT,3.5*DT,3*DT],atol=1e-25,rtol=0)
        if c.size:
            dm=np.max(abs(c[...,:18]-g[...,:18]),axis=(0,1,2))
            dv=np.max(abs(c[...,18:36]-g[...,18:36]),axis=(0,1,2))
            dr=np.max(abs(c[...,36:]-g[...,36:]),axis=(0,1,2))
            worst_mean=max(worst_mean,float(np.max(dm/FIELD_SCALES)))
            worst_variance=max(worst_variance,float(np.max(dv/FIELD_SCALES**2)))
            rms_raw=max(rms_raw,float(np.max(abs(c[...,36:]-g[...,36:]))))
            for n,name in enumerate(FIELD_NAMES):
                record_property(f'rank{rank}_{name}_mean_SI_maxabs',float(dm[n]))
                record_property(f'rank{rank}_{name}_variance_SI_maxabs',float(dv[n]))
                record_property(f'rank{rank}_{name}_rms_SI_maxabs',float(dr[n]))
        covered+=int(co.sum())
    assert covered==272
    assert worst_mean<=2e-10 and worst_variance<=2e-10
    record_property('mean_reference_scaled_maxabs',worst_mean)
    record_property('variance_reference_scaled_maxabs',worst_variance)
    record_property('rms_raw_SI_maxabs',rms_raw)
    record_property('topology',topology)


def test_air5_scalar_independent_time_weights(reference,tmp_path,record_property):
    _,topology,ranks,cases=reference
    # Integrate frozen endpoint samples, not a second fluid/chemistry solver.
    endpoint_weights=np.zeros(5)
    for n in range(4):
        a=max(n*DT,.5*DT); b=min((n+1)*DT,3.5*DT)
        if b<=a:
            continue
        alpha=(a-n*DT)/DT; beta=(b-n*DT)/DT
        endpoint_weights[n]+=(b-a)*(1-.5*(alpha+beta))
        endpoint_weights[n+1]+=(b-a)*.5*(alpha+beta)
    worst=0.
    for backend,case in cases.items():
        args=arguments(tmp_path/backend,topology); args.initial_restart=True
        early={}
        for end in (1,2):
            early[end],_=launch(args,backend,ranks,f'endpoint{end}',end,interval=1,
                insitu_config=config(),directory_budget_bytes=256*1024**2)
        raw=[]
        for step in range(5):
            source=early[1] if step<2 else early[2] if step==2 else case
            with h5py.File(source/f'outdat/new/checkpoints/step{step:012d}/statistics.h5') as state:
                raw.append(np.concatenate([np.stack([state[f'q{34*g+c:04d}'][...].transpose(2,1,0)[:,0,:]
                    for c in (5,6,7)],axis=-1) for g in range(6)],axis=-1))
        raw=np.stack(raw)
        mean=np.einsum('t,tikf->ikf',endpoint_weights,raw)/endpoint_weights.sum()
        variance=np.einsum('t,tikf->ikf',endpoint_weights,(raw-mean)**2)/endpoint_weights.sum()
        for rank in range(ranks):
            h,_,_,_,actual,_=read_statistics(case/f'outdat/render/sample.wall_statistics.step00000004.rank{rank:08d}.bin')
            if not actual.size:
                continue
            i,k=int(h[6]),int(h[8]); nx,nz=map(int,h[3:5])
            expected=mean[i:i+nx,k:k+nz]
            error=float(np.max(abs(actual[:,:,0,:18]-expected)/FIELD_SCALES))
            worst=max(worst,error)
            assert error<=2e-10
            assert np.max(abs(actual[:,:,0,18:36]-variance[i:i+nx,k:k+nz])/FIELD_SCALES**2)<=2e-10
        # Previous primitive fields must have the completed checkpoint's own phase.
        with h5py.File(case/FINAL/'statistics.h5') as stats,h5py.File(case/FINAL/'state.h5') as flow:
            indices=(23,24,25,26,28,29,27,30,31,32,33,34)
            for n,index in enumerate(indices):
                previous=stats[f'q{34*(n//3)+5+n%3:04d}'][...][:,0,:]
                np.testing.assert_allclose(previous,flow[f'q{index:04d}'][...][:,0,:],atol=2e-10,rtol=0)
    record_property('endpoint_oracle_reference_scaled_maxabs',worst)


@pytest.mark.parametrize('backend',('cpu','gpu'))
def test_air5_scalar_exact_restart(reference,backend,tmp_path):
    _,topology,ranks,cases=reference
    args=arguments(tmp_path,topology); args.initial_restart=True
    source=cases[backend]/'outdat/new/checkpoints/step000000000003'
    before={p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
    resumed,_=launch(args,backend,ranks,'restart',4,restore=source,interval=1,
        insitu_config=config(),directory_budget_bytes=256*1024**2)
    for name in ('state.h5','statistics.h5'):
        compare_fields(cases[backend]/FINAL/name,resumed/FINAL/name)
    for name in ('control.bin','insitu_control.bin'):
        assert (cases[backend]/FINAL/name).read_bytes()==(resumed/FINAL/name).read_bytes()
    for rank in range(ranks):
        name=f'outdat/render/sample.wall_statistics.step00000004.rank{rank:08d}.bin'
        assert (cases[backend]/name).read_bytes()==(resumed/name).read_bytes()
    assert before=={p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}


def test_air5_scalar_output_isolation(reference,tmp_path):
    _,topology,ranks,cases=reference
    for backend in ('cpu','gpu'):
        plain,_=launch(arguments(tmp_path/backend,topology),backend,ranks,'plain',4,interval=1)
        compare_fields(cases[backend]/FINAL/'state.h5',plain/FINAL/'state.h5')
    records=re.findall(r'ASTR_INSITU_WALL_STATS rank=(\d+) step=(\d+) samples=(\d+) field_download_bytes=(\d+)',
                       (cases['gpu']/'run.log').read_text())
    assert len(records)==5*ranks
    if topology==(1,2,1):
        assert all(int(size)==0 for rank,_,_,size in records if rank=='1')
    assert 'ASTR_INSITU_TRANSFER_TIMING' not in (cases['gpu']/'run.log').read_text()


def test_channel_wall_scalar_statistics(channel_reference,record_property):
    topology,ranks,cases=channel_reference
    worst=0.
    for rank in range(ranks):
        name=f'outdat/render/sample.wall_statistics.step00000004.rank{rank:08d}.bin'
        a=read_statistics(cases['cpu']/name); b=read_statistics(cases['gpu']/name)
        np.testing.assert_array_equal(a[0],b[0]); np.testing.assert_array_equal(a[1],b[1])
        np.testing.assert_allclose(a[4][...,:6],b[4][...,:6],rtol=0,atol=2e-10)
        worst=max(worst,float(np.max(abs(a[4][...,:6]-b[4][...,:6]))))
        assert a[2]==b[2]==5
        for backend,case in cases.items():
            _,_,_,_,actual,_=a if backend=='cpu' else b
            with h5py.File(case/FINAL/'statistics.h5') as state:
                assert 'wall_statistics' in state
                group=state['wall_statistics']
                assert group['identity'][6]==34
                for w in range(actual.shape[2]):
                    j_global=0 if (int(a[0][7])==0 and w==0) else 16
                    fields=np.stack([group[f'q{c:04d}'][...].transpose(2,1,0)[:,j_global,:] for c in (5,6,7)],axis=-1)
                    i,k=int(a[0][6]),int(a[0][8]); nx,nz=map(int,a[0][3:5])
                    assert np.isfinite(fields[i:i+nx,k:k+nz]).all()
                    endpoints=[]
                    for step in (3,4):
                        with h5py.File(case/f'outdat/new/checkpoints/step{step:012d}/statistics.h5') as checkpoint:
                            endpoints.append(np.stack([checkpoint['wall_statistics'][f'q{c:04d}'][...].transpose(2,1,0)
                                [i:i+nx,j_global,k:k+nz] for c in (5,6,7)],axis=-1))
                    np.testing.assert_allclose(actual[:,:,w,:3],.5*(endpoints[0]+endpoints[1]),atol=2e-10,rtol=0)
                    np.testing.assert_allclose(actual[:,:,w,3:6],.25*(endpoints[1]-endpoints[0])**2,atol=2e-10,rtol=0)
    record_property('channel_scalar_maxabs',worst)
    record_property('topology',topology)


def test_air5_scalar_render_restart(tmp_path):
    topology=(1,2,1); ranks=2
    args=arguments(tmp_path,topology)
    plain,_=launch(args,'gpu',ranks,'plain',4,interval=1,insitu_config=config(),
        directory_budget_bytes=256*1024**2)
    rendered,size=launch(args,'gpu',ranks,'render',4,interval=1,insitu_config=config(render=True),
        directory_budget_bytes=256*1024**2)
    check_resources(rendered,ranks,steps=(2,4),capture=False,dt=DT)
    for name in ('state.h5','statistics.h5'):
        compare_fields(plain/FINAL/name,rendered/FINAL/name)
    source=rendered/'outdat/new/checkpoints/step000000000003'
    resumed,_=launch(args,'gpu',ranks,'restart_render',4,restore=source,interval=1,
        insitu_config=config(render=True),directory_budget_bytes=256*1024**2)
    check_resources(resumed,ranks,steps=(4,),capture=False,dt=DT)
    for name in ('state.h5','statistics.h5'):
        compare_fields(rendered/FINAL/name,resumed/FINAL/name)
    for name in ('control.bin','insitu_control.bin'):
        assert (rendered/FINAL/name).read_bytes()==(resumed/FINAL/name).read_bytes()
    for product in PRODUCTS:
        base=f'outdat/render/{product}.step00000004'
        compare_geometry(rendered/(base+'.pvtp'),resumed/(base+'.pvtp'),FIELDS)
        for ext in ('jpeg','eps'):
            assert (rendered/(base+'.'+ext)).read_bytes()==(resumed/(base+'.'+ext)).read_bytes()
    for rank in range(ranks):
        name=f'outdat/render/sample.wall_statistics.step00000004.rank{rank:08d}.bin'
        assert (rendered/name).read_bytes()==(resumed/name).read_bytes()
    assert size<=256*1024**2


def test_air5_scalar_memory_safety(tmp_path):
    case,_=launch(arguments(tmp_path,(1,2,1)),'gpu',2,'memcheck',2,interval=1,
        insitu_config=config(),memcheck=True,directory_budget_bytes=256*1024**2)
    logs=list(case.glob('memcheck.*.log'))
    assert len(logs)==2
    assert all('ERROR SUMMARY: 0 errors' in p.read_text() for p in logs)
    assert 'rank=1 step=2 samples=3 field_download_bytes=0' in (case/'run.log').read_text()


@pytest.mark.parametrize('backend',('cpu','gpu'))
@pytest.mark.parametrize('defect',('metadata','nonfinite','off_wall'))
def test_air5_scalar_corrupt_state_rejected(tmp_path,backend,defect):
    args=arguments(tmp_path,(1,1,1))
    case,_=launch(args,backend,1,'source',1,interval=1,insitu_config=config())
    source=case/'outdat/new/checkpoints/step000000000001'
    root=tmp_path/'modified_source'
    shutil.copytree(source.parent.parent/'resources',root/'resources')
    damaged=root/'checkpoints'/source.name; shutil.copytree(source,damaged)
    with h5py.File(damaged/'statistics.h5','r+') as state:
        if defect=='metadata':
            state['metadata'][0]=99
        else:
            field=state['q0005']; field[0,0 if defect=='nonfinite' else 1,0]=np.nan if defect=='nonfinite' else 1.
    names=[line.split()[0] for line in (damaged/'MANIFEST').read_text().splitlines()[2:]]
    (damaged/'MANIFEST').write_text(seal_file_records(damaged,names,'ASTR_CHECKPOINT_BUNDLE 1'))
    size,crc=repair.fingerprint(damaged/'MANIFEST')
    (damaged/'COMPLETE').write_text(f'ASTR_COMPLETE_1 {size} {crc:016X}\n')
    before={p.name:p.read_bytes() for p in damaged.iterdir() if p.is_file()}
    message={'metadata':'wall scalar metadata mismatch','nonfinite':'invalid packed wall scalar state',
             'off_wall':'nonzero off-wall scalar checkpoint state'}[defect]
    launch(args,backend,1,'rejected',2,restore=damaged,interval=1,insitu_config=config(),reject=message)
    assert before=={p.name:p.read_bytes() for p in damaged.iterdir() if p.is_file()}


@pytest.mark.parametrize('kind,message',(
    ('host','statistics host allocation exceeds node budget'),
    ('device','statistics device allocation exceeds budget or free-memory headroom'),
    ('reserve','wall GPU budget'),
))
def test_air5_scalar_budget_rejected(tmp_path,kind,message):
    text=config()
    change={'host':('host_budget_bytes=4294967296','host_budget_bytes=1'),
            'device':('device_budget_bytes=2147483648','device_budget_bytes=131072'),
            'reserve':('device_reserve_bytes=1073741824','device_reserve_bytes=1152921504606846976')}[kind]
    text=text.replace(*change)
    launch(arguments(tmp_path,(1,2,1)),'gpu',2,'rejected',1,interval=1,
        insitu_config=text,reject=message)
