"""AIR5 completed-cache moments, open-face nodal measures and exact continuation."""
import re
import csv
import shutil
import h5py
import numpy as np
import pytest

from run_output_air5_restart_validation import launch,DT
from run_output_restart_validation import compare_fields
from test_insitu_air5_walls import arguments,TOPOLOGIES,FINAL,reference_scales
from test_insitu_wall_scalar_statistics import config as wall_config
from test_output_series_repair import seal_file_records,repair

SCALAR_NAMES=('T','Tv','Y_N2','Y_O2','Y_N','Y_O','Y_NO')
SCALAR_SCALES=np.array([reference_scales()['temperature']]*2+[1.]*5)
WINDOW=np.array([3.1*DT,3.6*DT])


def config():
    return wall_config().replace('statistics_window=5.d-11,3.5d-10',
        'statistics_window=3.1d-10,3.6d-10,air5_volume_statistics=t,air5_volume_reduction=t')


def read_statistics(path):
    with path.open('rb') as stream:
        assert stream.read(8)==b'ASTRAS01'
        header=np.fromfile(stream,'<i4',10); shape=tuple(header[3:6]); count=int(np.prod(shape))
        clocks=np.fromfile(stream,'<f8',5)
        samples=int(np.fromfile(stream,'<i8',1)[0])
        regional=np.fromfile(stream,'<f8',34)
        xyz=np.fromfile(stream,'<f8',3*count).reshape((*shape,3),order='F')
        weights=np.fromfile(stream,'<f8',count).reshape(shape,order='F')
        moments=np.fromfile(stream,'<f8',62*count).reshape((*shape,62),order='F')
        assert not stream.read(1)
    assert all(np.isfinite(a).all() for a in (clocks,regional,xyz,weights,moments))
    assert (weights>0).all() and (moments[...,48:55]>=0).all()
    np.testing.assert_allclose(moments[...,55:]**2,moments[...,48:55],atol=0,rtol=5e-16)
    return header,clocks,samples,regional,xyz,weights,moments


@pytest.fixture(scope='module',params=TOPOLOGIES,ids=('single','x','y','z'))
def reference(request,tmp_path_factory):
    topology=request.param; ranks=int(np.prod(topology)); cases={}
    args=arguments(tmp_path_factory.mktemp('air5_volume_statistics'),topology)
    for backend in ('cpu','gpu'):
        cases[backend],_=launch(args,backend,ranks,'reference',4,interval=1,
            insitu_config=config(),directory_budget_bytes=256*1024**2)
    return args,topology,ranks,cases


def exported(case,ranks):
    return [read_statistics(case/f'outdat/render/sample.air5_statistics.step00000004.rank{r:08d}.bin')
            for r in range(ranks)]


def test_cpu_gpu_moments(reference,record_property):
    _,topology,ranks,cases=reference
    mean_error=variance_error=velocity_error=0.
    for a,b in zip(exported(cases['cpu'],ranks),exported(cases['gpu'],ranks)):
        for n in (0,4,5):
            np.testing.assert_array_equal(a[n],b[n])
        np.testing.assert_allclose(a[1],b[1],atol=2e-10,rtol=0)
        assert a[2]==b[2]==5
        x,y=a[6].copy(),b[6].copy()
        x[...,8:14] **= 2; y[...,8:14] **= 2
        velocity_error=max(velocity_error,float(np.max(abs(x[...,:41]-y[...,:41]))))
        np.testing.assert_allclose(x[...,:41],y[...,:41],atol=2e-10,rtol=0)
        dm=np.max(abs(x[...,41:48]-y[...,41:48]),axis=(0,1,2))
        dv=np.max(abs(x[...,48:55]-y[...,48:55]),axis=(0,1,2))
        dr=np.max(abs(a[6][...,55:]-b[6][...,55:]),axis=(0,1,2))
        mean_error=max(mean_error,float(np.max(dm/SCALAR_SCALES)))
        variance_error=max(variance_error,float(np.max(dv/SCALAR_SCALES**2)))
        for name,m,v,r in zip(SCALAR_NAMES,dm,dv,dr):
            record_property(name+'_mean_SI_maxabs',float(m))
            record_property(name+'_variance_SI_maxabs',float(v))
            record_property(name+'_rms_SI_maxabs',float(r))
        # Regional signal also uses the unchanged absolute gate.
        np.testing.assert_allclose(a[3],b[3],atol=2e-10,rtol=0)
    assert mean_error<=2e-10 and variance_error<=2e-10
    record_property('velocity_statistics_maxabs',velocity_error)
    record_property('scalar_mean_reference_scaled_maxabs',mean_error)
    record_property('scalar_variance_reference_scaled_maxabs',variance_error)
    record_property('topology',topology)


def test_unique_nodes_and_physical_measure(reference,record_property):
    _,_,ranks,cases=reference
    for backend,case in cases.items():
        nodes={}; measure=0.
        for h,clock,count,_,xyz,weights,moments in exported(case,ranks):
            np.testing.assert_allclose(clock[:4],[4*DT,*WINDOW,WINDOW[1]-WINDOW[0]],atol=1e-25,rtol=0)
            np.testing.assert_allclose(clock[4],.08*.01*.002,atol=2e-18,rtol=0)
            assert count==5
            for index in np.ndindex(tuple(h[3:6])):
                identity=tuple(np.array(index)+h[6:9])
                assert identity not in nodes
                nodes[identity]=(xyz[index],weights[index])
            measure+=weights.sum()
            np.testing.assert_allclose(moments[...,0],WINDOW[1]-WINDOW[0],atol=1e-25,rtol=0)
        assert len(nodes)==17*17*16
        assert set(nodes)==set(np.ndindex(17,17,16))
        axes=[np.array([nodes[tuple(n if d==axis else 0 for d in range(3))][0][axis]
                       for n in range(17 if axis<2 else 16)]) for axis in range(3)]
        expected=[]
        for axis,line in enumerate(axes):
            if axis<2:
                left=np.r_[line[0],line[:-1]]; right=np.r_[line[1:],line[-1]]
            else:
                left=np.r_[line[-1]-.002,line[:-1]]; right=np.r_[line[1:],line[0]+.002]
            expected.append(.5*(right-left))
        for identity,(_,weight) in nodes.items():
            target=np.prod([expected[d][identity[d]] for d in range(3)])
            np.testing.assert_allclose(weight,target,atol=2e-21,rtol=0)
        np.testing.assert_allclose(measure,.08*.01*.002,atol=2e-18,rtol=0)
        record_property(backend+'_physical_volume',float(measure))


def test_independent_endpoint_moments_and_own_phase(reference,record_property):
    _,_,ranks,cases=reference
    alpha=(WINDOW[0]-3*DT)/DT; beta=(WINDOW[1]-3*DT)/DT
    f1=.5*(alpha+beta); f0=1-f1
    worst=0.
    cache_indices=(23,24,25,26,28,29,30,31,32,33,34)
    packed_indices=(4,5,6,7,39,40,41,73,74,75,107)
    for backend,case in cases.items():
        endpoints=[]
        for step in (3,4):
            with h5py.File(case/f'outdat/new/checkpoints/step{step:012d}/state.h5') as state:
                endpoints.append(np.stack([state[f'q{n:04d}'][...].transpose(2,1,0) for n in cache_indices],axis=-1))
        with h5py.File(case/FINAL/'statistics.h5') as checkpoint:
            stats=checkpoint['air5_volume_statistics']
            for p,c in zip(packed_indices,cache_indices):
                with h5py.File(case/FINAL/'state.h5') as state:
                    # z=L is a periodic duplicate, deliberately not an owned sample.
                    np.testing.assert_allclose(stats[f'q{p:04d}'][...][:-1],state[f'q{c:04d}'][...][:-1],
                        atol=2e-10,rtol=0)
        ra,rb=endpoints[0][...,0],endpoints[1][...,0]
        ua,ub=endpoints[0][...,1:4],endpoints[1][...,1:4]
        sa,sb=endpoints[0][...,4:],endpoints[1][...,4:]
        mean=f0*sa+f1*sb; variance=f0*f1*(sb-sa)**2
        rho=f0*ra+f1*rb
        vel=f0*ua+f1*ub; favre=(f0*ra[...,None]*ua+f1*rb[...,None]*ub)/rho[...,None]
        delta=ub-ua
        cr=f0*f1*delta[..., :,None]*delta[...,None,:]
        cf=(f0*f1*ra*rb/rho**2)[...,None,None]*delta[..., :,None]*delta[...,None,:]
        for h,_,_,_,_,_,actual in exported(case,ranks):
            i,j,k=map(int,h[6:9]); nx,ny,nz=map(int,h[3:6]); sl=np.s_[i:i+nx,j:j+ny,k:k+nz]
            error=float(np.max(abs(actual[...,41:48]-mean[sl])/SCALAR_SCALES))
            worst=max(worst,error); assert error<=2e-10
            assert np.max(abs(actual[...,48:55]-variance[sl])/SCALAR_SCALES**2)<=2e-10
            for start,end,target in ((1,2,rho[...,None]),(2,5,vel),(5,8,favre)):
                np.testing.assert_allclose(actual[...,start:end],target[sl],atol=2e-10,rtol=0)
            for offset,target in ((14,cr),(23,cf),(32,rho[...,None,None]*cf)):
                np.testing.assert_allclose(actual[...,offset:offset+9].reshape((*actual.shape[:3],3,3),order='F'),
                    target[sl],atol=2e-10,rtol=0)
    record_property('endpoint_oracle_scalar_scaled_maxabs',worst)


@pytest.mark.parametrize('backend',('cpu','gpu'))
def test_exact_continuation(reference,backend,tmp_path):
    _,topology,ranks,cases=reference
    args=arguments(tmp_path,topology); source=cases[backend]/'outdat/new/checkpoints/step000000000003'
    before={p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
    resumed,_=launch(args,backend,ranks,'restart',4,restore=source,interval=1,insitu_config=config(),
        directory_budget_bytes=256*1024**2)
    for name in ('state.h5','statistics.h5'):
        compare_fields(cases[backend]/FINAL/name,resumed/FINAL/name)
    for name in ('control.bin','insitu_control.bin'):
        assert (cases[backend]/FINAL/name).read_bytes()==(resumed/FINAL/name).read_bytes()
    name='outdat/render/sample.air5_volume_rms.step00000004.csv'
    assert (cases[backend]/name).read_bytes()==(resumed/name).read_bytes()
    for rank in range(ranks):
        name=f'outdat/render/sample.air5_statistics.step00000004.rank{rank:08d}.bin'
        assert (cases[backend]/name).read_bytes()==(resumed/name).read_bytes()
    assert before=={p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}


def test_statistics_isolation(reference,tmp_path):
    _,topology,ranks,cases=reference
    for backend in ('cpu','gpu'):
        plain,_=launch(arguments(tmp_path/backend,topology),backend,ranks,'plain',4,interval=1)
        compare_fields(cases[backend]/FINAL/'state.h5',plain/FINAL/'state.h5')
    records=re.findall(r'ASTR_INSITU_AIR5_VOLUME rank=(\d+) step=(\d+) samples=(\d+) owned_nodes=(\d+)',
        (cases['gpu']/'run.log').read_text())
    assert len(records)==5*ranks


def test_separate_rms_classes(reference):
    _,_,ranks,cases=reference
    comparisons=[]
    for case in cases.values():
        arrays=exported(case,ranks)
        numerator=sum(np.sum(a[5][...,None]*a[6][...,[14,18,22]],axis=(0,1,2)) for a in arrays)
        variance=numerator/arrays[0][1][4]
        path=case/'outdat/render/sample.air5_volume_rms.step00000004.csv'
        rows=list(csv.DictReader(line for line in path.read_text().splitlines() if not line.startswith('#')))
        assert [row['component'] for row in rows]==list('uvw')
        values=np.array([[float(row[name]) for name in
            ('local_variance_volume_mean','local_variance_volume_rms','regional_signal_variance','regional_signal_rms')]
            for row in rows])
        np.testing.assert_allclose(values[:,0],variance,atol=2e-10,rtol=0)
        np.testing.assert_allclose(values[:,[1,3]]**2,values[:,[0,2]],atol=0,rtol=5e-16)
        packed=arrays[0][3]
        if packed[7]>0:
            matrix=packed[15:24].reshape((3,3),order='F')/packed[7]
            np.testing.assert_allclose(values[:,2],np.diag(matrix),atol=2e-10,rtol=0)
        signals=[]
        for step in (3,4):
            with h5py.File(case/f'outdat/new/checkpoints/step{step:012d}/state.h5') as state:
                velocity=np.stack([state[f'q{n:04d}'][...].transpose(2,1,0) for n in (24,25,26)],axis=-1)
            total=np.zeros(3)
            for h,_,_,_,_,weights,_ in arrays:
                i,j,k=map(int,h[6:9]); nx,ny,nz=map(int,h[3:6])
                total+=np.sum(weights[...,None]*velocity[i:i+nx,j:j+ny,k:k+nz],axis=(0,1,2))
            signals.append(total/arrays[0][1][4])
        f1=.5*((WINDOW[0]+WINDOW[1])/DT-6)
        expected=f1*(1-f1)*(signals[1]-signals[0])**2
        np.testing.assert_allclose(values[:,2],expected,atol=2e-10,rtol=0)
        comparisons.append(values[:,[0,2]])
    np.testing.assert_allclose(*comparisons,atol=2e-10,rtol=0)


def test_gpu_memory_safety(tmp_path):
    case,_=launch(arguments(tmp_path,(2,1,1)),'gpu',2,'memcheck',2,interval=1,
        insitu_config=config(),memcheck=True,directory_budget_bytes=256*1024**2)
    logs=list(case.glob('memcheck.*.log')); assert len(logs)==2
    assert all('ERROR SUMMARY: 0 errors' in path.read_text() for path in logs)


@pytest.mark.parametrize('backend',('cpu','gpu'))
def test_volume_selection_restart_mismatch(tmp_path,backend):
    args=arguments(tmp_path,(1,1,1))
    case,_=launch(args,backend,1,'source',1,interval=1,insitu_config=config())
    source=case/'outdat/new/checkpoints/step000000000001'
    launch(args,backend,1,'rejected',2,restore=source,interval=1,
        insitu_config=config().replace('air5_volume_statistics=t','air5_volume_statistics=f').replace(
            'air5_volume_reduction=t','air5_volume_reduction=f'),
        reject='AIR5 volume statistics selection mismatch')


@pytest.mark.parametrize('backend',('cpu','gpu'))
def test_point_statistics_without_reduction(tmp_path,backend):
    args=arguments(tmp_path,(1,1,1)); settings=config().replace('air5_volume_reduction=t','air5_volume_reduction=f')
    case,_=launch(args,backend,1,'source',1,interval=1,insitu_config=settings)
    source=case/'outdat/new/checkpoints/step000000000001'
    resumed,_=launch(args,backend,1,'resumed',2,restore=source,interval=1,insitu_config=settings)
    header,*_=read_statistics(resumed/'outdat/render/sample.air5_statistics.step00000002.rank00000000.bin')
    assert header[9]==0
    assert not list(resumed.glob('outdat/render/*volume_rms*'))
    launch(args,backend,1,'rejected',2,restore=source,interval=1,insitu_config=config(),
        reject='volume statistics metadata mismatch')


@pytest.mark.parametrize('backend',('cpu','gpu'))
@pytest.mark.parametrize('defect',('metadata','nonfinite','off_owned'))
def test_corrupt_volume_state_rejected(tmp_path,backend,defect):
    args=arguments(tmp_path,(1,1,1))
    case,_=launch(args,backend,1,'source',1,interval=1,insitu_config=config())
    source=case/'outdat/new/checkpoints/step000000000001'
    root=tmp_path/'modified_source'; shutil.copytree(source.parent.parent/'resources',root/'resources')
    damaged=root/'checkpoints'/source.name; shutil.copytree(source,damaged)
    with h5py.File(damaged/'statistics.h5','r+') as file:
        state=file['air5_volume_statistics']
        if defect=='metadata':
            state['metadata'][0]=99
        else:
            state['q0005'][0 if defect=='nonfinite' else -1,0,0]=np.nan if defect=='nonfinite' else 1.
    names=[line.split()[0] for line in (damaged/'MANIFEST').read_text().splitlines()[2:]]
    (damaged/'MANIFEST').write_text(seal_file_records(damaged,names,'ASTR_CHECKPOINT_BUNDLE 1'))
    size,crc=repair.fingerprint(damaged/'MANIFEST')
    (damaged/'COMPLETE').write_text(f'ASTR_COMPLETE_1 {size} {crc:016X}\n')
    before={p.name:p.read_bytes() for p in damaged.iterdir() if p.is_file()}
    message={'metadata':'volume statistics metadata mismatch','nonfinite':'invalid volume packed statistics state',
             'off_owned':'nonzero off-owned volume statistics state'}[defect]
    launch(args,backend,1,'rejected',2,restore=damaged,interval=1,insitu_config=config(),reject=message)
    assert before=={p.name:p.read_bytes() for p in damaged.iterdir() if p.is_file()}
