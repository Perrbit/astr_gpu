"""IS6 approved periodic CURVE TGV field gate, not geometry/physics certification."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from generate_curvilinear_tgv_grid import mapped_grid
from run_insitu_gpu_derivatives import configuration, check_resources
from run_output_restart_validation import run_case, compare_fields

ROOT = Path(__file__).resolve().parents[2]
GPU = Path(os.environ.get('ASTR_OUTPUT_RUNTIME_EXE', ROOT/'build_insitu_gpu/bin/astr'))
CPU = Path(os.environ.get('ASTR_OUTPUT_CPU_EXE', ROOT/'build_insitu_check/bin/astr'))
MPI = Path(os.environ.get('ASTR_OUTPUT_MPIEXEC',
    '/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec'))
LIBRARY = Path(os.environ.get('ASTR_CATALYST_LIBRARY',
    '/home/dell/workspace/astr_dependencies/install/paraview-6.1.1-hpcx-gcc13/lib/catalyst'))
FIELDS = ('q1 q2 q3 q4 q5 rho u v w pressure temperature').split()
DERIVED = ('du_dx dv_dx dw_dx du_dy dv_dy dw_dy du_dz dv_dz dw_dz '
           'Q_rs divergence omega_x omega_y omega_z').split()
FINAL = 'outdat/new/checkpoints/step000000000004'


def arguments(output, backend, axis):
    return SimpleNamespace(output=output, executable=GPU if backend=='gpu' else CPU,
        mpiexec=MPI, case='tgv', mode='steps', restart_step=99, initial_dimension=0,
        legacy_statistics=False, statistics=False, initial_restart=False,
        filter_workspace='scalar', force='feedback', axis=axis, no_samples=True,
        directory_budget_bytes=256*1024**2)


def reference(case, step):
    """Periodic six-point difference of canonical q, contracted with all nine metrics."""
    with h5py.File(case/f'outdat/new/checkpoints/step{step:012d}/state.h5') as state:
        rho = state['q0001'][:-1,:-1,:-1]
        velocity = np.stack([state[f'q{c:04d}'][:-1,:-1,:-1]/rho for c in (2,3,4)], axis=-1)
    gradient = np.zeros(velocity.shape+(3,))
    with h5py.File(case/'outdat/new/resources/geometry.h5') as geometry:
        for computational, axis in enumerate((2,1,0)):
            directional = sum(w*(np.roll(velocity,-n,axis=axis)-np.roll(velocity,n,axis=axis))
                              for n,w in ((1,3/4),(2,-3/20),(3,1/60)))
            for physical in range(3):
                metric = geometry[f'q{5+computational+3*physical:04d}'][:-1,:-1,:-1]
                assert np.isfinite(metric).all()
                gradient[..., :, physical] += directional*metric[..., None]
    values = [gradient[..., c,d] for d in range(3) for c in range(3)]
    values += [-.5*np.einsum('...ij,...ji->...',gradient,gradient),
               np.trace(gradient,axis1=-2,axis2=-1),
               gradient[...,2,1]-gradient[...,1,2], gradient[...,0,2]-gradient[...,2,0],
               gradient[...,1,0]-gradient[...,0,1]]
    return [np.pad(value,[(0,1)]*3,mode='wrap') for value in values]


def index_coordinates(ranks, axis, rank):
    cells = np.array([32,32,32])
    origin = np.zeros(3,dtype=int)
    cells['xyz'.index(axis)] //= ranks
    origin['xyz'.index(axis)] = rank*cells['xyz'.index(axis)]
    xyz = np.meshgrid(*(np.arange(o,o+n+1) for o,n in zip(origin,cells)),indexing='ij')
    return tuple(value.ravel(order='F') for value in xyz)


def compare_captures(host, device, cpu, ranks, axis):
    maxima = {}
    xyz = mapped_grid(32,32,32,.15)[:3]
    for step in (0,2,4):
        discrete = reference(device,step) if step==4 else None
        for rank in range(ranks):
            idx = index_coordinates(ranks,axis,rank)
            name = f'fields.step{step:08d}.rank{rank:08d}.npz'
            with np.load(host/'outdat/render'/name) as a, np.load(device/'outdat/render'/name) as b:
                np.testing.assert_array_equal(a['xyz'],b['xyz'])
                np.testing.assert_allclose(b['xyz'],np.stack([v[idx] for v in xyz],axis=-1),rtol=0,atol=2e-14)
                assert set(a.files)==set(b.files)==set(['xyz']+FIELDS+DERIVED)
                for field in FIELDS+DERIVED:
                    error = float(np.max(abs(a[field]-b[field])))
                    assert error<=2e-10, (step,rank,field,error)
                    maxima[field]=max(maxima.get(field,0.),error)
                gradient = np.stack([b[name] for name in DERIVED[:9]],axis=-1).reshape(-1,3,3).transpose(0,2,1)
                np.testing.assert_allclose(b['Q_rs'],-.5*np.einsum('nij,nji->n',gradient,gradient),rtol=0,atol=2e-10)
                if discrete is not None:
                    selection = (idx[2],idx[1],idx[0])
                    for field, expected in zip(DERIVED,discrete):
                        error = float(np.max(abs(b[field]-expected[selection])))
                        assert error<=2e-10, ('independent full-metric stencil',field,error)
                        maxima['discrete_'+field]=max(maxima.get('discrete_'+field,0.),error)
                    with h5py.File(cpu/FINAL/'state.h5') as state:
                        for component,field in enumerate(FIELDS,1):
                            error = float(np.max(abs(b[field]-state[f'q{component:04d}'][:][selection])))
                            assert error<=2e-10, ('CPU solver',field,error)
                            maxima['cpu_'+field]=max(maxima.get('cpu_'+field,0.),error)
    return maxima


@pytest.mark.parametrize('ranks,axis',[(1,'x'),(2,'x'),(2,'y'),(2,'z')])
def test_curve_periodic_derivatives_and_restart(tmp_path,ranks,axis,record_property):
    args = arguments(tmp_path,'gpu',axis)
    config = configuration('gpu',LIBRARY,ROOT/'tests/gpu_validation/insitu_diagnostic_capture.py')
    kwargs = dict(grid='32,32,32',tgv_mapping='periodic')
    cpu,_ = run_case(arguments(tmp_path,'cpu',axis),ROOT,'cpu',ranks,'cpu_reference',4,
                     insitu_config='&insitu_run\n enabled=f\n/\n',**kwargs)
    cpu_seed,_ = run_case(arguments(tmp_path,'cpu',axis),ROOT,'cpu',ranks,'cpu_seed',3,
                         insitu_config='&insitu_run\n enabled=f\n/\n',**kwargs)
    cpu_restored,_ = run_case(arguments(tmp_path,'cpu',axis),ROOT,'cpu',ranks,'cpu_restored',4,
        insitu_config='&insitu_run\n enabled=f\n/\n',
        restore=cpu_seed/'outdat/new/checkpoints/step000000000003',**kwargs)
    compare_fields(cpu/FINAL/'state.h5',cpu_restored/FINAL/'state.h5')
    host,_ = run_case(args,ROOT,'gpu',ranks,'host_derivatives',4,
        insitu_config=configuration('cpu',LIBRARY,ROOT/'tests/gpu_validation/insitu_diagnostic_capture.py'),**kwargs)
    device,_ = run_case(args,ROOT,'gpu',ranks,'device_derivatives',4,insitu_config=config,**kwargs)
    for case in (host,device):
        check_resources(case,ranks)
    compare_fields(host/FINAL/'state.h5',device/FINAL/'state.h5')
    off,_ = run_case(args,ROOT,'gpu',ranks,'off',4,
                    insitu_config='&insitu_run\n enabled=f\n/\n',**kwargs)
    compare_fields(off/FINAL/'state.h5',device/FINAL/'state.h5')
    errors=compare_captures(host,device,cpu,ranks,axis)
    seed,_=run_case(args,ROOT,'gpu',ranks,'seed',3,insitu_config=config,**kwargs)
    restored,_=run_case(args,ROOT,'gpu',ranks,'restored',4,insitu_config=config,
        restore=seed/'outdat/new/checkpoints/step000000000003',**kwargs)
    check_resources(seed,ranks,steps=(0,2,3))
    check_resources(restored,ranks,steps=(4,))
    assert not (restored/'datin/grid.tgv.h5').exists()
    compare_fields(device/FINAL/'state.h5',restored/FINAL/'state.h5')
    for name in ('control.bin','insitu_control.bin'):
        assert (device/FINAL/name).read_bytes()==(restored/FINAL/name).read_bytes()
    for rank in range(ranks):
        name=f'fields.step00000004.rank{rank:08d}.npz'
        with np.load(device/'outdat/render'/name) as a, np.load(restored/'outdat/render'/name) as b:
            assert a.files==b.files
            for field in a.files:
                np.testing.assert_array_equal(a[field],b[field])
    record_property('field_errors',json.dumps(errors))


def test_unvalidated_curve_np4_rejected(tmp_path):
    config=configuration('gpu',LIBRARY,ROOT/'tests/gpu_validation/insitu_diagnostic_capture.py')
    run_case(arguments(tmp_path,'gpu','x'),ROOT,'gpu',4,'reject_np4',4,
        grid='32,32,32',tgv_mapping='periodic',insitu_config=config,
        reject='formal in situ requires a supported Cartesian candidate or 32^3 periodic CURVE TGV NP=1/2')


@pytest.mark.parametrize('kind',('host','device','reserve'))
def test_curve_statistics_resource_rejection(tmp_path,kind):
    from test_insitu_curve_statistics import configuration as statistics_configuration
    config=statistics_configuration()
    if kind=='host':
        config=config.replace('host_budget_bytes=4294967296','host_budget_bytes=1')
        message='statistics host allocation exceeds node budget'
    else:
        if kind=='device': config=config.replace('device_budget_bytes=2147483648','device_budget_bytes=1')
        else: config=config.replace('device_reserve_bytes=1073741824','device_reserve_bytes=1099511627776')
        message='statistics device allocation exceeds budget or free-memory headroom'
    case,_=run_case(arguments(tmp_path,'gpu','x'),ROOT,'gpu',2,'reject_'+kind,4,
        grid='32,32,32',tgv_mapping='periodic',insitu_config=config,reject=message)
    assert not list((case/'outdat/render').glob('*.jpeg'))


@pytest.mark.parametrize('axis',('x','y','z'))
def test_curve_derivative_memory_safety(tmp_path,axis):
    config=configuration('gpu',LIBRARY,ROOT/'tests/gpu_validation/insitu_diagnostic_capture.py')
    run_case(arguments(tmp_path,'gpu',axis),ROOT,'gpu',2,'memcheck',4,
        grid='32,32,32',tgv_mapping='periodic',insitu_config=config,memcheck=True)


def test_curve_q_real_render_and_restart(tmp_path,record_property):
    from PIL import Image
    from test_insitu_products import compare_geometry
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOXML import vtkXMLPPolyDataReader
    def check_q(case,steps):
        check_resources(case,2,steps=steps,capture=False)
        output=case/'outdat/render'
        for step in steps:
            for rank in range(2):
                record=json.loads((output/f'mesh_step{step:08d}_rank{rank}.json').read_text())
                assert record['step']==step and record['time']==step*.001
                assert set(record['products'])=={'q_surface'}
                assert record['units']==dict.fromkeys(('u','v','w','Q_rs'),'dimensionless')
            path=output/f'q_surface.step{step:08d}.jpeg'
            assert path.with_suffix('.eps').is_file()
            with Image.open(path) as image:
                pixels=np.asarray(image.convert('RGB'))
                assert pixels.shape==(600,800,3) and np.any(pixels<245)
            reader=vtkXMLPPolyDataReader(); reader.SetFileName(str(path.with_suffix('.pvtp'))); reader.Update()
            data=reader.GetOutput()
            assert data.GetNumberOfCells()>0
            points=vtk_to_numpy(data.GetPoints().GetData())
            assert np.isfinite(points).all() and np.min(points)>=-2e-10 and np.max(points)<=2*np.pi+2e-10
            np.testing.assert_allclose(vtk_to_numpy(data.GetPointData().GetArray('Q_rs')),.25,rtol=0,atol=2e-10)
            assert data.GetFieldData().GetArray('complete_step').GetValue(0)==step
            assert data.GetFieldData().GetArray('simulation_time').GetValue(0)==step*.001
            assert data.GetFieldData().GetAbstractArray('coordinate_space').GetValue(0)=='physical'
            units=data.GetFieldData().GetAbstractArray('field_units')
            assert {units.GetValue(i) for i in range(units.GetNumberOfValues())}=={
                f'{field}=dimensionless' for field in ('u','v','w','Q_rs')}
    config=configuration('gpu',LIBRARY,ROOT/'scripts/insitu/tgv_pipeline.py').replace(
        "derivative_backend='gpu',", "derivative_backend='gpu',products='q_surface',").replace(
        'initial_frame=t','initial_frame=f')
    args=arguments(tmp_path,'gpu','x')
    kwargs=dict(grid='32,32,32',tgv_mapping='periodic',checkpoint_interval=1)
    continuous,_=run_case(args,ROOT,'gpu',2,'real_q',4,insitu_config=config,**kwargs)
    check_q(continuous,[2,4])
    restored,_=run_case(args,ROOT,'gpu',2,'real_q_restored',4,insitu_config=config,
        restore=continuous/'outdat/new/checkpoints/step000000000003',**kwargs)
    check_q(restored,[4])
    compare_fields(continuous/FINAL/'state.h5',restored/FINAL/'state.h5')
    path=restored/'outdat/render/q_surface.step00000004.jpeg'
    with Image.open(path) as actual, Image.open(continuous/'outdat/render'/path.name) as expected:
        np.testing.assert_array_equal(np.asarray(actual),np.asarray(expected))
    geometry=path.with_suffix('.pvtp')
    compare_geometry(continuous/'outdat/render'/geometry.name,geometry,['u','v','w','Q_rs'])
    for part in (restored/'outdat/render').rglob('*.vtp'):
        assert part.read_bytes()==(continuous/'outdat/render'/part.relative_to(restored/'outdat/render')).read_bytes()
    record_property('display_frame',str(path))


@pytest.mark.parametrize('axis',('x','y','z'))
def test_curve_cross_rank_streamlines_and_restart(tmp_path,axis,record_property):
    from test_output_insitu_restart import check_render
    from test_insitu_products import compare_geometry
    config=configuration('gpu',LIBRARY,ROOT/'scripts/insitu/tgv_pipeline.py').replace(
        "derivative_backend='gpu',", "derivative_backend='gpu',products='q_streamlines',").replace(
        'initial_frame=t','initial_frame=f')
    args=arguments(tmp_path,'gpu',axis)
    kwargs=dict(grid='32,32,32',tgv_mapping='periodic',checkpoint_interval=1)
    continuous,_=run_case(args,ROOT,'gpu',2,'streams',4,insitu_config=config,**kwargs)
    check_render(continuous,2,[2,4],statistics=False)
    restored,_=run_case(args,ROOT,'gpu',2,'streams_restored',4,insitu_config=config,
        restore=continuous/'outdat/new/checkpoints/step000000000003',**kwargs)
    check_render(restored,2,[4],statistics=False)
    compare_fields(continuous/FINAL/'state.h5',restored/FINAL/'state.h5')
    for path in (restored/'outdat/render').rglob('*'):
        if path.suffix in ('.jpeg','.vtp'):
            assert path.read_bytes()==(continuous/'outdat/render'/path.relative_to(restored/'outdat/render')).read_bytes()
    for path in (restored/'outdat/render').glob('*.pvtp'):
        compare_geometry(continuous/'outdat/render'/path.name,path,['u','v','w','Q_rs'])
    errors=[]
    for step in (2,4):
        record=json.loads((continuous/f'outdat/render/mesh_step{step:08d}_rank0.json').read_text())
        errors.append(record['crossing_maxabs'])
    record_property('constant_physical_velocity_crossing_maxabs',max(errors))


def test_curve_default_mean_streamlines_and_statistics_restart(tmp_path):
    from test_output_insitu_restart import configuration as default_configuration,check_render
    from test_insitu_products import compare_geometry
    from test_insitu_curve_statistics import check_independent
    config=default_configuration(statistics=True,render=True,interval=2).replace(
        'statistics_window=0.0005,0.0115','statistics_window=0.003,0.004')
    args=arguments(tmp_path,'gpu','x')
    kwargs=dict(grid='32,32,32',tgv_mapping='periodic',checkpoint_interval=1)
    off,_=run_case(args,ROOT,'gpu',2,'default_off',4,
        insitu_config='&insitu_run\n enabled=f\n/\n',monitor_resources=True,**kwargs)
    baseline=json.loads((off/'resources.sampled.json').read_text())
    case,_=run_case(args,ROOT,'gpu',2,'default_mean',4,insitu_config=config,
        monitor_resources=True,resource_baseline=baseline,**kwargs)
    check_render(case,2,[2,4]); check_independent(case,2)
    compare_fields(case/FINAL/'state.h5',off/FINAL/'state.h5')
    observation=json.loads((case/'resources.sampled.json').read_text())
    assert observation['sampling_period_seconds']==.02 and observation['samples']>0
    assert observation['additional_host_peak_difference_bytes']<=4*1024**3
    assert max(observation['additional_device_peak_difference_bytes'].values())<=2*1024**3
    restored,_=run_case(args,ROOT,'gpu',2,'default_mean_restart',4,insitu_config=config,
        restore=case/'outdat/new/checkpoints/step000000000003',**kwargs)
    check_render(restored,2,[4])
    for name in ('state.h5','statistics.h5'):
        compare_fields(case/FINAL/name,restored/FINAL/name)
    for name in ('control.bin','insitu_control.bin'):
        assert (case/FINAL/name).read_bytes()==(restored/FINAL/name).read_bytes()
    for kind in ('reynolds','favre'):
        path=case/f'outdat/render/mean_{kind}_streamlines.step00000004.pvtp'
        assert path.is_file()
        compare_geometry(path,restored/'outdat/render'/path.name,[f'mean_{c}_{kind}' for c in ('u','v','w')])
    for path in (restored/'outdat/render').rglob('*'):
        if path.suffix in ('.jpeg','.vtp'):
            assert path.read_bytes()==(case/'outdat/render'/path.relative_to(restored/'outdat/render')).read_bytes()
