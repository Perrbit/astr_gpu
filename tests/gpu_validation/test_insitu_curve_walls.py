"""Bounded y-wavy bc41 TGV: geometric stress/heat projections, not channel physics."""
import json
import re

import h5py
import numpy as np
import pytest

from run_output_restart_validation import run_case,compare_fields
from test_insitu_curve_derivatives import arguments,ROOT,FINAL
from test_insitu_wall_scalar_statistics import read_statistics as read_wall_statistics
from test_insitu_channel_statistics import read_statistics as read_volume_statistics
from test_insitu_curve_statistics import check_independent
from test_insitu_geometry import integrate_numpy


def configuration(render=False):
    from test_output_insitu_restart import configuration as preset
    return preset(statistics=True,render=render,interval=2).replace(
        'statistics_window=0.0005,0.0115','statistics_window=0.003,0.004').replace(
        "schedule_mode='steps'","products='channel_walls',schedule_mode='steps'")


def read_wall(path):
    with path.open('rb') as stream:
        assert stream.read(8)==b'ASTRIW01'
        header=np.fromfile(stream,'<i4',9); clock=np.fromfile(stream,'<f8',2)
        shape=tuple(header[3:6]); nodes=int(np.prod(shape))
        xyz=np.fromfile(stream,'<f8',nodes*3).reshape((*shape,3),order='F')
        fields=np.fromfile(stream,'<f8',nodes*4).reshape((*shape,4),order='F')
        owned=np.fromfile(stream,'<i4',nodes).reshape(shape,order='F')
        assert not stream.read(1)
    assert np.isfinite(fields).all() and np.isfinite(xyz).all()
    assert np.isfinite(clock).all()
    expected=np.zeros(shape,dtype=int); expected[:-1,:-1,:]=1
    np.testing.assert_array_equal(owned,expected)
    return header,xyz,fields,owned


def independent_wall(case,step):
    with h5py.File(case/f'outdat/new/checkpoints/step{step:012d}/state.h5') as state:
        q=np.stack([state[f'q{c:04d}'][:].transpose(2,1,0) for c in range(1,6)],axis=-1)
    rho=q[:32,:,:32,0]
    u=q[:32,:,:32,1:4]/rho[...,None]
    pressure=(q[:32,:,:32,4]-.5*rho*np.sum(u*u,axis=-1))/2.5
    temperature=pressure/rho*(1.4*.1**2)
    # Analytic mapped coordinates differentiated with the independent six-point Fourier symbol.
    h=2*np.pi/32
    symbol=1.5*np.sin(h)-.3*np.sin(2*h)+np.sin(3*h)/30
    xi,zeta=np.meshgrid(np.arange(32)*h,np.arange(32)*h,indexing='ij')
    n=np.stack([-.15*(symbol/h)*np.cos(xi)*np.sin(zeta),np.ones_like(xi),
                -.15*(symbol/h)*np.sin(xi)*np.cos(zeta)],axis=-1)
    n/=np.linalg.norm(n,axis=-1)[...,None]
    t=np.zeros_like(n); t[...,0]=1; t-=n[...,0,None]*n
    t/=np.linalg.norm(t,axis=-1)[...,None]
    with h5py.File(case/'outdat/new/resources/geometry.h5') as geometry:
        metrics=np.stack([geometry[f'q{5+c+3*d:04d}'][:].transpose(2,1,0)
                          for d in range(3) for c in range(3)],axis=-1).reshape(33,33,33,3,3).swapaxes(-1,-2)
    results=[]
    for j,side in ((0,1),(32,-1)):
        directional=np.empty((32,32,3,3)); thermal=np.empty((32,32,3))
        for c,axis in ((0,0),(2,1)):
            directional[..., :,c]=sum(w*(np.roll(u[:,j,:,:],-s,axis=axis)-np.roll(u[:,j,:,:],s,axis=axis))
                                      for s,w in ((1,.75),(2,-.15),(3,1/60)))
            thermal[...,c]=sum(w*(np.roll(temperature[:,j,:],-s,axis=axis)-
                                  np.roll(temperature[:,j,:],s,axis=axis)) for s,w in ((1,.75),(2,-.15),(3,1/60)))
        directional[..., :,1]=side*(-1.5*u[:,j,:,:]+2*u[:,j+side,:,:]-.5*u[:,j+2*side,:,:])
        thermal[...,1]=side*(-1.5*temperature[:,j,:]+2*temperature[:,j+side,:]-.5*temperature[:,j+2*side,:])
        gradient=np.einsum('...cm,...md->...cd',directional,metrics[:32,j,:32,:,:])
        grad_t=np.einsum('...m,...md->...d',thermal,metrics[:32,j,:32,:,:])
        normal=side*n
        # The nondimensional perfect-gas backend defaults to 110.3 K.
        sutherland=110.3/273.15
        tw=temperature[:,j,:]
        mu=tw*np.sqrt(tw)*(1+sutherland)/(tw+sutherland)/1600
        stress=mu[...,None,None]*(gradient+gradient.swapaxes(-1,-2)-
            (2/3)*np.trace(gradient,axis1=-2,axis2=-1)[...,None,None]*np.eye(3))
        shear=np.einsum('...i,...ij,...j->...',t,stress,normal)
        heat=-((mu/.72)/(.4*.1**2))*np.sum(grad_t*normal,axis=-1)
        results.append(np.stack([pressure[:,j,:],shear,heat,normal[...,1]],axis=-1))
    return np.stack(results,axis=2)


def check_walls(case,ranks,step):
    expected=independent_wall(case,step)
    with h5py.File(case/'outdat/new/resources/geometry.h5') as geometry:
        coordinates=np.stack([geometry[f'q{c:04d}'][:].transpose(2,1,0) for c in (1,2,3)])
    _,areas=integrate_numpy(coordinates)
    coverage=np.zeros((32,32,2),dtype=int)
    result=np.empty((32,32,2,4)); worst=0.
    for rank in range(ranks):
        header,xyz,fields,owned=read_wall(case/f'outdat/sample.wall.step{step:08d}.rank{rank:08d}.bin')
        for i,k,w in np.ndindex(owned.shape):
            gi,gk=int(header[6]+i),int(header[8]+k)
            wall=0 if fields[i,k,w,3]>0 else 1
            np.testing.assert_array_equal(xyz[i,k,w],coordinates[:,gi,32*wall,gk])
            error=float(np.max(abs(fields[i,k,w]-expected[gi%32,gk%32,wall])))
            assert error<=2e-10,(rank,i,k,w,error)
            worst=max(worst,error)
            if owned[i,k,w]:
                coverage[gi,gk,wall]+=1; result[gi,gk,wall]=fields[i,k,w]
    np.testing.assert_array_equal(coverage,1)
    records=re.findall(r'ASTR_INSITU_CURVE_WALL wall=(\d+) area=\s*(\S+) area_means=\s*([^\n]+)',
                       (case/'run.log').read_text())
    assert len(records)>=2
    for wall,area,values in records[-2:]:
        w=int(wall)-1
        average=sum(np.roll(expected[:,:,w,:3],(-i,-k),axis=(0,1)) for i in (0,1) for k in (0,1))/4
        measured=areas[...,w]
        assert abs(float(area)-measured.sum())<=2e-10
        np.testing.assert_allclose(np.fromstring(values,sep=' '),
            np.einsum('ik,ikf->f',measured,average)/measured.sum(),rtol=0,atol=2e-10)
    return result,worst


def check_wall_moments(case,ranks):
    a,b=(independent_wall(case,step)[...,:3] for step in (3,4))
    mean=.5*(a+b); variance=.25*(b-a)**2
    for rank in range(ranks):
        header,meta,samples,xyz,values,owned=read_wall_statistics(
            case/f'outdat/render/sample.wall_statistics.step00000004.rank{rank:08d}.bin')
        assert samples==5
        np.testing.assert_allclose(meta,[.004,.003,.004,.001],rtol=0,atol=2e-15)
        for i,k,w in np.ndindex(owned.shape):
            gi,gk=int(header[6]+i)%32,int(header[8]+k)%32
            side=0 if xyz[i,k,w,1]<np.pi else 1
            np.testing.assert_allclose(values[i,k,w,:3],mean[gi,gk,side],rtol=0,atol=2e-10)
            np.testing.assert_allclose(values[i,k,w,3:6],variance[gi,gk,side],rtol=0,atol=2e-10)


@pytest.mark.parametrize('ranks,axis',[(1,'x'),(2,'x'),(2,'y'),(2,'z')])
def test_curve_wall_fields_stats_and_restart(tmp_path,ranks,axis,record_property):
    wall_results={}; statistics={}; errors=[]
    for backend in ('cpu','gpu'):
        args=arguments(tmp_path,backend,axis)
        kwargs=dict(grid='32,32,32',tgv_mapping='y-wavy',checkpoint_interval=1)
        sampled,_=run_case(args,ROOT,backend,ranks,'walls',4,**kwargs)
        wall_results[backend],error=check_walls(sampled,ranks,4); errors.append(error)
        case,_=run_case(args,ROOT,backend,ranks,'statistics',4,insitu_config=configuration(),**kwargs)
        check_independent(case,ranks,mapping='y-wavy')
        check_wall_moments(case,ranks)
        compare_fields(case/FINAL/'state.h5',sampled/FINAL/'state.h5')
        source=case/'outdat/new/checkpoints/step000000000003'
        restored,_=run_case(args,ROOT,backend,ranks,'restored',4,insitu_config=configuration(),restore=source,**kwargs)
        for name in ('state.h5','statistics.h5'):
            compare_fields(case/FINAL/name,restored/FINAL/name)
        for name in ('control.bin','insitu_control.bin'):
            assert (case/FINAL/name).read_bytes()==(restored/FINAL/name).read_bytes()
        for rank in range(ranks):
            for kind in ('statistics','wall_statistics'):
                name=f'outdat/render/sample.{kind}.step00000004.rank{rank:08d}.bin'
                assert (case/name).read_bytes()==(restored/name).read_bytes()
        statistics[backend]=case
    np.testing.assert_allclose(wall_results['cpu'],wall_results['gpu'],rtol=0,atol=2e-10)
    for rank in range(ranks):
        for kind,reader in (('statistics',read_volume_statistics),('wall_statistics',read_wall_statistics)):
            name=f'outdat/render/sample.{kind}.step00000004.rank{rank:08d}.bin'
            a=reader(statistics['cpu']/name); b=reader(statistics['gpu']/name)
            for left,right in zip(a,b):
                np.testing.assert_allclose(left,right,rtol=0,atol=2e-10)
    record_property('geometric_wall_oracle_maxabs',max(errors))
    record_property('cpu_gpu_wall_maxabs',float(np.max(abs(wall_results['cpu']-wall_results['gpu']))))


@pytest.mark.parametrize('axis',('x','y','z'))
def test_curve_wall_memory_safety(tmp_path,axis):
    run_case(arguments(tmp_path,'gpu',axis),ROOT,'gpu',2,'wall_memcheck',4,
        grid='32,32,32',tgv_mapping='y-wavy',checkpoint_interval=1,
        insitu_config=configuration(),memcheck=True)


@pytest.mark.parametrize('axis',('x','y','z'))
def test_curve_wall_render_geometry_and_restart(tmp_path,axis,record_property):
    from PIL import Image
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOXML import vtkXMLPPolyDataReader
    from test_insitu_products import compare_geometry
    from run_insitu_gpu_derivatives import check_resources
    names=('wall_pressure','wall_shear_x','wall_heat_into_gas','wall_normal_y')
    args=arguments(tmp_path,'gpu',axis)
    kwargs=dict(grid='32,32,32',tgv_mapping='y-wavy',checkpoint_interval=1)
    plain,_=run_case(args,ROOT,'gpu',2,'plain',4,insitu_config=configuration(),**kwargs)
    render_config=configuration(True).replace("products='channel_walls',","products='channel_walls',wall_mean_render=t,")
    case,_=run_case(args,ROOT,'gpu',2,'render',4,insitu_config=render_config,**kwargs)
    restored,_=run_case(args,ROOT,'gpu',2,'render_restart',4,insitu_config=render_config,
        restore=case/'outdat/new/checkpoints/step000000000003',**kwargs)
    for name in ('state.h5','statistics.h5'):
        compare_fields(case/FINAL/name,plain/FINAL/name)
        compare_fields(case/FINAL/name,restored/FINAL/name)
    for name in ('control.bin','insitu_control.bin'):
        assert (case/FINAL/name).read_bytes()==(restored/FINAL/name).read_bytes()
    check_resources(case,2,steps=(2,4),capture=False,statistics=True)
    check_resources(restored,2,steps=(4,),capture=False,statistics=True)
    with h5py.File(case/'outdat/new/resources/geometry.h5') as geometry:
        xyz=np.stack([geometry[f'q{c:04d}'][:].transpose(2,1,0) for c in (1,2,3)])
    _,area=integrate_numpy(xyz)
    worst=0.
    for step in (2,4):
        # The production keep-two policy retains steps 3/4, not the earlier render frame.
        expected=independent_wall(case,step) if step==4 else None
        for name in names[:3]:
            path=case/f'outdat/render/{name}.step{step:08d}.pvtp'
            reader=vtkXMLPPolyDataReader(); reader.SetFileName(str(path)); reader.Update()
            data=reader.GetOutput()
            assert reader.GetErrorCode()==0 and data.GetNumberOfCells()==2*32**2
            points=vtk_to_numpy(data.GetPoints().GetData())
            assert points.dtype==np.float64
            fields=np.stack([vtk_to_numpy(data.GetPointData().GetArray(n)) for n in names],axis=-1)
            assert np.isfinite(fields).all()
            seam_values={}
            for point,value in zip(points,fields):
                i,k=np.rint(point[[0,2]]/(2*np.pi/32)).astype(int)
                wall=0 if value[3]>0 else 1
                np.testing.assert_array_equal(point,xyz[:,i,32*wall,k])
                key=tuple(point)
                if key in seam_values:
                    np.testing.assert_array_equal(value,seam_values[key])
                seam_values[key]=value
                if expected is not None:
                    error=float(np.max(abs(value-expected[i%32,k%32,wall])))
                    assert error<=2e-10; worst=max(worst,error)
            measured=0.; cells=set()
            nodes=[.5-.5/np.sqrt(3),.5+.5/np.sqrt(3)]
            for index in range(data.GetNumberOfCells()):
                cell=data.GetCell(index)
                ids=[cell.GetPointId(n) for n in range(cell.GetNumberOfPoints())]
                assert len(ids)==4 and np.all(np.sign(fields[ids,3])==np.sign(fields[ids[0],3]))
                vertices=points[ids]; key=tuple(vertices.mean(axis=0))
                assert key not in cells; cells.add(key)
                a,b,c,d=vertices
                for s in nodes:
                    for t in nodes:
                        ds=(1-t)*(b-a)+t*(c-d); dt=(1-s)*(d-a)+s*(c-b)
                        measured+=np.linalg.norm(np.cross(ds,dt))/4
            assert abs(measured-area.sum())<=2e-10
            assert data.GetFieldData().GetArray('complete_step').GetValue(0)==step
            assert data.GetFieldData().GetArray('simulation_time').GetValue(0)==step*.001
            assert data.GetFieldData().GetAbstractArray('coordinate_space').GetValue(0)=='physical'
            units=data.GetFieldData().GetAbstractArray('field_units')
            assert f'{name}=dimensionless' in {units.GetValue(i) for i in range(units.GetNumberOfValues())}
            with Image.open(path.with_suffix('.jpeg')) as image:
                pixels=np.asarray(image.convert('RGB'))
                assert pixels.shape==(600,800,3) and np.any(pixels<245)
                border=np.concatenate((pixels[:10].reshape(-1,3),pixels[-10:].reshape(-1,3),
                    pixels[:,:10].reshape(-1,3),pixels[:,-10:].reshape(-1,3)))
                assert np.all(border>=245),'physical wall surface is clipped by the camera'
            assert path.with_suffix('.eps').read_bytes().startswith(b'%!PS-Adobe')
            if step==4:
                other=restored/'outdat/render'/path.name
                compare_geometry(path,other,list(names))
                for extension in ('.jpeg','.eps'):
                    assert path.with_suffix(extension).read_bytes()==other.with_suffix(extension).read_bytes()
    mean=.5*(independent_wall(case,3)[...,:3]+independent_wall(case,4)[...,:3])
    assert not list((case/'outdat/render').glob('mean_*.step00000002.pvtp'))
    for c,name in enumerate(names[:3]):
        path=case/f'outdat/render/mean_{name}.step00000004.pvtp'
        reader=vtkXMLPPolyDataReader(); reader.SetFileName(str(path)); reader.Update()
        data=reader.GetOutput()
        assert reader.GetErrorCode()==0 and data.GetNumberOfCells()==2*32**2
        points=vtk_to_numpy(data.GetPoints().GetData())
        values=vtk_to_numpy(data.GetPointData().GetArray('mean_'+name))
        for point,value in zip(points,values):
            i,k=np.rint(point[[0,2]]/(2*np.pi/32)).astype(int)
            wall=0 if point[1]<np.pi else 1
            assert abs(value-mean[i%32,k%32,wall,c])<=2e-10
        for n,expected_clock in (('statistics_duration',.001),('statistics_window_start',.003),
                                 ('statistics_window_end',.004)):
            np.testing.assert_allclose(vtk_to_numpy(data.GetPointData().GetArray(n)),expected_clock,rtol=0,atol=2e-15)
        other=restored/'outdat/render'/path.name
        compare_geometry(path,other,['mean_'+name])
        for extension in ('.jpeg','.eps'):
            assert path.with_suffix(extension).read_bytes()==other.with_suffix(extension).read_bytes()
    record_property('curve_wall_render_field_maxabs',worst)
    record_property('display_frame',str(case/'outdat/render/wall_shear_x.step00000004.jpeg'))
