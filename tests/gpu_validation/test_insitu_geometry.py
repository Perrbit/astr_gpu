"""Independent quadrature checks; geometry refinement is not a fluid simulation."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from generate_curvilinear_tgv_grid import mapped_grid

ROOT=Path(__file__).resolve().parents[2]
PROBE=Path(os.environ.get('ASTR_INSITU_GEOMETRY_PROBE',ROOT/'build_insitu_check/bin/insitu_geometry_probe'))


def integrate_numpy(coordinates):
    """Vectorized tensor Gauss rule using NumPy determinants, not the Fortran implementation."""
    nodes=np.array([.5-.5/np.sqrt(3),.5+.5/np.sqrt(3)])
    cells=tuple(n-1 for n in coordinates.shape[1:])
    volume=np.zeros(cells)
    for r in nodes:
        for q in nodes:
            for p in nodes:
                jac=np.zeros(cells+(3,3))
                for d in (0,1):
                    for b in (0,1):
                        for a in (0,1):
                            xyz=np.moveaxis(coordinates[:,a:a+cells[0],b:b+cells[1],d:d+cells[2]],0,-1)
                            shape=[(1-p,p)[a],(1-q,q)[b],(1-r,r)[d]]
                            for direction,index in enumerate((a,b,d)):
                                others=np.prod([v for c,v in enumerate(shape) if c!=direction])
                                jac[..., :,direction]+=xyz*(2*index-1)*others
                det=np.linalg.det(jac)
                assert np.isfinite(det).all() and np.min(det)>0
                volume+=det/8
    areas=np.zeros((cells[0],cells[2],2))
    for wall,j in enumerate((0,cells[1])):
        for q in nodes:
            for p in nodes:
                jac=np.zeros((cells[0],cells[2],3,2))
                for b in (0,1):
                    for a in (0,1):
                        xyz=np.moveaxis(coordinates[:,a:a+cells[0],j,b:b+cells[2]],0,-1)
                        jac[..., :,0]+=xyz*(2*a-1)*(1-q,q)[b]
                        jac[..., :,1]+=xyz*(1-p,p)[a]*(2*b-1)
                areas[...,wall]+=np.linalg.norm(np.cross(jac[..., :,0],jac[..., :,1]),axis=-1)/4
    return volume,areas


def run_probe(path,coordinates,reject=False):
    path.mkdir()
    dims=np.array(coordinates.shape[1:])-1
    with (path/'coordinates.bin').open('xb') as stream:
        stream.write(dims.astype('<i4').tobytes())
        stream.write(np.asarray(coordinates,dtype='<f8').ravel(order='F').tobytes())
    output=path/'measures.bin'
    result=subprocess.run([str(PROBE),str(path/'coordinates.bin'),str(output)],capture_output=True,text=True,timeout=60)
    if reject:
        assert result.returncode!=0 and not output.exists(),result.stdout+result.stderr
        return
    assert result.returncode==0,result.stdout+result.stderr
    packed=np.fromfile(output,dtype='<f8')
    count=int(np.prod(dims))
    volumes=packed[:count].reshape(tuple(dims),order='F')
    areas=packed[count:].reshape((dims[0],dims[2],2),order='F')
    assert np.isfinite(packed).all() and np.min(packed)>0
    assert sum(p.stat().st_size for p in path.iterdir())<=256*1024**2
    return volumes,areas


@pytest.mark.parametrize('mapping',('periodic','y-wavy'))
def test_curve_quadrature_and_area_refinement(tmp_path,mapping,record_property):
    nodes,weights=np.polynomial.legendre.leggauss(128)
    x=(nodes+1)*np.pi
    area_reference=np.pi**2*np.einsum('i,j,ij->',weights,weights,np.sqrt(1+.15**2*(
        np.cos(x)[:,None]**2*np.sin(x)[None,:]**2+
        np.sin(x)[:,None]**2*np.cos(x)[None,:]**2)))
    errors=[]
    for n in (16,32,64):
        coordinates=np.stack(mapped_grid(n,n,n,.15,mapping)[:3])
        volumes,areas=run_probe(tmp_path/f'grid{n}',coordinates)
        expected_volume,expected_area=integrate_numpy(coordinates)
        np.testing.assert_allclose(volumes,expected_volume,rtol=0,atol=2e-10)
        np.testing.assert_allclose(areas,expected_area,rtol=0,atol=2e-10)
        assert abs(volumes.sum()-(2*np.pi)**3)<=2e-10
        if mapping=='y-wavy':
            errors.append(abs(areas[...,0].sum()-area_reference))
            assert abs(areas[...,0].sum()-areas[...,1].sum())<=2e-10
    if errors:
        assert all(a>b for a,b in zip(errors,errors[1:])),errors
        record_property('area_refinement',json.dumps(dict(sizes=[16,32,64],absolute_error=errors,
            observed_orders=np.log2(np.array(errors[:-1])/errors[1:]).tolist())))


@pytest.mark.parametrize('defect',('inverted','collapsed','nan','folded'))
def test_invalid_geometry_stops(tmp_path,defect):
    coordinates=np.stack(np.meshgrid([0.,1.],[0.,1.],[0.,1.],indexing='ij'))
    if defect=='inverted':
        coordinates[0]*=-1
    elif defect=='collapsed':
        coordinates[1]=0
    elif defect=='nan':
        coordinates[2,0,0,0]=np.nan
    else:
        coordinates[:,1,1,1]=[-2.,-2.,-2.]
    run_probe(tmp_path/defect,coordinates,reject=True)


@pytest.mark.parametrize('defect',('none','parallel','no_inward','normal_x','nan'))
def test_wall_frame_orientation_and_rejection(tmp_path,defect):
    vectors=np.array([[1.,.2,0.],[0.,.3,1.],[0.,1.,0.]])
    if defect=='parallel': vectors[1]=vectors[0]
    if defect=='no_inward': vectors[2]=0
    if defect=='normal_x': vectors[:]=[[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]]
    if defect=='nan': vectors[0,0]=np.nan
    source=tmp_path/'vectors.bin'; output=tmp_path/'frame.bin'
    source.write_bytes(vectors.astype('<f8').tobytes())
    result=subprocess.run([str(PROBE),str(source),str(output),'frame'],capture_output=True,timeout=30)
    if defect!='none':
        assert result.returncode!=0 and not output.exists()
        return
    assert result.returncode==0,result.stderr
    normal,tangent=np.fromfile(output,'<f8').reshape(2,3)
    expected=np.cross(vectors[0],vectors[1]); expected/=np.linalg.norm(expected)
    if np.dot(expected,vectors[2])<0: expected=-expected
    np.testing.assert_allclose(normal,expected,rtol=0,atol=2e-10)
    assert np.dot(normal,vectors[2])>0
    assert abs(np.dot(normal,tangent))<=2e-10
    assert abs(np.linalg.norm(tangent)-1)<=2e-10 and tangent[0]>0
