"""TB3 local velocity-gradient eigensystems, before real-field admission."""
import ctypes
import json
import os
from pathlib import Path

import numpy as np
import pytest


def evaluate(matrices,backend,requested=3):
    path=os.environ.get('ASTR_VORTEX_'+backend.upper()+'_PROBE')
    if not path:
        pytest.skip('Select the root FP64 '+backend+' vortex mathematical probe')
    function=getattr(ctypes.CDLL(path),'astr_vortex_math_'+backend)
    real=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    integer=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    function.argtypes=[ctypes.c_int,real,ctypes.c_int,real,integer,integer]
    function.restype=ctypes.c_int
    packet=np.ascontiguousarray(np.asarray(matrices,dtype=np.float64).transpose(0,2,1))
    count=len(packet);values=np.empty((count,2));status=np.empty((count,2),dtype=np.int32);work=np.empty_like(status)
    assert function(count,packet,requested,values,status,work)==0
    return values,status,work


def analytic_cases():
    cases=[('zero',np.zeros((3,3)),[0.,0.]),('strain',np.diag([-3.,1.,2.]),[4.,0.]),
           ('expansion',np.eye(3)*2,[4.,0.]),('jordan',np.array([[0.,1.,0.],[0.,0.,1.],[0.,0.,0.]]),[0.,0.])]
    for axis in range(3):
        a=np.zeros((3,3));i,j=(axis+1)%3,(axis+2)%3;a[i,j]=-2.;a[j,i]=2.
        cases.append((f'rotation{axis}',a,[-4.,2.]))
        cases.append((f'rotation-expansion{axis}',a+.5*np.eye(3),[-3.75,2.]))
        for sign in (-1.,1.):
            shear=np.zeros((3,3));shear[i,j]=sign*3
            cases.append((f'shear{axis}-{sign}',shear,[0.,0.]))
    for exponent in (20,40,52):
        for sign in (-1.,0.,1.):
            a=np.array([[1.,1.,0.],[-1.+sign*2.**(-exponent),-1.,0.],[0.,0.,0.]])
            delta=1.+a[1,0]
            cases.append((f'coalescence{exponent}-{sign}',a,[delta,np.sqrt(max(0.,-delta))]))
    return cases


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_vortex_analytic_fields(tmp_path,backend):
    cases=analytic_cases();matrices=np.array([case[1] for case in cases]);expected=np.array([case[2] for case in cases])
    values,status,work=evaluate(matrices,backend)
    assert not np.any(status),[(cases[i][0],status[i].tolist()) for i in np.flatnonzero(np.any(status,axis=1))]
    error=np.abs(values-expected)/np.maximum(1.,np.maximum(np.abs(values),np.abs(expected)))
    (tmp_path/'analytic.json').write_text(json.dumps(dict(names=[case[0] for case in cases],values=values.tolist(),
        expected=expected.tolist(),status=status.tolist(),maximum_scaled_error=float(error.max()),work=work.tolist()),indent=2))
    assert np.isfinite(values).all() and error.max()<=2e-10


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_vortex_independent_dense_reference(tmp_path,backend):
    rng=np.random.default_rng(20261103)
    a=rng.normal(size=(512,3,3))
    a=np.concatenate([a,.5*(a[:64]+a[:64].transpose(0,2,1)),np.triu(a[:64])])
    square=a@a
    expected=np.column_stack([np.linalg.eigvalsh(.5*(square+square.transpose(0,2,1)))[:,1],
                              np.abs(np.linalg.eigvals(a).imag).max(axis=1)])
    values,status,work=evaluate(a,backend)
    (tmp_path/'dense.json').write_text(json.dumps(dict(failures=np.argwhere(status).tolist(),
        maximum_work=work.max(axis=0).tolist(),matrices=len(a)),indent=2))
    assert not np.any(status),np.argwhere(status)
    error=np.abs(values-expected)/np.maximum(1.,np.maximum(np.abs(values),np.abs(expected)))
    assert np.isfinite(values).all() and error.max()<=2e-10,(np.unravel_index(error.argmax(),error.shape),error.max())


def test_vortex_cpu_gpu_same_matrix(tmp_path):
    rng=np.random.default_rng(17)
    a=np.concatenate([np.array([case[1] for case in analytic_cases()]),
        np.array([case[1] for case in ill_conditioned_cases()]),rng.normal(size=(2048,3,3))])
    cpu,cpu_status,cpu_work=evaluate(a,'cpu');gpu,gpu_status,gpu_work=evaluate(a,'gpu')
    np.testing.assert_array_equal(cpu_status,gpu_status)
    assert not np.any(cpu_status)
    error=np.abs(cpu-gpu)/np.maximum(1.,np.maximum(np.abs(cpu),np.abs(gpu)))
    (tmp_path/'cpu_gpu.json').write_text(json.dumps(dict(maximum_scaled_error=float(error.max()),
        maximum_absolute_error=float(np.abs(cpu-gpu).max()),cpu_maximum_work=cpu_work.max(axis=0).tolist(),
        gpu_maximum_work=gpu_work.max(axis=0).tolist()),indent=2))
    assert np.isfinite(error).all() and error.max()<=2e-10


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_vortex_scale_and_unrequested_values(tmp_path,backend):
    # Finite swirl is not rejected because the unrequested squared criterion overflows.
    a=np.array([[[0.,-1e200,0.],[1e200,0.,0.],[0.,0.,0.]],
                [[0.,-1e-200,0.],[1e-200,0.,0.],[0.,0.,0.]],
                [[1e100,-1.,0.],[1.,1e100,0.],[0.,0.,1e100]]])
    values,status,_=evaluate(a,backend,requested=2)
    np.testing.assert_array_equal(status,np.tile([4,0],(3,1)))
    np.testing.assert_allclose(values[:,1],[1e200,1e-200,1.],rtol=2e-10,atol=0)
    values,status,_=evaluate(a[:2],backend)
    np.testing.assert_array_equal(status,[[3,0],[3,0]])
    assert np.isnan(values[:,0]).all()
    shear=np.array([[[0.,1e200,0.],[-1e-200,0.,0.],[0.,0.,0.]]])
    values,status,_=evaluate(shear,backend,requested=1)
    np.testing.assert_array_equal(status,[[0,4]])
    np.testing.assert_allclose(values[:,0],[-1.],rtol=2e-10,atol=0)
    offset=1e100*np.eye(3)+np.array([[0.,-3.,2.],[3.,0.,-1.],[-2.,1.,0.]])
    values,status,_=evaluate(offset[None,...],backend,requested=2)
    np.testing.assert_array_equal(status,[[4,0]])
    np.testing.assert_allclose(values[:,1],[np.sqrt(14.)],rtol=2e-10,atol=0)
    values,status,_=evaluate(np.diag([1e300,1.,1e-300])[None,...],backend)
    np.testing.assert_array_equal(status,[[0,0]])
    np.testing.assert_array_equal(values,[[1.,0.]])


@pytest.mark.parametrize('backend',['cpu','gpu'])
@pytest.mark.parametrize('requested',[1,2,3])
def test_vortex_nonfinite_is_not_a_zero_field(backend,requested):
    a=np.tile(np.eye(3),(2,1,1));a[0,0,1]=np.nan;a[1,2,2]=np.inf
    values,status,_=evaluate(a,backend,requested=requested)
    assert np.isnan(values).all()
    np.testing.assert_array_equal(status,np.tile([1 if requested&1 else 4,1 if requested&2 else 4],(2,1)))


def ill_conditioned_cases():
    q=np.array([[-1.,2.,2.],[2.,-1.,2.],[2.,2.,-1.]])/3.
    cases=[]
    for name,base in [('shear',np.array([[0.,1.,0.],[0.,0.,0.],[0.,0.,0.]])),
                      ('jordan',np.array([[0.,1.,0.],[0.,0.,1.],[0.,0.,0.]]))]:
        a=q@base@q.T
        for exponent in (0,10):
            value=np.ldexp(a,exponent)
            cases.append((name+f'-{exponent}',value))
            perturbed=value.copy();perturbed[0,0]=np.nextafter(perturbed[0,0],np.inf)
            cases.append((name+f'-{exponent}-one-ulp',perturbed))
    return cases


def high_precision_reference(matrices):
    import mpmath as mp
    output=[]
    with mp.workdps(100):
        for a in matrices:
            exact=mp.matrix([[mp.mpf(float(a[i,j])) for j in range(3)] for i in range(3)])
            square=exact*exact
            if exact*square==mp.zeros(3):
                output.append([0.,0.]);continue
            symmetric=(square+square.T)/2
            values=mp.eigsy(symmetric,eigvals_only=True)
            general=mp.eig(exact,left=False,right=False)
            output.append([float(values[1]),float(max(abs(mp.im(value)) for value in general))])
    return np.array(output)


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_vortex_near_defective_reference(tmp_path,backend):
    cases=ill_conditioned_cases();a=np.array([matrix for _,matrix in cases])
    reference=high_precision_reference(a)
    values,status,work=evaluate(a,backend)
    error=np.abs(values-reference)/np.maximum(1.,np.maximum(np.abs(values),np.abs(reference)))
    (tmp_path/'near_defective.json').write_text(json.dumps(dict(names=[name for name,_ in cases],
        input=a.tolist(),reference=reference.tolist(),observed=values.tolist(),status=status.tolist(),
        work=work.tolist(),scaled_errors=error.tolist()),indent=2))
    assert not np.any(status),status
    assert np.isfinite(error).all() and error.max()<=2e-10,(cases[int(np.unravel_index(error.argmax(),error.shape)[0])][0],error.max())
