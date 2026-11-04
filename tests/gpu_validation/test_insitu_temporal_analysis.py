"""TB4 streaming spectrum, using the solver's native FFT rather than a Python integrator."""
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from scipy.signal import welch

PROBE=os.environ.get('ASTR_TEMPORAL_ANALYSIS_PROBE')
pytestmark=pytest.mark.skipif(not PROBE,reason='Select the root temporal-analysis probe')


def run(tmp_path,values,length=32,spacing=.01,split=0,continuous=None,uniform=None):
    values=np.asarray(values,dtype=float)
    if values.ndim==1: values=values[:,None]
    n,components=values.shape
    h=np.broadcast_to(spacing,(n,))
    contiguous=np.ones(n,dtype=int) if continuous is None else np.asarray(continuous,dtype=int)
    equal=np.ones(n,dtype=int) if uniform is None else np.asarray(uniform,dtype=int)
    path=tmp_path/f'input-{len(list(tmp_path.glob("input-*")))}.txt'
    with path.open('w') as stream:
        stream.write(f'{length} {components} {n} {split}\n')
        np.savetxt(stream,np.column_stack([h,contiguous,equal,values]),fmt=['%.17e','%d','%d',*['%.17e']*components])
    result=subprocess.run([str(Path(PROBE).resolve()),str(path)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    lines=result.stdout.splitlines();header=lines[0].split()
    assert header[0]=='META',result.stdout
    state=dict(segments=int(header[1]),filled=int(header[2]),generation=int(header[3]),
               uniform=int(header[4]),spacing=float(header[5]),bytes=int(header[6]))
    output=np.array([[float(v) for v in line.split()] for line in lines[1:]])
    return state,output,result.stdout


def reference(values,length,spacing):
    window=.5*(1-np.cos(2*np.pi*np.arange(length)/length))
    frequency,spectrum=welch(values,fs=1/spacing,window=window,nperseg=length,noverlap=length//2,
                             detrend='constant',scaling='density',return_onesided=True,axis=0)
    return np.column_stack([frequency,spectrum])


@pytest.mark.parametrize('length',[2,30,32,50])
def test_streaming_spectrum_matches_explicit_scipy_reference(tmp_path,length,record_property):
    n=7*length+5;t=np.arange(n)*.01
    values=np.column_stack([3+np.sin(2*np.pi*7.3*t),np.cos(2*np.pi*13*t)+.2*np.sin(2*np.pi*3*t)])
    state,actual,_=run(tmp_path,values,length)
    expected=reference(values,length,.01)
    error=np.abs(actual-expected)/np.maximum(1,np.maximum(abs(actual),abs(expected)))
    record_property('maximum_scaled_error',float(error.max()))
    assert error.max()<=2e-10
    assert state['segments']==1+(n-length)//(length//2)
    # Density scaling integrates to mean windowed, demeaned segment energy, not the whole-record variance.
    segments=np.stack([values[i:i+length] for i in range(0,n-length+1,length//2)])
    window=.5*(1-np.cos(2*np.pi*np.arange(length)/length))
    energy=np.mean(np.sum(((segments-segments.mean(axis=1,keepdims=True))*window[None,:,None])**2,axis=1)
                   /np.sum(window**2),axis=0)
    np.testing.assert_allclose(actual[:,1:].sum(axis=0)/(length*.01),energy,rtol=2e-10,atol=2e-10)


@pytest.mark.parametrize('split',[7,17,32,49])
def test_checkpoint_preserves_partial_segment_overlap_and_cumulative_psd(tmp_path,split):
    values=np.random.default_rng(410).normal(size=(257,2))
    _,_,full=run(tmp_path,values)
    _,_,restored=run(tmp_path,values,split=split)
    assert full==restored


def test_true_gap_discards_partial_segment_but_keeps_completed_estimates(tmp_path):
    values=np.random.default_rng(411).normal(size=(110,2))
    flags=np.ones(len(values),dtype=int);flags[51]=0
    state,actual,_=run(tmp_path,values,continuous=flags,split=62)
    expected=[];counts=[]
    for piece in (values[:51],values[51:]):
        expected.append(reference(piece,32,.01));counts.append(1+(len(piece)-32)//16)
    combined=expected[0].copy()
    combined[:,1:]=(expected[0][:,1:]*counts[0]+expected[1][:,1:]*counts[1])/sum(counts)
    np.testing.assert_allclose(actual,combined,rtol=2e-10,atol=2e-10)
    assert state['segments']==sum(counts)


def test_partial_segment_is_not_shortened_or_zero_padded(tmp_path):
    state,result,_=run(tmp_path,np.ones((31,2)),split=17)
    assert result.size==0 and state['segments']==0 and state['filled']==31


def test_variable_timestep_does_not_publish_a_welch_spectrum(tmp_path):
    values=np.random.default_rng(412).normal(size=(100,2))
    state,result,_=run(tmp_path,values,spacing=np.linspace(.01,.03,100),uniform=np.zeros(100),split=51)
    assert result.size==0 and state['segments']==0 and state['filled']==0 and state['uniform']==0


def test_changed_fixed_spacing_starts_a_new_frequency_grid(tmp_path):
    values=np.random.default_rng(413).normal(size=(120,2))
    spacing=np.r_[np.full(48,.01),np.full(72,.02)]
    state,result,_=run(tmp_path,values,spacing=spacing,split=65)
    np.testing.assert_allclose(result,reference(values[48:],32,.02),rtol=2e-10,atol=2e-10)
    assert state['generation']==2 and state['segments']==3


def test_buffer_memory_depends_on_segment_not_run_duration(tmp_path):
    a,_,_=run(tmp_path,np.zeros((300,2)))
    b,_,_=run(tmp_path,np.zeros((10000,2)))
    assert a['bytes']==b['bytes']==(32*2+32+17*2)*8


@pytest.mark.parametrize('split',[0,1,6,7,8,19])
def test_line_blocks_keep_actual_coordinates_times_gaps_and_partial_restart(tmp_path,split):
    capacity=7;nodes=9;fields=2;n=25
    coordinate=np.column_stack([np.linspace(0,1,nodes)**2,np.full(nodes,.01),np.zeros(nodes)])
    times=np.cumsum(np.linspace(.01,.023,n));steps=np.arange(n)
    times[13:]+=.2;steps[13:]+=4
    continuous=np.ones(n,dtype=int);continuous[13]=0
    values=np.stack([np.sin(coordinate[:,0][None,:]-3*times[:,None]),
                     np.cos(2*coordinate[:,0][None,:]+times[:,None])],axis=2)
    path=tmp_path/'blocks.txt'
    with path.open('w') as stream:
        stream.write(f'{capacity} {nodes} {fields} {n} {split}\n')
        stream.write(' '.join(f'{v:.17e}' for v in coordinate.ravel())+'\n')
        for step,time,flag,value in zip(steps,times,continuous,values):
            stream.write(f'{step} {time:.17e} {flag} '+' '.join(f'{v:.17e}' for v in value.T.ravel())+'\n')
    result=subprocess.run([str(Path(PROBE).resolve()),str(path),'blocks'],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    lines=result.stdout.splitlines()
    blocks=[list(map(int,line.split()[1:])) for line in lines if line.startswith('BLOCK ')]
    assert blocks==[[0,7],[1,7],[2,7],[3,4]]
    coordinates=np.array([[float(v) for v in line.split()[2:]] for line in lines if line.startswith('COORD ')])
    np.testing.assert_array_equal(coordinates,coordinate)
    data=np.array([[float(v) for v in line.split()[1:]] for line in lines if line.startswith('DATA ')])
    np.testing.assert_array_equal(data[:,0],steps)
    np.testing.assert_array_equal(data[:,1],times)
    assert np.flatnonzero(data[:,2]).tolist()==[13]
    np.testing.assert_array_equal(data[:,3:],values.transpose(0,2,1).reshape(n,-1))
    assert lines[-1]=='SAMPLES 25'
