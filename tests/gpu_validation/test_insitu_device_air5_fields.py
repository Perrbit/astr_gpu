"""Completed-step AIR5 device fields against the existing CPU wall diagnostic.

The explicit test-only device download is separate from strict image admission.
"""
import re

import h5py
import numpy as np
import pytest

from run_output_air5_restart_validation import launch
from run_output_restart_validation import compare_fields
from test_insitu_air5_walls import (arguments, global_fields, TOPOLOGIES, FINAL,
    FIELD_NAMES, FIELD_SCALES, CHECKPOINT_SCALES)
from test_output_insitu_restart import ROOT


def current_arguments(output, topology):
    args=arguments(output,topology)
    args.executable=ROOT/'build_insitu_air5_device_render/bin/astr'
    args.runtime_timeout_seconds=300
    return args


def checks(case, topology, steps, mode):
    ranks=int(np.prod(topology)); log=(case/'run.log').read_text()
    errors=re.findall(r'ASTR_INSITU_DEVICE_AIR5_WALL_ERRORS rank=(\d+) raw_si_maxabs=([^\n]+)',log)
    assert len(errors)==ranks*steps,log
    raw=np.zeros(18)
    for _,row in errors:
        values=np.array([float(value) for value in row.split()])
        assert values.shape==(18,) and np.isfinite(values).all()
        raw=np.maximum(raw,values)
    assert np.max(raw/FIELD_SCALES)<=2e-10,raw
    entries=re.findall(r'ASTR_INSITU_DEVICE_WALL_CHECK rank=(\d+) backend=(\S+) maxabs=\s*(\S+) '
        r'oracle_download_bytes=(\d+) face_d2h_bytes=(\d+) face_h2d_bytes=(\d+)',log)
    assert len(entries)==ranks*steps
    for rank,transport,error,download,d2h,h2d in entries:
        assert transport==mode and np.isfinite(float(error)) and float(error)<=2e-10
        empty=topology==(1,2,1) and int(rank)==1
        assert (int(download)==0)==empty
        if mode=='device-aware' or empty:
            assert int(d2h)==int(h2d)==0
    if ranks==2 and mode=='device-aware':
        assert all(f'ASTR_GPU_PRE_MPI_DEVICE={rank} local_rank={rank}' in log for rank in range(2))
    return raw


@pytest.fixture(scope='module',params=TOPOLOGIES,ids=('single','x','y','z'))
def reference(request,tmp_path_factory):
    topology=request.param; ranks=int(np.prod(topology))
    args=current_arguments(tmp_path_factory.mktemp('air5_device_reference'),topology)
    cpu,_=launch(args,'cpu',ranks,'cpu',4,interval=1,wall_samples=True)
    return topology,cpu,global_fields(cpu,ranks)


@pytest.mark.parametrize('mode',('pinned','device-aware'))
def test_real_air5_device_fields(reference,tmp_path,mode,record_property):
    topology,cpu,expected=reference; ranks=int(np.prod(topology))
    args=current_arguments(tmp_path,topology)
    gpu,_=launch(args,'gpu',ranks,'fields',4,interval=1,wall_samples=True,device_sample_transport=mode)
    raw=checks(gpu,topology,4,mode)
    actual=global_fields(gpu,ranks)
    difference=np.max(abs(actual-expected),axis=(0,1))
    assert np.max(difference/FIELD_SCALES)<=2e-10,difference
    cache_indices=[23,24,25,26,28,29,27,30,31,32,33,34]
    for case,fields in ((cpu,expected),(gpu,actual)):
        with h5py.File(case/FINAL/'state.h5') as state:
            cached=np.stack([state[f'q{m:04d}'][...].transpose(2,1,0)[:,0,:16]
                for m in cache_indices],axis=-1)
        np.testing.assert_allclose(fields[...,:12],cached,atol=2e-10,rtol=0)
    with h5py.File(cpu/FINAL/'state.h5') as a,h5py.File(gpu/FINAL/'state.h5') as b:
        state_errors=np.array([np.max(abs(a[f'q{m:04d}'][...]-b[f'q{m:04d}'][...])) for m in range(1,35)])
    assert np.isfinite(state_errors).all() and np.max(state_errors/CHECKPOINT_SCALES)<=2e-10,state_errors
    record_property('topology',topology)
    record_property('transport',mode)
    record_property('same_state_device_reference_scaled_maxabs',float(np.max(raw/FIELD_SCALES)))
    record_property('cpu_gpu_reference_scaled_maxabs',float(np.max(difference/FIELD_SCALES)))
    record_property('complete_state_cache_reference_scaled_maxabs',float(np.max(state_errors/CHECKPOINT_SCALES)))
    for name,scale,oracle_error,field_error in zip(FIELD_NAMES,FIELD_SCALES,raw,difference):
        record_property(name+'_reference_scale',float(scale))
        record_property(name+'_device_cpu_oracle_si_maxabs',float(oracle_error))
        record_property(name+'_cpu_gpu_si_maxabs',float(field_error))
    source=gpu/'outdat/new/checkpoints/step000000000003'
    before={p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
    restarted,_=launch(args,'gpu',ranks,'restart',4,restore=source,interval=1,wall_samples=True,
        device_sample_transport=mode)
    checks(restarted,topology,1,mode)
    compare_fields(gpu/FINAL/'state.h5',restarted/FINAL/'state.h5')
    for name in ('control.bin','insitu_control.bin','air5_config.bin','air5_conservation.bin'):
        assert (gpu/FINAL/name).read_bytes()==(restarted/FINAL/name).read_bytes()
    for rank in range(ranks):
        name=f'outdat/sample.air5_wall.step00000004.rank{rank:08d}.bin'
        assert (gpu/name).read_bytes()==(restarted/name).read_bytes()
    assert before=={p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
    off,_=launch(args,'gpu',ranks,'off',4,interval=1)
    compare_fields(gpu/FINAL/'state.h5',off/FINAL/'state.h5')


@pytest.mark.parametrize('topology,mode',[((1,2,1),'device-aware'),((2,1,1),'pinned')])
def test_real_air5_device_fields_memcheck(tmp_path,topology,mode):
    case,_=launch(current_arguments(tmp_path,topology),'gpu',2,'memcheck',2,interval=1,
        wall_samples=True,device_sample_transport=mode,memcheck=True)
    checks(case,topology,2,mode)
    assert len(list(case.glob('memcheck.*.log')))==2
