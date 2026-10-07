"""Actual bc41 completed-step fields, not a strict rendering admission gate.

GPU oracle downloads are explicitly logged; solver snapshots remain unchanged.
"""
import os
from pathlib import Path
import re

import numpy as np
import pytest

from run_output_restart_validation import run_case, compare_fields
from test_insitu_channel_walls import arguments, check_wall, FINAL, ROOT
from test_insitu_is4 import compare_numerical


def current_arguments(output, samples=True):
    args=arguments(output, 'gpu', samples=samples)
    args.executable=Path(os.environ.get('ASTR_OUTPUT_INSITU_EXE',ROOT/'build_insitu_device_render/bin/astr'))
    return args


def checks(case, ranks, steps, mode):
    log=(case/'run.log').read_text()
    matches=re.findall(r'ASTR_INSITU_DEVICE_WALL_CHECK rank=(\d+) backend=(\S+) maxabs=\s*(\S+) '
        r'oracle_download_bytes=(\d+) face_d2h_bytes=(\d+) face_h2d_bytes=(\d+)',log)
    assert len(matches)==ranks*steps,log
    errors=[]
    for rank,backend,error,download,d2h,h2d in matches:
        assert backend==mode and np.isfinite(float(error)) and float(error)<=2e-10
        assert int(download)>0
        if mode=='device-aware':
            assert int(d2h)==int(h2d)==0
        errors.append(float(error))
    if mode=='device-aware' and ranks==2:
        assert 'ASTR_GPU_PRE_MPI_DEVICE=0 local_rank=0' in log
        assert 'ASTR_GPU_PRE_MPI_DEVICE=1 local_rank=1' in log
    return max(errors)


@pytest.mark.parametrize('topology',[(1,1,1),(2,1,1),(1,2,1),(1,1,2)],ids=['single','x','y','z'])
@pytest.mark.parametrize('mode',['pinned','device-aware'])
def test_real_bc41_device_fields(tmp_path, topology, mode, record_property):
    ranks=int(np.prod(topology)); args=current_arguments(tmp_path)
    gpu,_=run_case(args,ROOT,'gpu',ranks,'fields',4,topology=topology,checkpoint_interval=1,
        device_sample_transport=mode)
    error=checks(gpu,ranks,4,mode)
    actual,oracle_error=check_wall(gpu,ranks,4)
    cpu,_=run_case(args,ROOT,'cpu',ranks,'cpu',4,topology=topology,checkpoint_interval=1)
    expected,cpu_oracle_error=check_wall(cpu,ranks,4)
    field_error=float(np.max(abs(actual-expected)))
    assert field_error<=2e-10
    record_property('same_state_device_wall_maxabs',error)
    record_property('independent_wall_maxabs',max(oracle_error,cpu_oracle_error))
    record_property('cpu_gpu_wall_maxabs',field_error)
    record_property('cpu_gpu_state_maxabs',compare_numerical(cpu/FINAL/'state.h5',gpu/FINAL/'state.h5'))
    source=gpu/'outdat/new/checkpoints/step000000000003'
    restarted,_=run_case(args,ROOT,'gpu',ranks,'restart',4,restore=source,topology=topology,
        checkpoint_interval=1,device_sample_transport=mode)
    checks(restarted,ranks,1,mode)
    for name in ('state.h5','control.bin','insitu_control.bin'):
        if name.endswith('.h5'):
            compare_fields(gpu/FINAL/name,restarted/FINAL/name)
        else:
            assert (gpu/FINAL/name).read_bytes()==(restarted/FINAL/name).read_bytes()
    for rank in range(ranks):
        name=f'sample.wall.step00000004.rank{rank:08d}.bin'
        assert (gpu/'outdat'/name).read_bytes()==(restarted/'outdat'/name).read_bytes()
    args=current_arguments(tmp_path,samples=False)
    off,_=run_case(args,ROOT,'gpu',ranks,'off',4,topology=topology,checkpoint_interval=1)
    compare_fields(gpu/FINAL/'state.h5',off/FINAL/'state.h5')


def test_real_bc41_device_fields_memcheck(tmp_path):
    args=current_arguments(tmp_path)
    case,_=run_case(args,ROOT,'gpu',2,'memcheck',2,topology=(1,2,1),checkpoint_interval=1,
        device_sample_transport='device-aware',memcheck=True)
    checks(case,2,2,'device-aware')
    reports=list(case.glob('memcheck.*.log'))
    assert len(reports)==2
    for report in reports:
        assert 'ERROR SUMMARY: 0 errors' in report.read_text(),report
