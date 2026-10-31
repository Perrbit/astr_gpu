"""AP6 same-policy restart and real solver lifecycle; no inexact migration."""
from copy import copy
import json
import os
from pathlib import Path
import re

import h5py
import numpy as np
import pytest

from air5_checkpoint_contract import read_air5_contract
from run_output_air5_restart_validation import launch,state_sanity
from run_output_restart_validation import compare_fields
from test_air5_stage_coordination_holdout import check_statistics_finite
from test_output_air5_conservation import arguments


@pytest.mark.parametrize('backend,ranks,axis,case,source,workspace,history,compensation',[
    ('gpu',1,'x','hbl','frozen','scalar','previous','on'),
    ('gpu',2,'y','sbli','coupled','full','previous','on'),
    ('cpu',2,'z','hbl','frozen','scalar','previous','on'),
    ('gpu',2,'x','hbl','frozen','full','off','off')])
def test_same_threshold_exact_restart(tmp_path,backend,ranks,axis,case,source,workspace,history,compensation):
    args=arguments(tmp_path,case=case,axis=axis,workspace=workspace)
    args.low_activity=os.environ.get('ASTR_AIR5_ACTIVITY_TEST_THRESHOLD','1e-4')
    args.source_mode=source
    args.air5_filter_mode='conservative-flux'
    args.step_history=history
    args.compensation=compensation
    args.report_interval=4
    args.runtime_timeout_seconds=900
    whole,_=launch(args,backend,ranks,'whole',12,interval=12)
    first,_=launch(args,backend,ranks,'first',5,interval=5)
    seed=first/'outdat/new/checkpoints/step000000000005'
    resumed,_=launch(args,backend,ranks,'resumed',12,restore=seed,interval=12,override=True)
    final=Path('outdat/new/checkpoints/step000000000012')
    equal={name:compare_fields(whole/final/name,resumed/final/name) for name in ('state.h5','statistics.h5')}
    check_statistics_finite(resumed/final/'statistics.h5')
    assert (whole/final/'air5_config.bin').read_bytes()==(resumed/final/'air5_config.bin').read_bytes()
    contract=read_air5_contract((resumed/final/'air5_config.bin').read_bytes())
    assert contract['activity_threshold']==float(args.low_activity) and contract['activity_version']==1
    calls=[tuple(map(int,row)) for row in re.findall(
        r'AIR5_ACTIVITY half=(\d+) skipped_calls=(\d+) total_calls=(\d+)',(whole/'run.log').read_text())]
    assert calls and {row[0] for row in calls}=={1,2}
    if source=='frozen':
        assert all(skipped==total and total>0 for _,skipped,total in calls)
        with h5py.File(resumed/final/'state.h5') as state:
            if history=='previous':
                assert np.all(state['q0035'][:]==0)
            else:
                assert 'q0035' not in state
    changed=copy(args)
    changed.low_activity='off'
    launch(changed,backend,ranks,'changed',12,restore=seed,
        reject='AIR5 activity control contract mismatch')
    (tmp_path/'activity_restart.json').write_text(json.dumps(dict(
        backend=backend,ranks=ranks,axis=axis,source_mode=source,threshold=float(args.low_activity),
        history=history,compensation=compensation,
        equal_datasets=equal,reported_activity_calls=calls,sanity=state_sanity(resumed/final/'state.h5')),
        indent=2,allow_nan=False)+'\n')


@pytest.mark.parametrize('source',['coupled','frozen'])
def test_activity_mpi_memcheck(tmp_path,source):
    args=arguments(tmp_path,case='hbl',axis='y')
    args.low_activity='1e-4'
    args.source_mode=source
    args.air5_filter_mode='conservative-flux'
    args.step_history='previous'
    args.chemistry_dispatch='specialized'
    args.launch_tuning='auto'
    args.report_interval=4
    args.runtime_timeout_seconds=1800
    launch(args,'gpu',2,'memcheck',10,interval=10,memcheck=True,chemistry_diagnostics='report-steps')


def test_activity_coupled_cpu_gpu_fields(tmp_path):
    args=arguments(tmp_path,case='sbli')
    args.low_activity=os.environ.get('ASTR_AIR5_AP_ACTIVITY_THRESHOLD','1e-4')
    args.air5_filter_mode='conservative-flux'
    args.step_history='previous'
    args.runtime_timeout_seconds=900
    cpu,_=launch(args,'cpu',1,'cpu',12,interval=12)
    gpu,_=launch(args,'gpu',1,'gpu',12,interval=12)
    final=Path('outdat/new/checkpoints/step000000000012/state.h5')
    differences={}
    with h5py.File(cpu/final) as a,h5py.File(gpu/final) as b:
        for index in range(1,36):
            name=f'q{index:04d}'
            aa,bb=a[name][:],b[name][:]
            assert np.isfinite(aa).all() and np.isfinite(bb).all()
            scaled=float(np.max(abs(aa-bb)/(1+abs(aa))))
            differences[name]=dict(absolute=float(np.max(abs(aa-bb))),scaled=scaled)
            assert scaled<=2e-10,(name,differences[name])
    (tmp_path/'cpu_gpu_fields.json').write_text(json.dumps(dict(
        threshold=float(args.low_activity),differences=differences),indent=2,allow_nan=False)+'\n')
