"""Opt-in AIR5 image-only transfer/resource observation after numerical gates."""
import json
import os
import hashlib
from pathlib import Path
import re

import numpy as np
import pytest

from run_output_air5_restart_validation import launch
from run_output_restart_validation import compare_fields
from test_insitu_air5_walls import TOPOLOGIES
from test_insitu_device_air5_fields import current_arguments
from test_insitu_device_air5_render import configuration
from test_insitu_x4_observation import no_large_io, timing, ledger

pytestmark=pytest.mark.skipif(os.environ.get('ASTR_INSITU_X4_OBSERVE')!='1',
    reason='Select the approved, numerically validated AIR5 X4 executable')


@pytest.mark.parametrize('topology',TOPOLOGIES,ids=('single','x','y','z'))
@pytest.mark.parametrize('mode',('pinned','device-aware'))
@pytest.mark.parametrize('pipeline',('standard-device','direct-device'))
def test_actual_air5_observation(tmp_path,topology,mode,pipeline,record_property):
    ranks=int(np.prod(topology)); args=current_arguments(tmp_path,topology)
    config=configuration(pipeline,mode); disabled='&insitu_run enabled=f /\n'
    common=dict(checkpoint_enabled=False,insitu_timing=True,directory_budget_bytes=256*1024**2)
    off,_=launch(args,'gpu',ranks,'off',4,insitu_config=disabled,postprocess_transport=mode,**common)
    on,_=launch(args,'gpu',ranks,'on',4,insitu_config=config,postprocess_transport=mode,**common)
    for case in (off,on):
        no_large_io(case)
    off_timing,on_timing=timing(off,ranks),timing(on,ranks)
    record_property('topology',topology)
    record_property('pipeline',pipeline)
    record_property('transport',mode)
    record_property('matched_cost',json.dumps(dict(off=off_timing,on=on_timing,
        extra_completed_window_seconds=on_timing['completed_window']['seconds']-off_timing['completed_window']['seconds'],
        scope='one unprofiled four-step pair; initialization excluded; first-frame setup included; nested phases not summed')))
    baseline_case,_=launch(args,'gpu',ranks,'resource_off',2,insitu_config=disabled,
        postprocess_transport=mode,monitor_resources=True,**common)
    baseline=json.loads((baseline_case/'resources.sampled.json').read_text())
    observed,size=launch(args,'gpu',ranks,'resource_on',2,insitu_config=config,
        postprocess_transport=mode,monitor_resources=True,resource_baseline=baseline,**common)
    report=json.loads((observed/'resources.sampled.json').read_text())
    assert report['samples']>0 and report['sampling_period_seconds']==.02
    assert report['additional_host_peak_difference_bytes']<=4*1024**3
    assert all(value<=2*1024**3 for value in report['additional_device_peak_difference_bytes'].values())
    assert all(device['min_free_bytes']>=1024**3 for device in report['devices'].values())
    for case in (baseline_case,observed):
        no_large_io(case)
    trace=existing_trace(args.executable,topology,mode,pipeline,config,observed)
    if trace is None:
        trace,_=launch(args,'gpu',ranks,'trace',2,insitu_config=config,postprocess_transport=mode,
            nsys_trace=True,pixel_audit=True,**common)
    else:
        record_property('trace_reused_from',str(trace))
    no_large_io(trace)
    record_property('actual_transfer_ledger',json.dumps(ledger(trace,'air5-wall',mode,ranks)))
    record_property('external_20ms_resources',json.dumps(report))
    record_property('observed_directory_bytes',size)


def existing_trace(executable,topology,mode,pipeline,config,input_reference):
    roots=os.environ.get('ASTR_INSITU_AIR5_TRACE_ROOTS')
    if not roots:
        return None
    # This optional reuse is limited to the explicitly frozen X4 producer.
    assert hashlib.sha256(executable.read_bytes()).hexdigest()==(
        '5199203183cbe1d3faae5cc9bc03fc665ba257842b747640a60f04e11ecdba83')
    traces=sorted(p for root in roots.split(os.pathsep)
        for p in Path(root).glob('test_actual_air5_observation_*/gpu_np*_trace')
        if not p.parent.is_symlink())
    assert len(traces)==16
    matches=[]
    for trace in traces:
        log=(trace/'run.log').read_text()
        shape=tuple(map(int,re.search(r'forced mpi topology=\s*(\d+)\s+(\d+)\s+(\d+)',log).groups()))
        transport=re.search(r'ASTR_INSITU_DEVICE_WALL_FRAME .*?transport=(\S+)',log).group(1)
        receipt=json.loads((trace/'outdat/render/mesh_step00000002_rank0.json').read_text())
        if (shape,transport,receipt['rendering_pipeline'])==(topology,mode,pipeline):
            assert receipt['profile']=='air5_walls'
            assert (trace/'datin/insitu.nml').read_text()==config
            actual={p.relative_to(trace/'datin'):p.read_bytes()
                for p in (trace/'datin').rglob('*') if p.is_file()}
            expected={p.relative_to(input_reference/'datin'):p.read_bytes()
                for p in (input_reference/'datin').rglob('*') if p.is_file()}
            assert actual.keys()==expected.keys(),'Frozen trace input inventory differs'
            for name in actual:
                if name==Path('grid.h5'):
                    # The internal grid writer may serialize different HDF5 metadata.
                    compare_fields(trace/'datin'/name,input_reference/'datin'/name)
                else:
                    assert actual[name]==expected[name],('Frozen trace input differs',name)
            assert sum(p.stat().st_size for p in trace.rglob('*') if p.is_file())<=256*1024**2
            matches.append(trace)
    assert len(matches)==1,(topology,mode,pipeline,matches)
    return matches[0]


def test_existing_air5_trace_input_identity(record_property):
    root=os.environ.get('ASTR_INSITU_AIR5_MATCHED_ROOT')
    if not root:
        pytest.skip('Select the immutable matched-cost/resource matrix')
    cases=sorted(p for p in Path(root).glob('test_actual_air5_observation_*/gpu_np*_resource_on')
        if not p.parent.is_symlink())
    assert len(cases)==16
    identities=[]
    for case in cases:
        log=(case/'run.log').read_text()
        shape=tuple(map(int,re.search(r'forced mpi topology=\s*(\d+)\s+(\d+)\s+(\d+)',log).groups()))
        mode=re.search(r'ASTR_INSITU_DEVICE_WALL_FRAME .*?transport=(\S+)',log).group(1)
        receipt=json.loads((case/'outdat/render/mesh_step00000002_rank0.json').read_text())
        pipeline=receipt['rendering_pipeline']
        assert existing_trace(current_arguments(case,shape).executable,shape,mode,pipeline,
            (case/'datin/insitu.nml').read_text(),case).is_dir()
        identities.append((shape,mode,pipeline))
    expected={(shape,mode,pipeline) for shape in TOPOLOGIES
        for mode in ('pinned','device-aware') for pipeline in ('standard-device','direct-device')}
    assert len(set(identities))==16 and set(identities)==expected
    record_property('immutable_trace_input_sets_verified',len(identities))
