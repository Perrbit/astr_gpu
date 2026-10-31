"""HB6 approved variable-property channel: local-work, decomposition, and lifecycle."""
from argparse import Namespace
import json
import os
from pathlib import Path

import numpy as np
import pytest

from run_hb6_channel_case import run
from run_output_restart_validation import compare_fields
from compare_flowstate import read_flowstate
from test_high_order_wall_filter_runtime import compare_checkpoints


EXE=os.environ.get('ASTR_HB1B_EXE')
pytestmark=pytest.mark.skipif(not EXE,reason='Set ASTR_HB1B_EXE to the root-CMake FP64 executable')


@pytest.fixture(scope='module')
def channel(tmp_path_factory):
    root=tmp_path_factory.mktemp('hb6_joint')
    completed={}
    def launch(backend,grid=(32,32,16),topology=(1,1,1),steps=12,
               perturbation=0.,workspace='scalar',restore=None,wall=41,snapshots=False):
        key=(backend,grid,topology,steps,perturbation,workspace,str(restore),wall,snapshots)
        if key not in completed:
            directory=root/f'{backend}_bc{wall}_{len(completed):03d}'
            args=Namespace(executable=Path(EXE),output=directory,backend=backend,grid=grid,
                           topology=topology,steps=steps,dt=None,perturbation=perturbation,
                           workspace=workspace,restore=restore,wall=wall,snapshots=snapshots,
                           checkpoint_interval=1,wall_mode='hermite7')
            record=run(args)
            completed[key]=(directory,Path(record['checkpoint']),record)
        return completed[key]
    return launch


@pytest.mark.parametrize('grid',[(32,32,16),(64,64,32),(128,128,64)])
def test_reference_background_cpu_gpu(channel,grid):
    cpu,a,_=channel('cpu',grid)
    case,b,report=channel('gpu',grid)
    errors=compare_checkpoints(a,b)
    ha,va=read_flowstate(cpu); hb,vb=read_flowstate(case)
    assert ha==hb and va.shape==vb.shape
    errors['flow_statistics']=float(np.max(abs(va-vb)/(1+abs(va))))
    assert max(errors.values())<=2e-10,errors
    (case/'backend_errors.json').write_text(json.dumps(errors,indent=2)+'\n')
    assert .19<=report['maximum_cfl']<=.21


@pytest.mark.parametrize('topology',[(2,1,1),(1,2,1),(1,1,2)])
@pytest.mark.parametrize('wall',[41,42,421])
def test_decomposed_background(channel,topology,wall):
    _,a,_=channel('cpu',topology=topology,wall=wall)
    case,b,_=channel('gpu',topology=topology,wall=wall)
    _,single,_=channel('gpu',wall=wall)
    errors=compare_checkpoints(a,b)
    # Decomposition changes auxiliary layout, but not global physical fields.
    import h5py
    with h5py.File(single/'state.h5') as x,h5py.File(b/'state.h5') as y:
        for m in range(1,12):
            p,q=x[f'q{m:04d}'][:],y[f'q{m:04d}'][:]
            errors[f'decomposition_q{m}']=float(np.max(abs(p-q)/(1+abs(p))))
    assert max(errors.values())<=2e-10,errors
    (case/'backend_decomposition_errors.json').write_text(json.dumps(errors,indent=2)+'\n')


def test_perturbed_channel_and_storage(channel):
    _,a,_=channel('cpu',perturbation=1e-4)
    _,b,_=channel('gpu',perturbation=1e-4)
    _,full,_=channel('gpu',perturbation=1e-4,workspace='full')
    compare_checkpoints(a,b)
    compare_checkpoints(b,full)


@pytest.mark.parametrize('wall',[41,42,421])
@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_forced_or_unforced_exact_restart(channel,backend,wall):
    topology=(1,2,1)
    _,whole,_=channel(backend,topology=topology,wall=wall)
    _,part,_=channel(backend,topology=topology,steps=5,wall=wall)
    _,resumed,_=channel(backend,topology=topology,restore=part,wall=wall)
    compare_fields(whole/'state.h5',resumed/'state.h5')
    assert (whole/'control.bin').read_bytes()==(resumed/'control.bin').read_bytes()


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_reference_diagnostics_are_read_only(channel,backend):
    _,plain,_=channel(backend)
    _,sampled,report=channel(backend,snapshots=True)
    compare_fields(plain/'state.h5',sampled/'state.h5')
    assert max(stage['reconstruction_scaled'] for stage in report['first_step_budget']['stages'])<=2e-10
