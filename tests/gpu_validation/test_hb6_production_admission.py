"""Actual non-testing HB6 admission, compared with the validated testing build."""
from argparse import Namespace
import json
import os
from pathlib import Path

import pytest

from run_hb6_channel_case import run
from run_output_restart_validation import compare_fields
from test_high_order_wall_filter_runtime import compare_checkpoints


PRODUCTION=os.environ.get('ASTR_HB6_PRODUCTION_EXE')
REFERENCE=os.environ.get('ASTR_HB1B_EXE')
pytestmark=pytest.mark.skipif(not(PRODUCTION and REFERENCE),reason='Set HB6 production and testing executable paths')


@pytest.fixture(scope='module')
def launch(tmp_path_factory):
    root=tmp_path_factory.mktemp('hb6_non_testing')
    completed={}
    def execute(build,backend,wall=41,topology=(1,1,1),mode='hermite7',restore=None,steps=12,workspace='scalar'):
        key=(build,backend,wall,topology,mode,str(restore),steps,workspace)
        if key not in completed:
            path=root/f'{build}_{backend}_bc{wall}_{len(completed):03d}'
            args=Namespace(executable=Path(PRODUCTION if build=='production' else REFERENCE),
                           output=path,backend=backend,grid=(32,32,16),topology=topology,
                           steps=steps,dt=None,perturbation=0.,workspace=workspace,restore=restore,
                           wall=wall,snapshots=False,checkpoint_interval=1,wall_mode=mode)
            report=run(args)
            text=(path/'run.log').read_text()
            if build=='production' and mode=='hermite7':
                assert 'local-f6-v1 (FP64 Cartesian subset)' in text
            if mode in ('off','default'):
                assert 'high-order wall filter:' not in text
            completed[key]=(path,Path(report['checkpoint']))
        return completed[key]
    return execute


@pytest.mark.parametrize('wall',[41,42,421])
@pytest.mark.parametrize('topology',[(1,1,1),(1,2,1)])
def test_non_testing_fields_match_validated_build(launch,wall,topology):
    states={}
    for backend in ('cpu','gpu'):
        _,reference=launch('testing',backend,wall,topology)
        directory,production=launch('production',backend,wall,topology)
        errors=compare_checkpoints(reference,production)
        (directory/'testing_build_comparison.json').write_text(json.dumps(errors,indent=2)+'\n')
        states[backend]=production
    compare_checkpoints(states['cpu'],states['gpu'])


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_non_testing_default_is_off(launch,backend):
    _,default=launch('production',backend,mode='default')
    _,off=launch('production',backend,mode='off')
    compare_fields(default/'state.h5',off/'state.h5')
    assert (default/'control.bin').read_bytes()==(off/'control.bin').read_bytes()


@pytest.mark.parametrize('backend',['cpu','gpu'])
@pytest.mark.parametrize('wall',[41,42,421])
def test_non_testing_same_algorithm_exact_restart(launch,backend,wall):
    _,whole=launch('production',backend,wall,topology=(1,2,1))
    _,part=launch('production',backend,wall,topology=(1,2,1),steps=5)
    _,resumed=launch('production',backend,wall,topology=(1,2,1),restore=part)
    compare_fields(whole/'state.h5',resumed/'state.h5')
    assert (whole/'control.bin').read_bytes()==(resumed/'control.bin').read_bytes()


@pytest.mark.parametrize('wall',[41,42,421])
def test_non_testing_storage_paths(launch,wall):
    _,scalar=launch('production','gpu',wall,topology=(1,2,1))
    directory,full=launch('production','gpu',wall,topology=(1,2,1),workspace='full')
    errors=compare_checkpoints(scalar,full)
    (directory/'storage_comparison.json').write_text(json.dumps(errors,indent=2)+'\n')
