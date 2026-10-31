"""AP4 host-known diagnostic dispatch, without changing chemistry or failures."""
import json

import pytest

from run_output_air5_restart_validation import launch, state_sanity
from run_output_restart_validation import compare_fields
from test_output_air5_conservation import arguments
from test_air5_launch_learning import check_complete_learning


def options(tmp_path,case='sbli',axis='x',workspace='scalar'):
    args=arguments(tmp_path,case=case,axis=axis,workspace=workspace)
    args.air5_filter_mode='conservative-flux'
    args.primitive_reuse='chemistry'
    args.chemistry_reductions='packed'
    args.report_interval=4
    args.runtime_timeout_seconds=600
    return args


@pytest.mark.parametrize('ranks,axis,case,evaluation,workspace',[
    (1,'x','sbli','joint','scalar'),(2,'x','sbli','source-only','scalar'),
    (2,'y','hbl','joint','scalar'),(2,'z','sbli','source-only','full')])
def test_dispatch_preserves_fields_reports_and_restart(tmp_path,ranks,axis,case,evaluation,workspace):
    args=options(tmp_path,case,axis,workspace)
    args.source_evaluation=evaluation
    general,_=launch(args,'gpu',ranks,'general',24,interval=24,chemistry_diagnostics='report-steps')
    first,_=launch(args,'gpu',ranks,'first',17,interval=17,chemistry_diagnostics='report-steps')
    args.chemistry_dispatch='specialized'
    special,_=launch(args,'gpu',ranks,'specialized',24,interval=24,chemistry_diagnostics='report-steps')
    seed=first/'outdat/new/checkpoints/step000000000017'
    resumed,_=launch(args,'gpu',ranks,'resumed',24,restore=seed,interval=24,override=True,
                     chemistry_diagnostics='report-steps')
    final='outdat/new/checkpoints/step000000000024'
    for name in ('state.h5','statistics.h5'):
        compare_fields(general/final/name,special/final/name)
        compare_fields(special/final/name,resumed/final/name)
    reports=lambda case:[row for row in (case/'run.log').read_text().splitlines()
                             if row.startswith('AIR5_CHEMISTRY half=')]
    assert reports(general)==reports(special)
    assert (general/'flowstate.dat').read_bytes()==(special/'flowstate.dat').read_bytes()
    assert (general/final/'air5_config.bin').read_bytes()==(resumed/final/'air5_config.bin').read_bytes()
    (tmp_path/'dispatch.json').write_text(json.dumps(dict(evaluation=evaluation,ranks=ranks,axis=axis,
        scope='24-step software/numerical check; not physical acceptance',
        sanity=state_sanity(resumed/final/'state.h5')),indent=2,allow_nan=False)+'\n')


def test_specialized_dispatch_composes_with_auto_and_exact_restart(tmp_path):
    args=options(tmp_path)
    args.source_evaluation='source-only'
    args.rank_output=True
    general,_=launch(args,'gpu',1,'general66',66,interval=66,chemistry_diagnostics='report-steps')
    args.chemistry_dispatch='specialized'
    args.launch_tuning='auto'
    special,_=launch(args,'gpu',1,'auto66',66,interval=66,chemistry_diagnostics='report-steps')
    first,_=launch(args,'gpu',1,'first33',33,interval=33,chemistry_diagnostics='report-steps')
    resumed,_=launch(args,'gpu',1,'resumed66',66,restore=first/'outdat/new/checkpoints/step000000000033',
                     interval=66,override=True,chemistry_diagnostics='report-steps')
    final='outdat/new/checkpoints/step000000000066'
    for name in ('state.h5','statistics.h5'):
        compare_fields(general/final/name,special/final/name)
        compare_fields(special/final/name,resumed/final/name)
    record=check_complete_learning(special,1,'x')
    record['restart']=check_complete_learning(resumed,1,'x',33)
    (tmp_path/'dispatch_auto.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')


def test_specialized_dispatch_memcheck(tmp_path):
    args=options(tmp_path,case='hbl',axis='y')
    args.source_evaluation='source-only'
    args.chemistry_dispatch='specialized'
    args.launch_tuning='auto'
    args.runtime_timeout_seconds=1800
    launch(args,'gpu',2,'memcheck',22,interval=22,memcheck=True,chemistry_diagnostics='report-steps')
