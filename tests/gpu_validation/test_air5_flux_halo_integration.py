"""AP5 copies flux faces only; compare full states and exact continuation."""
import json
import os

import pytest

from run_output_air5_restart_validation import launch,state_sanity
from run_output_restart_validation import compare_fields
from test_output_air5_conservation import arguments


CASES=[(1,'x','hbl','full_state','packed','pinned',None),
    (1,'x','sbli','layered','packed','pinned',None),
    (2,'x','sbli','layered','packed-overlap','pinned',None),
    (2,'y','hbl','layered','packed','pinned',None),
    (2,'z','sbli','layered','packed-overlap','pinned',None),
    (2,'x','sbli','layered','packed-overlap','device-aware',None),
    (4,'x','sbli','layered','packed-overlap','pinned',(2,2,1)),
    (4,'x','hbl','layered','packed-overlap','device-aware',(2,1,2))]


def options(tmp_path,case,axis,backend='pinned',diffusion='layered',topology=None):
    args=arguments(tmp_path,case=case,axis=axis)
    args.air5_filter_mode='conservative-flux'
    args.source_evaluation='source-only'
    args.chemistry_block_size=32
    args.primitive_reuse='chemistry'
    args.chemistry_reductions='packed'
    args.halo_transport=backend
    args.diffusion_limiter=diffusion
    args.topology=topology
    args.share_gpus=topology is not None
    args.report_interval=4
    args.runtime_timeout_seconds=900
    return args


@pytest.mark.parametrize('ranks,axis,case,diffusion,mode,backend,topology',CASES)
def test_flux_halo_preserves_full_fields_and_restart(tmp_path,ranks,axis,case,diffusion,mode,backend,topology):
    args=options(tmp_path,case,axis,backend,diffusion,topology)
    baseline,_=launch(args,'gpu',ranks,'separate',24,interval=24,chemistry_diagnostics='report-steps')
    first,_=launch(args,'gpu',ranks,'first',17,interval=17,chemistry_diagnostics='report-steps')
    args.flux_halo=mode
    candidate,_=launch(args,'gpu',ranks,'packed',24,interval=24,chemistry_diagnostics='report-steps')
    seed=first/'outdat/new/checkpoints/step000000000017'
    resumed,_=launch(args,'gpu',ranks,'resumed',24,restore=seed,interval=24,override=True,
                     chemistry_diagnostics='report-steps')
    final='outdat/new/checkpoints/step000000000024'
    equal={}
    for name in ('state.h5','statistics.h5'):
        equal[name]=compare_fields(baseline/final/name,candidate/final/name)
        compare_fields(candidate/final/name,resumed/final/name)
    assert (baseline/final/'air5_config.bin').read_bytes()==(resumed/final/'air5_config.bin').read_bytes()
    assert (baseline/'flowstate.dat').read_bytes()==(candidate/'flowstate.dat').read_bytes()
    (tmp_path/'flux_halo.json').write_text(json.dumps(dict(ranks=ranks,axis=axis,topology=topology,
        diffusion_limiter=diffusion,flux_halo=mode,backend=backend,equal_datasets=equal,
        sanity=state_sanity(resumed/final/'state.h5')),indent=2,allow_nan=False)+'\n')


def test_flux_halo_memcheck(tmp_path):
    args=options(tmp_path,'sbli','x',topology=(2,2,1))
    args.flux_halo='packed-overlap'
    args.runtime_timeout_seconds=1800
    launch(args,'gpu',4,'memcheck',6,interval=6,memcheck=True,chemistry_diagnostics='report-steps')


def test_combined_multiaxis_learning_exact_restart(tmp_path):
    from test_air5_launch_learning import learning_records
    from test_air5_stage_coordination_holdout import check_statistics_finite

    args=options(tmp_path,'sbli','x',topology=(2,2,1))
    args.low_activity=os.environ.get('ASTR_AIR5_AP_ACTIVITY_THRESHOLD','off')
    if args.low_activity!='off':
        args.step_history='previous'
    args.flux_halo='packed-overlap'
    args.launch_tuning='auto'
    args.chemistry_dispatch='specialized'
    args.rank_output=True
    whole,_=launch(args,'gpu',4,'combined',66,interval=66,chemistry_diagnostics='report-steps')
    first,_=launch(args,'gpu',4,'first',33,interval=33,chemistry_diagnostics='report-steps')
    seed=first/'outdat/new/checkpoints/step000000000033'
    resumed,_=launch(args,'gpu',4,'resumed',66,restore=seed,interval=66,override=True,
                     chemistry_diagnostics='report-steps')
    final='outdat/new/checkpoints/step000000000066'
    equal={name:compare_fields(whole/final/name,resumed/final/name)
           for name in ('state.h5','statistics.h5')}
    check_statistics_finite(whole/final/'statistics.h5')
    check_statistics_finite(resumed/final/'statistics.h5')
    assert (whole/final/'air5_config.bin').read_bytes()==(resumed/final/'air5_config.bin').read_bytes()
    for case in (whole,first,resumed):
        choices,samples,registrations=learning_records(case,4)
        assert len(choices)==len(registrations)==20
        for key,values in samples.items():
            assert min(row[4] for row in values)==6
            assert max(row[4] for row in values)<=33
            assert choices[key]['rounds']==6
            assert choices[key]['reason']=='samples-complete'
            assert len(values)==6*len(registrations[key]['legal'])
    (tmp_path/'combined_restart.json').write_text(json.dumps(dict(
        topology=[2,2,1],continuous_steps=66,restart_step=33,equal_datasets=equal,
        sanity=state_sanity(resumed/final/'state.h5')),indent=2,allow_nan=False)+'\n')
