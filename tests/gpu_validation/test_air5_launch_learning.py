"""AP3 normal-launch sampling, startup lifecycle and exact continuation."""
import json
import math
import re

import h5py
import pytest

from run_output_air5_restart_validation import launch, state_sanity
from run_output_restart_validation import compare_fields
from test_output_air5_conservation import arguments


def learning_records(case,ranks):
    choices, samples, registrations = {}, {}, {}
    sources=list((case/'mpi_output').rglob('stdout'))
    assert len(sources)==ranks
    for line in '\n'.join(path.read_text() for path in sorted(sources)).splitlines():
        if line.startswith('AIR5_TUNING_REGISTER '):
            match = re.fullmatch(r'AIR5_TUNING_REGISTER rank=(\d+) context=(\S+) original=(\d+) legal=([TF ]+)', line)
            assert match, line
            rank,context,original,mask=match.groups()
            key=(int(rank),context)
            assert key not in registrations
            registrations[key]=dict(original=int(original),legal=[i for i,v in enumerate(mask.split(),1) if v=='T'])
        elif line.startswith('AIR5_TUNING_CHOICE '):
            match = re.fullmatch(r'AIR5_TUNING_CHOICE rank=(\d+) context=(\S+) candidate=(\d+) '
                r'block=\s*(\d+)\s+(\d+)\s+(\d+)\s+complete_rounds=(\d+) reason=(\S+)', line)
            assert match, line
            rank, context, candidate, x, y, z, rounds, reason = match.groups()
            key = (int(rank), context)
            assert key not in choices
            choices[key] = dict(candidate=int(candidate), block=(int(x),int(y),int(z)),
                                rounds=int(rounds), reason=reason)
        elif line.startswith('AIR5_TUNING_SAMPLE '):
            match = re.fullmatch(r'AIR5_TUNING_SAMPLE rank=(\d+) context=(\S+) candidate=(\d+) '
                r'round=(\d+) startup_step=(\d+) used=([TF]) ms=\s*(\S+)', line)
            assert match, line
            rank, context, candidate, number, step, used, duration = match.groups()
            key = (int(rank), context)
            samples.setdefault(key, []).append((int(candidate), int(number), used, float(duration), int(step)))
    return choices, samples, registrations


def check_complete_learning(case, ranks, axis, started_at=0):
    choices, samples, registrations = learning_records(case,ranks)
    assert len(choices)==5*ranks
    assert choices.keys()==registrations.keys()
    for key, choice in choices.items():
        rank, context = key
        inactive = axis=='y' and rank<ranks-1 and context.endswith('_top')
        if inactive:
            assert choice['rounds']==0 and choice['reason'] in ('no-complete-round','run-ended-while-learning')
            assert choice['block']==(128,1,1)
            assert key not in samples
            continue
        legal=registrations[key]['legal']
        count=len(legal)
        assert choice['rounds']==6 and choice['reason']=='samples-complete'
        values = samples[key]
        assert len(values)==count*6 and all(v[2]=='T' and math.isfinite(v[3]) and v[3]>=0 for v in values)
        assert min(v[4] for v in values)==6 and max(v[4] for v in values)<=64
        assert {(v[0],v[1]) for v in values}=={(c,r) for c in legal for r in range(1,7)}
        # Event medians, not minimum samples or a shared rank winner.
        medians = {}
        for candidate in legal:
            times = sorted(v[3] for v in values if v[0]==candidate)
            medians[candidate] = (times[2]+times[3])*0.5
        assert medians[choice['candidate']]==min(medians.values())
        original=registrations[key]['original']
        if medians[original]==min(medians.values()):
            assert choice['candidate']==original
    return dict(started_at=started_at,
        choices={f'{rank}/{context}':row for (rank,context),row in choices.items()},
        samples={f'{rank}/{context}':row for (rank,context),row in samples.items()})


@pytest.mark.parametrize('ranks,axis,case', [(1,'x','sbli'),(2,'x','sbli'),(2,'y','hbl'),(2,'z','sbli')])
def test_auto_keeps_fixed_fields_and_relearns_after_restart(tmp_path,ranks,axis,case):
    args = arguments(tmp_path,case=case,axis=axis,initial=True)
    if not args.executable.is_file():
        pytest.skip('ASTR executable is unavailable')
    args.air5_filter_mode='conservative-flux'
    args.source_evaluation='source-only'
    args.primitive_reuse='chemistry'
    args.chemistry_reductions='packed'
    args.report_interval=8
    args.rank_output=True
    args.runtime_timeout_seconds=600
    fixed,_=launch(args,'gpu',ranks,'fixed',66,interval=66,chemistry_diagnostics='report-steps')
    args.launch_tuning='auto'
    automatic,_=launch(args,'gpu',ranks,'auto',66,interval=66,chemistry_diagnostics='report-steps')
    first,_=launch(args,'gpu',ranks,'first',33,interval=33,chemistry_diagnostics='report-steps')
    seed=first/'outdat/new/checkpoints/step000000000033'
    resumed,_=launch(args,'gpu',ranks,'resumed',66,restore=seed,interval=66,override=True,
                     chemistry_diagnostics='report-steps')
    final='outdat/new/checkpoints/step000000000066'
    equal={}
    for name in ('state.h5','statistics.h5'):
        equal[name]=compare_fields(fixed/final/name,automatic/final/name)
        compare_fields(automatic/final/name,resumed/final/name)
    assert (fixed/final/'air5_config.bin').read_bytes()==(automatic/final/'air5_config.bin').read_bytes()
    assert (automatic/final/'air5_config.bin').read_bytes()==(resumed/final/'air5_config.bin').read_bytes()
    with h5py.File(resumed/final/'state.h5') as state:
        assert int(state['identity'][8])==66
    records=check_complete_learning(automatic,ranks,axis)
    records['restart']=check_complete_learning(resumed,ranks,axis,33)
    records['equal_datasets']=equal
    records['sanity']=state_sanity(resumed/final/'state.h5')
    (tmp_path/'launch_learning.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')


def test_auto_kernel_candidates_memcheck(tmp_path):
    args=arguments(tmp_path,case='hbl',axis='y',initial=True)
    args.air5_filter_mode='conservative-flux'
    args.source_evaluation='source-only'
    args.launch_tuning='auto'
    args.rank_output=True
    args.report_interval=10
    args.runtime_timeout_seconds=1800
    case,_=launch(args,'gpu',2,'auto_memcheck',10,interval=10,memcheck=True,
                  chemistry_diagnostics='report-steps')
    _,samples,registrations=learning_records(case,2)
    for key,values in samples.items():
        assert {v[0] for v in values}==set(registrations[key]['legal'])
        assert min(v[4] for v in values)==6 and max(v[4] for v in values)==10
