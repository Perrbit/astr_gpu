"""Reject an unavailable GPU statistics request before committing any run option."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from run_adaptive_tgv_gate import run,compare_state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('gpu','mpiexec','replay-library','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    for name in ('gpu','mpiexec','replay_library','output'):
        setattr(args,name,getattr(args,name).resolve())
    args.output.mkdir(parents=True,exist_ok=False)
    args.size=32;args.scheme='643e';args.analyze_only=False
    baseline='&run_control max_steps=12,reload_steps=1 /\n&timestep initial_dt=0.001 /\n&logging list_steps=0 /\n'
    packets={step:baseline for step in range(13)}
    packets[1]='&run_control max_steps=99,reload_steps=1 /\n&timestep cfl_target=0.3 /\n&statistics enabled=t /\n'
    source,clocks=run(args,'gpu',2,'continuous',runtime_packets=packets,replay_library=args.replay_library)
    log=(source/'run.log').read_text()
    if re.findall(r'ASTR_RUN update rejected at complete_step=(\d+)',log)!=['1']:
        raise AssertionError('unavailable statistics must reject the candidate once')
    if 'ASTR_RUN accepted' in log:
        raise AssertionError('a rejected candidate changed the observed snapshot')
    expected=0.001
    for row in clocks:
        np.testing.assert_allclose(row[3],expected,rtol=0,atol=64*np.finfo(float).eps)
        expected*=1.1
    resumed,resumed_clocks=run(args,'gpu',2,'resumed',restore=source/'captured/checkpoints/step000000000005',
        runtime_packets=packets,replay_library=args.replay_library)
    if resumed_clocks!=clocks[5:]:
        raise AssertionError('rejected request altered exact continuation clocks')
    final=Path('outdat/new/checkpoints/step000000000012')
    compare_state(source/final/'state.h5',resumed/final/'state.h5',True)
    for name in ('control.bin','archives.bin','insitu_control.bin'):
        if (source/final/name).read_bytes()!=(resumed/final/name).read_bytes():
            raise AssertionError(f'exact rejection history differs: {name}')
    report=dict(status='passed',rejected_steps=[1],complete_steps=12,unchanged_growth_sequence=True,
        exact_state_and_controls=True,clocks=clocks)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
