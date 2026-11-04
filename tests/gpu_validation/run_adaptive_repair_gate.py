"""Actual repair hot-switch lifecycle and saved effective-mode restoration."""
import argparse
import json
from pathlib import Path
import re

import h5py
import numpy as np

from run_adaptive_tgv_gate import run,compare_state


def packet(step):
    return ('&run_control max_steps=12,reload_steps=1 /\n&timestep initial_dt=0.001 /\n'
        f'&logging list_steps={0 if step<6 else 2} /\n'
        f'&repair crashfix={"t" if 2<=step<6 else "f"} /\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cpu','gpu','mpiexec','replay-library','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--unsupported',action='store_true',help='GPU compact-convection repair must reject before commit')
    parser.add_argument('--backends',nargs='+',choices=('cpu','gpu'),default=['cpu','gpu'])
    args=parser.parse_args()
    for name in ('cpu','gpu','mpiexec','replay_library','output'):
        setattr(args,name,getattr(args,name).resolve())
    args.output.mkdir(parents=True,exist_ok=False)
    args.size=32;args.scheme='643c' if args.unsupported else '643e';args.analyze_only=False
    packets={step:packet(step) for step in range(13)}
    if args.unsupported:
        packets={step:packet(0) for step in range(13)}
        packets[1]='&run_control max_steps=99,reload_steps=1 /\n&timestep cfl_target=0.3 /\n&repair crashfix=t /\n'
    final=Path('outdat/new/checkpoints/step000000000012')
    report={}
    for backend in (('gpu',) if args.unsupported else args.backends):
        source,clocks=run(args,backend,2,backend,runtime_packets=packets,replay_library=args.replay_library)
        log=(source/'run.log').read_text()
        rejected=re.findall(r'ASTR_RUN update rejected at complete_step=(\d+)',log)
        if args.unsupported:
            if rejected!=['1'] or 'ASTR_RUN accepted' in log:
                raise AssertionError('unsupported repair did not reject the whole candidate')
        else:
            if rejected or re.findall(r'ASTR_RUN accepted complete_step=(\d+)',log)!=['2','6']:
                raise AssertionError('repair switch sequence differs from complete-step requests')
            with h5py.File(source/'captured/checkpoints/step000000000005/state.h5') as state:
                if 'crashfix_history' not in state or state['crashfix_layout'][1]!=1:
                    raise AssertionError('enabled repair did not reach the actual checkpoint provider')
        resumed_packets=dict(packets)
        if not args.unsupported:
            # Unedited startup defaults must not override the effective repair mode saved at step 5.
            resumed_packets[5]=packet(0)
        resumed,resumed_clocks=run(args,backend,2,backend+'_resumed',
            restore=source/'captured/checkpoints/step000000000005',runtime_packets=resumed_packets,
            replay_library=args.replay_library)
        if resumed_clocks!=clocks[5:]:
            raise AssertionError('repair restore changed complete-step clocks')
        compare_state(source/final/'state.h5',resumed/final/'state.h5',True)
        for name in ('control.bin','archives.bin','insitu_control.bin'):
            if (source/final/name).read_bytes()!=(resumed/final/name).read_bytes():
                raise AssertionError(f'repair exact continuation differs in {name}')
        if backend=='gpu' and not args.unsupported:
            with h5py.File(source/final/'state.h5') as state:
                if 'crashfix_history' not in state or state['crashfix_layout'][1]!=0:
                    raise AssertionError('disabling repair discarded the existing device history')
        report[backend]=dict(rejected_steps=rejected,exact_state_and_controls=True,clocks=clocks)
    report['status']='passed'
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
