"""One real-solver hot-update sequence, with exact continuation of the same sequence."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from run_adaptive_tgv_gate import run,compare_state


def packet(timestep):
    return '&run_control max_steps=12, reload_steps=1 /\n'+timestep+'\n&logging list_steps=0 /\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cpu','gpu','mpiexec','replay-library','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    for name in ('cpu','gpu','mpiexec','replay_library','output'):
        setattr(args,name,getattr(args,name).resolve(strict=name!='output'))
    args.size=32;args.scheme='643e';args.analyze_only=False
    args.output.mkdir(parents=True,exist_ok=False)
    packets={0:packet('&timestep mode="fixed", fixed_dt=0.001 /'),
             1:packet('&timestep mode="auto", max_dt=0.0015 /'),
             2:packet('&timestep mode="bad-mode", fixed_dt=1.0 /'),
             3:packet('&timestep mode="fixed", fixed_dt=0.0008, max_dt=0.00001 /'),
             4:packet('&timestep mode="auto", max_dt=0.0007 /'),
             5:packet('&timestep mode="auto" /')}
    # Frame 6 is deliberately absent: a failed open must not reset the controller.
    packets.update({step:packet('&timestep mode="auto", cfl_target=0.3 /') for step in range(7,13)})
    expected_next=[0.0011,0.00121,0.0008,0.0007,0.00077,0.000847]
    for _ in range(6):
        expected_next.append(expected_next[-1]*1.1)
    final=Path('outdat/new/checkpoints/step000000000012')
    report={};references={}
    for backend in ('cpu','gpu'):
        source,clocks=run(args,backend,2,backend,runtime_packets=packets,replay_library=args.replay_library)
        np.testing.assert_allclose([record[4] for record in clocks],expected_next,rtol=0,atol=64*np.finfo(float).eps)
        log=(source/'run.log').read_text()
        if re.findall(r'ASTR_RUN update rejected at complete_step=(\d+)',log)!=['2','6']:
            raise AssertionError('invalid and missing snapshots must each reject exactly once')
        if re.findall(r'ASTR_CONTROLLER_TEST_READ completed_step=(\d+)',log)!=list(map(str,range(13))):
            raise AssertionError('one root-only read per selected complete step is required')
        resumed,resumed_clocks=run(args,backend,2,backend+'_resumed',
            restore=source/'captured/checkpoints/step000000000005',
            runtime_packets=packets,replay_library=args.replay_library)
        if resumed_clocks!=clocks[5:]:
            raise AssertionError('exact restore overwrote saved dt_next or changed subsequent hot-update clocks')
        compare_state(source/final/'state.h5',resumed/final/'state.h5',True)
        for name in ('control.bin','archives.bin','insitu_control.bin'):
            if (source/final/name).read_bytes()!=(resumed/final/name).read_bytes():
                raise AssertionError(f'hot-update exact control history differs: {name}')
        report[backend]=dict(rejected_steps=[2,6],root_reads=13,clocks=clocks,exact_restart=True)
        references[backend]=source/final/'state.h5'
    comparison={}
    compare_state(references['cpu'],references['gpu'],False,comparison)
    if report['cpu']['clocks']!=report['gpu']['clocks']:
        raise AssertionError('CPU/GPU control identities or physical times differ')
    report['cpu_gpu']=comparison
    report['status']='passed'
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
