"""Separate the observed dt-sequence sensitivity from span-grid sensitivity."""
import argparse
import json
from pathlib import Path
import re

import h5py
import numpy as np

from run_h1_hcw_span_grid import BASELINE,analyze_case,clock
from test_high_order_boundary_runtime import EXE,H1_VARIANT,launch_hcw

ROOT=Path(__file__).resolve().parents[2]
REFERENCE=ROOT/'tests/gpu_validation/out/h1_span_grid_20261105/window_after_archive_fix/span2mm_n16'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path,default=REFERENCE)
    parser.add_argument('--baseline',type=Path,default=BASELINE)
    parser.add_argument('--replay-library',type=Path,required=True)
    parser.add_argument('--analyze-only',action='store_true')
    args=parser.parse_args()
    assert EXE and H1_VARIANT=='inlet-compatible','Select the approved root executable and H1 candidate'
    args.output.mkdir(parents=True,exist_ok=True)
    reference=clock(args.reference)
    coarse_auto=analyze_case(args.baseline)
    fine_auto=analyze_case(args.reference)
    case=args.output/'span2mm_n5_common_fine_dt'
    if not args.analyze_only:
        launch_hcw(case,'gpu',steps=len(reference),z_intervals=5,span_m=.002,
            replay_clocks=reference,replay_library=args.replay_library,
            checkpoint_interval=1000000,volume_time_interval=coarse_auto['final_time']/10)
    actual=clock(case)
    np.testing.assert_array_equal(actual,reference)
    reads=[int(value) for value in re.findall(r'ASTR_CONTROLLER_TEST_READ completed_step=(\d+)',
                                             (case/'run.log').read_text())]
    assert reads==list(range(len(reference)+1))
    coarse_replay=analyze_case(case)
    for result in (coarse_auto,fine_auto,coarse_replay):
        text=(Path(result['case'])/'run.log').read_text()
        upper=[float(row.split()[-1]) for row in re.findall(
            r'ASTR_CFL directional/local_sum/upper_bound=([^\n]+)',text)]
        result['observed_cfl_upper_range']=[min(upper),max(upper)]
    names=coarse_replay['final']['spanwise_deviation_rms']
    ratios={name:dict(
        coarse_auto_over_coarse_fine_dt=coarse_auto['final']['spanwise_deviation_rms'][name]/
                                       coarse_replay['final']['spanwise_deviation_rms'][name],
        coarse_over_fine_common_dt=coarse_replay['final']['spanwise_deviation_rms'][name]/
                                  fine_auto['final']['spanwise_deviation_rms'][name]) for name in names}
    end=f'outdat/new/checkpoints/step{len(reference):012d}/state.h5'
    mean_errors={}
    with h5py.File(case/end) as a,h5py.File(args.reference/end) as b:
        for name,index in zip(('density','u','v','w','pressure','temperature'),range(6,12)):
            left=a[f'q{index:04d}'][:-1].mean(axis=0)
            right=b[f'q{index:04d}'][:-1].mean(axis=0)
            mean_errors[name]=float((abs(left-right)/(1+abs(right))).max())
    report=dict(scope='Unforced roundoff-seeded sensitivity; not modal physical convergence or backend equivalence',
        timestep_source='Observed automatic fine-grid sequence replayed using test-only fixed-dt input updates',
        common_clocks_exact=True,coarse_auto=coarse_auto,fine_auto=fine_auto,coarse_replay=coarse_replay,
        coarse_auto_endpoint_offset=float(coarse_replay['final_time']-coarse_auto['final_time']),
        common_dt_spanwise_mean_scaled_differences=mean_errors,ratios=ratios)
    (args.output/'span_timestep.json').write_text(json.dumps(report,indent=2)+'\n')
    print('common-clock coarse grid:',coarse_replay['steps'],coarse_replay['final_time'])
    print(json.dumps(ratios,indent=2))


if __name__=='__main__':main()
