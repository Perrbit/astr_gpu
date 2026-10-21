"""Same-state M12 proof for quiet timing; not a performance measurement."""

import argparse
import json
from pathlib import Path

from run_native_precision_m12_gate import run_case, LABELS
from compare_q_validation_snapshots import compare_snapshot_sets
from run_compact_execution import crashfix_summary, window_times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-fp32', type=Path, required=True)
    parser.add_argument('--gpu-fp64', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('/home/dell/workspace/M12_C13'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(exist_ok=False)
    report = dict(topology=[2, 1, 1], complete_steps=10, blowing='original legacy_random', cases={})
    for precision in ('fp32', 'fp64'):
        exe = getattr(args, 'gpu_'+precision).resolve()
        for scheme in ('643e', '643c'):
            tag = precision+'_'+scheme
            baseline, quiet = root/tag/'baseline', root/tag/'quiet'
            a = run_case(baseline, exe, args.source, True, (2, 1, 1), precision, True, 'scalar', diffusion=scheme)
            b = run_case(quiet, exe, args.source, True, (2, 1, 1), precision, True, 'scalar', diffusion=scheme,
                         buffer_output=True, statistics_interval=0)
            comparison = compare_snapshot_sets(baseline/'diagnostics/state', quiet/'diagnostics/state', LABELS,
                atol=0., rtol=0., active_only=True, cpu_precision=precision, gpu_precision=precision)
            if not comparison.passed or comparison.max_abs != 0:
                raise AssertionError('Quiet output controls alter M12 advancement')
            first, second = (path/'run.log' for path in (baseline, quiet))
            repair = crashfix_summary(first.read_text())
            if repair != crashfix_summary(second.read_text()):
                raise AssertionError('Buffered crashfix summary changes repair diagnostics')
            window_times(second.read_text(), 2, 10, 0)
            if list(quiet.rglob('flowstate.dat')):
                raise AssertionError('Quiet M12 run emitted scalar statistics')
            report['cases'][tag] = dict(field_max_abs=0., repair=repair, baseline=a, quiet=b)
            (root/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            print(tag+': state and repair diagnostics unchanged', flush=True)
    report['status'] = 'passed'
    (root/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
