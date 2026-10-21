"""Cross-kind adaptive event/window history and native follow-on continuation."""

import argparse
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

from run_native_precision_restart_gate import check_state
from run_native_precision_representative_cross_restart_gate import check_probe
from run_output_restart_validation import run_case


ROOT = Path(__file__).resolve().parents[2]
CONFIG = '''&adaptive_output
 enabled=t,monitor_steps=1,event_ids='decay',s_ref=1,t_ref=1,
 r_on=0.74,r_off=0.723,hold_time=0.005,
 window_ids='burst','clockwindow',window_modes='steps','time',
 window_steps(:,1)=3,7,window_times(:,2)=0.003,0.007
/
'''
GROUPS = '''&volume
 enabled=t,interval_steps=5,adaptive=t,dense_interval_steps=1,
 window_ids='burst',event_ids='decay'
/
&slices
 enabled=t,mode='time',interval_time=0.006,adaptive=t,dense_interval_time=0.002,
 window_ids='clockwindow',event_ids='decay',k_indices=4
/
'''


def snapshot(root):
    return {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}


def check_log(case, first, last):
    text = (case / 'run.log').read_text()
    if 'ASTR_AP_RESET' in text:
        raise AssertionError('unchanged adaptive configuration was reset')
    steps = [int(value) for value in re.findall(r'ASTR_AP_MONITOR step=(\d+)', text)]
    if steps != list(range(first + 1, last + 1)):
        raise AssertionError(f'adaptive monitor identities differ: {steps}')
    return dict(monitor_steps=steps, products=re.findall(r'ASTR_AP_PRODUCT id=(\S+) step=(\d+)', text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for backend in ('cpu', 'gpu'):
        for precision in ('fp32', 'fp64'):
            parser.add_argument(f'--{backend}-{precision}', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backends', nargs='+', choices=['cpu', 'gpu'], default=['cpu', 'gpu'])
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(exist_ok=False)
    os.environ.update(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_osc='pt2pt',
                      OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_opal_cuda_support='0')
    report = dict(grid=[16]*3, ranks=2, cases={})
    for backend in args.backends:
        for old, new in (('fp32', 'fp64'), ('fp64', 'fp32')):
            tag = f'{backend}_{old}_to_{new}'
            local = SimpleNamespace(output=args.output, executable=getattr(args, f'{backend}_{old}').resolve(),
                mpiexec=args.mpiexec.resolve(), case='tgv', mode='steps', restart_step=5,
                initial_dimension=0, legacy_statistics=False, statistics=True, initial_restart=False,
                filter_workspace='scalar', force='feedback', axis='x', no_samples=True, wall_samples=False)
            options = dict(gpu_precision_mode='', archive_groups=GROUPS, adaptive_config=CONFIG,
                           buffer_bytes=4096, tgv_reynolds=1)
            a, _ = run_case(local, ROOT, backend, 2, old + '_seed_for_' + new, 5, **options)
            source = a / 'outdat/new/checkpoints/step000000000005'
            before = snapshot(a)
            local.executable = getattr(args, f'{backend}_{new}').resolve()
            b, _ = run_case(local, ROOT, backend, 2, new + '_from_' + old, 12,
                restore=source, exact_restart=False, restore_probe=True, **options)
            result = dict(restored_fields=check_probe(source, b, new, (True, True, True)),
                state=check_state(b / 'outdat/new/checkpoints/step000000000012/state.h5', new),
                log=check_log(b, 5, 12))
            before_b = snapshot(b)
            c, _ = run_case(local, ROOT, backend, 2, new + '_follow_on_from_' + old, 14,
                restore=b / 'outdat/new/checkpoints/step000000000012', exact_restart=False, **options)
            result['follow_on_state'] = check_state(c / 'outdat/new/checkpoints/step000000000014/state.h5', new)
            result['follow_on_log'] = check_log(c, 12, 14)
            if snapshot(a) != before or snapshot(b) != before_b:
                raise AssertionError('source run changed during adaptive continuation')
            result['source_unchanged'] = True
            report['cases'][tag] = result
            (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(tag + ': adaptive history and target-kind follow-on passed', flush=True)
    report['status'] = 'passed'
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
