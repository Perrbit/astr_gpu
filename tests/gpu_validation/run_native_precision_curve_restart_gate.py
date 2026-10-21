"""Native static CURVE exact continuation; ASTR alone advances the flow."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from run_native_precision_restart_gate import check_state
from run_output_restart_validation import compare_fields, run_case


ROOT = Path(__file__).resolve().parents[2]
FINAL = Path('outdat/new/checkpoints/step000000000012')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision', choices=['fp32', 'fp64'], required=True)
    parser.add_argument('--cpu', type=Path, required=True)
    parser.add_argument('--gpu', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backends', nargs='+', choices=['cpu', 'gpu'], default=['cpu', 'gpu'])
    parser.add_argument('--mappings', nargs='+', choices=['periodic', 'y-wavy'], default=['periodic', 'y-wavy'])
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(exist_ok=False)
    os.environ.update(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp',
                      OMPI_MCA_osc='pt2pt', OMPI_MCA_coll='^cuda,hcoll,ucc',
                      OMPI_MCA_opal_cuda_support='0')
    dt = float(np.float32(2e-5)) if args.precision == 'fp32' else 2e-5
    report = dict(precision=args.precision, complete_steps=12, restart_step=5,
                  intended_dt=2e-5, native_dt=dt, maximum_cfl=.5, cases={})
    for mapping in args.mappings:
        for backend in args.backends:
            for ranks in (1, 2):
                label = f'{backend}_{mapping}_np{ranks}'
                local = SimpleNamespace(output=args.output, executable=getattr(args, backend).resolve(),
                    mpiexec=args.mpiexec.resolve(), case='tgv', mode='steps', restart_step=5,
                    initial_dimension=0, legacy_statistics=False, statistics=False, initial_restart=False,
                    filter_workspace='scalar', force='feedback', axis='y' if mapping == 'y-wavy' else 'x',
                    no_samples=True, wall_samples=False, scale_timestep=dt, maximum_cfl=.5)
                local.native_precision = args.precision
                options = dict(gpu_precision_mode='', grid='32,32,32', tgv_mapping=mapping)
                a, _ = run_case(local, ROOT, backend, ranks, mapping + '_continuous', 12, **options)
                b, _ = run_case(local, ROOT, backend, ranks, mapping + '_seed', 5, **options)
                resources = b / 'outdat/new/resources'
                for path in resources.glob('*.h5'):
                    path.chmod(0o444)
                before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in resources.iterdir() if p.is_file()}
                c, _ = run_case(local, ROOT, backend, ranks, mapping + '_resumed', 12,
                    restore=b / 'outdat/new/checkpoints/step000000000005', **options)
                result = dict(state=check_state(c / FINAL / 'state.h5', args.precision),
                              exact_datasets=compare_fields(a / FINAL / 'state.h5', c / FINAL / 'state.h5'))
                for name in ('control.bin', 'archives.bin', 'insitu_control.bin'):
                    if (a / FINAL / name).read_bytes() != (c / FINAL / name).read_bytes():
                        raise AssertionError(f'{label}: {name} differs')
                if any((p.stat().st_size, p.stat().st_mtime_ns) != saved for p, saved in before.items()):
                    raise AssertionError('frozen CURVE source resources were modified')
                result['source_resources_unchanged'] = True
                result['max_cfl_upper'] = max(json.loads((p / 'cfl_gate.json').read_text())['maximum'] for p in (a, b, c))
                report['cases'][label] = result
                (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
                print(label + ': exact state/control and immutable frozen grid', flush=True)
    report['status'] = 'passed'
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
