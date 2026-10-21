"""Ma12 MP7-LD/NSCBC continuation with the original wall RNG and GPU/CPU solver."""

import argparse
import json
from math import isfinite
import os
from pathlib import Path
import re
import subprocess

from prepare_m12_local_case import prepare, replace_value
from run_compact_flatplate import configure
from run_native_precision_restart_gate import check_state
from run_output_restart_validation import compare_fields


ROOT = Path(__file__).resolve().parents[2]
GRID = (96, 48, 32)
FINAL = Path('outdat/new/checkpoints/step000000000012')


def launch(args, backend, ranks, name, steps, restore=None, exact=True, restore_probe=False):
    case = args.output / f'{backend}_np{ranks}_{name}'
    prepare(args.source, case, backend, GRID)
    configure(case, 'nscbc', args.diffusion, True, 't' if args.filter else 'f', True)
    controller = case / 'datin/controller'
    lines = controller.read_text().splitlines()
    replace_value(lines, 'maxstep,feqchkpt', f'{steps - 1},1,9999,9999,1,9999')
    controller.write_text('\n'.join(lines) + '\n')
    (case / 'outdat/new').mkdir()
    (case / 'datin/input.output').write_text(f"""&output
 directory='outdat/new',restore_directory='{restore or ''}',exact_restart={'.true.' if exact else '.false.'},
 host_budget_bytes=67108864,device_budget_bytes=67108864,buffer_bytes=1048576,
 checkpoint_statistics=t
/
&checkpoint
 enabled=t,mode='steps',interval_steps=5,keep=2
/
&volume
 enabled=f
/
&slices
 enabled=f
/
""")
    if restore:
        # All four physical resources must come from the frozen source bundle.
        for filename in ('grid.2d', 'flowini2d.h5', 'inlet.prof', 'wallbs.dat'):
            (case / 'datin' / filename).unlink()
    env = {k: v for k, v in os.environ.items() if not k.startswith(('ASTR_', 'OMPI_MCA_', 'UCX_'))}
    env.update(ASTR_FORCE_MPI_TOPOLOGY=f'{ranks},1,1', ASTR_WALL_BLOWING_MODE='legacy_random',
        ASTR_NSCBC_FARFIELD_MODE='incoming_only', ASTR_NSCBC_FARFIELD_RHO='1', ASTR_NSCBC_FARFIELD_U='1',
        ASTR_NSCBC_FARFIELD_V='0', ASTR_NSCBC_FARFIELD_W='0', ASTR_NSCBC_FARFIELD_T='1',
        ASTR_GPU_FILTER_WORKSPACE='scalar', ASTR_GPU_SYNC_MODE='explicit', ASTR_GPU_HALO_TRANSPORT='pinned',
        ASTR_GPU_COMPACT_SYNC='explicit', ASTR_GPU_COMPACT_TRANSPORT='pinned',
        OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_opal_cuda_support='0')
    if restore_probe:
        env['ASTR_CHECKPOINT_TEST_RESTORE_PROBE'] = '1'
    with (case / 'run.log').open('w') as log:
        completed = subprocess.run([str(args.mpiexec.resolve()), '--bind-to', 'none', '-np', str(ranks),
            str(getattr(args, backend).resolve()), 'run', 'datin/input.dat'], cwd=case, env=env,
            stdout=log, stderr=subprocess.STDOUT, timeout=180)
    if completed.returncode:
        raise RuntimeError(f'Ma12 restart solver failed: {case / "run.log"}')
    text = (case / 'run.log').read_text()
    first = 5 if restore else 0
    indices = [int(n) for n in re.findall(r'ASTR_CFL complete_step=(\d+)', text)]
    cfl = [float(row.split()[-1]) for row in re.findall(r'ASTR_CFL directional/local_sum/upper_bound=([^\n]+)', text)]
    if (indices != list(range(first, steps)) or len(cfl) != steps - first or
            not all(isfinite(v) and v >= 0 for v in cfl) or max(cfl) > .5):
        raise AssertionError('Ma12 complete-step/CFL gate differs')
    disk = sum(p.stat().st_size for p in case.rglob('*') if p.is_file())
    if disk > 128 * 1024**2:
        raise RuntimeError(f'Ma12 exact-checkpoint directory exceeds 128 MiB: {disk}')
    return case, max(cfl)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision', choices=['fp32', 'fp64'], required=True)
    parser.add_argument('--cpu', type=Path, required=True)
    parser.add_argument('--gpu', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('/home/dell/workspace/M12_C13'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backends', nargs='+', choices=['cpu', 'gpu'], default=['cpu', 'gpu'])
    parser.add_argument('--ranks', nargs='+', type=int, choices=[1, 2], default=[1, 2])
    parser.add_argument('--diffusion', choices=['643e', '643c'], default='643e')
    parser.add_argument('--filter', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(exist_ok=False)
    report = dict(precision=args.precision, grid_intervals=GRID, scheme=f'743e/{args.diffusion}', filter=args.filter,
                  boundary=[11, 22, 41, 52, 1, 1], blowing='legacy_random', dt_input=.005, cases={})
    for backend in args.backends:
        for ranks in args.ranks:
            a, cfl_a = launch(args, backend, ranks, 'continuous', 12)
            b, cfl_b = launch(args, backend, ranks, 'seed', 5)
            for resource in (b / 'outdat/new/resources').iterdir():
                if resource.suffix == '.h5' or resource.name == 'grid.2d':
                    resource.chmod(0o444)
            before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in b.rglob('*') if p.is_file()}
            c, cfl_c = launch(args, backend, ranks, 'resumed', 12,
                restore=b / 'outdat/new/checkpoints/step000000000005')
            result = dict(state=check_state(c / FINAL / 'state.h5', args.precision),
                          exact_datasets=compare_fields(a / FINAL / 'state.h5', c / FINAL / 'state.h5'),
                          max_cfl_upper=max(cfl_a, cfl_b, cfl_c))
            for name in ('control.bin', 'archives.bin', 'insitu_control.bin'):
                if (a / FINAL / name).read_bytes() != (c / FINAL / name).read_bytes():
                    raise AssertionError(f'Ma12 {backend} NP={ranks}: exact {name} differs')
            after = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in b.rglob('*') if p.is_file()}
            if before != after:
                raise AssertionError('Ma12 source run was modified during frozen-resource restore')
            result['source_unchanged'] = True
            report['cases'][f'{backend}_np{ranks}'] = result
            (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(f'Ma12 {args.precision} {backend} NP={ranks}: exact state, RNG/control, frozen resources', flush=True)
    report['status'] = 'passed'
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
