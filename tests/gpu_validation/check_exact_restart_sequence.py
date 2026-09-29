#!/usr/bin/env python3
"""Compare a four-step channel run with a two-step sequence checkpoint restart."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
from prepare_tgv_case import set_controller_sequence, set_controller_steps, set_restart


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--gpu', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(ASTR_FORCE_MPI_TOPOLOGY='2,1,1', ASTR_GPU_SYNC_MODE='explicit',
               ASTR_GPU_HALO_TRANSPORT='pinned', ASTR_GPU_PRECISION_MODE='fp64',
               ASTR_CHANNEL_FORCE_MODE='fixed', ASTR_CHANNEL_FORCE_FIXED='1.d-4')
    final = {}
    for name in ('continuous', 'restart'):
        case = out/name
        shutil.copytree(args.reference/'datin', case/'datin')
        for directory in ('outdat', 'bakup', 'islice', 'jslice', 'kslice', 'monitor', 'testout'):
            (case/directory).mkdir()
        set_restart(case/'datin/input.chl', 'f')
        for step in ([4] if name == 'continuous' else [2, 4]):
            set_controller_steps(case/'datin/controller', step, step, 1)
            set_controller_sequence(case/'datin/controller', 't', step)
            with (case/f'run{step}.log').open('w') as log:
                subprocess.run(['mpirun', '-np', '2', str(args.gpu.resolve()),
                                'run', 'datin/input.chl'], cwd=case, env=env,
                               stdout=log, stderr=subprocess.STDOUT, timeout=60, check=True)
            fields = sorted((case/'outdat').glob('flowfield*.h5'))
            selected = []
            for path in fields:
                with h5py.File(path) as handle:
                    if handle['nstep'][()].item() == step:
                        selected.append(path)
            assert len(selected) == 1, selected
            final[name] = selected[0]
            for rank in range(2):
                assert (case/f'outdat/restart_q.step{step:010d}.rank{rank:08d}.bin').is_file()
            if step == 2:
                auxiliary = selected[0].with_name(selected[0].stem.replace('flowfield', 'auxiliary')+'.txt')
                shutil.copy2(auxiliary, case/'outdat/auxiliary.txt')
                set_restart(case/'datin/input.chl', 't')
    subprocess.run([sys.executable, str(Path(__file__).with_name('compare_flowfield_h5.py')),
                    '--cpu', str(final['continuous']), '--gpu', str(final['restart']),
                    '--atol', '0', '--rtol', '0', '--report', str(out/'field.txt')], check=True)


if __name__ == '__main__':
    main()
