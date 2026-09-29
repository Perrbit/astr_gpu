#!/usr/bin/env python3
"""Capture existing stage diagnostics without changing release solver code."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from prepare_tgv_case import set_controller_steps, set_restart


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(ASTR_FORCE_MPI_TOPOLOGY='2,1,1', ASTR_GPU_SYNC_MODE='explicit',
               ASTR_GPU_HALO_TRANSPORT='pinned', ASTR_GPU_PRECISION_MODE='fp64',
               ASTR_GPU_FILTER_WORKSPACE='scalar', ASTR_CHANNEL_FORCE_MODE='fixed',
               ASTR_CHANNEL_FORCE_FIXED='1.d-4', ASTR_VALIDATION_RHS_STEP='49',
               ASTR_VALIDATION_RHS_STEP_SECONDARY='50')
    for mode in ('continuous', 'restart'):
        case = out/mode
        shutil.copytree(args.reference/'datin', case/'datin')
        for directory in ('outdat', 'bakup', 'monitor', 'testout', 'islice', 'jslice', 'kslice'):
            (case/directory).mkdir()
        if mode == 'restart':
            shutil.copytree(args.checkpoint, case/'outdat', dirs_exist_ok=True)
        set_restart(case/'datin/input.chl', 't' if mode == 'restart' else 'f')
        set_controller_steps(case/'datin/controller', 51, 1, 1)
        runtime = dict(env, ASTR_VALIDATION_RHS_PREFIX=str(case/'trace'))
        (case/'runtime.json').write_text(json.dumps({k:v for k,v in runtime.items()
            if k.startswith('ASTR_')}, indent=2)+'\n')
        with (case/'run.log').open('w') as log:
            subprocess.run(['mpirun', '-np', '2', str(args.exe.resolve()),
                'run', 'datin/input.chl'], cwd=case, env=runtime, stdout=log,
                stderr=subprocess.STDOUT, check=True, timeout=180)
        if 'The job is done!' not in (case/'run.log').read_text():
            raise ValueError(f'failed run: {case}')
    print(out)


if __name__ == '__main__':
    main()
