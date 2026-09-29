#!/usr/bin/env python3
"""Exercise corrupt/mismatched exact checkpoints using a channel step-50 fixture."""
import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess

from prepare_tgv_case import set_controller_steps, set_restart


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--gpu', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(ASTR_FORCE_MPI_TOPOLOGY='2,1,1', ASTR_GPU_SYNC_MODE='explicit',
               ASTR_GPU_HALO_TRANSPORT='pinned', ASTR_GPU_PRECISION_MODE='fp64',
               ASTR_CHANNEL_FORCE_MODE='fixed', ASTR_CHANNEL_FORCE_FIXED='1.d-4')
    cases = {
        'missing': 'missing or incomplete',
        'pending': 'missing or incomplete',
        'truncated': 'payload is truncated or oversized',
        'nonfinite': 'contains nonfinite state',
        'generation': 'metadata mismatch',
        'topology': 'metadata mismatch',
        'oversized': 'payload is truncated or oversized',
        'version': 'Unsupported GPU exact checkpoint version',
        'backup_failure': 'backup rotation failed',
        'legacy': 'Legacy checkpoint: exact GPU continuation is not guaranteed',
    }
    results = {}
    for name, message in cases.items():
        case = out/name
        shutil.copytree(args.source/'datin', case/'datin')
        shutil.copytree(args.source/'checkpoint_step50', case/'outdat')
        for directory in ('bakup', 'monitor', 'islice', 'jslice', 'kslice', 'testout'):
            (case/directory).mkdir()
        set_restart(case/'datin/input.chl', 't')
        set_controller_steps(case/'datin/controller', 51, 51, 1)
        state = case/'outdat/restart_q.rank00000000.bin'
        data = bytearray(state.read_bytes())
        if name == 'missing':
            state.unlink()
        elif name == 'pending':
            state.with_suffix('.bin.tmp').touch()
        elif name == 'truncated':
            state.write_bytes(data[:-8])
        elif name == 'nonfinite':
            struct.pack_into('<d', data, 364, float('nan'))
            state.write_bytes(data)
        elif name == 'generation':
            struct.pack_into('<q', data, 132, 0)
            state.write_bytes(data)
        elif name == 'topology':
            struct.pack_into('<i', data, 8+6*4, 1)
            state.write_bytes(data)
        elif name == 'oversized':
            state.write_bytes(data+b'x')
        elif name == 'backup_failure':
            (case/'bakup'/state.name).mkdir()
        elif name in ('version', 'legacy'):
            aux = case/'outdat/auxiliary.txt'
            lines = aux.read_text().splitlines()
            lines = [('exact_gpu_restart_version=99' if name == 'version' else '')
                     if 'exact_gpu_restart_version=' in line else line for line in lines]
            aux.write_text('\n'.join(lines)+'\n')
        with (case/'run.log').open('w') as log:
            result = subprocess.run(['mpirun', '-np', '2', str(args.gpu.resolve()),
                                     'run', 'datin/input.chl'], cwd=case, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=60)
        text = (case/'run.log').read_text()
        passed = message in text and ((result.returncode == 0) == (name == 'legacy'))
        results[name] = dict(passed=passed, returncode=result.returncode)
        (out/'result.json').write_text(json.dumps(results, indent=2)+'\n')
        if not passed:
            raise RuntimeError(f'{name}: expected diagnostic missing or incorrect exit status')
    print(f'{len(cases)} checkpoint lifecycle checks passed')


if __name__ == '__main__':
    main()
