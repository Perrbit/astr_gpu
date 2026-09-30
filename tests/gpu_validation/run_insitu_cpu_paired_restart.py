"""Exact CPU TGV paired continuation, without changing legacy restart files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
from prepare_tgv_case import set_controller_deltat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('cpu', 'mpiexec', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'checks': []}

    def prepare(name, batch=None):
        case = args.output/name
        subprocess.run([sys.executable, str(root/'tests/gpu_validation/prepare_tgv_case.py'),
                        '--src-case', str(root/'examples/Taylor_Green_Vortex'),
                        '--dst-case', str(case), '--use-gpu', 'f', '--grid', '32,32,32',
                        '--maxstep', '4', '--feqchkpt', '2', '--deltat', '1.d-3',
                        '--lfilter', 't', '--diffterm', 't', '--scheme', '643e'], check=True)
        if batch:
            path = case/'datin/input.tgv'
            lines = path.read_text().splitlines()
            for i, line in enumerate(lines):
                if line.lstrip().startswith('# lrestar'):
                    for j in range(i+1, len(lines)):
                        if lines[j].strip() and not lines[j].lstrip().startswith('#'):
                            lines[j] = 't'
                            break
                    break
            else:
                raise ValueError('restart input marker missing')
            path.write_text('\n'.join(lines)+'\n')
        (case/'insitu.nml').write_text(
            '&insitu_run enabled=t, statistics=t, render=f,\n'
            "statistics_window=0.0005,0.0035, output_directory='outdat',\n"
            "host_budget_bytes=4294967296, batch_prefix='outdat/paired',\n"
            f"restore_batch='{batch or ''}'\n/\n")
        return case

    def run(case, ranks, expected=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith('ASTR_')}
        env.update(ASTR_INSITU_CONFIG=str(case/'insitu.nml'),
                   ASTR_FORCE_MPI_TOPOLOGY=f'{ranks},1,1')
        with (case/'run.log').open('w') as log:
            result = subprocess.run([str(args.mpiexec.resolve()), '--mca', 'coll_hcoll_enable',
                                     '0', '-np', str(ranks), str(args.cpu.resolve()),
                                     'run', 'datin/input.tgv'], cwd=case, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=180)
        if expected:
            assert result.returncode != 0 and expected in (case/'run.log').read_text()
        else:
            assert result.returncode == 0, case/'run.log'

    def hashes(batch):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in batch.iterdir() if p.is_file()}

    try:
        for ranks in (1, 2):
            continuous = prepare(f'np{ranks}_continuous')
            run(continuous, ranks)
            batch = continuous/'outdat/paired.step00000002'
            before = hashes(batch)
            resumed = prepare(f'np{ranks}_resumed', batch)
            run(resumed, ranks)
            for rank in range(ranks):
                names = [f'sample.statistics.step00000005.rank{rank:08d}.bin',
                         f'paired.step00000004/statistics.rank{rank:08d}.bin',
                         f'paired.step00000004/insitu_cpu_q.rank{rank:08d}.bin']
                for name in names:
                    assert (continuous/'outdat'/name).read_bytes() == (resumed/'outdat'/name).read_bytes(), name
            with h5py.File(continuous/'outdat/flowfield.h5') as a, h5py.File(resumed/'outdat/flowfield.h5') as b:
                for name in ('ro', 'u1', 'u2', 'u3', 'p', 't', 'nstep', 'time'):
                    assert a[name][...].tobytes() == b[name][...].tobytes(), name
            assert hashes(batch) == before
            report['checks'].append({'np': ranks, 'exact_q': 'byte-identical',
                                     'statistics': 'byte-identical', 'HDF': 'bitwise-identical'})
            if ranks == 2:
                for fault in ('missing', 'corrupt'):
                    broken = args.output/f'batch_{fault}'
                    shutil.copytree(batch, broken)
                    victim = broken/'insitu_cpu_q.rank00000001.bin'
                    if fault == 'missing':
                        victim.unlink()
                    else:
                        with victim.open('r+b') as stream:
                            stream.seek(-1, 2)
                            value = stream.read(1)
                            stream.seek(-1, 2)
                            stream.write(bytes([value[0] ^ 1]))
                    case = prepare(f'reject_{fault}', broken)
                    run(case, ranks, 'paired batch incomplete or changed')
                    report['checks'].append({'rejected': fault})
                for fault, expected in (
                        ('topology', 'paired batch incomplete or changed'),
                        ('window', 'paired point statistics record failed'),
                        ('parameters', 'CPU exact checkpoint metadata mismatch')):
                    case = prepare(f'reject_{fault}', batch)
                    if fault == 'window':
                        path = case/'insitu.nml'
                        path.write_text(path.read_text().replace('0.0005,0.0035', '0.0006,0.0035'))
                    elif fault == 'parameters':
                        set_controller_deltat(case/'datin/controller', '9.d-4')
                    run(case, 1 if fault == 'topology' else ranks, expected)
                    report['checks'].append({'rejected': fault})
            print(f'PASS CPU NP={ranks}: exact paired continuation', flush=True)
        report['status'] = 'passed-bounded-cpu-paired-restart'
    except BaseException as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
