"""Check the solver clock during live controller timestep reloads (Linux)."""
import argparse
import errno
import json
import math
import os
from pathlib import Path
import pty
import re
import select
import shutil
import signal
import struct
import subprocess
import sys
import time

from prepare_tgv_case import set_controller_deltat


CLOCK = re.compile(r"ASTR_CFL complete_step=(\d+) state_time=\s*(\S+) dt=\s*(\S+)")
MAXSTEP = 11  # The existing loop includes step label MAXSTEP.


def check_clock(records, step, state_time, dt):
    if step != len(records) or not math.isfinite(dt) or dt <= 0:
        raise ValueError(f"invalid clock record: {step}, {state_time}, {dt}")
    expected = (records[-1]['expected_time'] if records else 0.0) + dt
    error = abs(state_time - expected)
    if not math.isfinite(state_time) or error > 8 * math.ulp(expected):
        raise ValueError(f"step label {step}: state_time={state_time:.17g}, "
                         f"sum(dt_used)={expected:.17g}, error={error:.17g}")
    return dict(step=step, time=state_time, dt=dt, expected_time=expected, error=error)


def run_case(root, args, backend, ranks, mode):
    case = args.output / f'{backend}_np{ranks}_{mode}'
    subprocess.run([
        sys.executable, str(root / 'tests/gpu_validation/prepare_tgv_case.py'),
        '--src-case', str(root / 'examples/Taylor_Green_Vortex'), '--dst-case', str(case),
        '--use-gpu', 't' if backend == 'gpu' else 'f', '--grid', '16,16,16',
        '--maxstep', str(MAXSTEP), '--feqchkpt', '1', '--deltat', '1.d-3',
        '--lfilter', 't', '--diffterm', 't', '--scheme', '643e'], check=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(ASTR_FORCE_MPI_TOPOLOGY=f'{ranks},1,1', ASTR_GPU_SYNC_MODE='explicit',
               ASTR_GPU_HALO_TRANSPORT='pinned', ASTR_GPU_PRECISION_MODE='fp64',
               ASTR_INSITU_SAMPLE_PREFIX='outdat/clock_sample')
    command = [str(args.mpiexec), '--mca', 'coll_hcoll_enable', '0', '-np', str(ranks),
               str(args.executable), 'run', 'datin/input.tgv']
    report = dict(status='running', backend=backend, np=ranks, mode=mode,
                  command=command, records=[], changes=[])
    master, slave = pty.openpty()
    process = None
    try:
        with (case / 'run.log').open('wb') as log:
            # A PTY exposes progress promptly; the test does not guess solver speed.
            process = subprocess.Popen(command, cwd=case, env=env, stdin=subprocess.DEVNULL,
                                       stdout=slave, stderr=slave, start_new_session=True)
            os.close(slave)
            slave = -1
            pending = b''
            deadline = time.monotonic() + 180
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError('clock test exceeded 180 seconds')
                if not select.select([master], [], [], 0.2)[0]:
                    continue
                try:
                    data = os.read(master, 65536)
                except OSError as error:
                    if error.errno != errno.EIO:
                        raise
                    break
                if not data:
                    break
                log.write(data)
                log.flush()
                pending += data
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    match = CLOCK.search(line.decode(errors='replace'))
                    if not match:
                        continue
                    step = int(match[1])
                    record = check_clock(report['records'], step, float(match[2]), float(match[3]))
                    report['records'].append(record)
                    # Replace atomically: a rank reads either complete old or new input.
                    target = '2.d-3' if step == 0 else '5.d-4' if step == 5 else None
                    if mode == 'reload' and target:
                        controller = case / 'datin/controller'
                        temporary = controller.with_name('controller.next')
                        shutil.copyfile(controller, temporary)
                        set_controller_deltat(temporary, target)
                        temporary.replace(controller)
                        report['changes'].append(dict(after_record=step, requested_dt=target))
            if process.wait(timeout=10) != 0:
                raise RuntimeError(f'solver failed; see {case / "run.log"}')
        records = report['records']
        if len(records) != MAXSTEP + 1:
            raise ValueError(f'expected {MAXSTEP + 1} clock records, found {len(records)}')
        observed = {r['dt'] for r in records}
        expected = {0.001, 0.002, 0.0005} if mode == 'reload' else {0.001}
        if observed != expected:
            raise ValueError(f'controller changes were not exercised: {observed}')
        for record in records:
            for rank in range(ranks):
                path = case / 'outdat' / (
                    f'clock_sample.step{record["step"] + 1:08d}.rank{rank:08d}.bin')
                with path.open('rb') as stream:
                    magic = stream.read(8)
                    header = struct.unpack('<9i', stream.read(36))
                    clock = struct.unpack('<2d', stream.read(16))
                if (magic != b'ASTRIS01' or header[:3] != (1, record['step'] + 1, rank)
                        or clock != (record['time'], record['dt'])):
                    raise ValueError(f'completed-step sample/CFL clock mismatch: {path}')
        report.update(status='passed', samples_checked=len(records) * ranks,
                      max_time_error=max(r['error'] for r in records))
    except BaseException as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        os.close(master)
        if slave >= 0:
            os.close(slave)
        (case / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('executable', 'mpiexec', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--backends', nargs='+', choices=('cpu', 'gpu'), default=['cpu', 'gpu'])
    parser.add_argument('--ranks', nargs='+', type=int, choices=(1, 2), default=[1, 2])
    parser.add_argument('--modes', nargs='+', choices=('fixed', 'reload'), default=['fixed', 'reload'])
    args = parser.parse_args()
    args.executable = args.executable.resolve(strict=True)
    args.mpiexec = args.mpiexec.resolve(strict=True)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    report = dict(status='running', checks=[])
    try:
        for backend in args.backends:
            for ranks in args.ranks:
                for mode in args.modes:
                    result = run_case(root, args, backend, ranks, mode)
                    report['checks'].append(result)
                    print(f'PASS {backend} NP={ranks} {mode}: '
                          f'{result["samples_checked"]} sample clocks', flush=True)
        report['status'] = 'passed-bounded-completed-step-clock'
    except BaseException as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
