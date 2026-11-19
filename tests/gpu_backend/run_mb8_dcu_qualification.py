"""Collect MB8 qualification records in an approved allocation (no scheduler calls).

Use --local-rehearsal only on a two-GPU NVIDIA workstation. Verify the G1
reference and the HDF5 data offline; this collector needs only the stdlib.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

from run_mb67_consumer_batch import assignment, clocks, command, environment


def launch(argv, work, env, log):
    command(argv, work, env, log)
    return log.read_text()


def phase_devices(text, ranks, devices):
    bus = r'[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]'
    records = re.findall(
        rf'^AIR5_CHEMISTRY_PHASE_PASS rank=(\d+) cases=9 global_failures=2 device_words=exact device=(\d+) pci=({bus})$',
        text, flags=re.IGNORECASE | re.MULTILINE)
    rows = []
    for rank, device, pci in records:
        domain, address = pci.split(':', 1)
        rows.append((int(rank), int(device), f'{int(domain, 16):04x}:{address.lower()}'))
    rows.sort()
    assert [(r, d) for r, d, _ in rows] == [(r, r % devices) for r in range(ranks)], rows
    # On local four-rank rehearsal, ranks sharing an ordinal must share its
    # PCI identity. Platform qualification needs a distinct device per rank.
    assert len({(d, pci) for _, d, pci in rows}) == len({pci for _, _, pci in rows}) == min(ranks, devices), rows
    return rows


def collect_components(args, root):
    ieee = root / 'ieee'
    ieee.mkdir()
    shutil.copy2(args.inputs / 'ieee/input.in', ieee / 'input.in')
    env = environment('pinned')
    launch([args.bin_dir / 'astr_backend_air5_ros2', 'contracts', 'input.in', 'transactions.dat'],
           ieee, env, ieee / 'run.log')
    records = (ieee / 'transactions.dat').read_text().splitlines()
    rows = [list(map(int, line.split())) for line in records if line and not line.startswith('#')]
    assert len(rows) == 24 and all(len(r) == 10 and r[2] == r[3] and r[4] == 1 for r in rows)
    assert [line for line in records if line.startswith('# AIR5_IEEE ')] == [
        '# AIR5_IEEE compensated_add=1 fp64_subnormal_carry=1 compensated_rk=1']

    g1 = root / 'g1'
    g1.mkdir()
    frozen = args.inputs / 'g1'
    reference = json.loads((frozen / 'reference.json').read_text())
    assert len(reference['cases']) == 11
    for case in reference['cases']:
        name = case['name']
        shutil.copy2(frozen / (name + '.in'), g1 / (name + '.in'))
        argv = [args.bin_dir / 'astr_backend_air5_coupling', 'batch', name + '.in', name + '.out',
                '1e-9', '1e-13', '4', 'history']
        if args.specialized:
            argv.append('specialized-counted')
        launch(argv, g1, env | {'ASTR_AIR5_SOURCE_EVALUATION': 'joint'}, g1 / (name + '.log'))
    shutil.copy2(frozen / 'reference.json', g1 / 'reference.json')
    if args.g1_checker is not None:
        with (g1 / 'gate.log').open('x') as log:
            subprocess.run([sys.executable, '-B', args.g1_checker, g1 / 'reference.json', g1], cwd=g1,
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    phase = root / 'chemistry_phase'
    phase.mkdir()
    shutil.copy2(args.inputs / 'phase/input.bin', phase / 'input.bin')
    runs = []
    for dims in ((1, 1, 1), (2, 1, 1), (2, 2, 1)):
        ranks = math.prod(dims)
        name = ''.join(map(str, dims))
        exe = args.bin_dir / 'astr_backend_air5_chemistry_phase'
        argv = [args.mpiexec, '--bind-to', 'none', '-np', ranks, exe, 'input.bin', *dims]
        if args.specialized:
            argv.append('specialized')
        text = launch(argv, phase, env, phase / (name + '.log'))
        devices = 2 if args.local_rehearsal else 4
        bound = phase_devices(text, ranks, devices)
        runs.append(dict(topology=dims, rank_devices=[(r, d) for r, d, _ in bound],
                         rank_pci=[(r, pci) for r, _, pci in bound],
                         rank_cases=9*ranks, failure_observations=2*ranks))
    return dict(ieee_transactions=24, g1_collected=11, chemistry_phase=runs)


def run_flow(args, work, dims, transport, first, count, controls):
    ranks = math.prod(dims)
    env = environment(transport, dims)
    env.pop('ASTR_VALIDATION_RHS_STEP')
    env.pop('ASTR_VALIDATION_RHS_PREFIX')
    if args.local_rehearsal and transport == 'device':
        env['OMPI_MCA_opal_cuda_support'] = '1'
    env.update(controls)
    if args.existing_paths:
        env.update(ASTR_AIR5_FILTER_VALIDATION='on', ASTR_AIR5_TOP_GPU_VALIDATION='on',
                   ASTR_AIR5_TOP_RESTART_VALIDATION='on')
    if args.specialized:
        env['ASTR_AIR5_CHEMISTRY_DISPATCH'] = 'specialized'
    argv = [args.mpiexec, '--bind-to', 'none', '-np', ranks, args.bin_dir / 'astr', 'run', 'datin/input.case']
    log = launch(argv, work, env, work / 'run.log')
    expected = 'cuda' if args.local_rehearsal else 'gpufusion'
    mapping = sorted((int(r), b, int(d)) for r,b,d in re.findall(
        r'ASTR_GPU_CPP rank=(\d+) backend=(\S+) device=(\d+)', log))
    assert mapping == [(r, expected, r) for r in range(ranks)], mapping
    assert 'The job is done!' in log and 'ASTR_GPU_CPP AIR5 resident RK3; Strang=T' in log
    assert f'ASTR_GPU_CPP transport={transport};' in log
    return dict(topology=dims, transport=transport, rank_devices=mapping, clocks=clocks(work, first=first, count=count))


def hbl(args, root, name, dims, transport, steps=4, restore=None, quiet=False):
    work = root / 'hbl' / name
    src = args.inputs / 'hbl'
    shutil.copytree(src / 'datin', work / 'datin')
    (work / 'outdat/new').mkdir(parents=True)
    runtime = work / 'datin/input.run'
    text = assignment(runtime.read_text(), 'max_steps', str(steps))
    if quiet:
        text = assignment(text, 'enabled', '.false.')
    runtime.write_text(text)
    output = work / 'datin/input.output'
    text = output.read_text()
    if restore is not None:
        assert text.count("restore_directory=''") == 1
        text = text.replace("restore_directory=''", f"restore_directory='{restore}'")
    if quiet:
        text, count = re.subn(r'(?ms)^&volume\b.*?^/', lambda m: assignment(m[0], 'enabled', '.false.'), text)
        assert count == 1
    output.write_text(text)
    products = work / 'input.products.jsonl'
    if quiet:
        products.write_text('{"catalog_schema":"astr-insitu-products-v1"}\n')
    else:
        shutil.copy2(src / products.name, products)
    first = 1 if restore is None else 3
    controls = dict(ASTR_AIR5_PRIMITIVE_REUSE='chemistry', ASTR_AIR5_C4_CONSERVATION='f' if quiet else 't')
    return work, run_flow(args, work, dims, transport, first, steps-first+1, controls)


def incident_sbli(args, root):
    work = root / 'incident_sbli'
    shutil.copytree(args.inputs / 'incident_sbli/datin', work / 'datin')
    (work / 'outdat/new').mkdir(parents=True)
    runtime = work / 'datin/input.run'
    runtime.write_text(assignment(runtime.read_text(), 'max_steps', '1'))
    controls = dict(ASTR_AIR5_PRIMITIVE_REUSE='off', ASTR_AIR5_C4_CONSERVATION='f')
    return run_flow(args, work, (2, 1, 1), 'pinned', 1, 1, controls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'bin-dir', 'mpiexec', 'output'):
        parser.add_argument('--' + name, type=lambda s: Path(s).resolve(), required=True)
    parser.add_argument('--local-rehearsal', action='store_true')
    parser.add_argument('--specialized', action='store_true')
    parser.add_argument('--existing-paths', action='store_true',
                        help='collect the combined existing flow paths using specialized chemistry, without component probes')
    parser.add_argument('--incident-sbli', action='store_true',
                        help='add a single completed-step incident SBLI check to the existing-paths matrix')
    parser.add_argument('--g1-checker', type=lambda s: Path(s).resolve())
    args = parser.parse_args()
    if args.incident_sbli and not args.existing_paths:
        parser.error('--incident-sbli requires --existing-paths')
    if args.existing_paths:
        case = (args.inputs / 'hbl/datin/input.case').read_text()
        for key, value in (('convection_limiter', "'symmetric_species'"), ('top_mode', "'characteristic'"),
                           ('filter_mode', "'conservative-flux'"), ('lfilter', '.true.')):
            if not re.search(rf'^\s*{key}\s*=\s*{re.escape(value)}\s*$', case, re.MULTILINE):
                parser.error(f'--existing-paths requires {key}={value} in the HBL input')
        args.specialized = True
    if args.incident_sbli:
        case = (args.inputs / 'incident_sbli/datin/input.case').read_text()
        if not re.search(r"^\s*flowtype\s*=\s*'air5sbli'\s*$", case, re.MULTILINE):
            parser.error('--incident-sbli requires the incident SBLI input')
    if not args.existing_paths and args.specialized and args.g1_checker is None:
        parser.error('specialized qualification requires the frozen G1 check before phase runs')
    root = args.output
    root.mkdir()
    names = ('astr',) if args.existing_paths else (
        'astr', 'astr_backend_air5_ros2', 'astr_backend_air5_coupling', 'astr_backend_air5_chemistry_phase')
    report = dict(local_rehearsal=args.local_rehearsal, chemistry_dispatch='specialized' if args.specialized else 'general',
                  existing_paths=args.existing_paths, g1_checked_before_phase=not args.existing_paths and args.g1_checker is not None,
                  scope='collection; full HDF5 numerical checks offline',
                  binaries={name: hashlib.sha256((args.bin_dir / name).read_bytes()).hexdigest()
                            for name in names})
    if not args.existing_paths:
        report.update(collect_components(args, root))
    cases = [('111_pinned', (1, 1, 1), 'pinned'), ('211_pinned', (2, 1, 1), 'pinned'),
             ('112_device', (1, 1, 2), 'device')]
    if not args.local_rehearsal:
        cases.extend([('221_pinned', (2, 2, 1), 'pinned'), ('221_device', (2, 2, 1), 'device')])
    report['hbl'] = {}
    for name,dims,transport in cases:
        _, report['hbl'][name] = hbl(args, root, name, dims, transport)
    name,dims,transport = cases[-1]
    seed, report['hbl']['seed'] = hbl(args, root, 'seed', dims, transport, steps=2)
    source = seed / 'outdat/new/checkpoints/step000000000002'
    _, report['hbl']['resumed'] = hbl(args, root, 'resumed', dims, transport, restore=source)
    _, report['hbl']['quiet'] = hbl(args, root, 'quiet', dims, transport, quiet=True)
    full = report['hbl'][name]['clocks']
    assert full == report['hbl']['seed']['clocks'] + report['hbl']['resumed']['clocks']
    assert full == report['hbl']['quiet']['clocks']
    report['continuation_baseline'] = name
    if args.incident_sbli:
        report['incident_sbli'] = incident_sbli(args, root)
    (root / 'collection.json').write_text(json.dumps(report, indent=2) + '\n')
    print(root / 'collection.json')


if __name__ == '__main__':
    main()
