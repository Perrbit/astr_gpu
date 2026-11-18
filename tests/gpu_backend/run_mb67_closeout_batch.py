"""Approved-allocation task body for remaining MB6/MB7 consumers; no scheduler calls."""
import argparse
from array import array
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct

from run_mb67_consumer_batch import assignment, command, environment, clocks, archive_schedule


def read_snapshot(path, precision):
    with path.open('rb') as stream:
        header = struct.unpack('<5i', stream.read(20))
        values = array('d' if precision == 'fp64' else 'f')
        values.frombytes(stream.read())
    widths = [n+2*header[3]+1 for n in header[:3]]
    if header[4] != 5 or len(values) != math.prod(widths)*5:
        raise AssertionError('snapshot component/precision layout')
    return header, widths, values


def compare_snapshot(actual, reference, precision, absolute_limit):
    h, widths, values = read_snapshot(actual, precision)
    rh, rw, expected = read_snapshot(reference, precision)
    if h != rh or widths != rw:
        raise AssertionError('same-topology snapshot header')
    nodes = math.prod(widths); maximum = [0.]*5; scale = [1.]*5
    minimum = [math.inf, math.inf]
    for k in range(h[2]+1):
        for j in range(h[1]+1):
            for i in range(h[0]+1):
                p = ((k+h[3])*widths[1]+j+h[3])*widths[0]+i+h[3]
                q = [values[p+c*nodes] for c in range(5)]
                r = [expected[p+c*nodes] for c in range(5)]
                if not all(math.isfinite(v) for v in q+r) or q[0] <= 0:
                    raise AssertionError('snapshot physical finiteness/density')
                internal = q[4]-.5*sum(v*v for v in q[1:4])/q[0]
                if not math.isfinite(internal) or internal <= 0:
                    raise AssertionError('snapshot physical internal energy')
                minimum = [min(minimum[0], q[0]), min(minimum[1], internal)]
                for c in range(5):
                    maximum[c] = max(maximum[c], abs(q[c]-r[c])); scale[c] = max(scale[c], abs(r[c]))
    if absolute_limit is not None and max(maximum) > absolute_limit:
        raise AssertionError(('mixed trajectory absolute gate', maximum))
    return dict(max_abs=maximum, scaled=[a/b for a, b in zip(maximum, scale)],
                density_min=minimum[0], internal_energy_min=minimum[1])


def physical_snapshot(path, precision):
    h, widths, values = read_snapshot(path, precision); nodes = math.prod(widths)
    minimum = [math.inf, math.inf]
    for k in range(h[2]+1):
        for j in range(h[1]+1):
            for i in range(h[0]+1):
                p = ((k+h[3])*widths[1]+j+h[3])*widths[0]+i+h[3]
                q = [values[p+c*nodes] for c in range(5)]
                if not all(math.isfinite(v) for v in q) or q[0] <= 0:
                    raise AssertionError('snapshot physical finiteness/density')
                internal = q[4]-.5*sum(v*v for v in q[1:4])/q[0]
                if not math.isfinite(internal) or internal <= 0:
                    raise AssertionError('snapshot physical internal energy')
                minimum = [min(minimum[0], q[0]), min(minimum[1], internal)]
    return dict(density_min=minimum[0], internal_energy_min=minimum[1])


def compare_sensor(actual, reference):
    def read(path):
        with path.open('rb') as stream:
            header = struct.unpack('<3i', stream.read(12)); nodes = math.prod(n+1 for n in header)
            raw = struct.unpack('<'+'d'*nodes, stream.read(nodes*8)); mask = stream.read(nodes)
            if len(mask) != nodes or stream.read(1) or not all(math.isfinite(v) for v in raw):
                raise AssertionError('physical sensor packet')
        return header, raw, mask
    ah, a, am = read(actual); bh, b, bm = read(reference)
    if ah != bh or am != bm:
        raise AssertionError('MP3 stage integer masks differ')
    return max(abs(x-y) for x, y in zip(a, b))


def output(case, phase, restore):
    path = case/'datin/input.output'; text = path.read_text()
    if restore:
        text = assignment(text, 'restore_directory', repr(str(restore)))
    if phase in ('quiet', 'monitor_off'):
        # Disable checkpoint/field sections, but keep the AP monitor in the quiet AP case.
        text = re.sub(r'(?ms)^&(checkpoint|volume|slices)\b.*?^/',
            lambda m: assignment(m[0], 'enabled', '.false.'), text)
    path.write_text(text)


def run_case(build, template, target, family, phase, mpiexec, restore=None, cpu=False):
    source = template/('monitor_off_datin' if phase == 'monitor_off' else 'datin')
    shutil.copytree(source, target/'datin', copy_function=shutil.copyfile)
    (target/'outdat/new').mkdir(parents=True)
    if family.get('wall_csv'):
        (target/'outdat/render').mkdir()
    if (template/'inflow').is_dir():
        shutil.copytree(template/'inflow', target/'inflow')
    if family.get('hit_resource') and restore:
        (target/'datin/velocity.h5').unlink()
    product = template/'input.products.jsonl'
    (target/'input.products.jsonl').write_text(product.read_text() if family['catalog'] and phase != 'quiet'
        else '{"catalog_schema":"astr-insitu-products-v1"}\n')
    if phase == 'quiet' and family.get('wall_csv'):
        path = target/'datin/input.insitu'; path.write_text(assignment(path.read_text(), 'enabled', '.false.'))
    steps = 2 if phase == 'prefix' else family['steps']
    path = target/'datin/input.run'; text = assignment(path.read_text(), 'max_steps', str(steps))
    if phase == 'quiet' and family.get('cpu_reference'):
        text = assignment(text, 'enabled', '.false.')
    path.write_text(text)
    if cpu:
        path = target/'datin/input.case'; path.write_text(assignment(path.read_text(), 'use_gpu', '.false.'))
    output(target, phase, restore)
    transport = 'pinned' if cpu else family['transport']
    env = environment(transport, family['topology'])
    if family.get('wall_csv'):
        env['ASTR_INSITU_CONFIG'] = 'datin/input.insitu'
    if family['backend'] == 'cuda' and transport == 'device':
        env['OMPI_MCA_opal_cuda_support'] = '1'
    env['ASTR_VALIDATION_RHS_STEP'] = '0' if family.get('sensor') else str(family['steps'])
    env['ASTR_VALIDATION_RHS_STEP_END'] = str(family['steps'])
    executable = build/('build_'+family['precision'])/'bin/astr'
    command([mpiexec, '--bind-to', 'none', '-n', math.prod(family['topology']), executable, 'run'], target, env, target/'run.log')
    if not cpu:
        text = (target/'run.log').read_text()
        expected = family['backend']
        devices = re.findall(r'ASTR_GPU_CPP rank=(\d+) backend=(\S+) device=(\d+)', text)
        if sorted(devices) != [(str(i), expected, str(i)) for i in range(math.prod(family['topology']))]:
            raise AssertionError('actual backend and rank/device mapping')
    return target


def compare_csv(actual, reference, exact):
    a = [line for line in actual.read_text().splitlines() if line and not line.startswith('#')]
    b = [line for line in reference.read_text().splitlines() if line and not line.startswith('#')]
    if exact:
        if actual.read_bytes() != reference.read_bytes():
            raise AssertionError('resumed wall curve words differ')
        return 0.
    if len(a) != len(b) or a[0] != b[0]:
        raise AssertionError('wall curve row/column identity')
    maximum = 0.
    for x, y in zip(a[1:], b[1:]):
        xx = [float(v) for v in x.split(',')]; yy = [float(v) for v in y.split(',')]
        if len(xx) != len(yy) or not all(math.isfinite(v) for v in xx+yy):
            raise AssertionError('wall curve finite fields')
        maximum = max(maximum, *(abs(u-v)/max(1., abs(v)) for u, v in zip(xx, yy)))
    if maximum > 2e-10:
        raise AssertionError('wall curve scaled difference')
    return maximum


def wall_curves(case):
    result = {}
    for path in (case/'outdat/new/wall_products').glob('*/version*/segment*/frame.*.csv'):
        key = path.parents[2].name+'/'+path.name
        if key in result:
            raise AssertionError('multiple versions of a fixed wall product')
        result[key] = path
    return result


def execute(build, validation, root, mpiexec, backend):
    root.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((validation/'manifest.json').read_text()); summary = dict(families={})
    for name, settings in manifest['families'].items():
        family = settings | dict(backend=backend); template = validation/'families'/name
        print('CLOSEOUT', name, flush=True); paths = {}
        if family.get('cpu_reference'):
            paths['cpu'] = run_case(build, template, root/name/'cpu', family, 'full', mpiexec, cpu=True)
        paths['full'] = run_case(build, template, root/name/'full', family, 'full', mpiexec)
        full = paths['full']; row = dict(snapshots=[]); step = family['steps']
        base_clock = clocks(full, count=step)
        if not family.get('cpu_reference') and clocks(template, count=step) != base_clock:
            raise AssertionError('frozen reference clock')
        for rank in range(math.prod(family['topology'])):
            packet = f'outdat/rhs.cpp_final.step{step:08d}.rk01.rank{rank:08d}.bin'
            frozen = template/f'reference.rank{rank:08d}.bin'
            if family['reference_kind'] == 'snapshot':
                row['snapshots'].append(compare_snapshot(full/packet, frozen, family['precision'],
                    2e-6 if family.get('sensor') else None))
            else:
                row['snapshots'].append(physical_snapshot(full/packet, family['precision']))
        comparator = build/('build_'+family['precision'])/'bin/astr_backend_compare'
        final = Path(f'outdat/new/checkpoints/step{step:012d}')
        if family['precision'] == 'fp64' and not family.get('sensor'):
            reference = paths['cpu']/final/'state.h5' if family.get('cpu_reference') else template/'reference.h5'
            command([comparator, 'field', reference, full, step], full, os.environ, full/'reference_compare.log')
        if family.get('sensor'):
            row['raw_sensor_max_abs'] = 0.; count = 0
            for frozen in sorted((template/'sensor').glob('*.bin')):
                row['raw_sensor_max_abs'] = max(row['raw_sensor_max_abs'], compare_sensor(full/'outdat'/frozen.name, frozen))
                count += 1
            if count != 12:
                raise AssertionError('six phases on two ranks required')
            row['integer_sensor_masks_exact'] = True
        if family.get('wall_csv'):
            curves = wall_curves(full)
            if sorted(curves) != family['wall_csv_inventory']:
                raise AssertionError('actual wall curve inventory')
            row['wall_csv_scaled'] = [compare_csv(curve, template/'curves'/key, False) for key, curve in curves.items()]
        phases = ('quiet', 'prefix', 'tail') if family['checkpoint'] else ('quiet',)
        for phase in phases:
            restore = paths['prefix']/'outdat/new/checkpoints/step000000000002' if phase == 'tail' else None
            paths[phase] = run_case(build, template, root/name/phase, family, phase, mpiexec, restore)
        if clocks(paths['quiet'], count=step) != base_clock:
            raise AssertionError('quiet completed-step clock')
        for rank in range(math.prod(family['topology'])):
            packet = f'outdat/rhs.cpp_final.step{step:08d}.rk01.rank{rank:08d}.bin'
            if (full/packet).read_bytes() != (paths['quiet']/packet).read_bytes():
                raise AssertionError('quiet q/halo words differ')
        if family['checkpoint']:
            tail = paths['tail']
            if clocks(paths['prefix'], count=2) != base_clock[:2] or clocks(tail, 3, step-2) != base_clock[2:]:
                raise AssertionError('prefix/tail completed-step clock')
            for filename in ['state.h5', *family['histories']]:
                command([comparator, 'exact', full/final/filename, tail/final/filename], full, os.environ,
                        full/(filename+'.exact.log'))
            for filename in ('control.bin', 'insitu_control.bin'):
                if (full/final/filename).read_bytes() != (tail/final/filename).read_bytes():
                    raise AssertionError('exact controller history')
            if archive_schedule(full/final/'archives.bin') != archive_schedule(tail/final/'archives.bin'):
                raise AssertionError('exact archive schedule')
            if family.get('hit_resource'):
                if (tail/'datin/velocity.h5').exists() or (full/'outdat/new/resources/velocity.h5').read_bytes() != (tail/'outdat/new/resources/velocity.h5').read_bytes():
                    raise AssertionError('self-contained HIT resource')
            row['exact_state_and_enabled_history'] = True
        if family.get('wall_csv'):
            for key, curve in wall_curves(paths['tail']).items():
                compare_csv(curve, wall_curves(full)[key], True)
        if family.get('adaptive'):
            a = monitor(full); b = monitor(template)
            identity = lambda row: [row['step'], row['time'], row['active'], row['windows']]
            if [identity(x) for x in a] != [identity(x) for x in b]:
                raise AssertionError('adaptive monitor identities/flags')
            row['monitor_max_abs'] = max(abs(u-v) for x, y in zip(a, b)
                for u, v in zip([x['energy'], *x['rates']], [y['energy'], *y['rates']]))
            if row['monitor_max_abs'] > 2e-10:
                raise AssertionError('adaptive monitor numerical difference')
            if monitor(paths['tail']) != [x for x in a if x['step'] > 2]:
                raise AssertionError('resumed adaptive monitor values/flags')
            off = run_case(build, template, root/name/'monitor_off', family, 'monitor_off', mpiexec)
            for rank in range(math.prod(family['topology'])):
                packet = f'outdat/rhs.cpp_final.step{step:08d}.rk01.rank{rank:08d}.bin'
                if (off/packet).read_bytes() != (full/packet).read_bytes():
                    raise AssertionError('disabled adaptive monitor changed q/halo')
            inventory = lambda case: sorted(p.parent.name for p in (case/'outdat/new/fields').glob('*/*/data.h5'))
            if inventory(full) != inventory(template) or inventory(paths['prefix'])+inventory(paths['tail']) != inventory(full):
                raise AssertionError('adaptive actual volume frame inventory')
        summary['families'][name] = row
        (root/'report.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def monitor(case):
    pattern = re.compile(r'ASTR_AP_MONITOR step=(\d+) time=\s*(\S+) kinetic_energy=\s*(\S+) rates=(.*?) active=(.*?) windows=(.*?) wall_seconds=')
    rows = []
    for match in pattern.finditer((case/'run.log').read_text()):
        rates = list(map(float, match[4].split()))
        rows.append(dict(step=int(match[1]), time=float(match[2]), energy=float(match[3]), rates=rates,
                         active=match[5].split(), windows=match[6].split()))
    if not rows:
        raise AssertionError('adaptive monitor records missing')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('build-root', 'validation', 'output', 'mpiexec'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--backend', choices=('cuda', 'gpufusion'), required=True)
    args = parser.parse_args()
    execute(args.build_root.resolve(), args.validation.resolve(), args.output.resolve(), args.mpiexec.resolve(), args.backend)


if __name__ == '__main__':
    main()
