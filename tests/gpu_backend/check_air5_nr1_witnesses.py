"""Find and check FP64 witnesses for frozen NR1 final projection inputs."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np

from diagnose_air5_projection_geometry import analyze, sequential_sum


def ratios(x, flux, geometry):
    sf, sw, tolerance = (float(geometry[k]) for k in ('scale_flux', 'scale_weight', 'tolerance'))
    coefficient = 64 * math.ulp(1.)
    scaled = [f / sf for f in flux]
    products = [(r / sw) * f for r, f in zip(x[16:21], scaled)]
    rounded = [abs(sequential_sum(v) - target) / (tolerance + coefficient * sequential_sum(map(abs, v)))
               for v, target in ((scaled, x[15] / sf), (products, (x[21] / sw) / sf))]
    # Independent check before any rounding of the normalization or products.
    scaled_exact = [Fraction(f) / Fraction(sf) for f in flux]
    products_exact = [Fraction(r) * f / Fraction(sw) for r, f in zip(x[16:21], scaled_exact)]
    exact = [abs(sum(v) - target) / (Fraction(tolerance) + Fraction(coefficient) * sum(map(abs, v)))
             for v, target in ((scaled_exact, Fraction(x[15]) / Fraction(sf)),
                               (products_exact, Fraction(x[21]) / Fraction(sw) / Fraction(sf)))]
    return rounded, exact


def witness(x, geometry, max_steps):
    start = geometry['rounded_physical_witness']['flux']
    candidates = [(0, None, None, start)]
    for s in range(5):
        for direction in (-1, 1):
            f = list(start)
            for n in range(1, max_steps + 1):
                f[s] = math.nextafter(f[s], direction * math.inf)
                candidates.append((n, s, direction, list(f)))
    for steps, species, direction, f in sorted(candidates, key=lambda row: row[0]):
        if not all(l <= v <= u for l, v, u in zip(x[5:10], f, x[10:15])):
            continue
        rounded, exact = ratios(x, f, geometry)
        if max(*rounded, *exact) <= 1:
            return dict(flux=f, exact_mass_start=start, ulps=steps, species_index=species,
                        direction=direction, rounded_budget_ratios=rounded, exact_budget_ratios=list(map(float, exact)))
    raise AssertionError('no witness in the tested neighborhood; this does not prove infeasibility')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('trace', 'binary', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    trace, out, binary = (getattr(args, k).resolve() for k in ('trace', 'output', 'binary'))
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out/'driver.py')
    shutil.copy2(binary, out/'probe')
    report = dict(rows=[], source_hashes={}, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    for r in json.loads((trace/'report.json').read_text())['cases']:
        assert int(r['projection']) == 13 and r['status'] == 201
        index = r['index']
        log = trace/f'face{index}.log'
        data = log.read_bytes(); lines = data.decode().splitlines()
        assert len(lines) == 2 and lines[1].startswith('INPUT ')
        x = list(map(float, lines[1].split()[1:]))
        assert len(x) == 22
        geometry = analyze(np.asarray(x))
        selected = witness(x, geometry, 2)
        face = trace/f'face{index}.bin'
        blob = face.read_bytes()
        n, *face_data = struct.unpack('<i80d', blob)
        assert n == 1 and face_data[79] == 4
        report['source_hashes'][str(index)] = [hashlib.sha256(v).hexdigest() for v in (data, blob)]
        report['rows'].append(dict(index=index, beta=float(r['beta']), **selected))
        with (out/'inputs.bin').open('ab') as f:
            f.write(struct.pack('<108d', *face_data, float(r['beta']), *x, *selected['flux']))
    cmd = [str(out/'probe'), str(out/'inputs.bin')]
    report['device_command'] = cmd
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    with (out/'device.log').open('w') as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    lines = (out/'device.log').read_text().splitlines()
    assert len(lines) == len(report['rows'])
    for row, line in zip(report['rows'], lines):
        tag, mass, moment, left, right = line.split()
        assert tag == 'AIR5_WITNESS' and max(float(mass), float(moment)) <= 1
        row['device_budget_ratios'] = [float(mass), float(moment)]
        row['left_admissible'], row['right_admissible'] = (bool(int(x)) for x in (left, right))
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f"{len(lines)} verified FP64 projection witnesses; max ULPs: {max(r['ulps'] for r in report['rows'])}; "
          f"both adjacent states admissible: {sum(r['left_admissible'] and r['right_admissible'] for r in report['rows'])}")


if __name__ == '__main__':
    main()
