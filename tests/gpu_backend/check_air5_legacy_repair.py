"""Prepare and compare legacy AIR5 point-repair fixtures (no filter stencil)."""
import argparse
import json
from pathlib import Path
import struct

import numpy as np

from run_air5_properties_components import read


def prepare(properties, out):
    out.mkdir(parents=True, exist_ok=False)
    statuses, values = read(properties)
    assert np.all(statuses[120] == 0)
    q = values[120, :11].copy()
    floor = values[0, 109:114]
    rho = q[0]
    rows, labels = [], []
    for comp in (0, 1):
        tiny = np.ldexp(q, -58)
        def add(label, expected, raw, base=None, carry=None):
            base = q if base is None else base
            carry = np.ldexp(raw, -58) if carry is None else carry
            labels.append(dict(case=label, compensated=bool(comp), expected_status=expected))
            rows.append(struct.pack('<i33d', comp, *base, *raw, *carry))

        for exponent in (-40, -70, -110):
            raw = q.copy();raw[9] = 2.**exponent
            add(f'trace_{exponent}', 0, raw)
        for factor in (1., .71, .09):
            raw = q.copy();raw[9] = -q[9]*factor
            add(f'negative_{factor}', 0, raw)
        base = q.copy();base[5:10] = rho*np.array([.40,.40,.04,.08,.08]);raw = q.copy();raw[9] = -q[9]
        add('closure_tie', 0, raw, base)
        base[5],base[6] = .30*rho,.50*rho
        add('oxygen_closure', 0, raw, base)
        raw[9] = 0.
        add('zero_support', 0, raw)
        for key, index, value, status in [('invalid_density',0,-rho,2),('nonfinite_species',9,float('nan'),7),
                                          ('ev_after_repair',10,-1.,3),('et_after_repair',4,-1.,3)]:
            raw = q.copy();raw[5] += .01*rho;raw[index] = value
            add(key,status,raw)
        for key, index, value in [('negative_base',9,-q[9]),('base_mass_mismatch',5,0.)]:
            base = q.copy();base[index] = value
            add(key,3,q,base)
        if comp:
            raw = q.copy();raw[9] = 0.
            add('negative_represented_trace',3,raw,carry=tiny)
            carry = tiny.copy();carry[1] = float('nan')
            add('nonfinite_carry',7,q,carry=carry)
    (out/'input.bin').write_bytes(struct.pack('<i',len(rows))+b''.join(rows))
    (out/'floor.bin').write_bytes(struct.pack('<5d',*floor))
    (out/'labels.json').write_text(json.dumps(labels,indent=2)+'\n')


def compare(out, old, new=None):
    labels = json.loads((out/'labels.json').read_text())
    row = struct.Struct('<i22d')
    old_bytes = old.read_bytes()
    old = list(row.iter_unpack(old_bytes))
    assert len(old) == len(labels)
    assert [x[0] for x in old] == [x['expected_status'] for x in labels], list(zip(labels,[x[0] for x in old]))
    if new is None:
        print(f'{len(labels)} frozen CUDA Fortran repair cases meet their specified status')
        return
    new_bytes = new.read_bytes()
    new = list(row.iter_unpack(new_bytes))
    assert len(new) == len(old)
    for label, ref, actual in zip(labels,old,new):
        assert actual[0] == ref[0], (label,ref[0],actual[0])
        expected, got = ref[1:], actual[1:]
        if label['case'].startswith('trace_'):
            trace = 2.**int(label['case'][6:])
            assert got[9] == expected[9] == trace, (label,got[9],expected[9])
        if label['case'] == 'zero_support':
            assert got[9] == expected[9] == 0.
    assert old_bytes == new_bytes, 'repair output differs from frozen CUDA Fortran at the word level'
    summary = dict(rows=len(labels),state_carry_status_bitwise=True,traces_exact=True)
    (out/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--properties',type=Path)
    parser.add_argument('--old',type=Path)
    parser.add_argument('--new',type=Path)
    args = parser.parse_args()
    if args.properties is not None:
        prepare(args.properties,args.directory)
    else:
        compare(args.directory,args.old,args.new)
