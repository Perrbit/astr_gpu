"""Exact-rational audit of a stopped native T3-P1 pure-filter snapshot."""
import argparse
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

from test_air5_filter_multidim_probe import raw_faces
from air5_radau_reference import Air5RadauReference


def read_failure(path):
    with path.open('rb') as stream:
        header = np.fromfile(stream, dtype=np.int32, count=9)
        shape = tuple(int(x) for x in header[:3])
        fields = [np.moveaxis(np.fromfile(stream, dtype=np.float64, count=11*np.prod(shape))
                             .reshape((11,)+shape, order='F'), 0, -1) for _ in range(6)]
        tr, v = np.fromfile(stream, dtype=np.float64, count=10).reshape(2, 5)
        assert stream.read() == b''
    return header, fields, tr, v


def vibration_audit(q, carry, floor):
    state = q-carry
    sequential = state[..., 10].copy()
    for s in range(5):
        sequential -= state[..., s+5]*floor[s]
    # Select all signs that can be affected by the five rounded products/subtractions.
    scale = abs(state[..., 10])+np.sum(abs(state[..., 5:10]*floor), axis=-1)
    operations = 16*np.finfo(float).eps/2
    bound = operations/(1-operations)*scale
    near = np.argwhere(abs(sequential) <= bound)
    signs = {'exact_negative': 0, 'exact_zero': 0, 'exact_positive': 0,
             'false_rejections': 0, 'false_acceptances': 0}
    examples = {}
    fractions = [Fraction(float(value)) for value in floor]
    exact_min = None
    for index in near:
        location = tuple(index)
        high, low = q[location], carry[location]
        represented = [Fraction(float(h))-Fraction(float(c)) for h, c in zip(high, low)]
        exact = represented[10]-sum(represented[s+5]*fractions[s] for s in range(5))
        sign = 'exact_negative' if exact < 0 else 'exact_positive' if exact > 0 else 'exact_zero'
        signs[sign] += 1
        exact_min = exact if exact_min is None else min(exact_min, exact)
        false_reject = sequential[location] < 0 <= exact
        false_accept = exact < 0 <= sequential[location]
        if false_reject:
            signs['false_rejections'] += 1
        if false_accept:
            signs['false_acceptances'] += 1
        if sign not in examples or false_reject and 'false_rejection' not in examples:
            key = 'false_rejection' if false_reject else sign
            examples[key] = {'index': list(map(int, index)),
                             'sequential_margin': float(sequential[location]),
                             'exact_margin': float(exact),
                             'exact_fraction': str(exact)}
    return {'near_bound_nodes': len(near), 'sequential_negative': int(np.count_nonzero(sequential < 0)),
            'exact_min_near_bound': float(exact_min) if exact_min is not None else None,
            **signs, 'examples': examples}


def pressure_residual(state, formation):
    return (state[4]-state[10]-sum(state[s+5]*formation[s] for s in range(5))
            -sum(state[s]**2 for s in (1, 2, 3))/(2*state[0])-2500)


def pressure_box_lower_bound(center, radius, formation):
    rho = center[0]-radius[0]
    assert rho > 0, 'diagnostic pressure box crosses zero density'
    return (center[4]-radius[4]-center[10]-radius[10]
            -sum(center[s+5]*formation[s]+radius[s+5]*abs(formation[s]) for s in range(5))
            -sum((abs(center[s])+radius[s])**2 for s in (1, 2, 3))/(2*rho)-2500)


def pressure_commit_audit(fields, axis, node, carry_mode):
    model = Air5RadauReference(Path(__file__).resolve().parents[2]/'chemMech/air5_kimjo12.json')
    formation = list(map(Fraction, map(float, model.formation_energy)))
    q, carry, source, source_c, _, flux = fields
    previous = list(node); previous[axis] = (previous[axis]-1) % source.shape[axis]
    previous = tuple(previous)
    rational = lambda values: list(map(Fraction, map(float, values)))
    h, c, right, left = map(rational, (source[node], source_c[node], flux[node], flux[previous]))
    base = [a-b for a, b in zip(h, c)]
    target = [a+b-c for a, b, c in zip(base, right, left)]
    represented = [a-b for a, b in zip(rational(q[node]), rational(carry[node]))]
    collapsed = rational(q[node]-carry[node])
    u = Fraction(1, 2**53)
    gamma = lambda n: n*u/(1-n*u)
    if carry_mode == 'off':
        assert not np.any(source_c) and not np.any(carry)
        represented_gamma = collapsed_gamma = gamma(2)
        pair_gamma = u
    else:
        represented_gamma, collapsed_gamma, pair_gamma = gamma(11), gamma(13), gamma(3)
    scale = [abs(a)+abs(b)+abs(r)+abs(l) for a, b, r, l in zip(h, c, right, left)]
    error = [a-b for a, b in zip(represented, target)]
    collapsed_error = [a-b for a, b in zip(collapsed, target)]
    assert all(abs(e) <= represented_gamma*s for e, s in zip(error, scale))
    assert all(abs(e) <= collapsed_gamma*s for e, s in zip(collapsed_error, scale))
    construction, boxes = [], []
    rounded_base = source[node]-source_c[node]
    for face, sign in ((right, 1), (left, -1)):
        exact = [a+2*sign*f for a, f in zip(base, face)]
        rounded = rational(rounded_base+2*sign*np.array(list(map(float, face))))
        face_scale = [abs(a)+abs(b)+2*abs(f) for a, b, f in zip(h, c, face)]
        assert all(abs(a-b) <= pair_gamma*s for a, b, s in zip(rounded, exact, face_scale))
        radius = [(collapsed_gamma+pair_gamma)*s for s in face_scale]
        construction.append({'exact_margin': float(pressure_residual(exact, formation)),
                             'rounded_margin': float(pressure_residual(rounded, formation)),
                             'max_component_error': float(max(abs(a-b) for a, b in zip(rounded, exact)))})
        boxes.append(float(pressure_box_lower_bound(rounded, radius, formation)))
    radius = [collapsed_gamma*s for s in scale]
    lower = pressure_box_lower_bound(target, radius, formation)
    final_residual = pressure_residual(collapsed, formation)
    assert lower <= final_residual
    return {'node': list(node), 'carry_mode': carry_mode,
            'base_margin': float(pressure_residual(base, formation)),
            'exact_target_margin': float(pressure_residual(target, formation)),
            'represented_output_margin': float(pressure_residual(represented, formation)),
            'collapsed_output_margin': float(final_residual),
            'collapsed_output_margin_fraction': str(final_residual),
            'represented_commit_max_component_error': float(max(map(abs, error))),
            'collapsed_commit_max_component_error': float(max(map(abs, collapsed_error))),
            'commit_error_bounds_hold': True, 'pair_construction': construction,
            'proposed_pair_box_lower_margins': boxes,
            'commit_box_lower_margin': float(lower),
            'scope': 'exact-rational diagnosis/design only; no limiter or CFD execution'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pressure-node', type=int, nargs=3)
    parser.add_argument('--carry', choices=('off', 'on'), default='off')
    args = parser.parse_args()
    header, fields, tr, v = read_failure(args.snapshot)
    q, carry, source, source_c, high, flux = fields
    axis = int(header[8])-1
    assert header[7] == 1, 'independent periodic roll audit requires NP=1'
    raw_difference = float(np.max(abs(high-raw_faces(source, source_c, axis))))
    update_difference = float(np.max(abs(q-((source+flux)-np.roll(flux, 1, axis=axis)))))
    result = {'snapshot': str(args.snapshot), 'axis': axis+1,
              'raw_face_max_abs_difference': raw_difference,
              'ordinary_commit_max_abs_difference': update_difference,
              'source': vibration_audit(source, source_c, v),
              'final': vibration_audit(q, carry, v)}
    if args.pressure_node is not None:
        result['pressure_commit'] = pressure_commit_audit(fields, axis, tuple(args.pressure_node), args.carry)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
