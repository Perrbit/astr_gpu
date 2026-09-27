#!/usr/bin/env python3
"""Check complete-step AIR5 CFL against saved states on uniform Cartesian grids."""
import argparse
import json
from pathlib import Path
import re

import h5py
import numpy as np

from air5_radau_reference import Air5RadauReference
from check_air5_c5_hbl import _assemble_global, _load_parallel_layout
from check_air5_c5_normal_shock import primitive_metrics
from check_air5_mach4_error_replay import cartesian_jacobian


def check(case):
    contract = json.loads((case.parent / 'contract.json').read_text())
    cartesian_jacobian(case)  # Reject nonuniform/skewed grids before using spacings.
    with h5py.File(case / 'datin/grid.h5') as grid:
        spacing = np.array([np.diff(grid[n][()], axis=a).flat[0]
                            for n, a in [('x', 2), ('y', 1), ('z', 0)]])
    thermo = Air5RadauReference(Path(__file__).resolve().parents[2] /
                               'chemMech/air5_kimjo12.json')
    layouts = _load_parallel_layout(case / 'datin/parallel.info')
    log = (case / 'run.log').read_text()
    if 'The job is done!' not in log:
        raise ValueError('run has not completed')
    records = {}
    for line in log.splitlines():
        if line.startswith('ASTR_CFL complete_step='):
            match = re.fullmatch(r'ASTR_CFL complete_step=(\d+) state_time=\s*(\S+) dt=\s*(\S+)', line)
            step, time, dt = match.groups()
            record = records[int(step)] = dict(time=float(time), dt=float(dt), locations={}, planes={})
        elif line.startswith('ASTR_CFL directional/local_sum/upper_bound='):
            record['values'] = np.array(list(map(float, line.split('=')[1].split())))
        elif line.startswith('ASTR_CFL maximum='):
            match = re.fullmatch(r'ASTR_CFL maximum=(\d+) global_ijk=\s*(\d+)\s+(\d+)\s+(\d+)', line)
            axis, *ijk = map(int, match.groups())
            record['locations'][axis-1] = tuple(ijk)
        elif line.startswith('ASTR_CFL_Y step='):
            match = re.fullmatch(r'ASTR_CFL_Y step=(\d+) j=(\d+)\s+(.*)', line)
            step, j, values = match.groups()
            records[int(step)]['planes'][int(j)] = list(map(float, values.split()))
    if not records:
        raise ValueError('missing CFL records')
    expected_steps = {step for step in range(contract['start_step'],
                      contract['start_step']+contract['updates'])
                      if step % contract['checkpoint_interval'] == 0}
    if set(records) != expected_steps:
        raise ValueError('incomplete CFL sampling')
    results = []
    for step, record in records.items():
        expected_time = contract['start_time'] + (step-contract['start_step']+1)*contract['dt']
        if record['dt'] != contract['dt'] or abs(record['time']-expected_time) > 1e-18:
            raise ValueError('CFL time phase mismatch')
        files = list((case / 'validation').glob(f'air5.post_chemistry.step{step:08d}.rk02.rank*.bin'))
        if files:
            q = _assemble_global(case / 'validation/air5', 'post_chemistry', 2, layouts, step)
            source = 'post_chemistry'
        else:
            # A checkpoint at step+1 contains the preceding complete-step state.
            with h5py.File(case / 'outdat/flowfield.h5') as data:
                if int(data['nstep'][()].item()) != step+1:
                    raise ValueError(f'no matched complete-step state for {step}')
                q = np.stack([data[f'acq{i:02}'][()].transpose(2, 1, 0)
                              for i in range(1, 12)], axis=-1)
            source = 'next_checkpoint'
        rho, species, temperature, _, _ = primitive_metrics(q, thermo)
        rden, cvden = species @ thermo.gas_constant, species @ thermo.cv_tr
        sound = np.sqrt((1+rden/cvden)*rden*temperature/rho)
        rates = (abs(q[..., 1:4]/rho[..., None])+sound[..., None])/spacing
        rates = np.concatenate((rates, rates.sum(axis=-1, keepdims=True)), axis=-1)*record['dt']
        maximum = rates.max(axis=(0, 1, 2))
        expected = np.r_[maximum, maximum[:3].sum()]
        np.testing.assert_allclose(record['values'], expected, rtol=5e-13, atol=1e-14)
        if set(record['locations']) != set(range(4)):
            raise ValueError('missing maximum locations')
        for axis, ijk in record['locations'].items():
            np.testing.assert_allclose(rates[ijk+(axis,)], maximum[axis], rtol=5e-13, atol=1e-14)
        if contract['cfl_profile_y']:
            if set(record['planes']) != set(range(q.shape[1])):
                raise ValueError('incomplete y profile')
            actual = np.array([record['planes'][j] for j in range(q.shape[1])])
            np.testing.assert_allclose(actual, rates.max(axis=(0, 2)), rtol=5e-13, atol=1e-14)
        results.append(dict(step=step, source=source, maxima=expected.tolist(),
                            maximum_absolute_difference=float(abs(record['values']-expected).max())))
    return dict(passed=True, case=str(case), records=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(check(args.case), indent=2)+'\n'
    if args.output:
        args.output.write_text(result)
    print(result, end='')
