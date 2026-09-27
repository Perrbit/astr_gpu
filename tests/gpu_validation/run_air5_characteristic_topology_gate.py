#!/usr/bin/env python3
"""Current-policy coupled AIR5 topology gate before physical SBLI admission."""
import argparse
import json
from pathlib import Path

from run_air5_characteristic_backend_gate import execute, compare, TOLERANCE
from check_reconciled_solution_halos import check
from run_air5_sbli_preflight import ROOT


def run(output, executable):
    output.mkdir(parents=True, exist_ok=False)
    contract = dict(source_mode='coupled', viscous=True, temperature=3000., tv=1500.,
                    dt=1.25e-9, convection_limiter='symmetric_species',
                    diffusion_limiter='layered', tolerance=TOLERANCE,
                    scope='three-update implementation equivalence, not physical admission')
    (output/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    results = []
    reference = None
    cases = [(False, '1,1,1'), *[(True, t) for t in
             ('1,1,1', '2,1,1', '1,2,1', '1,1,2', '2,2,2')]]
    for gpu, topology in cases:
        name = ('gpu_' if gpu else 'cpu_')+topology.replace(',', 'x')
        case = output/name
        data = execute(case, executable, gpu, topology, source_mode='coupled',
                       viscous=True, temperature=3000., tv=1500., dt=1.25e-9,
                       convection_limiter='symmetric_species', diffusion_limiter='layered')
        if reference is None:
            reference = data
        fields = compare(reference, data)
        halos = [check(case, step, stage) for step in (0, 2) for stage in (1, 2, 3)]
        if not all(row['passed'] for row in halos):
            raise ValueError(f'{name}: halo/donor mismatch')
        results.append(dict(case=name, fields=fields, halos=halos,
                            maximum=max(row['max_scaled'] for row in fields.values())))
        (output/'result.json').write_text(json.dumps(dict(passed=False, completed=results), indent=2)+'\n')
        print(name, results[-1]['maximum'], flush=True)
    (output/'result.json').write_text(json.dumps(dict(passed=True, completed=results), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--executable', type=Path, default=ROOT/'build_gpu_probe/bin/astr')
    args = parser.parse_args()
    run(args.output.resolve(), args.executable.resolve())
