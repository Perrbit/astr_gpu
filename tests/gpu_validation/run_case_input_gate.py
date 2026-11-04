"""Exercise the actual named reader/broadcast, without constructing a CFD state."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from test_case_input_conversion import dimensional_text, tgv_text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/input'))
from convert_legacy_input import convert_case, namelist, values, logical, RealLiteral, MODEL_ENVIRONMENT


def parse_probe(text):
    result = {}
    for match in re.finditer(r'(?ms)&(\w+)\s+(.*?)^\s*/', text):
        fields = {}
        positions = list(re.finditer(r'\b(\w+)\s*=', match[2]))
        for index, position in enumerate(positions):
            end = positions[index+1].start() if index+1 < len(positions) else len(match[2])
            fields[position[1].lower()] = values(match[2][position.end():end].replace('\n', ' '))
        result[match[1].lower()] = fields
    return result


def check_values(expected, actual):
    receipt = actual['model_option_receipt']
    model = dict(zip(map(str.strip,receipt['option_names']), map(str.strip,receipt['option_values']), strict=True))
    for group, fields in expected.items():
        for field, value in fields.items():
            wanted = value if isinstance(value, list) else [value]
            found = actual[group][field] if group in actual else [model[field]]
            if len(wanted) != len(found):
                raise AssertionError(f'{group}.{field} has wrong array length')
            for x, y in zip(wanted, found, strict=True):
                if isinstance(x, bool):
                    equal = x == logical(y)
                elif isinstance(x, (int, RealLiteral)):
                    equal = float(str(x).lower().replace('d', 'e')) == float(y.lower().replace('d', 'e'))
                else:
                    equal = x == y.strip()
                if not equal:
                    raise AssertionError(f'{group}.{field}: expected {x!r}, read {y!r}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.executable = args.executable.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_coll='^cuda,hcoll,ucc',
               OMPI_MCA_opal_cuda_support='0')
    report = {}

    def run(label, text, expected=None, reason=None, retired=None):
        directory = (args.output/label).resolve()
        directory.mkdir()
        (directory/'input.case').write_bytes(text.encode())
        local_env = env.copy()
        local_env.update(retired or {})
        with (directory/'run.log').open('w') as log:
            completed = subprocess.run([str(args.mpiexec), '-n', '2', str(args.executable),
                                        'test', 'case', 'input.case'], cwd=directory, env=local_env,
                                       stdout=log, stderr=subprocess.STDOUT)
        output = (directory/'run.log').read_text()
        if reason:
            if completed.returncode == 0 or 'ASTR_CASE rejected' not in output or reason not in output:
                raise AssertionError(f'{label}: expected rejection {reason}, see {directory}')
            report[label] = {'rejected': reason}
            return
        if completed.returncode:
            raise AssertionError(f'{label}: reader failed, see {directory}')
        a = (directory/'case_input.rank000000.nml').read_text()
        b = (directory/'case_input.rank000001.nml').read_text()
        if a != b:
            raise AssertionError(f'{label}: broadcast lost named fields')
        check_values(expected, parse_probe(a))
        report[label] = {'named_fields_match': True, 'rank_reports_equal': True}
        if retired:
            if output.count('ASTR_CASE ignored retired environment') != len(retired):
                raise AssertionError('retired environment warnings must occur only once on the root')
            report[label]['retired_ignored_once'] = len(retired)

    tgv = convert_case(tgv_text())
    dimensional = convert_case(dimensional_text())
    run('tgv', namelist(tgv), tgv)
    # Group and field order are independent; CRLF and uppercase names are accepted.
    reordered = {k: dict(reversed(list(v.items()))) for k, v in reversed(list(dimensional.items()))}
    run('dimensional_reordered_crlf', namelist(reordered).replace('\n', '\r\n'), dimensional)
    settings = {name: '1.25d0' if kind is RealLiteral else 'off'
                for name, (_, _, kind) in MODEL_ENVIRONMENT.items()}
    settings.update(ASTR_PROFILE_INFLOW_MODE='mach_pressure', ASTR_WALL_BLOWING_MODE='legacy_random',
                    ASTR_NSCBC_FARFIELD_MODE='incoming_only', ASTR_CONSERVATIVE_BOUNDARY_FILE='datin/boundary state.nml',
                    ASTR_CHANNEL_FORCE_MODE='fixed', ASTR_CHANNEL_FORCE_FIXED='1.d-4',
                    ASTR_GPU_PRECISION_MODE='fp64', ASTR_GPU_MIXED_CANDIDATE='flux', ASTR_GPU_MP_POSITIVITY='flux',
                    ASTR_AIR5_SOURCE_MODE='coupled', ASTR_AIR5_CHEMISTRY_TOLERANCE='tight',
                    ASTR_AIR5_FILTER_MODE='conservative-flux', ASTR_AIR5_CONVECTION_LIMITER='symmetric_species',
                    ASTR_AIR5_DIFFUSION_LIMITER='layered', ASTR_AIR5_TOP_MODE='characteristic',
                    ASTR_AIR5_TOP_TAU='1.d-6', ASTR_AIR5_COMPENSATION='on',
                    ASTR_AIR5_COMPENSATION_RESTART='restore', ASTR_AIR5_STEP_HISTORY='previous',
                    ASTR_AIR5_LOW_ACTIVITY='1.d-10', ASTR_AIR5_SOURCE_EVALUATION='source-only')
    models = convert_case(tgv_text(), model_environment=settings)
    run('model_options', namelist(models), models,
        retired={name: 'invalid-retired-value' for name in MODEL_ENVIRONMENT})
    run('old_positional_rejected', tgv_text(), reason='required groups')
    run('missing_mach', namelist(tgv).replace('  mach = 0.1d0\n', ''), reason='reynolds and mach')
    run('unknown_group', namelist(tgv)+'&typo\n/\n', reason='unknown group')
    run('unknown_field', namelist(tgv).replace('&numerics', '&numerics\n does_not_exist=1'), reason='numerics:')
    run('duplicate_group', namelist(tgv)+'&mesh\n/\n', reason='duplicate group')
    no_species = {k: v for k, v in dimensional.items() if k != 'species_transport'}
    run('missing_schmidt', namelist(no_species), reason='Schmidt numbers')
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
