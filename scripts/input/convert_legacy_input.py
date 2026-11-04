#!/usr/bin/env python3
"""One-time positional ASTR input/controller to named input conversion.

No solver fallback is installed. Numeric real literals keep their original text.
Auxiliary mesh, profile, HDF5 and chemistry resources are never rewritten.
"""
import argparse
import math
from pathlib import Path
import shlex


class RealLiteral(str):
    """A validated Fortran real token, not a character namelist value."""


MODEL_ENVIRONMENT = {
    'ASTR_PROFILE_INFLOW_MODE': ('boundary_options', 'profile_inflow_mode', str),
    'ASTR_WALL_BLOWING_MODE': ('boundary_options', 'wall_blowing_mode', str),
    'ASTR_NSCBC_FARFIELD_MODE': ('boundary_options', 'nscbc_farfield_mode', str),
    'ASTR_HIGH_ORDER_WALL_MODE': ('boundary_options', 'high_order_wall_mode', str),
    'ASTR_CONSERVATIVE_BOUNDARY_FILE': ('boundary_options', 'conservative_boundary_file', str),
    'ASTR_NSCBC_FARFIELD_RHO': ('boundary_options', 'nscbc_farfield_rho', RealLiteral),
    'ASTR_NSCBC_FARFIELD_U': ('boundary_options', 'nscbc_farfield_u', RealLiteral),
    'ASTR_NSCBC_FARFIELD_V': ('boundary_options', 'nscbc_farfield_v', RealLiteral),
    'ASTR_NSCBC_FARFIELD_W': ('boundary_options', 'nscbc_farfield_w', RealLiteral),
    'ASTR_NSCBC_FARFIELD_T': ('boundary_options', 'nscbc_farfield_t', RealLiteral),
    'ASTR_NSCBC_FARFIELD_SHOCK_X': ('boundary_options', 'nscbc_farfield_shock_x', RealLiteral),
    'ASTR_NSCBC_FARFIELD_SHOCK_ANGLE_DEG': ('boundary_options', 'nscbc_farfield_shock_angle_deg', RealLiteral),
    'ASTR_PERFECT_GAS_PRANDTL': ('transport_options', 'perfect_gas_prandtl', RealLiteral),
    'ASTR_SUTHERLAND_TEMPERATURE_K': ('transport_options', 'sutherland_temperature_k', RealLiteral),
    'ASTR_CHANNEL_FORCE_MODE': ('transport_options', 'channel_force_mode', str),
    'ASTR_CHANNEL_FORCE_FIXED': ('transport_options', 'channel_force_fixed', RealLiteral),
    'ASTR_SHUOSHER_SHOCK_X': ('initial_conditions', 'shuosher_shock_x', RealLiteral),
    'ASTR_GPU_PRECISION_MODE': ('precision_options', 'gpu_precision_mode', str),
    'ASTR_GPU_MIXED_CANDIDATE': ('precision_options', 'gpu_mixed_candidate', str),
    'ASTR_GPU_MP_POSITIVITY': ('precision_options', 'gpu_mp_positivity', str),
    'ASTR_AIR5_SOURCE_MODE': ('chemistry', 'source_mode', str),
    'ASTR_AIR5_CHEMISTRY_TOLERANCE': ('chemistry', 'chemistry_tolerance', str),
    'ASTR_AIR5_FILTER_MODE': ('chemistry', 'filter_mode', str),
    'ASTR_AIR5_CONVECTION_LIMITER': ('chemistry', 'convection_limiter', str),
    'ASTR_AIR5_DIFFUSION_LIMITER': ('chemistry', 'diffusion_limiter', str),
    'ASTR_AIR5_TOP_MODE': ('chemistry', 'top_mode', str),
    'ASTR_AIR5_TOP_TAU': ('chemistry', 'top_tau', RealLiteral),
    'ASTR_AIR5_COMPENSATION': ('chemistry', 'compensation', str),
    'ASTR_AIR5_COMPENSATION_RESTART': ('chemistry', 'compensation_restart', str),
    'ASTR_AIR5_STEP_HISTORY': ('chemistry', 'step_history', str),
    'ASTR_AIR5_LOW_ACTIVITY': ('chemistry', 'low_activity_threshold', RealLiteral),
    'ASTR_AIR5_SOURCE_EVALUATION': ('chemistry', 'source_evaluation', str),
}


def values(line):
    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace = ' ,\t'
    lexer.whitespace_split = True
    lexer.commenters = '!'
    result = []
    for token in lexer:
        count, star, value = token.partition('*')
        if star and count.isdecimal():
            if not value:
                raise ValueError('null repeated values cannot be converted')
            result.extend([value]*int(count))
        else:
            result.append(token)
    return result


def data_lines(text):
    return [line.strip() for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith(('#', '!'))]


def logical(value):
    token = value.strip('.').lower()
    if token not in ('t', 'true', 'f', 'false'):
        raise ValueError(f'invalid logical token: {value}')
    return token in ('t', 'true')


def real(value):
    number = float(value.lower().replace('d', 'e'))
    if not math.isfinite(number):
        raise ValueError(f'nonfinite real input: {value}')
    return RealLiteral(value)


def literal_environment(text):
    """Read literal assignment records, never execute shell commands or expand $VAR."""
    result = {}
    for line in text.splitlines():
        entries = shlex.split(line, comments=True)
        if not entries:
            continue
        if entries[0] == 'export':
            entries = entries[1:]
        if any(entry in ('if', 'then', 'else', 'elif', 'fi', 'for', 'do', 'done', 'case', 'esac') for entry in entries):
            raise ValueError('environment conversion requires unconditional literal assignments')
        for entry in entries:
            name, equal, value = entry.partition('=')
            if equal and name in MODEL_ENVIRONMENT:
                if '$' in value or '`' in value or any(c in value for c in ';|&'):
                    raise ValueError(f'resolve shell expressions manually for {name}')
                result[name] = value
    return result


def add_model_environment(groups, environment):
    for name, value in environment.items():
        if name not in MODEL_ENVIRONMENT:
            continue  # Hardware, communication and diagnostic settings are not case inputs.
        group, field, kind = MODEL_ENVIRONMENT[name]
        if name == 'ASTR_AIR5_LOW_ACTIVITY' and value == 'off':
            value = '0'
        if kind is RealLiteral:
            value = real(value)
        groups.setdefault(group, {})[field] = value


def convert_case(text, comb=False, model_environment=None):
    lines = iter(data_lines(text))
    groups = {name: {} for name in ('case_options', 'mesh', 'physics', 'numerics',
                                   'boundary', 'initialization', 'sponge')}
    case, mesh, physics, numerics, boundary, initial, sponge = groups.values()
    shared = {}

    def take(count=None):
        try:
            row = values(next(lines))
        except StopIteration as error:
            raise ValueError('incomplete positional input') from error
        if count is not None and len(row) != count:
            raise ValueError(f'expected {count} values, got {row}')
        return row

    def shared_value(name, value):
        comparable = float(str(value).lower().replace('d', 'e')) if isinstance(value, RealLiteral) else value
        if name in shared and shared[name] != comparable:
            raise ValueError(f'conflicting shared boundary parameter {name}; review manually')
        shared[name] = comparable
        boundary[name] = value

    case['flowtype'] = take(1)[0]
    mesh.update(zip(('ia', 'ja', 'ka'), map(int, take(3)), strict=True))
    mesh.update(zip(('lihomo', 'ljhomo', 'lkhomo'), map(logical, take(3)), strict=True))
    flags = take()
    if len(flags) not in (7, 8, 9):
        raise ValueError('runtime flags require seven, eight or nine values')
    flags = list(map(logical, flags)) + [False]*(9-len(flags))
    physics.update(zip(('nondimen', 'diffterm'), flags[:2], strict=True))
    numerics['lfilter'] = flags[2]
    mesh.update(zip(('lreadgrid', 'lfftk'), flags[3:5], strict=True))
    initial['limmbou'] = flags[5]
    case['ltimrpt'] = flags[6]
    physics['lcomb'] = flags[7]
    case['use_gpu'] = flags[8]
    initial['lrestart'] = logical(take(1)[0])
    alpha, cutoff = take(2)
    numerics.update(alfa_filter=real(alpha), kcutoff=int(cutoff))
    references = ('ref_tem', 'reynolds', 'mach') if physics['nondimen'] else (
        'ref_tem', 'ref_vel', 'ref_len', 'ref_den')
    physics.update(zip(references, map(real, take(len(references))), strict=True))
    schemes = take()
    if len(schemes) not in (3, 4) or (comb and len(schemes) != 4):
        raise ValueError('expected conschm, difschm, rkscheme and optional COMB odetype')
    numerics.update(zip(('conschm', 'difschm', 'rkscheme'), schemes[:3], strict=True))
    if comb:
        physics['odetype'] = schemes[3]
    # A fourth scheme token was not consumed by non-COMB readinput.
    recon, characteristic, bfac, sensor = take(4)
    numerics.update(recon_schem=int(recon), lchardecomp=logical(characteristic),
                    bfacmpld=real(bfac), shkcrt=real(sensor))
    species = take()
    if not comb:
        physics['num_species'] = int(species[0])
        if physics['num_species'] < 0:
            raise ValueError('negative num_species')
        while len(species) < physics['num_species']+1:
            species.extend(take())
        if len(species) != physics['num_species']+1:
            raise ValueError('Schmidt-number count differs from num_species')
        if physics['num_species']:
            groups['species_transport'] = {'schmidt': list(map(real, species[1:]))}
    physics['turbmode'], case['iomode'] = take(2)
    boundary['bctype'] = []
    boundary['twall'] = [RealLiteral('0')]*6
    for face in range(6):
        entry = take()
        code = int(entry[0])
        boundary['bctype'].append(code)
        expected = 1
        if code == 41:
            expected = 2
            if len(entry) == expected:
                boundary['twall'][face] = real(entry[1])
        elif code in (411, 421):
            expected = 3 if code == 411 else 2
            if len(entry) == expected:
                shared_value('xslip', real(entry[1]))
                if code == 411:
                    boundary['twall'][face] = real(entry[2])
        elif code == 11:
            expected = 2
            if len(entry) == expected:
                shared_value('turbinf', entry[1])
        elif code in (21, 22):
            if len(entry) not in (1, 2):
                raise ValueError(f'invalid outlet entry: {entry}')
            shared_value('pout', real(entry[1] if len(entry) == 2 else '1'))
            expected = len(entry)
        elif code == 51 and case['flowtype'] == 'swbli':
            expected = 3
            if len(entry) == expected:
                shared_value('xrhjump', real(entry[1]))
                shared_value('angshk', real(entry[2]))
        if len(entry) != expected:
            raise ValueError(f'boundary face {face+1}, bctype={code} needs {expected} values: {entry}')
    initial['ninit'] = int(take(1)[0])
    widths = take()
    if len(widths) not in (6, 7):
        raise ValueError('sponge needs six widths and optional spg_def')
    sponge.update(zip(('spg_i0', 'spg_im', 'spg_j0', 'spg_jm', 'spg_k0', 'spg_km'),
                      map(int, widths[:6]), strict=True))
    sponge['spg_def'] = widths[6] if len(widths) == 7 else 'layer'
    try:
        mesh['gridfile'] = next(lines).strip().strip('"\'')
    except StopIteration as error:
        raise ValueError('missing gridfile record') from error
    if initial['limmbou']:
        initial['ibmode'], initial['solidfile'] = take(2)
    if comb:
        try:
            physics['chemfile'] = next(lines).strip().strip('"\'')
        except StopIteration as error:
            raise ValueError('missing COMB chemfile') from error
    remaining = list(lines)
    if remaining:
        raise ValueError(f'unrecognized trailing input, review manually: {remaining}')
    add_model_environment(groups, model_environment or {})
    return groups


def convert_controller(text):
    rows = [values(line) for line in data_lines(text)]
    if len(rows) != 3 or len(rows[0]) != 4 or len(rows[1]) != 6 or len(rows[2]) != 1:
        raise ValueError('controller needs four flags, six counters and deltat')
    _, _, average, repair = map(logical, rows[0])
    maximum, reload, _, _, listing, sample = map(int, rows[1])
    dt = real(rows[2][0])
    if maximum < 0 or reload <= 0 or listing < 0 or (average and sample <= 0) or float(dt.lower().replace('d', 'e')) <= 0:
        raise ValueError('invalid controller stop, cadence or deltat')
    return {
        'run_control': {'max_steps': maximum+1, 'reload_steps': reload},
        'timestep': {'mode': 'fixed', 'fixed_dt': dt},
        'logging': {'list_steps': listing},
        'repair': {'crashfix': repair},
        'statistics': {'enabled': average, 'sample_clock': 'steps',
                       'sample_steps': sample if average else 1, 'write_steps': 0},
    }


def namelist(groups):
    def token(value):
        if isinstance(value, RealLiteral):
            return str(value)
        if isinstance(value, bool):
            return '.true.' if value else '.false.'
        if isinstance(value, str):
            return "'"+value.replace("'", "''")+"'"
        return str(value)
    result = []
    for group, fields in groups.items():
        result.append('&'+group)
        for key, value in fields.items():
            entries = value if isinstance(value, list) else [value]
            result.append('  '+key+' = '+', '.join(map(token, entries)))
        result.append('/\n')
    return '\n'.join(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--controller', type=Path)
    parser.add_argument('--run-output', type=Path)
    parser.add_argument('--comb', action='store_true', help='source was read by a COMB build')
    parser.add_argument('--environment', type=Path, help='literal old NAME=value/export file; never executed')
    args = parser.parse_args()
    environment = literal_environment(args.environment.read_text()) if args.environment else {}
    converted = namelist(convert_case(args.input.read_text(), comb=args.comb, model_environment=environment))
    run = namelist(convert_controller(args.controller.read_text())) if args.controller else None
    with args.output.open('x') as target:
        target.write(converted)
    print(args.output)
    if run is not None:
        path = args.run_output or args.output.with_name('input.run')
        with path.open('x') as target:
            target.write(run)
        print(path)
        print('Fixed dt retained; max_steps=old maxstep+1. Field output stays in input.output.')


if __name__ == '__main__':
    main()
