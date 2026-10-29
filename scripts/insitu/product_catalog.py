"""Publish complete runtime catalogs and explicitly re-enable one product."""
import argparse
import json
import os
from pathlib import Path
import tempfile

SCHEMA = 'astr-insitu-products-v1'
MAX_REQUEST = (1 << 63) - 1


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON object key: ' + key)
        result[key] = value
    return result


def _constant(value):
    raise ValueError('nonstandard JSON number: ' + value)


def _decode(line):
    return json.loads(line, object_pairs_hook=_object, parse_constant=_constant)


def inspect_catalog(path):
    """Check framing and report body syntax; ASTR owns product/type admission."""
    lines = Path(path).read_bytes().decode('utf-8').splitlines(keepends=True)
    header = False
    records, warnings = [], []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if not header:
            value = _decode(line)
            if value != {'catalog_schema': SCHEMA}:
                raise ValueError('unsupported or invalid catalog header')
            header = True
            continue
        try:
            value = _decode(line)
            if not isinstance(value, dict):
                raise ValueError('product record must be an object')
            records.append((index, value))
        except (ValueError, json.JSONDecodeError) as exc:
            warnings.append(f'line {index + 1}: {exc}')
    if not header:
        raise ValueError('missing catalog header; an empty catalog still needs its header')
    return lines, records, warnings


def publish(candidate, formal):
    """Rename a completed candidate once; body errors remain per-product errors."""
    candidate, formal = Path(candidate), Path(formal)
    if candidate.absolute() == formal.absolute():
        raise ValueError('candidate and formal file must be different paths')
    _, _, warnings = inspect_catalog(candidate)
    os.replace(candidate, formal)
    return warnings


def enable(formal, product_id):
    """Change only the requested record, then publish an internal candidate."""
    formal = Path(formal)
    lines, records, warnings = inspect_catalog(formal)
    matches = [(index, value) for index, value in records if value.get('id') == product_id]
    if len(matches) != 1:
        raise ValueError('enable needs exactly one identifiable record for ' + product_id)
    index, value = matches[0]
    number = value.get('enable_request')
    if type(number) is not int or not 1 <= number < MAX_REQUEST:
        raise ValueError('enable_request must be a positive int64 that can be incremented')
    value['enable_request'] = number + 1
    ending = '\r\n' if lines[index].endswith('\r\n') else '\n' if lines[index].endswith('\n') else ''
    lines[index] = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + ending
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=formal.parent, prefix=formal.name + '.candidate.', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(''.join(lines).encode('utf-8'))
        os.chmod(temporary, formal.stat().st_mode & 0o777)
        os.replace(temporary, formal)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return number + 1, warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    submit = commands.add_parser('publish', help='atomically replace the formal file with a candidate')
    submit.add_argument('candidate', type=Path)
    submit.add_argument('formal', type=Path)
    retry = commands.add_parser('enable', help='increase one enable_request and publish atomically')
    retry.add_argument('formal', type=Path)
    retry.add_argument('product_id')
    args = parser.parse_args()
    try:
        if args.command == 'publish':
            warnings = publish(args.candidate, args.formal)
        else:
            number, warnings = enable(args.formal, args.product_id)
            print(f'enable_request={number}')
    except (OSError, UnicodeError, ValueError) as exc:
        parser.exit(1, str(exc) + '\n')
    for warning in warnings:
        print('WARNING: ' + warning)
    print('Published. Runtime acceptance is reported by ASTR in products.status.json.')


if __name__ == '__main__':
    main()
