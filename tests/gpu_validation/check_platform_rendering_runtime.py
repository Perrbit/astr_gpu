"""Verify that platform executables load the explicitly selected rendering DSO."""
import argparse
from pathlib import Path
import re
import subprocess


SONAME = 'libvtkRenderingOpenGL2-pv6.1.so.1'


def check_library(binary, expected):
    expected = Path(expected).resolve(strict=True)
    result = subprocess.run(['ldd', '-r', str(binary)], check=True,
                            text=True, capture_output=True)
    listing = result.stdout + result.stderr
    if 'not found' in listing or 'undefined symbol' in listing:
        raise ValueError(f'Unresolved runtime dependency for {binary}:\n{listing}')
    match = re.search(r'^\s*' + re.escape(SONAME) + r'\s+=>\s+(\S+)', listing, re.M)
    if not match:
        raise ValueError(f'Missing rendering dependency for {binary}')
    actual = Path(match[1]).resolve(strict=True)
    if actual != expected:
        raise ValueError(f'Wrong rendering library for {binary}: loaded {actual}, expected {expected}')
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected', type=Path, required=True)
    parser.add_argument('binaries', type=Path, nargs='+')
    args = parser.parse_args()
    for binary in args.binaries:
        library = check_library(binary, args.expected)
        print(f'RENDERING_RUNTIME=PASS binary={binary} library={library}', flush=True)


if __name__ == '__main__':
    main()
