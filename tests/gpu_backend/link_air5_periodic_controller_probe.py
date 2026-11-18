"""Link a test main against the actual root-CMake CUDA Fortran solver objects."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    work, out = args.build.resolve()/'src', args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    target = work/'CMakeFiles/astr.dir'
    for source in ('src_gpu/halo_exchange_gpu.cuf', 'src_gpu/chemistry_compensation_gpu.cuf',
                   'tests/gpu_backend/air5_periodic_controller_probe.cuf'):
        shutil.copy2(root/source, out/Path(source).name)
    flags = dict(line.split(' = ', 1) for line in (target/'flags.make').read_text().splitlines() if ' = ' in line)
    command = []
    skip = False
    for token in shlex.split((target/'link.txt').read_text()):
        if skip:
            skip = False
        elif token == '-o':
            skip = True
        elif not token.endswith('/astr.F90.o') and not token.startswith('-Wl,--dependency-file='):
            command.append(token)
    for name in ('Fortran_INCLUDES', 'Fortran_DEFINES'):
        command += shlex.split(flags[name])
    command += ['-I', str(work), str(out/'air5_periodic_controller_probe.cuf'), '-o', str(out/'probe')]
    (out/'link_command.json').write_text(json.dumps(command, indent=2)+'\n')
    with (out/'build.log').open('w') as log:
        subprocess.run(command, cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
