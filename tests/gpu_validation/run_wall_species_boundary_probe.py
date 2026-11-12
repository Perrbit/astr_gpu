"""Link a boundary-only driver against the actual root-built solver objects."""
import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--mpiexec', type=Path, required=True)
parser.add_argument('--backend', choices=['cpu','gpu'], required=True)
args = parser.parse_args()
OUT = args.output.resolve()
OUT.mkdir(parents=True, exist_ok=False)
BUILD = args.build.resolve()
link = shlex.split((BUILD / 'src/CMakeFiles/astr.dir/link.txt').read_text())
compiler = link[0]
source = Path(__file__).with_name('wall_species_boundary_probe.F90')
obj = OUT / 'probe.o'
exe = OUT / 'probe'
includes, = [line.split('=', 1)[1] for line in
             (BUILD / 'src/CMakeFiles/astr.dir/flags.make').read_text().splitlines()
             if line.startswith('Fortran_INCLUDES =')]
compile_command = [compiler, '-Mpreprocess', '-Mextend', '-traceback', '-g', '-Mnostack_arrays', '-O2',
                   '-I' + str(BUILD / 'src'), *shlex.split(includes), '-c', str(source), '-o', str(obj)]
if args.backend=='gpu':
    compile_command += ['-cuda','-gpu=lineinfo','-DTEST_CUDA']
subprocess.run(compile_command, cwd=OUT, check=True)
assert link.count('CMakeFiles/astr.dir/astr.F90.o') == 1
link[link.index('CMakeFiles/astr.dir/astr.F90.o')] = str(obj)
link[link.index('-o') + 1] = str(exe)
link = [value for value in link if not value.startswith('-Wl,--dependency-file=')]
subprocess.run(link, cwd=BUILD / 'src', check=True)
report = {'root_build': str(BUILD), 'backend': args.backend, 'compile': compile_command,
          'link': link, 'cases': [], 'status': 'running'}
env = dict(os.environ, OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_osc='pt2pt',
           OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_opal_cuda_support='0')
mpiexec = args.mpiexec.resolve()
for ranks in (1, 2):
    for wall in (41, 42):
        states=[(0,1),(0,3)] if args.backend=='gpu' else [(0,1),(0,3),(1,0),(1,1),(1,2)]
        for species, variant in states:
            name = f'np{ranks}_bc{wall}_species{species}_variant{variant}'
            command = [str(mpiexec), '--mca', 'coll_hcoll_enable', '0', '-np', str(ranks),
                       str(exe), str(wall), str(species), str(variant)]
            with (OUT / (name + '.log')).open('w') as stream:
                result = subprocess.run(command, cwd=OUT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'wall boundary probe exited {result.returncode}: {OUT / (name + ".log")}')
            log = (OUT / (name + '.log')).read_text()
            line, = [line for line in log.splitlines() if line.startswith('ERROR lowerY')]
            metrics = list(map(float, line.split('=')[1].split()))
            record = dict(name=name, ranks=ranks, wall=wall, species=species, variant=variant,
                          returncode=result.returncode, errors=metrics)
            report['cases'].append(record)
            print(json.dumps(record), flush=True)
            (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
report['status'] = 'passed'
(OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
