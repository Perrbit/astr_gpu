"""Frozen-input diagnostic storage/isolation probes; main MPI tests are separate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('binary', 'packets', 'mpiexec', 'sanitizer', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve();root.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve()
    shutil.copyfile(__file__, root/'runner.py');shutil.copyfile(binary, root/'tested_probe')
    env = os.environ | dict(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_opal_cuda_support='0')
    reports = []
    for layout, topology, tool in (('np1', (1,1,1), None), ('np2_x', (2,1,1), None),
            ('np2_y', (1,2,1), 'memcheck'), ('np2_z', (1,1,2), 'initcheck')):
        case = root/layout;case.mkdir()
        ranks = topology[0]*topology[1]*topology[2]
        command = [str(args.mpiexec.resolve()), '--bind-to', 'none', '-n', str(ranks)]
        if tool:
            command += [str(args.sanitizer.resolve()), '--tool', tool, '--error-exitcode', '99',
                '--log-file', str(case/'sanitizer.%p.log')]
            if tool == 'memcheck':command += ['--leak-check', 'full']
        command += [str(binary), str(args.packets.resolve()/layout), ','.join(map(str,topology))]
        with (case/'run.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, check=True)
        values = re.findall(r'CPP_MIXED_DERIVATIVE_PASS\s+(\d+)\s+(\d+)\s+([\d.Ee+\-]+)\s+(\d+)',
            (case/'run.log').read_text())
        if len(values) != ranks or any(int(n)!=12 or float(error)>1e-8 or int(count)<=0 for _,n,error,count in values):
            raise AssertionError('missing diagnostic conversion/isolation evidence')
        if tool:
            logs = list(case.glob('sanitizer.*.log'))
            if len(logs)!=ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in logs):
                raise AssertionError('diagnostic gradient memory-safety failure')
            if tool=='memcheck' and any('LEAK SUMMARY: 0 bytes leaked' not in p.read_text() for p in logs):
                raise AssertionError('diagnostic gradient allocation leaked')
        result = dict(layout=layout, topology=topology, tool=tool, cases=sum(int(v[1]) for v in values),
            exact_float_stores=sum(int(v[3]) for v in values), old_rounded_gradient_max_abs=max(float(v[2]) for v in values))
        reports.append(result);print(json.dumps(result), flush=True)
    (root/'report.json').write_text(json.dumps(dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        records=reports, scope='Frozen same-state local gradients, final FP32 storage and FP64 solver/q/RHS isolation; no main MPI claim'), indent=2)+'\n')


if __name__ == '__main__':main()
