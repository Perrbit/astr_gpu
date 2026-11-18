"""Locate a failed final trial in the frozen NR1 candidate, not production."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from run_air5_flux_components import read
from diagnose_air5_projection_geometry import analyze


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('components', 'replay', 'compiler', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = a.components/'source'
    for name in ('chemistry_flux_math.cuh', 'chemistry_properties_math.cuh', 'chemistry_air5_data.h'):
        shutil.copy2(source/'src_gpu'/name, out/name)
    shutil.copy2(source/'tests/gpu_backend/air5_flux_math_probe.cu', out/'probe.cu')
    shutil.copy2(__file__, out/'driver.py')
    header = out/'chemistry_flux_math.cuh'
    text = header.read_text()
    key = '        if(iteration==ratio_iterations+2) {'
    assert text.count(key) == 1
    # Report only scalars already calculated by the production phase.
    extra = '            printf("FINAL beta=%.17g last_accepted=%.17g projection=%d admissible=%d\\n",' \
            'beta,beta_lo,projection,int(admissible));\n' \
            '            if(projection!=ok) {\n' \
            '                printf("INPUT");\n' \
            '                for(const double *p : {trial+5, lower, upper}) for(int s=0;s<5;++s)printf(" %.17g",p[s]);\n' \
            '                printf(" %.17g",trial[0]);\n' \
            '                for(int s=0;s<5;++s)printf(" %.17g",gas[s]);\n' \
            '                printf(" %.17g\\n",dot5(gas,trial+5));\n' \
            '            }\n'
    header.write_text('#include <cstdio>\n#include <initializer_list>\n'+text.replace(key, key+'\n'+extra))
    command = [str(a.compiler.resolve()), '-O3', '--fmad=false', '--ftz=false', '-std=c++14',
               '-I'+str(out), str(out/'probe.cu'), '-o', str(out/'probe')]
    with (out/'build.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    n = np.fromfile(a.replay/'face_neighbors.bin', '<i4', count=1)[0]
    cases = np.fromfile(a.replay/'face_neighbors.bin', '<f8', offset=4).reshape(n,80)
    status, values = read(a.replay/'face_neighbors_cpp_0.bin')
    rows = []
    for i in np.flatnonzero(status):
        path = out/f'face{i}.bin'
        with path.open('wb') as f:
            np.array([1],'<i4').tofile(f)
            cases[i].tofile(f)
        target = out/f'output{i}.bin'
        run = subprocess.run([str(out/'probe'), str(path), str(target), '0'], cwd=out, capture_output=True, text=True, check=True)
        (out/f'face{i}.log').write_text(run.stdout+run.stderr)
        ts, tv = read(target)
        assert ts[0] == status[i] and tv[0].tobytes() == values[i].tobytes()
        lines = run.stdout.splitlines()
        fields = dict(word.split('=') for word in lines[0].split()[1:])
        assert int(fields['admissible']) == 0
        if int(fields['projection']) != 0:
            assert lines[1].startswith('INPUT ')
            geometry = analyze(np.asarray(list(map(float, lines[1].split()[1:]))))
            fields['geometry'] = geometry
        rows.append(dict(index=int(i), status=int(ts[0]), **fields))
    (out/'report.json').write_text(json.dumps(dict(command=command, cases=rows),indent=2)+'\n')
    print(json.dumps([dict(index=r['index'], projection=r['projection'],
                          physical_witness_passes=r.get('geometry',{}).get('rounded_physical_witness',{}).get('passes_original_final_checks'))
                      for r in rows],indent=2))


if __name__ == '__main__':
    main()
