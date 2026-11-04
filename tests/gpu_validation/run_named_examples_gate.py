"""Read named example fields on two ranks; do not allocate or advance a CFD field."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from run_case_input_gate import parse_probe

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--mpiexec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    executable=args.executable.resolve()
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    env={key:value for key,value in os.environ.items() if not key.startswith('ASTR_')}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_coll='^cuda,hcoll,ucc',
        OMPI_MCA_opal_cuda_support='0')
    report={'read':{},'requires_comb':[]}
    for primary in sorted((ROOT/'examples').rglob('*.case')):
        if primary.parent.name!='datin':
            continue
        # The COMB-only physics group is not compiled in the selected AIR5/non-COMB binary.
        expected=parse_probe(primary.read_text())
        if 'chemfile' in expected.get('physics',{}):
            report['requires_comb'].append(str(primary.relative_to(ROOT)))
            continue
        case=args.output/primary.relative_to(ROOT).parent.parent
        case=case/primary.name
        case.mkdir(parents=True)
        with (case/'run.log').open('w') as log:
            process=subprocess.run([str(args.mpiexec),'-n','2',str(executable),'test','case',str(primary)],
                cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT)
        if process.returncode:
            raise AssertionError(f'named reader failed: {case}/run.log')
        a=(case/'case_input.rank000000.nml').read_text()
        b=(case/'case_input.rank000001.nml').read_text()
        if a!=b:
            raise AssertionError(f'named fields differ across ranks: {primary}')
        actual=parse_probe(a)
        receipt=actual['model_option_receipt']
        model=dict(zip(map(str.strip,receipt['option_names']),map(str.strip,receipt['option_values']),strict=True))
        def scalar(value):
            token=value.strip()
            logical=token.strip('.').lower()
            if logical in ('true','t','false','f'):
                return logical in ('true','t')
            if re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?',token):
                return float(token.lower().replace('d','e'))
            return token
        for group,fields in expected.items():
            for name,value in fields.items():
                found=actual[group][name] if group in actual else [model[name]]
                equal=list(map(scalar,value))==list(map(scalar,found))
                if not equal:
                    raise AssertionError(f'field changed: {primary} {group}.{name}: {value} != {found}')
        report['read'][str(primary.relative_to(ROOT))]={'rank_fields_equal':True,'all_declared_fields_match':True}
    report['status']='passed'
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
