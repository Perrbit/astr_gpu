"""Check tiled TGV initialization and benchmark I/O with the local CUDA executable."""
import argparse
import json
from math import prod
import os
from pathlib import Path
import re
import subprocess

import numpy as np

from prepare_dcu_weak_scaling import (CASES,case_configuration,normalized_enstrophy,
                                     placement_check,prepare,reduce_run,write_inputs)
from run_cpp_tgv_gate import compare, snapshot


def audit_matrix(root):
    prepare(root)
    campaign = json.loads((root/'campaign.json').read_text())
    assert len(campaign)==27
    expected_counts = dict(explicit=29750853632,compact=22105827328,air5=9710862336)
    for name,config in campaign.items():
        base = CASES[config['kind']][0]
        assert prod(config['mpi'])==4*config['nodes']
        assert [g//p for g,p in zip(config['grid'],config['mpi'])]==[base//2,base//2,base]
        if config['nodes']==256:
            assert prod(config['grid'])==expected_counts[config['kind']]
        px,py,_ = config['mpi']
        def rows(blocks):
            return [dict(rank=r,host=str(i),coords=[r%px,r//px%py,r//(px*py)])
                    for i,block in enumerate(blocks) for r in block]
        placement_check(config,rows(config['rank_blocks']))
        if name=='explicit_004nodes':
            try:
                placement_check(config,rows([range(i,i+4) for i in range(0,16,4)]))
            except ValueError:
                pass
            else:
                raise AssertionError('contiguous four-rank placement was not rejected')
    return expected_counts


def run_case(binary, mpiexec, root, kind, tiles, mpi):
    config = case_configuration(kind,16,tiles,mpi)
    write_inputs(root,config)
    env = os.environ | config['environment'] | dict(
        OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',
        ASTR_VALIDATION_RHS_PREFIX='outdat/probe',ASTR_VALIDATION_RHS_STEP='0',
        ASTR_VALIDATION_RHS_STEP_SECONDARY='5')
    ranks = prod(mpi)
    with (root/'run.log').open('w') as log:
        subprocess.run([str(mpiexec),'-n',str(ranks),str(binary),'run'],cwd=root,env=env,
                       stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,check=True)
    log = (root/'run.log').read_text()
    if 'ASTR_GPU_BENCHMARK_NO_FIELD_IO enabled' not in log or 'The job is done!' not in log:
        raise AssertionError('incomplete no-field-I/O benchmark')
    if any(root.rglob('*.h5')) or any(root.rglob('*.xdmf')) or any(root.rglob('*.plt')):
        raise AssertionError('unexpected grid or field output')
    clocks = reduce_run(log,config)
    spacing = tuple(map(float,re.search(r'grid spacing:\s*(\S+)\s*~\s*(\S+)',log).groups()))
    cfl = [float(line.split()[-1]) for line in re.findall(r'^ASTR_CFL directional/local_sum/upper_bound=.+$',log,re.M)]
    if len(cfl)!=5 or any(not 0<c<1 for c in cfl):
        raise AssertionError('missing or inadmissible CFL')
    def field(label, step, rank):
        return snapshot(root/'outdat'/f'probe.{label}.step{step:08d}.rk01.rank{rank:08d}.bin')
    return dict(timing=clocks,spacing=spacing,max_cfl=max(cfl)),[field('initialized',0,r) for r in range(ranks)], \
           [field('cpp_final',5,r) for r in range(ranks)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--mpiexec',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    root = a.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    report = dict(full_size_counts=audit_matrix(root/'production_templates'),local={})
    for kind in CASES:
        reference = root/f'{kind}_base'
        entry,initial,terminal = run_case(a.binary.resolve(),a.mpiexec.resolve(),reference,kind,(1,1,1),(1,1,1))
        report['local'][kind] = {'base':entry}
        variants = [('x',(2,1,1))] if kind!='air5' else [('x',(2,1,1)),('z',(1,1,2))]
        for axis,tiles in variants:
            case = root/f'{kind}_tile_{axis}'
            observed,i_blocks,t_blocks = run_case(a.binary.resolve(),a.mpiexec.resolve(),case,kind,tiles,tiles)
            np.testing.assert_allclose(observed['spacing'],entry['spacing'],rtol=5e-7,atol=0)
            # The two ranks each cover one tile, including both copies of the interface node.
            observed['initial_difference'] = max(max(compare(f,initial[0])) for f in i_blocks)
            observed['terminal_difference'] = max(max(compare(f,terminal[0])) for f in t_blocks)
            if kind!='air5':
                base = np.loadtxt(reference/'flowstate.dat',skiprows=1,ndmin=2)
                tiled = np.loadtxt(case/'flowstate.dat',skiprows=1,ndmin=2)
                tiled[:,3] = normalized_enstrophy(tiled[:,3],tiles)
                # flowstate.dat has formatted, not full-precision, scalar output.
                np.testing.assert_allclose(tiled,base,rtol=2e-6,atol=1e-12)
            report['local'][kind][axis] = observed
            print(kind,axis,'same spacing, periodic tile and five complete steps without field I/O',flush=True)
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
