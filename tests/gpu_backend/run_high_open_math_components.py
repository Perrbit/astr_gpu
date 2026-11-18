"""Original H1 production algebra versus native C++ device components, not full H1."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cpu-probe','old-gpu-probe','cpp-probe','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--sanitizer',type=Path)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(scope='FP64 H1 production mathematical dependencies only; not ordinary-main H1 or candidate promotion',
        cases=10,interface_positions=72,grids=[32,64,128],tolerance=2e-10,commands={},binaries={})

    def run(name,command):
        command=[str(x) for x in command];report['commands'][name]=command
        with (out/f'{name}.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        return (out/f'{name}.log').read_text()

    (out/'bin').mkdir()
    for name,source in [('cpu',args.cpu_probe),('old_gpu',args.old_gpu_probe),('cpp',args.cpp_probe)]:
        source=source.resolve();binary=out/'bin'/name;shutil.copy2(source,binary)
        report['binaries'][name]=dict(source=str(source),frozen=str(binary),sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    for name in ('cpu','old_gpu'):
        log=run(name,[out/'bin'/name,out/f'{name}.bin'])
        if 'HB_COMPONENT_ACCURACY_PASS_NOT_SOLVER_STABILITY' not in log:
            raise AssertionError(f'{name} did not complete original component assertions')
    if (out/'cpu.bin').read_bytes()!=(out/'old_gpu.bin').read_bytes():
        raise AssertionError('CPU reference packet differs between original probe builds')
    log=run('cpp',[out/'bin/cpp',out/'cpu.bin'])
    if 'HIGH_OPEN_CPP_COMPONENT_PASS cases=10 positions=72 grids=32,64,128' not in log:
        raise AssertionError('C++ component assertions did not complete')
    errors={name:float(value) for name,value in re.findall(r'HIGH_OPEN_(\w+)_MAX_ABS\s+(\S+)',log)}
    rows=[[float(value) for value in row] for row in re.findall(r'HIGH_OPEN_COMPOSITE\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)',log)]
    orders=[float(value) for value in re.findall(r'HIGH_OPEN_COMPOSITE_ORDER\s+(\S+)',log)]
    if len(errors)!=7 or any(not (0<=value<=2e-10) for value in errors.values()):
        raise AssertionError(f'component differences: {errors}')
    if [int(row[0]) for row in rows]!=[32,64,128] or any(not (0<=value<=2e-10) for row in rows for value in row[1:3]):
        raise AssertionError(f'composite reference differences: {rows}')
    if len(orders)!=2 or any(not math.isfinite(value) or value<5.5 for value in orders):
        raise AssertionError(f'composite convergence: {orders}')
    report.update(max_absolute_errors=errors,composite=rows,composite_orders=orders,
        constant_field_exact=True,interface_positions_exact=True)
    if args.sanitizer:
        log=run('cpp_memcheck',[args.sanitizer.resolve(),'--tool','memcheck','--leak-check','full',
            '--error-exitcode','99','--log-file',out/'memcheck.log',out/'bin/cpp',out/'cpu.bin'])
        tool=(out/'memcheck.log').read_text()
        if 'HIGH_OPEN_CPP_COMPONENT_PASS' not in log or 'ERROR SUMMARY: 0 errors' not in tool or \
                'LEAK SUMMARY: 0 bytes leaked in 0 allocations' not in tool:
            raise AssertionError('missing clean component memory-safety result')
        report['memcheck_zero_errors_and_leaks']=True
    sources=['src/boundary_high_order_body.inc','src/astr_precision.h','src_gpu/high_open_math.cuh',
        'src_gpu/nscbc_incident_math.cuh','tests/gpu_validation/high_order_boundary_probe.F90',
        'tests/gpu_backend/high_open_probe.cu','tests/gpu_backend/run_high_open_math_components.py',
        'tests/gpu_backend/CMakeLists.txt']
    report['source_sha256']={}
    for name in sources:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        report['source_sha256'][name]=hashlib.sha256(target.read_bytes()).hexdigest()
    report['reference_packet_sha256']=hashlib.sha256((out/'cpu.bin').read_bytes()).hexdigest()
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({name:report[name] for name in ('max_absolute_errors','composite','composite_orders')},indent=2))


if __name__=='__main__':main()
