"""Relink the root CUDA Fortran build with the observation-only stage entry."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build=args.build.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]
    source=root/'tests/gpu_validation/mp_positivity_stage_reference.cuf'
    shutil.copy2(source,out/source.name);shutil.copy2(__file__,out/'preparation.py')
    compiler='/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpif90'
    compile_command=[compiler,'-Mpreprocess','-Mextend','-cuda','-gpu=rdc','-O2',
        '-I'+str(build/'src'),'-I/opt/hdf5-1.14.6/include','-c',str(out/source.name),'-o',str(out/'probe.o')]
    with (out/'compile.log').open('w') as log:
        subprocess.run(compile_command,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
    command=[]
    for argument in shlex.split((build/'src/CMakeFiles/astr.dir/link.txt').read_text()):
        command.extend(shlex.split((build/'src'/argument[1:]).read_text()) if argument.startswith('@') else [argument])
    matches=[i for i,word in enumerate(command) if word.endswith('/astr.F90.o')]
    assert len(matches)==1
    command[matches[0]]=str(out/'probe.o');command[command.index('-o')+1]=str(out/'probe')
    with (out/'link.log').open('w') as log:
        subprocess.run(command,cwd=build/'src',stdout=log,stderr=subprocess.STDOUT,check=True)
    (out/'build.json').write_text(json.dumps(dict(compile=compile_command,link=command,
        binary_sha256=hashlib.sha256((out/'probe').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),indent=2)+'\n')
    print(out/'probe')


if __name__=='__main__':main()
