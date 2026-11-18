"""Relink root-built legacy objects with the typed crashfix observation entry."""
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
    source=root/'tests/gpu_validation/crashfix_reference_probe.cuf'
    shutil.copy2(source,out/source.name);shutil.copy2(__file__,out/'preparation.py')
    command=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpif90',
        '-Mpreprocess','-Mextend','-cuda','-gpu=rdc','-O2','-DASTR_BUILD_TESTING',
        '-I'+str(build/'src'),'-I/opt/hdf5-1.14.6/include','-c',str(out/source.name),'-o',str(out/'probe.o')]
    with (out/'compile.log').open('w') as log:
        subprocess.run(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
    link=[]
    for word in shlex.split((build/'src/CMakeFiles/astr.dir/link.txt').read_text()):
        link.extend(shlex.split((build/'src'/word[1:]).read_text()) if word.startswith('@') else [word])
    entries=[i for i,word in enumerate(link) if word.endswith('/astr.F90.o')]
    assert len(entries)==1
    link[entries[0]]=str(out/'probe.o');link[link.index('-o')+1]=str(out/'probe')
    with (out/'link.log').open('w') as log:
        subprocess.run(link,cwd=build/'src',stdout=log,stderr=subprocess.STDOUT,check=True)
    (out/'build.json').write_text(json.dumps(dict(compile=command,link=link,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        sha256=hashlib.sha256((out/'probe').read_bytes()).hexdigest()),indent=2)+'\n')
    print(out/'probe')


if __name__=='__main__':main()
