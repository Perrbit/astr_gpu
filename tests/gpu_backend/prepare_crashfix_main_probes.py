"""Relink root-CMake objects with a test-only initial-state injection entry."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('cpp-build','old-build','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).with_name('crashfix_main_probe.F90').resolve()
    shutil.copy2(source,out/source.name);shutil.copy2(__file__,out/'preparation.py')
    compiler='/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpif90'
    report={}
    for mode,build in [('cpp',args.cpp_build.resolve()),('old_gpu',args.old_build.resolve())]:
        case=out/mode;case.mkdir();obj=case/'probe.o';binary=case/'probe'
        compile_command=[compiler,'-Mpreprocess','-Mextend','-O2','-I'+str(build/'src'),
            '-I'+str(build/'gpu_cpp/state_modules'),'-I/opt/hdf5-1.14.6/include']
        if mode=='cpp':compile_command+=['-fPIE']
        compile_command+=['-DASTR_GPU_CPP'] if mode=='cpp' else ['-D_CUDA','-cuda','-gpu=rdc']
        compile_command+=['-c',str(source),'-o',str(obj)]
        with (case/'compile.log').open('w') as log:
            subprocess.run(compile_command,cwd=case,stdout=log,stderr=subprocess.STDOUT,check=True)
        command=[]
        for arg in shlex.split((build/'src/CMakeFiles/astr.dir/link.txt').read_text()):
            command.extend(shlex.split((build/'src'/arg[1:]).read_text()) if arg.startswith('@') else [arg])
        matches=[i for i,x in enumerate(command) if x.endswith('/astr.F90.o')]
        if len(matches)!=1:raise AssertionError('root main object identity')
        command[matches[0]]=str(obj);command[command.index('-o')+1]=str(binary)
        with (case/'link.log').open('w') as log:
            subprocess.run(command,cwd=build/'src',stdout=log,stderr=subprocess.STDOUT,check=True)
        report[mode]=dict(compile=compile_command,link=command,
            sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            root_astr_sha256=hashlib.sha256((build/'bin/astr').read_bytes()).hexdigest())
    (out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Root-built CPU orchestration and GPU state/RK/output objects retained in both probes')


if __name__=='__main__':main()
