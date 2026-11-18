"""Reproduce the NVHPC AIR5 wrapper source-mode difference without changing math.

Both modes compile identical bytes and original predicate bodies at O2. Earlier
unsuccessful predicate rewrites remain only in their frozen diagnostic results.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from run_air5_flux_components import read

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--compiler",type=Path,required=True);args=p.parse_args()
    ref=args.reference.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    x=np.fromfile(ref/"input.bin","<f8",offset=4).reshape(-1,80);selected=x[:,79]==3
    cpu=read(ref/"cpu0.bin");wrapper=ref/"source/tests/gpu_backend/air5_flux_reference.F90"
    report=dict(scope="identical wrapper bytes, original functions, O2; no production modification",modes={})
    for suffix in ["F90","cuf"]:
        work=out/suffix;work.mkdir();source=work/("wrapper."+suffix)
        shutil.copy2(wrapper,source)
        shutil.copy2(ref/"old_gpu/frozen_flux_functions.inc",work/"frozen_flux_functions.inc")
        flags=[str(args.compiler.resolve()),"-O2","-Mpreprocess","-Mextend","-cuda","-gpu=rdc,maxregcount:128",
            "-D_CUDA","-DASTR_AIR5_REFERENCE_GPU","-I"+str(work),"-I"+str(ref/"old_gpu"),"-I"+str(ref/"source/src")]
        objects=[ref/"old_gpu"/(name+".o") for name in ["chemistry_air5_data","chemistry_core","chemistry_properties",
            "chemistry_core_gpu","chemistry_flow_state_gpu"]]
        commands=[flags+[str(source),"-c","-o",str(work/"wrapper.o")],
            flags+list(map(str,objects))+[str(work/"wrapper.o"),"-o",str(work/"probe")],
            [str(work/"probe"),str(ref/"input.bin"),str(work/"output.bin"),"0"]]
        for i,command in enumerate(commands):
            with (work/f"command{i}.log").open("w") as stream:
                subprocess.run(command,cwd=work,stdout=stream,stderr=subprocess.STDOUT,check=True)
        dryrun=flags+["-dryrun",str(source),"-c","-o",str(work/"unused.o")]
        with (work/"dryrun.log").open("w") as stream:
            subprocess.run(dryrun,cwd=work,stdout=stream,stderr=subprocess.STDOUT,check=True)
        status,values=read(work/"output.bin")
        result=dict(commands=commands,dryrun=dryrun,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            mask_differences_to_cpu=int(np.sum(np.any(values[selected,1:3]!=cpu[1][selected,1:3],axis=1))),
            ratio_max_abs_to_cpu=float(np.max(abs(values[selected,0]-cpu[1][selected,0]))),
            status_differences=int(np.sum(status!=cpu[0])),
            fort2ex_passes=sum("/tools/fort2ex " in line for line in (work/"dryrun.log").read_text().splitlines()))
        report["modes"][suffix]=result
    assert report["modes"]["F90"]["source_sha256"]==report["modes"]["cuf"]["source_sha256"]
    assert report["modes"]["cuf"]["mask_differences_to_cpu"]==0
    assert report["modes"]["cuf"]["ratio_max_abs_to_cpu"]==0
    assert report["modes"]["cuf"]["status_differences"]==0
    shutil.copy2(__file__,out/"driver.py");(out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:{a:b for a,b in v.items() if a not in ["commands","dryrun","source_sha256"]} for k,v in report["modes"].items()},indent=2))

if __name__=="__main__":main()
