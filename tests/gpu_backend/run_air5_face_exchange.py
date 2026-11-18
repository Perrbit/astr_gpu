"""Exact private face-plane MPI checks, independent of the AIR5 limiter."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]
FILES=["src_gpu/halo_transport.cu","src_gpu/chemistry_convection.cu","src_gpu/chemistry_transport.h",
       "src_gpu/chemistry_transport.F90","src_gpu/device_state.cu","src_gpu/device_state_internal.cuh",
       "src_gpu/device_state.h","tests/gpu_backend/air5_face_exchange_probe.cu",
       "tests/gpu_backend/air5_convection_abi.F90","tests/gpu_backend/run_air5_face_exchange.py",
       "tests/gpu_backend/CMakeLists.txt","src_gpu/CMakeLists.txt"]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",required=True,type=Path)
    parser.add_argument("--mpiexec",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--memcheck",type=Path)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=out/"probe";shutil.copy2(args.binary.resolve(),binary)
    hashes={}
    for name in FILES:
        dst=out/"source"/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst)
        hashes[name]=hashlib.sha256(dst.read_bytes()).hexdigest()
    env=os.environ|dict(OMPI_MCA_pml="ucx",OMPI_MCA_opal_cuda_support="1",
                        OMPI_MCA_coll="^cuda,hcoll,ucc",OMPI_MCA_osc="pt2pt")
    report=dict(source_sha256=hashes,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                mpi_environment={k:v for k,v in env.items() if k.startswith("OMPI_MCA_")},runs={})
    for topology in [(1,1,1),(2,1,1),(1,2,1),(1,1,2)]:
        ranks=topology[0]*topology[1]*topology[2];name="np%d_%d%d%d"%(ranks,*topology)
        command=[str(args.mpiexec),"--bind-to","none","-np",str(ranks),str(binary),*map(str,topology)]
        result=subprocess.run(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (out/(name+".log")).write_text(result.stdout);result.check_returncode()
        lines=[line for line in result.stdout.splitlines() if line.startswith("AIR5_FACE_EXCHANGE_PASS")]
        if len(lines)!=8*ranks:raise AssertionError((name,len(lines),8*ranks))
        report["runs"][name]=dict(command=command,contexts=len(lines),face_exchange_calls=6*len(lines),result="exact")
        (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        print(name,len(lines),"contexts exact",flush=True)
    if args.memcheck:
        command=[str(args.mpiexec),"--bind-to","none","-np","2",str(args.memcheck),"--tool","memcheck",
                 "--leak-check","full","--error-exitcode","99",str(binary),"1","2","1","pinned"]
        mem_env=env|dict(OMPI_MCA_pml="ob1",OMPI_MCA_btl="self,vader,tcp",OMPI_MCA_opal_cuda_support="0")
        result=subprocess.run(command,env=mem_env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (out/"memcheck.log").write_text(result.stdout);result.check_returncode()
        if result.stdout.count("ERROR SUMMARY: 0 errors")!=2:raise AssertionError("missing per-rank zero-error summaries")
        if result.stdout.count("LEAK SUMMARY: 0 bytes leaked")!=2:raise AssertionError("missing per-rank zero-leak summaries")
        report["memcheck"]=dict(command=command,transport="pinned",zero_error_ranks=2,zero_leak_ranks=2,
                                mpi_environment={k:v for k,v in mem_env.items() if k.startswith("OMPI_MCA_")})
        (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        print("pinned memcheck two ranks zero errors/leaks",flush=True)

if __name__=="__main__":main()
