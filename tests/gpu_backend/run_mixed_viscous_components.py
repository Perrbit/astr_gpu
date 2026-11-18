"""Actual mixed flux producer/RHS, typed halo bits and memory-safety component checks."""
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
    for key in ("binary", "packets", "mpiexec", "sanitizer", "output"):
        parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve()
    shutil.copyfile(__file__, root / "runner.py")
    shutil.copyfile(binary, root / "tested_probe")
    records = []
    plan = [("np1", (1, 1, 1), "pinned", None), ("np2_x", (2, 1, 1), "pinned", None),
        ("np2_y", (1, 2, 1), "pinned", None), ("np2_z", (1, 1, 2), "pinned", None),
        ("np2_x_device", (2, 1, 1), "device", None), ("np2_z_device", (1, 1, 2), "device", None),
        ("np2_y_memcheck", (1, 2, 1), "pinned", "memcheck"),
        ("np1_initcheck", (1, 1, 1), "pinned", "initcheck"),
        ("np1_racecheck", (1, 1, 1), "pinned", "racecheck")]
    env = os.environ | dict(OMPI_MCA_pml="ob1", OMPI_MCA_btl="self,vader,tcp", OMPI_MCA_osc="pt2pt",
        OMPI_MCA_coll="^cuda,hcoll,ucc", OMPI_MCA_opal_cuda_support="0")
    for name, topology, transport, tool in plan:
        case = root / name
        case.mkdir()
        ranks = topology[0]*topology[1]*topology[2]
        layout = "np1" if ranks == 1 else "np2_"+"xyz"[topology.index(2)]
        command = [str(args.mpiexec.resolve()), "--bind-to", "none", "-n", str(ranks)]
        if tool:
            command += [str(args.sanitizer.resolve()), "--tool", tool, "--error-exitcode", "99",
                        "--log-file", str(case / "sanitizer.%p.log")]
            if tool == "memcheck":
                command += ["--leak-check", "full"]
        command += [str(binary), str(args.packets.resolve() / layout), ",".join(map(str, topology)), transport]
        execution = env if transport == "pinned" else env | dict(OMPI_MCA_pml="ucx", OMPI_MCA_opal_cuda_support="1")
        with (case / "run.log").open("w") as log:
            subprocess.run(command, env=execution, stdin=subprocess.DEVNULL, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        text = (case / "run.log").read_text()
        values = re.findall(r"CPP_MIXED_VISCOUS_PASS\s+(\d+)\s+(\d+)\s+([\d.Ee+\-]+)\s+([\d.Ee+\-]+)\s+([\d.Ee+\-]+)\s+(\d+)", text)
        if len(values) != ranks or any(int(n) != 12 or max(map(float, (error, grad, cell))) > 2e-10 or int(contrast) == 0
            for _, n, error, grad, cell, contrast in values):
            raise AssertionError((name, "missing phase/cell/precision-order evidence"))
        storage = re.findall(r"CPP_MIXED_VISCOUS_STORAGE\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", text)
        if len(storage) != 2*ranks or any(int(used)*(2 if int(mode) else 1) != int(fp64) for _, mode, used, fp64 in storage):
            raise AssertionError("mixed storage byte ratio")
        if tool:
            logs = list(case.glob("sanitizer.*.log"))
            marker = "RACECHECK SUMMARY: 0 hazards" if tool == "racecheck" else "ERROR SUMMARY: 0 errors"
            if len(logs) != ranks or any(marker not in p.read_text() for p in logs):
                raise AssertionError((name, "memory-safety log"))
            if tool == "memcheck" and any("LEAK SUMMARY: 0 bytes leaked" not in p.read_text() for p in logs):
                raise AssertionError("mixed workspace leaked")
        record = dict(name=name, topology=topology, transport=transport, tool=tool,
            cases=sum(int(v[1]) for v in values), max_scaled=max(float(v[2]) for v in values),
            gradient_scaled=max(float(v[3]) for v in values), constitutive_fp64_max_abs=max(float(v[4]) for v in values),
            storage=[dict(rank=int(r),mode=int(m),bytes=int(b),fp64_bytes=int(f)) for r,m,b,f in storage],command=command)
        records.append(record)
        print(json.dumps(record), flush=True)
    report = dict(binary=str(binary),sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),records=records,
        scope="FP64 state/algebra with FP32 stored viscous flux; components only, not native FP32 or ordinary mixed-mode restart")
    (root / "report.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
