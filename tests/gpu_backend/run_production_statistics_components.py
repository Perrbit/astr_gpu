"""Check original compact GPU statistics, repeat/restore bits and observation isolation."""
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
    for key in ("binary", "reference", "mpiexec", "output", "sanitizer"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument('--precision',choices=('fp64','fp32'),default='fp64')
    parser.add_argument('--cases',nargs='+',choices=('np1','np2_device_isolation','np1_memcheck','np1_initcheck','np1_racecheck'))
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve()
    reference = args.reference.resolve()
    env = os.environ | dict(OMPI_MCA_pml="ob1", OMPI_MCA_btl="self,vader,tcp",
        OMPI_MCA_osc="pt2pt", OMPI_MCA_coll="^cuda,hcoll,ucc", OMPI_MCA_opal_cuda_support="0")
    records = []
    for name, ranks, tool in (("np1", 1, None), ("np2_device_isolation", 2, None),
                             ("np1_memcheck", 1, "memcheck"), ("np1_initcheck", 1, "initcheck"),
                             ("np1_racecheck", 1, "racecheck")):
        if args.cases and name not in args.cases:continue
        case = root / name
        case.mkdir()
        command = [str(args.mpiexec.resolve()), "--bind-to", "none", "-n", str(ranks)]
        if tool:
            command += [str(args.sanitizer.resolve()), "--tool", tool, "--error-exitcode", "99",
                        "--log-file", str(case / "sanitizer.%p.log")]
            if tool == "memcheck":
                command += ["--leak-check", "full"]
        command += [str(binary), str(reference / "statistics")]
        with (case / "run.log").open("w") as log:
            subprocess.run(command, env=env, stdin=subprocess.DEVNULL, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        text = (case / "run.log").read_text()
        results = re.findall(r"CPP_PRODUCTION_STATISTICS_PASS\s+(\d+)\s+(\d+)\s+([\d.Ee+\-]+)", text)
        if len(results) != ranks or any(int(n) != 28 or (args.precision=='fp64' and float(error) > 2e-10) for _, n, error in results):
            raise AssertionError(f"incomplete statistic comparison: {name}")
        widths=re.findall(r'CPP_PRODUCTION_STATISTICS_REAL_BYTES\s+(\d+)\s+(\d+)',text)
        if sorted((int(r),int(w)) for r,w in widths)!=[(r,4 if args.precision=='fp32' else 8) for r in range(ranks)]:
            raise AssertionError('production statistics actual numerical width')
        if tool:
            logs = list(case.glob("sanitizer.*.log"))
            expected = "RACECHECK SUMMARY: 0 hazards" if tool == "racecheck" else "ERROR SUMMARY: 0 errors"
            if len(logs) != ranks or any(expected not in p.read_text() for p in logs):
                raise AssertionError(f"statistic {tool} evidence: {name}")
            if tool == "memcheck" and any("LEAK SUMMARY: 0 bytes leaked" not in p.read_text() for p in logs):
                raise AssertionError("statistic leaked device allocation")
        record = dict(name=name, ranks=ranks, tool=tool, command=command,
                      rank_cases=sum(int(n) for _, n, _ in results),
                      max_scaled_error=max(float(error) for _, _, error in results))
        rows=re.findall(r'CPP_PRODUCTION_STATISTICS_NATIVE_DIFFERENCE\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)',text)
        if len(rows)!=798*ranks:raise AssertionError('native per-channel statistic observations incomplete')
        channels={}
        for rank,kind,debug,repeat,sample,buffer,component,absolute,scaled in rows:
            key=f'kind{kind}/buffer{buffer}/component{component}'
            previous=channels.setdefault(key,dict(max_abs=0.,max_scaled=0.))
            previous['max_abs']=max(previous['max_abs'],float(absolute))
            previous['max_scaled']=max(previous['max_scaled'],float(scaled))
        record['per_channel_report_only']=channels
        records.append(record)
        print(json.dumps(record), flush=True)
    report = dict(binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        reference=str(reference),
        reference_binary_sha256=hashlib.sha256((reference / "compact_statistics_probe").read_bytes()).hexdigest(),
        records=records,
        precision=args.precision,
        scope="Native-kind nondimensional compact plane/wall component; NP2 repeats frozen data on separate devices, not a spatial MPI integration or main/checkpoint gate")
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copyfile(__file__, root / "runner.py")
    shutil.copyfile(binary, root / "tested_probe")


if __name__ == "__main__":
    main()
