"""Integration gate for optional perfect-gas transport, not SBLI validation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/gpu_validation"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    for key in ("ASTR_PERFECT_GAS_PRANDTL", "ASTR_SUTHERLAND_TEMPERATURE_K"):
        env.pop(key, None)
    env.update(OMPI_MCA_sharedfp="individual", MAXSTEP="2", FEQCHKPT="2",
               LFILTER="t", DIFFTERM="t", SCHEME="643e", ATOL="1e-10", RTOL="1e-10")
    binaries = {kind: Path(env.get(f"{kind}_EXE", ROOT / f"build_{kind.lower()}_probe/bin/astr")).resolve()
                for kind in ("CPU", "GPU")}
    for kind, exe in binaries.items():
        env[f"{kind}_EXE"] = str(exe)
    report = {"status": "running", "scope": "transport configuration TGV regression",
              "binaries": {k: {"path": str(v), "sha256": hashlib.sha256(v.read_bytes()).hexdigest()}
                           for k, v in binaries.items()}, "checks": []}
    report_path = out / "summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    def run(name, command, run_env, cwd=ROOT, expect_error=False):
        log = out / f"{name}.log"
        start = time.monotonic()
        with log.open("w") as stream:
            result = subprocess.run(["timeout", "--kill-after=10s", "300s", *map(str, command)],
                                    cwd=cwd, env=run_env, stdout=stream, stderr=subprocess.STDOUT)
        passed = result.returncode == 0
        if expect_error:
            passed = (result.returncode == 1 and
                      "Invalid ASTR perfect-gas transport environment parameter" in log.read_text())
        report["checks"].append({"name": name, "command": list(map(str, command)),
                                 "returncode": result.returncode, "pass": passed,
                                 "elapsed_s": time.monotonic() - start, "log": str(log)})
        if not passed:
            report["status"] = "fail"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        if not passed:
            raise RuntimeError(f"Failed {name}; see {log}")

    for mode in ("default", "reference"):
        for np in (1, 2):
            case = out / f"{mode}_np{np}"
            run_env = dict(env, OUT_DIR=str(case), MPI_NP=str(np), TOPOLOGY=f"{np},1,1")
            if mode == "reference":
                run_env.update(ASTR_PERFECT_GAS_PRANDTL="0.71d0", ASTR_SUTHERLAND_TEMPERATURE_K="110.4d0")
            run(f"{mode}_np{np}_field", ["bash", TESTS / "run_tgv_mpirank2_field_compare.sh"], run_env)
            run(f"{mode}_np{np}_stats", [sys.executable, TESTS / "compare_flowstate.py",
                "--cpu", case / "cpu", "--gpu", case / "gpu", "--report", case / "flowstate_compare.txt",
                "--atol", "1e-10", "--rtol", "1e-10"], run_env)
    for kind, exe in binaries.items():
        for key in ("ASTR_PERFECT_GAS_PRANDTL", "ASTR_SUTHERLAND_TEMPERATURE_K"):
            bad_env = dict(env, ASTR_FORCE_MPI_TOPOLOGY="2,1,1")
            bad_env[key] = "bad"
            run(f"reject_{kind}_{key}", ["mpirun", "-np", "2", exe, "run", "datin/input.tgv"],
                bad_env, cwd=out / "reference_np2" / kind.lower(), expect_error=True)
    report["status"] = "pass"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(report_path)


if __name__ == "__main__":
    main()
