"""Fail-closed admission checks for the optional conservative SBLI mode."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
CONFIG_KEY = "ASTR_CONSERVATIVE_BOUNDARY_FILE"
TEST_KEY = "ASTR_CONSERVATIVE_ADMISSION_TEST"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = out / "valid.nml"
    config.write_text(
        "&conservative_boundary\n schema=1, split_x=40,\n"
        " q_left=1.00000596004,1.00000268202,.00565001630205,0,.94644428042,\n"
        " q_right=1.129734572,1.0921171,-.058866065,0,1.0590824\n/\n",
        encoding="ascii",
    )
    cases = {
        "valid": None,
        "wrong_flow": "flowtype=bl",
        "wrong_scheme": "conschm=543e",
        "filter": "lfilter=false",
        "bad_pr": "Prandtl number must be explicitly 0.71 or 0.72",
        "bad_tw": "wall temperature must be finite and positive",
        "wrong_ninit": "ninit=3",
        "restart": None,
        "wrong_bctype": "boundary topology contract",
        "wrong_topology": "supported NP=1/2/4 topology",
    }
    report = {"status": "running", "scope": "host production-mode admission only", "checks": []}
    summary = out / "summary.json"
    for kind in ("cpu", "gpu"):
        exe = ROOT / f"build_{kind}_probe/bin/astr"
        digest = hashlib.sha256(exe.read_bytes()).hexdigest()
        for mode, rejection in cases.items():
            env = dict(os.environ, OMPI_MCA_sharedfp="individual", **{
                CONFIG_KEY: str(config), TEST_KEY: mode,
            })
            command = ["timeout", "--kill-after=5s", "45s", "mpirun", "-np", "1",
                       str(exe), "test", "bcad"]
            log = out / f"{kind}_{mode}.log"
            with log.open("w") as stream:
                result = subprocess.run(command, cwd=out, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, check=False)
            output = log.read_text()
            if rejection is None:
                passed = result.returncode == 0 and output.count("CONSERVATIVE_ADMISSION_PASS") == 1
            else:
                passed = (result.returncode == 1 and rejection in output and
                          "CONSERVATIVE_ADMISSION_PASS" not in output)
            report["checks"].append({"name": f"{kind}_{mode}", "pass": passed,
                                     "returncode": result.returncode, "command": command,
                                     "binary_sha256": digest, "log": str(log)})
            report["status"] = "running" if passed else "fail"
            summary.write_text(json.dumps(report, indent=2) + "\n")
            if not passed:
                raise RuntimeError(f"Failed {kind}_{mode}: {log}")
    report["status"] = "pass"
    summary.write_text(json.dumps(report, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
