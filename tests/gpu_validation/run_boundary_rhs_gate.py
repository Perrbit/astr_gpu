"""Check physical-face Euler RHS against an analytic affine-density state."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
CPU_MARKERS = ("BOUNDARY_RHS_MANUFACTURED_PASS", "BOUNDARY_RHS_ROE_PASS", "BOUNDARY_RHS_DIFFUSION_PASS",
               "BOUNDARY_RHS_STRETCHED_PHYSICAL_PASS", "BOUNDARY_RHS_STRETCHED_ROE_PASS",
               "BOUNDARY_RHS_FIXED_EPS_CONTROL_PASS", "BOUNDARY_RHSCAL_COMPOSITION_PASS",
               "BOUNDARY_RHSCAL_METRIC_PASS")
GPU_MARKERS = CPU_MARKERS + ("BOUNDARY_RHS_GPU_PHYSICAL_PASS", "BOUNDARY_RHS_GPU_ROE_PASS",
                             "BOUNDARY_RHS_GPU_DIFFUSION_PASS")
GPU_MARKERS += ("BOUNDARY_RHS_GPU_STRETCHED_PHYSICAL_PASS", "BOUNDARY_RHS_GPU_STRETCHED_ROE_PASS")
GPU_MARKERS += tuple(f"BOUNDARY_RHS_{case}_{mode}_PASS" for case in ("SMOOTH", "JUMP_X", "JUMP_Y")
                     for mode in ("PHYSICAL", "ROE"))


def check_output(output, markers, sanitizer=False):
    errors = {}
    for marker in markers:
        values = re.findall(rf"{marker} max_abs=\s*(\S+)", output)
        if len(values) != 1:
            raise ValueError(f"Expected exactly one {marker} result")
        value = float(values[0].replace("D", "E"))
        if not math.isfinite(value) or not 0 <= value <= 1e-10:
            raise ValueError(f"Invalid manufactured-solution error: {marker}={value}")
        errors[marker] = value
    if sanitizer and "ERROR SUMMARY: 0 errors" not in output:
        raise ValueError("Missing clean Compute Sanitizer summary")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cpu", type=Path, default=ROOT / "build_cpu_probe/bin/astr")
    parser.add_argument("--gpu", type=Path, default=ROOT / "build_gpu_probe/bin/astr")
    args = parser.parse_args()
    binaries = {key: path.resolve() for key, path in (("cpu", args.cpu), ("gpu", args.gpu))}
    report = {"status": "running", "scope": "NP1 CPU/GPU Euler and viscous RHS; not SBLI validation",
              "tolerance_abs": 1e-10,
              "binaries": {key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                           for key, path in binaries.items()}, "checks": []}
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMPI_MCA_sharedfp="individual")
    summary = out / "summary.json"
    summary.write_text(json.dumps(report, indent=2) + "\n")
    for name, kind, sanitizer in (("cpu", "cpu", False), ("gpu", "gpu", False),
                                  ("gpu_memcheck", "gpu", True)):
        command = ["mpirun", "-np", "1"]
        if sanitizer:
            command += ["compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "1"]
        command += [str(binaries[kind]), "test", "bcrh"]
        command = ["timeout", "--kill-after=10s", "120s", *command]
        start = time.monotonic()
        log = out / f"{name}.log"
        entry = {"name": name, "command": command, "log": str(log), "pass": False}
        run_env = dict(env)
        if sanitizer:
            # This NP1 kernel probe has no MPI transport to validate. Avoid UCX's
            # pre-kernel CUDA context probes without suppressing CUDA API errors.
            overrides = {"OMPI_MCA_pml": "ob1", "OMPI_MCA_osc": "pt2pt",
                         "OMPI_MCA_btl": "self,vader,tcp"}
            run_env.update(overrides)
            entry["environment_overrides"] = overrides
        report["checks"].append(entry)
        try:
            with log.open("w") as stream:
                result = subprocess.run(command, cwd=out, env=run_env, stdout=stream,
                                        stderr=subprocess.STDOUT, check=False)
            entry["returncode"] = result.returncode
            if result.returncode:
                raise RuntimeError(f"{name} returned {result.returncode}; see {log}")
            entry["max_abs"] = check_output(log.read_text(), GPU_MARKERS if kind == "gpu" else CPU_MARKERS,
                                             sanitizer)
            entry["pass"] = True
        except Exception as error:
            report["status"] = "fail"
            entry["error"] = str(error)
            raise
        finally:
            entry["elapsed_s"] = time.monotonic() - start
            summary.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "pass"
    summary.write_text(json.dumps(report, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
