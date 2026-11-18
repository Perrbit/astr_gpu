"""Freeze and exercise AIR5 IEEE transactions and persistent multi-region chemistry."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--memcheck", action="store_true")
    args = p.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    build = args.build.resolve()
    for target in ["astr_backend_air5_ros2", "astr_backend_air5_coupling_state", "astr_backend_air5_coupling"]:
        shutil.copy2(build/"bin"/target, out/target)
    shutil.copy2(args.input, out/"input.in")
    shutil.copy2(__file__, out/"driver.py")
    files = ["src_gpu/chemistry_coupling.cu", "src_gpu/chemistry_coupling.F90", "src_gpu/chemistry_coupling.h",
             "src_gpu/chemistry_ros2_math.cuh", "src_gpu/chemistry_kinetics_math.cuh", "src_gpu/chemistry_compensation_math.cuh",
             "src_gpu/chemistry_properties.cu", "src_gpu/chemistry_properties_math.cuh", "src_gpu/chemistry_air5_data.h",
             "src_gpu/device_state.cu", "src_gpu/device_state.h", "src_gpu/device_state_internal.cuh", "src_gpu/device_state.F90",
             "src_gpu/CMakeLists.txt", "tests/gpu_backend/CMakeLists.txt", "tests/gpu_backend/air5_coupling_probe.F90",
             "tests/gpu_backend/air5_ros2_probe.cu", "tests/gpu_backend/air5_coupling_state_probe.cu"]
    for name in files:
        path = out/"source"/name; path.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name, path)
    for name in ["CMakeCache.txt", "gpu_cpp/CMakeFiles/astr_device_state.dir/flags.make",
                 "gpu_backend/CMakeFiles/astr_backend_air5_coupling.dir/link.txt"]:
        path = out/"build_identity"/name; path.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(build/name, path)
    report = dict(status="running", scope="local chemistry components; not transport, MPI or checkpoint acceptance",commands=[],
                  source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files})
    commands = [([str(out/"astr_backend_air5_ros2"), "contracts", str(out/"input.in"), str(out/"transactions.dat")], "transactions.log"),
                ([str(out/"astr_backend_air5_coupling_state"), str(out/"input.in")], "fields.log")]
    if args.memcheck:
        sanitizer = shutil.which("compute-sanitizer")
        if sanitizer is None: raise RuntimeError("requested Compute Sanitizer is unavailable")
        commands.append(([sanitizer, "--tool", "memcheck", "--leak-check", "full", "--error-exitcode", "77",
                          "--log-file", str(out/"memcheck.log"), str(out/"astr_backend_air5_coupling_state"), str(out/"input.in")], "memcheck_stdout.log"))
    (out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    for command, log in commands:
        report["commands"].append(command)
        with (out/log).open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    rows = np.loadtxt(out/"transactions.dat", dtype=int)
    assert rows.shape == (24, 10) and np.all(rows[:, 2] == rows[:, 3]) and np.all(rows[:, 4] == 1)
    ieee = [line for line in (out/"transactions.dat").read_text().splitlines() if line.startswith("# AIR5_IEEE ")]
    assert len(ieee) == 1
    fields = dict(word.split("=") for word in ieee[0].split()[2:])
    assert all(v == "1" for v in fields.values()) and len(fields) == 3
    output = (out/"fields.log").read_text().splitlines()
    expected = [f"AIR5_CONTEXT test={i} active=62 field_mapping_exact=1" for i in range(8)]
    assert output == expected
    if args.memcheck:
        text = (out/"memcheck.log").read_text()
        assert "ERROR SUMMARY: 0 errors" in text and "0 bytes leaked in 0 allocations" in text
    report.update(status="component-contracts-passed", transaction_rows=len(rows), ieee=fields,
                  field_variants=8, field_shape=[8, 6, 4], halo=5, active_nodes_per_region_group=62,
                  checks=["distinct per-node density/carry/history mapping", "non-active/halo/corner values unchanged",
                          "enabled primitive caches match accepted q", "carry and history allocations independent",
                          "general/specialized and diagnostic isolation", "array restore continuation exact",
                          "earlier-region failure retained through later success", "positive-extrema bit reductions exact"])
    (out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
