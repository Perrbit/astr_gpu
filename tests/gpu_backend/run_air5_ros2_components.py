"""Use the existing eleven-state G1 reference and checker for the shared C++ ROS-2."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"tests/gpu_validation"))
from air5_radau_reference import Air5RadauReference
from check_air5_actual_error import compare_case, require_g1
from run_air5_state_workload import TOLERANCES, parse_results


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--tolerance", choices=TOLERANCES, default="baseline")
    p.add_argument("--source-evaluation", choices=("joint", "source-only"), default="joint")
    p.add_argument("--segments", type=int, default=4)
    p.add_argument("--history", action="store_true")
    p.add_argument("--activity-threshold", type=float, default=0.)
    p.add_argument("--backend", choices=("cpp", "context", "old-gpu", "cpu"), default="cpp")
    args = p.parse_args()
    if args.backend == "cpu" and args.source_evaluation != "joint":
        p.error("the existing CPU probe exposes joint evaluation only")
    reference = json.loads(args.reference.read_text())
    assert reference["status"] == "reference-convergence-collected-awaiting-G1"
    mechanism = ROOT/"chemMech/air5_kimjo12.json"
    assert reference["mechanism_sha256"] == digest(mechanism)
    assert reference["reference_code_sha256"] == digest(ROOT/"tests/gpu_validation/air5_radau_reference.py")
    assert len(reference["cases"]) == 11
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.candidate, out/"candidate")
    shutil.copy2(args.reference, out/"reference.json")
    shutil.copy2(__file__, out/"driver.py")
    files = ["src_gpu/chemistry_ros2_math.cuh", "src_gpu/chemistry_kinetics_math.cuh",
             "src_gpu/chemistry_compensation_math.cuh",
             "src_gpu/chemistry_properties_math.cuh", "src_gpu/chemistry_air5_data.h",
             "tests/gpu_backend/air5_ros2_probe.cu"]
    if args.backend == "context":
        files = files[:-1]+["src_gpu/chemistry_coupling.cu", "src_gpu/chemistry_coupling.F90",
                  "src_gpu/chemistry_coupling.h", "src_gpu/device_state_internal.cuh",
                  "src_gpu/device_state.h", "src_gpu/device_state.F90", "src_gpu/device_state.cu",
                  "src_gpu/chemistry_properties.cu", "src_gpu/chemistry_properties.F90",
                  "src_gpu/chemistry_properties.h", "src_gpu/CMakeLists.txt",
                  "tests/gpu_backend/air5_coupling_probe.F90"]
    elif args.backend != "cpp":
        files = ["src/chemistry_air5_data.F90", "src/chemistry_core.F90", "src/chemistry_properties.F90",
                 "src/chemistry_kinetics.F90", "src_gpu/chemistry_core_gpu.cuf", "src_gpu/chemistry_kinetics_gpu.cuf",
                 "src_gpu/chemistry_relaxation_gpu.cuf", "src/chemistry_pure_vt_body.inc",
                 "src/chemistry_pure_source_body.inc", "src/chemistry_activity_index_body.inc",
                 "src/chemistry_activity_gate_body.inc", "src/chemistry_vibrational_difference_body.inc",
                 "tests/gpu_validation/chemistry_ros2_compensation_gpu_probe.cuf"]
    for name in files: shutil.copy2(ROOT/name, out/Path(name).name)
    model = Air5RadauReference(mechanism)
    rtol, factor = TOLERANCES[args.tolerance]
    report = dict(status="running", scope="existing G1 eleven-state actual error; not full-flow or physical acceptance",
                  reference_sha256=digest(args.reference), candidate_sha256=digest(out/"candidate"),
                  source_sha256={name: digest(ROOT/name) for name in files},
                  tolerance=args.tolerance, source_evaluation=args.source_evaluation, segments=args.segments,
                  history=args.history, activity_threshold=args.activity_threshold, backend=args.backend, cases=[])
    (out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    for case in reference["cases"]:
        stem = out/case["name"]
        input_path, result_path = stem.with_suffix(".in"), stem.with_suffix(".out")
        row = np.r_[case["rho"], case["momentum"], case["q5"], case["initial"], case["carry"]]
        with input_path.open("w") as stream:
            stream.write(f'1 {case["duration"]:.17e} '+str({"coupled": 0, "chemical": 1, "vt": 2}[case["mode"]])+"\n")
            np.savetxt(stream, row[None, :], fmt="%.17e")
        command = [str(out/"candidate"), "batch-cpu" if args.backend == "cpu" else "batch", str(input_path), str(result_path),
                   f"{rtol:.17e}", f"{factor:.17e}", str(args.segments), "history" if args.history else "reset"]
        env = dict(os.environ, ASTR_AIR5_SOURCE_EVALUATION=args.source_evaluation,
                   ASTR_AIR5_LOW_ACTIVITY=str(args.activity_threshold))
        with stem.with_suffix(".log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if args.backend != "context": parse_results(result_path, 1)
        result = np.loadtxt(result_path, ndmin=2)[0]
        if args.backend == "context":
            assert result.shape == (24,) and np.isfinite(result).all() and result[1] == 0
            assert np.array_equal(result[6:11], row[:5])
            observed = compare_case(model, case, result[11:17], result[17:23])
            observed["saved_step_history"] = float(result[23])
        else:
            observed = compare_case(model, case, result[6:12], result[12:18])
            observed["suggested_step"] = float(result[18])
        observed.update(work_counts=result[2:6].astype(int).tolist(),
                        command=command, input_sha256=digest(input_path), output_sha256=digest(result_path))
        report["cases"].append(observed)
        (out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
        require_g1(observed)
        print(case["name"], max(observed["comparisons"]["q"]["normalized_error"]), flush=True)
    report["status"] = "g1-actual-error-passed"
    report["max_normalized_error"] = max(max(c["comparisons"][name]["normalized_error"])
                                          for c in report["cases"] for name in ("q", "q-minus-carry"))
    (out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print("G1 maximum normalized error:", report["max_normalized_error"])


if __name__ == "__main__": main()
