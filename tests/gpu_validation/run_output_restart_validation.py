"""Bounded real-solver TGV acceptance for completed-step checkpoint restart."""
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys

import h5py
import numpy as np


def run_case(args, root, backend, ranks, name, steps, restore=None, enabled=True,
             reject=None, change_input=False):
    case = args.output / f"{backend}_np{ranks}_{name}"
    subprocess.run([
        sys.executable, str(root / "tests/gpu_validation/prepare_tgv_case.py"),
        "--src-case", str(root / "examples/Taylor_Green_Vortex"), "--dst-case", str(case),
        "--use-gpu", "t" if backend == "gpu" else "f", "--grid", "16,16,16",
        "--maxstep", str(steps - 1), "--feqchkpt", "1", "--deltat", "1.d-3",
        "--lfilter", "t", "--diffterm", "t", "--scheme", "643e"], check=True)
    (case / "outdat/new").mkdir(parents=True)
    config = case / "datin/input.output"
    config.write_text(f"""&output
 directory='outdat/new', restore_directory='{restore or ''}',
 host_budget_bytes=67108864,device_budget_bytes=67108864,buffer_bytes=67108864
/
&checkpoint
 enabled={'.true.' if enabled else '.false.'},mode='{args.mode}',
 interval_steps={5 if args.mode == 'steps' else 0},interval_time={0.004 if args.mode == 'time' else 0},keep=2
/
&volume
 enabled=.false.
/
&slices
 enabled=.false.
/
""")
    if change_input:
        with (case / "datin/input.tgv").open("a") as stream:
            stream.write("\n! changed primary input identity\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("ASTR_")}
    env.update(ASTR_OUTPUT_CONFIG="datin/input.output", ASTR_FORCE_MPI_TOPOLOGY=f"{ranks},1,1",
               ASTR_GPU_SYNC_MODE="explicit", ASTR_GPU_HALO_TRANSPORT="pinned",
               ASTR_GPU_PRECISION_MODE="fp64", ASTR_INSITU_SAMPLE_PREFIX="outdat/sample")
    command = [str(args.mpiexec), "--mca", "coll_hcoll_enable", "0", "-np", str(ranks),
               str(args.executable), "run", "datin/input.tgv"]
    with (case / "run.log").open("wb") as log:
        process = subprocess.Popen(command, cwd=case, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=180)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                process.wait()
            raise
    if reject:
        if returncode == 0 or reject not in (case / "run.log").read_text():
            raise AssertionError(f"expected rejection not observed: {reject}: {case}")
        if list((case / "outdat/new").rglob("COMPLETE")):
            raise AssertionError("failed restore published a checkpoint")
    elif returncode:
        raise RuntimeError(f"solver failed: {case / 'run.log'}")
    if (case / "outdat/flowfield.h5").exists():
        raise AssertionError("new output emitted a legacy flowfield file")
    disk_bytes = sum(p.stat().st_size for p in case.rglob("*") if p.is_file())
    if disk_bytes > 64 * 1024**2:
        raise RuntimeError(f"test directory budget exceeded: {case}: {disk_bytes}")
    return case, disk_bytes


def compare_fields(left, right):
    with h5py.File(left, "r") as a, h5py.File(right, "r") as b:
        if set(a) != set(b):
            raise AssertionError("dataset inventory mismatch")
        differences = {}
        for name in a:
            aa, bb = a[name][:], b[name][:]
            if aa.shape != bb.shape or aa.dtype != bb.dtype or aa.tobytes() != bb.tobytes():
                differences[name] = float(np.max(np.abs(aa.astype(float) - bb.astype(float))))
        if differences:
            raise AssertionError(f"exact state mismatch: {differences}")
        return sorted(a)


def compare_statistics(continuous, resumed, disabled, backend):
    names = ["flowstate.dat"]
    if backend == "gpu":
        names += ["gpu_kenergy.dat", "gpu_enstophy.dat", "gpu_dissipation.dat"]
    counts = {}
    for name in names:
        full = np.loadtxt(continuous / name, skiprows=1, ndmin=2)
        tail = np.loadtxt(resumed / name, skiprows=1, ndmin=2)
        off = np.loadtxt(disabled / name, skiprows=1, ndmin=2)
        expected = full[full[:, 0] >= 5]
        if full.tobytes() != off.tobytes() or expected.tobytes() != tail.tobytes():
            raise AssertionError(f"statistics mismatch: {name}")
        counts[name] = len(tail)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("executable", "mpiexec", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--backends", nargs="+", choices=("cpu", "gpu"), default=["cpu", "gpu"])
    parser.add_argument("--ranks", nargs="+", type=int, choices=(1, 2), default=[1, 2])
    parser.add_argument("--mode", choices=("steps", "time"), default="steps")
    parser.add_argument("--no-samples", action="store_true",
                        help="For non-testing builds: skip test-only per-step field dumps; still compare final restart states")
    args = parser.parse_args()
    args.executable = args.executable.resolve(strict=True)
    args.mpiexec = args.mpiexec.resolve(strict=True)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    report = {"status": "running", "checks": []}
    try:
        for backend in args.backends:
            for ranks in args.ranks:
                continuous, a_size = run_case(args, root, backend, ranks, "continuous", 12)
                first, b_size = run_case(args, root, backend, ranks, "first", 5)
                source = first / "outdat/new/checkpoints/step000000000005"
                resumed, c_size = run_case(args, root, backend, ranks, "resumed", 12, restore=source)
                final = "outdat/new/checkpoints/step000000000012/state.h5"
                datasets = compare_fields(continuous / final, resumed / final)
                control = "outdat/new/checkpoints/step000000000012/control.bin"
                if (continuous / control).read_bytes() != (resumed / control).read_bytes():
                    raise AssertionError("final control/schedule state mismatch")
                compare_fields(continuous / "outdat/new/resources/geometry.h5",
                               resumed / "outdat/new/resources/geometry.h5")
                disabled, d_size = run_case(args, root, backend, ranks, "disabled", 12, enabled=False)
                statistics = compare_statistics(continuous, resumed, disabled, backend)
                expected_batches = ([10, 12] if args.mode == "steps" else [8, 12])
                for case in (continuous, resumed):
                    actual_batches = sorted(p.name for p in (case / "outdat/new/checkpoints").glob("step*"))
                    if actual_batches != [f"step{step:012d}" for step in expected_batches]:
                        raise AssertionError(f"unexpected retained checkpoints: {actual_batches}")
                sample_count = 0
                for step in ([] if args.no_samples else range(1, 13)):
                    for rank in range(ranks):
                        filename = f"sample.step{step:08d}.rank{rank:08d}.bin"
                        actual = (continuous / "outdat" / filename).read_bytes()
                        if actual != (disabled / "outdat" / filename).read_bytes():
                            raise AssertionError(f"output changed completed sample: {filename}")
                        if step > 5 and actual != (resumed / "outdat" / filename).read_bytes():
                            raise AssertionError(f"restart changed completed sample: {filename}")
                        sample_count += 1
                report["checks"].append(dict(backend=backend, np=ranks, mode=args.mode, datasets=datasets,
                                              output_switch_field_check=not args.no_samples,
                                              samples=sample_count, statistics=statistics,
                                              bytes=[a_size, b_size, c_size, d_size]))
                if backend == "cpu" and ranks == 2:
                    run_case(args, root, backend, ranks, "reject_input", 12, restore=source,
                             reject="numerical/executable/controller contract mismatch", change_input=True)
                    corrupt = args.output / f"{backend}_np{ranks}_corrupt_resource"
                    shutil.copytree(first / "outdat/new", corrupt)
                    with (corrupt / "resources/input.txt").open("ab") as stream:
                        stream.write(b"changed")
                    run_case(args, root, backend, ranks, "reject_resource", 12,
                             restore=corrupt / "checkpoints/step000000000005",
                             reject="invalid new checkpoint bundle")
                    report["checks"][-1]["rejections"] = ["primary_input_mismatch", "resource_corruption"]
                (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
