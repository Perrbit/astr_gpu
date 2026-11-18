"""Compare coupled ordinary astr runs with the CUDA Fortran reference."""
import argparse
import json
import os
import re
import shutil
import sys
from math import prod
from pathlib import Path

import numpy as np

from run_air5_filter_fields import command

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpu_validation"))
from compare_q_validation_snapshots import compare_snapshot_sets, comparison_values, read_q_snapshot


PHASES = {"pre_chemistry": (1,), "post_chemistry": (1, 2), "post_update": (1, 2, 3), "post_transport": (1,)}


def prepare(destination, source, steps):
    shutil.copytree(source, destination / "datin")
    control = destination / "datin/input.run"
    control.write_text(re.sub(r"(?m)^  max_steps = [0-9]+$", f"  max_steps = {steps}", control.read_text()))
    (destination / "validation").mkdir()


def reference_check(old, new, ranks, steps):
    phases = tuple(PHASES)
    reference = old / "validation/air5"
    candidate = new / "validation/air5"
    result = compare_snapshot_sets(reference, candidate, phases, 1.e-10, 1.e-10, active_only=True)
    assert result.file_count == steps*ranks*sum(map(len, PHASES.values())), result.file_count
    failures = [x for x in result.comparisons if not x.passed]
    assert not failures, failures[:3]
    source_change = 0.
    for rank in range(ranks):
        def state(label, phase):
            suffix = f"{label}.step00000000.rk{phase:02d}.rank{rank:08d}.bin"
            return comparison_values(read_q_snapshot(Path(f"{candidate}.{suffix}")), True)
        source_change = max(source_change, float(np.max(abs(state("post_chemistry", 1)-state("pre_chemistry", 1)))))
    assert source_change > 0., "the nominally coupled run did not exercise a nonzero source update"
    logs = [(p / "run.log").read_text(errors="replace") for p in (old, new)]
    counts = [int(x) for x in re.findall(r"(?m)^AIR5_CHEMISTRY half=[12].* max_acc/rej/rhs/jac=\s*([0-9]+)", logs[1])]
    assert len(counts) >= 2 and all(x > 0 for x in counts), counts
    timesteps = [[line for line in log.splitlines() if line.startswith("ASTR_TIMESTEP complete_step=")] for log in logs]
    assert timesteps[0] == timesteps[1] and len(timesteps[0]) == steps, timesteps
    return dict(files=result.file_count, max_abs=result.max_abs, max_scaled=result.max_scaled,
                first_half_max_abs_update=source_change, chemistry_max_accepted=counts, timesteps=timesteps[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "candidate", "mpiexec", "input", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--topology", type=int, nargs=3, default=(1, 1, 1))
    parser.add_argument("--transport", choices=("pinned", "device"), default="pinned")
    parser.add_argument("--primitive-reuse", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir()
    ranks = prod(args.topology)
    env = os.environ | dict(OMPI_MCA_pml="ob1", OMPI_MCA_btl="self,vader,tcp", OMPI_MCA_osc="pt2pt",
                            OMPI_MCA_coll="^cuda,hcoll,ucc", OMPI_MCA_opal_cuda_support="0",
                            ASTR_FORCE_MPI_TOPOLOGY=",".join(map(str, args.topology)),
                            ASTR_VALIDATION_RHS_PREFIX="validation/air5", ASTR_GPU_SYNC_MODE="explicit",
                            ASTR_AIR5_C4_CONSERVATION="f", ASTR_VALIDATION_RHS_STEP="0",
                            ASTR_VALIDATION_RHS_STEP_END=str(args.steps-1),
                            ASTR_GPU_HALO_TRANSPORT=args.transport)
    if args.transport == "device":
        env |= dict(OMPI_MCA_pml="ucx", OMPI_MCA_opal_cuda_support="1")
    commands = {}
    cases = [("old", args.reference), ("new", args.candidate)]
    if args.primitive_reuse:
        cases.append(("full_refresh", args.candidate))
    for name, exe in cases:
        work = out / name
        prepare(work, args.input.resolve(), args.steps)
        cmd = [args.mpiexec.resolve(), "--bind-to", "none", "-np", ranks, exe.resolve(), "run", "datin/input.case"]
        commands[name] = list(map(str, cmd))
        transport = "device-aware" if name == "old" and args.transport == "device" else args.transport
        reuse = "chemistry" if args.primitive_reuse and name != "full_refresh" else "off"
        command(cmd, work, "run.log", env | dict(ASTR_GPU_HALO_TRANSPORT=transport, ASTR_AIR5_PRIMITIVE_REUSE=reuse))
        assert "The job is done!" in (work / "run.log").read_text(errors="replace"), work
    assert "AIR5 resident RK3; Strang=T" in (out / "new/run.log").read_text(errors="replace")
    report = reference_check(out / "old", out / "new", ranks, args.steps)
    if args.primitive_reuse:
        lines = (out / "new/run.log").read_text()
        refreshed = re.findall(r"ASTR_AIR5_PRIMITIVE_SHELL rank=(\d+) step=(\d+) half=(\d+) "
                               r"recovered_nodes=(\d+) physical_nodes=(\d+)", lines)
        assert {int(row[0]) for row in refreshed} == set(range(ranks))
        assert {int(row[2]) for row in refreshed} == {1, 2}
        assert all(0 < int(row[3]) < int(row[4]) for row in refreshed)
        assert "ASTR_AIR5_PRIMITIVE_SHELL" not in (out / "full_refresh/run.log").read_text()
        comparisons = 0
        for label, phases in PHASES.items():
            for stage in phases:
                for step in range(args.steps):
                    for rank in range(ranks):
                        name = f"air5.{label}.step{step:08d}.rk{stage:02d}.rank{rank:08d}.bin"
                        actual = comparison_values(read_q_snapshot(out/"new/validation"/name), True)
                        baseline = comparison_values(read_q_snapshot(out/"full_refresh/validation"/name), True)
                        assert actual.tobytes() == baseline.tobytes(), name
                        comparisons += 1
        report.update(reuse_phase_comparisons=comparisons, reuse_phase_words_exact=True,
                      shell_nodes=[[int(value) for value in row] for row in refreshed])
    (out / "report.json").write_text(json.dumps(report | dict(commands=commands), indent=2) + "\n")
    print(out / "report.json")


if __name__ == "__main__":
    main()
