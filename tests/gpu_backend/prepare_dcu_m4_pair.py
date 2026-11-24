"""Prepare the fixed M4 paired workload and reduce one uninterrupted window per run."""
import argparse
import json
from math import isclose, isfinite
from pathlib import Path
import re
from statistics import fmean

from prepare_dcu_weak_scaling import TIMESTEP, case_configuration, rank_blocks, write_inputs


ORDER = (("a1", "baseline"), ("b1", "regional"), ("b2", "regional"), ("a2", "baseline"))
GATE = (("gate_a", "baseline"), ("gate_b", "regional"))
BASELINE_SOURCE = "a0e9217558e84850da41c59ec8ea82241d8748cd"
WINDOW = re.compile(r"^ASTR_BENCHMARK_WALL_WINDOW\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)$", re.M)


def prepare(output):
    config = case_configuration("explicit", 488, (2, 2, 2), (4, 4, 2), steps=105)
    config.update(warmup=5, timed=100, timing="wall_window", rank_blocks=rank_blocks((2, 2, 2)))
    config["environment"].update(ASTR_GPU_RK_TIMING="0", ASTR_COMPLETE_STEP_TIMING="0", ASTR_INSITU_TIMING="0",
                                 ASTR_BENCHMARK_WALL_WINDOW="5,100")
    output.mkdir(parents=True, exist_ok=False)
    runs = []
    for name, variant in ORDER:
        source = config["environment"].copy()
        if variant == "regional":
            source.update(ASTR_CPP_VISCOUS_HALO="separate", ASTR_CPP_VISCOUS_OVERLAP_AXIS="x")
        runs.append(dict(name=name, variant=variant, environment=source))
        write_inputs(output / name, config)
    config.update(baseline_source=BASELINE_SOURCE, runs=runs)
    gate = case_configuration("explicit", 16, (1, 1, 1), (2, 2, 1), steps=4)
    gate.update(grid=(17, 19, 16), nodes=2, rank_blocks=[[0, 1], [2, 3]])
    gate["environment"].update(ASTR_GPU_RK_TIMING="0", ASTR_GPU_BENCHMARK_NO_FIELD_IO="0",
                               ASTR_VALIDATION_RHS_PREFIX="outdat/rhs", ASTR_VALIDATION_RHS_STEP="0",
                               ASTR_VALIDATION_RHS_STEP_END="3")
    gate["runs"] = []
    for name, variant in GATE:
        source = gate["environment"].copy()
        if variant == "regional":
            source.update(ASTR_CPP_VISCOUS_HALO="separate", ASTR_CPP_VISCOUS_OVERLAP_AXIS="x")
        gate["runs"].append(dict(name=name, variant=variant, environment=source))
        write_inputs(output / name, gate)
    config["qualification"] = gate
    (output / "campaign.json").write_text(json.dumps(config, indent=2) + "\n")


def check_progress(log, config, variant):
    expected = "z" if variant == "baseline" else "xyz"
    axes = re.findall(r"^ASTR_GPU_CPP viscous_overlap_axis=(\w+)$", log, re.M)
    if axes != [expected]:
        raise ValueError(f"unexpected executed viscous schedule for {variant}: {axes}")
    if any(re.search(rf"^{prefix} ", log, re.M) for prefix in ("ASTR_GPU_RANK_RK_TIMING", "ASTR_CPP_HALO_TIMING",
                                                                  "ASTR_COMPLETE_STEP_TIMING")):
        raise ValueError("per-step profiling was enabled in the wall window")
    clocks = TIMESTEP.findall(log)
    dt = config["dt"]
    if len(clocks) != config["timesteps"] or any(int(step) != idx or float(used) != dt or float(next_dt) != dt
            or not isclose(float(start), (idx - 1) * dt, rel_tol=1e-12)
            or not isclose(float(end), idx * dt, rel_tol=1e-12)
            for idx, (step, start, end, used, next_dt) in enumerate(clocks, 1)):
        raise ValueError("wrong fixed-step clock or incomplete run")


def reduce_run(log, config, variant):
    check_progress(log, config, variant)
    records = WINDOW.findall(log)
    if len(records) != 1:
        raise ValueError("expected exactly one rank-zero wall window record")
    warmup, timed, elapsed, per_step = records[0]
    if (int(warmup), int(timed)) != (config["warmup"], config["timed"]):
        raise ValueError("unexpected timing window")
    elapsed, per_step = float(elapsed), float(per_step)
    if not isfinite(elapsed) or elapsed <= 0 or not isclose(per_step, elapsed / int(timed), rel_tol=1e-13):
        raise ValueError("invalid timing record")
    return dict(window_seconds=elapsed, seconds_per_step=per_step)


def summarize(output):
    config = json.loads((output / "campaign.json").read_text())
    measurements = {name: reduce_run((output / name / "run.log").read_text(), config, variant) for name, variant in ORDER}
    a1, b1, b2, a2 = (measurements[name]["seconds_per_step"] for name, _ in ORDER)
    a, b = fmean((a1, a2)), fmean((b1, b2))
    return dict(runs=measurements, baseline_mean=a, regional_mean=b, speedup=a / b, fractional_time_reduction=1 - b / a,
                paired_seconds_saved=[a1 - b1, a2 - b2], baseline_repeat_spread=abs(a1 - a2),
                regional_repeat_spread=abs(b1 - b2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "summarize"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.output)
    else:
        print(json.dumps(summarize(args.output), indent=2))


if __name__ == "__main__":
    main()
