"""Prepare and reduce the 100-step nonreacting DCU strong/weak wall-clock matrix."""
import argparse
import json
from math import prod
from pathlib import Path
import re
from statistics import fmean

from check_dcu_m4_qualification import layouts
from prepare_dcu_m4_pair import reduce_run, reduce_wall_window
from prepare_dcu_weak_scaling import (CASES, NODE_BLOCK, TILES, case_configuration,
                                      placement_check, rank_blocks, shape_product, write_inputs)


WARMUP, TIMED = 5, 100
STRONG_GRID = {"explicit": 512, "compact": 448}
STRONG_TILES = [(1, 1, 1), (1, 1, 2), (1, 1, 4), (1, 2, 4), (2, 2, 4)]
INPUTS = ("input.case", "input.run", "input.output")


def make_config(scaling, kind, placement):
    mesh_tiles = (1, 1, 1) if scaling == "strong" else placement
    base = STRONG_GRID[kind] if scaling == "strong" else CASES[kind][0]
    mpi = shape_product(placement, NODE_BLOCK)
    cfg = case_configuration(kind, base, mesh_tiles, mpi, steps=WARMUP + TIMED)
    nodes = prod(placement)
    cfg.update(scaling=scaling, nodes=nodes, warmup=WARMUP, timed=TIMED, timing="wall_window",
               rank_blocks=rank_blocks(placement), repetitions=3 if scaling == "strong" or nodes == 1 else 2)
    cfg["environment"].update(ASTR_BENCHMARK_WALL_WINDOW=f"{WARMUP},{TIMED}", ASTR_GPU_RK_TIMING="0",
                              ASTR_COMPLETE_STEP_TIMING="0", ASTR_INSITU_TIMING="0")
    if kind == "explicit":
        cfg["environment"].update(ASTR_CPP_VISCOUS_HALO="separate", ASTR_CPP_VISCOUS_OVERLAP_AXIS="x")
    return cfg


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    configs = {}
    for scaling, placements in (("strong", STRONG_TILES), ("weak", TILES)):
        for kind in STRONG_GRID:
            for placement in placements:
                cfg = make_config(scaling, kind, placement)
                name = f'{scaling}_{kind}_{cfg["nodes"]:03d}nodes'
                write_inputs(root / "cases" / name, cfg)
                configs[name] = cfg
    (root / "campaign.json").write_text(json.dumps(configs, indent=2) + "\n")
    print(f"Prepared {len(configs)} nonreacting cases: {WARMUP} warmup + {TIMED} timed steps per run.")


def check_run(log, cfg):
    layouts(log, cfg["grid"], cfg["mpi"])
    if "ASTR_GPU_BENCHMARK_NO_FIELD_IO enabled" not in log:
        raise ValueError("field I/O was not disabled")
    cfl = [float(line.split()[-1]) for line in re.findall(r"^ASTR_CFL directional/local_sum/upper_bound=.+$", log, re.M)]
    if len(cfl) != cfg["timesteps"] or not all(0 < value < 1 for value in cfl):
        raise ValueError("incomplete or inadmissible CFL history")
    if cfg["kind"] == "explicit":
        timing = reduce_run(log, cfg, "regional")
    else:
        compact = re.findall(r"^ASTR_CPP_COMPACT transport/batch/explicit=([^\n]+)$", log, re.M)
        if ([tuple(map(int, row.split())) for row in compact] != [(1, 65535, 0)]
                or re.findall(r"^ASTR_CPP_COMPACT pipeline=(\d+)\b", log, re.M) != ["0"]):
            raise ValueError("not the frozen compact device PCR configuration")
        if re.search(r"^ASTR_CPP_COMPACT_PHASE_TIMING\b", log, re.M):
            raise ValueError("compact profiling was enabled in the wall window")
        timing = reduce_wall_window(log, cfg)
    return dict(timing, max_cfl=max(cfl))


def read_point(expected, path, cfg):
    repetitions = []
    for i in range(1, cfg["repetitions"] + 1):
        run = path / f"r{i}"
        for filename in INPUTS:
            relative = Path("datin") / filename
            if (run / relative).read_bytes() != (expected / relative).read_bytes():
                raise ValueError(f"{run}: unexpected {filename}")
        if any(run.rglob("*.h5")):
            raise ValueError(f"{run}: unexpected HDF5 output")
        repetitions.append(check_run((run / "run.log").read_text(), cfg))
    placement_check(cfg, json.loads((path / "layout.json").read_text()))
    checksum = (path / "astr.sha256").read_text().split()[0]
    if not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError(f"{path}: invalid binary SHA-256")
    seconds = fmean(run["seconds_per_step"] for run in repetitions)
    return dict(seconds_per_step=seconds, intervals_per_second=prod(cfg["grid"]) / seconds,
                binary_sha256=checksum, repetitions=repetitions)


def summarize(prepared, runs):
    configs = json.loads((prepared / "campaign.json").read_text())
    points, missing = {}, []
    for name, cfg in configs.items():
        path = runs / name
        absent = [str(path / f"r{i}/run.log") for i in range(1, cfg["repetitions"] + 1)
                  if not (path / f"r{i}/run.log").is_file()]
        missing.extend(absent)
        if not absent:
            points[name] = read_point(prepared / "cases" / name, path, cfg)
    for name, data in points.items():
        cfg = configs[name]
        anchor = points.get(f'{cfg["scaling"]}_{cfg["kind"]}_001nodes')
        if anchor is None:
            continue
        if data["binary_sha256"] != anchor["binary_sha256"]:
            raise ValueError(f"{name}: binary differs from the one-node reference")
        ratio = anchor["seconds_per_step"] / data["seconds_per_step"]
        if cfg["scaling"] == "strong":
            data.update(strong_speedup=ratio, strong_efficiency=ratio / cfg["nodes"])
        else:
            data["weak_efficiency"] = ratio
    return dict(cases=points, missing_logs=missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("prepare").add_argument("output", type=Path)
    summary = sub.add_parser("summarize")
    summary.add_argument("prepared", type=Path)
    summary.add_argument("runs", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.output)
    else:
        print(json.dumps(summarize(args.prepared, args.runs), indent=2))


if __name__ == "__main__":
    main()
