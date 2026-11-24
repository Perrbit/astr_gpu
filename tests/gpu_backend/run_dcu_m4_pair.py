"""Run M4 qualification and the AB/BA timing pair inside one existing allocation."""
import argparse
from collections import defaultdict
import importlib.util
import json
from math import prod
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK / "source/tests/gpu_backend"))
from check_dcu_m4_qualification import compare_stages, layouts
from prepare_dcu_m4_pair import check_progress, reduce_run, summarize
from prepare_dcu_weak_scaling import placement_check


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute(mapping, case, helper, variant):
    rank, local, host = load(helper).identity()
    row = read(mapping)[rank]
    if (rank, local, host) != (row["rank"], row["local"], row["host"]):
        raise ValueError(f"rank {rank}: unexpected host or DCU binding")
    os.environ["UCX_NET_DEVICES"] = row["port"]
    print(f'M4_BIND rank={rank} host={host} local={local} cpus={sorted(os.sched_getaffinity(0))} '
          f'dcu={row["bdf"]} port={row["port"]}', flush=True)
    binary = WORK / "bin" / f"astr_{variant}"
    os.chdir(case)
    os.execv(binary, [str(binary), "run"])


def save_layout(root, rows):
    write(root / "layout.json", rows)
    (root / "rankfile").write_text(''.join(f'rank {r["rank"]}={r["host"]} slot={r["core"]}\n' for r in rows))


def gate_layout(layout, config):
    by_host = defaultdict(list)
    for row in layout:
        by_host[row["host"]].append(row)
    rows = []
    for host in sorted(by_host)[:2]:
        for device in sorted(by_host[host], key=lambda row: row["local"])[:2]:
            rank = len(rows)
            rows.append(dict(device, rank=rank, coords=[rank % 2, rank // 2, 0]))
    placement_check(config, rows)
    return rows


def run(root, legacy, helper, probe, mpi):
    inputs = WORK / "inputs"
    cfg = read(inputs / "campaign.json")
    root.mkdir()
    write(root / "campaign.json", cfg)
    launch = [str(mpi), "--mca", "ras", "slurm", "--mca", "plm", "slurm", "--nooversubscribe"]
    mpi_size = str(prod(cfg["mpi"]))
    with (root / "inventory.log").open("w") as log:
        subprocess.run([*launch, "--map-by", "slot", "--rank-by", "slot", "--bind-to", "none", "-n", mpi_size,
                        sys.executable, "-B", str(legacy), "inventory", str(root), str(helper), str(probe)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    load(legacy).place(root, cfg, load(helper))
    all_ranks = read(root / "layout.json")
    gate = root / "gate-placement"
    gate.mkdir()
    gate_ranks = gate_layout(all_ranks, cfg["qualification"])
    save_layout(gate, gate_ranks)

    def measure(entry, config, placement, binding, require_window):
        name, variant = entry["name"], entry["variant"]
        case = root / name
        shutil.copytree(inputs / name, case)
        env = {k: v for k, v in os.environ.items() if not k.startswith("ASTR_")} | entry["environment"]
        print(f"M4_RUN {name} {variant}", flush=True)
        with (case / "run.log").open("w") as log:
            subprocess.run([*launch, "--rankfile", str(placement / "rankfile"), "-n", str(prod(config["mpi"])),
                            sys.executable, "-B", str(Path(__file__).resolve()), "execute", str(placement / "layout.json"),
                            str(case), str(helper), variant], env=env, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        output = (case / "run.log").read_text()
        bound = sorted((int(rank), int(device)) for rank, device in re.findall(
            r'^ASTR_GPU_CPP rank=(\d+) backend=gpufusion device=(\d+)\b', output, re.M))
        if bound != [(row["rank"], row["local"]) for row in binding]:
            raise ValueError(f"{name}: solver used different DCUs")
        layouts(output, config["grid"], config["mpi"])
        cfl = [float(line.split()[-1]) for line in re.findall(r"^ASTR_CFL directional/local_sum/upper_bound=.+$", output, re.M)]
        if len(cfl) != config["timesteps"] or not all(0 < u < 1 for u in cfl):
            raise ValueError(f"{name}: incomplete or inadmissible CFL")
        if require_window:
            if "ASTR_GPU_BENCHMARK_NO_FIELD_IO enabled" not in output or any(case.rglob("*.h5")):
                raise ValueError(f"{name}: field I/O was enabled")
            write(case / "summary.json", dict(reduce_run(output, config, variant), max_cfl=max(cfl)))
        else:
            check_progress(output, config, variant)

    q = cfg["qualification"]
    for entry in q["runs"]:
        measure(entry, q, gate, gate_ranks, False)
    result = compare_stages(root / "gate_a", root / "gate_b", q["grid"], q["mpi"], q["timesteps"])
    write(root / "qualification.json", result)
    print("M4_QUALIFICATION", json.dumps(result), flush=True)
    for entry in cfg["runs"]:
        measure(entry, cfg, root, all_ranks, True)
    result = summarize(root)
    write(root / "summary.json", result)
    print("M4_WALL_SUMMARY", json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name, names in (("run", ("root", "legacy", "helper", "probe", "mpi")),
                        ("execute", ("mapping", "case", "helper"))):
        cmd = sub.add_parser(name)
        for arg in names:
            cmd.add_argument(arg, type=Path)
        if name == "execute":
            cmd.add_argument("variant", choices=("baseline", "regional"))
    args = vars(parser.parse_args())
    action = args.pop("action")
    (run if action == "run" else execute)(**args)
