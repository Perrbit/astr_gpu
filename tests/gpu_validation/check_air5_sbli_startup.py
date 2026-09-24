#!/usr/bin/env python3
"""Check bounded SBLI startup states and prescribed top BC; no physical-pass claim."""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from air5_radau_reference import Air5RadauReference
from check_air5_c5_normal_shock import primitive_metrics
from check_air5_c5_hbl import _load_parallel_layout
from check_air5_numq11_shock_tube import q_array, read_sensor_mask, snapshot_rank
from compare_q_validation_snapshots import read_q_snapshot


def compare_sensors(cpu, gpu):
    files = [sorted((p / "validation").glob("air5.sensor.*.bin")) for p in (cpu, gpu)]
    if not files[0] or [p.name for p in files[0]] != [p.name for p in files[1]]:
        raise ValueError("CPU/GPU sensor snapshot sets differ or are empty")
    maximum = 0.
    for left, right in zip(*files):
        a, ma = read_sensor_mask(left)
        b, mb = read_sensor_mask(right)
        np.testing.assert_allclose(a, b, atol=2e-11, rtol=2e-10,
                                   err_msg=f"raw sensor mismatch: {left.name}")
        np.testing.assert_array_equal(ma, mb, err_msg=f"shock mask mismatch: {left.name}")
        maximum = max(maximum, float(abs(a-b).max()))
    return dict(files=len(files[0]), maximum_raw_difference=maximum, masks_equal=True)


def check(case, topology):
    tx, ty, tz = topology
    meta = json.loads((case / "incident_shock_metadata.json").read_text())
    intervals = tuple(map(int, meta["grid"].split(",")))
    layouts = _load_parallel_layout(case / "datin/parallel.info")
    if set(layouts) != set(range(tx*ty*tz)):
        raise ValueError("MPI layout rank count does not match topology")
    thermo = Air5RadauReference(Path(__file__).resolve().parents[2] / "chemMech/air5_kimjo12.json")
    ev_bounds = [np.array([thermo.species_vibrational_energy(s, t) for s in range(5)])
                 for t in thermo.temperature_bounds]
    metrics = dict(minimum_density=float("inf"), minimum_species_density=float("inf"),
        minimum_temperature=float("inf"), maximum_temperature=0., mass_closure=0.,
        top_max_scaled_error=0., wall_temperature_error=0., wall_speed=0.,
        marked_sensor_entries=0, snapshots=0)
    ranks = set()
    top_states_seen = set()
    for label in ("post_chemistry", "pre_rhs", "post_update", "post_transport"):
        files = sorted((case / "validation").glob(f"air5.{label}.*.bin"))
        if not files:
            raise ValueError(f"missing {label} snapshots")
        for path in files:
            rank = snapshot_rank(path)
            ranks.add(rank)
            snapshot = read_q_snapshot(path)
            im, jm, km, hm, _ = snapshot.header
            q = q_array(snapshot)
            active = q[hm:hm+im+1, hm:hm+jm+1, hm:hm+km+1]
            rho, species, temp, pressure, _ = primitive_metrics(active, thermo)
            if np.min(rho) <= 0 or np.min(species) < 0:
                raise ValueError(f"{path}: negative accepted species or invalid density")
            if np.min(temp) < thermo.temperature_bounds[0] or np.max(temp) > thermo.temperature_bounds[1]:
                raise ValueError(f"{path}: translational temperature outside model domain")
            if np.min(pressure) < thermo.pressure_bounds[0] or np.max(pressure) > thermo.pressure_bounds[1]:
                raise ValueError(f"{path}: pressure outside model domain")
            if np.any(active[..., 10] < species @ ev_bounds[0]) or np.any(active[..., 10] > species @ ev_bounds[1]):
                raise ValueError(f"{path}: vibrational energy outside model domain")
            metrics["snapshots"] += 1
            metrics["minimum_density"] = min(metrics["minimum_density"], float(rho.min()))
            metrics["minimum_species_density"] = min(metrics["minimum_species_density"], float(species.min()))
            metrics["minimum_temperature"] = min(metrics["minimum_temperature"], float(temp.min()))
            metrics["maximum_temperature"] = max(metrics["maximum_temperature"], float(temp.max()))
            metrics["mass_closure"] = max(metrics["mass_closure"], float(np.max(abs(species.sum(axis=-1)-rho)/rho)))
            if label != "pre_rhs":
                continue
            layout = layouts[rank]
            if (im, jm, km) != (layout.im, layout.jm, layout.km):
                raise ValueError("snapshot extent differs from recorded MPI layout")
            x = (layout.i0+np.arange(im+1))*meta["domain"][0]/intervals[0]
            side = (x >= meta["top_x"]).astype(int)
            expected = np.array([meta["upstream_q"], meta["downstream_q"]])[side]
            if layout.j0+jm == intervals[1]:
                top_states_seen.update(side.tolist())
                top = q[hm:hm+im+1, hm+jm:hm+jm+hm+1, hm:hm+km+1]
                scale = np.maximum(abs(expected), 1.)
                error = np.max(abs(top-expected[:, None, None, :])/scale[:, None, None, :])
                metrics["top_max_scaled_error"] = max(metrics["top_max_scaled_error"], float(error))
            if layout.j0 == 0:
                wall = active[:, 0]
                _, _, wall_t, _, _ = primitive_metrics(wall, thermo)
                metrics["wall_temperature_error"] = max(metrics["wall_temperature_error"], float(abs(wall_t-meta["wall_temperature"]).max()))
                metrics["wall_speed"] = max(metrics["wall_speed"], float(abs(wall[..., 1:4]/wall[..., :1]).max()))
    if ranks != set(range(tx*ty*tz)) or top_states_seen != {0, 1}:
        raise ValueError("missing ranks or an incident top state")
    if metrics["mass_closure"] > 2e-12 or metrics["top_max_scaled_error"] > 2e-11:
        raise ValueError(f"closure/top boundary gate failed: {metrics}")
    if metrics["wall_temperature_error"] > 1e-8 or metrics["wall_speed"] > 1e-10:
        raise ValueError(f"wall boundary gate failed: {metrics}")
    for path in (case / "validation").glob("air5.sensor.*.bin"):
        _, mask = read_sensor_mask(path)
        metrics["marked_sensor_entries"] += int(np.count_nonzero(mask))
    if not metrics["marked_sensor_entries"]:
        raise ValueError("no shock sensor activation")
    logs = list(case.glob("*.log"))
    cfl = [float(v) for p in logs for v in re.findall(r"current CFL:\s*(\S+)", p.read_text())]
    if not cfl or not np.isfinite(cfl).all() or max(cfl) >= 1:
        raise ValueError(f"missing/invalid CFL: {cfl}")
    metrics.update(status="startup-contract-pass-not-physical-pass", max_cfl=max(cfl))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--topology", default="1,1,1")
    parser.add_argument("--compare-sensors-with", type=Path)
    args = parser.parse_args()
    result = check(args.case, tuple(map(int, args.topology.split(","))))
    if args.compare_sensors_with:
        result["sensor_comparison"] = compare_sensors(args.compare_sensors_with, args.case)
    (args.case / "startup_contract.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
