"""Replay one continuous AIR5 RK3 transport step from frozen old-astr output."""
import argparse
import json
import os
import sys
from math import prod
from pathlib import Path

import h5py
import numpy as np

from run_air5_filter_fields import command

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpu_validation"))
from air5_radau_reference import Air5RadauReference
from compare_q_validation_snapshots import read_q_snapshot


# Same state/RHS thresholds as run_air5_c5_frozen_transport_compare.sh.
STATE_ATOL, STATE_RTOL = 1.e-10, 1.e-10
RHS_ATOL, RHS_RTOL = 1.e-8, 2.e-10
DT = 2.e-10
CASES = (("wave", (1, 1, 1), 0, 0), ("diffusion", (2, 1, 1), 1, 0),
         ("normalshock", (2, 1, 1), 0, 1))


def snapshot(directory, label, stage, rank):
    return directory / f"air5.{label}.step00000000.rk{stage:02d}.rank{rank:08d}.bin"


def pack(directory, prefix, topology, floor, normal):
    with h5py.File(directory / "geometry.h5") as file:
        jac = file["jacob"][...]
        metric = np.stack([file[f"dxi{i}{j}"][...] for j in (1, 2, 3) for i in (1, 2, 3)])
    for rank in range(prod(topology)):
        q = read_q_snapshot(snapshot(directory, "initialized", 1, rank))
        local = np.array(q.header[:3])
        halo = q.header[3]
        whole = local * topology
        assert jac.shape == tuple(whole[::-1] + 1)
        offset = np.array(np.unravel_index(rank, topology)) * local
        indices = []
        for axis, (n, h, start) in enumerate(zip(local, whole, offset)):
            line = np.arange(start-halo, start+n+halo+1)
            indices.append(np.clip(line, 0, h) if normal and axis == 0 else
                           np.where((line < 0) | (line > h), line % h, line))
        index = np.ix_(*indices[::-1])
        with Path(f"{prefix}.rank{rank:08d}.bin").open("xb") as file:
            np.array(q.header[:4], dtype="<i4").tofile(file)
            np.r_[DT, floor].astype("<f8").tofile(file)
            q.values.tofile(file)
            jac[index].astype("<f8").tofile(file)
            metric[(slice(None), *index)].astype("<f8").tofile(file)


def check_final_cache(state, cache, model):
    density = state[0]
    velocity = state[1:4] / density
    species = state[5:10]
    dot = lambda table: np.einsum("s,s...->...", table, species)
    temperature = (state[4]-state[10]-0.5*np.sum(state[1:4]*velocity,axis=0)
                   -dot(model.formation_energy))/dot(model.cv_tr)
    pressure = temperature*dot(model.gas_constant)
    vibrating = model.theta_v > 0.
    theta = model.theta_v[vibrating, None, None, None]
    ev = np.sum(species[vibrating]*model.gas_constant[vibrating, None, None, None]*theta /
                np.expm1(theta/cache[6]), axis=0)
    expected = np.concatenate(([density], velocity, [pressure, temperature, state[10]]))
    actual = np.concatenate((cache[:6], [ev]))
    ratio = abs(actual-expected)/(STATE_ATOL+STATE_RTOL*abs(expected))
    assert np.isfinite(cache).all() and np.max(ratio) <= 1., float(np.max(ratio))
    return float(np.max(ratio))


def compare(directory, prefix, topology, normal, model):
    report = {}
    for stage in (1, 2, 3):
        worst = np.zeros(2)
        counts = np.zeros(2, dtype=int)
        cache_error = 0.
        for rank in range(prod(topology)):
            q = read_q_snapshot(snapshot(directory, "post_update", stage, rank))
            shape = q.header[:3][::-1]
            h = q.header[3]
            state = q.values.reshape(11, *(n+2*h+1 for n in shape))[:, h:-h, h:-h, h:-h]
            path = snapshot(directory, "full", stage, rank)
            np.testing.assert_array_equal(np.fromfile(path, "<i4", 4), (*q.header[:3], 11))
            rhs = np.fromfile(path, "<f8", offset=16).reshape(state.shape)
            with Path(f"{prefix}.rk{stage}.rank{rank:08d}.bin").open("rb") as stream:
                got = np.fromfile(stream, "<f8", 2*state.size).reshape(2, *state.shape)
                if normal:
                    nodes = prod(n+1 for n in shape)
                    mask = np.fromfile(stream, "i1", nodes)
                    old_mask = np.fromfile(snapshot(directory, "sensor", stage, rank), "i1", offset=12+8*nodes)
                    np.testing.assert_array_equal(mask, old_mask)
                    counts += np.bincount(mask, minlength=2)
                if stage == 3:
                    cache = np.fromfile(stream, "<f8", 7*state[0].size).reshape(7, *state.shape[1:])
                    cache_error = max(cache_error, check_final_cache(got[0], cache, model))
                assert stream.read(1) == b""
            assert np.isfinite(got).all()
            for column, (values, ref, atol, rtol) in enumerate(
                zip(got, (state, rhs), (STATE_ATOL, RHS_ATOL), (STATE_RTOL, RHS_RTOL))
            ):
                ratio = abs(values-ref)/(atol+rtol*abs(ref))
                idx = np.unravel_index(ratio.argmax(), ratio.shape)
                assert ratio[idx] <= 1., (stage, rank, ("q", "rhs")[column], idx, ratio[idx], ref[idx], values[idx])
                worst[column] = max(worst[column], ratio[idx])
        report[stage] = dict(state=float(worst[0]), rhs=float(worst[1]))
        if stage == 3:
            report[stage]["final_cache"] = cache_error
        if normal:
            assert np.all(counts > 0), counts
            report[stage]["sensor_mask_counts"] = counts.tolist()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "candidate", "mpiexec", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--memcheck", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir()
    model = Air5RadauReference(ROOT / "chemMech/air5_kimjo12.json")
    floor = [model.species_vibrational_energy(s, 300.) for s in range(5)]
    pinned = os.environ | dict(OMPI_MCA_pml="ob1", OMPI_MCA_btl="self,vader,tcp", OMPI_MCA_osc="pt2pt",
                              OMPI_MCA_coll="^cuda,hcoll,ucc", OMPI_MCA_opal_cuda_support="0")
    report = {}
    for case, topology, viscous, normal in CASES:
        work = out / case
        work.mkdir()
        ref = args.reference.resolve() / case / "old" / "validation"
        source = work / "input"
        pack(ref, source, topology, floor, normal)
        for mode in ((0, 1) if prod(topology) > 1 else (0,)):
            target = work / f"mode{mode}"
            launch = [args.mpiexec.resolve(), "--bind-to", "none", "-np", prod(topology)]
            cmd = [*launch, args.candidate.resolve(), source, target, *topology, viscous, mode, normal]
            env = pinned if mode == 0 else pinned | dict(OMPI_MCA_pml="ucx", OMPI_MCA_opal_cuda_support="1")
            command(cmd, work, f"mode{mode}.log", env)
            report[f"{case}/{mode}"] = dict(stages=compare(ref, target, topology, normal, model), command=list(map(str, cmd)))
            if mode:
                for file in work.glob("mode0.rk*.bin"):
                    other = work / file.name.replace("mode0.", "mode1.", 1)
                    assert file.read_bytes() == other.read_bytes(), file
            elif args.memcheck and prod(topology) > 1:
                checked = work / "memcheck"
                cuda_check = [args.memcheck.resolve(), "--tool", "memcheck", "--leak-check", "full", "--error-exitcode", "99"]
                check_cmd = [*launch, *cuda_check, args.candidate.resolve(), source, checked, *topology, viscous, mode, normal]
                command(check_cmd, work, "memcheck.log", env)
                report[f"{case}/{mode}"]["memcheck_stages"] = compare(ref, checked, topology, normal, model)
            (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"{case}/{mode} passed", flush=True)


if __name__ == "__main__":
    main()
