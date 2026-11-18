"""Check complete AIR5 output and same-topology restart of ordinary C++ astr."""
import argparse
import json
import os
from math import prod
from pathlib import Path

import h5py
import numpy as np

from run_air5_filter_fields import command
from run_cpp_air5_strang import prepare
from run_output_restart_validation import compare_fields


FIELDS = ("density", "velocity_x", "velocity_y", "velocity_z", "pressure", "temperature",
          "vibrational_temperature", "mass_fraction_N2", "mass_fraction_O2", "mass_fraction_N",
          "mass_fraction_O", "mass_fraction_NO")


def checkpoint(work, step):
    return work / f"outdat/new/checkpoints/step{step:012d}/state.h5"


def archive_frame(work, kind, step):
    pattern = f"step{step:012d}.h5" if kind == "slice_products" else f"step{step:012d}/data.h5"
    frames = list((work / f"outdat/new/{kind}").rglob(pattern))
    assert len(frames) == 1, frames
    return frames[0]


def check_archives(work, step):
    with (h5py.File(checkpoint(work, step)) as state,
          h5py.File(archive_frame(work, "fields", step)) as volume,
          h5py.File(archive_frame(work, "slice_products", step)) as plane):
        for component, name in enumerate(FIELDS, start=23):
            values = state[f"q{component:04d}"][...]
            assert np.isfinite(values).all(), name
            assert volume[name][...].tobytes() == values.tobytes(), name
            assert plane[name][...].tobytes() == values[..., 7].tobytes(), name
    return len(FIELDS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "mpiexec", "input", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--topology", type=int, nargs=3, default=(2, 1, 1))
    parser.add_argument("--transport", choices=("pinned", "device"), default="pinned")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir()
    env = os.environ | dict(OMPI_MCA_pml="ob1", OMPI_MCA_btl="self,vader,tcp", OMPI_MCA_osc="pt2pt",
                            OMPI_MCA_coll="^cuda,hcoll,ucc", OMPI_MCA_opal_cuda_support="0",
                            ASTR_GPU_HALO_TRANSPORT=args.transport, ASTR_GPU_SYNC_MODE="explicit",
                            ASTR_FORCE_MPI_TOPOLOGY=",".join(map(str, args.topology)),
                            ASTR_AIR5_C4_CONSERVATION="f")
    if args.transport == "device":
        env |= dict(OMPI_MCA_pml="ucx", OMPI_MCA_opal_cuda_support="1")
    launch = [args.mpiexec.resolve(), "--bind-to", "none", "-np", prod(args.topology),
              args.binary.resolve(), "run", "datin/input.case"]

    def run(name, steps, archive=True, restore=None, exact=True, new_dt=None):
        work = root / name
        prepare(work, args.input.resolve(), steps)
        primary = work / "datin/input.case"
        primary.write_text(primary.read_text().replace("&chemistry\n", "&chemistry\n  step_history = 'previous'\n"))
        if new_dt is not None:
            runtime = work / "datin/input.run"
            lines = runtime.read_text().splitlines()
            runtime.write_text("\n".join(f"  fixed_dt = {new_dt}" if "fixed_dt =" in line else line
                                         for line in lines) + "\n")
        (work / "outdat/new").mkdir(parents=True)
        enabled = "t" if archive else "f"
        (work / "datin/input.output").write_text(f"""&output
 directory='outdat/new',restore_directory='{restore or ''}',exact_restart={'t' if exact else 'f'},
 host_budget_bytes=-1,device_budget_bytes=-1,buffer_bytes=16384
/
&checkpoint
 enabled=t,mode='steps',interval_steps=2,keep=2,initial_frame=f
/
&volume
 enabled={enabled},mode='steps',interval_steps=1
/
&slices
 enabled=f
/
""")
        catalog = [dict(catalog_schema="astr-insitu-products-v1")]
        if archive:
            catalog.append(dict(id="air5_i7", kind="slice", enable_request=1, source="instantaneous", fields=FIELDS,
                                sampling=dict(mode="grid_index", axis="i", index=7, index_ranges=dict(j=[0, 16], k=[0, 16])),
                                output=dict(channel="data", format="hdf5-xdmf"),
                                schedule=dict(mode="steps", interval=1)))
        (work / "input.products.jsonl").write_text("".join(json.dumps(item)+"\n" for item in catalog))
        probe = {"ASTR_CHECKPOINT_TEST_RESTORE_PROBE": "1"} if restore else {}
        command(launch, work, "run.log", env | probe)
        assert "The job is done!" in (work / "run.log").read_text(errors="replace"), work
        return work

    uninterrupted = run("continuous", 4)
    seed = run("seed", 2)
    seed_path = checkpoint(seed, 2)
    with h5py.File(seed_path) as state:
        assert state["identity"][6] == 35 and state["identity"][12] == 7
        assert state["metadata"][:].tolist() == [1, 1]
        assert any(np.count_nonzero(state[f"q{c:04d}"][...]) for c in range(12, 23))
        assert np.count_nonzero(state["q0035"][...])
    resumed = run("resumed", 4, restore=seed_path.parent)
    compare_fields(seed_path, resumed / "outdat/restore_probe.h5")
    datasets = compare_fields(checkpoint(uninterrupted, 4), checkpoint(resumed, 4))
    for kind in ("fields", "slice_products"):
        compare_fields(archive_frame(uninterrupted, kind, 4), archive_frame(resumed, kind, 4))
    bare = run("without_archives", 4, archive=False)
    compare_fields(checkpoint(uninterrupted, 4), checkpoint(bare, 4))
    count = check_archives(uninterrupted, 4)
    nonexact = run("nonexact", 3, restore=seed_path.parent, exact=False, new_dt="2.d-10")
    compare_fields(seed_path, nonexact / "outdat/restore_probe.h5")
    with h5py.File(checkpoint(nonexact, 3)) as state:
        assert state["identity"][12] == 7 and state["identity"][6] == 35
        assert np.asarray(state["identity"][10], dtype="<i8").view("<f8").item() == 2.e-10
    report = dict(topology=args.topology, transport=args.transport, fields=count, state_datasets=len(datasets),
                  exact_restore=True, exact_continuation=True, output_isolation=True, nonexact_retains_full_state=True,
                  nonexact_dt=2.e-10, command=list(map(str, launch)))
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(root / "report.json")


if __name__ == "__main__":
    main()
