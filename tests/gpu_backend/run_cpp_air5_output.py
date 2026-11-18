"""Check complete AIR5 output and same-topology restart of ordinary C++ astr."""
import argparse
import json
import os
import struct
from subprocess import CalledProcessError
from math import prod
from pathlib import Path

import h5py
import numpy as np

from run_air5_filter_fields import command
from run_cpp_air5_strang import comparison_values, prepare, read_q_snapshot
from run_output_restart_validation import compare_fields


FIELDS = ("density", "velocity_x", "velocity_y", "velocity_z", "pressure", "temperature",
          "vibrational_temperature", "mass_fraction_N2", "mass_fraction_O2", "mass_fraction_N",
          "mass_fraction_O", "mass_fraction_NO")
MEAN_FIELDS = {
    "reynolds": dict(zip(FIELDS[:5], (1, 28, 29, 30, 5))),
    "favre": dict(zip(FIELDS[1:4] + (FIELDS[5],), (2, 3, 4, 6))),
}


def checkpoint(work, step):
    return work / f"outdat/new/checkpoints/step{step:012d}/state.h5"


def archive_frame(work, kind, step, product="air5_i7"):
    pattern = f"step{step:012d}.h5" if kind == "slice_products" else f"step{step:012d}/data.h5"
    directory = work / f"outdat/new/{kind}"
    if kind == "slice_products":
        directory /= product
    frames = list(directory.rglob(pattern))
    assert len(frames) == 1, frames
    return frames[0]


def statistics(work, step):
    return checkpoint(work, step).with_name("statistics.h5")


def check_statistics_window(path, start, steps, duration):
    with h5py.File(path) as state:
        assert state["identity"][6] == 58 and state["identity"][12] == 6
        meta = state["metadata"][:]
        assert meta[:4].tolist() == [3, steps - start + 1, start, steps], meta
        assert meta[5:7].tolist() == [1, 1], meta
        assert meta[8:9].view(np.float64)[0] == duration, meta
        mass_integral = state["q0001"][...]
        if duration > 0.:
            assert np.all(mass_integral > 0.)
        else:
            assert not np.any(mass_integral)


def compare_mean_reference(candidate, reference):
    with h5py.File(candidate) as new, h5py.File(reference) as old:
        for name in ("identity", "metadata", "partitions"):
            np.testing.assert_array_equal(new[name][...], old[name][...], err_msg=name)
        errors = {}
        for index in range(1, 59):
            name = f"q{index:04d}"
            a, b = new[name][...], old[name][...]
            errors[name] = float(np.max(np.abs(a - b)) / max(1., np.max(np.abs(b))))
        return errors


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


def check_mean_slices(continuous, resumed, step):
    count = 0
    with h5py.File(statistics(continuous, step)) as state:
        measure = state["metadata"][8:9].view(np.float64)[0]
        for source, fields in MEAN_FIELDS.items():
            product = f"air5_i7_{source}"
            frame = archive_frame(continuous, "slice_products", step, product)
            compare_fields(frame, archive_frame(resumed, "slice_products", step, product))
            assert not list((continuous / f"outdat/new/slice_products/{product}").rglob("step000000000000.h5"))
            with h5py.File(frame) as plane:
                divisor = state["q0001"][..., 7] if source == "favre" else measure
                for name, slot in fields.items():
                    values = state[f"q{slot:04d}"][..., 7] / divisor
                    assert plane[name][...].tobytes() == values.tobytes(), (source, name)
                    count += 1
                provenance = plane.attrs["source_measure"]
                if isinstance(provenance, bytes):
                    provenance = provenance.decode()
                assert "statistics_source=solver-samples" in provenance and "phase=completed_step" in provenance
                weight = "density_time_integral" if source == "favre" else "time_integral"
                assert f"weight={weight}" in provenance and "covered_duration=" in provenance
    return count


def conservation_records(work, start):
    return np.loadtxt(work / f"outdat/new/air5_conservation_from_step{start:012d}.dat", ndmin=2)


def check_conservation_values(work, records):
    """Integrate frozen nodal q in higher precision, with one copy of each cell."""
    with h5py.File(work / "outdat/new/resources/geometry.h5") as geometry:
        jacobian = geometry["q0004"][...].transpose()
        partitions = geometry["partitions"][...].reshape(-1, 8)
        worst = 0.
        for row in records:
            step, phase = map(int, row[:2])
            label, rk, trajectory_step = ("post_update", 3, step-1) if phase else ("post_chemistry", 1, step)
            totals = np.zeros(11, dtype=np.longdouble)
            scale = totals.copy()
            for rank, part in enumerate(partitions):
                origin, cells = part[:3], part[3:6]
                snapshot = read_q_snapshot(work / f"validation/conservation.{label}.step{trajectory_step:08d}."
                                           f"rk{rk:02d}.rank{rank:08d}.bin")
                shape = tuple(cells+1)
                values = comparison_values(snapshot, True).reshape((*shape, 11), order="F").astype(np.longdouble)
                metrics = jacobian[tuple(slice(a, a+b) for a, b in zip(origin, shape))]
                weighted = values * metrics[..., None]
                for k in range(2):
                    for j in range(2):
                        for i in range(2):
                            cells_q = weighted[i:i+cells[0], j:j+cells[1], k:k+cells[2], :]
                            totals += cells_q.sum(axis=(0, 1, 2)) / 8
                            scale += np.abs(cells_q).sum(axis=(0, 1, 2)) / 8
            residual = np.abs(row[2:13] - totals)
            nonzero = scale > 0
            assert not np.any(residual[~nonzero]), row[:2]
            worst = max(worst, float(np.max(residual[nonzero] / scale[nonzero], initial=0)))
    assert worst <= 1.e-14, worst
    return worst


def check_conservation_history(continuous, seed, resumed, nonexact):
    name = "air5_conservation.bin"
    seed_file = checkpoint(seed, 2).with_name(name).read_bytes()
    assert len(seed_file) == 168
    assert seed_file == (resumed / "outdat/restore_conservation_probe.bin").read_bytes()
    assert seed_file[40:] == (nonexact / "outdat/restore_conservation_probe.bin").read_bytes()[40:]
    final = checkpoint(continuous, 4).with_name(name).read_bytes()
    assert final == checkpoint(resumed, 4).with_name(name).read_bytes()
    tag, step, _, _, _, enabled, baseline, first, last, samples, *integrals = struct.unpack("<8sq3d5q11d", final)
    assert (tag, step, enabled, baseline, first, last, samples) == (b"ASTRA5C1", 4, 1, 1, 0, 4, 4)
    records = conservation_records(continuous, 0)
    assert records[:, :2].tolist() == [[0, 0], [1, 1], [2, 1], [3, 1], [4, 1]]
    np.testing.assert_array_equal(records[0, 2:13], integrals)
    np.testing.assert_array_equal(conservation_records(resumed, 2), records[[0, 3, 4]])
    np.testing.assert_array_equal(conservation_records(nonexact, 2)[:, :2], records[[0, 3], :2])
    return check_conservation_values(continuous, records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "mpiexec", "input", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--topology", type=int, nargs=3, default=(2, 1, 1))
    parser.add_argument("--transport", choices=("pinned", "device"), default="pinned")
    parser.add_argument("--mpi-btl", default="self,vader,tcp")
    parser.add_argument("--mean-statistics", action="store_true")
    parser.add_argument("--uncompensated", action="store_true")
    parser.add_argument("--conservation", action="store_true")
    parser.add_argument("--primitive-reuse", action="store_true")
    parser.add_argument("--reference-binary", type=Path)
    args = parser.parse_args()
    if args.reference_binary and not (args.mean_statistics or args.conservation):
        parser.error("--reference-binary requires --mean-statistics or --conservation")
    root = args.output.resolve()
    root.mkdir()
    env = os.environ | dict(OMPI_MCA_pml="ob1", OMPI_MCA_btl=args.mpi_btl, OMPI_MCA_osc="pt2pt",
                            OMPI_MCA_coll="^cuda,hcoll,ucc", OMPI_MCA_opal_cuda_support="0",
                            ASTR_GPU_HALO_TRANSPORT=args.transport, ASTR_GPU_SYNC_MODE="explicit",
                            ASTR_FORCE_MPI_TOPOLOGY=",".join(map(str, args.topology)))
    if args.transport == "device":
        env |= dict(OMPI_MCA_pml="ucx", OMPI_MCA_opal_cuda_support="1")
    launch = [args.mpiexec.resolve(), "--bind-to", "none", "-np", prod(args.topology),
              args.binary.resolve(), "run", "datin/input.case"]

    def run(name, steps, archive=True, restore=None, exact=True, new_dt=None, mean=None, binary=None, conservation=None,
            reuse=None, reject=None):
        work = root / name
        prepare(work, args.input.resolve(), steps)
        primary = work / "datin/input.case"
        case = primary.read_text().replace("&chemistry\n", "&chemistry\n  step_history = 'previous'\n")
        if args.uncompensated:
            case = case.replace("compensation = 'on'", "compensation = 'off'")
        primary.write_text(case)
        runtime = work / "datin/input.run"
        mean_enabled = args.mean_statistics if mean is None else mean
        if mean_enabled:
            config = runtime.read_text().replace("enabled = .false.", "enabled = .true.")
            runtime.write_text(config.replace("write_steps = 0", "write_steps = 2"))
        if new_dt is not None:
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
            if mean_enabled:
                for source, fields in MEAN_FIELDS.items():
                    catalog.append(catalog[1] | dict(id=f"air5_i7_{source}", source=source, fields=list(fields),
                                                      statistics_source="solver-samples"))
        (work / "input.products.jsonl").write_text("".join(json.dumps(item)+"\n" for item in catalog))
        probe = {"ASTR_CHECKPOINT_TEST_RESTORE_PROBE": "1"} if restore else {}
        if name == "continuous" and args.conservation:
            probe |= dict(ASTR_VALIDATION_RHS_STEP="0", ASTR_VALIDATION_RHS_STEP_END=str(steps-1),
                          ASTR_VALIDATION_RHS_PREFIX="validation/conservation")
        executable = launch if binary is None else launch[:-3] + [binary.resolve()] + launch[-2:]
        transport = {"ASTR_GPU_HALO_TRANSPORT": "device-aware"} if binary and args.transport == "device" else {}
        observe = args.conservation if conservation is None else conservation
        use_reuse = args.primitive_reuse if reuse is None else reuse
        controls = dict(ASTR_AIR5_C4_CONSERVATION="t" if observe else "f",
                        ASTR_AIR5_PRIMITIVE_REUSE="chemistry" if use_reuse else "off")
        try:
            command(executable, work, "run.log", env | probe | transport | controls)
        except CalledProcessError:
            if reject is None:
                raise
            log = (work / "run.log").read_text(errors="replace")
            assert reject in log and "ASTR_TIMESTEP complete_step=" not in log, log[-2000:]
            return work
        assert reject is None, "unexpected successful run with incompatible checkpoint controls"
        assert "The job is done!" in (work / "run.log").read_text(errors="replace"), work
        return work

    uninterrupted = run("continuous", 4)
    seed = run("seed", 2)
    seed_path = checkpoint(seed, 2)
    with h5py.File(seed_path) as state:
        assert state["identity"][6] == 35 and state["identity"][12] == 7
        assert state["metadata"][:].tolist() == [1, int(not args.uncompensated)]
        assert any(np.count_nonzero(state[f"q{c:04d}"][...]) for c in range(12, 23)) == (not args.uncompensated)
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
                  nonexact_dt=2.e-10, compensated=not args.uncompensated, command=list(map(str, launch)))
    if args.mean_statistics:
        restored = resumed / "outdat/restore_statistics_probe.h5"
        keys = compare_fields(statistics(uninterrupted, 4), statistics(resumed, 4))
        compare_fields(statistics(seed, 2), restored)
        compare_fields(statistics(uninterrupted, 4), uninterrupted / "outdat/statistics.step000000000004.h5")
        check_statistics_window(statistics(uninterrupted, 4), 0, 4, 1.e-9)
        check_statistics_window(nonexact / "outdat/restore_statistics_probe.h5", 2, 2, 0.)
        off = run("without_means", 4, mean=False)
        compare_fields(checkpoint(uninterrupted, 4), checkpoint(off, 4))
        assert not statistics(off, 4).exists()
        mean_fields = check_mean_slices(uninterrupted, resumed, 4)
        report.update(mean_continuation=True, mean_datasets=len(keys), mean_observation_isolation=True,
                      mean_slice_fields=mean_fields)
    if args.conservation:
        error = check_conservation_history(uninterrupted, seed, resumed, nonexact)
        off = run("without_conservation", 4, conservation=False)
        compare_fields(checkpoint(uninterrupted, 4), checkpoint(off, 4))
        if args.mean_statistics:
            compare_fields(statistics(uninterrupted, 4), statistics(off, 4))
        assert not list((off / "outdat/new").glob("air5_conservation_from_step*.dat"))
        assert not checkpoint(off, 4).with_name("air5_conservation.bin").exists()
        report.update(conservation_history_exact=True, conservation_isolation=True, conservation_quadrature_error=error)
    if args.primitive_reuse:
        full = run("without_primitive_reuse", 4, archive=False, reuse=False)
        compare_fields(checkpoint(uninterrupted, 4), checkpoint(full, 4))
        if args.mean_statistics:
            compare_fields(statistics(uninterrupted, 4), statistics(full, 4))
        run("changed_reuse_contract", 4, restore=seed_path.parent, reuse=False,
            reject="AIR5 source/limiter/compensation/top contract mismatch")
        report.update(primitive_reuse_state_exact=True, primitive_reuse_contract_rejected=True)
    if args.reference_binary:
        reference = run("old_gpu", 4, archive=False, binary=args.reference_binary)
        if args.mean_statistics:
            report["old_gpu_mean_scaled_errors"] = compare_mean_reference(statistics(uninterrupted, 4),
                                                                         statistics(reference, 4))
        if args.conservation:
            old = conservation_records(reference, 0)
            new = conservation_records(uninterrupted, 0)
            np.testing.assert_array_equal(old[:, :2], new[:, :2])
            difference = np.abs(new[:, 2:13] - old[:, 2:13])
            scale = np.max(np.abs(old[:, 2:13]), axis=0)
            report["old_gpu_conservation_max_scaled_integral_difference"] = float(np.max(difference / np.maximum(1., scale)))
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(root / "report.json")


if __name__ == "__main__":
    main()
