"""Freeze the five FP64 linear stencils used by the original GPU filter."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from run_air5_flux_components import extract


ROOT = Path(__file__).resolve().parents[2]
STENCILS = ("filter10_value", "filter6_value", "filter8_value",
            "filter6_boundary_point1", "filter6_boundary_point2")
AXES = ("filter_x_global_kernel", "filter_y_pong_to_q_halo_global_kernel",
        "filter_z_halo_global_kernel", "filter_x_scalar_global_kernel",
        "filter_y_scalar_to_q_halo_global_kernel", "filter_z_scalar_global_kernel")


def build(cmd, out, name):
    with (out / name).open("w") as log:
        subprocess.run(cmd, cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    names = ["src/astr_precision.F90", "src/constdef.F90", "src_gpu/solver_gpu.cuf",
             "tests/gpu_backend/air5_legacy_linear_reference.cuf",
             "tests/gpu_backend/freeze_air5_legacy_linear.py",
             "src/boundary_high_order_body.inc", "src_gpu/boundary_high_order_gpu.cuf",
             "tests/gpu_backend/air5_legacy_linear_axes.cuf"]
    paths = [out / name for name in names]
    for name, path in zip(names, paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, path)
    solver = paths[2].read_text()
    (out / "frozen_linear.inc").write_text(
        "\n".join(extract(solver, name) for name in STENCILS))
    (out / "frozen_axes.inc").write_text("\n".join(extract(solver, name) for name in
        ("wall_filter6_active", "wall_filter6_global_value", *AXES)))
    rng = np.random.default_rng(205)
    samples = rng.uniform(-1., 1., size=(4096, 11))
    samples[2048:] = np.ldexp(samples[2048:], rng.integers(-90, 91, (2048, 11)))
    samples[:32] = np.ldexp(samples[:32], -60) + 2.**-10
    samples = np.concatenate((samples, np.eye(11), np.ones((1, 11))))
    with (out / "lines.bin").open("wb") as f:
        np.array([len(samples)], dtype="<i4").tofile(f)
        samples.astype("<f8").tofile(f)
    flags = [str(args.compiler.resolve()), "-O2", "-Mpreprocess", "-Mextend", "-cuda",
             "-gpu=rdc,maxregcount:128,keep", "-D_CUDA", "-I" + str(out), *names[:2]]
    cmd = flags + [names[3], "-o", "probe"]
    build(cmd, out, "build.log")
    subprocess.run(["./probe", "lines.bin", "values.bin"], cwd=out, check=True)
    values = np.fromfile(out / "values.bin", dtype="<f8").reshape(-1, len(STENCILS))
    assert values.shape == (len(samples), len(STENCILS)) and np.isfinite(values).all()
    assert np.array_equal(values[-1], np.ones(len(STENCILS)))
    field = rng.uniform(-1., 1., size=(11, 21, 22, 23))
    field[8] = np.ldexp(field[8], -60) + 2.**-10
    field[9] = np.ldexp(field[9], -90)
    field[10] = 1.
    field.astype("<f8").tofile(out / "axes_input.bin")
    axis_cmd = flags + names[6:8] + ["-o", "probe_axes"]
    build(axis_cmd, out, "build_axes.log")
    subprocess.run(["./probe_axes", "axes_input.bin", "axes_values.bin"], cwd=out, check=True)
    axes = np.fromfile(out / "axes_values.bin", dtype="<f8").reshape(2, 3, 6, 11, 11, 12, 13)
    assert np.isfinite(axes).all() and np.all(axes[:, :, :, 10] == 1.)
    equal = axes[0].view(np.uint64) == axes[1].view(np.uint64)
    mismatch = np.count_nonzero(~equal, axis=(2, 3, 4, 5)).tolist()
    max_diff = np.max(abs(axes[0] - axes[1]), axis=(2, 3, 4, 5)).tolist()
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in zip(names, paths)}
    (out / "provenance.json").write_text(json.dumps(
        dict(source_sha256=hashes, commands=[cmd, axis_cmd], stencils=STENCILS, lines=len(samples),
             axes=AXES, sides=["no physical end", "lower", "upper", "both", "wall6", "wall6 adiabatic"],
             scalar_full_mismatches=mismatch, scalar_full_max_abs=max_diff), indent=2) + "\n")
    print(f"Frozen {len(samples)} lines for each of {len(STENCILS)} FP64 stencils: {out}")
    print(f"Scalar/full bit mismatches by axis and closure: {mismatch}")
    print(f"Scalar/full maximum absolute difference: {max_diff}")


if __name__ == "__main__":
    main()
