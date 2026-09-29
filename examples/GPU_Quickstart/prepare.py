#!/usr/bin/env python3
"""Prepare self-contained short ASTR examples; no flow integration is done here."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
CASES = ("tgv", "channel", "flatplate", "air5_tgv", "air5_flatplate")


def record(lines, marker):
    matches = [i for i, line in enumerate(lines) if line.startswith("# " + marker)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one input marker: {marker}")
    index = matches[0] + 1
    if not lines[index].strip() or lines[index].startswith("#"):
        raise ValueError(f"missing value after {marker}")
    return index


def positive_real(value):
    number = float(value.lower().replace("d", "e"))
    if not math.isfinite(number) or number <= 0:
        raise ValueError("time step must be finite and positive")
    return number


def flatplate_data(datin, grid):
    import h5py
    import numpy as np

    im, jm, km = grid
    xline = np.linspace(0., 10., im + 1)
    eta = np.linspace(0., 1., jm + 1)
    yline = np.expm1(3. * eta) / np.expm1(3.)
    zline = np.linspace(0., .25, km + 1)
    x, y, z = np.meshgrid(xline, yline, zline, indexing="ij")
    with h5py.File(datin / "grid.h5", "w") as handle:
        for name, array in zip(("x", "y", "z"), (x, y, z)):
            handle.create_dataset(name, data=array.transpose(2, 1, 0))
    with (datin / "inlet.prof").open("w", encoding="ascii") as handle:
        handle.write("Analytic flat-plate startup profile\n")
        handle.write("rho u v temperature\n")
        handle.write("0.08 0.026 0.018 0.01\n")
        handle.write("rho u v temperature\n")
        for y in yline:
            u = 1. - math.exp(-(float(y) / .08)**2)
            handle.write(f"1.0 {u:.17e} 0.0 1.0\n")


def air5_flatplate_data(datin):
    # Frozen values from the existing Mach-4 precursor startup specification.
    length = 3.75482142813887026e-4
    thickness = 9.76415315871868058e-4
    velocity = 3.11235677152851986e3
    gas = 8.31446261815324030 * (.767 / .028 + .233 / .032)
    domain = (80. * length, 12. * length, 2. * length)
    (datin / "air5_hbl_domain.dat").write_text(
        "air5_hbl_domain_v1\n" + " ".join(f"{v:.17e}" for v in domain) + "\n",
        encoding="ascii",
    )
    with (datin / "air5_hbl_profile.dat").open("w", encoding="ascii") as handle:
        handle.write("# Analytic startup seed, not a developed boundary layer.\n")
        handle.write(f"# x_origin={20. * length:.17e}\n")
        handle.write("# y rho u v w p T Tv Y_N2 Y_O2 Y_N Y_O Y_NO\n")
        heights = [i * thickness / 4096 for i in range(4097)] + [domain[1]]
        for y in heights:
            s = min(y / thickness, 1.)
            shape = 2. * s - 2. * s**3 + s**4
            temperature = 3000. - 1500. * shape
            row = (y, 20000. / (gas * temperature), velocity * shape,
                   0., 0., 20000., temperature, temperature, .767, .233, 0., 0., 0.)
            handle.write(" ".join(f"{v:.17e}" for v in row) + "\n")


def prepare(case, destination, mode="gpu", np=1, topology="1,1,1",
            steps=None, deltat=None):
    if case not in CASES or mode not in ("cpu", "gpu"):
        raise ValueError("unsupported case or execution mode")
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    source = HERE / case / "datin"
    lines = (source / "input.dat").read_text(encoding="ascii").splitlines()
    grid = tuple(int(v) for v in lines[record(lines, "ia,ja,ka")].split(","))
    topo = tuple(int(v) for v in topology.split(","))
    if len(topo) != 3 or min(topo) < 1 or np < 1 or math.prod(topo) != np:
        raise ValueError("positive topology dimensions must multiply to NP")
    if any(n // ranks < 6 for n, ranks in zip(grid, topo)):
        raise ValueError("these examples require local extents >= 6")
    flags_index = record(lines, "nondimen,")
    flags = lines[flags_index].split(",")
    if len(flags) != 9:
        raise ValueError("template must contain nine logical flags")
    flags[-1] = "t" if mode == "gpu" else "f"
    lines[flags_index] = ",".join(flags)
    controller = (source / "controller").read_text(encoding="ascii").splitlines()
    controls_index = record(controller, "maxstep,")
    controls = controller[controls_index].split(",")
    if steps is not None:
        if steps < 1:
            raise ValueError("STEPS must be positive")
        controls[0] = str(steps - 1)
        controls[1] = str(max(steps - 1, 1))
        controller[controls_index] = ",".join(controls)
    dt_index = record(controller, "deltat")
    if deltat is not None:
        positive_real(deltat)
        controller[dt_index] = deltat
    dt = positive_real(controller[dt_index])
    # Resolve optional dependencies before creating an output directory.
    if case == "flatplate":
        import h5py  # noqa: F401
        import numpy  # noqa: F401
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    datin = destination / "datin"
    shutil.copytree(source, datin)
    (datin / "input.dat").write_text("\n".join(lines) + "\n", encoding="ascii")
    (datin / "controller").write_text("\n".join(controller) + "\n", encoding="ascii")
    if case == "flatplate":
        flatplate_data(datin, grid)
    elif case == "air5_flatplate":
        air5_flatplate_data(datin)
    metadata = dict(case=case, mode=mode, ranks=np, topology=topo, grid_upper_bounds=grid,
                    updates=int(controls[0]) + 1, deltat=dt,
                    expected_final_time=(int(controls[0]) + 1) * dt,
                    initial_restart=False, scope="short startup, not physical validation")
    (destination / "case.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                          encoding="ascii")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--mode", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--np", type=int, default=1)
    parser.add_argument("--topology", default="1,1,1")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--deltat")
    args = parser.parse_args()
    print(prepare(args.case, args.destination, args.mode, args.np, args.topology,
                  args.steps, args.deltat))


if __name__ == "__main__":
    main()
