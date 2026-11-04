#!/usr/bin/env python3
"""Prepare self-contained short ASTR examples; no flow integration is done here."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent
CASES = ("tgv", "channel", "flatplate", "air5_tgv", "air5_flatplate")


def template_value(text, name):
    """Read one scalar assignment in the bundled one-field-per-line templates."""
    matches = re.findall(rf"^\s*{re.escape(name)}\s*=\s*(.*?)\s*$", text, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"expected one named template assignment: {name}")
    return matches[0]


def set_template_value(text, name, value):
    template_value(text, name)
    return re.sub(rf"^(\s*{re.escape(name)}\s*=).*$", lambda match: f"{match[1]} {value}",
                  text, flags=re.MULTILINE)


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
    primary = (source / "input.case").read_text(encoding="ascii")
    runtime = (source / "input.run").read_text(encoding="ascii")
    grid = tuple(int(template_value(primary, name)) for name in ("ia", "ja", "ka"))
    topo = tuple(int(v) for v in topology.split(","))
    if len(topo) != 3 or min(topo) < 1 or np < 1 or math.prod(topo) != np:
        raise ValueError("positive topology dimensions must multiply to NP")
    if any(n // ranks < 6 for n, ranks in zip(grid, topo)):
        raise ValueError("these examples require local extents >= 6")
    primary = set_template_value(primary, "use_gpu", ".true." if mode == "gpu" else ".false.")
    if steps is not None:
        if steps < 1:
            raise ValueError("STEPS must be positive")
        runtime = set_template_value(runtime, "max_steps", steps)
        runtime = set_template_value(runtime, "reload_steps", max(steps - 1, 1))
    updates = int(template_value(runtime, "max_steps"))
    if deltat is not None:
        positive_real(deltat)
        runtime = set_template_value(runtime, "fixed_dt", deltat)
    dt = positive_real(template_value(runtime, "fixed_dt"))
    # Resolve optional dependencies before creating an output directory.
    if case == "flatplate":
        import h5py  # noqa: F401
        import numpy  # noqa: F401
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    datin = destination / "datin"
    shutil.copytree(source, datin, ignore=shutil.ignore_patterns("input.dat", "controller"))
    (datin / "input.case").write_text(primary, encoding="ascii")
    (datin / "input.run").write_text(runtime, encoding="ascii")
    shutil.copyfile(HERE / "input.output", datin / "input.output")
    if case.startswith("air5_"):
        config = (datin / "input.output").read_text()
        config = config.replace(" directory=", " exact_restart=.true.,\n directory=")
        (datin / "input.output").write_text(config)
    (destination / "outdat/output").mkdir(parents=True)
    if case == "flatplate":
        flatplate_data(datin, grid)
    elif case == "air5_flatplate":
        air5_flatplate_data(datin)
    metadata = dict(case=case, mode=mode, ranks=np, topology=topo, grid_upper_bounds=grid,
                    updates=updates, deltat=dt,
                    expected_final_time=updates * dt,
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
