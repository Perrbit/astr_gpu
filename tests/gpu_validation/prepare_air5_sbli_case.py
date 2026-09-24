#!/usr/bin/env python3
"""Prepare the bounded incident-shock/HBL startup gate, not a developed SBLI."""
import argparse
import json
from pathlib import Path

import numpy as np

from air5_htr_profile import load_htr_similarity_profile, map_htr_profile_to_air5
from air5_radau_reference import Air5RadauReference
from generate_air5_oblique_shock_states import frozen_oblique_jump, jump_metadata
from prepare_air5_c4_case import prepare_case, replace_after_marker


def prepare(destination, grid, maxstep, deltat, use_gpu):
    root = Path(__file__).resolve().parents[2]
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence: {destination}")
    mechanism = root / "chemMech/air5_kimjo12.json"
    chemistry = Air5RadauReference(mechanism)
    profile, _ = map_htr_profile_to_air5(load_htr_similarity_profile(
        Path(__file__).parent / "data/htr_multispecies_tbl_mach6_similarity.dat"),
        pressure=101325.)
    edge = profile[-1]
    y = np.array(edge.mass_fraction)
    gas = float(y @ chemistry.gas_constant)
    gamma = 1 + gas / float(y @ chemistry.cv_tr)
    speed = np.hypot(edge.u, edge.v)
    jump = frozen_oblique_jump(chemistry,
        mach=speed/np.sqrt(gamma*gas*edge.temperature),
        temperature=edge.temperature, tv=edge.tv, pressure=edge.pressure,
        shock_angle_deg=20., inflow_angle_deg=np.degrees(np.arctan2(edge.v, edge.u)),
        mass_fraction=y)
    ref_len = 4.41262150017878821e-5
    meta = jump_metadata(jump, top_x=20*ref_len, top_y=8*ref_len)
    if not 0 < meta["geometric_wall_intersection_x"] < 80*ref_len:
        raise ValueError("incident shock geometric foot is outside the domain")
    if meta["max_scaled_normal_flux_residual"] > 2e-12:
        raise ValueError("incident shock violates frozen normal-flux conservation")
    input_file = prepare_case(root / "examples/Taylor_Green_Vortex_SI/datin",
        destination, grid, maxstep, deltat, "t", "f", use_gpu,
        "high-enthalpy-boundary-layer", 1, "uniform")
    lines = input_file.read_text().splitlines()
    replace_after_marker(lines, "flowtype", "air5sbli")
    replace_after_marker(lines, "recon_schem, lchardecomp,bfacmpld,shkcrt", "3,f,0.3d0,0.05d0")
    input_file.write_text("\n".join(lines)+"\n")
    rows = [[meta["top_x"], meta["top_y"], *jump.normal], jump.upstream_q, jump.downstream_q]
    (destination / "datin/air5_incident_shock.dat").write_text(
        "air5_incident_shock_v1\n" + "\n".join(
            " ".join(f"{v:.17e}" for v in row) for row in rows)+"\n")
    meta.update(flowtype="air5sbli", grid=grid, domain=[80*ref_len,8*ref_len,2*ref_len],
        maxstep=maxstep, deltat=deltat, initial_field="uniform-x HBL profile",
        wall_temperature=2925., lfilter=False,
        validation_scope="startup coupling only; not physical SBLI acceptance")
    (destination / "incident_shock_metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
    (destination / "validation").mkdir()
    return input_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--grid", default="31,31,7")
    parser.add_argument("--maxstep", type=int, default=2)
    parser.add_argument("--deltat", default="1.d-11")
    parser.add_argument("--use-gpu", choices=("t", "f"), required=True)
    args = parser.parse_args()
    if args.maxstep < 0:
        parser.error("maxstep must be non-negative")
    print(prepare(args.destination, args.grid, args.maxstep, args.deltat, args.use_gpu))
