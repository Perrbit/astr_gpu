"""Subset frozen old-GPU TGV states; no time advancement or replacement flux solver."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from run_cpp_multi_tgv_gate import raw_field


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    for label, topology in (("np1", (1, 1, 1)), ("np2_x", (2, 1, 1)), ("np2_y", (1, 2, 1)), ("np2_z", (1, 1, 2))):
        destination = root / label
        destination.mkdir()
        for mode in (0, 1):
            source = args.reference / ("old_gpu_mixed" if mode else "old_gpu_fp64") / "outdat"
            with h5py.File(source / "geometry.h5") as geometry:
                jacobian = geometry["jacob"][...].transpose(2, 1, 0)
                metric = np.empty((*jacobian.shape, 3, 3))
                for a in range(3):
                    for b in range(3):
                        metric[..., a, b] = geometry[f"dxi{a+1}{b+1}"][...].transpose(2, 1, 0)
            first = ".step00000000.rk01.rank00000000.bin"
            h, base = raw_field(source / ("rhs.pre_rhs" + first))
            global_upper, halo = np.asarray(h[:3]), int(h[3])
            indices = np.indices(base.shape[:3])-halo
            wrapped = tuple(indices[a] % global_upper[a] for a in range(3))
            metric_halo = metric[wrapped].copy()
            jac_halo = jacobian[wrapped].copy()
            physical = tuple(slice(halo, halo+n+1) for n in global_upper)
            metric_halo[physical] = metric
            jac_halo[physical] = jacobian
            for stage in (1, 2, 3):
                suffix = f".step00000000.rk{stage:02d}.rank00000000.bin"
                _, q = raw_field(source / ("rhs.pre_rhs" + suffix))
                _, primitives = raw_field(source / ("rhs.pre_rhs_primitives" + suffix), primitive=True)
                _, next_q = raw_field(source / ("rhs.post_update" + suffix))
                with (source / ("rhs.gradients" + suffix)).open("rb") as f:
                    gh = np.fromfile(f, np.int32, 5)
                    count = int(np.prod(global_upper+1))
                    grad = np.fromfile(f, np.float64, count*9).reshape((*global_upper+1, 3, 3), order="F")
                    thermal = np.fromfile(f, np.float64, count*3).reshape((*global_upper+1, 3), order="F")
                    if not np.array_equal(gh[:3], global_upper):
                        raise AssertionError("original gradient dimensions")
                rhs = []
                for field in ("conv", "full"):
                    with (source / ("rhs." + field + suffix)).open("rb") as f:
                        rh = np.fromfile(f, np.int32, 4)
                        rhs.append(np.fromfile(f, np.float64).reshape((*global_upper+1, 5), order="F"))
                        if not np.array_equal(rh, [*global_upper, 5]):
                            raise AssertionError("original RHS dimensions")
                for rank in range(np.prod(topology)):
                    coordinate = np.array([rank % topology[0], (rank//topology[0]) % topology[1], rank//(topology[0]*topology[1])])
                    upper = global_upper//topology
                    origin = coordinate*upper
                    volume = tuple(slice(int(o), int(o+n+2*halo+1)) for o, n in zip(origin, upper))
                    body = tuple(slice(int(o), int(o+n+1)) for o, n in zip(origin, upper))
                    next_body = tuple(slice(int(o+halo), int(o+halo+n+1)) for o, n in zip(origin, upper))
                    path = destination / f"mode{mode}.stage{stage}.rank{rank:08d}.bin"
                    with path.open("xb") as f:
                        np.array([*global_upper, *upper, halo, *origin, mode, stage], np.int32).tofile(f)
                        # Same existing TGV constants as rhs_probe.F90.
                        np.array([1600., .72, (1.4-1)*.1**2, 110.3/273.15, 1+110.3/273.15,
                                  1e-4, 1.4*.1**2, 1/(1.4-1)], np.float64).tofile(f)
                        for a in (base[volume], q[volume], primitives[volume], metric_halo[volume], jac_halo[volume],
                                  grad[body], thermal[body], rhs[0][body], rhs[1][body], next_q[next_body]):
                            a.ravel(order="F").tofile(f)
    (root / "source.json").write_text(json.dumps(dict(reference=str(args.reference.resolve()),
        scope="frozen same-state 643e periodic gradients, conv/full RHS and RK stages, FP64 and the selected original storage candidate"), indent=2)+"\n")
    for p in root.rglob("*"):
        p.chmod(0o555 if p.is_dir() else 0o444)
    root.chmod(0o555)


if __name__ == "__main__":
    main()
