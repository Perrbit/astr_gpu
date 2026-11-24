"""Compare the frozen and regional RK stages before running the M4 timing pair."""
import argparse
from array import array
from itertools import product
import json
from math import isfinite, prod
from pathlib import Path
import re
import struct


PHASES = {"pre_rhs": (5, True), "pre_rhs_primitives": (6, True), "conv": (5, False),
          "full": (5, False), "post_update": (5, True)}
RTOL = 2e-10


def layouts(log, shape, topology):
    result = {}
    for line in log.splitlines():
        if line.startswith("ASTR_GPU_CPP_LAYOUT "):
            data = {name: tuple(map(int, text.split())) for name, text in
                    re.findall(r"(\w+)=\s*((?:-?\d+\s*)+)", line)}
            rank, = data.pop("rank")
            if rank in result or data["global"] != tuple(shape) or data["topology"] != tuple(topology):
                raise ValueError("unexpected or duplicate qualification decomposition")
            result[rank] = data
    if sorted(result) != list(range(prod(topology))):
        raise ValueError("incomplete qualification rank layout")
    return result


class Field:
    def __init__(self, path, layout, count, halo, primitive=False):
        header_words = 4 if primitive or not halo else 5
        raw = path.read_bytes()
        header = struct.unpack_from(f"={header_words}i", raw)
        self.upper = upper = layout["upper"]
        self.halo = h = layout["halo"][0] if halo else 0
        expected = (*upper, h) if primitive else (*upper, h, count) if halo else (*upper, count)
        self.widths = tuple(n + 2 * h + 1 for n in upper)
        self.size = prod(self.widths)
        self.values = array("d")
        self.values.frombytes(raw[4 * header_words:])
        self.bits = memoryview(self.values).cast("B").cast("Q")
        if header != expected or len(self.values) != self.size * count:
            raise ValueError(f"invalid field shape: {path}")

    def offset(self, point, comp):
        nx, ny, _ = self.widths
        i, j, k = (p + self.halo for p in point)
        return i + nx * (j + ny * k) + self.size * comp


def scaled_difference(a, b, components):
    worst = 0.
    points = list(product(*(range(n + 1) for n in a.upper)))
    for c in range(components):
        maximum, scale = 0., 1.
        for point in points:
            x, y = a.values[a.offset(point, c)], b.values[b.offset(point, c)]
            if not isfinite(x) or not isfinite(y):
                raise ValueError("nonfinite owned field value")
            maximum, scale = max(maximum, abs(x - y)), max(scale, abs(y))
        worst = max(worst, maximum / scale)
    return worst


def face_bits(fields, layout, components):
    for rank, own in fields.items():
        for axis in range(3):
            transverse = [a for a in range(3) if a != axis]
            for high in (False, True):
                peer = layout[rank]["higher" if high else "lower"][axis] if layout[rank]["topology"][axis] > 1 else rank
                rhs = fields[peer]
                if any((own.upper[a], layout[rank]["origin"][a]) != (rhs.upper[a], layout[peer]["origin"][a])
                       for a in transverse):
                    raise ValueError("transverse face layouts differ")
                for where in product(*(range(own.upper[a] + 1) for a in transverse)):
                    p, q = [0, 0, 0], [0, 0, 0]
                    for a, value in zip(transverse, where):
                        p[a] = q[a] = value
                    for h in range(1, own.halo + 1):
                        p[axis], q[axis] = (own.upper[axis] + h, h) if high else (-h, rhs.upper[axis] - h)
                        for c in range(components):
                            if own.bits[own.offset(p, c)] != rhs.bits[rhs.offset(q, c)]:
                                raise ValueError(f"face halo differs: rank={rank} axis={axis} high={high} depth={h}")


def compare_stages(baseline, regional, shape, topology, steps):
    layout = layouts((regional / "run.log").read_text(), shape, topology)
    if layout != layouts((baseline / "run.log").read_text(), shape, topology):
        raise ValueError("phase comparison requires the same decomposition")
    maxima = dict.fromkeys(PHASES, 0.)
    face_checks = 0
    for step, rk in product(range(steps), range(1, 4)):
        for label, (count, halo) in PHASES.items():
            candidate = {}
            for rank, local in layout.items():
                path = f"outdat/rhs.{label}.step{step:08d}.rk{rk:02d}.rank{rank:08d}.bin"
                a, b = (Field(root / path, local, count, halo, label == "pre_rhs_primitives")
                        for root in (regional, baseline))
                scaled = scaled_difference(a, b, count)
                if scaled > RTOL:
                    raise ValueError(f"scaled discrepancy {scaled} exceeds {RTOL}: {path}")
                maxima[label] = max(maxima[label], scaled)
                if label == "pre_rhs_primitives":
                    for point in product(*(range(n + 1) for n in a.upper)):
                        if any(a.values[a.offset(point, c)] <= 0 for c in (0, 4, 5)):
                            raise ValueError(f"nonpositive density, pressure or temperature: {path}")
                candidate[rank] = a
            if label in ("pre_rhs", "pre_rhs_primitives"):
                face_bits(candidate, layout, count)
                face_checks += 6 * len(layout)
    return dict(maximum_scaled_difference=maxima, exact_regional_face_checks=face_checks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    setup = json.loads((args.root / "campaign.json").read_text())["qualification"]
    print(json.dumps(compare_stages(args.root / "gate_a", args.root / "gate_b", setup["grid"],
                                    setup["mpi"], setup["timesteps"]), indent=2))
