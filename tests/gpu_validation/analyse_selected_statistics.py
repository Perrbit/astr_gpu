#!/usr/bin/env python3
"""Compare existing completed-step slice evidence; never launches a solver."""
import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np


def cases_in(root):
    cases = {}
    visited = set()
    for directory in sorted(root.glob("test_real_flatplate*/?pu")):
        directory = directory.resolve()
        if directory in visited:
            continue
        visited.add(directory)
        log = (directory / "run.log").read_text()
        match = re.search(r"forced mpi topology=\s*(\d+)\s+(\d+)\s+(\d+)", log)
        if not match or "The job is done!" not in log:
            raise ValueError(f"missing complete solver/topology evidence: {directory}")
        identity = directory.name, tuple(map(int, match.groups()))
        if identity in cases:
            raise ValueError(f"duplicate case identity: {identity}")
        cases[identity] = directory
    return cases


def datasets(case):
    result = {}
    for product in (case / "outdat/new/slice_products").iterdir():
        segments = list(product.glob("version*/segment*"))
        if len(segments) != 1:
            raise ValueError(f"expected one immutable product segment: {product}")
        result[product.name] = {path.name: path for path in segments[0].glob("step*.h5")}
    return result


def compare(left, right):
    a, b = datasets(left), datasets(right)
    if a.keys() != b.keys():
        raise ValueError("product mismatch")
    errors = {}
    for product in sorted(a, key=lambda name: (name != "raw", name)):
        if a[product].keys() != b[product].keys():
            raise ValueError(f"frame mismatch: {product}")
        for filename, left_file in sorted(a[product].items()):
            with h5py.File(left_file) as reference, h5py.File(b[product][filename]) as actual:
                np.testing.assert_array_equal(reference["clock"], actual["clock"])
                np.testing.assert_array_equal(reference["field_ids"], actual["field_ids"])
                for field in ("density", "velocity_x", "pressure", "temperature"):
                    if field not in reference:
                        continue
                    x, y = reference[field][...], actual[field][...]
                    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
                        raise ValueError(f"invalid field shape/values: {product}/{filename}/{field}")
                    difference = np.abs(x - y)
                    entry = errors.setdefault(product + "/" + field, dict(max_abs=0., max_scaled=0.))
                    entry["max_abs"] = max(entry["max_abs"], float(difference.max()))
                    scaled = difference / np.maximum(1., np.maximum(np.abs(x), np.abs(y)))
                    entry["max_scaled"] = max(entry["max_scaled"], float(scaled.max()))
                    if entry["max_scaled"] > 2e-10:
                        raise ValueError(f"scaled gate failed: {left} vs {right}: {product}/{field}: {entry}")
    return errors


def analyse(root, scope="all"):
    cases = cases_in(root)
    topologies = ((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 1), (2, 1, 2), (1, 2, 2))
    required = {(backend, topology) for backend in ("cpu", "gpu") for topology in topologies}
    if set(cases) != required:
        raise ValueError(f"case matrix mismatch: missing {required - set(cases)}, extra {set(cases) - required}")
    result = {"scope": scope, "cases": {str(key): str(path) for key, path in cases.items()}, "comparisons": {}}
    if scope in ("all", "same-topology"):
        for topology in topologies:
            result["comparisons"]["cpu-vs-gpu:" + str(topology)] = compare(cases["cpu", topology], cases["gpu", topology])
    if scope in ("all", "cross-topology"):
        for backend in ("cpu", "gpu"):
            for topology in topologies[1:]:
                result["comparisons"][backend + ":np1-vs-" + str(topology)] = compare(cases[backend, topologies[0]], cases[backend, topology])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scope", choices=("all", "same-topology", "cross-topology"), default="all")
    args = parser.parse_args()
    result = analyse(args.root, args.scope)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    entries = [entry for comparison in result["comparisons"].values() for entry in comparison.values()]
    print(json.dumps(dict(comparisons=len(result["comparisons"]), max_abs=max(e["max_abs"] for e in entries),
                          max_scaled=max(e["max_scaled"] for e in entries))))
