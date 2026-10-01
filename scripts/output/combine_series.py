"""Build a separate native field index by following only sealed, explicit parent edges."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

from repair_series import (directory, fingerprint, frame_catalog, segment_record,
                           validate_frame, validate_units)


def plain_path(path):
    # Check before resolving: a symbolic link followed by /.. is still an alias.
    path = Path(path).absolute()
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        directory(current)
    return path.resolve(strict=True)


def retained_bytes(value):
    if isinstance(value, dict):
        return sys.getsizeof(value) + sum(retained_bytes(k) + retained_bytes(v) for k, v in value.items())
    if isinstance(value, (list, tuple, set)):
        return sys.getsizeof(value) + sum(retained_bytes(v) for v in value)
    return sys.getsizeof(value)


def parent_chain(head, budget):
    chain, seen = [], set()
    head = plain_path(head)
    product = head.parent.name
    if product not in ("fields", "slices"):
        raise ValueError("unknown native archive product")
    current = head
    expected = None
    while True:
        if not re.fullmatch(r"segment[0-9]{8}", current.name) or current.parent.name != product:
            raise ValueError("parent edge changes product or segment identity")
        if str(current) in seen:
            raise ValueError("cycle in explicit segment parent chain")
        record = segment_record(current, require_parent=True)
        if expected is not None and record["fingerprint"] != expected:
            raise ValueError("explicit parent segment checksum mismatch")
        seen.add(str(current))
        chain.append((str(current), record))
        if retained_bytes(chain) + retained_bytes(seen) > budget:
            raise ValueError("parent chain exceeds its explicit catalog byte budget")
        if record["parent"] is None:
            break
        expected = record["parent_fingerprint"]
        current = plain_path(current / record["parent"])
    chain.reverse()
    for (_, parent), (_, child) in zip(chain, chain[1:]):
        if parent["step"] > child["step"] or parent["time"] > child["time"]:
            raise ValueError("restore point predates parent segment origin")
    return chain, retained_bytes(chain) + retained_bytes(seen)


def combine_series(head, *, output=None, catalog_bytes=1048576):
    """Sources must be stopped/immutable; output is exclusive and outside source products."""
    if catalog_bytes <= 0:
        raise ValueError("invalid catalog byte budget")
    chain, chain_bytes = parent_chain(head, catalog_bytes)
    destination = None
    if output is not None:
        output = Path(output).absolute()
        destination = plain_path(output.parent) / output.name
        if ":" in str(destination) or any(ord(c) < 32 or ord(c) == 127 for c in str(destination)):
            raise ValueError("output path cannot be represented in an HDF reference")
        for name, _ in chain:
            if destination == Path(name).parent or destination.is_relative_to(Path(name).parent):
                raise ValueError("combined output must be outside source product directories")
        destination.mkdir(exist_ok=False)
    ledger = xml = None
    count = excluded = ignored = 0
    scan_bytes = sum(record["crc_scan_bytes"] for _, record in chain)
    peak = chain_bytes
    previous_step, previous_time, signature = -1, -math.inf, None
    try:
        if destination is not None:
            ledger = (destination / "lineage.frames.tmp").open("x", encoding="utf-8")
            xml = (destination / "lineage.xdmf.tmp").open("xb")
            ledger.write("ASTR_PARENT_FRAME_SERIES_1\n")
            xml.write(b'<?xml version="1.0"?><Xdmf Version="3.0"><Domain>'
                      b'<Grid Name="lineage" GridType="Collection" CollectionType="Temporal">\n')
        for position, (name, record) in enumerate(chain):
            segment = Path(name)
            directory(segment.parent / "resources")
            remaining = catalog_bytes - chain_bytes
            names, candidates, used = frame_catalog(segment, remaining)
            ignored += candidates
            peak = max(peak, chain_bytes + used)
            geometry = segment.parent / "resources/data.h5"
            geometry_fp = fingerprint(geometry)
            scan_bytes += geometry_fp[0]
            cutoff = chain[position + 1][1] if position + 1 < len(chain) else None
            for frame_name in names:
                if cutoff is not None and int(frame_name[4:]) > cutoff["step"]:
                    excluded += 1
                    continue
                step, time, layout, grid, scanned = validate_frame(
                    segment / frame_name, geometry, geometry_fp, record["fingerprint"])
                scan_bytes += scanned
                if step < record["step"] or time < record["time"]:
                    raise ValueError("frame clock predates its segment origin")
                if cutoff is not None and time > cutoff["time"]:
                    raise ValueError("parent frame time exceeds its declared restore point")
                if step <= previous_step or time <= previous_time:
                    raise ValueError("duplicate or nonmonotonic clocks in explicit parent chain")
                if signature is not None and signature != layout:
                    raise ValueError("field/plane layout changes across the parent chain")
                validate_units(segment / "input.txt", layout[1])
                if destination is not None:
                    for item in grid.iter("DataItem"):
                        path, dataset = item.text.split(":", 1)
                        item.text = os.path.relpath(segment / path, destination) + ":" + dataset
                    ledger.write(f"{step} {time:.17g} {os.path.relpath(segment / frame_name, destination)}\n")
                    xml.write(ET.tostring(grid, encoding="utf-8") + b"\n")
                previous_step, previous_time, signature = step, time, layout
                count += 1
        if count == 0:
            raise ValueError("no sealed frames in explicit parent chain")
        if xml is not None:
            xml.write(b"</Grid></Domain></Xdmf>\n")
    finally:
        if ledger is not None:
            ledger.close()
        if xml is not None:
            xml.close()
    if destination is not None:
        for name in ("lineage.frames", "lineage.xdmf"):
            os.rename(destination / (name + ".tmp"), destination / name)
    return dict(format="ASTR_PARENT_FRAME_SERIES_1", published=destination is not None,
                segments=len(chain), frames=count, excluded_parent_frames=excluded,
                ignored_candidates=ignored, catalog_peak_bytes=peak,
                field_array_read_bytes=0, crc_scan_bytes=scan_bytes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("head", type=Path)
    parser.add_argument("--output", type=Path,
                        help="exclusive new index directory; explicitly assert that all source segments are stopped")
    parser.add_argument("--catalog-bytes", type=int, default=1048576)
    args = parser.parse_args()
    print(json.dumps(combine_series(args.head, output=args.output, catalog_bytes=args.catalog_bytes), indent=2))


if __name__ == "__main__":
    main()
