#!/usr/bin/env python3
"""Tiny CPU/MPI check of publication and retention on the actual output filesystem."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def preflight(probe, mpiexec, directory):
    work = Path(tempfile.mkdtemp(prefix="checkpoint-publication-", dir=directory))
    report = dict(status="failed", directory=str(work), checks=[])

    def run(path, mode, name=None, success=True):
        command = [str(mpiexec), "--mca", "pml", "ob1", "--mca", "btl", "self,tcp",
                   "--mca", "coll_hcoll_enable", "0", "--mca", "opal_cuda_support", "false",
                   "-n", "1", str(probe), str(path), mode]
        if name is not None:
            command.append(str(name))
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=120)
        number = len(report["checks"])
        (work / f"check{number:02d}.log").write_text(result.stdout)
        report["checks"].append(dict(mode=mode, name=name, returncode=result.returncode,
                                     expected_success=success))
        if (result.returncode == 0) != success or ("PASS bundle" if success else "REJECT bundle") not in result.stdout:
            raise RuntimeError(f"Checkpoint preflight failed: {mode}; {work / f'check{number:02d}.log'}")

    try:
        for keep in (1, 2):
            root = work / f"keep{keep}"
            root.mkdir(mode=0o700)
            candidate = root / "batch1.tmp"
            candidate.mkdir()
            (candidate / "state.h5").write_bytes(b"ASTR filesystem preflight\n")
            run(candidate, "seal")
            payload = {p.name: p.read_bytes() for p in candidate.iterdir()}
            for number in (2, 3, 4):
                shutil.copytree(candidate, root / f"batch{number}.tmp")
            run(root, "retain", keep)
            if (root / "LATEST").read_text() != "batch4\n":
                raise RuntimeError("Wrong LATEST after retention")
            expected = {f"batch{number}" for number in range(5-keep, 5)}
            if {p.name for p in root.iterdir() if p.is_dir()} != expected:
                raise RuntimeError("Wrong retained generations")
            for name in sorted(expected):
                run(root / name, "validate")
            shutil.copytree(root / "batch4", root / "batch4.tmp")
            run(root, "publish", "batch4", success=False)
            if payload != {p.name: p.read_bytes() for p in (root / "batch4").iterdir()}:
                raise RuntimeError("Existing checkpoint was overwritten")
            if (root / "LATEST").read_text() != "batch4\n":
                raise RuntimeError("Conflict changed LATEST")
        report["status"] = "passed"
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"CHECKPOINT_PUBLICATION_PREFLIGHT={report['status']} report={work / 'report.json'}", flush=True)
    return work


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--mpiexec", type=Path, default=Path(shutil.which("mpiexec") or "mpiexec"))
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error("--directory must already exist on the intended output filesystem")
    preflight(args.probe.resolve(strict=True), args.mpiexec, args.directory.resolve(strict=True))


if __name__ == "__main__":
    main()
