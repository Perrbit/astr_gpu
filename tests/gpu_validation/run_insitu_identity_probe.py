"""Bounded analytic Catalyst tests; requires the external dependency probe build."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image


def check_image(directory):
    with Image.open(directory / "sphere.jpg") as image:
        if image.size != (640, 480):
            raise RuntimeError("Unexpected image dimensions")
        rgb = image.convert("RGB")
        if max(high-low for low, high in rgb.getextrema()) <= 100:
            raise RuntimeError("Blank or low-contrast image")
        rgb.save(directory / "sphere.eps", format="EPS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("probe", "pipeline", "backend", "output", "mpiexec"):
        parser.add_argument("--" + option, required=True, type=Path)
    parser.add_argument("--render-backend", choices=("egl", "osmesa"), default="egl")
    parser.add_argument("--software-library-dir", type=Path)
    args = parser.parse_args()
    software = args.render_backend == "osmesa"
    if software and (args.software_library_dir is None or
                     not (args.software_library_dir / "libOSMesa.so.8").is_file()):
        raise RuntimeError("OSMesa requires an explicit directory containing libOSMesa.so.8")
    for path in (args.probe, args.pipeline, args.backend, args.mpiexec):
        if not path.exists():
            raise RuntimeError(f"Missing input: {path}")
    args.output.mkdir(parents=True, exist_ok=False)
    # Fresh processes are essential: CUDA visibility is cached after initialization.
    variants = [("np1_native", 1, None), ("np2_reversed", 2, "1,0"),
                ("np1_masked", 1, "1")]
    if software:
        variants = [("osmesa_np1", 1, ""), ("osmesa_np2", 2, "")]
    summary = []
    wrapper = Path(__file__).with_name("insitu_device_identity.py")
    for name, ranks, visibility in variants:
        directory = (args.output / name).resolve()
        directory.mkdir()
        env = os.environ.copy()
        for variable in ("DISPLAY", "PYTHONPATH", "CUDA_VISIBLE_DEVICES",
                         "CATALYST_IMPLEMENTATION_PREFER_ENV"):
            env.pop(variable, None)
        if visibility is not None:
            env["CUDA_VISIBLE_DEVICES"] = visibility
        env["ASTR_PROBE_OUTPUT"] = str(directory)
        env["ASTR_PROBE_RENDER_BACKEND"] = args.render_backend
        launch = [sys.executable, str(wrapper.resolve()), "--"]
        if software:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [str(args.software_library_dir.resolve()),
                                                                  env.get("LD_LIBRARY_PATH")]))
            env["LP_NUM_THREADS"] = "2"
            env["GALLIUM_DRIVER"] = "llvmpipe"
            subprocess.run([sys.executable, "-c", "import ctypes; ctypes.CDLL('libOSMesa.so.8')"],
                           env=env, check=True, timeout=15)
            launch = []
        command = [str(args.mpiexec.resolve()), "--mca", "coll_hcoll_enable", "0",
                   "-np", str(ranks), *launch,
                   str(args.probe.resolve()), str(args.pipeline.resolve()), str(args.backend.resolve())]
        start = time.monotonic()
        with (directory / "run.log").open("w") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=120)
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect {directory / 'run.log'}")
        records = [json.loads((directory / f"rank{rank}.json").read_text()) for rank in range(ranks)]
        mappings = [] if software else [json.loads((directory / f"mapping{rank}.json").read_text()) for rank in range(ranks)]
        if software:
            for record in records:
                if record["window_class"] != "vtkOSOpenGLRenderWindow" or "llvmpipe" not in record["capabilities"].lower():
                    raise RuntimeError("Explicit software renderer was not used")
                libraries = [Path(p) for p in record["runtime_libraries"] if "libOSMesa" in p]
                if not libraries or any(p.parent != args.software_library_dir.resolve() for p in libraries):
                    raise RuntimeError("OSMesa library identity not recorded")
        for record, mapping in zip(records, mappings):
            if record["actual_egl_uuid"] != mapping["uuid"]:
                raise RuntimeError("Actual context UUID mismatch")
        if sum(record["mesh_cells"] for record in records) != 3375:
            raise RuntimeError("Incorrect distributed mesh cell count")
        if sum(record["contour_cells"] for record in records) != 1052:
            raise RuntimeError("Incorrect distributed contour cell count")
        if (directory / "sphere.jpg").stat().st_size < 1000:
            raise RuntimeError("Missing or unexpectedly small image")
        check_image(directory)
        summary.append({"variant": name, "np": ranks, "seconds": time.monotonic()-start,
                        "mapping": mappings, "actual_uuid": [r["actual_egl_uuid"] for r in records],
                        "gpu_memory_after_screenshot": [r["gpu_memory_after_screenshot"] for r in records],
                        "peak_rss_kib_per_rank": [r["peak_rss_kib"] for r in records]})
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"PASS {name}: explicit {args.render_backend} backend verified", flush=True)


if __name__ == "__main__":
    main()
