"""Freeze CPU/CUDA Fortran AIR5 properties, then compare the C++ device equations."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ["src/chemistry_air5_data.F90", "src/chemistry_core.F90",
           "src/chemistry_properties.F90", "src_gpu/chemistry_core_gpu.cuf",
           "src_gpu/chemistry_flow_state_gpu.cuf", "src_gpu/chemistry_transport_gpu.cuf",
           "tests/gpu_backend/air5_properties_reference.F90", "chemMech/air5_kimjo12.json",
           "src/chemistry_pure_vt_body.inc", "src/chemistry_vibrational_difference_body.inc"]
GROUPS = [("q", 1, 0, 11), ("primitive", 2, 11, 23), ("pressure", 0, 23, 24),
          ("dT", 3, 24, 30), ("dTv", 4, 30, 36), ("transport", 5, 36, 74),
          ("flux", 6, 74, 104), ("reuse", 6, 104, 114)]


def run(command, directory, log):
    with log.open("w") as stream:
        subprocess.run(list(map(str, command)), cwd=directory, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def make_cases():
    cases = []
    compositions = [[.7653, .2347, 0., 0., 0.], [.7, .2, .04, .03, .03],
                    [1.-4e-14, 1e-14, 1e-14, 1e-14, 1e-14], [1., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0.], [0., 0., 0., 0., 1.], [.1, .1, .3, .3, .2]]
    for t in [300., 500., 2200., 5000., 8000.]:
        for tv in [300., 1200., 4000., 7600., 8000.]:
            for y in compositions:
                rho = 1.e5/(t*sum(a*8.31446261815324/m for a, m in zip(y, [.028,.032,.014,.016,.030])))
                cases.append([rho, 180., -20., 11., t, *y, tv])
    rng = np.random.default_rng(851112)
    for _ in range(256):
        y = rng.dirichlet(np.ones(5))
        y[0] = 1.-sum(y[1:])
        t, tv = rng.uniform(301., 7999., 2)
        p = np.exp(rng.uniform(np.log(1100.), np.log(9e5)))
        rho = p/(t*sum(y*8.31446261815324/np.array([.028,.032,.014,.016,.030])))
        cases.append([rho, *rng.uniform(-1000., 1000., 3), t, *y, tv])
    valid_count = len(cases)
    base = [.09, 180., -20., 11., 5100., .42, .12, .18, .18, .1, 3600.]
    for position, value in [(0, 0.), (0, -1.), (0, np.nan), (1, np.inf), (4, 299.),
                             (4, 8001.), (4, np.nan), (10, 299.), (10, 8001.),
                             (10, np.nan), (5, -.1), (5, .5)]:
        case = base.copy(); case[position] = value; cases.append(case)
    case = base.copy(); case[5:10] = [0., 0., .6, .4, 0.]; cases.append(case)
    values = np.zeros((len(cases), 41), dtype="<f8")
    values[:, :11] = cases
    values[:, 11:] = rng.uniform(-.2, .2, (len(cases), 30))
    # A true composition gradient sums to zero. No correction of cell states.
    for d in range(3):
        offset = 26+5*d
        values[:, offset] = -values[:, offset+1:offset+5].sum(axis=1)
    return values, valid_count


def read(path):
    with path.open("rb") as stream:
        n = int(np.fromfile(stream, "<i4", 1)[0])
        status = np.fromfile(stream, "<i4", 7*n).reshape(n, 7)
        values = np.fromfile(stream, "<f8", 114*n).reshape(n, 114)
        assert stream.read() == b""
    return status, values


def compare(actual, reference):
    st, values = read(actual); sr, ref = read(reference)
    assert np.array_equal(st, sr), (actual, np.argwhere(st != sr).tolist())
    result = {}
    for name, index, start, stop in GROUPS:
        active = sr[:, index] == 0
        a, b = values[active, start:stop], ref[active, start:stop]
        assert np.isfinite(a).all() and np.isfinite(b).all(), (actual, name)
        # The existing chemistry_flow_gpu_probe absolute + relative budget.
        scale = np.maximum(np.max(abs(b), axis=1, keepdims=True), 1.)
        ratio = np.max(abs(a-b)/(1e-13*scale+5e-12*abs(b)))
        result[name] = float(ratio)
        assert ratio <= 1., (actual, name, ratio)
    return result


def run_context(candidate, reference_directory, output):
    """Reuse the frozen references; exercise context ownership and face boxes."""
    out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=False)
    refdir = Path(reference_directory).resolve()
    report = json.loads((refdir/"report.json").read_text())
    n = report["valid_cases"]
    with (refdir/"input.bin").open("rb") as stream:
        count = int(np.fromfile(stream, "<i4", 1)[0])
        x = np.fromfile(stream, "<f8").reshape(count, 41)[:n]
    with (out/"input.bin").open("wb") as stream:
        np.array([n], dtype="<i4").tofile(stream); x.tofile(stream)
    shutil.copy2(candidate, out/"candidate")
    state_sources = ["src_gpu/chemistry_properties.cu", "src_gpu/chemistry_properties.F90",
                     "src_gpu/chemistry_properties.h", "src_gpu/chemistry_properties_math.cuh",
                     "src_gpu/chemistry_air5_data.h", "src_gpu/device_state.cu",
                     "src_gpu/device_state_internal.cuh", "src_gpu/device_state.h",
                     "src_gpu/device_state.F90", "src_gpu/CMakeLists.txt",
                     "tests/gpu_backend/air5_state_probe.cu"]
    for name in state_sources:
        path = out/"source"/name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, path)
    results = {}
    for mode in ["closure", "all"]:
        path = out/f"cpp_{mode}.bin"
        run([out/"candidate", out/"input.bin", path, mode], out, out/f"{mode}.log")
        status, values = read(path)
        assert np.all(status == 0)
        _, reference = read(refdir/f"cpu_{mode}.bin")
        result = {}
        for name, _, start, stop in GROUPS[:3]:
            a, b = values[:, start:stop], reference[:n, start:stop]
            scale = np.maximum(np.max(abs(b), axis=1, keepdims=True), 1.)
            ratio = float(np.max(abs(a-b)/(1e-13*scale+5e-12*abs(b))))
            assert np.isfinite(a).all() and ratio <= 1., (mode, name, ratio)
            result[name] = ratio
        _, standalone = read(refdir/f"cpp_{mode}.bin")
        assert np.array_equal(values[:, :23], standalone[:n, :23])
        result["standalone_q_and_decoded_primitives_bitwise_equal"] = True
        result["temperature_roundtrip_relative"] = float(np.max(abs(values[:, 15]-x[:, 4])/x[:, 4]))
        result["tv_roundtrip_relative"] = float(np.max(abs(values[:, 21]-x[:, 10])/x[:, 10]))
        assert result["temperature_roundtrip_relative"] <= 1e-12
        assert result["tv_roundtrip_relative"] <= 1e-12
        results[mode] = result
    results["reference_directory"] = str(refdir)
    results["reference_report_sha256"] = hashlib.sha256((refdir/"report.json").read_bytes()).hexdigest()
    results["candidate_sha256"] = hashlib.sha256((out/"candidate").read_bytes()).hexdigest()
    results["source_sha256"] = {name: hashlib.sha256((out/"source"/name).read_bytes()).hexdigest()
                                 for name in state_sources}
    (out/"report.json").write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--nvfortran", required=True, type=Path)
    parser.add_argument("--context-candidate", type=Path)
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    source = out/"reference_source"
    for name in SOURCES:
        path = source/name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, path)
    for name in ["src_gpu/chemistry_properties_math.cuh", "src_gpu/chemistry_air5_data.h",
                 "tests/gpu_backend/air5_properties_probe.cu"]:
        shutil.copy2(ROOT/name, out/Path(name).name)
    shutil.copy2(__file__, out/"driver.py")
    shutil.copy2(args.candidate, out/"candidate")
    commands = {}
    for backend in ["cpu", "old_gpu"]:
        build = out/backend; build.mkdir()
        files = SOURCES[:3]
        flags = [str(args.nvfortran), "-O2", "-Mpreprocess", "-Mextend", "-I"+str(source/"src")]
        if backend == "old_gpu":
            files += SOURCES[3:6]
            flags += ["-cuda", "-gpu=rdc,maxregcount:128", "-D_CUDA", "-DASTR_AIR5_REFERENCE_GPU"]
        files += [SOURCES[6]]
        commands[backend] = []
        objects = []
        for index, name in enumerate(files):
            obj = build/(Path(name).stem+".o"); objects.append(obj)
            command = flags+["-c", str(source/name), "-o", str(obj)]
            commands[backend].append(command)
            run(command, build, build/f"compile{index}.log")
        command = flags+list(map(str, objects))+["-o", str(build/"probe")]
        commands[backend].append(command)
        run(command, build, build/"link.log")
    x, valid_count = make_cases()
    with (out/"input.bin").open("wb") as stream:
        np.array([len(x)], dtype="<i4").tofile(stream); x.tofile(stream)
    report = dict(cases=len(x), valid_cases=valid_count, build_commands=commands,
                  comparison_budget="1e-13*max(1,maxabs(reference group per state))+5e-12*abs(reference)",
                  modes={})
    for mode in ["closure", "all"]:
        for backend in ["cpu", "old_gpu", "cpp"]:
            exe = out/"candidate" if backend == "cpp" else out/backend/"probe"
            run([exe, out/"input.bin", out/f"{backend}_{mode}.bin", mode], out, out/f"{backend}_{mode}.log")
        ref = out/f"cpu_{mode}.bin"
        report["modes"][mode] = {
            "old_gpu_cpu": compare(out/f"old_gpu_{mode}.bin", ref),
            "cpp_cpu": compare(out/f"cpp_{mode}.bin", ref),
            "cpp_old_gpu": compare(out/f"cpp_{mode}.bin", out/f"old_gpu_{mode}.bin")}
        st, values = read(out/f"cpp_{mode}.bin")
        assert np.all(st[:valid_count] == 0), np.argwhere(st[:valid_count] != 0)
        y = values[:valid_count, 16:21]
        assert np.all(y[x[:valid_count, 5:10] > 0] > 0)
        assert np.max(abs(y.sum(axis=1)-1.)) <= 128*np.finfo(float).eps
        sf = values[:valid_count, 83:98].reshape(valid_count, 3, 5)
        closure = abs(sf.sum(axis=2))
        assert np.all(closure <= 128*np.finfo(float).eps*np.maximum(1., abs(sf).sum(axis=2)))
        report["modes"][mode]["species_flux_closure_abs"] = float(closure.max())
    report["source_sha256"] = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in source.rglob("*") if p.is_file()}
    report["binary_sha256"] = {name: hashlib.sha256((out/name).read_bytes()).hexdigest()
                               for name in ["candidate", "cpu/probe", "old_gpu/probe"]}
    (out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report["modes"], indent=2))
    if args.context_candidate:
        run_context(args.context_candidate, out, out/"context")


if __name__ == "__main__":
    main()
