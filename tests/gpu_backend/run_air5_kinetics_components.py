"""AIR5 source, analytic Jacobian, activity and LU6 against frozen native references."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from decimal import Decimal, localcontext

import numpy as np
from run_air5_properties_components import run, read as read_properties

ROOT = Path(__file__).resolve().parents[2]
CPU = ["src/chemistry_air5_data.F90", "src/chemistry_core.F90",
       "src/chemistry_properties.F90", "src/chemistry_kinetics.F90"]
GPU = ["src_gpu/gpu_check.cuf", "src_gpu/chemistry_core_gpu.cuf",
       "src_gpu/chemistry_relaxation_gpu.cuf", "src_gpu/chemistry_kinetics_gpu.cuf"]
INCLUDES = ["src/chemistry_pure_vt_body.inc", "src/chemistry_pure_source_body.inc",
            "src/chemistry_activity_index_body.inc", "src/chemistry_activity_gate_body.inc",
            "src/chemistry_vibrational_difference_body.inc"]
PROBE = "tests/gpu_backend/air5_kinetics_reference.F90"
GROUPS = [("source", 0, 0, 6), ("jacobian", 0, 6, 42), ("progress", 0, 42, 54),
          ("pure_source", 1, 54, 60), ("activity_source", 2, 60, 66),
          ("activity_terms", 2, 66, 80), ("activity_index", 2, 80, 90),
          ("lu", 3, 90, 126), ("solution", 4, 126, 132)]


def read(path):
    with path.open("rb") as stream:
        n = int(np.fromfile(stream, "<i4", 1)[0])
        st = np.fromfile(stream, "<i4", 11*n).reshape(n, 11)
        x = np.fromfile(stream, "<f8", 132*n).reshape(n, 132)
        assert stream.read() == b""
    return st, x


def compare(actual, reference):
    st, a = read(actual); sr, b = read(reference)
    assert np.array_equal(st[:, :5], sr[:, :5]), (actual, np.argwhere(st[:, :5] != sr[:, :5]))
    active = sr[:, 3] == 0
    assert np.array_equal(st[active, 5:], sr[active, 5:]), (actual, "LU pivot mismatch")
    result = {}
    for name, status, start, end in GROUPS:
        active = sr[:, status] == 0
        if not active.any():
            result[name] = None  # Invalid source modes deliberately evaluate no source values.
            continue
        av, bv = a[active, start:end], b[active, start:end]
        assert np.isfinite(av).all() and np.isfinite(bv).all(), (actual, name)
        scale = np.maximum(1., abs(bv).max(axis=1, keepdims=True))
        ratios = abs(av-bv)/(1e-13*scale+5e-12*abs(bv))
        maximum = float(ratios.max())
        result[name] = maximum
        assert maximum <= 1., (actual, name, maximum, np.unravel_index(ratios.argmax(), ratios.shape))
    return result


def compare_temperature(actual, reference):
    st, a = read(actual); sr, b = read(reference)
    assert np.array_equal(st, sr), (actual, "fixed-temperature statuses")
    assert np.all(st[:, :4] == 0)
    result = {}
    groups = [("energy_difference", 0, 5), ("pure_vt", 5, 6), ("joint_vt", 6, 7),
              ("vt_derivatives", 7, 13), ("dT", 13, 19), ("dTv", 19, 25)]
    for name, start, end in groups:
        av, bv = a[:, start:end], b[:, start:end]
        assert np.isfinite(av).all() and np.isfinite(bv).all()
        scale = np.maximum(1., abs(bv).max(axis=1, keepdims=True))
        maximum = float(np.max(abs(av-bv)/(1e-13*scale+5e-12*abs(bv))))
        assert maximum <= 1., (actual, name, maximum)
        result[name] = maximum
    return result


def check_energy_difference(x, values):
    data = json.loads((ROOT/"chemMech/air5_kimjo12.json").read_text())
    maximum = 0.
    with localcontext() as ctx:
        ctx.prec = 80
        ru = Decimal.from_float(8.3144626181532403)
        for row, output in zip(x, values):
            t, tv = map(Decimal.from_float, row[11:13])
            for s, item in enumerate(data["species"]):
                theta = Decimal.from_float(item["vibrational_temperature_k"])
                expected = Decimal(0)
                if theta:
                    r = ru/Decimal.from_float(item["molecular_weight_kg_per_mol"])
                    expected = r*theta*(1/((theta/t).exp()-1)-1/((theta/tv).exp()-1))
                if not expected:
                    assert output[s] == 0.
                else:
                    relative = abs(Decimal.from_float(output[s])-expected)/abs(expected)
                    maximum = max(maximum, float(relative))
                    assert relative <= Decimal("5e-12"), (s, t, tv, output[s], expected)
    return maximum


def prepare(directory, properties, compiler, component):
    directory.mkdir(parents=True, exist_ok=False)
    source = directory/"source"
    files = CPU+GPU+INCLUDES+[PROBE, "chemMech/air5_kimjo12.json"]
    for name in files:
        p = source/name; p.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name, p)
    commands = {}
    for backend in ["cpu", "old_gpu"]:
        build = directory/backend; build.mkdir()
        flags = [str(compiler), "-O2", "-Mpreprocess", "-Mextend", "-I"+str(source/"src")]
        if backend == "old_gpu": flags += ["-cuda", "-gpu=rdc,maxregcount:128", "-D_CUDA", "-DASTR_AIR5_REFERENCE_GPU"]
        sources = CPU+(GPU if backend == "old_gpu" else [])+[PROBE]
        commands[backend] = []; objects = []
        for i, name in enumerate(sources):
            obj = build/(Path(name).stem+".o"); objects.append(str(obj))
            command = flags+["-c", str(source/name), "-o", str(obj)]
            commands[backend].append(command); run(command, build, build/f"compile{i}.log")
        command = flags+objects+["-o", str(build/"probe")]
        commands[backend].append(command); run(command, build, build/"link.log")
    properties_report = json.loads((properties/"report.json").read_text())
    _, prop = read_properties(properties/"cpu_all.bin")
    q = prop[:properties_report["valid_cases"], :11].copy()
    base = q[112].copy()
    extra = []
    for trace in [0., 1e-20, 6.301775586262159e-165, 1e-250, np.finfo(float).tiny,
                  np.nextafter(0., 1.)]:
        case = base.copy(); case[5] += case[9]-trace; case[9] = trace; extra.append(case)
    for index, value in [(0, 0.), (0, -1.), (1, np.nan), (4, np.inf), (6, -1.),
                          (10, -1.), (4, -1e20)]:
        case = base.copy(); case[index] = value; extra.append(case)
    case = base.copy(); case[5:10] = [0., 0., .6*case[0], .4*case[0], 0.]; extra.append(case)
    q = np.concatenate([q, extra])
    x = np.zeros((len(q), 53), dtype="<f8"); x[:, :11] = q
    rng = np.random.default_rng(851206)
    for i in range(len(q)):
        matrix = rng.normal(size=(6, 6))+3.*np.eye(6)
        matrix *= 10.**rng.uniform(-10., 10., size=(6, 1))
        if i == len(q)-3: matrix[0] = 0.
        if i == len(q)-2: matrix[2] = matrix[1]
        if i == len(q)-1: matrix[0, 0] = np.nan
        x[i, 11:47] = matrix.ravel(order="F")
        x[i, 47:] = matrix @ np.array([1., -2., 3., -4., 5., -6.])
    with (directory/"input.bin").open("wb") as stream:
        if component == "temperature":
            with (properties/"input.bin").open("rb") as prop_stream:
                count = int(np.fromfile(prop_stream, "<i4", 1)[0])
                props = np.fromfile(prop_stream, "<f8").reshape(count, 41)
            n = properties_report["valid_cases"]
            x = x[:n].copy(); x[:, 11:] = 0
            x[:, 11] = props[:n, 4]; x[:, 12] = props[:n, 10]
            extra = []
            for t, tv in [(300., np.nextafter(300., 8000.)), (np.nextafter(300., 8000.), 300.),
                          (np.nextafter(8000., 300.), 8000.), (8000., np.nextafter(8000., 300.))]:
                row = x[112].copy(); row[11:13] = [t, tv]; extra.append(row)
            x = np.concatenate([x, extra])
        np.array([len(x)], dtype="<i4").tofile(stream); x.tofile(stream)
    report = dict(cases=len(x), component=component, build_commands=commands, properties_reference=str(properties),
                  source_sha256={name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in files},
                  modes={})
    (directory/"reference_identity.json").write_text(json.dumps(report, indent=2)+"\n")
    modes = {"temperature": [99], "linear": [-1], "chemical": [1]}.get(component, range(-1, 5))
    for mode in modes:
        for backend in ["cpu", "old_gpu"]:
            run([directory/backend/"probe", directory/"input.bin", directory/f"{backend}_{mode}.bin", mode],
                directory, directory/f"{backend}_{mode}.log")
        comparison = compare_temperature if component == "temperature" else compare
        report["modes"][mode] = comparison(directory/f"old_gpu_{mode}.bin", directory/f"cpu_{mode}.bin")
        if component == "temperature":
            for backend in ["cpu", "old_gpu"]:
                _, values = read(directory/f"{backend}_{mode}.bin")
                report[f"{backend}_energy_difference_80digit_max_relative"] = check_energy_difference(x, values)
    (directory/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report["modes"], indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--references", type=Path)
    p.add_argument("--properties-reference", type=Path)
    p.add_argument("--compiler", type=Path)
    p.add_argument("--candidate", type=Path)
    p.add_argument("--component", choices=("all", "linear", "temperature", "chemical"), default="all")
    args = p.parse_args()
    out = args.output.resolve()
    if args.references is None:
        if args.compiler is None or args.properties_reference is None: p.error("reference preparation needs --compiler and --properties-reference")
        prepare(out, args.properties_reference.resolve(), args.compiler.resolve(), args.component)
        return
    if args.candidate is None: p.error("candidate comparison needs --candidate")
    ref = args.references.resolve(); out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.candidate, out/"candidate")
    shutil.copy2(__file__, out/"driver.py")
    for name in ["src_gpu/chemistry_kinetics_math.cuh", "src_gpu/chemistry_properties_math.cuh",
                 "src_gpu/chemistry_air5_data.h", "tests/gpu_backend/air5_kinetics_probe.cu"]:
        shutil.copy2(ROOT/name, out/Path(name).name)
    report = dict(reference_directory=str(ref), component=args.component, modes={})
    modes = {"temperature": [99], "linear": [-1], "chemical": [1]}.get(args.component, range(-1, 5))
    comparison = compare_temperature if args.component == "temperature" else compare
    for mode in modes:
        actual = out/f"cpp_{mode}.bin"
        run([out/"candidate", ref/"input.bin", actual, mode], out, out/f"{mode}.log")
        report["modes"][mode] = {
            "cpu": comparison(actual, ref/f"cpu_{mode}.bin"),
            "old_gpu": comparison(actual, ref/f"old_gpu_{mode}.bin")}
    with (ref/"input.bin").open("rb") as stream:
        n = int(np.fromfile(stream, "<i4", 1)[0]); x = np.fromfile(stream, "<f8").reshape(n, 53)
    if args.component == "temperature":
        _, values = read(out/"cpp_99.bin")
        report["energy_difference_80digit_max_relative"] = check_energy_difference(x, values)
        (out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2))
        return
    st, values = read(out/f"cpp_{-1 if args.component == 'linear' else (1 if args.component == 'chemical' else 0)}.bin")
    active = st[:, 4] == 0
    matrices = x[active, 11:47].reshape(-1, 6, 6).transpose(0, 2, 1)
    rhs = x[active, 47:]; solution = values[active, 126:132]
    # Row-scaled residual avoids hiding a small row beneath a large one.
    scale = abs(matrices).max(axis=2)
    residual = abs(np.einsum("nij,nj->ni", matrices/scale[:, :, None], solution)-rhs/scale)
    denominator = np.maximum(1., abs(rhs/scale)+np.einsum("nij,nj->ni", abs(matrices)/scale[:, :, None], abs(solution)))
    report["lu_max_row_scaled_relative_residual"] = float(np.max(residual/denominator))
    assert report["lu_max_row_scaled_relative_residual"] <= 1e-12
    assert np.max(abs(solution-np.array([1., -2., 3., -4., 5., -6.]))) <= 1e-10
    report["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob("*.cu*")}
    report["candidate_sha256"] = hashlib.sha256((out/"candidate").read_bytes()).hexdigest()
    (out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
