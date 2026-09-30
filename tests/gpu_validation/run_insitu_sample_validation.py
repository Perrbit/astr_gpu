"""Bounded completed-step sampling and optional persistent Catalyst bridge gate."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np


def read_sample(path, canonical=False):
    with path.open("rb") as stream:
        if stream.read(8) != (b"ASTRIC01" if canonical else b"ASTRIS01"):
            raise ValueError("Invalid sample signature")
        header = np.fromfile(stream, dtype="<i4", count=9)
        clock = np.fromfile(stream, dtype="<f8", count=2)
        shape = tuple(header[3:6])
        points = int(np.prod(shape))
        xyz = np.fromfile(stream, dtype="<f8", count=3*points).reshape((*shape, 3), order="F")
        fields = np.fromfile(stream, dtype="<f8", count=11*points).reshape((*shape, 11), order="F")
        if stream.read(1) or header[0] != 1 or not np.isfinite(fields).all():
            raise ValueError("Invalid sample payload")
    step = int(header[1])
    for axis in range(3):
        line = (np.arange(shape[axis])+header[6+axis])*(2*np.pi/32)
        broadcast_shape = [1, 1, 1]
        broadcast_shape[axis] = shape[axis]
        np.testing.assert_allclose(xyz[..., axis], np.broadcast_to(line.reshape(broadcast_shape), shape),
                                   rtol=0, atol=2e-15)
    np.testing.assert_allclose(clock, [step*1e-3, 1e-3 if step else 0.], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(fields[..., 0], fields[..., 5])
    rho, velocity = fields[..., 5], fields[..., 6:9]
    np.testing.assert_allclose(fields[..., 1:4]/rho[..., None], velocity, rtol=0, atol=2e-13)
    pressure = (fields[..., 4]-.5*rho*np.sum(velocity*velocity, axis=-1))*(1.4-1)
    np.testing.assert_allclose(pressure, fields[..., 9], rtol=0, atol=2e-13)
    np.testing.assert_allclose(pressure/rho*(1.4*.1**2), fields[..., 10], rtol=0, atol=2e-13)
    return header, xyz, fields


def read_derived(path):
    with path.open('rb') as stream:
        if stream.read(8) != b'ASTRID01':
            raise ValueError('Invalid diagnostic signature')
        header = np.fromfile(stream, '<i4', 9)
        clock = np.fromfile(stream, '<f8', 2)
        values = np.fromfile(stream, '<f8').reshape((*header[3:6], 14), order='F')
    np.testing.assert_allclose(clock, [header[1]*1e-3, 1e-3 if header[1] else 0.], atol=1e-15, rtol=0)
    return header, values


def diagnostics(gradient):
    q = -.5*np.einsum('...ij,...ji->...', gradient, gradient)
    div = np.trace(gradient, axis1=-2, axis2=-1)
    curl = np.stack([gradient[...,2,1]-gradient[...,1,2],
                     gradient[...,0,2]-gradient[...,2,0],
                     gradient[...,1,0]-gradient[...,0,1]], axis=-1)
    flat = np.stack([gradient[...,i,j] for j in range(3) for i in range(3)], axis=-1)
    return np.concatenate([flat, q[...,None], div[...,None], curl], axis=-1)


def check_diagnostics(case, ranks):
    h = 2*np.pi/32
    symbol = (1.5*np.sin(h)-.3*np.sin(2*h)+np.sin(3*h)/30)/h
    steps = sorted(int(p.name.split('.step')[1].split('.')[0])
                   for p in (case/'outdat').glob('sample.step*.rank00000000.bin'))
    for step in steps:
        global_velocity = np.empty((32,32,32,3))
        global_q = np.empty((32,32,32,5))
        ownership = np.zeros((32,32,32), dtype=int)
        parts = []
        for rank in range(ranks):
            suffix = f'.step{step:08d}.rank{rank:08d}.bin'
            header, xyz, values = read_sample(case/'outdat'/('sample'+suffix))
            offset, extent = header[6:9], header[3:6]-1
            region = tuple(slice(int(o),int(o+n)) for o,n in zip(offset,extent))
            global_q[region] = values[:-1,:-1,:-1,0:5]
            ownership[region] += 1
            parts.append((suffix,header,xyz))
        np.testing.assert_array_equal(ownership, np.ones_like(ownership))
        global_velocity = global_q[...,1:4]/global_q[...,0,None]
        gradient = np.stack([(.75*(np.roll(global_velocity,-1,a)-np.roll(global_velocity,1,a))
                              -.15*(np.roll(global_velocity,-2,a)-np.roll(global_velocity,2,a))
                              +(np.roll(global_velocity,-3,a)-np.roll(global_velocity,3,a))/60)/h
                             for a in range(3)], axis=-1)
        reference = diagnostics(gradient)
        for suffix,header,xyz in parts:
            indexes = [np.arange(int(o),int(o+n)) % 32 for o,n in zip(header[6:9],header[3:6])]
            canonical_header, _, canonical = read_sample(case/'outdat'/('sample.canonical'+suffix), canonical=True)
            np.testing.assert_array_equal(canonical_header, header)
            expected_q = global_q[np.ix_(*indexes)]
            np.testing.assert_array_equal(canonical[...,0:5], expected_q)
            if canonical[...,0:5].tobytes() != expected_q.tobytes():
                raise ValueError('Canonical ownership must copy conserved values bitwise')
            derived_header, actual = read_derived(case/'outdat'/('sample.derived'+suffix))
            np.testing.assert_array_equal(derived_header, header)
            np.testing.assert_allclose(actual, reference[np.ix_(*indexes)], atol=2e-10, rtol=0)
            if step == 1:
                a = np.zeros((*xyz.shape[:3],3,3))
                a[...,0,0] = symbol*np.cos(xyz[...,0])
                a[...,0,1] = symbol*np.cos(xyz[...,1])
                a[...,1,0] = -symbol*np.cos(xyz[...,0])
                a[...,1,1] = symbol*np.cos(xyz[...,1])
                a[...,2,2] = symbol*np.cos(xyz[...,2])
                _, manufactured = read_derived(case/'outdat'/f'sample.manufactured.rank{header[2]:08d}.bin')
                np.testing.assert_allclose(manufactured, diagnostics(a), atol=2e-12, rtol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cpu", "gpu", "output", "mpiexec"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument('--topology', choices=('2,1,1','1,2,1','1,1,2'), default='2,1,1')
    parser.add_argument('--skip-single', action='store_true')
    parser.add_argument('--render-backend-dir', type=Path)
    parser.add_argument('--render-initial', action='store_true')
    parser.add_argument('--render-final', action='store_true')
    parser.add_argument('--on-demand', action='store_true')
    parser.add_argument('--statistics', action='store_true', help='accumulate every step over clipped window [0.0005,0.0025]')
    parser.add_argument('--statistics-roundtrip', action='store_true', help='release and restore device statistics after step 1')
    parser.add_argument('--observe-resources', action='store_true', help='sample local GPU on/off process memory against approved D8 limits')
    cadence = parser.add_mutually_exclusive_group()
    cadence.add_argument('--render-step-interval', type=int)
    cadence.add_argument('--render-time-interval', type=float)
    args = parser.parse_args()
    if args.statistics_roundtrip and not args.statistics:
        parser.error('--statistics-roundtrip requires --statistics')
    if args.statistics:
        args.render_initial = True
    if args.on_demand and not args.render_backend_dir:
        parser.error('--on-demand requires --render-backend-dir')
    expected_render = {}
    if args.render_time_interval is not None:
        if not np.isfinite(args.render_time_interval) or args.render_time_interval<=0:
            parser.error('render time interval must be finite and positive')
        ordinal=1
        for step in (1,2,3):
            crossed=0
            while ordinal*args.render_time_interval<=step*1.e-3:
                ordinal+=1
                crossed+=1
                if ordinal>10000:
                    parser.error('bounded test permits at most 10000 targets')
            if crossed:
                expected_render[step]=crossed
    else:
        interval=args.render_step_interval if args.render_step_interval is not None else 1
        if interval<=0:
            parser.error('render step interval must be positive')
        expected_render={s:1 for s in (1,2,3) if s%interval==0}
    if args.render_initial:
        expected_render={0:0, **expected_render}
    if args.render_final:
        expected_render.setdefault(3,0)
    if args.render_backend_dir and not expected_render:
        parser.error('bounded render gate requires at least one frame')
    root = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "two_rank_topology": args.topology, "checks": []}
    if args.render_backend_dir:
        report['render_schedule'] = {
            'mode': 'time' if args.render_time_interval is not None else 'steps',
            'interval': args.render_time_interval if args.render_time_interval is not None else interval,
            'origin_step': 0, 'origin_time': 0.0, 'initial': args.render_initial, 'final': args.render_final,
            'expected_crossed_targets_by_step': expected_render}
        report['on_demand'] = args.on_demand
    env = {k:v for k,v in os.environ.items() if not k.startswith("ASTR_")}
    env.update(ASTR_GPU_SYNC_MODE="explicit", ASTR_GPU_HALO_TRANSPORT="pinned",
               ASTR_GPU_PRECISION_MODE="fp64")

    def run(command, directory, log, runtime, observe=False, baseline=None):
        with log.open("w") as stream:
            if observe:
                from insitu_resource_monitor import run_monitored
                return run_monitored(command, directory, runtime, stream,
                                     directory/'resources.json', baseline)
            subprocess.run(command, cwd=directory, env=runtime, stdout=stream,
                           stderr=subprocess.STDOUT, check=True, timeout=180)

    try:
        for ranks in ((2,) if args.skip_single else (1, 2)):
            env["ASTR_FORCE_MPI_TOPOLOGY"] = args.topology if ranks==2 else '1,1,1'
            for backend, exe in (("cpu", args.cpu), ("gpu", args.gpu)):
                baseline_resources = None
                for sampling in (False, True):
                    case = (args.output/f"np{ranks}_{backend}_{'on' if sampling else 'off'}").resolve()
                    run([sys.executable, str(root/"tests/gpu_validation/prepare_tgv_case.py"),
                         "--src-case", str(root/"examples/Taylor_Green_Vortex"),
                         "--dst-case", str(case), "--use-gpu", "t" if backend=="gpu" else "f",
                         "--grid", "32,32,32", "--maxstep", "2", "--feqchkpt", "2",
                         "--deltat", "1.d-3", "--lfilter", "t", "--diffterm", "t", "--scheme", "643e"],
                        root, args.output/(case.name+"_prepare.log"), env)
                    runtime = dict(env)
                    launch = []
                    if sampling:
                        runtime["ASTR_INSITU_SAMPLE_PREFIX"] = "outdat/sample"
                        runtime['ASTR_INSITU_TEST_INITIAL']=str(int(args.render_initial))
                        runtime['ASTR_INSITU_TEST_FINAL']=str(int(args.render_final))
                        if args.statistics:
                            runtime['ASTR_INSITU_TEST_STATISTICS_WINDOW']='0.0005 0.0025'
                            if args.statistics_roundtrip:
                                runtime['ASTR_INSITU_TEST_STATISTICS_ROUNDTRIP']='1'
                    if sampling and backend=="gpu" and args.render_backend_dir:
                        (case/'outdat').mkdir(exist_ok=True)
                        for key in ('DISPLAY','PYTHONPATH','CATALYST_IMPLEMENTATION_PREFER_ENV'):
                            runtime.pop(key, None)
                        runtime.update(ASTR_INSITU_TEST_BACKEND=str(args.render_backend_dir.resolve()),
                                       ASTR_INSITU_TEST_PIPELINE=str(root/'tests/gpu_validation/insitu_tgv_pipeline.py'),
                                       ASTR_PROBE_OUTPUT=str(case/'outdat'))
                        runtime['ASTR_INSITU_TEST_EXPECTED_STEPS']=json.dumps(list(expected_render))
                        runtime['ASTR_INSITU_TEST_ON_DEMAND']=str(int(args.on_demand))
                        if args.render_time_interval is not None:
                            runtime['ASTR_INSITU_TEST_TIME_INTERVAL']=repr(args.render_time_interval)
                        else:
                            runtime['ASTR_INSITU_TEST_STEP_INTERVAL']=str(interval)
                        launch = [sys.executable, str(root/'tests/gpu_validation/insitu_device_identity.py'), '--']
                    resources = run([str(args.mpiexec.resolve()), "--mca", "coll_hcoll_enable", "0", "-np", str(ranks),
                         *launch, str(exe.resolve()), "run", "datin/input.tgv"], case, case/"run.log", runtime,
                         observe=args.observe_resources and backend=='gpu', baseline=baseline_resources)
                    if not sampling:
                        baseline_resources = resources
                    if "The job is done!" not in (case/"run.log").read_text():
                        raise ValueError("Solver did not complete")
                    samples = sorted((case/"outdat").glob("sample.step*.bin"))
                    expected_samples=(3+int(args.render_initial)) if sampling else 0
                    if launch and args.on_demand and not args.render_final and not args.statistics:
                        expected_samples=len(expected_render)
                    if len(samples) != expected_samples*ranks:
                        raise ValueError("Incorrect sample count")
                    for path in samples:
                        read_sample(path)
                    if sampling:
                        check_diagnostics(case, ranks)
                        if args.statistics:
                            from insitu_statistics_reference import check_case
                            report['checks'].append({'np':ranks,'backend':backend,
                                'statistics_oracle':check_case(case,ranks,read_sample,device=backend=='gpu')})
                    if launch:
                        from PIL import Image
                        for rank in range(ranks):
                            lifecycle = json.loads((case/'outdat'/f'lifecycle_rank{rank}.json').read_text())
                            if lifecycle != {'frames':list(expected_render), 'finalized':True}:
                                raise ValueError('Invalid backend lifecycle')
                        if len(list((case/'outdat').glob('q_surface.step*.jpeg'))) != len(expected_render):
                            raise ValueError('Unexpected number of rendered frames')
                        for step in expected_render:
                            records = [json.loads((case/'outdat'/f'mesh_step{step:08d}_rank{rank}.json').read_text())
                                       for rank in range(ranks)]
                            if sum(r['mesh_cells'] for r in records) != 32**3:
                                raise ValueError('Missing or duplicate distributed cells')
                            if any(r['crossed_targets'] != expected_render[step] for r in records):
                                raise ValueError('Incorrect number of crossed targets')
                            for product in ('q_surface','velocity_slice'):
                                if sum(r['products'][product]['local_cells'] for r in records) <= 0:
                                    raise ValueError('Empty visualization geometry')
                                with Image.open(case/'outdat'/f'{product}.step{step:08d}.jpeg') as image:
                                    rgb = np.asarray(image.convert('RGB'))
                                    if image.size != (800,600) or np.count_nonzero(np.min(rgb,axis=2)<220)<1000:
                                        raise ValueError('Blank or wrong-sized image')
                                    image.convert('RGB').save(case/'outdat'/f'{product}.step{step:08d}.eps')
                            report['checks'].append({'np':ranks,'step':step,'mesh_render':records})
                on = args.output/f"np{ranks}_{backend}_on/outdat/flowfield.h5"
                off = args.output/f"np{ranks}_{backend}_off/outdat/flowfield.h5"
                with h5py.File(on) as a, h5py.File(off) as b:
                    for field in ("ro", "u1", "u2", "u3", "p", "t", "nstep", "time"):
                        np.testing.assert_array_equal(a[field][...], b[field][...])
                        if a[field][...].tobytes() != b[field][...].tobytes():
                            raise ValueError(f"Sampling changed bit patterns in {field}")
                report["checks"].append(f"NP={ranks} {backend} sampling on/off output bitwise identical")
            maximum = np.zeros(11)
            if args.statistics:
                from insitu_statistics_reference import read_statistics, compare
                for gpu in sorted((args.output/f'np{ranks}_gpu_on/outdat').glob('sample.statistics.step*.bin')):
                    cpu = args.output/f'np{ranks}_cpu_on/outdat'/gpu.name
                    hc,mc,vc = read_statistics(cpu)
                    hg,mg,vg = read_statistics(gpu)
                    np.testing.assert_array_equal(hc,hg)
                    np.testing.assert_array_equal(mc[:3],mg[:3])
                    # Volume summation order differs between host and device reductions.
                    np.testing.assert_allclose(mc[3],mg[3],atol=2e-10,rtol=0)
                    np.testing.assert_allclose(mc[4:]**2,mg[4:]**2,atol=2e-10,rtol=0)
                    report['checks'].append({'np':ranks,'statistics_cpu_gpu':compare(vc,vg)})
            derived_maximum = np.zeros(14)
            for gpu in sorted((args.output/f"np{ranks}_gpu_on/outdat").glob("sample.step*.bin")):
                cpu = args.output/f"np{ranks}_cpu_on/outdat"/gpu.name
                h1, x1, f1 = read_sample(cpu)
                h2, x2, f2 = read_sample(gpu)
                np.testing.assert_array_equal(h1, h2)
                np.testing.assert_array_equal(x1, x2)
                np.testing.assert_allclose(f1, f2, atol=2e-10, rtol=0)
                maximum = np.maximum(maximum, np.max(np.abs(f1-f2), axis=(0,1,2)))
                _, d1 = read_derived(cpu.with_name(cpu.name.replace('sample.step','sample.derived.step')))
                _, d2 = read_derived(gpu.with_name(gpu.name.replace('sample.step','sample.derived.step')))
                np.testing.assert_allclose(d1,d2,atol=2e-10,rtol=0)
                derived_maximum = np.maximum(derived_maximum,np.max(np.abs(d1-d2),axis=(0,1,2)))
            report["checks"].append({"np": ranks, "cpu_gpu_max_abs_by_field": maximum.tolist(),
                                     "derived_max_abs_by_field": derived_maximum.tolist(),
                                     "manufactured_and_global_stencil": "passed"})
            print(f"PASS NP={ranks}: sampling phase, field consistency, CPU/GPU, on/off", flush=True)
        report["status"] = "passed-bounded-TGV-sampling-not-production-visualization"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (args.output/"summary.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
