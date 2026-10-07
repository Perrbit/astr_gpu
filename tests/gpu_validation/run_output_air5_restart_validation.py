"""Bounded completed-step AIR5 HBL/SBLI restart gate, not a physical benchmark."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess

import h5py
import numpy as np

from air5_radau_reference import Air5RadauReference
from prepare_air5_c4_case import prepare_case, replace_boundary_types
from run_air5_sbli_preflight import set_value
from run_output_restart_validation import archive_schedule_payload, compare_fields, controlled_checkpoint_buffers
from check_air5_mass_closure_stages import metrics
from generate_air5_oblique_shock_states import frozen_oblique_jump, jump_metadata

ROOT = Path(__file__).resolve().parents[2]
DT = 1e-10
DOMAIN = (.08, .01, .002)
REFERENCE_DENSITY = .05
REFERENCE_TEMPERATURE = 3000.
REFERENCE_MASS_FRACTIONS = (.767, .233, 0., 0., 0.)


def reference_scales():
    """Frozen input/mechanism scales, never estimated from computed fields."""
    model = Air5RadauReference(ROOT / "chemMech/air5_kimjo12.json")
    ys = np.array(REFERENCE_MASS_FRACTIONS)
    gas = float(ys @ model.gas_constant)
    gamma = 1 + gas / float(ys @ model.cv_tr)
    sound = float(np.sqrt(gamma * gas * REFERENCE_TEMPERATURE))
    pressure = REFERENCE_DENSITY * gas * REFERENCE_TEMPERATURE
    return dict(density=REFERENCE_DENSITY, temperature=REFERENCE_TEMPERATURE,
                velocity=sound, pressure=pressure, shear=pressure, heat=pressure*sound)


def compare_air5_resources(continuous, resumed, incident):
    names = [f"air5_hbl_{name}.dat" for name in ("domain", "profile", "initial_field")]
    if incident:
        names.append("air5_incident_shock.dat")
    sizes = {}
    for name in names:
        expected = (continuous / "outdat/new/resources" / name).read_bytes()
        if expected != (resumed / "outdat/new/resources" / name).read_bytes():
            raise AssertionError(f"frozen AIR5 resource differs: {name}")
        if (resumed / "datin" / name).exists():
            raise AssertionError(f"original AIR5 resource remains: {name}")
        sizes[name] = len(expected)
    geometry = "outdat/new/resources/geometry.h5"
    return dict(file_bytes=sizes,
                geometry_datasets=compare_fields(continuous / geometry, resumed / geometry))


def state_sanity(path):
    with h5py.File(path) as state:
        q = np.stack([state[f"q{m:04}"][:] for m in range(1, 12)], axis=-1)
        row = metrics(q, (0, 0, 0))
        if row["min_species_density"] < 0 or row["transport_sum_violations"]:
            raise AssertionError("completed AIR5 state violates species positivity/closure")
        for m in range(12, 35):
            values = state[f"q{m:04}"][:]
            if not np.isfinite(values).all():
                raise AssertionError(f"nonfinite physical cache/carry component {m}")
            if m in (23, 27, 28, 29) and np.any(values <= 0):
                raise AssertionError(f"nonpositive physical cache component {m}")
        return {key: row[key] for key in ("min_species_density", "transport_sum_violations",
                                         "max_extended_relative_residual")}


def prepare(case, backend, steps, incident=False, reconstruction=5, mean_statistics=False, sample_interval=2):
    if case.exists():
        raise FileExistsError(case)
    inp = prepare_case(ROOT / "examples/Taylor_Green_Vortex_SI/datin", case,
                       "16,16,16", steps - 1, "1.d-10", "t", "t",
                       "t" if backend == "gpu" else "f", "species-wave", list_frequency=1)
    set_value(inp, "flowtype", "air5sbli" if incident else "air5hbl")
    set_value(inp, "lihomo,ljhomo,lkhomo", "f,f,t")
    set_value(inp, "lrestar", "f")
    set_value(inp, "conschm,difschm,rkscheme,odetype", "643e,643e,rk3")
    set_value(inp, "recon_schem, lchardecomp,bfacmpld,shkcrt", f"{reconstruction},f,0.3d0,0.05d0")
    lines = inp.read_text().splitlines()
    replace_boundary_types(lines, ("11,free", 50, "41,3000.d0", 51, 1, 1))
    inp.write_text("\n".join(lines) + "\n")
    set_value(case / "datin/controller", "lwsequ,lwslic,lavg,lcracon",
              f"f,f,{'t' if mean_statistics else 'f'},f")
    set_value(case / "datin/controller", "maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg",
              f"{steps-1},1,1000000,1000000,1,{sample_interval}")
    model = Air5RadauReference(ROOT / "chemMech/air5_kimjo12.json")
    ys = np.array(REFERENCE_MASS_FRACTIONS)
    gas = float(ys @ model.gas_constant)
    rho, temperature = REFERENCE_DENSITY, REFERENCE_TEMPERATURE
    pressure = rho * gas * temperature
    sound = np.sqrt((1 + gas / float(ys @ model.cv_tr)) * gas * temperature)
    speed = 4 * sound if incident else 0.
    (case / "datin/air5_hbl_domain.dat").write_text(
        "air5_hbl_domain_v1\n" + " ".join(map(str, DOMAIN)) + "\n")
    rows = []
    for y in (0., .001, .01):
        tv = 1500. + 1500. * max(0., 1. - y / .001)
        rows.append([y, rho, speed * min(y/.001, 1.), 0., 0., pressure, temperature, tv, *ys])
    (case / "datin/air5_hbl_profile.dat").write_text(
        "# Bounded hot-gas restart test\n# x_origin=1.0\n" +
        "\n".join(" ".join(f"{v:.17e}" for v in row) for row in rows) + "\n")
    field = ["# ASTR_AIR5_HBL_INITIAL_FIELD_V1", "# Restart gate, not a production profile", "17 17"]
    for x in np.linspace(0., DOMAIN[0], 17):
        for y in np.linspace(0., DOMAIN[1], 17):
            v = .01 * np.sin(np.pi*x/DOMAIN[0]) * np.sin(np.pi*y/DOMAIN[1])
            momentum = rho * np.array([speed * min(y/.001, 1.), v, 0.])
            partial = rho * ys
            tv = 1500. + 1500. * max(0., 1. - y/.001)
            ev = model.ev_from_tv(partial, tv)
            energy = model.q5_from_state(rho, momentum, partial, ev, temperature)
            field.append(" ".join(f"{z:.17e}" for z in [x, y, rho, *momentum, energy, *partial, ev]))
    (case / "datin/air5_hbl_initial_field.dat").write_text("\n".join(field) + "\n")
    if incident:
        jump = frozen_oblique_jump(model, mach=4., temperature=temperature, tv=1500.,
                                   pressure=pressure, shock_angle_deg=25., mass_fraction=ys)
        meta = jump_metadata(jump, top_x=.035, top_y=DOMAIN[1])
        if (meta["max_scaled_normal_flux_residual"] > 2e-12 or
                not 0 < meta["geometric_wall_intersection_x"] < DOMAIN[0]):
            raise ValueError("bounded incident jump fails its geometry/flux contract")
        rows = [[meta["top_x"], meta["top_y"], *jump.normal], jump.upstream_q, jump.downstream_q]
        (case / "datin/air5_incident_shock.dat").write_text(
            "air5_incident_shock_v1\n" + "\n".join(
                " ".join(f"{value:.17e}" for value in row) for row in rows) + "\n")
        (case / "incident_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return DOMAIN[1] / sound


def launch(args, backend, ranks, name, steps, restore=None, interval=5, fault=None, statistics=None,
           archive_groups=None, buffer_bytes=67108864, checkpoint_keep=2, reuse_root=None,
           conservation=False, publication_hook=None, reject=None, memcheck=False,
           device_budget_bytes=67108864, restore_probe=False, device_reserve_bytes=0, override=False,
           wall_samples=False, insitu_config=None, directory_budget_bytes=64*1024**2,
           device_sample_transport=None, postprocess_transport=None, checkpoint_enabled=True,
           insitu_timing=False, nsys_trace=False, monitor_resources=False, resource_baseline=None, pixel_audit=False,
           wall_mean_oracle=False):
    case = args.output / f"{backend}_np{ranks}_{name}"
    tau = prepare(case, backend, steps, incident=args.case == "sbli", reconstruction=args.reconstruction,
                  mean_statistics=args.mean_statistics if statistics is None else statistics,
                  sample_interval=args.sample_interval)
    if fault == "statistics_activation":
        set_value(case / "datin/controller", "lwsequ,lwslic,lavg,lcracon",
                  f"f,f,{'f' if args.mean_statistics else 't'},f")
    elif fault == "sample_interval":
        changed_interval = 3 if args.sample_interval == 2 else 2
        set_value(case / "datin/controller", "maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg",
                  f"{steps-1},1,1000000,1000000,1,{changed_interval}")
    (case / "outdat/new").mkdir(parents=True)
    if reuse_root is not None:
        if restore is None:
            raise ValueError("reuse test needs an explicit checkpoint")
        shutil.copytree(reuse_root, case / "outdat/new", dirs_exist_ok=True)
        restore = case / "outdat/new/checkpoints" / restore.name
    prior_complete = {str(path.relative_to(case / "outdat/new")): path.read_bytes()
                      for path in (case / "outdat/new").rglob("COMPLETE")}
    (case / "datin/input.output").write_text(f"""&output
 directory='outdat/new',restore_directory='{restore or ''}',
 restart_output='{'override' if override else 'saved'}',
 host_budget_bytes=67108864,device_budget_bytes={1 if fault == 'statistics_device_budget' else device_budget_bytes},buffer_bytes={buffer_bytes}
 device_reserve_bytes={device_reserve_bytes}
/
&checkpoint
 enabled={'t' if checkpoint_enabled else 'f'},mode='{args.mode}',interval_steps={interval if args.mode == 'steps' else 0},
 interval_time={interval*DT if args.mode == 'time' else 0},keep={checkpoint_keep},initial_frame={'t' if args.initial_restart else 'f'}
/
&volume
 enabled=f
/
&slices
 enabled=f
/
""")
    if archive_groups is not None:
        config = case / "datin/input.output"
        original = config.read_text()
        original = original.replace("&volume\n enabled=f\n/\n&slices\n enabled=f\n/\n", archive_groups)
        config.write_text(original)
    env = {k: v for k, v in os.environ.items() if not k.startswith("ASTR_")}
    topology = [1, 1, 1]
    topology["xyz".index(args.axis)] = ranks
    env.update(ASTR_OUTPUT_CONFIG="datin/input.output", ASTR_GPU_SYNC_MODE="explicit",
               ASTR_GPU_PRECISION_MODE="fp64", ASTR_GPU_FILTER_WORKSPACE=args.filter_workspace,
               ASTR_GPU_HALO_TRANSPORT="pinned", ASTR_FORCE_MPI_TOPOLOGY=",".join(map(str, topology)),
               ASTR_AIR5_SOURCE_MODE="coupled", ASTR_AIR5_COMPENSATION="on",
               ASTR_AIR5_CONVECTION_LIMITER="symmetric_species", ASTR_AIR5_DIFFUSION_LIMITER="layered",
               ASTR_AIR5_TOP_MODE=args.top_mode, ASTR_AIR5_TOP_TAU=f"{tau:.17e}",
               ASTR_AIR5_TOP_GPU_VALIDATION="on", ASTR_AIR5_FILTER_VALIDATION="on")
    if wall_samples:
        env['ASTR_INSITU_SAMPLE_PREFIX']='outdat/sample'
    if insitu_timing:
        env['ASTR_INSITU_TIMING']='1'
    if pixel_audit:
        env['ASTR_VTK_PIXEL_AUDIT']='1'
    if wall_mean_oracle:
        if insitu_config is None or 'wall_mean_render=t' not in insitu_config or nsys_trace or monitor_resources:
            raise ValueError('AIR5 wall mean oracle requires explicit means, separate from transfer/resource gates')
        env['ASTR_INSITU_TEST_WALL_MEAN_PREFIX']='outdat/render/wall_mean_oracle'
    if device_sample_transport is not None:
        if device_sample_transport not in ('pinned','device-aware') or not wall_samples:
            raise ValueError('AIR5 device oracle requires explicit wall samples and transport')
        env['ASTR_INSITU_TEST_DEVICE_TRANSPORT']=device_sample_transport
    if postprocess_transport is not None and (postprocess_transport not in ('pinned','device-aware') or insitu_config is None):
        raise ValueError('AIR5 device products require explicit configuration and transport')
    active_transport=postprocess_transport if postprocess_transport is not None else device_sample_transport
    if active_transport=='device-aware':
        env.update(OMPI_MCA_pml='ucx', OMPI_MCA_coll='^hcoll,ucc,cuda',
                   OMPI_MCA_coll_hcoll_enable='0', OMPI_MCA_osc='pt2pt',
                   UCX_MEMTYPE_CACHE='n', UCX_CUDA_COPY_ENABLE_FABRIC='no',
                   UCX_CUDA_COPY_DMABUF='no', UCX_CUDA_IPC_ENABLE_MNNVL='no',
                   UCX_TLS='self,sm,cuda_copy,cuda_ipc')
    elif active_transport=='pinned':
        env.update(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,tcp', OMPI_MCA_osc='pt2pt',
                   OMPI_MCA_opal_cuda_support='false', OMPI_MCA_coll_ucc_enable='0')
    if insitu_config is not None:
        (case / 'outdat/render').mkdir()
        (case / 'datin/insitu.nml').write_text(insitu_config)
        env['ASTR_INSITU_CONFIG']='datin/insitu.nml'
        for key in ('DISPLAY','PYTHONPATH','CATALYST_IMPLEMENTATION_PREFER_ENV','VTK_EGL_DEVICE_INDEX'):
            env.pop(key,None)
    if conservation:
        env["ASTR_AIR5_C4_CONSERVATION"] = "1"
    if restore_probe:
        if restore is None:
            raise ValueError("restore probe requires a checkpoint")
        env["ASTR_CHECKPOINT_TEST_RESTORE_PROBE"] = "1"
    if publication_hook is not None:
        library, target_step = publication_hook
        env.update(LD_PRELOAD=str(Path(library).resolve(strict=True)),
                   ASTR_CHECKPOINT_TEST_FAULT="protect_batch",
                   ASTR_CHECKPOINT_TEST_ROOT=str((case / "outdat/new/checkpoints").resolve()),
                   ASTR_CHECKPOINT_TEST_TARGET=f"step{target_step:012d}")
    if restore:
        for name in ("domain", "profile", "initial_field"):
            (case / f"datin/air5_hbl_{name}.dat").unlink()
        if args.case == "sbli":
            (case / "datin/air5_incident_shock.dat").unlink()
    if fault == "tau":
        env["ASTR_AIR5_TOP_TAU"] = f"{2*tau:.17e}"
    elif fault == "compensation":
        env["ASTR_AIR5_COMPENSATION"] = "off"
    elif fault == "source":
        env["ASTR_AIR5_SOURCE_MODE"] = "frozen"
    elif fault == "conservation":
        env["ASTR_AIR5_C4_CONSERVATION"] = "1"
    elif fault == "top_mode":
        env["ASTR_AIR5_TOP_MODE"] = "prescribed" if args.top_mode == "characteristic" else "characteristic"
    solver_command = [str(args.executable), "run", "datin/input.air5_c4"]
    if nsys_trace:
        profiler=shutil.which('nsys')
        if profiler is None or backend!='gpu' or memcheck or monitor_resources or reject or fault:
            raise ValueError('AIR5 Nsight observation requires a validated GPU gate and separate resource run')
        solver_command=[profiler,'profile','--trace=cuda,nvtx,mpi','--mpi-impl=openmpi',
            '--sample=none','--cpuctxsw=none','--cuda-memory-usage=true',
            '--cuda-um-cpu-page-faults=true','--cuda-um-gpu-page-faults=true',
            '--export=sqlite','--output='+str(case/'trace.rank%q{OMPI_COMM_WORLD_RANK}'),*solver_command]
    if memcheck:
        sanitizer = shutil.which("compute-sanitizer")
        if sanitizer is None:
            raise RuntimeError("compute-sanitizer is required for this explicit memory gate")
        # Keep MPI's optional CUDA context probes outside this kernel-memory gate.
        aware=active_transport=='device-aware'
        if not aware:
            env.update(OMPI_MCA_opal_cuda_support="false", OMPI_MCA_pml="ob1",
                       OMPI_MCA_osc="pt2pt", OMPI_MCA_btl="self,vader,tcp", OMPI_MCA_coll_ucc_enable="0")
        solver_command = [sanitizer, "--tool", "memcheck", "--target-processes", "all",
                          "--error-exitcode", "88", "--log-file", str(case / "memcheck.%p.log"),
                          *(['--suppressions',str(ROOT / 'tests/gpu_validation/compute_sanitizer_ucx_cuda_aware.supp.xml')]
                            if aware else []),
                          *solver_command]
    command = [str(args.mpiexec), "--mca", "coll_hcoll_enable", "0", "-np", str(ranks), *solver_command]
    with (case / "run.log").open("wb") as log:
        if monitor_resources:
            if backend!='gpu' or reject or fault or memcheck:
                raise ValueError('AIR5 resource observation requires a successful uninstrumented GPU gate')
            from insitu_resource_monitor import run_monitored
            run_monitored(command,case,env,log,case/'resources.sampled.json',baseline=resource_baseline)
            code=0
        else:
            process = subprocess.Popen(command, cwd=case, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=getattr(args,'runtime_timeout_seconds',180))
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                except ProcessLookupError:
                    process.wait()
                raise
    if fault or reject:
        message = reject or ("AIR5 conservation diagnostic is GPU-only" if fault == "conservation" and backend == "cpu" else
                   "invalid new checkpoint bundle" if fault == "incident_resource" else
                   "numerical/executable/controller contract mismatch" if fault in
                   ("statistics_activation", "sample_interval") else
                   "AIR5 mean statistics device budget" if fault == "statistics_device_budget" else
                   "AIR5 source/limiter/compensation/top/resource contract mismatch")
        if not code or message not in (case / "run.log").read_text():
            raise AssertionError(f"fault not rejected: {fault}: {case}")
        completed = {str(path.relative_to(case / "outdat/new")): path.read_bytes()
                     for path in (case / "outdat/new").rglob("COMPLETE")}
        if completed != prior_complete:
            raise AssertionError("failed restore published a batch")
    elif code:
        raise RuntimeError(f"solver failed: {case / 'run.log'}")
    if memcheck:
        counts = [int(count) for path in case.glob("memcheck.*.log")
                  for count in re.findall(r"ERROR SUMMARY: (\d+) errors", path.read_text())]
        if len(counts) < ranks or any(counts):
            raise AssertionError(f"missing successful memcheck summary: {case}")
    if (case / "outdat/flowfield.h5").exists():
        raise AssertionError("legacy AIR5 checkpoint emitted")
    disk = sum(p.stat().st_size for p in case.rglob("*") if p.is_file())
    if disk > directory_budget_bytes:
        raise RuntimeError(f"directory budget exceeded: {case}: {disk}")
    return case, disk


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("executable", "mpiexec", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--backends", nargs="+", choices=("cpu", "gpu"), default=["cpu", "gpu"])
    p.add_argument("--ranks", nargs="+", type=int, choices=(1, 2), default=[1, 2])
    p.add_argument("--axis", choices=("x", "y", "z"), default="x")
    p.add_argument("--mode", choices=("steps", "time"), default="steps")
    p.add_argument("--case", choices=("hbl", "sbli"), default="hbl")
    p.add_argument("--top-mode", choices=("characteristic", "prescribed"), default="characteristic")
    p.add_argument("--reconstruction", type=int, choices=(0, 3, 5),
                   help="Existing AIR5 selector: default 3 (selective shock) for SBLI, 5 for HBL")
    p.add_argument("--filter-workspace", choices=("scalar", "full"), default="scalar")
    p.add_argument("--rejections", action="store_true")
    p.add_argument("--mean-statistics", action="store_true", help="Preserve raw 44-field legacy statistics")
    p.add_argument("--sample-interval", type=int, choices=(1, 2, 3), default=2)
    p.add_argument("--initial-restart", action="store_true", help="Restore the zero-sample step-0 batch")
    args = p.parse_args()
    if args.reconstruction is None:
        args.reconstruction = 3 if args.case == "sbli" else 5
    args.executable = args.executable.resolve(strict=True)
    args.mpiexec = args.mpiexec.resolve(strict=True)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "checks": []}
    try:
        for backend in args.backends:
            for ranks in args.ranks:
                continuous, a = launch(args, backend, ranks, "continuous", 12)
                restart_step = 0 if args.initial_restart else 5
                first, b = launch(args, backend, ranks, "first", max(1, restart_step))
                seed = first / f"outdat/new/checkpoints/step{restart_step:012d}"
                with h5py.File(seed / "state.h5") as state:
                    nonzero = sum(np.count_nonzero(state[f"q{m:04}"][:]) for m in range(12, 23))
                    if (state["identity"][12] != 7 or state["metadata"][:].tolist() != [1, 1] or
                            (not nonzero and restart_step > 0)):
                        raise AssertionError("seed does not exercise AIR5 compensated state role")
                if args.mean_statistics and args.initial_restart:
                    with h5py.File(seed / "statistics.h5") as state:
                        if state["metadata"][:].tolist() != [2, 0, 0, 0, 0, 0, 0, 0]:
                            raise AssertionError("initial statistics metadata is not empty")
                        if any(np.any(state[f"q{m:04}"][:]) for m in range(1, 45)):
                            raise AssertionError("initial statistics payload is not empty")
                resumed, c = launch(args, backend, ranks, "resumed", 12, restore=seed)
                off, d = launch(args, backend, ranks, "final_only", 12, interval=1000000)
                final = Path("outdat/new/checkpoints/step000000000012")
                datasets = compare_fields(continuous / final / "state.h5", resumed / final / "state.h5")
                compare_fields(continuous / final / "state.h5", off / final / "state.h5")
                mean_info = None
                if args.mean_statistics:
                    relative = final / "statistics.h5"
                    stat_datasets = compare_fields(continuous / relative, resumed / relative)
                    compare_fields(continuous / relative, off / relative)
                    without, no_stat_disk = launch(args, backend, ranks, "no_mean", 12, statistics=False)
                    compare_fields(continuous / final / "state.h5", without / final / "state.h5")
                    if (without / relative).exists():
                        raise AssertionError("statistics disabled but a statistics file was written")
                    with h5py.File(continuous / relative) as state:
                        metadata = state["metadata"][:]
                        count = 11 // args.sample_interval
                        if (state["identity"][12] != 6 or metadata[:4].tolist() !=
                                [2, count, args.sample_interval, count*args.sample_interval]):
                            raise AssertionError("mean statistics lost or duplicated a sample")
                        if not np.any(state["q0034"][:] != 0):
                            raise AssertionError("mean statistics did not exercise viscous stress")
                        for m in range(1, 45):
                            if not np.isfinite(state[f"q{m:04}"][:]).all():
                                raise AssertionError("nonfinite mean payload")
                        stat_bytes = (continuous / relative).stat().st_size
                        stat_transfer = []
                        for row in state["partitions"][:].reshape(-1, 8):
                            stat_transfer.append(int(np.prod(row[3:6]+1))*44*8)
                    mean_info = dict(datasets=stat_datasets, metadata=metadata.tolist(),
                        file_bytes=stat_bytes, output_activation_preserves_flow=True,
                        disabled_directory_bytes=no_stat_disk,
                        controlled_host_buffer_bounds=controlled_checkpoint_buffers(continuous / relative, 0),
                        persistent_device_bytes=[n+4 for n in stat_transfer] if backend == "gpu" else [],
                        declared_gpu_statistics_bytes_per_transfer=stat_transfer if backend == "gpu" else [])
                sanity = {name: state_sanity(case / final / "state.h5") for name, case in
                          (("continuous", continuous), ("resumed", resumed), ("final_only", off))}
                metadata_files = ["control.bin", "air5_config.bin", "archives.bin"]
                if backend == "gpu":
                    metadata_files.append("air5_conservation.bin")
                    empty = (continuous / final / "air5_conservation.bin").read_bytes()
                    if len(empty) != 168 or empty[:8] != b"ASTRA5C1" or any(empty[40:]):
                        raise AssertionError("disabled diagnostic did not save canonical empty history")
                for name in metadata_files:
                    read = archive_schedule_payload if name == "archives.bin" else Path.read_bytes
                    if read(continuous / final / name) != read(resumed / final / name):
                        raise AssertionError(f"control/config continuation mismatch: {name}")
                resource_info = compare_air5_resources(continuous, resumed, args.case == "sbli")
                incident_info = None
                if args.case == "sbli":
                    relative = "outdat/new/resources/air5_incident_shock.dat"
                    expected = (continuous / relative).read_bytes()
                    if expected != (resumed / relative).read_bytes():
                        raise AssertionError("incident resource differs after restart")
                    if (resumed / "datin/air5_incident_shock.dat").exists():
                        raise AssertionError("original incident resource was not removed")
                    contract = (continuous / final / "air5_config.bin").read_bytes()
                    if len(contract) != 1640 or contract[:8] != b"ASTRA502":
                        raise AssertionError("SBLI configuration layout mismatch")
                    top = np.frombuffer(contract[1288:1608], dtype="<f8")
                    if top[4] != 1 or top[38] != int(args.top_mode == "characteristic"):
                        raise AssertionError("incident/top mode missing from completed-step contract")
                    targets = top[16:38].reshape(2, 11)
                    if targets[1, 0] <= targets[0, 0]:
                        raise AssertionError("incident targets do not contain a compressive jump")
                    meta = json.loads((continuous / "incident_metadata.json").read_text())
                    incident_info = dict(resource_bytes=len(expected), top_x=float(top[3]),
                        pressure_ratio=meta["pressure"][1]/meta["pressure"][0],
                        max_scaled_normal_flux_residual=meta["max_scaled_normal_flux_residual"],
                        original_source_removed=True)
                for name in ("flowstate.dat",):
                    full = np.loadtxt(continuous / name, skiprows=1, ndmin=2)
                    tail = np.loadtxt(resumed / name, skiprows=1, ndmin=2)
                    if full[full[:, 0] >= restart_step].tobytes() != tail.tobytes():
                        raise AssertionError(f"scalar statistics continuation mismatch: {name}")
                faults = []
                if args.rejections:
                    rejection_modes = ["tau", "compensation", "source", "conservation", "top_mode"]
                    if args.mean_statistics:
                        rejection_modes.extend(("statistics_activation", "sample_interval"))
                        if backend == "gpu":
                            rejection_modes.append("statistics_device_budget")
                    for fault in rejection_modes:
                        launch(args, backend, ranks, "reject_" + fault, 12, restore=seed, fault=fault)
                        faults.append(fault)
                    if args.case == "sbli":
                        corrupt = args.output / f"{backend}_np{ranks}_corrupt_incident"
                        shutil.copytree(first / "outdat/new", corrupt)
                        with (corrupt / "resources/air5_incident_shock.dat").open("ab") as stream:
                            stream.write(b"changed incident state")
                        launch(args, backend, ranks, "reject_incident", 12,
                               restore=corrupt / f"checkpoints/step{restart_step:012d}", fault="incident_resource")
                        faults.append("incident_resource")
                buffer_bounds = controlled_checkpoint_buffers(continuous / final / "state.h5", 0, backend)
                transfer_bytes = []
                if backend == "gpu":
                    with h5py.File(continuous / final / "state.h5") as state:
                        for row in state["partitions"][:].reshape(-1, 8):
                            transfer_bytes.append(int(np.prod(row[3:6] + 2*row[6] + 1))*23*8 +
                                                  int(np.prod(row[3:6] + 1))*11*8)
                report["checks"].append(dict(backend=backend, ranks=ranks, axis=args.axis,
                    datasets=datasets, seed_nonzero_carry=int(nonzero), rejected=faults,restart_step=restart_step,
                    directory_bytes=[a, b, c, d], complete_step=12, filter_workspace=args.filter_workspace,
                    case=args.case, convection_scheme="643e", reconstruction=args.reconstruction,
                    selective_shock_capturing=args.reconstruction==3,
                    checkpoint_mode=args.mode,sample_interval=args.sample_interval,
                    top_mode=args.top_mode, incident=incident_info, sanity=sanity,
                    resource_readback=resource_info,
                    state_file_bytes=(continuous / final / "state.h5").stat().st_size,
                    configuration_bytes=(continuous / final / "air5_config.bin").stat().st_size,
                    controlled_host_buffer_bounds=buffer_bounds,
                    mean_statistics=mean_info,
                    declared_gpu_field_bytes_per_transfer=transfer_bytes))
        report["status"] = "pass"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
