"""Bounded real-solver acceptance for completed-step checkpoint restart."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
from time import perf_counter

import h5py
import numpy as np
from prepare_tgv_case import next_data_line, set_controller_deltat, set_ninit, set_restart


def archive_schedule_payload(path):
    """Compare schedules exactly, not the intentionally different segment receipts."""
    payload = path.read_bytes()
    if len(payload) < 56 or payload[:8] not in (b"ASTROA02", b"ASTROA03"):
        raise AssertionError("invalid archive history version/tail")
    ids, sizes, crcs = np.frombuffer(payload[-48:], dtype="<i8").reshape(3, 2)
    if not np.all(((ids == -1) & (sizes == 0) & (crcs == 0)) |
                  ((ids >= 0) & (ids <= 99999999) & (sizes > 0))):
        raise AssertionError("invalid saved segment receipts")
    return payload[:-48]


def prepare_initial_resource(case, input_name, dimension):
    """Exercise the original reader with smooth positive, nonconstant fields."""
    primary = case / "datin" / input_name
    set_ninit(primary, dimension)
    coordinates = np.meshgrid(*([np.linspace(0, 2*np.pi, 17)] * dimension), indexing="ij")
    x = coordinates[-1]
    variation = sum(np.cos(c) for c in coordinates) / dimension
    fields = {"ro": 1 + 0.01 * variation, "u1": 0.05 * np.sin(x),
              "t": 1 + 0.02 * variation}
    if dimension >= 2:
        fields["u2"] = 0.03 * np.sin(coordinates[-2])
    if dimension == 3:
        fields["u3"] = 0.02 * np.sin(coordinates[0])
    path = case / "datin" / f"flowini{dimension}d.h5"
    with h5py.File(path, "x") as state:
        for name, values in fields.items():
            state[name] = np.asarray(values, dtype=np.float64)
    return path


def run_case(args, root, backend, ranks, name, steps, restore=None, enabled=True,
             reject=None, change_input=False, checkpoint_interval=None, override=False,
             source_fault=None, archive_groups=None, buffer_bytes=67108864,
             device_budget_bytes=67108864, checkpoint_keep=2, publication_fault=None, reuse_root=None,
             lfilter=True, insitu_config=None, topology=None, controller_replay=None, initial_resource=None,
             rhs_snapshot_step=None, memcheck=False, device_reserve_bytes=0,
             monitor_resources=False, resource_baseline=None,
             output_config_override=None, omit_output_config=False,
             legacy_restart=False, legacy_output=False, no_field_io=False, grid=None,adaptive_config=None,tgv_reynolds=None,
             resident_audit=False, pixel_audit=False,
             test_fault=None, failure_after_start=False, tgv_mapping=None, insitu_timing=False,
             device_sample_transport=None, postprocess_transport=None, nsys_trace=False, plane_oracle=False,
             checkpoint_enabled=None, curve_surface_oracle=False, wall_mean_oracle=False, curve_trace_oracle=False,
             ncu_kernel=None, curve_trace_step_scale=None):
    if ncu_kernel is not None and (not ncu_kernel or backend != 'gpu' or ranks != 1 or
            nsys_trace or memcheck or monitor_resources or reject):
        raise ValueError('Kernel profiling requires a separate successful NP=1 GPU run')
    if failure_after_start and not reject:
        raise ValueError('post-start failure checks require an expected rejection')
    if checkpoint_enabled is not None and type(checkpoint_enabled) is not bool:
        raise ValueError('checkpoint_enabled must be an explicit boolean')
    if checkpoint_enabled is False and (restore is not None or args.initial_restart):
        raise ValueError('checkpoint-free observation cannot request restart')
    timeout=getattr(args,'runtime_timeout_seconds',180)
    if timeout<=0:
        raise ValueError('test runtime timeout must be positive')
    directory_budget = getattr(args, 'directory_budget_bytes', 64 * 1024**2)
    if directory_budget <= 0:
        raise ValueError('test directory budget must be positive')
    output_host_budget = getattr(args, 'output_host_budget_bytes', 64 * 1024**2)
    if type(output_host_budget) is not int or output_host_budget < buffer_bytes:
        raise ValueError('output host budget must contain the requested packing buffer')
    if grid is not None and (args.case != 'tgv' or args.initial_dimension != 0):
        raise ValueError('explicit validation grid requires internally initialized TGV')
    if tgv_mapping is not None and (args.case != 'tgv' or args.initial_dimension != 0 or
                                   grid not in ('32,32,32', '64,64,64', '128,128,128', '256,256,256') or
                                   tgv_mapping not in ('periodic','y-wavy')):
        raise ValueError('CURVE fixture requires internally initialized 32^3, 64^3, 128^3 or 256^3 TGV')
    case = args.output / f"{backend}_np{ranks}_{name}"
    channel = args.case == "channel"
    dynamic = args.case == "dynamic"
    curve = args.case in ("curve", "dynamic")
    input_name = "input.flatplate" if curve else ("input.chl" if channel else "input.tgv")
    dt = 6e-6 if dynamic else (1e-5 if curve else 1e-3)
    scale_dt = getattr(args, 'scale_timestep', None)
    cfl_limit = getattr(args, 'maximum_cfl', None)
    if scale_dt is not None:
        if tgv_mapping is None or not np.isfinite(scale_dt) or scale_dt <= 0 or cfl_limit is None:
            raise ValueError('Scale timestep override requires CURVE TGV and an explicit CFL gate')
        dt = scale_dt
    if curve:
        subprocess.run([
            sys.executable, str(root / "tests/gpu_validation/prepare_s1_flatplate_case.py"),
            "--dst-case", str(case), "--use-gpu", "t" if backend == "gpu" else "f",
            "--im", "16", "--jm", "16", "--km", "16", "--warp-x", "0.08", "--warp-y", "0.04",
            "--maxstep", str(max(1, steps - 1)), "--feqchkpt", "1", "--deltat", str(dt),
            "--lfilter", "t" if lfilter else "f", "--diffterm", "t", "--conschm", "543e",
            "--turbinf", "intp" if dynamic else "prof"], check=True)
        if dynamic:
            subprocess.run([
                sys.executable, str(root / "tests/gpu_validation/generate_dynamic_inflow_slices.py"),
                "--output", str(case / "inflow"), "--jm", "16", "--km", "16",
                "--count", str(args.inflow_count), "--delta-time", "1e-5",
                "--temporal-mode", "nonpolynomial"], check=True)
            if source_fault == "hole":
                (case / "inflow/islice00001.h5").unlink()
            elif source_fault == "external":
                with h5py.File(case / "inflow/islice00000.h5", "r+") as source:
                    del source["ro"]
                    source["ro"] = h5py.ExternalLink("islice00001.h5", "ro")
            elif source_fault == "initial_field":
                primary = case / "datin/input.flatplate"
                lines = primary.read_text().splitlines()
                index = next_data_line(lines, next(i for i, line in enumerate(lines) if line.strip() == "# ninit"))
                lines[index] = "3"
                primary.write_text("\n".join(lines) + "\n")
            elif source_fault is not None and source_fault != "initial_external":
                raise ValueError("unknown dynamic source fault")
        if steps == 1:
            controller = case / "datin/controller"
            lines = controller.read_text().splitlines()
            index = next_data_line(lines, next(i for i, line in enumerate(lines)
                                              if "maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg" in line))
            fields = lines[index].split(",")
            if len(fields) != 6:
                raise ValueError("unexpected controller counter inventory")
            fields[0] = "0"  # The solver's inclusive loop performs one complete step.
            lines[index] = ",".join(fields)
            controller.write_text("\n".join(lines) + "\n")
    else:
        subprocess.run([
        sys.executable, str(root / "tests/gpu_validation/prepare_tgv_case.py"),
        "--src-case", str(root / ("examples/Channel" if channel else "examples/Taylor_Green_Vortex")),
        "--dst-case", str(case), "--input-name", input_name,
        "--homogeneous", "t,f,t" if channel else "t,t,t",
        "--use-gpu", "t" if backend == "gpu" else "f", "--grid", grid or "16,16,16",
        "--maxstep", str(steps - 1), "--feqchkpt", "1", "--deltat", str(dt),
        *(["--feqlist", "1"] if scale_dt is not None else []),
        "--lfilter", "t" if lfilter else "f", "--diffterm", "t", "--scheme", "643e"], check=True)
    if tgv_reynolds is not None:
        if args.case!='tgv' or args.initial_dimension!=0 or not np.isfinite(tgv_reynolds) or tgv_reynolds<=0:
            raise ValueError('reference Reynolds override requires internally initialized TGV and a finite positive value')
        primary=case/'datin'/input_name
        lines=primary.read_text().splitlines()
        index=next_data_line(lines,next(i for i,line in enumerate(lines) if 'ref_t,reynolds,mach' in line))
        fields=lines[index].split(',')
        if len(fields)!=3:
            raise ValueError('unexpected reference value inventory')
        fields[1]=f'{tgv_reynolds:.17e}'
        lines[index]=','.join(fields)
        primary.write_text('\n'.join(lines)+'\n')
    if tgv_mapping is not None:
        from prepare_tgv_case import set_gridfile, set_runtime_flags, set_homogeneous, set_bctype
        subprocess.run([
            sys.executable, str(root / 'tests/gpu_validation/generate_curvilinear_tgv_grid.py'),
            '--grid', grid, '--amplitude', '0.15', '--mapping', tgv_mapping,
            '--output', str(case / 'datin/grid.tgv.h5'),
            '--report', str(case / 'grid_report.txt')], check=True)
        primary = case / 'datin' / input_name
        set_gridfile(primary, 'datin/grid.tgv.h5')
        set_runtime_flags(primary, 't' if backend == 'gpu' else 'f', None, None, 't')
        if tgv_mapping=='y-wavy':
            set_homogeneous(primary,'t,f,t')
            set_bctype(primary,'1;1;41,273.15d0;41,273.15d0;1;1')
    if args.initial_dimension and source_fault != "initial_field":
        if initial_resource is None:
            initial_path = prepare_initial_resource(case, input_name, args.initial_dimension)
        else:
            if not channel or args.initial_dimension != 3 or source_fault is not None:
                raise ValueError("frozen external initial resource is limited to the channel gate")
            set_ninit(case / "datin" / input_name, 3)
            initial_path = case / "datin/flowini3d.h5"
            shutil.copyfile(initial_resource, initial_path)
        if source_fault == "initial_external":
            with h5py.File(initial_path, "r+") as state:
                del state["ro"]
                state["ro"] = h5py.ExternalLink("absent.h5", "ro")
        if restore:
            initial_path.unlink()
    if args.legacy_statistics:
        controller = case / "datin/controller"
        lines = controller.read_text().splitlines()
        flags = next_data_line(lines, next(i for i, line in enumerate(lines) if "lwsequ,lwslic,lavg,lcracon" in line))
        counters = next_data_line(lines, next(i for i, line in enumerate(lines) if "maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg" in line))
        fields = lines[flags].split(",")
        intervals = lines[counters].split(",")
        if len(fields) != 4 or len(intervals) != 6:
            raise ValueError("unexpected controller field inventory")
        fields[2], intervals[5] = "t", "1"
        lines[flags], lines[counters] = ",".join(fields), ",".join(intervals)
        controller.write_text("\n".join(lines) + "\n")
    if legacy_output:
        controller = case / "datin/controller"
        lines = controller.read_text().splitlines()
        flags = next_data_line(lines, next(i for i, line in enumerate(lines) if "lwsequ,lwslic,lavg,lcracon" in line))
        fields = lines[flags].split(",")
        fields[:2] = ["t", "t"]
        lines[flags] = ",".join(fields)
        controller.write_text("\n".join(lines) + "\n")
    if legacy_restart:
        set_restart(case / "datin" / input_name, "t")
    (case / "outdat/new").mkdir(parents=True)
    if reuse_root is not None:
        if restore is None:
            raise ValueError("reuse test needs an explicit checkpoint")
        shutil.copytree(reuse_root, case / "outdat/new", dirs_exist_ok=True)
        restore = case / "outdat/new/checkpoints" / restore.name
    interval = checkpoint_interval
    if interval is None:
        interval = (args.restart_step if enabled else 1000000000) if args.mode == "steps" else (4*dt if enabled else 1.0)
    config = case / "datin/input.output"
    write_checkpoint = (enabled or args.statistics or args.legacy_statistics or args.case != 'tgv' or args.initial_dimension)
    if checkpoint_enabled is not None:
        write_checkpoint = checkpoint_enabled
    config.write_text(f"""&output
 directory='outdat/new', restore_directory='{restore or ''}',
 restart_output='{'override' if override else 'saved'}',
 host_budget_bytes={output_host_budget},device_budget_bytes={device_budget_bytes},buffer_bytes={buffer_bytes}
 device_reserve_bytes={device_reserve_bytes}
/
&checkpoint
        enabled={'.true.' if write_checkpoint else '.false.'},mode='{args.mode}',
 interval_steps={interval if args.mode == 'steps' else 0},
 interval_time={interval if args.mode == 'time' else 0},keep={checkpoint_keep},
 initial_frame={'.true.' if args.initial_restart else '.false.'}
/
&volume
 enabled=.false.
/
&slices
 enabled=.false.
/
""")
    if archive_groups is not None:
        original = config.read_text()
        original = original.replace("&volume\n enabled=.false.\n/\n&slices\n enabled=.false.\n/\n", archive_groups)
        config.write_text(original)
    if adaptive_config is not None:
        config.write_text(config.read_text() + adaptive_config)
    if change_input:
        with (case / "datin" / input_name).open("a") as stream:
            stream.write("\n! changed primary input identity\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("ASTR_")}
    if topology is None:
        topology = [1, 1, 1]
        topology["xyz".index(args.axis)] = ranks
    if len(topology) != 3 or any(n < 1 for n in topology) or np.prod(topology) != ranks:
        raise ValueError("test topology must contain three positive extents matching ranks")
    if omit_output_config:
        config.unlink()
    elif output_config_override:
        destination = case / output_config_override
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination != config:
            config.rename(destination)
    if output_config_override is not None:
        env["ASTR_OUTPUT_CONFIG"] = output_config_override
    env.update(ASTR_FORCE_MPI_TOPOLOGY=",".join(map(str, topology)),
               ASTR_GPU_SYNC_MODE="explicit", ASTR_GPU_HALO_TRANSPORT="pinned",
               ASTR_GPU_PRECISION_MODE="fp64", ASTR_INSITU_SAMPLE_PREFIX="outdat/sample")
    env["ASTR_GPU_FILTER_WORKSPACE"] = args.filter_workspace
    if insitu_timing:
        env['ASTR_INSITU_TIMING']='1'
    if resident_audit:
        env['ASTR_INSITU_RESIDENT_AUDIT']='1'
    if pixel_audit:
        env['ASTR_VTK_PIXEL_AUDIT']='1'
    env.pop('ASTR_INSITU_TEST_PLANE_PREFIX',None)
    env.pop('ASTR_INSITU_TEST_CURVE_Q_PREFIX',None)
    env.pop('ASTR_INSITU_TEST_CURVE_TRACE_PREFIX',None)
    env.pop('ASTR_INSITU_TEST_CURVE_STEP_SCALE',None)
    env.pop('ASTR_INSITU_TEST_WALL_MEAN_PREFIX',None)
    if wall_mean_oracle:
        if insitu_config is None or 'wall_mean_render=t' not in insitu_config or nsys_trace or monitor_resources:
            raise ValueError('wall mean oracle requires explicit means, separate from transfer/resource gates')
        env['ASTR_INSITU_TEST_WALL_MEAN_PREFIX']='outdat/render/wall_mean_oracle'
    if curve_surface_oracle:
        if tgv_mapping not in ('periodic', 'y-wavy') or insitu_config is None or not any(
                f"products='{profile}'" in insitu_config for profile in ('q_surface', 'curve_demo')):
            raise ValueError('CURVE Q oracle requires an admitted physical contour configuration')
        env['ASTR_INSITU_TEST_CURVE_Q_PREFIX']='outdat/render/curve_q_oracle'
    if curve_trace_oracle:
        if tgv_mapping is None or insitu_config is None or not any(
                f"products='{p}'" in insitu_config for p in ('streamlines','curve_demo')) or \
                nsys_trace or monitor_resources:
            raise ValueError('CURVE trace oracle requires explicit physical streamlines, separate from transfer/resource gates')
        env['ASTR_INSITU_TEST_CURVE_TRACE_PREFIX']='outdat/render/curve_trace_oracle'
    if curve_trace_step_scale is not None:
        if curve_trace_step_scale not in (1.,.5) or not curve_trace_oracle or checkpoint_enabled is not False or \
                restore is not None or nsys_trace or monitor_resources or ncu_kernel:
            raise ValueError('Step sensitivity requires separate checkpoint-free trace diagnostics with scale 1 or 0.5')
        env['ASTR_INSITU_TEST_CURVE_STEP_SCALE']='1' if curve_trace_step_scale==1. else '0.5'
    if plane_oracle:
        if insitu_config is None or "slice_definition='plane'" not in insitu_config:
            raise ValueError('plane oracle requires the explicit physical-plane configuration')
        env['ASTR_INSITU_TEST_PLANE_PREFIX']='outdat/render/plane_oracle'
    if device_sample_transport is not None:
        if device_sample_transport not in ('device-aware','pinned'):
            raise ValueError('device sample diagnostic requires an explicit supported transport')
        env['ASTR_INSITU_TEST_DEVICE_TRANSPORT']=device_sample_transport
    if postprocess_transport is not None:
        if postprocess_transport not in ('device-aware','pinned') or insitu_config is None:
            raise ValueError('native device products require a declared transport and configuration')
    active_transport=postprocess_transport if postprocess_transport is not None else device_sample_transport
    if active_transport is not None:
        if active_transport == 'device-aware':
            env.update(OMPI_MCA_pml='ucx', OMPI_MCA_coll='^hcoll,ucc,cuda',
                       OMPI_MCA_coll_hcoll_enable='0', OMPI_MCA_osc='pt2pt',
                       UCX_MEMTYPE_CACHE='n', UCX_CUDA_COPY_ENABLE_FABRIC='no',
                       UCX_CUDA_COPY_DMABUF='no', UCX_CUDA_IPC_ENABLE_MNNVL='no',
                       UCX_TLS='self,sm,cuda_copy,cuda_ipc')
        else:
            env.update(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,tcp', OMPI_MCA_osc='pt2pt',
                       OMPI_MCA_opal_cuda_support='false', OMPI_MCA_coll_ucc_enable='0')
    if no_field_io:
        if args.case=='tgv' and args.initial_dimension==0 and tgv_mapping is None:
            # The copied example grid is unused by internally generated no-I/O TGV.
            (case/'datin/grid.h5').unlink(missing_ok=True)
        env["ASTR_GPU_BENCHMARK_NO_FIELD_IO"] = "1"
        env["ASTR_GPU_RK_TIMING" if backend == "gpu" else "ASTR_CPU_RK_TIMING"] = "1"
    if rhs_snapshot_step is not None:
        if not 0 <= rhs_snapshot_step < steps:
            raise ValueError("RHS diagnostic step must be within this bounded run")
        env.update(ASTR_VALIDATION_RHS_PREFIX="outdat/rhs",
                   ASTR_VALIDATION_RHS_STEP=str(rhs_snapshot_step))
    if args.case != "tgv" and not (channel and getattr(args, "wall_samples", False)):
        env.pop("ASTR_INSITU_SAMPLE_PREFIX")
    if args.case == 'tgv' and getattr(args, 'wall_samples', None) is False:
        env.pop('ASTR_INSITU_SAMPLE_PREFIX', None)
    if channel:
        env["ASTR_CHANNEL_FORCE_MODE"] = args.force
        if args.force == "fixed":
            env["ASTR_CHANNEL_FORCE_FIXED"] = "1.d-4"
    if args.statistics:
        (case / "datin/input.insitu").write_text("""&insitu_run
 enabled=t, statistics=t, render=f, statistics_window=0.0005,0.0115,
 output_directory='outdat', host_budget_bytes=67108864,
 device_budget_bytes=67108864, device_reserve_bytes=1073741824
/
""")
        env["ASTR_INSITU_CONFIG"] = "datin/input.insitu"
    if insitu_config is not None:
        (case / "outdat/render").mkdir()
        (case / "datin/input.insitu").write_text(insitu_config)
        env["ASTR_INSITU_CONFIG"] = "datin/input.insitu"
        for key in ("DISPLAY", "PYTHONPATH", "CATALYST_IMPLEMENTATION_PREFER_ENV", "VTK_EGL_DEVICE_INDEX"):
            env.pop(key, None)
    if curve and restore:
        # The frozen shared resources must suffice without the original inputs.
        (case / "datin/grid.flatplate.h5").unlink()
        (case / "datin/inlet.prof").unlink()
        if dynamic:
            shutil.rmtree(case / "inflow")
    if tgv_mapping is not None and restore:
        (case / 'datin/grid.tgv.h5').unlink()
    if publication_fault is not None:
        library, phase, target_step, retired_step = publication_fault
        if phase not in ("batch_create", "batch_rename", "latest_rename", "retire_marker", "retire_payload", "protect_batch"):
            raise ValueError("unknown publication fault")
        env.update(LD_PRELOAD=str(Path(library).resolve(strict=True)),
                   ASTR_CHECKPOINT_TEST_ROOT=str((case / "outdat/new/checkpoints").resolve()),
                   ASTR_CHECKPOINT_TEST_TARGET=f"step{target_step:012d}",
                   ASTR_CHECKPOINT_TEST_RETIRE=f"step{retired_step:012d}",
                   ASTR_CHECKPOINT_TEST_FAULT=phase)
    if controller_replay is not None:
        if publication_fault is not None or args.case != "tgv":
            raise ValueError("controller replay is isolated to the TGV clock test")
        library, start_step, dt_next = controller_replay
        if not 0 <= start_step < steps or len(dt_next) != steps:
            raise ValueError("controller replay must cover every completed step")
        replay = case / "controller_replay"
        replay.mkdir()
        controller = case / "datin/controller"
        shutil.copyfile(controller, replay / "controller000000000000")
        for completed in range(start_step + 1, steps + 1):
            value = dt_next[completed - 1]
            if not np.isfinite(value) or value <= 0:
                raise ValueError("controller replay needs finite positive steps")
            target = replay / f"controller{completed:012d}"
            shutil.copyfile(controller, target)
            set_controller_deltat(target, f"{value:.17e}")
        env.update(LD_PRELOAD=str(Path(library).resolve(strict=True)),
                   ASTR_CONTROLLER_TEST_FILE=str(controller.resolve(strict=True)),
                   ASTR_CONTROLLER_TEST_REPLAY=str(replay.resolve()),
                   ASTR_CONTROLLER_TEST_START_STEP=str(start_step))
    solver_command = [str(args.executable), "run", "datin/" + input_name]
    if ncu_kernel is not None:
        profiler = shutil.which('ncu')
        if profiler is None:
            raise RuntimeError('Nsight Compute is required for explicit kernel profiling')
        solver_command = [profiler, '--target-processes', 'all', '--kernel-name-base', 'demangled',
            '--kernel-name', 'regex:'+ncu_kernel,
            '--launch-count', '1', '--section', 'LaunchStats', '--section', 'Occupancy',
            '--section', 'SpeedOfLight', '--section', 'SchedulerStats',
            '--section', 'MemoryWorkloadAnalysis', '--export', str(case/'kernel'), *solver_command]
    if nsys_trace:
        profiler=shutil.which('nsys')
        if profiler is None or memcheck or monitor_resources or reject:
            raise ValueError('Nsight trace requires a successful uninstrumented GPU gate and nsys')
        if backend!='gpu':
            raise ValueError('Device trace requires GPU')
        trace_domains = getattr(args, 'nsys_trace_domains', 'cuda,nvtx,mpi')
        solver_command=[profiler,'profile','--trace='+trace_domains,'--mpi-impl=openmpi',
            '--sample=none','--cpuctxsw=none','--cuda-memory-usage=true',
            '--cuda-um-cpu-page-faults=true','--cuda-um-gpu-page-faults=true',
            '--export=sqlite','--output='+str(case/'trace.rank%q{OMPI_COMM_WORLD_RANK}'),*solver_command]
    if test_fault is not None:
        if publication_fault is not None or controller_replay is not None:
            raise ValueError('IS4 fault preload cannot be combined with other preloads')
        library,phase,target_step=test_fault
        env.update(LD_PRELOAD=str(Path(library).resolve(strict=True)),
                   ASTR_IS4_TEST_PHASE=phase,
                   ASTR_IS4_TEST_TARGET=str((case/'outdat/new/checkpoints'/f'step{target_step:012d}.tmp').resolve()))
    if memcheck:
        sanitizer = shutil.which("compute-sanitizer")
        if sanitizer is None:
            raise RuntimeError("compute-sanitizer is required for this explicit memory gate")
        # Host transport can isolate optional MPI CUDA pointer probes. A device
        # buffer diagnostic must retain its explicitly declared aware-MPI setup.
        aware = device_sample_transport == 'device-aware' or postprocess_transport == 'device-aware'
        if not aware:
            env.update(OMPI_MCA_opal_cuda_support="false", OMPI_MCA_pml="ob1",
                       OMPI_MCA_osc="pt2pt", OMPI_MCA_btl="self,vader,tcp",
                       OMPI_MCA_coll_ucc_enable="0")
        solver_command = [sanitizer, "--tool", "memcheck", "--target-processes", "all",
                          "--error-exitcode", "99", "--log-file", str(case / "memcheck.%p.log"),
                          *(["--suppressions",str(root / 'tests/gpu_validation/compute_sanitizer_ucx_cuda_aware.supp.xml')]
                            if aware else []),*solver_command]
    command = [str(args.mpiexec), "--mca", "coll_hcoll_enable", "0", "-np", str(ranks),
               *solver_command]
    launch_started=perf_counter()
    cfl_gate = None
    if cfl_limit is not None:
        from insitu_cfl_gate import CflGate
        if reject or scale_dt is None:
            raise ValueError('CFL gate requires an explicit successful scale test')
        first = int(restore.name.removeprefix('step')) if restore is not None else 0
        cfl_gate = CflGate(case/'run.log', cfl_limit, dt, first, steps)
    with (case / "run.log").open("wb") as log:
        if monitor_resources:
            if backend != "gpu" or reject or memcheck:
                raise ValueError("resource sampling requires a successful GPU gate without memcheck")
            from insitu_resource_monitor import run_monitored
            run_monitored(command, case, env, log, case / "resources.sampled.json", baseline=resource_baseline,
                device_extra_budget_bytes=getattr(args, 'resource_device_extra_budget_bytes', 2*1024**3),
                host_extra_budget_bytes=getattr(args, 'resource_host_extra_budget_bytes', 4*1024**3),
                device_reserve_bytes=getattr(args, 'resource_device_reserve_bytes', 1024**3),
                timeout_seconds=timeout, check_progress=cfl_gate.check if cfl_gate else None)
            returncode = 0
        else:
            process = subprocess.Popen(command, cwd=case, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                if cfl_gate is None:
                    returncode = process.wait(timeout=timeout)
                else:
                    deadline = perf_counter()+timeout
                    while True:
                        cfl_gate.check()
                        if perf_counter() >= deadline:
                            raise TimeoutError('CFL-monitored solver exceeded runtime budget')
                        try:
                            returncode = process.wait(timeout=.02)
                            break
                        except subprocess.TimeoutExpired:
                            continue
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
    if cfl_gate is not None and returncode == 0:
        cfl_gate.check(final=True)
        (case/'cfl_gate.json').write_text(json.dumps(dict(maximum=cfl_gate.maximum, limit=cfl_limit,
            dt=dt, samples=steps-first, scope='per-step log gate; termination after log becomes visible'), indent=2))
    if insitu_timing:
        (case/'timing.launch.json').write_text(json.dumps(dict(
            seconds=perf_counter()-launch_started,
            scope='mpiexec launch through child exit, includes external monitor when selected'),indent=2))
    if ncu_kernel is not None and not (case/'kernel.ncu-rep').is_file():
        raise RuntimeError('Kernel profiling produced no report; inspect run.log before accepting the capture')
    if reject:
        if returncode == 0 or reject not in (case / "run.log").read_text():
            raise AssertionError(f"expected rejection not observed: {reject}: {case}")
        if not failure_after_start and publication_fault is None and reuse_root is None and list((case / "outdat/new").rglob("COMPLETE")):
            raise AssertionError("failed restore published a checkpoint")
    elif returncode:
        raise RuntimeError(f"solver failed: {case / 'run.log'}")
    if memcheck:
        summaries = list(case.glob("memcheck.*.log"))
        counts = [int(count) for path in summaries
                  for count in re.findall(r"ERROR SUMMARY: (\d+) errors", path.read_text())]
        if len(counts) < ranks or any(counts):
            raise AssertionError(f"missing successful memcheck summary: {case}")
    if (case / "outdat/flowfield.h5").exists():
        raise AssertionError("new output emitted a legacy flowfield file")
    disk_bytes = sum(p.stat().st_size for p in case.rglob("*") if p.is_file())
    if disk_bytes > directory_budget:
        raise RuntimeError(f"test directory budget exceeded: {case}: {disk_bytes}")
    return case, disk_bytes


def compare_fields(left, right):
    with h5py.File(left, "r") as a, h5py.File(right, "r") as b:
        names_a, names_b = [], []
        a.visit(names_a.append)
        b.visit(names_b.append)
        if names_a != names_b:
            raise AssertionError("dataset inventory mismatch")
        differences = {}
        for name in names_a:
            if isinstance(a[name], h5py.Group) != isinstance(b[name], h5py.Group):
                raise AssertionError("group/dataset layout mismatch")
            if isinstance(a[name], h5py.Group):
                continue
            aa, bb = a[name][:], b[name][:]
            if aa.shape != bb.shape or aa.dtype != bb.dtype or aa.tobytes() != bb.tobytes():
                differences[name] = float(np.max(np.abs(aa.astype(float) - bb.astype(float))))
        if differences:
            raise AssertionError(f"exact state mismatch: {differences}")
        return names_a


def check_geometry_padding(path, homogeneous):
    """Check padding by location, never by the magnitude of stored values."""
    checked = 0
    with h5py.File(path, "r") as state:
        global_shape = state["identity"][3:6]
        if state["identity"][6] != 13:
            raise AssertionError("expected 13-component geometry resource")
        offset = 0
        for rank, partition in enumerate(state["partitions"][:].reshape(-1, 8)):
            origin, cells = partition[:3], partition[3:6]
            halo, extra_count = map(int, partition[6:8])
            shape = cells + 2 * halo + 1
            indices = np.indices(tuple(shape))
            physical = (indices >= halo) & (indices <= (halo + cells)[:, None, None, None])
            face_or_interior = np.count_nonzero(~physical, axis=0) <= 1
            owned = cells + (origin + cells == global_shape - 1)
            owner = np.all((indices >= halo) &
                           (indices < (halo + owned)[:, None, None, None]), axis=0)
            extra_nodes = ~owner.ravel(order="F")
            width = int(np.count_nonzero(extra_nodes))
            if extra_count != 13 * width:
                raise AssertionError("geometry extras layout mismatch")
            payload = state["rank_extras"][offset:offset + extra_count].reshape(13, width)
            metric_defined = face_or_interior.copy()
            for axis, periodic in enumerate(homogeneous):
                if periodic:
                    continue
                if origin[axis] == 0:
                    metric_defined &= indices[axis] >= halo
                if origin[axis] + cells[axis] == global_shape[axis] - 1:
                    metric_defined &= indices[axis] <= halo + cells[axis]
            for component in range(13):
                defined = face_or_interior if component < 3 else metric_defined
                padding = ~defined.ravel(order="F")[extra_nodes]
                # Bitwise +0 excludes tiny uninitialized data and signed-zero drift.
                if np.any(payload[component, padding].view(np.uint64) != 0):
                    raise AssertionError(f"noncanonical geometry padding: rank={rank}, component={component + 1}")
                checked += int(np.count_nonzero(padding))
            offset += extra_count
        if offset != state["rank_extras"].size:
            raise AssertionError("geometry extras length mismatch")
    return checked


def controlled_checkpoint_buffers(path, persistent_bytes, backend=None):
    """Explicit allocations only; not RSS, HDF5/MPI peaks or compiler temporaries."""
    with h5py.File(path) as state:
        components, role = int(state["identity"][6]), int(state["identity"][12])
        if role not in (1, 2, 5, 6, 7, 8):
            raise ValueError("buffer estimator does not cover this state provider")
        if role == 7 and backend not in ("cpu", "gpu"):
            raise ValueError("AIR5 buffer estimate requires the actual backend")
        partitions = state["partitions"][:].reshape(-1, 8)
        bounds = []
        for row in partitions:
            extents = row[3:6] + 2 * row[6] + 1
            packing = int(np.prod(extents)) * components * 8
            extras = max(1, int(row[7])) * 8
            tables = len(partitions) * 128
            staging = 0
            if role == 5:
                staging = int((row[3] + 1) * (row[4] + 1)) * 13 * 8
                if row[1] == 0:
                    staging += int(row[3] + 1) * 4 * 8
            if role == 7 and backend == "gpu" and state["metadata"][1] == 1:
                staging = int(np.prod(row[3:6] + 1)) * 11 * 8
            bounds.append(packing + extras + tables + staging + persistent_bytes)
        if max(bounds) > 64 * 1024**2:
            raise AssertionError("controlled checkpoint host budget exceeded")
        return bounds


def compare_statistics(continuous, resumed, disabled, backend, restart_step, case):
    names = ["flowstate.dat"]
    if backend == "gpu" and case == "tgv":
        names += ["gpu_kenergy.dat", "gpu_enstophy.dat", "gpu_dissipation.dat"]
    counts = {}
    for name in names:
        full = np.loadtxt(continuous / name, skiprows=1, ndmin=2)
        tail = np.loadtxt(resumed / name, skiprows=1, ndmin=2)
        off = np.loadtxt(disabled / name, skiprows=1, ndmin=2)
        expected = full[full[:, 0] >= restart_step]
        if full.tobytes() != off.tobytes() or expected.tobytes() != tail.tobytes():
            raise AssertionError(f"statistics mismatch: {name}")
        counts[name] = len(tail)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("executable", "mpiexec", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--backends", nargs="+", choices=("cpu", "gpu"), default=["cpu", "gpu"])
    parser.add_argument("--ranks", nargs="+", type=int, choices=(1, 2), default=[1, 2])
    parser.add_argument("--mode", choices=("steps", "time"), default="steps")
    parser.add_argument("--case", choices=("tgv", "channel", "curve", "dynamic"), default="tgv")
    parser.add_argument("--restart-step", type=int, choices=range(1, 12), default=5)
    parser.add_argument("--inflow-count", type=int, default=12)
    parser.add_argument("--initial-dimension", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--force", choices=("feedback", "fixed", "frozen"), default="feedback")
    parser.add_argument("--axis", choices=("x", "y", "z"), default="x")
    parser.add_argument("--legacy-statistics", action="store_true",
                        help="Include existing CPU accumulated moments or GPU compact curve statistics")
    parser.add_argument("--filter-workspace", choices=("scalar", "full"), default="scalar")
    parser.add_argument("--statistics", action="store_true", help="Include formal statistics; compare periodic versus final-only checkpoints")
    parser.add_argument("--initial-restart", action="store_true",
                        help="Restore the initial step-0 checkpoint instead of step 5")
    parser.add_argument("--schedule-checks", action="store_true",
                        help="Check changed-schedule rejection and explicit restart-relative override")
    parser.add_argument("--no-samples", action="store_true",
                        help="For non-testing builds: skip test-only per-step field dumps; still compare final restart states")
    args = parser.parse_args()
    if args.legacy_statistics and "gpu" in args.backends and args.case not in ("curve", "dynamic"):
        parser.error("GPU legacy statistics currently require a flatplate case")
    if args.inflow_count < 12 or args.inflow_count > 256:
        parser.error("bounded 16^3 test inflow count must be 12:256; not a solver production limit")
    if args.legacy_statistics and args.statistics:
        parser.error("formal and legacy statistics require separate runs")
    if args.case != "tgv" and args.statistics:
        parser.error("formal in-situ statistics remain restricted to TGV")
    if args.case == "curve" and (args.initial_restart or args.schedule_checks):
        parser.error("curve initial and override tests require a separate bounded matrix")
    if args.schedule_checks and args.restart_step != 5:
        parser.error("schedule override checks currently use restart step 5")
    if args.statistics or args.legacy_statistics or args.case != "tgv":
        args.no_samples = True
    args.executable = args.executable.resolve(strict=True)
    # HPC-X wrappers select the underlying executable from their invoked name.
    args.mpiexec = args.mpiexec.absolute()
    if not args.mpiexec.is_file():
        parser.error("MPI launcher does not exist")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    report = {"status": "running", "checks": []}
    try:
        for backend in args.backends:
            for ranks in args.ranks:
                continuous, a_size = run_case(args, root, backend, ranks, "continuous", 12)
                restart_step = 0 if args.initial_restart else args.restart_step
                first, b_size = run_case(args, root, backend, ranks, "first", max(1, restart_step))
                source = first / f"outdat/new/checkpoints/step{restart_step:012d}"
                resumed, c_size = run_case(args, root, backend, ranks, "resumed", 12, restore=source)
                final = "outdat/new/checkpoints/step000000000012/state.h5"
                datasets = compare_fields(continuous / final, resumed / final)
                control = "outdat/new/checkpoints/step000000000012/control.bin"
                if (continuous / control).read_bytes() != (resumed / control).read_bytes():
                    raise AssertionError("final control/schedule state mismatch")
                history = "outdat/new/checkpoints/step000000000012/archives.bin"
                if archive_schedule_payload(continuous / history) != archive_schedule_payload(resumed / history):
                    raise AssertionError("final field/slice schedule history mismatch")
                compare_fields(continuous / "outdat/new/resources/geometry.h5",
                               resumed / "outdat/new/resources/geometry.h5")
                homogeneous = {"tgv": (True, True, True), "channel": (True, False, True),
                               "curve": (False, False, True), "dynamic": (False, False, True)}[args.case]
                geometry_padding = check_geometry_padding(
                    continuous / "outdat/new/resources/geometry.h5", homogeneous)
                check_geometry_padding(resumed / "outdat/new/resources/geometry.h5", homogeneous)
                disabled, d_size = run_case(args, root, backend, ranks, "disabled", 12, enabled=False)
                if args.case != "tgv" or args.initial_dimension:
                    compare_fields(continuous / final, disabled / final)
                initial_info = None
                if args.initial_dimension:
                    resource = f"flowini{args.initial_dimension}d.h5"
                    relative = "outdat/new/resources/" + resource
                    expected = (continuous / relative).read_bytes()
                    if expected != (resumed / relative).read_bytes():
                        raise AssertionError("frozen initialization resource differs")
                    if (resumed / "datin" / resource).exists():
                        raise AssertionError("original initialization resource still exists")
                    initial_info = dict(dimension=args.initial_dimension, original_source_removed=True,
                                        file_bytes=len(expected),
                                        flow_controlled_host_bounds=controlled_checkpoint_buffers(
                                            continuous / final, 16*args.inflow_count if args.case == "dynamic" else 0))
                inflow_info = None
                if args.case == "dynamic":
                    inflow_final = "outdat/new/checkpoints/step000000000012/inflow.h5"
                    compare_fields(continuous / inflow_final, resumed / inflow_final)
                    compare_fields(continuous / inflow_final, disabled / inflow_final)
                    with h5py.File(source / "inflow.h5") as seed, h5py.File(continuous / inflow_final) as end:
                        initial_cursor, final_cursor = int(seed["metadata"][2]), int(end["metadata"][2])
                        if final_cursor <= initial_cursor or seed["identity"][12] != 8:
                            raise AssertionError("dynamic inflow did not cross a cache rollover")
                    index = "outdat/new/resources/inflow_index.bin"
                    if (continuous / index).read_bytes() != (resumed / index).read_bytes():
                        raise AssertionError("dynamic inflow frozen-source index differs")
                    source_names = sorted(p.name for p in (resumed / "outdat/new/resources").glob("islice*.h5"))
                    if source_names != [f"islice{i:05d}.h5" for i in range(args.inflow_count)]:
                        raise AssertionError("dynamic inflow frozen-source inventory differs")
                    inflow_info = dict(seed_cursor=initial_cursor, final_cursor=final_cursor,
                                       frozen_frames=len(source_names), original_sources_removed=True)
                    persistent_bytes = 16 * args.inflow_count
                    inflow_info["fingerprint_host_bytes"] = persistent_bytes
                    inflow_info["resource_metadata_host_bound_bytes"] = 288 * (args.inflow_count + 6) + 131072
                    for name, filename in [("inflow", inflow_final), ("flow", final),
                                           ("geometry", "outdat/new/resources/geometry.h5"),
                                           ("statistics", "outdat/new/checkpoints/step000000000012/statistics.h5")]:
                        if (continuous / filename).is_file():
                            inflow_info[name + "_file_bytes"] = (continuous / filename).stat().st_size
                            inflow_info[name + "_controlled_host_bounds"] = controlled_checkpoint_buffers(
                                continuous / filename, persistent_bytes)
                if args.statistics or args.legacy_statistics:
                    stats_file = "outdat/new/checkpoints/step000000000012/statistics.h5"
                    compare_fields(continuous / stats_file, resumed / stats_file)
                    compare_fields(continuous / stats_file, disabled / stats_file)
                    compare_fields(continuous / final, disabled / final)
                    if args.legacy_statistics:
                        with h5py.File(continuous / stats_file) as state:
                            metadata = state["metadata"][:]
                            if backend == "cpu":
                                dt = 6e-6 if args.case == "dynamic" else (1e-5 if args.case == "curve" else 1e-3)
                                expected_time = np.cumsum(np.full(11, dt, dtype=np.float64))[-1]
                                if (state["identity"][12] != 6 or metadata.shape != (8,)
                                        or metadata[:4].tolist() != [2, 11, 1, 11]
                                        or metadata[7:8].view(np.float64)[0] != expected_time):
                                    raise AssertionError("CPU raw-moment count/last-sample identity mismatch")
                            elif state["identity"][12] != 5 or metadata[7] != 11 or metadata[9] != 11:
                                raise AssertionError("GPU compact count/last-sample identity mismatch")
                    for rank in range(ranks) if args.statistics else []:
                        name = f"outdat/sample.statistics.step00000012.rank{rank:08d}.bin"
                        expected = (continuous / name).read_bytes()
                        if expected != (resumed / name).read_bytes() or expected != (disabled / name).read_bytes():
                            raise AssertionError("final statistics output differs")
                statistics = compare_statistics(continuous, resumed, disabled, backend, restart_step, args.case)
                expected_batches = (sorted(set(range(args.restart_step, 13, args.restart_step)) | {12})[-2:]
                                    if args.mode == "steps" else [8, 12])
                for case in (continuous, resumed):
                    actual_batches = sorted(p.name for p in (case / "outdat/new/checkpoints").glob("step*"))
                    if actual_batches != [f"step{step:012d}" for step in expected_batches]:
                        raise AssertionError(f"unexpected retained checkpoints: {actual_batches}")
                sample_count = 0
                for step in ([] if args.no_samples else range(1, 13)):
                    for rank in range(ranks):
                        filename = f"sample.step{step:08d}.rank{rank:08d}.bin"
                        actual = (continuous / "outdat" / filename).read_bytes()
                        if actual != (disabled / "outdat" / filename).read_bytes():
                            raise AssertionError(f"output changed completed sample: {filename}")
                        if step > restart_step and actual != (resumed / "outdat" / filename).read_bytes():
                            raise AssertionError(f"restart changed completed sample: {filename}")
                        sample_count += 1
                report["checks"].append(dict(backend=backend, np=ranks, axis=args.axis, mode=args.mode, datasets=datasets,
                                              case=args.case, force=args.force if args.case == "channel" else None,
                                              filter_workspace=args.filter_workspace,
                                              restart_step=restart_step,
                                              formal_statistics=args.statistics,
                                              legacy_statistics=args.legacy_statistics,
                                              checkpoint_comparison="periodic_vs_final_only" if args.statistics or args.legacy_statistics or args.case != "tgv" or args.initial_dimension else "on_vs_off",
                                              output_switch_field_check=not args.no_samples,
                                              samples=sample_count, statistics=statistics,
                                              geometry_padding_values=geometry_padding,
                                              dynamic_inflow=inflow_info,
                                              initial_resource=initial_info,
                                              bytes=[a_size, b_size, c_size, d_size]))
                if backend == "cpu" and ranks == 2:
                    run_case(args, root, backend, ranks, "reject_input", 12, restore=source,
                             reject="numerical/executable/controller contract mismatch", change_input=True)
                    corrupt = args.output / f"{backend}_np{ranks}_corrupt_resource"
                    shutil.copytree(first / "outdat/new", corrupt)
                    with (corrupt / "resources/input.txt").open("ab") as stream:
                        stream.write(b"changed")
                    run_case(args, root, backend, ranks, "reject_resource", 12,
                             restore=corrupt / f"checkpoints/step{restart_step:012d}",
                             reject="invalid new checkpoint bundle")
                    report["checks"][-1]["rejections"] = ["primary_input_mismatch", "resource_corruption"]
                    if args.initial_dimension:
                        initial_corrupt = args.output / f"{backend}_np{ranks}_corrupt_initial"
                        shutil.copytree(first / "outdat/new", initial_corrupt)
                        with (initial_corrupt / "resources" / f"flowini{args.initial_dimension}d.h5").open("ab") as stream:
                            stream.write(b"changed initial field")
                        run_case(args, root, backend, ranks, "reject_initial", 12,
                                 restore=initial_corrupt / f"checkpoints/step{restart_step:012d}",
                                 reject="invalid new checkpoint bundle")
                        run_case(args, root, backend, ranks, "reject_initial_external", 12,
                                 reject="cannot fingerprint external initialization resource",
                                 source_fault="initial_external")
                        report["checks"][-1]["rejections"] += ["initial_resource_corruption", "initial_hdf5_dependency"]
                    if args.case == "dynamic":
                        for fault, message in [
                            ("hole", "dynamic inflow requires a frozen contiguous regular-file sequence"),
                            ("external", "dynamic inflow source identity mismatch"),
                            ("initial_field", "cannot fingerprint external initialization resource"),
                        ]:
                            run_case(args, root, backend, ranks, "reject_" + fault, 12,
                                     reject=message, source_fault=fault)
                        frame_corrupt = args.output / f"{backend}_np{ranks}_corrupt_frame"
                        shutil.copytree(first / "outdat/new", frame_corrupt)
                        with (frame_corrupt / "resources/islice00001.h5").open("ab") as stream:
                            stream.write(b"changed frame")
                        run_case(args, root, backend, ranks, "reject_frame", 12,
                                 restore=frame_corrupt / f"checkpoints/step{restart_step:012d}",
                                 reject="invalid new checkpoint bundle")
                        report["checks"][-1]["rejections"] += ["source_hole", "hdf5_dependency",
                                                               "missing_initial_field", "source_frame_corruption"]
                if args.schedule_checks:
                    interval = 4 if args.mode == "steps" else 0.003
                    run_case(args, root, backend, ranks, "reject_schedule", 12, restore=source,
                             checkpoint_interval=interval,
                             reject="saved output schedule differs; select explicit override")
                    overridden, _ = run_case(args, root, backend, ranks, "override_schedule", 12,
                                             restore=source, checkpoint_interval=interval, override=True)
                    compare_fields(continuous / final, overridden / final)
                    if args.statistics:
                        compare_fields(continuous / stats_file, overridden / stats_file)
                    expected = ([8, 12] if restart_step == 0 else [9, 12]) if args.mode == "steps" else [11, 12]
                    if args.mode == "time" and restart_step == 0:
                        expected = [9, 12]
                    batches = sorted(p.name for p in (overridden / "outdat/new/checkpoints").glob("step*"))
                    if batches != [f"step{step:012d}" for step in expected]:
                        raise AssertionError(f"restart-relative schedule mismatch: {batches}")
                    report["checks"][-1]["schedule_override"] = expected
                    if args.mode == "time":
                        run_case(args, root, backend, ranks, "reject_tiny_interval", 12,
                                 checkpoint_interval=1e-300,
                                 reject="invalid checkpoint schedule clock or unrepresentable time targets")
                        report["checks"][-1]["unrepresentable_schedule_rejected"] = True
                (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
