#!/usr/bin/env python3
"""Static contracts for GPU-resident compact production statistics."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]


def source(relative: str) -> str:
    path = ROOT / relative
    assert path.is_file(), f"missing production source: {relative}"
    return path.read_text(encoding="utf-8")


def function_body(text: str, name: str) -> str:
    match = re.search(
        rf"(?:logical function|subroutine)\s+{name}\b(.*?)end (?:function|subroutine)\s+{name}",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    assert match is not None, f"missing routine {name}"
    return match.group(1)


def test_cmake_orders_production_statistics_before_gpu_runtime():
    cmake = source("src/CMakeLists.txt")
    assert "src_gpu/production_statistics_gpu.cuf" in cmake
    assert cmake.index("src_gpu/production_statistics_gpu.cuf") < cmake.index("src_gpu/gpu_runtime.cuf")
    assert "compact_statistics_probe" in cmake


def test_capability_gate_matches_approved_bl_scope():
    text = source("src_gpu/production_statistics_gpu.cuf")
    body = function_body(text, "compact_statistics_supported")
    compact = body.replace(" ", "").replace("\n", "").lower()
    assert "trim(flowtype)=='bl'" in compact
    assert "trim(flowtype)=='swbli'" in compact
    assert "ndims==3" in compact
    assert "numq==5" in compact
    assert "num_species==0" in compact
    assert "trim(turbmode)=='none'" in compact
    assert "lkhomo" in compact
    assert "all(bctype(5:6)==1)" in compact
    for wall_kind in (41, 42, 411, 421):
        assert f"bctype(3)=={wall_kind}" in compact


def test_request_gate_does_not_capture_non_bl_lavg_cases():
    text = source("src_gpu/production_statistics_gpu.cuf")
    requested = function_body(text, "compact_statistics_requested")
    compact = requested.replace(" ", "").replace("\n", "").lower()
    assert "use_gpu.and.lavg" in compact
    assert "trim(flowtype)=='bl'" in compact
    assert "trim(flowtype)=='swbli'" in compact


def test_storage_is_compact_and_wall_storage_is_ymin_owned():
    text = source("src_gpu/production_statistics_gpu.cuf")
    assert re.search(r"plane_measure_d\s*\(:,:\)", text)
    assert re.search(r"plane_sum_d\s*\(:,:,:\)", text)
    assert re.search(r"wall_measure_d\s*\(:\)", text)
    assert re.search(r"wall_sum_d\s*\(:,:\)", text)
    initialize = function_body(text, "initialize_compact_statistics_gpu")
    assert re.search(r"if\s*\(jrk\s*==\s*0\)\s*then", initialize, flags=re.IGNORECASE)
    assert "allocate(wall_measure_d(0:im))" in initialize
    assert "allocate(wall_sum_d(0:im,1:3))" in initialize


def test_measure_kernel_owns_each_local_z_segment_once():
    text = source("src_gpu/production_statistics_gpu.cuf")
    body = function_body(text, "compact_plane_measure_kernel")
    assert re.search(r"do\s+k\s*=\s*1\s*,\s*km", body, flags=re.IGNORECASE)
    assert "x(i,j,k,1)-x(i,j,k-1,1)" in body.replace(" ", "")
    assert "sqrt(dx*dx+dy*dy+dz*dz)" in body.replace(" ", "")


def test_every_measure_kernel_launch_is_explicitly_synchronized():
    text = source("src_gpu/production_statistics_gpu.cuf")
    initialize = function_body(text, "initialize_compact_statistics_gpu")
    assert "compact_plane_measure_kernel<<<" in initialize
    assert "sync_after_kernel('compact_plane_measure_kernel')" in initialize
    assert "compact_wall_measure_kernel<<<" in initialize
    assert "sync_after_kernel('compact_wall_measure_kernel')" in initialize


def test_capability_failure_uses_mpi_abort_without_host_fallback():
    text = source("src_gpu/production_statistics_gpu.cuf")
    initialize = function_body(text, "initialize_compact_statistics_gpu")
    assert "mpi_abort" in initialize.lower()
    assert "copy_flow_from_gpu" not in text
    assert "meanflowcal" not in text.lower()


def test_plane_accumulation_has_sampling_gate_and_explicit_sync():
    text = source("src_gpu/production_statistics_gpu.cuf")
    kernel = function_body(text, "compact_plane_moment_kernel").replace(" ", "").lower()
    assert "dok=1,km" in kernel
    for expression in (
        "rho*u",
        "rho*v",
        "rho*w",
        "rho*t",
        "rho*u*u",
        "rho*v*v",
        "rho*w*w",
        "rho*u*v",
        "rho*u*w",
        "rho*v*w",
    ):
        assert expression in kernel
    accumulate = function_body(text, "accumulate_compact_statistics_gpu")
    compact = accumulate.replace(" ", "").replace("\n", "").lower()
    assert "nstep<=0" in compact
    assert "mod(nstep,feqavg)/=0" in compact
    assert "int(nstep,int64)==last_sample_step" in compact
    assert "compact_plane_moment_kernel<<<" in accumulate
    assert "sync_after_kernel('compact_plane_moment_kernel')" in accumulate
    assert compact.index("sync_after_kernel") < compact.index("sample_count=sample_count+1")


def test_wall_statistics_use_geometric_projection_and_explicit_sync():
    text = source("src_gpu/production_statistics_gpu.cuf")
    point = function_body(text, "compact_wall_point_values").replace(" ", "").lower()
    assert "dot_product(di,normal)" in point
    assert "di-dot_product(di,normal)*normal" in point
    assert "dot_product(tangent,traction)" in point
    assert "dot_product(grad_t,normal)" in point
    assert "tempconst1" in point
    kernel = function_body(text, "compact_wall_moment_kernel")
    assert re.search(r"do\s+k\s*=\s*1\s*,\s*km", kernel, flags=re.IGNORECASE)
    accumulate = function_body(text, "accumulate_compact_statistics_gpu")
    assert "if(jrk==0)" in accumulate.replace(" ", "")
    assert "compact_wall_moment_kernel<<<" in accumulate
    assert "sync_after_kernel('compact_wall_moment_kernel')" in accumulate


def test_gpu_runtime_owns_compact_statistics_lifecycle():
    runtime = source("src_gpu/gpu_runtime.cuf")
    after_flowinit = function_body(runtime, "gpu_after_flowinit")
    finalize = function_body(runtime, "gpu_before_finalize")
    assert "initialize_compact_statistics_gpu" in after_flowinit
    assert "release_compact_statistics_gpu" in finalize
    requested = function_body(runtime, "gpu_compact_statistics_requested")
    assert "compact_statistics_requested" in requested
    accumulate = function_body(runtime, "gpu_accumulate_compact_statistics")
    assert "accumulate_compact_statistics_gpu" in accumulate


def test_gpu_runtime_releases_all_persistent_application_allocations():
    commarray = source("src_gpu/commarray_gpu.cuf")
    release_core = function_body(commarray, "release_gpu_arrays")
    for name in (
        "q_d", "qrhs_d", "qsave_d", "qwork_d", "filter_work_d",
        "rho_d", "prs_d", "tmp_d", "vel_d", "jacob_d", "x_d", "dxi_d",
        "bnorm_i0_d", "bnorm_im_d", "bnorm_j0_d", "bnorm_jm_d",
        "bnorm_k0_d", "bnorm_km_d", "dvel_d", "dtmp_d", "sigma_d",
        "qflux_d", "flux_work_d", "flux_characteristic_work_d",
        "shock_sensor_d", "shock_mask_d",
    ):
        assert re.search(rf"if\s*\(\s*allocated\s*\(\s*{name}\s*\)\s*\)\s*deallocate\s*\(\s*{name}\s*\)", release_core, re.I)

    halo = source("src_gpu/halo_exchange_gpu.cuf")
    release_halo = function_body(halo, "release_halo_exchange_buffers_gpu")
    for name in re.findall(r"(?:device|target)\s*::\s*([^\n]+)", halo, re.I):
        for variable in re.findall(r"\b([a-z][a-z0-9_]*_[dh])\s*\(", name, re.I):
            assert f"allocated({variable})" in release_halo.replace(" ", "").lower()

    boundary = source("src_gpu/boundary_gpu.cuf")
    release_boundary = function_body(boundary, "release_boundary_gpu")
    for name in ("s1_bl_rho_prof_d", "s1_bl_prs_prof_d", "s1_bl_tmp_prof_d", "s1_bl_vel_prof_d"):
        assert f"allocated({name})" in release_boundary.replace(" ", "").lower()

    assert "allocated(sponge_im_coef_d)" in function_body(
        source("src_gpu/sponge_gpu.cuf"), "release_sponge_gpu"
    ).replace(" ", "").lower()
    assert "allocated(xline_d)" in function_body(
        source("src_gpu/conservative_boundary_stage_gpu.cuf"),
        "release_conservative_stage_gpu",
    ).replace(" ", "").lower()

    finalize = function_body(source("src_gpu/gpu_runtime.cuf"), "gpu_before_finalize")
    for routine in (
        "release_compact_statistics_gpu", "release_dynamic_inflow_gpu",
        "release_boundary_gpu", "release_sponge_gpu",
        "release_conservative_stage_gpu", "release_host_buffers",
        "release_halo_exchange_buffers_gpu", "release_gpu_arrays",
    ):
        assert f"call {routine}()" in finalize.lower()


def test_s1_bl_reduction_kernel_uses_explicit_workspace_shape():
    text = source("src_gpu/statistic_gpu.cuf")
    kernel = function_body(text, "s1_bl_partial_kernel")
    compact = kernel.replace(" ", "").replace("\n", "").lower()
    assert "integer,value::im,jm,km,hm,nblocks,apply_down" in compact
    assert "real(8),device::partial(nblocks,6)" in compact
    assert "real(8),device::partial(:,:)" not in compact


def test_initialisation_skips_cpu_meanflow_restore_under_compact_ownership():
    initialisation = source("src/initialisation.F90")
    flowinit = function_body(initialisation, "flowinit")
    compact = flowinit.replace(" ", "").replace("\n", "").lower()
    assert "gpu_compact_statistics_requested" in compact
    assert "if(.not.(use_gpu.and.gpu_compact_statistics_requested()))" in compact
    assert compact.index("if(.not.(use_gpu.and.gpu_compact_statistics_requested()))") < compact.index(
        "callreadmeanflow(mode='h')"
    )


def test_mainloop_samples_between_preparation_and_snapshot_restore():
    mainloop = source("src/mainloop.F90")
    integration = function_body(mainloop, "time_integration_rk")
    prepare = integration.index("call gpu_prepare_rkfirst_stats()")
    sample = integration.index("call gpu_accumulate_compact_statistics()")
    restore = integration.index("call gpu_restore_stats_snapshot()")
    assert prepare < sample < restore


def test_sampling_updates_checkpoint_sample_count_without_full_field_copy():
    text = source("src_gpu/production_statistics_gpu.cuf")
    accumulate = function_body(text, "accumulate_compact_statistics_gpu")
    assert "use statistic, only: nsamples" in accumulate
    compact = accumulate.replace(" ", "").replace("\n", "").lower()
    assert "nsamples=int(sample_count)" in compact
    assert "copy_flow_from_gpu" not in accumulate


def test_checkpoint_transaction_wraps_existing_flow_checkpoint():
    mainloop = source("src/mainloop.F90")
    integration = function_body(mainloop, "time_integration_rk")
    prepare = integration.index("call gpu_prepare_compact_statistics_checkpoint()")
    flow = integration.index("call writechkpt()", prepare)
    commit = integration.index("call gpu_commit_compact_statistics_checkpoint()", flow)
    assert prepare < flow < commit


def test_sidecar_uses_fixed_width_stream_and_atomic_publish():
    text = source("src_gpu/production_statistics_gpu.cuf")
    prepare = function_body(text, "prepare_compact_statistics_checkpoint").lower()
    assert "access='stream'" in prepare
    assert "convert='little_endian'" in prepare
    assert "integer(int32)" in prepare
    assert "integer(int64)" in prepare
    assert "compact_stats.rank" in text.lower()
    assert "trim(final_path)//'.tmp'" in text.lower()
    assert "plane_measure_d" in prepare
    assert "plane_sum_d" in prepare
    assert "copy_flow_from_gpu" not in prepare
    commit = function_body(text, "commit_compact_statistics_checkpoint").lower()
    assert "c_rename" in commit
    assert "mpi_barrier" in commit
    assert "bakup/compact_stats.rank" in text.lower()


def test_restart_requires_same_checkpoint_generation_and_topology():
    text = source("src_gpu/production_statistics_gpu.cuf")
    restore = function_body(text, "restore_compact_statistics_checkpoint").lower()
    for field in (
        "global_dims",
        "topology",
        "rank_coords",
        "offsets",
        "local_dims",
        "checkpoint_step",
        "checkpoint_time",
        "sample_count",
    ):
        assert field in restore
    assert "mpi_abort" in restore
    assert "last_sample_step=int(nstep,int64)" in restore.replace(" ", "").replace("\n", "")


def test_gpu_runtime_exposes_checkpoint_facade():
    runtime = source("src_gpu/gpu_runtime.cuf")
    assert "gpu_prepare_compact_statistics_checkpoint" in runtime
    assert "gpu_commit_compact_statistics_checkpoint" in runtime


def test_cpu_compact_validation_snapshot_follows_meanflow_sample():
    mainloop = source("src/mainloop.F90")
    rkfirst = function_body(mainloop, "rkfirst")
    sample = rkfirst.index("call meanflowcal")
    snapshot = rkfirst.index("call write_compact_statistics_validation_snapshot", sample)
    assert sample < snapshot

    validation = source("src/validation_io.F90")
    writer = function_body(validation, "write_compact_statistics_validation_snapshot")
    assert "ASTR_VALIDATION_COMPACT_PREFIX" in validation
    compact = writer.replace(" ", "").replace("\n", "").lower()
    assert "nstep<=0" in compact
    assert "mod(nstep,feqavg)/=0" in compact
    assert "access='stream'" in writer.lower()
    for field in ("x", "rho", "vel", "prs", "tmp", "dvel", "dtmp", "bnorm_j0"):
        assert field in writer


def test_compact_statistics_matrix_covers_required_solver_topologies():
    driver = source("tests/gpu_validation/run_compact_statistics_matrix.sh")
    for selector in (
        "static-np1",
        "dynamic-np1",
        "curve-filter-np4",
        "z-slab-np2",
        "xyz-np8",
        "restart-np1",
        "all",
    ):
        assert selector in driver
    for topology in ("1,1,1", "2,2,1", "1,1,2", "2,2,2"):
        assert topology in driver
    assert "ASTR_VALIDATION_COMPACT_PREFIX" in driver
    assert "compact_statistics_host_reference.py" in driver
    assert "assemble_compact_statistics.py" in driver
    assert "refusing to overwrite" in driver
    assert "CUDA_VISIBLE_DEVICES" in driver


def test_compact_statistics_profile_has_safety_residency_and_overhead_gates():
    driver = source("tests/gpu_validation/run_compact_statistics_profile.sh")
    for mode in ("memcheck", "residency", "performance"):
        assert mode in driver
    assert "--tool memcheck" in driver
    assert "--leak-check full" in driver
    assert "--tool racecheck" in driver
    assert "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)" in driver
    assert "OMPI_MCA_opal_cuda_support=0" in driver
    assert "OMPI_MCA_coll=^hcoll,ucc" in driver
    assert "UCX_MEMTYPE_CACHE=n" in driver
    assert "nsys profile" in driver
    assert "analyze_nsys_rk_residency.py" in driver
    assert "ASTR_GPU_COMPLETE_STEP_TIMING=1" in driver
    assert "REPEATS=\"${REPEATS:-5}\"" in driver
    assert "MAX_OVERHEAD_PERCENT=\"${MAX_OVERHEAD_PERCENT:-10.0}\"" in driver
    assert "refusing to overwrite" in driver
    for metadata in ("git_commit", "executable_sha256", "gpu_model", "topology", "grid", "feqavg"):
        assert metadata in driver


def test_complete_step_timing_includes_compact_statistics_and_rk_advance():
    runtime = source("src_gpu/gpu_runtime.cuf")
    assert "ASTR_GPU_COMPLETE_STEP_TIMING" in runtime
    assert "gpu_begin_complete_step_timing" in runtime
    assert "gpu_end_complete_step_timing" in runtime
    mainloop = source("src/mainloop.F90")
    integration = function_body(mainloop, "time_integration_rk")
    begin = integration.index("call gpu_begin_complete_step_timing()")
    compact = integration.index("call gpu_accumulate_compact_statistics()")
    advance = integration.index("call gpu_time_integration_rk(.true.,.false.)")
    end = integration.index("call gpu_end_complete_step_timing()")
    assert begin < compact < advance < end
