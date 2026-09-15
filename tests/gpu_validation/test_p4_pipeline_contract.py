"""Source contracts for the first P4 host-staged halo pipeline."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
TRANSPORT = (ROOT / "src_gpu/halo_transport_gpu.cuf").read_text(encoding="utf-8")
EXCHANGE = (ROOT / "src_gpu/halo_exchange_gpu.cuf").read_text(encoding="utf-8")
GPU_CHECK = (ROOT / "src_gpu/gpu_check.cuf").read_text(encoding="utf-8")
RUNTIME = (ROOT / "src_gpu/gpu_runtime.cuf").read_text(encoding="utf-8")
MAINLOOP = (ROOT / "src_gpu/mainloop_gpu.cuf").read_text(encoding="utf-8")
SOLVER = (ROOT / "src_gpu/solver_gpu.cuf").read_text(encoding="utf-8")
STATISTIC = (ROOT / "src_gpu/statistic_gpu.cuf").read_text(encoding="utf-8")
DRIVER = (ROOT / "tests/gpu_validation/run_tgv_256_performance_benchmark.sh").read_text(
    encoding="utf-8"
)
FIELD_DRIVER = (ROOT / "tests/gpu_validation/run_tgv_mpirank2_field_compare.sh").read_text(
    encoding="utf-8"
)
PERFORMANCE_SUMMARY = (
    ROOT / "tests/gpu_validation/summarize_tgv_performance.py"
).read_text(encoding="utf-8")


def compact(source: str) -> str:
    return re.sub(r"\s+", "", source).lower()


class P4PipelineContract(unittest.TestCase):
    def test_tgv_reduction_kernels_do_not_allocate_device_descriptors(self) -> None:
        source = compact(STATISTIC)
        for kernel in (
            "kenergy_partial_kernel",
            "enstophy_partial_kernel",
            "dissipation_partial_kernel",
            "maxq_partial_kernel",
            "enstophy_partial_sp_kernel",
            "dissipation_partial_sp_kernel",
        ):
            start = source.index(f"subroutine{kernel}")
            end = source.index(f"endsubroutine{kernel}", start)
            routine = source[start:end]
            self.assertIn("real(8),device::partial(*)", routine)
            self.assertNotIn("real(8),device::partial(:)", routine)

    def test_pipeline_backend_has_collective_fail_closed_setup(self) -> None:
        source = compact(TRANSPORT)
        self.assertIn("if(trim(value)=='pinned-pipeline')choice=3", source)
        self.assertIn("halo_pipeline_enabled=choice==3", source)
        self.assertIn(
            "cudastreamcreatewithflags(pipeline_contexts(axis)%stream,cudastreamnonblocking)",
            source,
        )
        self.assertIn(
            "cudaeventcreatewithflags(pipeline_contexts(axis)%send_ready_event,cudaeventdisabletiming)",
            source,
        )
        self.assertIn(
            "cudaeventcreatewithflags(pipeline_contexts(axis)%halo_ready_event,cudaeventdisabletiming)",
            source,
        )
        self.assertIn(
            "cudastreamcreatewithflags(pipeline_compute_stream,cudastreamnonblocking)",
            source,
        )
        self.assertIn(
            "cudaeventcreatewithflags(pipeline_compute_done_event,cudaeventdisabletiming)",
            source,
        )
        self.assertIn("callmpi_allreduce(resource_failed,any_resource_failed", source)

    def test_pipeline_context_owns_request_state_transitions(self) -> None:
        source = compact(TRANSPORT)
        for state in (
            "pipeline_idle",
            "pipeline_receive_posted",
            "pipeline_device_packing",
            "pipeline_mpi_active",
            "pipeline_host_complete",
            "pipeline_device_uploading",
        ):
            self.assertIn(state, source)
        self.assertIn("subroutinebegin_host_pair_pipeline", source)
        self.assertIn("subroutinecomplete_host_pair_d2h", source)
        self.assertIn("subroutinefinish_host_pair_pipeline", source)
        self.assertIn("callmpi_irecv", source)
        self.assertIn("callmpi_isend", source)
        self.assertIn("callmpi_testall", source)

    def test_pipeline_transport_owns_one_context_per_mpi_axis(self) -> None:
        source = compact(TRANSPORT)
        self.assertIn("integer,parameter,public::halo_axis_x=1", source)
        self.assertIn("integer,parameter,public::halo_axis_y=2", source)
        self.assertIn("integer,parameter,public::halo_axis_z=3", source)
        self.assertIn("type::halo_pipeline_context", source)
        self.assertIn("type(halo_pipeline_context),save::pipeline_contexts(3)", source)
        self.assertIn("subroutineconfigure_pipeline_axes(active)", source)
        self.assertIn("pipeline_axis_enabled=active", source)

    def test_pipeline_api_indexes_context_and_splits_mpi_start_from_wait(self) -> None:
        source = compact(TRANSPORT)
        self.assertIn("subroutinebegin_host_pair_pipeline(axis,", source)
        self.assertIn("subroutinebegin_host_pair_device_pack(axis,stream)", source)
        self.assertIn("subroutinestart_host_pair_mpi(axis,send_low,send_high)", source)
        self.assertIn("subroutinecomplete_host_pair_mpi(axis,work)", source)
        self.assertIn("subroutinebegin_host_pair_device_upload(axis,stream)", source)
        self.assertIn("subroutinefinish_host_pair_pipeline(axis)", source)
        self.assertIn("pipeline_contexts(axis)%requests", source)

    def test_dependency_mode_uses_stream_ordering_for_nonforced_kernels(self) -> None:
        source = compact(GPU_CHECK)
        self.assertIn("case('dependency')", source)
        self.assertIn("dependency_sync_enabled=choice==2", source)
        self.assertIn("logicalfunctiongpu_dependency_sync_enabled", source)
        self.assertIn(
            "must_sync=.not.(selective_sync_enabled.or.dependency_sync_enabled)",
            source,
        )
        self.assertIn("must_sync=must_sync.or.force_sync", source)

    def test_pipeline_has_default_stream_source_dependency_event(self) -> None:
        source = compact(TRANSPORT)
        self.assertIn(
            "cudaeventcreatewithflags(pipeline_contexts(axis)%source_ready_event,cudaeventdisabletiming)",
            source,
        )
        start = source.index("subroutinebegin_host_pair_device_pack(axis,stream)")
        end = source.index("endsubroutinebegin_host_pair_device_pack", start)
        routine = source[start:end]
        record = routine.index(
            "cudaeventrecord(pipeline_contexts(axis)%source_ready_event,0_cuda_stream_kind)"
        )
        wait = routine.index(
            "cudastreamwaitevent(pipeline_contexts(axis)%stream,&pipeline_contexts(axis)%source_ready_event,0)"
        )
        self.assertLess(record, wait)

        start = source.index(
            "subroutinebegin_pipeline_compute(axis,stream,wait_for_source)"
        )
        end = source.index("endsubroutinebegin_pipeline_compute", start)
        routine = source[start:end]
        self.assertIn("logical,intent(in),optional::wait_for_source", routine)
        self.assertIn("if(source_dependency)then", routine)
        self.assertIn(
            "cudastreamwaitevent(pipeline_compute_stream,&pipeline_contexts(axis)%source_ready_event,0)",
            routine,
        )

    def test_runtime_reports_dependency_mode(self) -> None:
        source = compact(RUNTIME)
        self.assertIn("gpu_dependency_sync_enabled", source)
        self.assertIn("**gpusynchronizationmode:dependency", source)

    def test_x_solution_halo_uses_async_pipeline(self) -> None:
        source = compact(EXCHANGE)
        self.assertIn("if(halo_pipeline_enabled)then", source)
        self.assertIn("callbegin_host_pair_pipeline", source)
        self.assertIn("<<<grid,block,0,pipeline_stream>>>", source)
        self.assertGreaterEqual(source.count("cudamemcpyasync("), 4)
        self.assertIn("callcomplete_host_pair_d2h", source)
        self.assertIn("callfinish_host_pair_pipeline", source)

    def test_solution_pipeline_exposes_begin_finish_for_independent_work(self) -> None:
        source = compact(EXCHANGE)
        self.assertIn("subroutinebegin_solution_halo_pipeline_gpu", source)
        self.assertIn("subroutinefinish_solution_halo_pipeline_gpu", source)
        begin = source.index("subroutinebegin_solution_halo_pipeline_gpu")
        finish = source.index("subroutinefinish_solution_halo_pipeline_gpu")
        self.assertLess(begin, finish)

    def test_rhs_zeroing_is_launched_inside_solution_halo_window(self) -> None:
        source = compact(MAINLOOP)
        begin = source.index("callbegin_solution_halo_pipeline_gpu(.true.)")
        compute = source.index(
            "callbegin_pipeline_compute(solution_pipeline_axis,solution_compute_stream,.true.)", begin
        )
        zero = source.index(
            "callzero_rhs_kernel<<<grid1,block1,0,solution_compute_stream>>>",
            compute,
        )
        d2h = source.index("callstage_solution_halo_d2h_gpu()", zero)
        finish = source.index("callfinish_solution_halo_pipeline_gpu()", begin)
        self.assertLess(begin, zero)
        self.assertLess(zero, d2h)
        self.assertLess(d2h, finish)
        self.assertIn("callcheck_kernel_launch('pipeline_zero_rhs_kernel')", source)
        self.assertIn("callrecord_pipeline_compute()", source)
        self.assertIn("callfinish_pipeline_compute()", source)

    def test_periodic_x_filter_has_disjoint_interior_and_boundary_kernels(self) -> None:
        source = compact(SOLVER)
        self.assertIn("subroutinefilter_x_periodic_interior_global_kernel", source)
        self.assertIn("subroutinefilter_x_periodic_boundary_global_kernel", source)
        self.assertIn("ni=im-2*hm+1", source)
        self.assertIn("if(i>=hm.and.i<=im-hm)return", source)

    def test_filter_interior_is_launched_inside_x_halo_window(self) -> None:
        source = compact(MAINLOOP)
        begin = source.index("callbegin_x_filter_halo_pipeline_gpu()")
        compute = source.index(
            "callbegin_pipeline_compute(halo_axis_x,filter_compute_stream,.true.)", begin
        )
        interior = source.index(
            "callfilter_x_periodic_interior_global_kernel<<<grid_x_flat,block_x,0,filter_compute_stream>>>",
            compute,
        )
        d2h = source.index("callstage_x_filter_halo_d2h_gpu()", interior)
        finish = source.index("callfinish_x_filter_halo_pipeline_gpu()", begin)
        boundary = source.index(
            "callfilter_x_periodic_boundary_global_kernel<<<", finish
        )
        self.assertLess(begin, interior)
        self.assertLess(interior, d2h)
        self.assertLess(d2h, finish)
        self.assertLess(finish, boundary)

    def test_pipeline_mpi_phase_accepts_independent_work(self) -> None:
        source = compact(TRANSPORT)
        start = source.index("subroutinecomplete_host_pair_d2h(axis,send_low,send_high,work)")
        end = source.index("endsubroutinecomplete_host_pair_d2h", start)
        routine = source[start:end]
        self.assertIn("callstart_host_pair_mpi(axis,send_low,send_high)", routine)
        self.assertIn("callcomplete_host_pair_mpi(axis,work)", routine)

    def test_diffusion_pipeline_fuses_sigma_and_qflux_halos(self) -> None:
        source = compact(EXCHANGE)
        self.assertIn("integer,parameter::max_field_vars=9", source)
        self.assertIn("subroutinepack_diffusion_x_fused_kernel", source)
        self.assertIn("subroutineunpack_diffusion_x_fused_kernel", source)
        start = source.index("subroutineexchange_diffusion_flux_halo_gpu(work)")
        end = source.index("endsubroutineexchange_diffusion_flux_halo_gpu", start)
        routine = source[start:end]
        self.assertIn("if(halo_pipeline_enabled)then", routine)
        self.assertIn("callexchange_diffusion_x_fused_pipeline_gpu(work)", routine)

    def test_nine_component_buffers_are_pipeline_only(self) -> None:
        source = compact(EXCHANGE)
        start = source.index("subroutineinit_halo_transport_gpu()")
        end = source.index("endsubroutineinit_halo_transport_gpu", start)
        routine = source[start:end]
        self.assertIn(
            "if(halo_pipeline_enabled)then"
            "callensure_field_buffers(max_field_vars)"
            "else"
            "callensure_field_buffers(6)",
            routine,
        )
        self.assertIn(
            "allocate(field_send_left_d(0:hm-1,0:jm,0:km,1:capacity))",
            source,
        )
        self.assertIn(
            "allocate(field_send_down_d(0:im,0:hm-1,0:km,1:capacity))",
            source,
        )
        self.assertIn(
            "allocate(field_send_back_d(0:im,0:jm,0:hm-1,1:capacity))",
            source,
        )

    def test_fused_diffusion_pipeline_uses_one_host_pair_transaction(self) -> None:
        source = compact(EXCHANGE)
        start = source.index("subroutineexchange_diffusion_x_fused_pipeline_gpu(work)")
        end = source.index("endsubroutineexchange_diffusion_x_fused_pipeline_gpu", start)
        routine = source[start:end]
        self.assertEqual(routine.count("callbegin_host_pair_pipeline"), 1)
        self.assertEqual(routine.count("callcomplete_host_pair_d2h"), 1)
        self.assertEqual(routine.count("callfinish_host_pair_pipeline"), 1)
        self.assertIn("callcomplete_host_pair_d2h(", routine)
        self.assertIn(",work)", routine)

    def test_pipeline_enables_periodic_diffusion_overlap(self) -> None:
        source = compact(MAINLOOP)
        self.assertIn(
            "overlap_diffusion=(halo_overlap_enabled.or.halo_pipeline_enabled).and.diffterm",
            source,
        )

    def test_pipeline_progress_polls_mpi_against_compute_event(self) -> None:
        source = compact(TRANSPORT)
        start = source.index("subroutineprogress_host_pair_during_pipeline_compute(axis,requests)")
        end = source.index(
            "endsubroutineprogress_host_pair_during_pipeline_compute", start
        )
        routine = source[start:end]
        self.assertIn("callmpi_testall(4,requests,done,statuses,ierr)", routine)
        self.assertIn("cudaeventquery(pipeline_compute_done_event)", routine)

    def test_diffusion_interior_uses_dedicated_pipeline_stream(self) -> None:
        source = compact(MAINLOOP)
        start = source.index("subroutinediffusion_interior_work(requests,axis)")
        end = source.index("endsubroutinediffusion_interior_work", start)
        routine = source[start:end]
        self.assertIn(
            "if(halo_pipeline_enabled.and..not.mixed_viscous_flux_workspace)then",
            routine,
        )
        self.assertIn(
            "callbegin_pipeline_compute(axis,diffusion_compute_stream,.true.)", routine
        )
        self.assertIn(
            "calldiffusion_rhs_x_stored_global_kernel<<<grid_x_flat,block_x,0,diffusion_compute_stream>>>",
            routine,
        )
        self.assertIn("callrecord_pipeline_compute()", routine)
        self.assertIn("callprogress_host_pair_during_pipeline_compute(axis,requests)", routine)
        self.assertIn("callfinish_pipeline_compute()", routine)

    def test_pipeline_accepts_periodic_tgv_multi_axis_topologies(self) -> None:
        source = compact(EXCHANGE)
        self.assertIn("trim(flowtype)/='tgv'", source)
        self.assertIn(
            "active_axes=merge(1,0,isize>1)+merge(1,0,jsize>1)+"
            "merge(1,0,ksize>1)",
            source,
        )
        self.assertIn("active_axes<1", source)
        self.assertNotIn("active_axes/=1", source)
        self.assertIn("callconfigure_pipeline_axes((/isize>1,jsize>1,ksize>1/))", source)
        self.assertNotIn("isize<=1.or.jsize/=1.or.ksize/=1", source)
        self.assertIn(".not.(lihomo.and.ljhomo.and.lkhomo)", source)

    def test_solution_pipeline_dispatches_all_slab_axes(self) -> None:
        source = compact(EXCHANGE)
        begin = source.index("subroutinebegin_solution_halo_pipeline_gpu")
        finish = source.index("endsubroutinefinish_solution_halo_pipeline_gpu", begin)
        routine = source[begin:finish]
        for axis in ("x", "y", "z"):
            self.assertIn(f"callbegin_{axis}_halo_pipeline_gpu()", routine)
            self.assertIn(f"callfinish_{axis}_halo_pipeline_gpu()", routine)
        self.assertIn("subroutinestage_solution_halo_d2h_gpu", source)
        self.assertIn("callstage_y_solution_halo_d2h_gpu()", source)
        self.assertIn("callstage_z_solution_halo_d2h_gpu()", source)
        self.assertIn("callstage_solution_halo_d2h_gpu()", compact(MAINLOOP))

    def test_solution_pipeline_starts_all_active_axes_before_ordered_finish(self) -> None:
        source = compact(EXCHANGE)
        begin = source.index("subroutinebegin_solution_halo_pipeline_gpu")
        end = source.index("endsubroutinebegin_solution_halo_pipeline_gpu", begin)
        routine = source[begin:end]
        self.assertIn("if(isize>1)then", routine)
        self.assertIn("if(jsize>1)then", routine)
        self.assertIn("if(ksize>1)then", routine)
        self.assertNotIn("elseif(jsize>1)", routine)
        self.assertIn("subroutinestart_solution_halo_mpi_gpu", source)
        finish = source.index("subroutinefinish_solution_halo_pipeline_gpu")
        finish_end = source.index("endsubroutinefinish_solution_halo_pipeline_gpu", finish)
        finish_routine = source[finish:finish_end]
        start_all = finish_routine.index("callstart_solution_halo_mpi_gpu()")
        finish_x = finish_routine.index("callfinish_x_halo_pipeline_gpu()")
        finish_y = finish_routine.index("callfinish_y_halo_pipeline_gpu()")
        finish_z = finish_routine.index("callfinish_z_halo_pipeline_gpu()")
        self.assertLess(start_all, finish_x)
        self.assertLess(finish_x, finish_y)
        self.assertLess(finish_y, finish_z)

    def test_periodic_filter_pipeline_has_y_and_z_interior_boundary_pairs(self) -> None:
        exchange = compact(EXCHANGE)
        solver = compact(SOLVER)
        mainloop = compact(MAINLOOP)
        for axis in ("y", "z"):
            self.assertIn(f"subroutinebegin_{axis}_filter_halo_pipeline_gpu", exchange)
            self.assertIn(f"subroutinestage_{axis}_filter_halo_d2h_gpu", exchange)
            self.assertIn(f"subroutinefinish_{axis}_filter_halo_pipeline_gpu", exchange)
            self.assertIn(f"subroutinefilter_{axis}_periodic_interior_global_kernel", solver)
            self.assertIn(f"subroutinefilter_{axis}_periodic_boundary_global_kernel", solver)
            self.assertIn(f"callbegin_{axis}_filter_halo_pipeline_gpu()", mainloop)
            self.assertIn(f"callfinish_{axis}_filter_halo_pipeline_gpu()", mainloop)

    def test_fused_diffusion_pipeline_dispatches_all_slab_axes(self) -> None:
        source = compact(EXCHANGE)
        for axis in ("x", "y", "z"):
            self.assertIn(f"subroutinepack_diffusion_{axis}_fused_kernel", source)
            self.assertIn(f"subroutineunpack_diffusion_{axis}_fused_kernel", source)
            self.assertIn(f"callexchange_diffusion_{axis}_fused_pipeline_gpu(work)", source)

    def test_fused_diffusion_pipeline_visits_every_active_axis_once(self) -> None:
        source = compact(EXCHANGE)
        start = source.index("subroutineexchange_diffusion_flux_halo_gpu(work)")
        end = source.index("endsubroutineexchange_diffusion_flux_halo_gpu", start)
        routine = source[start:end]
        self.assertIn("if(isize>1)then", routine)
        self.assertIn("if(jsize>1)then", routine)
        self.assertIn("if(ksize>1)then", routine)
        self.assertNotIn("elseif(jsize>1)", routine)
        self.assertIn("work_used=.true.", routine)

    def test_pipeline_field_buffers_expand_only_on_active_slab_axis(self) -> None:
        source = compact(EXCHANGE)
        self.assertIn("subroutineensure_field_y_buffers(nvar)", source)
        self.assertIn("subroutineensure_field_z_buffers(nvar)", source)
        self.assertGreaterEqual(source.count("capacity=max_field_vars"), 3)

    def test_benchmark_driver_separates_pipeline_and_dependency_sync(self) -> None:
        self.assertIn('"pinned-pipeline"', DRIVER)
        self.assertIn(
            'HALO_TRANSPORT=pinned-pipeline does not support SYNC_MODE=selective',
            DRIVER,
        )
        self.assertIn(
            'SYNC_MODE=dependency requires HALO_TRANSPORT=pinned-pipeline', DRIVER
        )

    def test_runtime_allows_explicit_or_dependency_pipeline_sync(self) -> None:
        source = compact(EXCHANGE)
        self.assertIn("if(gpu_selective_sync_enabled())then", source)
        self.assertIn(
            "astr_gpu_halo_transport=pinned-pipelinedoesnotsupportastr_gpu_sync_mode=selective",
            source,
        )
        self.assertIn("elseif(gpu_dependency_sync_enabled())then", source)

    def test_field_driver_forwards_pipeline_configuration(self) -> None:
        self.assertIn('SYNC_MODE="${SYNC_MODE:-explicit}"', FIELD_DRIVER)
        self.assertIn('HALO_TRANSPORT="${HALO_TRANSPORT:-pageable}"', FIELD_DRIVER)
        self.assertIn('ASTR_GPU_SYNC_MODE="$SYNC_MODE"', FIELD_DRIVER)
        self.assertIn('ASTR_GPU_HALO_TRANSPORT="$HALO_TRANSPORT"', FIELD_DRIVER)

    def test_performance_summary_accepts_dependency_mode(self) -> None:
        self.assertRegex(
            PERFORMANCE_SUMMARY,
            r"choices=\([^\n]*[\"']explicit[\"'][^\n]*[\"']selective[\"'][^\n]*[\"']dependency[\"']",
        )


if __name__ == "__main__":
    unittest.main()
