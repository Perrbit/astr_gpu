!=======================================================================
! DNS Solver for Compressible Flow - ASTR
!=======================================================================
! Re-mastered on: 2021-02-06
! Author        : Fang Jian
! Contact       : fangjian19@gmail.com
!=======================================================================
program astr

  !---------------------------------------------------------------------
  ! Module usage
  !---------------------------------------------------------------------
  use parallel,      only: mpiinitial,mpistop,mpisizedis,lio,          &
                           parallelini,parapp
  use readwrite,     only: statement,readinput,readcont,fileini,infodisp
  use commarray,     only: allocommarray
  use commvar,       only: use_gpu,prandtl,flowtype,ndims,lihomo,     &
                           ljhomo,lkhomo,nstep,time
  use benchmark_runtime, only: configure_benchmark_runtime,configure_insitu_timing, &
                               insitu_clock,report_insitu_timing
  use insitu_runtime, only: insitu_check
  use solver,        only: refcal
  use initialisation,only: flowinit
  use sponge_layer,  only: spongelayerini
  use mainloop,      only: steploop
  use output_runtime,only: bootstrap_output_resources,prepare_preview_paths, &
                           configure_preview_runtime,restore_preview_runtime
  use insitu_session,only: render_preview_insitu
  use gridgeneration,only: gridgen
  use cmdefne,       only: getcmd,listcmd
  use pp,            only: ppentrance
  use geom,          only: geomcal
  use ibmethod,      only: ibprocess
  use test,          only: codetest
  use comsolver,     only: solvrinit
  use bc,            only: twall,bctype,configure_high_order_boundary
  use conservative_boundary_runtime, only: conservative_boundary, &
                           load_conservative_boundary_environment, &
                           validate_conservative_sbli_mode,        &
                           initialize_conservative_boundary_state
#ifdef _CUDA
  use gpu_runtime,   only: gpu_pre_mpi_bind_device,gpu_bind_device, &
                           gpu_after_refcal,gpu_after_alloc, &
                           gpu_after_flowinit,gpu_before_finalize
#endif

  implicit none

  !---------------------------------------------------------------------
  ! Local variables
  !---------------------------------------------------------------------
  character(len=16) :: cmd
  logical :: preview_mode
  real(8) :: initialization_started,solver_started

  !---------------------------------------------------------------------
  ! MPI Initialization and Command Processing
  !---------------------------------------------------------------------
#ifdef _CUDA
  call gpu_pre_mpi_bind_device()
#endif
  call mpiinitial
  call configure_insitu_timing()
  solver_started=insitu_clock()
  initialization_started=solver_started

  call statement

  call listcmd

  call getcmd(cmd)

  !---------------------------------------------------------------------
  ! Select operation based on command
  !---------------------------------------------------------------------
  if (trim(cmd) == 'insitu-check') then
    call insitu_check()
    call mpistop
  elseif (trim(cmd) == 'pp') then

    ! Pre/Post-processing
    call ppentrance

  elseif (trim(cmd) == 'test') then

    ! code tests
    call codetest

  elseif (trim(cmd) == 'run'.or.trim(cmd)=='insitu-preview') then
    preview_mode=trim(cmd)=='insitu-preview'

    ! Main simulation run
    call readinput
    if(preview_mode) call prepare_preview_paths()

    call mpisizedis

    call parapp

    call parallelini

    call refcal
    call configure_high_order_boundary()

    call configure_benchmark_runtime(use_gpu,flowtype,ndims,lihomo,ljhomo,lkhomo, &
                                     all(bctype(1:6)==1))

    call load_conservative_boundary_environment()
    call validate_conservative_sbli_mode(twall(3))
    if(conservative_boundary%enabled .and. lio) then
      write(*,'(A,F5.2)') ' ASTR_CONSERVATIVE_SBLI_MODE enabled Pr=',prandtl
    endif
#ifdef _CUDA
    if(use_gpu) call gpu_bind_device()
    if(use_gpu) call gpu_after_refcal()
#endif

    if(preview_mode) then
      call readcont
      call configure_preview_runtime()
    endif
    if(.not.preview_mode) call fileini

    call infodisp

    call allocommarray
#ifdef _CUDA
    if(use_gpu) call gpu_after_alloc()
#endif

    if(preview_mode) then
      call restore_preview_runtime()
      call render_preview_insitu(nstep,time)
    else
    call ibprocess

    call bootstrap_output_resources()
    call gridgen

    call solvrinit

    call geomcal

    call spongelayerini

    call flowinit

    if(conservative_boundary%enabled) &
      call initialize_conservative_boundary_state(twall(3))
#ifdef _CUDA
    if(use_gpu) call gpu_after_flowinit()
#endif

    call report_insitu_timing('solver_initialization',initialization_started,-1)
    call steploop
    endif
#ifdef _CUDA
    if(use_gpu) call gpu_before_finalize()
#endif

    call report_insitu_timing('solver_total_after_mpi',solver_started,-1)
    call mpistop

  else

    if (lio) print *, ' ** All jobs (nothing) done. **'
    stop

  end if

end program astr
!=======================================================================
! End of program ASTR
!=======================================================================
