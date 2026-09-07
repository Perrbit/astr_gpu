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
  use readwrite,     only: statement,readinput,fileini,infodisp
  use commarray,     only: allocommarray
  use commvar,       only: use_gpu,prandtl
  use solver,        only: refcal
  use initialisation,only: flowinit
  use sponge_layer,  only: spongelayerini
  use mainloop,      only: steploop
  use gridgeneration,only: gridgen
  use cmdefne,       only: getcmd,listcmd
  use pp,            only: ppentrance
  use geom,          only: geomcal
  use ibmethod,      only: ibprocess
  use test,          only: codetest
  use comsolver,     only: solvrinit
  use bc,            only: twall
  use conservative_boundary_runtime, only: conservative_boundary, &
                           load_conservative_boundary_environment, &
                           validate_conservative_sbli_mode,        &
                           initialize_conservative_boundary_state
#ifdef _CUDA
  use gpu_runtime,   only: gpu_bind_device,gpu_after_refcal,gpu_after_alloc, &
                           gpu_after_flowinit,gpu_before_finalize
#endif

  implicit none

  !---------------------------------------------------------------------
  ! Local variables
  !---------------------------------------------------------------------
  character(len=16) :: cmd

  !---------------------------------------------------------------------
  ! MPI Initialization and Command Processing
  !---------------------------------------------------------------------
  call mpiinitial

  call statement

  call listcmd

  call getcmd(cmd)

  !---------------------------------------------------------------------
  ! Select operation based on command
  !---------------------------------------------------------------------
  if (trim(cmd) == 'pp') then

    ! Pre/Post-processing
    call ppentrance

  elseif (trim(cmd) == 'test') then

    ! code tests
    call codetest

  elseif (trim(cmd) == 'run') then

    ! Main simulation run
    call readinput

    call mpisizedis

    call parapp

    call parallelini

    call refcal

    call load_conservative_boundary_environment()
    call validate_conservative_sbli_mode(twall(3))
    if(conservative_boundary%enabled .and. lio) then
      write(*,'(A,F5.2)') ' ASTR_CONSERVATIVE_SBLI_MODE enabled Pr=',prandtl
    endif
#ifdef _CUDA
    if(use_gpu) call gpu_bind_device()
    if(use_gpu) call gpu_after_refcal()
#endif

    call fileini

    call infodisp

    call allocommarray
#ifdef _CUDA
    if(use_gpu) call gpu_after_alloc()
#endif

    call ibprocess

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

    call steploop
#ifdef _CUDA
    if(use_gpu) call gpu_before_finalize()
#endif

    call mpistop

  else

    if (lio) print *, ' ** All jobs (nothing) done. **'
    stop

  end if

end program astr
!=======================================================================
! End of program ASTR
!=======================================================================
