! The normal steploop owns repair, RK, output and restoration. Only the
! deliberately invalid initial node below differs from the ordinary program.
program crashfix_main_probe
  use astr_precision, only: real_kind,real_bytes
  use iso_fortran_env, only: int32,int64
  use parallel, only: mpiinitial,mpistop,mpisizedis,parapp,parallelini,mpirank
  use readwrite, only: readinput,fileini
  use commarray, only: allocommarray,q,crinod
  use commvar, only: use_gpu,im,jm,km,hm,prandtl,flowtype,ndims,lihomo,ljhomo,lkhomo
  use solver, only: refcal
  use initialisation, only: flowinit
  use gridgeneration, only: gridgen
  use geom, only: geomcal
  use comsolver, only: solvrinit
  use sponge_layer, only: spongelayerini
  use ibmethod, only: ibprocess
  use mainloop, only: steploop
  use bc, only: twall,bctype,configure_high_order_boundary,validate_high_order_wall_geometry
  use conservative_boundary_runtime, only: conservative_boundary,load_conservative_boundary_environment, &
    validate_conservative_sbli_mode,initialize_conservative_boundary_state
  use output_runtime, only: bootstrap_output_resources
  use benchmark_runtime, only: configure_benchmark_runtime
  use cmdefne, only: getcmd
#ifdef ASTR_GPU_CPP
  use iso_c_binding, only: c_size_t,c_sizeof
  use device_state, only: device_state_upload,device_state_download,device_q
  use fluid_runtime, only: cpp_after_flowinit,cpp_before_finalize,cpp_device_context
  use crashfix_operator_cpp, only: crashfix_history_download_cpp
#else
  use cudafor
  use commarray_gpu, only: q_d,crinod_d
  use gpu_runtime, only: gpu_pre_mpi_bind_device,gpu_bind_device,gpu_after_refcal,gpu_after_alloc, &
    gpu_after_flowinit,gpu_before_finalize
#endif
  implicit none
  integer :: status
  integer,parameter :: state_bits_kind=merge(int32,int64,real_bytes==4)
  character(16) :: command
#ifdef ASTR_GPU_CPP
  character(32) :: probe_mode
#endif
#ifndef ASTR_GPU_CPP
  call gpu_pre_mpi_bind_device()
#endif
  call mpiinitial
  call getcmd(command)
  if(command/='run') error stop 'crashfix steploop probe expects run'
  call readinput
  if(.not.use_gpu) error stop 'crashfix steploop probe requires the selected GPU backend'
  call mpisizedis;call parapp;call parallelini;call refcal
  call configure_high_order_boundary()
  call configure_benchmark_runtime(use_gpu,flowtype,ndims,lihomo,ljhomo,lkhomo,all(bctype==1))
  call load_conservative_boundary_environment()
  call validate_conservative_sbli_mode(twall(3))
#ifndef ASTR_GPU_CPP
  call gpu_bind_device();call gpu_after_refcal()
#endif
  call fileini;call allocommarray
#ifndef ASTR_GPU_CPP
  call gpu_after_alloc()
#endif
  call ibprocess;call bootstrap_output_resources();call gridgen;call solvrinit;call geomcal
  call validate_high_order_wall_geometry();call spongelayerini;call flowinit
  if(conservative_boundary%enabled) call initialize_conservative_boundary_state(twall(3))
#ifdef ASTR_GPU_CPP
  call cpp_after_flowinit()
  status=device_state_download(cpp_device_context(),device_q,q,size(q,kind=c_size_t)*c_sizeof(0._real_kind))
  if(status/=0) error stop 'initial crashfix test download'
  call get_environment_variable('ASTR_VALIDATION_CRASHFIX_HOT_CONTROL',probe_mode)
  if(probe_mode=='1') then
    call hot_control_probe()
    call cpp_before_finalize()
    call mpistop
  endif
#else
  call gpu_after_flowinit()
  q=q_d
#endif
  if(mpirank==0) q(im,jm/2,km/2,1)=-1.d0
#ifdef ASTR_GPU_CPP
  status=device_state_upload(cpp_device_context(),device_q,q,size(q,kind=c_size_t)*c_sizeof(0._real_kind))
  if(status/=0) error stop 'initial crashfix test upload'
#else
  q_d=q
#endif
  call steploop
  call dump_history()
#ifdef ASTR_GPU_CPP
  call cpp_before_finalize()
#else
  call gpu_before_finalize()
#endif
  call mpistop
contains
  subroutine dump_history()
    character(1024) :: prefix
    character(1200) :: path
    integer(1),allocatable :: markers(:,:,:)
    integer(state_bits_kind),allocatable :: before(:)
    integer :: err,unit
    call get_environment_variable('ASTR_VALIDATION_CRASHFIX_HISTORY_PREFIX',prefix,status=err)
    if(err==1) return
    if(err/=0.or.len_trim(prefix)==0) error stop 'crashfix history probe path'
    allocate(markers(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
#ifdef ASTR_GPU_CPP
    err=device_state_download(cpp_device_context(),device_q,q,size(q,kind=c_size_t)*c_sizeof(0._real_kind))
    if(err/=0) error stop 'history observer initial q download'
#else
    q=q_d
#endif
    before=transfer(q,[0_state_bits_kind],size(q))
#ifdef ASTR_GPU_CPP
    err=crashfix_history_download_cpp(cpp_device_context(),markers,size(markers,kind=c_size_t))
    if(err/=0) error stop 'crashfix history probe download'
#else
    crinod=crinod_d
    markers=int(merge(1,0,crinod),1)
#endif
#ifdef ASTR_GPU_CPP
    err=device_state_download(cpp_device_context(),device_q,q,size(q,kind=c_size_t)*c_sizeof(0._real_kind))
    if(err/=0) error stop 'history observer final q download'
#else
    q=q_d
#endif
    if(any(before/=transfer(q,[0_state_bits_kind],size(q)))) error stop 'history observation changed q'
    write(path,'(a,a,i8.8,a)') trim(prefix),'.rank',mpirank,'.bin'
    open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='new')
    write(unit) [im,jm,km,hm,real_bytes],markers
    close(unit)
    write(*,'(a,i0)') 'CRASHFIX_HISTORY_OBSERVER_Q_WORDS_PASS rank=',mpirank
  end subroutine
#ifdef ASTR_GPU_CPP
  subroutine hot_control_probe()
    use iso_c_binding, only: c_int8_t
    use iso_fortran_env, only: int64
    use mpi, only: MPI_COMM_WORLD,MPI_Barrier,MPI_Allreduce,MPI_INTEGER,MPI_SUM
    use commarray, only: crinod,rho,vel,prs,tmp
    use astr_precision, only: real_kind
    use crashfix_operator_cpp, only: crashfix_history_download_cpp
    use device_state, only: device_not_allocated
    use run_control_runtime, only: active_run,poll_run_control
    use readwrite, only: apply_run_input
    use output_runtime, only: validate_output_run_control
    use fluid_runtime, only: cpp_check_state,cpp_export_host_state
    use ieee_arithmetic, only: ieee_is_finite
    integer(c_int8_t),allocatable :: history(:,:,:),observed(:,:,:)
    integer(state_bits_kind),allocatable :: before(:)
    integer :: unit,ierr,step,local_marks,global_marks
    logical :: changed,ok,enabled
    character(512) :: message
    crinod=.false.
    if(active_run%crashfix) error stop 'hot-control must start with repair off'
    allocate(history(-hm:im+hm,-hm:jm+hm,-hm:km+hm),observed(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
    call cpp_check_state()
    status=crashfix_history_download_cpp(cpp_device_context(),history,size(history,kind=c_size_t))
    if(status/=device_not_allocated) error stop 'fresh disabled state allocated critical history'
    do step=1,3
      enabled=step/=2
      if(mpirank==0) then
        open(newunit=unit,file='datin/input.run',status='replace',action='write')
        write(unit,'(a)') '&run_control max_steps=4, reload_steps=1 /', &
          '&timestep mode="fixed", fixed_dt=1e-6 /', &
          '&logging list_steps=1 /','&statistics enabled=.false. /'
        if(enabled) then
          write(unit,'(a)') '&repair crashfix=.true. /'
        else
          write(unit,'(a)') '&repair crashfix=.false. /'
        endif
        close(unit)
      endif
      call MPI_Barrier(MPI_COMM_WORLD,ierr)
      call poll_run_control(int(step,int64),0._real_kind,MPI_COMM_WORLD,changed,ok,message,validate_output_run_control)
      if(.not.ok) then
        print *, trim(message)
        error stop 'hot-control candidate rejected'
      endif
      if(.not.changed.or.(active_run%crashfix.neqv.enabled)) error stop 'actual input polling did not switch repair'
      call apply_run_input()
      if(step==1) then
        if(mpirank==0) q(im,jm/2,km/2,1)=-1.d0
        status=device_state_upload(cpp_device_context(),device_q,q,size(q,kind=c_size_t)*c_sizeof(0._real_kind))
        if(status/=0) error stop 'hot-control invalid node upload'
      endif
      call cpp_check_state()
      call cpp_export_host_state(.true.)
      status=crashfix_history_download_cpp(cpp_device_context(),observed,size(observed,kind=c_size_t))
      if(status/=0) error stop 'hot-control history download'
      if(step==1) then
        local_marks=count(observed/=0)
        call MPI_Allreduce(local_marks,global_marks,1,MPI_INTEGER,MPI_SUM,MPI_COMM_WORLD,ierr)
        if(global_marks==0) error stop 'hot activation produced no repaired history'
        history=observed
        before=transfer(q,[0_state_bits_kind],size(q))
      else
        if(any(history/=observed).or.any(before/=transfer(q,[0_state_bits_kind],size(q)))) &
          error stop 'hot switch changed healthy q or discarded critical history'
      endif
      if(.not.all(ieee_is_finite(rho(0:im,0:jm,0:km))).or.any(rho(0:im,0:jm,0:km)<=0).or. &
        .not.all(ieee_is_finite(prs(0:im,0:jm,0:km))).or.any(prs(0:im,0:jm,0:km)<=0).or. &
        .not.all(ieee_is_finite(tmp(0:im,0:jm,0:km))).or.any(tmp(0:im,0:jm,0:km)<=0).or. &
        .not.all(ieee_is_finite(vel(0:im,0:jm,0:km,:)))) error stop 'hot-control physical caches'
      write(*,'(a,3(1x,i0),1x,l1)') 'CPP_CRASHFIX_HOT_CONTROL_PASS',mpirank,step,count(observed/=0),enabled
    enddo
    write(*,'(a,i0)') 'CPP_CRASHFIX_HOT_REAL_BYTES=',c_sizeof(0._real_kind)
  end subroutine
#endif
end program
