module insitu_catalog_m12
  use mpi
  use iso_fortran_env, only: int64,real64
  use astr_precision, only: real_kind
  use commvar, only: use_gpu,flowtype,numq,im,ia
  use commarray, only: x
  use parallel, only: mpirank,ig0
  use insitu_run_config, only: insitu_options
  use insitu_session, only: catalog_product_options
  use insitu_product_schedule, only: runtime_product_state,runtime_catalog_count,runtime_catalog_state, &
    runtime_catalog_due,runtime_catalog_success,runtime_catalog_disable,runtime_catalog_reject,runtime_active
  use index_slice_archive, only: runtime_product_segment,refresh_runtime_product_status
  use checkpoint_state_io, only: checkpoint_state_identity
  use benchmark_runtime, only: benchmark_field_io_disabled
#ifdef _CUDA
  use insitu_m12_gpu, only: m12_diagnostic_layout_supported,m12_probe_count,capture_m12_probes_gpu,capture_m12_wall_gpu
#endif
  implicit none
  private
  public :: observe_catalog_m12_diagnostics,finish_catalog_m12_diagnostics
  type :: probe_stream
    integer :: unit=0
    logical :: opened=.false.
    character(4096) :: directory=''
  end type
  type(probe_stream),allocatable,save :: streams(:)
  type(checkpoint_state_identity),save :: last_identity
  logical,save :: observed=.false.
contains
  logical function agreed(ok) result(all_ok)
    logical,intent(in) :: ok
    integer :: bad,global_bad,ierr,ignored
    bad=merge(0,1,ok)
    call MPI_Allreduce(bad,global_bad,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,ignored)
    all_ok=global_bad==0
  end function

  subroutine observe_catalog_m12_diagnostics(identity)
    type(checkpoint_state_identity),intent(in) :: identity
    type(insitu_options) :: options
    type(runtime_product_state) :: state
    type(probe_stream),allocatable :: grown(:)
    character(:),allocatable :: directory
    character(4096) :: file
    character(512) :: message
    integer :: n,p,i,j,unit,status,closed
    logical :: enabled,ok,supported,dirty,requested
#ifdef _CUDA
    real(real_kind),allocatable :: probes(:,:),wall(:,:)
#endif
    if(benchmark_field_io_disabled()) return
    ! The original M12 providers consume completed RK states, not the cold initial field.
    if(identity%step==0) return
    n=runtime_catalog_count()
    if(n==0) return
    requested=.false.
    do p=1,n
      call runtime_catalog_state(p,state)
      requested=requested.or.state%definition%view%kind=='m12_probes'.or.state%definition%view%kind=='m12_wall'
    enddo
    if(.not.requested) return
    if(.not.allocated(streams)) allocate(streams(0))
    if(size(streams)<n) then
      allocate(grown(n));grown(:size(streams))=streams
      call move_alloc(grown,streams)
    endif
    dirty=.false.
    do p=1,n
      call runtime_catalog_state(p,state)
      if(state%definition%view%kind/='m12_probes'.and.state%definition%view%kind/='m12_wall') cycle
      if(state%status/=runtime_active) then
        if(streams(p)%opened) then
          close(streams(p)%unit,iostat=status);streams(p)%opened=.false.
        endif
        cycle
      endif
      call catalog_product_options(options,enabled)
      supported=.false.
#ifdef _CUDA
      supported=enabled.and.use_gpu.and.trim(flowtype)=='bl'.and.numq==5.and.m12_diagnostic_layout_supported()
#endif
      if(.not.agreed(supported)) then
        call runtime_catalog_reject(p,'original M12 diagnostics need the admitted GPU M12 layout')
        dirty=.true.;cycle
      endif
      if(.not.runtime_catalog_due(p,identity%step,real(identity%time,real64))) cycle
      call runtime_product_segment(p,state,directory,ok,message)
      if(.not.ok) then
        call runtime_catalog_disable(p,identity%step,real(identity%time,real64),trim(message))
        dirty=.true.;cycle
      endif
      status=0;closed=0;ok=.false.
#ifdef _CUDA
      if(state%definition%view%kind=='m12_probes') then
        if(.not.allocated(probes)) allocate(probes(6,m12_probe_count))
        call capture_m12_probes_gpu(probes,ok)
        ok=agreed(ok)
        if(ok.and.mpirank==0) then
          if(streams(p)%opened.and.trim(streams(p)%directory)/=directory) then
            close(streams(p)%unit,iostat=status);streams(p)%opened=.false.
          endif
          if(status==0.and..not.streams(p)%opened) then
            open(newunit=streams(p)%unit,file=directory//'/probes.csv',status='new',action='write',iostat=status)
            if(status==0) then
              streams(p)%opened=.true.;streams(p)%directory=directory
              dirty=.true.
              write(streams(p)%unit,'(a)',iostat=status) 'step,time,probe,u_Uinf,v_Uinf,w_Uinf,rho_rhoinf,p_pinf,T_Tinf'
            endif
          endif
          if(status==0) then
            do i=1,m12_probe_count
              write(streams(p)%unit,'(i0,a,es26.17e3,a,i0,6(a,es26.17e3))',iostat=status) &
                identity%step,',',identity%time,',',i,(',',probes(j,i),j=1,6)
              if(status/=0) exit
            enddo
            if(status==0) flush(streams(p)%unit,iostat=status)
          endif
        endif
      else
        if(.not.allocated(wall)) allocate(wall(3,im+1))
        call capture_m12_wall_gpu(wall,ok)
        ok=agreed(ok)
        if(ok) then
          write(file,'(a,"/wall.step",i0.12,".rank",i4.4,".csv")') directory,identity%step,mpirank
          open(newunit=unit,file=trim(file),status='new',action='write',iostat=status)
          if(status==0) then
            write(unit,'(a)',iostat=status) 'global_i,x,Cf,q_into_wall_rhoinf_Uinf3'
            do i=0,im
              if(status/=0) exit
              if(i==im.and.ig0+im<ia) cycle
              write(unit,'(i0,3(a,es26.17e3))',iostat=status) ig0+i,',',x(i,0,0,1),',',wall(2,i+1),',',wall(3,i+1)
            enddo
            close(unit,iostat=closed)
          endif
        endif
      endif
#endif
      ok=agreed(ok.and.status==0.and.closed==0)
      if(ok) then
        call runtime_catalog_success(p,identity%step,real(identity%time,real64))
        if(state%definition%view%kind=='m12_wall') dirty=.true.
        last_identity=identity;observed=.true.
      else
        call runtime_catalog_disable(p,identity%step,real(identity%time,real64),'original M12 diagnostic capture or CSV failed')
        dirty=.true.
      endif
    enddo
    ! Probe CSV is already flushed each step; publish status only on events.
    if(agreed(.not.dirty)) return
    call refresh_runtime_product_status(identity)
  end subroutine

  subroutine finish_catalog_m12_diagnostics()
    integer :: p,status
    if(.not.allocated(streams)) return
    do p=1,size(streams)
      if(.not.streams(p)%opened) cycle
      close(streams(p)%unit,iostat=status)
      if(status/=0) print '(a,i0)','ASTR_M12_PROBE_CSV_CLOSE_FAILED product=',p
    enddo
    deallocate(streams)
    if(observed) call refresh_runtime_product_status(last_identity)
  end subroutine
end module
