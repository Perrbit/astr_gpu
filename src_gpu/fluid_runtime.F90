module fluid_runtime
  use iso_c_binding, only: c_ptr,c_null_ptr,c_int,c_double,c_size_t,c_associated,c_sizeof
  use iso_fortran_env, only: int64
  use astr_precision, only: real_kind,clock_kind
  use mpi
  use commvar, only: im,jm,km,hm,ia,ja,ka,numq,num_species,num_modequ,use_gpu,flowtype,ndims, &
    lihomo,ljhomo,lkhomo,lreadgrid,nondimen,lcomb,lfilter,diffterm,conschm,difschm,rkscheme,turbmode, &
    const2,const5,const6,mach,gamma,rgas,reynolds,prandtl,tempconst,tempconst1,nstep,deltat,lcracon,lavg, &
    roinf,uinf,xmax,rkstep
  use commarray, only: q,jacob,dxi
  use parallel, only: mpirank,lio,ig0,jg0,kg0,isize,jsize,ksize, &
    mpileft,mpiright,mpidown,mpiup,mpiback,mpifront
  use device_state
  use halo_transport
  use fluid_explicit
  use fluid_step, only: fluid_prepare_stage,fluid_advance_prepared_stage
  implicit none
  private
  public :: cpp_after_flowinit,cpp_before_finalize,cpp_time_integration,cpp_check_state
  public :: collect_timestep_rates_cpp,collect_cfl_cpp
  type(c_ptr),save :: context=c_null_ptr
#ifdef ASTR_BUILD_TESTING
  integer(c_int),save :: exchange_sequence=0
#endif
contains
  subroutine require(condition,message)
    logical,intent(in) :: condition
    character(*),intent(in) :: message
    integer :: ignored
    if(condition) return
    write(*,'(a,i0,2a)') 'ASTR_GPU_CPP rank=',mpirank,': ',message
    call MPI_Abort(MPI_COMM_WORLD,1,ignored)
  end subroutine
  subroutine check(status,label)
    integer,intent(in) :: status
    character(*),intent(in) :: label
    if(status/=0) write(*,'(a,i0,a,i0)') 'ASTR_GPU_CPP rank=',mpirank,' status=',status
    call require(status==0,label)
  end subroutine
  subroutine cpp_after_flowinit()
    use bc, only: bctype
    use numerical_forms, only: convection_form,viscous_form,filter_scheme
    use case_model_options, only: get_case_option
    integer :: node,local_rank,local_size,ierr,choice,low,high,env_status
    integer(c_int) :: visible
    character(32) :: sync,value
    call require(trim(flowtype)=='tgv'.and.ndims==3.and.all(bctype==1).and. &
      lihomo.and.ljhomo.and.lkhomo.and..not.lreadgrid.and.nondimen.and. &
      numq==5.and.num_species==0.and.num_modequ==0.and..not.lcomb.and.trim(turbmode)=='none', &
      'implemented C++ flow path is FP64 Cartesian periodic five-variable TGV')
    call require(conschm=='643e'.and.difschm=='643e'.and.rkscheme=='rk3'.and. &
      convection_form=='standard'.and.viscous_form=='standard'.and.filter_scheme=='explicit10', &
      'C++ flow operators require standard 643e/643e, explicit10 and RK3')
    call require(.not.lcracon.and..not.lavg,'crashfix and mean-field statistics are not connected to the C++ state')
    call get_case_option('gpu_precision_mode',value,status=env_status)
    call require(env_status==1.or.(env_status==0.and.value=='fp64'),'C++ numerical fields currently require fp64')
    if(lfilter) then
      value=''
      call get_environment_variable('ASTR_GPU_FILTER_WORKSPACE',value,status=env_status)
      if(env_status==1) value='full'
      call require((env_status==0.or.env_status==1).and.value=='full', &
        'C++ filter implements the full workspace; scalar workspace migration is pending')
      if(lio) write(*,'(a)') 'ASTR_GPU_CPP filter_workspace=full'
    endif
    if(any([isize,jsize,ksize]>1)) then
      value=''
      call get_environment_variable('ASTR_GPU_HALO_TRANSPORT',value,status=env_status)
      if(env_status==1) value='pinned'
      call require((env_status==0.or.env_status==1).and.value=='pinned', &
        'C++ transport implements pinned host staging; GPU-aware transport is not connected')
    endif
    call MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,node,ierr)
    call check(ierr,'node communicator')
    call MPI_Comm_rank(node,local_rank,ierr)
    call check(ierr,'local rank')
    call MPI_Comm_size(node,local_size,ierr)
    call check(ierr,'local size')
    call check(device_count(visible),'visible devices')
    call require(local_size<=visible,'one visible device per local rank is required')
    call check(device_bind(int(local_rank,c_int)),'bind device')
    call MPI_Comm_free(node,ierr)
    call check(ierr,'release node communicator')
    sync=''
    call get_environment_variable('ASTR_GPU_SYNC_MODE',sync,status=env_status)
    if(env_status==1) sync='dependency'
    choice=-1
    if(sync=='dependency') choice=0
    if(sync=='explicit') choice=1
    call require((env_status==0.or.env_status==1).and.choice>=0,'ASTR_GPU_SYNC_MODE must be dependency or explicit')
    call MPI_Allreduce(choice,low,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
    call check(ierr,'sync configuration minimum')
    call MPI_Allreduce(choice,high,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    call check(ierr,'sync configuration maximum')
    call require(low==high,'sync mode differs across ranks')
    call check(device_state_create(int(local_rank,c_int),int(im,c_int),int(jm,c_int),int(km,c_int), &
      int(hm,c_int),int(choice,c_int),context),'create state')
    ! TGV scalar enstrophy/dissipation consume gradients even for inviscid runs.
    call check(fluid_prepare(context,1_c_int),'prepare primitive and gradient fields')
    call check(fluid_rhs_prepare(context,int(merge(1,0,diffterm),c_int)),'prepare RHS fields')
    if(lfilter) call check(fluid_filter_prepare(context),'prepare filter workspace')
    call check(device_transport_prepare(context,int(MPI_COMM_WORLD,c_int), &
      int([mpileft,mpidown,mpiback],c_int),int([mpiright,mpiup,mpifront],c_int), &
      int([isize,jsize,ksize],c_int),int(merge(6,5,diffterm),c_int)),'prepare pinned halo transport')
    call check(device_state_upload(context,device_q,q,bytes(size(q,kind=c_size_t))),'initial q upload')
    call check(device_state_upload(context,device_metric,dxi,bytes(size(dxi,kind=c_size_t))),'metric upload')
    call check(device_state_upload(context,device_jacobian,jacob,bytes(size(jacob,kind=c_size_t))),'Jacobian upload')
    call check(fluid_primitives(context,const2,const6),'initial primitive reconstruction')
    call cpp_check_state()
    write(*,'(a,i0,2a,a,i0,2a)') 'ASTR_GPU_CPP rank=',mpirank,' backend=',ASTR_GPU_CPP_NAME, &
      ' device=',local_rank,' sync=',trim(sync)
    write(*,'(a,i0,3(a,3(i0,1x)),a,i0,3(a,3(i0,1x)))') 'ASTR_GPU_CPP_LAYOUT rank=',mpirank, &
      ' global=',ia,ja,ka,' origin=',ig0,jg0,kg0,' upper=',im,jm,km,' halo=',hm, &
      ' topology=',isize,jsize,ksize,' lower=',mpileft,mpidown,mpiback,' higher=',mpiright,mpiup,mpifront
    if(lio) write(*,'(a)') 'ASTR_GPU_CPP transport=pinned; shared reconciliation then halo-only refresh'
  end subroutine
  integer(c_size_t) function bytes(count)
    integer(c_size_t),intent(in) :: count
    bytes=count*c_sizeof(0.d0)
  end function
  subroutine cpp_check_state()
    call check(fluid_check(context),'finite positive physical state')
  end subroutine
  subroutine collect_timestep_rates_cpp(values,valid)
    real(real_kind),intent(out) :: values(3)
    logical,intent(out) :: valid
    integer :: status
    status=fluid_cfl(context,int(merge(1,0,nondimen),c_int),mach,gamma,rgas,values)
    valid=status==0
    if(.not.valid) write(*,'(a,i0,a,i0)') 'ASTR_GPU_CPP wave collector rank=',mpirank,' status=',status
  end subroutine
  subroutine collect_cfl_cpp(values,keys,yvalues,by_plane,ny)
    integer,intent(in) :: ny
    real(real_kind),intent(out) :: values(4),yvalues(4,0:ny-1)
    integer(int64),intent(out) :: keys(4)
    logical,intent(in) :: by_plane
    yvalues=0
    call check(fluid_cfl_diagnostics(context,int(merge(1,0,nondimen),c_int),mach,gamma,rgas, &
      int([ia,ja,ka],c_int),int([ig0,jg0,kg0],c_int),int(merge(ny,0,by_plane),c_int), &
      values,keys,yvalues),'CFL diagnostic')
  end subroutine
  subroutine cpp_time_integration(statistics_started)
    real(clock_kind),intent(in) :: statistics_started
    do rkstep=1,3
#ifdef ASTR_BUILD_TESTING
      exchange_sequence=0
#endif
      call check(fluid_prepare_stage(context,const2,const6,lfilter,cpp_exchange_axis),'prepare RK stage')
#ifdef ASTR_BUILD_TESTING
      call snapshot_prepared_stage()
#endif
      if(rkstep==1.and.nstep>0) call publish_tgv_statistics(statistics_started)
      call check(fluid_advance_prepared_stage(context,rkstep,deltat,const2,const6,reynolds,prandtl, &
        const5,tempconst,tempconst1,diffterm,cpp_exchange_axis),'advance RK stage')
    enddo
  end subroutine
  integer(c_int) function cpp_exchange_axis(handle,field,axis,reconcile) bind(C,name='astr_cpp_exchange_axis') result(status)
    type(c_ptr),value :: handle
    integer(c_int),value :: field,axis,reconcile
    status=device_exchange_axis(handle,field,axis,reconcile)
#ifdef ASTR_BUILD_TESTING
    if(status==0) call snapshot_exchange(handle,field,axis,reconcile)
#endif
  end function
#ifdef ASTR_BUILD_TESTING
  subroutine snapshot_exchange(handle,field,axis,reconcile)
    use validation_io, only: rhs_validation_requested
    type(c_ptr),intent(in) :: handle
    integer(c_int),intent(in) :: field,axis,reconcile
    type(device_view) :: view
    real(c_double),allocatable :: data(:)
    character(1024) :: prefix
    character(1400) :: path
    integer :: unit,ignored
    integer(c_int) :: header(9)
    if(.not.rhs_validation_requested()) return
    exchange_sequence=exchange_sequence+1
    call check(device_state_view(handle,field,view),'exchange observation view')
    allocate(data(product(int(view%extent,c_size_t))))
    call check(device_state_snapshot_axis(handle,field,axis,data,bytes(size(data,kind=c_size_t))), &
      'exchange observation readback')
    call get_environment_variable('ASTR_VALIDATION_RHS_PREFIX',prefix,status=ignored)
    write(path,'(a,".cpp_halo.step",i8.8,".rk",i2.2,".exchange",i2.2,".rank",i8.8,".bin")') &
      trim(prefix),nstep,rkstep,exchange_sequence,mpirank
    header=int([im,jm,km,hm,view%extent(4)*view%extent(5),field,axis,reconcile,exchange_sequence],c_int)
    open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='new',action='write')
    write(unit) header
    write(unit) data
    close(unit)
    deallocate(data)
  end subroutine
  subroutine snapshot_prepared_stage()
    use validation_io, only: rhs_validation_requested,write_q_validation_snapshot,write_primitive_validation_snapshot
    use commarray, only: rho,vel,prs,tmp
    if(.not.rhs_validation_requested()) return
    call check(device_state_download(context,device_q,q,bytes(size(q,kind=c_size_t))),'stage validation q')
    call check(device_state_download(context,device_rho,rho,bytes(size(rho,kind=c_size_t))),'stage validation density')
    call check(device_state_download(context,device_velocity,vel,bytes(size(vel,kind=c_size_t))),'stage validation velocity')
    call check(device_state_download(context,device_pressure,prs,bytes(size(prs,kind=c_size_t))),'stage validation pressure')
    call check(device_state_download(context,device_temperature,tmp,bytes(size(tmp,kind=c_size_t))),'stage validation T')
    call write_q_validation_snapshot('pre_rhs',nstep,rkstep)
    call write_primitive_validation_snapshot('pre_rhs_primitives',nstep,rkstep)
  end subroutine
#endif
  subroutine publish_tgv_statistics(started)
    use statistic, only: kenergy,enstophy,dissipation,max_q,min_q,statout
    real(clock_kind),intent(in) :: started
    real(c_double) :: local(13),global(13),nodes,length
    integer :: ierr
    call check(fluid_gradient(context),'statistics gradient')
    call check(fluid_statistics(context,reynolds,tempconst,tempconst1,local),'TGV scalar statistics')
    call MPI_Allreduce(local(1:3),global(1:3),3,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr)
    call check(ierr,'statistics sums')
    call MPI_Allreduce(local(4:8),global(4:8),5,MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,ierr)
    call check(ierr,'statistics maxima')
    call MPI_Allreduce(local(9:13),global(9:13),5,MPI_DOUBLE_PRECISION,MPI_MIN,MPI_COMM_WORLD,ierr)
    call check(ierr,'statistics minima')
    nodes=real(ia,c_double)*real(ja,c_double)*real(ka,c_double)
    length=xmax/(2.d0*acos(-1.d0))
    kenergy=.5d0*global(1)/nodes/(roinf*uinf*uinf)
    enstophy=.5d0*global(2)/nodes/(roinf*(uinf/length)**2)
    dissipation=global(3)/nodes
    if(.not.allocated(max_q)) allocate(max_q(5),min_q(5))
    max_q=global(4:8);min_q=global(9:13)
    call statout(started)
  end subroutine
  subroutine cpp_before_finalize()
#ifdef ASTR_BUILD_TESTING
    use validation_io, only: rhs_validation_requested,write_q_validation_snapshot
    if(rhs_validation_requested()) then
      call check(device_state_download(context,device_q,q,bytes(size(q,kind=c_size_t))),'terminal validation q')
      call write_q_validation_snapshot('cpp_final',nstep,1)
    endif
#endif
    call require(c_associated(context),'missing C++ device context at finalize')
    call check(device_state_destroy(context),'release device state')
  end subroutine
end module
