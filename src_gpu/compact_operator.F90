module compact_flow_cpp
  use iso_c_binding, only: c_ptr,c_int,c_size_t,c_associated
  use astr_precision, only: real_kind
  use mpi
  use commvar, only: im,jm,km,conschm,difschm,lfilter,alfa_filter,gamma
  use compact_runtime, only: configure_global_compact,compact_partitions,compact_line_comms
  use compact_operator, only: compact_derivative_core,compact_filter_core
  use compact_line_solver, only: compact_matrix
  use compact_line_cpp
  implicit none
  private
  public :: prepare_compact_flow_cpp,release_compact_flow_cpp
  public :: compact_gradient_cpp,compact_convection_cpp,compact_viscous_cpp,compact_filter_cpp
  public :: compact_keep_convection_cpp
  public :: cpp_compact_derivative,cpp_compact_filter
  logical,save :: cpp_compact_derivative=.false.,cpp_compact_filter=.false.
  integer,save :: filter_components=1
  integer,save :: keep_bases=0
  type(compact_distributed_cpp_plan),save :: derivatives(3),filters(3)
  type(compact_stencil_cpp),save :: stencils(3)
  interface
    integer(c_int) function clear_rhs(parent) bind(C,name='astr_compact_rhs_clear')
      import c_ptr,c_int
      type(c_ptr),value :: parent
    end function
    integer(c_int) function configure_sync(parent,explicit_sync) bind(C,name='astr_compact_configure_sync')
      import c_ptr,c_int
      type(c_ptr),value :: parent
      integer(c_int),value :: explicit_sync
    end function
  end interface
contains
  subroutine prepare_compact_flow_cpp(parent,components)
    use commvar, only: is,ie,js,je,ks,ke
    use bc, only: bctype
    use numerical_forms, only: compact_filter_selected,convection_form
    use keep_operator, only: keep_split_plan,build_keep_split_plan
    use filter, only: coef6i,coef8i,coef10i,coefb
    use parallel, only: lio
    type(c_ptr),intent(in) :: parent
    integer,intent(in) :: components
    type(compact_matrix) :: matrix
    type(keep_split_plan) :: split
    real(real_kind) :: edge(2)
    integer :: axis,lines,dims(3),batch,choice(3),lo(3),hi(3),ierr,status,ios
    character(32) :: value
    cpp_compact_derivative=conschm=='643c'.or.difschm=='643c'
    cpp_compact_filter=lfilter.and.compact_filter_selected(conschm,difschm)
    if(.not.cpp_compact_derivative.and..not.cpp_compact_filter) return
    call configure_global_compact(bctype)
    choice=[0,65535,1]
    call get_environment_variable('ASTR_GPU_COMPACT_TRANSPORT',value,status=status)
    if(status==0) then
      choice(1)=-1
      if(value=='pinned') choice(1)=0
      if(value=='device-aware') choice(1)=1
    elseif(status/=1) then
      choice(1)=-1
    endif
    call get_environment_variable('ASTR_GPU_COMPACT_BATCH_RHS',value,status=status)
    if(status==0) then
      read(value,*,iostat=ios) choice(2)
      if(ios/=0) choice(2)=-1
    elseif(status/=1) then
      choice(2)=-1
    endif
    call get_environment_variable('ASTR_GPU_COMPACT_SYNC',value,status=status)
    if(status==0) then
      choice(3)=-1
      if(value=='explicit') choice(3)=1
      if(value=='dependency') choice(3)=0
    elseif(status/=1) then
      choice(3)=-1
    endif
    call MPI_Allreduce(choice,lo,3,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
    call MPI_Allreduce(choice,hi,3,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(any(lo/=hi).or.any(lo<0).or.lo(2)<1.or.lo(2)>65535) &
      error stop 'C++ compact execution requires consistent transport, sync and batch RHS in 1:65535'
    status=configure_sync(parent,int(choice(3),c_int))
    if(status/=0) error stop 'C++ compact synchronization configuration'
    filter_components=components
    dims=[im,jm,km]
    do axis=1,3
      lines=product(dims+1)/(dims(axis)+1)
      stencils(axis)%axis=axis-1
      stencils(axis)%first=compact_partitions(axis)%first
      stencils(axis)%total=compact_partitions(axis)%intervals
      stencils(axis)%periodic=merge(1,0,compact_partitions(axis)%periodic)
      stencils(axis)%active_first=[is,js,ks];stencils(axis)%active_last=[ie,je,ke]
      if(cpp_compact_derivative) then
        call compact_derivative_core(compact_partitions(axis),matrix,edge)
        batch=min(5*lines,choice(2))
        call prepare_distributed_compact_cpp(parent,matrix,edge,compact_partitions(axis), &
          compact_line_comms(axis),batch,int(choice(1),c_int),derivatives(axis))
      endif
      if(cpp_compact_filter) then
        call compact_filter_core(compact_partitions(axis),alfa_filter,matrix,edge)
        batch=min(filter_components*lines,choice(2))
        call prepare_distributed_compact_cpp(parent,matrix,edge,compact_partitions(axis), &
          compact_line_comms(axis),batch,int(choice(1),c_int),filters(axis))
        stencils(axis)%c6=coef6i;stencils(axis)%c8=coef8i;stencils(axis)%c10=coef10i
        stencils(axis)%boundary=coefb(0:2,0:6)
      endif
    enddo
    if(convection_form=='keep'.and.conschm=='643c') then
      call build_keep_split_plan(split)
      status=compact_keep_prepare_cpp(parent,int(split%terms,c_int),int(split%bases,c_int),split%left,split%right, &
        split%component,split%derivative,split%weight)
      if(status/=0) error stop 'C++ KEEP constant split table preparation'
      keep_bases=split%bases
    endif
    if(lio) write(*,'(a,3(i0,1x))') 'ASTR_CPP_COMPACT transport/batch/explicit=',choice
  end subroutine

  integer(c_int) function apply(parent,plan,axis,kind,components,offset,wall_band) result(status)
    type(c_ptr),intent(in) :: parent
    type(compact_distributed_cpp_plan),intent(inout) :: plan
    integer,intent(in) :: axis,kind,components,offset
    logical,optional,intent(in) :: wall_band
    type(compact_stencil_cpp) :: stencil
    integer :: lines,start,nrhs,dims(3)
    logical :: bounded
    bounded=.false.
    if(present(wall_band)) bounded=wall_band
    dims=[im,jm,km];lines=product(dims+1)/(dims(axis)+1)
    stencil=stencils(axis);stencil%kind=kind;stencil%component_offset=offset
    status=0
    do start=0,lines*components-1,plan%capacity
      nrhs=min(plan%capacity,lines*components-start);stencil%start=start
      status=compact_field_rhs_cpp(parent,plan%handle,stencil,int(nrhs,c_int))
      if(status/=0) return
      call solve_distributed_compact_cpp(plan,nrhs)
      call restore_distributed_compact_cpp(plan,nrhs)
      if(bounded) then
        status=compact_field_commit_wall_band_cpp(parent,plan%handle,stencil,int(nrhs,c_int))
      else
        status=compact_field_commit_cpp(parent,plan%handle,stencil,int(nrhs,c_int))
      endif
      if(status/=0) return
    enddo
  end function

  integer(c_int) function compact_gradient_cpp(parent) result(status)
    type(c_ptr),value :: parent
    integer :: axis
    do axis=1,3
      status=apply(parent,derivatives(axis),axis,1,4,1)
      if(status/=0) return
    enddo
  end function

  integer(c_int) function compact_convection_cpp(parent,retain_rhs) result(status)
    type(c_ptr),value :: parent
    logical,optional,intent(in) :: retain_rhs
    integer :: axis
    status=0
    if(present(retain_rhs)) then
      if(.not.retain_rhs) status=clear_rhs(parent)
    else
      status=clear_rhs(parent)
    endif
    if(status/=0) return
    do axis=1,3
      status=apply(parent,derivatives(axis),axis,2,5,1)
      if(status/=0) return
    enddo
  end function

  integer(c_int) function compact_viscous_cpp(parent,wall_band) result(status)
    type(c_ptr),value :: parent
    logical,optional,intent(in) :: wall_band
    integer :: axis
    do axis=1,3
      status=apply(parent,derivatives(axis),axis,3,4,1,wall_band)
      if(status/=0) return
    enddo
  end function

  integer(c_int) function compact_keep_convection_cpp(parent,retain_rhs) result(status)
    type(c_ptr),value :: parent
    logical,optional,intent(in) :: retain_rhs
    type(compact_stencil_cpp) :: stencil
    integer :: axis,lines,start,nrhs,dims(3)
    status=0
    if(present(retain_rhs)) then
      if(.not.retain_rhs) status=clear_rhs(parent)
    else
      status=clear_rhs(parent)
    endif
    if(status/=0) return
    dims=[im,jm,km]
    do axis=1,3
      lines=product(dims+1)/(dims(axis)+1);stencil=stencils(axis)
      stencil%kind=2;stencil%component_offset=1
      do start=0,lines*keep_bases-1,derivatives(axis)%capacity
        nrhs=min(derivatives(axis)%capacity,lines*keep_bases-start);stencil%start=start
        status=compact_keep_rhs_cpp(parent,derivatives(axis)%handle,stencil,int(nrhs,c_int),gamma)
        if(status/=0) return
        call solve_distributed_compact_cpp(derivatives(axis),nrhs)
        call restore_distributed_compact_cpp(derivatives(axis),nrhs)
        status=compact_keep_commit_cpp(parent,derivatives(axis)%handle,stencil,int(nrhs,c_int),gamma)
        if(status/=0) return
      enddo
    enddo
  end function

  integer(c_int) function compact_filter_cpp(parent) result(status)
    use device_state, only: device_q
    use halo_transport, only: device_exchange_axis
    type(c_ptr),value :: parent
    integer :: axis,offset
    do axis=1,3
      status=device_exchange_axis(parent,device_q,int(axis-1,c_int),0_c_int)
      if(status/=0) return
      do offset=1,5,filter_components
        status=apply(parent,filters(axis),axis,4,filter_components,offset)
        if(status/=0) return
      enddo
    enddo
  end function

  subroutine release_compact_flow_cpp()
    integer :: axis
    do axis=1,3
      if(c_associated(derivatives(axis)%handle)) call release_distributed_compact_cpp(derivatives(axis))
      if(c_associated(filters(axis)%handle)) call release_distributed_compact_cpp(filters(axis))
    enddo
    cpp_compact_derivative=.false.;cpp_compact_filter=.false.
    keep_bases=0
  end subroutine
end module
