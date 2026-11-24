module compact_flow_cpp
  use iso_c_binding, only: c_ptr,c_double,c_int,c_int64_t,c_size_t,c_associated
  use astr_precision, only: real_kind
  use mpi
  use commvar, only: im,jm,km,conschm,difschm,lfilter,alfa_filter,gamma
  use compact_runtime, only: configure_global_compact,compact_partitions,compact_line_comms, &
    wall5_filter_plan,make_wall5_filter_plan,wall5_filter_layout
  use compact_operator, only: compact_derivative_core,compact_filter_core
  use compact_line_solver, only: compact_matrix
  use numerical_forms, only: compact_filter_axis_selected,filter_scope
  use compact_line_cpp
  implicit none
  private
  public :: prepare_compact_flow_cpp,release_compact_flow_cpp
  public :: compact_gradient_cpp,compact_convection_cpp,compact_viscous_cpp,compact_filter_cpp
  public :: compact_keep_convection_cpp
  public :: report_compact_phases_cpp
  public :: cpp_compact_derivative,cpp_compact_filter
  logical,save :: cpp_compact_derivative=.false.,cpp_compact_filter=.false.
  integer,save :: filter_components=1
  integer,save :: keep_bases=0
  logical,save :: phase_timing=.false.
  type(compact_distributed_cpp_plan),save :: derivatives(2,3),filters(2,3)
  type(compact_stencil_cpp),save :: stencils(3)
  type(wall5_filter_plan),save :: wall5_plan
  integer,save :: wall5_walls=0,wall5_peers=0,wall5_receivers=0
  interface
    integer(c_int) function filter_wall5(parent,plan,walls) bind(C,name='astr_compact_filter_wall5')
      import c_ptr,c_int,wall5_filter_plan
      type(c_ptr),value :: parent
      type(wall5_filter_plan),intent(in) :: plan
      integer(c_int),value :: walls
    end function
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
    use numerical_forms, only: compact_filter_selected,convection_form,viscous_form
    use keep_operator, only: keep_split_plan,build_keep_split_plan
    use filter, only: coef6i,coef8i,coef10i,coefb
    use parallel, only: lio
    type(c_ptr),intent(in) :: parent
    integer,intent(in) :: components
    type(compact_matrix) :: matrix
    type(keep_split_plan) :: split
    real(real_kind) :: edge(2)
    integer :: axis,lines,dims(3),batch,choice(5),lo(5),hi(5),ierr,status,ios
    character(32) :: value
    cpp_compact_derivative=conschm=='643c'.or.difschm=='643c'
    cpp_compact_filter=lfilter.and.compact_filter_selected(conschm,difschm)
    if(.not.cpp_compact_derivative.and..not.cpp_compact_filter) return
    call configure_global_compact(bctype)
    choice=[0,65535,1,0,0]
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
    call get_environment_variable('ASTR_GPU_COMPACT_PIPELINE',value,status=status)
    if(status==0) then
      choice(4)=-1
      if(value=='off') choice(4)=0
      if(value=='double') choice(4)=1
    elseif(status/=1) then
      choice(4)=-1
    endif
    call get_environment_variable('ASTR_GPU_COMPACT_PHASE_TIMING',value,status=status)
    if(status==0) then
      choice(5)=-1
      if(value=='0') choice(5)=0
      if(value=='1') choice(5)=1
    elseif(status/=1) then
      choice(5)=-1
    endif
    call MPI_Allreduce(choice,lo,5,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
    call MPI_Allreduce(choice,hi,5,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(lo(4)/=hi(4).or.lo(4)<0) error stop 'ASTR_GPU_COMPACT_PIPELINE requires consistent off or double'
    if(lo(5)/=hi(5).or.lo(5)<0) error stop 'ASTR_GPU_COMPACT_PHASE_TIMING requires consistent 0 or 1'
    if(any(lo/=hi).or.any(lo<0).or.lo(2)<1.or.lo(2)>65535) &
      error stop 'C++ compact execution requires consistent transport, sync and batch RHS in 1:65535'
    if(choice(4)==1.and.(choice(3)/=0.or..not.all(bctype==1).or.conschm/='643c'.or.difschm/='643c'.or. &
      convection_form/='standard'.or.viscous_form/='standard')) &
      error stop 'double compact pipeline requires periodic standard 643c/643c with dependency synchronization'
    status=configure_sync(parent,int(choice(3),c_int))
    if(status/=0) error stop 'C++ compact synchronization configuration'
    filter_components=components
    phase_timing=choice(5)==1
    if(cpp_compact_filter.and.filter_scope=='y-wall5') then
      call wall5_filter_layout(wall5_walls,wall5_peers,wall5_receivers)
      if(wall5_walls/=0) wall5_plan=make_wall5_filter_plan()
    endif
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
          compact_line_comms(axis),batch,int(choice(1),c_int),derivatives(1,axis))
        if(choice(4)==1.and.compact_partitions(axis)%ranks>1.and.batch<5*lines) &
          call prepare_distributed_compact_cpp(parent,matrix,edge,compact_partitions(axis), &
            compact_line_comms(axis),batch,int(choice(1),c_int),derivatives(2,axis))
      endif
      if(cpp_compact_filter.and.compact_filter_axis_selected(axis)) then
        call compact_filter_core(compact_partitions(axis),alfa_filter,matrix,edge)
        batch=min(filter_components*lines,choice(2))
        call prepare_distributed_compact_cpp(parent,matrix,edge,compact_partitions(axis), &
          compact_line_comms(axis),batch,int(choice(1),c_int),filters(1,axis))
        if(choice(4)==1.and.compact_partitions(axis)%ranks>1.and.batch<filter_components*lines) &
          call prepare_distributed_compact_cpp(parent,matrix,edge,compact_partitions(axis), &
            compact_line_comms(axis),batch,int(choice(1),c_int),filters(2,axis))
        stencils(axis)%c6=coef6i;stencils(axis)%c8=coef8i;stencils(axis)%c10=coef10i
        stencils(axis)%boundary=coefb(0:2,0:6)
      endif
      if(phase_timing) then
        call enable_axis_phases(derivatives(:,axis))
        call enable_axis_phases(filters(:,axis))
      endif
    enddo
    if(convection_form=='keep'.and.conschm=='643c') then
      call build_keep_split_plan(split)
      status=compact_keep_prepare_cpp(parent,int(split%terms,c_int),int(split%bases,c_int),split%left,split%right, &
        split%component,split%derivative,split%weight)
      if(status/=0) error stop 'C++ KEEP constant split table preparation'
      keep_bases=split%bases
    endif
    if(lio) then
      write(*,'(a,3(i0,1x))') 'ASTR_CPP_COMPACT transport/batch/explicit=',choice(1:3)
      write(*,'(a,i0,2(a,i0))') 'ASTR_CPP_COMPACT pipeline=',choice(4), &
        ' derivative_axes=',count([(c_associated(derivatives(2,axis)%handle),axis=1,3)]), &
        ' filter_axes=',count([(c_associated(filters(2,axis)%handle),axis=1,3)])
    endif
  end subroutine

  subroutine enable_axis_phases(plans)
    type(compact_distributed_cpp_plan),intent(in) :: plans(2)
    integer :: slot
    do slot=1,2
      if(c_associated(plans(slot)%handle)) call compact_distributed_phase_enable_cpp(plans(slot))
    enddo
  end subroutine

  subroutine report_compact_phases_cpp()
    use commvar, only: nstep
    use parallel, only: mpirank
    integer :: axis
    if(.not.phase_timing) return
    do axis=1,3
      if(cpp_compact_derivative) call report_axis_phases(derivatives(:,axis),axis,'derivative')
      if(cpp_compact_filter.and.compact_filter_axis_selected(axis)) call report_axis_phases(filters(:,axis),axis,'filter')
    enddo
  contains
    subroutine report_axis_phases(plans,axis,op)
      type(compact_distributed_cpp_plan),intent(in) :: plans(2)
      integer,intent(in) :: axis
      character(*),intent(in) :: op
      real(c_double) :: times(5),total(5)
      integer(c_int64_t) :: bytes(2),total_bytes(2)
      integer :: slot
      total=0
      total_bytes=0
      do slot=1,2
        if(.not.c_associated(plans(slot)%handle)) cycle
        call compact_distributed_phase_seconds_cpp(plans(slot)%handle,times)
        call compact_distributed_metrics_cpp(plans(slot),bytes,.true.)
        total=total+times
        total_bytes=total_bytes+bytes
      enddo
      write(*,'(a,3(1x,i0),1x,a,5(1x,es24.16e3),2(1x,i0))') &
        'ASTR_CPP_COMPACT_PHASE_TIMING',mpirank,nstep,axis,op,total,total_bytes
    end subroutine
  end subroutine

  integer(c_int) function apply(parent,plan,axis,kind,components,offset,wall_band) result(status)
    type(c_ptr),intent(in) :: parent
    type(compact_distributed_cpp_plan),intent(in) :: plan(2)
    integer,intent(in) :: axis,kind,components,offset
    logical,optional,intent(in) :: wall_band
    type(compact_stencil_cpp) :: stencil,upcoming
    integer :: lines,start,nrhs,dims(3),capacity,total,slot,next,next_start,next_rhs
    logical :: bounded,pipeline
    bounded=.false.
    if(present(wall_band)) bounded=wall_band
    dims=[im,jm,km];lines=product(dims+1)/(dims(axis)+1)
    stencil=stencils(axis);stencil%kind=kind;stencil%component_offset=offset
    capacity=plan(1)%capacity
    total=lines*components
    pipeline=c_associated(plan(2)%handle).and.total>capacity
    slot=1
    status=0
    if(pipeline) then
      stencil%start=0
      status=prepare_batch(parent,plan(slot),stencil,capacity)
      if(status/=0) return
    endif
    do start=0,total-1,capacity
      nrhs=min(capacity,total-start);stencil%start=start
      if(pipeline) then
        status=compact_distributed_begin_cpp(plan(slot)%handle,int(nrhs,c_int))
        if(status/=0) return
        next=3-slot;next_start=start+capacity
        if(next_start<total) then
          next_rhs=min(capacity,total-next_start)
          upcoming=stencil;upcoming%start=next_start
          ! Lines/components are disjoint within this direction, including
          ! in-place filtering. The other plan owns separate local scratch.
          status=prepare_batch(parent,plan(next),upcoming,next_rhs)
          if(status/=0) return
        endif
        status=compact_distributed_finish_cpp(plan(slot)%handle,int(nrhs,c_int))
        if(status/=0) return
      else
        status=compact_field_rhs_cpp(parent,plan(slot)%handle,stencil,int(nrhs,c_int))
        if(status/=0) return
        call solve_distributed_compact_cpp(plan(slot),nrhs)
      endif
      call restore_distributed_compact_cpp(plan(slot),nrhs)
      if(bounded) then
        status=compact_field_commit_wall_band_cpp(parent,plan(slot)%handle,stencil,int(nrhs,c_int))
      else
        status=compact_field_commit_cpp(parent,plan(slot)%handle,stencil,int(nrhs,c_int))
      endif
      if(status/=0) return
      if(pipeline) slot=next
    enddo
  end function

  integer(c_int) function prepare_batch(parent,plan,stencil,nrhs) result(status)
    type(c_ptr),intent(in) :: parent
    type(compact_distributed_cpp_plan),intent(in) :: plan
    type(compact_stencil_cpp),intent(in) :: stencil
    integer,intent(in) :: nrhs
    status=compact_field_rhs_cpp(parent,plan%handle,stencil,int(nrhs,c_int))
    if(status==0) status=compact_distributed_local_cpp(plan%handle,int(nrhs,c_int))
  end function

  integer(c_int) function compact_gradient_cpp(parent) result(status)
    type(c_ptr),value :: parent
    integer :: axis
    do axis=1,3
      status=apply(parent,derivatives(:,axis),axis,1,4,1)
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
      status=apply(parent,derivatives(:,axis),axis,2,5,1)
      if(status/=0) return
    enddo
  end function

  integer(c_int) function compact_viscous_cpp(parent,wall_band) result(status)
    type(c_ptr),value :: parent
    logical,optional,intent(in) :: wall_band
    integer :: axis
    do axis=1,3
      status=apply(parent,derivatives(:,axis),axis,3,4,1,wall_band)
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
      do start=0,lines*keep_bases-1,derivatives(1,axis)%capacity
        nrhs=min(derivatives(1,axis)%capacity,lines*keep_bases-start);stencil%start=start
        status=compact_keep_rhs_cpp(parent,derivatives(1,axis)%handle,stencil,int(nrhs,c_int),gamma)
        if(status/=0) return
        call solve_distributed_compact_cpp(derivatives(1,axis),nrhs)
        call restore_distributed_compact_cpp(derivatives(1,axis),nrhs)
        status=compact_keep_commit_cpp(parent,derivatives(1,axis)%handle,stencil,int(nrhs,c_int),gamma)
        if(status/=0) return
      enddo
    enddo
  end function

  integer(c_int) function compact_filter_cpp(parent) result(status)
    use device_state, only: device_q
    use halo_transport, only: device_exchange_axis,device_exchange_owned_plane
    type(c_ptr),value :: parent
    integer :: axis,offset
#ifdef ASTR_BUILD_TESTING
    status=snapshot_filter_state(parent,'pre_filter')
    if(status/=0) return
#endif
    if(filter_scope=='y-wall5') then
      status=device_exchange_axis(parent,device_q,1_c_int,0_c_int)
      if(status/=0) return
      if(wall5_walls/=0) status=filter_wall5(parent,wall5_plan,int(wall5_walls,c_int))
      if(status/=0) return
      if(wall5_peers/=0) status=device_exchange_owned_plane(parent,device_q,1_c_int, &
        int(wall5_peers,c_int),int(wall5_receivers,c_int))
    else
      do axis=1,3
        if(.not.compact_filter_axis_selected(axis)) cycle
        status=device_exchange_axis(parent,device_q,int(axis-1,c_int),0_c_int)
        if(status/=0) return
        do offset=1,5,filter_components
          status=apply(parent,filters(:,axis),axis,4,filter_components,offset)
          if(status/=0) return
        enddo
      enddo
    endif
#ifdef ASTR_BUILD_TESTING
    if(status==0) status=snapshot_filter_state(parent,'post_filter')
#endif
  end function

#ifdef ASTR_BUILD_TESTING
  integer(c_int) function snapshot_filter_state(parent,label) result(status)
    use commarray, only: q
    use device_state, only: device_q,device_state_download
    use validation_io, only: rhs_validation_requested,write_q_validation_snapshot
    type(c_ptr),value :: parent
    character(*),intent(in) :: label
    status=0
    if(.not.rhs_validation_requested()) return
    status=device_state_download(parent,device_q,q,size(q,kind=c_size_t)*int(storage_size(0.e0_real_kind)/8,c_size_t))
    if(status==0) call write_q_validation_snapshot(label)
  end function
#endif

  subroutine release_compact_flow_cpp()
    integer :: axis,slot
    do axis=1,3
      do slot=1,2
        if(c_associated(derivatives(slot,axis)%handle)) call release_distributed_compact_cpp(derivatives(slot,axis))
        if(c_associated(filters(slot,axis)%handle)) call release_distributed_compact_cpp(filters(slot,axis))
      enddo
    enddo
    cpp_compact_derivative=.false.;cpp_compact_filter=.false.
    phase_timing=.false.
    keep_bases=0
  end subroutine
end module
