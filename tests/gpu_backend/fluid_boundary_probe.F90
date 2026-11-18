program fluid_boundary_probe
  use iso_c_binding
  use ieee_arithmetic, only: ieee_is_finite
  use mpi
  use astr_precision, only: real_kind,real_bytes,real_to_word
  use device_state
  use fluid_explicit
  use compact_operator, only: compact_line_partition,partition_compact_line,compact_derivative_core,compact_filter_core
  use compact_line_solver, only: compact_matrix
  use compact_line_cpp
  implicit none
  integer,parameter :: upper(3)=[4,7,9],hm=5
  real(real_kind),parameter :: gamma=1.4e0_real_kind,velocity(3)=[.1e0_real_kind,.2e0_real_kind,.3e0_real_kind], &
    slope(3)=[.01e0_real_kind,.02e0_real_kind,.03e0_real_kind]
  type(c_ptr) :: state
  type(compact_distributed_cpp_plan) :: plans(3)
  type(compact_stencil_cpp) :: stencils(3),s
  type(compact_line_partition) :: part
  type(compact_matrix) :: matrix
  real(real_kind),allocatable :: q(:,:,:,:),vel(:,:,:,:),prs(:,:,:),jac(:,:,:),metric(:,:,:,:,:), &
    rhs(:,:,:,:),expected(:,:,:,:),temperature(:,:,:),stress(:,:,:,:),heat(:,:,:,:), &
    gradient(:,:,:,:,:),thermal(:,:,:,:),filter_q(:,:,:,:)
  logical,allocatable :: body(:,:,:)
  integer(c_int) :: lower(3),higher(3),first(3),last(3)
  real(real_kind) :: edge(2),target(5),error,maximum=0
  integer :: i,j,k,a,c,debug,mask,exclude,nrhs,count=0,ierr
  character(16) :: method
  logical :: compact
  call get_command_argument(1,method)
  if(len_trim(method)==0) method='explicit'
  if(method/='explicit'.and.method/='compact') error stop 'boundary flux method must be explicit/compact'
  compact=method=='compact'
  call check(device_bind(0_c_int),'bind')
  if(compact) call MPI_Init(ierr)
  allocate(q(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,5), &
    vel(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,3), &
    prs(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm), &
    jac(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm), &
    metric(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,3,3), &
    rhs(0:upper(1),0:upper(2),0:upper(3),5),expected(0:upper(1),0:upper(2),0:upper(3),5), &
    body(0:upper(1),0:upper(2),0:upper(3)),gradient(0:upper(1),0:upper(2),0:upper(3),3,3), &
    thermal(0:upper(1),0:upper(2),0:upper(3),3))
  allocate(temperature,mold=prs);allocate(stress(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,6))
  allocate(heat,mold=vel);allocate(filter_q,mold=q)
  do k=-hm,upper(3)+hm
  do j=-hm,upper(2)+hm
  do i=-hm,upper(1)+hm
    prs(i,j,k)=2+dot_product(slope,real([i,j,k],real_kind))
    temperature(i,j,k)=32+dot_product(slope,real([i,j,k],real_kind))
    vel(i,j,k,:)=velocity
    q(i,j,k,1)=1;q(i,j,k,2:4)=velocity
    q(i,j,k,5)=prs(i,j,k)/(gamma-1)+.5e0_real_kind*sum(velocity**2)
  enddo
  enddo
  enddo
  jac=1;metric=0
  do a=1,3
    metric(:,:,:,a,a)=1
  enddo
  ! Constant density/velocity and linear pressure give an analytic Euler divergence.
  target=[7.e0_real_kind,7-slope,7-gamma/(gamma-1)*dot_product(velocity,slope)]
  do debug=0,1
    call check(device_state_create(0_c_int,int(upper(1),c_int),int(upper(2),c_int),int(upper(3),c_int), &
      int(hm,c_int),int(debug,c_int),state),'create')
    call check(fluid_prepare(state,1_c_int),'primitive fields')
    call check(fluid_rhs_prepare(state,1_c_int),'RHS fields')
    call check(device_state_upload(state,device_q,q,bytes(size(q))),'q')
    call check(device_state_upload(state,device_velocity,vel,bytes(size(vel))),'velocity')
    call check(device_state_upload(state,device_pressure,prs,bytes(size(prs))),'pressure')
    call check(device_state_upload(state,device_temperature,temperature,bytes(size(temperature))),'temperature')
    call check(device_state_upload(state,device_metric,metric,bytes(size(metric))),'metric')
    call check(device_state_upload(state,device_jacobian,jac,bytes(size(jac))),'Jacobian')
    if(compact) then
      do a=1,3
        ! Each line has its complete nonperiodic matrix. Only destination ownership varies below.
        call partition_compact_line(upper(a),0,1,.false.,part)
        call compact_derivative_core(part,matrix,edge)
        nrhs=5*product(upper+1)/(upper(a)+1)
        call prepare_distributed_compact_cpp(state,matrix,edge,part,MPI_COMM_SELF,nrhs,0_c_int,plans(a))
        stencils(a)%axis=a-1;stencils(a)%kind=2;stencils(a)%first=0;stencils(a)%total=upper(a)
        stencils(a)%periodic=0;stencils(a)%start=0;stencils(a)%component_offset=1
        stencils(a)%active_first=0;stencils(a)%active_last=upper
        call check(compact_field_rhs_cpp(state,plans(a)%handle,stencils(a),int(nrhs,c_int)),'compact Euler RHS')
        call solve_distributed_compact_cpp(plans(a),nrhs)
        call restore_distributed_compact_cpp(plans(a),nrhs)
      enddo
    endif
    do mask=0,63
      do a=1,3
        lower(a)=merge(1,0,btest(mask,2*a-2));higher(a)=merge(1,0,btest(mask,2*a-1))
      enddo
      do exclude=0,1
        first=exclude*lower;last=upper-exclude*higher
        body=.false.;body(first(1):last(1),first(2):last(2),first(3):last(3))=.true.
        rhs=7;expected=7
        do c=1,5
          where(body) expected(:,:,:,c)=target(c)
        enddo
        call check(device_state_upload(state,device_rhs,rhs,bytes(size(rhs))),'sentinel RHS')
        if(compact) then
          do a=1,3
            s=stencils(a);s%active_first=first;s%active_last=last
            nrhs=5*product(upper+1)/(upper(a)+1)
            call check(compact_field_commit_cpp(state,plans(a)%handle,s,int(nrhs,c_int)),'compact body-box commit')
          enddo
        else
          call check(fluid_flux_divergence_basic(state,0_c_int,lower,higher,first,last),'explicit body-box accumulate')
        endif
        call check(device_state_download(state,device_rhs,rhs,bytes(size(rhs))),'RHS observation')
        if(.not.all(ieee_is_finite(rhs))) error stop 'nonfinite analytic Euler RHS'
        error=0
        do c=1,5
          error=max(error,maxval(abs(rhs(:,:,:,c)-expected(:,:,:,c)))/max(1.e0_real_kind,maxval(abs(expected(:,:,:,c)))))
        enddo
        if(real_bytes==8.and..not.(error<=2.e-10_real_kind)) then
          write(*,*) 'BOUNDARY_FLUX_DIFFERENCE ',trim(method),debug,mask,exclude,error
          error stop 'Euler divergence or body-box ownership differs'
        endif
        do k=0,upper(3)
        do j=0,upper(2)
        do i=0,upper(1)
          if(body(i,j,k)) cycle
          if(any(real_to_word(rhs(i,j,k,:))/=real_to_word(expected(i,j,k,:)))) &
            error stop 'convection changed boundary-owned RHS bits'
        enddo
        enddo
        enddo
        maximum=max(maximum,error);count=count+1
      enddo
    enddo
    if(compact) then
      call compact_fields()
      do a=1,3
        call release_distributed_compact_cpp(plans(a))
      enddo
    endif
    call check(device_state_destroy(state),'destroy')
  enddo
  if(compact) call MPI_Finalize(ierr)
  write(*,'(a,1x,a,1x,i0,1x,es24.16e3)') 'CPP_BOUNDARY_FLUX_PASS',trim(method),count,maximum
  write(*,'(a,1x,i0)') 'CPP_COMPACT_FIELD_ELEMENT_BYTES',real_bytes
contains
  subroutine compact_fields()
    type(compact_distributed_cpp_plan) :: filter_plans(3)
    type(compact_stencil_cpp) :: stencil
    real(real_kind),parameter :: constant(5)=[1._real_kind,.125_real_kind,.25_real_kind,.5_real_kind,16._real_kind]
    real(real_kind),parameter :: viscous_target(4)=[1._real_kind/16,1._real_kind/8,1._real_kind/4,7._real_kind/8]
    real(real_kind) :: difference
    integer :: axis,lines,columns,mode,component,components
    do axis=1,3
      stencil=stencils(axis);stencil%kind=1
      columns=4*product(upper+1)/(upper(axis)+1)
      call apply_field(plans(axis),stencil,columns)
    enddo
    call check(device_state_download(state,device_dvelocity,gradient,bytes(size(gradient))),'compact gradients')
    call check(device_state_download(state,device_dtemperature,thermal,bytes(size(thermal))),'compact thermal gradients')
    if(.not.all(ieee_is_finite(gradient)).or..not.all(ieee_is_finite(thermal))) error stop 'compact gradient nonfinite'
    if(any(gradient/=0)) error stop 'compact constant velocity derivative'
    difference=0
    do axis=1,3
      difference=max(difference,maxval(abs(thermal(:,:,:,axis)-slope(axis))))
    enddo
    if(real_bytes==8.and.difference>2.e-10_real_kind) error stop 'compact affine thermal gradient'
    write(*,'(a,1x,i0,1x,es24.16e3)') 'CPP_COMPACT_FIELD_GRADIENT',debug,difference

    stress=0
    do k=-hm,upper(3)+hm
    do j=-hm,upper(2)+hm
    do i=-hm,upper(1)+hm
      stress(i,j,k,1)=real(i,real_kind)/16;stress(i,j,k,4)=real(j,real_kind)/8
      stress(i,j,k,6)=real(k,real_kind)/4
      heat(i,j,k,:)=[real(i,real_kind)/8,real(j,real_kind)/4,real(k,real_kind)/2]
    enddo
    enddo
    enddo
    call check(device_state_upload(state,device_stress,stress,bytes(size(stress))),'manufactured stress')
    call check(device_state_upload(state,device_heat_flux,heat,bytes(size(heat))),'manufactured heat flux')
    rhs=0
    call check(device_state_upload(state,device_rhs,rhs,bytes(size(rhs))),'clear manufactured viscous RHS')
    do axis=1,3
      stencil=stencils(axis);stencil%kind=3
      columns=4*product(upper+1)/(upper(axis)+1)
      call apply_field(plans(axis),stencil,columns)
    enddo
    call check(device_state_download(state,device_rhs,rhs,bytes(size(rhs))),'compact viscous divergence')
    if(.not.all(ieee_is_finite(rhs)).or.any(rhs(:,:,:,1)/=0)) error stop 'compact viscous finiteness/mass invariant'
    difference=0
    do component=2,5
      difference=max(difference,maxval(abs(rhs(:,:,:,component)- &
        viscous_target(component-1))))
    enddo
    if(real_bytes==8.and.difference>2.e-10_real_kind) error stop 'compact affine stress/heat divergence'
    write(*,'(a,1x,i0,1x,es24.16e3)') 'CPP_COMPACT_FIELD_VISCOUS',debug,difference

    ! An identity periodic filter checks scalar/full coefficient and field layouts exactly.
    do axis=1,3
      call partition_compact_line(upper(axis),0,1,.true.,part)
      call compact_filter_core(part,0._real_kind,matrix,edge)
      columns=5*product(upper+1)/(upper(axis)+1)
      call prepare_distributed_compact_cpp(state,matrix,edge,part,MPI_COMM_SELF,columns,0_c_int,filter_plans(axis))
    enddo
    do mode=1,2
      components=merge(5,1,mode==1)
      do component=1,5
        filter_q(:,:,:,component)=constant(component)
      enddo
      call check(device_state_upload(state,device_q,filter_q,bytes(size(filter_q))),'constant filter state')
      do axis=1,3
        stencil=stencils(axis);stencil%kind=4;stencil%periodic=1
        stencil%c10=0;stencil%c10(0)=.5_real_kind
        lines=product(upper+1)/(upper(axis)+1)
        do component=1,5,components
          stencil%component_offset=component
          call apply_field(filter_plans(axis),stencil,components*lines)
        enddo
      enddo
      call check(device_state_download(state,device_q,filter_q,bytes(size(filter_q))),'filtered constant state')
      do component=1,5
        if(any(real_to_word(filter_q(0:upper(1),0:upper(2),0:upper(3),component))/= &
          real_to_word(constant(component)))) error stop 'compact scalar/full constant filter bits'
      enddo
    enddo
    do axis=1,3
      call release_distributed_compact_cpp(filter_plans(axis))
    enddo
    write(*,'(a,1x,i0)') 'CPP_COMPACT_FIELD_FILTER_EXACT',debug
  end subroutine

  subroutine apply_field(plan,stencil,columns)
    type(compact_distributed_cpp_plan),intent(in) :: plan
    type(compact_stencil_cpp),intent(in) :: stencil
    integer,intent(in) :: columns
    call check(compact_field_rhs_cpp(state,plan%handle,stencil,int(columns,c_int)),'compact field RHS')
    call solve_distributed_compact_cpp(plan,columns)
    call restore_distributed_compact_cpp(plan,columns)
    call check(compact_field_commit_cpp(state,plan%handle,stencil,int(columns,c_int)),'compact field commit')
  end subroutine

  integer(c_size_t) function bytes(n)
    integer,intent(in) :: n
    bytes=int(n,c_size_t)*real_bytes
  end function
  subroutine check(status,label)
    integer(c_int),intent(in) :: status
    character(*),intent(in) :: label
    if(status==0) return
    write(*,'(a,1x,a,1x,i0)') 'BOUNDARY_FLUX_STATUS',label,status
    error stop 'boundary flux probe failed'
  end subroutine
end program
