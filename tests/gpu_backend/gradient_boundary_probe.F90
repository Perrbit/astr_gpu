program gradient_boundary_probe
  use iso_c_binding
  use ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_quiet_nan
  use device_state
  use fluid_explicit
  use mpi
  use compact_line_cpp
  use compact_operator, only: compact_line_partition,partition_compact_line,compact_derivative_core
  use compact_line_solver, only: compact_matrix
  implicit none
  type(c_ptr) :: state
  type(compact_distributed_cpp_plan) :: plans(3)
  type(compact_stencil_cpp) :: stencils(3)
  integer :: h(4),im,jm,km,hm,unit,axis,backend,stage,debug,poison,c,b,count=0,manufactured=0
  integer(c_int) :: low(3),high(3)
  real(c_double),allocatable :: p(:,:,:,:),seed(:,:,:,:),after(:,:,:,:),metric(:,:,:,:,:), &
    gradient(:,:,:,:,:),reference(:,:,:,:,:),dtmp(:,:,:,:),tref(:,:,:,:)
  real(c_double) :: error,maximum=0,analytic_maximum=0,nan
  character(1024) :: root,path
  character(80) :: suffix
  character(3) :: label
  character(32) :: method,transport
  integer :: ierr,a
  integer(c_int) :: line_mode=0
  logical :: compact,hermite,sbp
  character(1),parameter :: letters(3)=['x','y','z']
  call get_command_argument(1,root)
  call get_command_argument(2,method)
  compact=trim(method)=='compact'
  hermite=trim(method)=='hermite7'
  sbp=trim(method)=='sbp63'
  if(len_trim(method)>0.and.method/='compact'.and.method/='explicit'.and..not.hermite.and..not.sbp) &
    error stop 'gradient method must be explicit/compact/hermite7/sbp63'
  call get_command_argument(3,transport)
  if(len_trim(transport)>0) then
    select case(trim(transport))
    case('pinned');line_mode=0
    case('device-aware');line_mode=1
    case default;error stop 'gradient compact transport must be pinned/device-aware'
    end select
  endif
  if(compact) then
    call check(device_bind(0_c_int))
    call MPI_Init(ierr)
  endif
  nan=ieee_value(0.d0,ieee_quiet_nan)
  do axis=0,2
  if((hermite.or.sbp).and.axis/=1) cycle
  do backend=1,2
    label='cpu'
    if(backend==2) label='gpu'
    do stage=1,3
      write(suffix,'(a,i2.2,a)') '/gradient_input.rk',stage,'.bin'
      path=trim(root)//'/'//trim(label)//'_'//letters(axis+1)//trim(suffix)
      open(newunit=unit,file=trim(path),status='old',access='stream',form='unformatted')
      read(unit) h
      im=h(1);jm=h(2);km=h(3);hm=h(4)
      allocate(p(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6),seed(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6), &
        after(-hm:im+hm,-hm:jm+hm,-hm:km+hm,4),metric(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3), &
        gradient(0:im,0:jm,0:km,3,3),reference(0:im,0:jm,0:km,3,3), &
        dtmp(0:im,0:jm,0:km,3),tref(0:im,0:jm,0:km,3))
      read(unit) p,metric,reference,tref
      close(unit)
      low=0;high=0;low(axis+1)=1;high(axis+1)=1
      do debug=0,1
        call check(device_state_create(0_c_int,im,jm,km,hm,int(debug,c_int),state))
        call check(fluid_prepare(state,1_c_int));call check(fluid_rhs_prepare(state,0_c_int))
        call check(device_state_upload(state,device_metric,metric,bytes(size(metric))))
        if(compact) call prepare_lines()
        do poison=0,1
          seed=p
          if(poison==1) then
            select case(axis)
            case(0);seed(-hm:-1,:,:,:)=nan;seed(im+1:im+hm,:,:,:)=nan
            case(1);seed(:,-hm:-1,:,:)=nan;seed(:,jm+1:jm+hm,:,:)=nan
            case(2);seed(:,:,-hm:-1,:)=nan;seed(:,:,km+1:km+hm,:)=nan
            end select
          endif
          call check(device_state_upload(state,device_velocity,seed(:,:,:,2:4),bytes(3*size(seed(:,:,:,1)))))
          call check(device_state_upload(state,device_temperature,seed(:,:,:,6),bytes(size(seed(:,:,:,1)))))
          if(compact) then
            call compact_gradient()
          elseif(hermite) then
            call check(fluid_gradient_high_wall(state,int(axis,c_int),low,high))
          elseif(sbp) then
            call check(fluid_sbp63_gradient(state,int(jm,c_int),0_c_int,[1.d0,1.d0]))
          else
            call check(fluid_gradient_basic_boundary(state,low,high))
          endif
          call check(device_state_download(state,device_dvelocity,gradient,bytes(size(gradient))))
          call check(device_state_download(state,device_dtemperature,dtmp,bytes(size(dtmp))))
          if(.not.all(ieee_is_finite(gradient)).or..not.all(ieee_is_finite(dtmp))) error stop 'physical ghost read'
          error=0
          do b=1,3
            error=max(error,maxval(abs(dtmp(:,:,:,b)-tref(:,:,:,b)))/max(1.d0,maxval(abs(tref(:,:,:,b)))))
            do c=1,3
              error=max(error,maxval(abs(gradient(:,:,:,c,b)-reference(:,:,:,c,b)))/ &
                max(1.d0,maxval(abs(reference(:,:,:,c,b)))))
            enddo
          enddo
          if(.not.(error<=2.d-10)) error stop 'gradient differs from CPU/old GPU snapshot'
          call check(device_state_download(state,device_velocity,after(:,:,:,1:3),bytes(3*size(after(:,:,:,1)))))
          call check(device_state_download(state,device_temperature,after(:,:,:,4),bytes(size(after(:,:,:,1)))))
          if(any(transfer(after(:,:,:,1:3),[0_c_int64_t],3*size(after(:,:,:,1)))/= &
                 transfer(seed(:,:,:,2:4),[0_c_int64_t],3*size(after(:,:,:,1))))) error stop 'velocity modified'
          if(any(transfer(after(:,:,:,4),[0_c_int64_t],size(after(:,:,:,1)))/= &
                 transfer(seed(:,:,:,6),[0_c_int64_t],size(after(:,:,:,1))))) error stop 'temperature modified'
          maximum=max(maximum,error);count=count+1
        enddo
        if(compact) then
          do a=1,3
            call release_distributed_compact_cpp(plans(a))
          enddo
        endif
        call check(device_state_destroy(state))
      enddo
      deallocate(p,seed,after,metric,gradient,reference,dtmp,tref)
    enddo
  enddo
  enddo
  if(.not.compact.and..not.sbp) call quadratic_fields()
  write(*,'(a,1x,i0,1x,es24.16e3)') 'CPP_BOUNDARY_GRADIENT_PASS',count,maximum
  if(.not.compact.and..not.sbp) &
    write(*,'(a,1x,i0,1x,es24.16e3)') 'CPP_BOUNDARY_GRADIENT_QUADRATIC_PASS',manufactured,analytic_maximum
  if(compact) call MPI_Finalize(ierr)
contains
  subroutine prepare_lines()
    type(compact_line_partition) :: part
    type(compact_matrix) :: matrix
    integer :: a,dims(3),lines
    real(c_double) :: edge(2)
    dims=[im,jm,km]
    do a=1,3
      call partition_compact_line(dims(a),0,1,a/=axis+1,part)
      call compact_derivative_core(part,matrix,edge)
      lines=product(dims+1)/(dims(a)+1)
      call prepare_distributed_compact_cpp(state,matrix,edge,part,MPI_COMM_SELF,min(4*lines,65535),line_mode,plans(a))
      stencils(a)%axis=a-1;stencils(a)%first=0;stencils(a)%total=dims(a);stencils(a)%periodic=merge(1,0,part%periodic)
      stencils(a)%kind=1;stencils(a)%start=0;stencils(a)%component_offset=1
      stencils(a)%active_first=0;stencils(a)%active_last=dims
    enddo
  end subroutine
  subroutine compact_gradient()
    integer :: a,columns,dims(3),start,total
    dims=[im,jm,km]
    do a=1,3
      total=4*product(dims+1)/(dims(a)+1)
      do start=0,total-1,plans(a)%capacity
        stencils(a)%start=start;columns=min(plans(a)%capacity,total-start)
        call check(compact_field_rhs_cpp(state,plans(a)%handle,stencils(a),int(columns,c_int)))
        call solve_distributed_compact_cpp(plans(a),columns)
        call restore_distributed_compact_cpp(plans(a),columns)
        call check(compact_field_commit_cpp(state,plans(a)%handle,stencils(a),int(columns,c_int)))
      enddo
    enddo
  end subroutine
  subroutine quadratic_fields()
    integer :: width,mask,i,j,k,a,n,indices(3),last(3),selected_axis
    real(c_double) :: x(3),derivative(3),g(3,3),value
    ! An exact quadratic checks all 64 physical-end combinations, including
    ! intersections and a four-interval axis, independently of stencil weights.
    im=4;jm=7;km=9
    if(hermite) then
      im=8;jm=9;km=10
    endif
    last=[im,jm,km]
    do width=3,5,2
      hm=width
      allocate(seed(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6), &
        metric(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3),gradient(0:im,0:jm,0:km,3,3), &
        reference(0:im,0:jm,0:km,3,3),dtmp(0:im,0:jm,0:km,3),tref(0:im,0:jm,0:km,3))
      do debug=0,1
        call check(device_state_create(0_c_int,im,jm,km,hm,int(debug,c_int),state))
        call check(fluid_prepare(state,1_c_int));call check(fluid_rhs_prepare(state,0_c_int))
        do mask=0,63
          do a=1,3
            low(a)=merge(1,0,btest(mask,2*a-2));high(a)=merge(1,0,btest(mask,2*a-1))
          enddo
          if(hermite.and.sum(merge(1,0,low+high>0))>1) cycle
          metric=0
          do k=-hm,km+hm
          do j=-hm,jm+hm
          do i=-hm,im+hm
            indices=[i,j,k];x=real(indices,c_double)/8.d0
            seed(i,j,k,:)=0
            do n=1,4
              c=n+1
              if(n==4) c=6
              seed(i,j,k,c)=2.d0+n*(x(1)**2+2*x(2)**2+3*x(3)**2+x(1)*x(2)-x(2)*x(3))*.01d0
            enddo
            do a=1,3
              if((low(a)==1.and.indices(a)<0).or.(high(a)==1.and.indices(a)>last(a))) then
                seed(i,j,k,:)=nan
              endif
            enddo
            if(any(indices<0).or.any(indices>[im,jm,km])) cycle
            do b=1,3
            do a=1,3
              g(a,b)=merge(1.d0,.05d0,a==b)+.01d0*(x(a)+x(b))
            enddo
            enddo
            metric(i,j,k,:,:)=g
            derivative=[2*x(1)+x(2),4*x(2)+x(1)-x(3),6*x(3)-x(2)]*.01d0/8.d0
            do b=1,3
              value=sum(derivative*g(:,b))
              do c=1,3
                reference(i,j,k,c,b)=c*value
              enddo
              tref(i,j,k,b)=4*value
            enddo
          enddo
          enddo
          enddo
          call check(device_state_upload(state,device_velocity,seed(:,:,:,2:4),bytes(3*size(seed(:,:,:,1)))))
          call check(device_state_upload(state,device_temperature,seed(:,:,:,6),bytes(size(seed(:,:,:,1)))))
          call check(device_state_upload(state,device_metric,metric,bytes(size(metric))))
          if(hermite) then
            selected_axis=maxloc(low+high,dim=1)-1
            call check(fluid_gradient_high_wall(state,int(selected_axis,c_int),low,high))
          else
            call check(fluid_gradient_basic_boundary(state,low,high))
          endif
          call check(device_state_download(state,device_dvelocity,gradient,bytes(size(gradient))))
          call check(device_state_download(state,device_dtemperature,dtmp,bytes(size(dtmp))))
          if(.not.all(ieee_is_finite(gradient)).or..not.all(ieee_is_finite(dtmp))) error stop 'quadratic ghost read'
          error=max(maxval(abs(gradient-reference)),maxval(abs(dtmp-tref)))
          if(.not.(error<=2.d-10)) error stop 'quadratic gradient differs from analytic projection'
          if(mask==0) then
            reference=gradient;tref=dtmp
            call check(fluid_gradient(state))
            call check(device_state_download(state,device_dvelocity,gradient,bytes(size(gradient))))
            call check(device_state_download(state,device_dtemperature,dtmp,bytes(size(dtmp))))
            if(any(transfer(gradient,[0_c_int64_t],size(gradient))/= &
                   transfer(reference,[0_c_int64_t],size(reference))).or. &
               any(transfer(dtmp,[0_c_int64_t],size(dtmp))/=transfer(tref,[0_c_int64_t],size(tref)))) &
              error stop 'no-boundary path changed centered gradient bits'
          endif
          analytic_maximum=max(analytic_maximum,error);manufactured=manufactured+1
        enddo
        call check(device_state_destroy(state))
      enddo
      deallocate(seed,metric,gradient,reference,dtmp,tref)
    enddo
  end subroutine
  integer(c_size_t) function bytes(n)
    integer,intent(in) :: n
    bytes=int(n,c_size_t)*8_c_size_t
  end function
  subroutine check(status)
    integer,intent(in) :: status
    if(status==0) return
    print *, 'boundary gradient status=',status
    error stop 1
  end subroutine
end program
