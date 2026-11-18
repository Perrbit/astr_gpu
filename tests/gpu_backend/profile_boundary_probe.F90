program profile_boundary_probe
  use iso_c_binding
  use ieee_arithmetic, only: ieee_is_finite
  use mpi
  use device_state
  use fluid_explicit
  use boundary_state_cpp
  implicit none
  type(c_ptr) :: state
  type(device_view) :: target_before,target_after
  integer :: rank,ranks,local_rank,ierr,split_axis=1,h(4),g(3),upper(3),origin(3),hm,unit, &
    backend,stage,debug,operation,trial,j,k,c,i,cases=0,analytic=0
  logical :: own_inlet,own_top,own_outlet
  real(c_double),allocatable :: gq(:,:,:,:),gp(:,:,:,:),gt(:,:,:),q(:,:,:,:),p(:,:,:,:), &
    seedq(:,:,:,:),seedp(:,:,:,:),observedq(:,:,:,:),observedp(:,:,:,:),target(:,:,:), &
    rhs(:,:,:,:),expected_rhs(:,:,:,:),expectedq(:,:,:,:),expectedp(:,:,:,:),metric(:,:,:,:,:)
  real(c_double),parameter :: mach=2.d0,c2=5.6d0,c6=2.5d0
  real(c_double) :: den,pressure,temperature,u(3),css,vnormal,error,maximum=0,analytic_max=0,ratio
  character(1024) :: root,path
  character(80) :: text,suffix
  character(3) :: name
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text,status=ierr)
  if(ierr/=0) error stop 'profile probe needs matching Open MPI launcher'
  read(text,*) local_rank
  call check(device_bind(int(local_rank,c_int)))
  call MPI_Init(ierr);call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  if(ranks/=1.and.ranks/=2) error stop 'profile probe covers NP=1/2'
  call get_command_argument(1,root)
  if(ranks==2) then
    call get_command_argument(2,text);read(text,*) split_axis
    if(split_axis<1.or.split_axis>3) error stop 'profile split axis must be 1/2/3'
  endif
  do backend=1,2
    name='cpu'
    if(backend==2) name='gpu'
    do stage=1,3
      write(suffix,'(a,i2.2,a)') '.rk',stage,'.bin'
      path=trim(root)//'/'//name//trim(suffix)
      open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='old')
      read(unit) h;g=h(:3);hm=h(4);upper=g;origin=0
      if(ranks==2) then
        upper(split_axis)=g(split_axis)/2;origin(split_axis)=rank*upper(split_axis)
      endif
      own_inlet=origin(1)==0;own_top=origin(2)+upper(2)==g(2)
      own_outlet=origin(1)+upper(1)==g(1)
      allocate(gq(-hm:g(1)+hm,-hm:g(2)+hm,-hm:g(3)+hm,5),gp(-hm:g(1)+hm,-hm:g(2)+hm,-hm:g(3)+hm,6), &
        gt(0:g(2),0:g(3),9),q(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,5), &
        p(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,6),target(0:upper(2),0:upper(3),9))
      read(unit) gq,gp,gt;close(unit)
      q=gq(origin(1)-hm:origin(1)+upper(1)+hm,origin(2)-hm:origin(2)+upper(2)+hm,origin(3)-hm:origin(3)+upper(3)+hm,:)
      p=gp(origin(1)-hm:origin(1)+upper(1)+hm,origin(2)-hm:origin(2)+upper(2)+hm,origin(3)-hm:origin(3)+upper(3)+hm,:)
      target=gt(origin(2):origin(2)+upper(2),origin(3):origin(3)+upper(3),:)
      allocate(seedq(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,5), &
        seedp(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,6), &
        observedq(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,5), &
        observedp(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,6), &
        expectedq(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,5), &
        expectedp(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,6), &
        rhs(0:upper(1),0:upper(2),0:upper(3),5),expected_rhs(0:upper(1),0:upper(2),0:upper(3),5), &
        metric(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,3,3))
      metric=0
      do c=1,3
        metric(:,:,:,c,c)=1
      enddo
      do debug=0,1
        target=gt(origin(2):origin(2)+upper(2),origin(3):origin(3)+upper(3),:)
        call check(device_state_create(int(local_rank,c_int),upper(1),upper(2),upper(3),hm,int(debug,c_int),state))
        call check(fluid_prepare(state,0_c_int));call check(fluid_rhs_prepare(state,0_c_int))
        call check(device_state_upload(state,device_metric,metric,bytes(size(metric))))
        if(device_state_view(state,device_inflow_target,target_before)/=device_not_allocated) error stop 'unrequested profile storage'
        if(own_inlet) then
          call check(profile_inflow_prepare(state));call check(device_state_view(state,device_inflow_target,target_before))
          call check(profile_inflow_prepare(state));call check(device_state_view(state,device_inflow_target,target_after))
          if(.not.c_associated(target_before%data,target_after%data)) error stop 'profile prepare changed storage'
          if(any(target_before%extent/=[1,upper(2)+1,upper(3)+1,9,1])) error stop 'profile is not plane-only'
          call check(device_state_upload(state,device_inflow_target,target,bytes(size(target))))
        endif
        do operation=0,2
          if(operation==0.and..not.own_inlet) cycle
          if(operation==1.and..not.own_top) cycle
          if(operation==2.and..not.own_outlet) cycle
          seedq=q;seedp=p;rhs=7;expected_rhs=7
          if(operation==0) then
            do k=0,upper(3)
            do j=0,upper(2)
              call seed_face(0,j,k)
            enddo
            enddo
            expected_rhs(0,:,:,:)=0
          elseif(operation==1) then
            do k=0,upper(3)
            do i=0,upper(1)
              call seed_face(i,upper(2),k)
            enddo
            enddo
            expected_rhs(:,upper(2),:,:)=0
          else
            expected_rhs(upper(1),:,:,:)=0
          endif
          expectedq=seedq;expectedp=seedp
          call upload()
          if(operation==0) then
            do k=0,upper(3)
            do j=0,upper(2)
              call target_values(j,k)
              call set_expected(0,j,k)
            enddo
            enddo
            call check(profile_inflow_x(state,0_c_int,mach,c2,c6))
          elseif(operation==1) then
            do k=0,upper(3)
            do i=0,upper(1)
              den=(4*p(i,upper(2)-1,k,1)-p(i,upper(2)-2,k,1))/3
              pressure=(4*p(i,upper(2)-1,k,5)-p(i,upper(2)-2,k,5))/3
              temperature=pressure/den*c2
              u=(4*p(i,upper(2)-1,k,2:4)-p(i,upper(2)-2,k,2:4))/3
              call set_expected(i,upper(2),k)
            enddo
            enddo
            call check(farfield51_upper_basic(state,c2,c6))
          else
            do k=0,upper(3)
            do j=0,upper(2)
              call outlet_values(j,k,1.d0/c2)
              call set_expected(upper(1),j,k)
            enddo
            enddo
            call check(outflow21_x_basic(state,mach,c2,c6,1.d0/c2))
          endif
          call observe();call compare_all(error);maximum=max(maximum,error);cases=cases+1
          if(operation==0) then
            do j=0,upper(2)
              if(origin(2)+j==0.or.origin(2)+j==g(2)) cycle
              do c=1,6
                error=maxval(abs(observedp(0,j,0:upper(3),c)-p(0,j,0:upper(3),c)))/max(1.d0,maxval(abs(p(0,j,0:upper(3),c))))
                if(.not.(error<=2.d-10)) error stop 'profile differs from frozen boundary interior'
                maximum=max(maximum,error)
              enddo
            enddo
          elseif(operation==1) then
            do c=1,6
              error=maxval(abs(observedp(0:upper(1),upper(2),0:upper(3),c)-p(0:upper(1),upper(2),0:upper(3),c)))/ &
                max(1.d0,maxval(abs(p(0:upper(1),upper(2),0:upper(3),c))))
              if(.not.(error<=2.d-10)) error stop 'bc51 differs from frozen original top'
              maximum=max(maximum,error)
            enddo
          else
            do j=0,upper(2)
              if(origin(2)+j==0.or.origin(2)+j==g(2)) cycle
              do c=1,6
                error=maxval(abs(observedp(upper(1),j,0:upper(3),c)-p(upper(1),j,0:upper(3),c)))/ &
                  max(1.d0,maxval(abs(p(upper(1),j,0:upper(3),c))))
                if(.not.(error<=2.d-10)) error stop 'bc21 differs from frozen outlet interior'
                maximum=max(maximum,error)
              enddo
            enddo
          endif
        enddo
        if(own_inlet.and.backend==1.and.stage==1) call mach_pressure_cases()
        if(own_outlet.and.backend==1.and.stage==1) call outlet_mach_cases()
        call check(device_state_destroy(state))
      enddo
      deallocate(gq,gp,gt,q,p,target,seedq,seedp,observedq,observedp,expectedq,expectedp,rhs,expected_rhs,metric)
    enddo
  enddo
  write(*,'(a,2(1x,i0),1x,es24.16e3)') 'CPP_PROFILE_BOUNDARY_PASS',rank,cases,maximum
  write(*,'(a,2(1x,i0),1x,es24.16e3)') 'CPP_PROFILE_MACH_PASS',rank,analytic,analytic_max
  call MPI_Finalize(ierr)
contains
  subroutine outlet_values(y,z,target_pressure)
    integer,intent(in) :: y,z
    real(c_double),intent(in) :: target_pressure
    real(c_double) :: pe,roe,ce
    integer :: n
    n=upper(1)
    u=(4*seedp(n-1,y,z,2:4)-seedp(n-2,y,z,2:4))/3
    pe=(4*seedp(n-1,y,z,5)-seedp(n-2,y,z,5))/3
    temperature=seedp(n,y,z,6)
    if(seedp(n,y,z,2)>=sqrt(temperature)/mach) then
      pressure=pe;temperature=(4*seedp(n-1,y,z,6)-seedp(n-2,y,z,6))/3
    else
      pressure=target_pressure
      roe=(4*seedp(n-1,y,z,1)-seedp(n-2,y,z,1))/3
      ce=(4*sqrt(seedp(n-1,y,z,6))/mach-sqrt(seedp(n-2,y,z,6))/mach)/3
      u(1)=u(1)+(pe-pressure)/roe/ce
    endif
    den=pressure/temperature*c2
  end subroutine
  subroutine outlet_mach_cases()
    real(c_double) :: pe
    integer :: n,x
    operation=2;n=upper(1)
    do trial=0,2
      seedq=q;seedp=p;rhs=7;expected_rhs=7;expected_rhs(n,:,:,:)=0
      do k=0,upper(3)
      do j=0,upper(2)
        seedp(n,j,k,:)=[1.d0,.5d0,.1d0,-.1d0,4.d0/c2,4.d0]
        if(trial==1) seedp(n,j,k,2)=1.5d0
        if(trial==2) seedp(n,j,k,2)=1.d0
        seedp(n-1,j,k,:)=[1.2d0,1.1d0,.1d0,-.1d0,1.2d0*9.d0/c2,9.d0]
        seedp(n-2,j,k,:)=[1.5d0,.8d0,.1d0,-.1d0,1.5d0/c2,1.d0]
        do x=n-2,n
          seedq(x,j,k,:)=[seedp(x,j,k,1),seedp(x,j,k,1)*seedp(x,j,k,2:4), &
            seedp(x,j,k,5)*c6+.5d0*seedp(x,j,k,1)*sum(seedp(x,j,k,2:4)**2)]
        enddo
      enddo
      enddo
      expectedq=seedq;expectedp=seedp
      pe=41.7d0/(3.d0*c2)
      do k=0,upper(3)
      do j=0,upper(2)
        u=[1.2d0,.1d0,-.1d0];temperature=35.d0/3.d0;pressure=pe
        if(trial==0) then
          pressure=.2d0;temperature=4;u(1)=u(1)+(pe-.2d0)/(11.d0/10.d0)/(11.d0/6.d0)
        endif
        den=pressure/temperature*c2;call set_expected(n,j,k)
      enddo
      enddo
      call upload();call check(outflow21_x_basic(state,mach,c2,c6,.2d0))
      call observe();call compare_all(error);analytic_max=max(analytic_max,error);analytic=analytic+1
    enddo
  end subroutine
  subroutine mach_pressure_cases()
    operation=0
    do trial=0,2
      target=0
      do k=0,upper(3)
      do j=0,upper(2)
        target(j,k,1)=1.d0+.01d0*(origin(2)+j)
        target(j,k,6)=1.d0+.02d0*(origin(3)+k)
        target(j,k,5)=target(j,k,1)*target(j,k,6)/c2
        css=sqrt(target(j,k,6))/mach
        ratio=.5d0
        if(trial==1) ratio=1.5d0
        target(j,k,3)=.2d0;target(j,k,4)=.1d0
        target(j,k,7)=.6d0;target(j,k,8)=.8d0
        target(j,k,2)=(ratio*css-.8d0*target(j,k,3))/.6d0
        if(trial==2) then
          target(j,k,6)=4;target(j,k,5)=target(j,k,1)*4/c2
          target(j,k,2)=1;target(j,k,3:4)=0;target(j,k,7)=1;target(j,k,8)=0
        endif
      enddo
      enddo
      seedq=q;seedp=p;rhs=7;expected_rhs=7;expected_rhs(0,:,:,:)=0
      seedp(1,:,:,5)=1.5d0;seedp(2,:,:,5)=.6d0
      do k=0,upper(3)
      do j=0,upper(2)
      do i=1,2
        seedp(i,j,k,6)=seedp(i,j,k,5)/seedp(i,j,k,1)*c2
        seedq(i,j,k,1)=seedp(i,j,k,1)
        seedq(i,j,k,2:4)=seedp(i,j,k,1)*seedp(i,j,k,2:4)
        seedq(i,j,k,5)=seedp(i,j,k,5)*c6+.5d0*seedp(i,j,k,1)*sum(seedp(i,j,k,2:4)**2)
      enddo
      enddo
      enddo
      expectedq=seedq;expectedp=seedp
      do k=0,upper(3)
      do j=0,upper(2)
        call target_values(j,k)
        if(trial==0) then
          temperature=target(j,k,6);pressure=1.8d0;den=pressure/temperature*c2
        endif
        call set_expected(0,j,k)
      enddo
      enddo
      call upload();call check(device_state_upload(state,device_inflow_target,target,bytes(size(target))))
      call check(profile_inflow_x(state,1_c_int,mach,c2,c6));call observe();call compare_all(error)
      analytic_max=max(analytic_max,error);analytic=analytic+1
    enddo
  end subroutine
  subroutine seed_face(x,y,z)
    integer,intent(in) :: x,y,z
    seedp(x,y,z,:)=[2.d0,.1d0,.2d0,.3d0,2.d0,5.6d0]
    seedq(x,y,z,:)=[2.d0,.2d0,.4d0,.6d0,2.d0*c6+sum([.1d0,.2d0,.3d0]**2)]
  end subroutine
  subroutine target_values(y,z)
    integer,intent(in) :: y,z
    den=target(y,z,1);u=target(y,z,2:4);pressure=target(y,z,5);temperature=pressure/den*c2
  end subroutine
  subroutine set_expected(x,y,z)
    integer,intent(in) :: x,y,z
    expectedp(x,y,z,:)=[den,u,pressure,temperature]
    expectedq(x,y,z,:)=[den,den*u,pressure*c6+.5d0*den*sum(u*u)]
  end subroutine
  subroutine upload()
    call check(device_state_upload(state,device_q,seedq,bytes(size(seedq))))
    call check(device_state_upload(state,device_rho,seedp(:,:,:,1),bytes(size(seedp(:,:,:,1)))))
    call check(device_state_upload(state,device_velocity,seedp(:,:,:,2:4),bytes(3*size(seedp(:,:,:,1)))))
    call check(device_state_upload(state,device_pressure,seedp(:,:,:,5),bytes(size(seedp(:,:,:,1)))))
    call check(device_state_upload(state,device_temperature,seedp(:,:,:,6),bytes(size(seedp(:,:,:,1)))))
    call check(device_state_upload(state,device_rhs,rhs,bytes(size(rhs))))
  end subroutine
  subroutine observe()
    call check(device_state_download(state,device_q,observedq,bytes(size(observedq))))
    call check(device_state_download(state,device_rho,observedp(:,:,:,1),bytes(size(seedp(:,:,:,1)))))
    call check(device_state_download(state,device_velocity,observedp(:,:,:,2:4),bytes(3*size(seedp(:,:,:,1)))))
    call check(device_state_download(state,device_pressure,observedp(:,:,:,5),bytes(size(seedp(:,:,:,1)))))
    call check(device_state_download(state,device_temperature,observedp(:,:,:,6),bytes(size(seedp(:,:,:,1)))))
    call check(device_state_download(state,device_rhs,rhs,bytes(size(rhs))))
    call check(fluid_check(state))
  end subroutine
  subroutine compare_all(value)
    real(c_double),intent(out) :: value
    integer :: component,x,y,z
    real(c_double) :: qscale(5),pscale(6)
    logical :: selected
    value=0
    do component=1,5
      if(operation/=1) then
        qscale(component)=max(1.d0,maxval(abs(expectedq(merge(0,upper(1),operation==0),0:upper(2),0:upper(3),component))))
      else
        qscale(component)=max(1.d0,maxval(abs(expectedq(0:upper(1),upper(2),0:upper(3),component))))
      endif
    enddo
    do component=1,6
      if(operation/=1) then
        pscale(component)=max(1.d0,maxval(abs(expectedp(merge(0,upper(1),operation==0),0:upper(2),0:upper(3),component))))
      else
        pscale(component)=max(1.d0,maxval(abs(expectedp(0:upper(1),upper(2),0:upper(3),component))))
      endif
    enddo
    do z=-hm,upper(3)+hm
    do y=-hm,upper(2)+hm
    do x=-hm,upper(1)+hm
      selected=(operation==0.and.x==0.and.y>=0.and.y<=upper(2).and.z>=0.and.z<=upper(3)).or. &
        (operation==1.and.y==upper(2).and.x>=0.and.x<=upper(1).and.z>=0.and.z<=upper(3)).or. &
        (operation==2.and.x==upper(1).and.y>=0.and.y<=upper(2).and.z>=0.and.z<=upper(3))
      if(selected) then
        if(.not.all(ieee_is_finite(observedq(x,y,z,:))).or..not.all(ieee_is_finite(observedp(x,y,z,:)))) &
          error stop 'nonfinite boundary command result'
        value=max(value,maxval(abs(observedq(x,y,z,:)-expectedq(x,y,z,:))/qscale))
        value=max(value,maxval(abs(observedp(x,y,z,:)-expectedp(x,y,z,:))/pscale))
      else
        if(any(transfer(observedq(x,y,z,:),[0_c_int64_t],5)/=transfer(seedq(x,y,z,:),[0_c_int64_t],5)).or. &
          any(transfer(observedp(x,y,z,:),[0_c_int64_t],6)/=transfer(seedp(x,y,z,:),[0_c_int64_t],6))) &
          error stop 'boundary command changed another node'
      endif
    enddo
    enddo
    enddo
    if(.not.(value<=2.d-10).or.any(transfer(rhs,[0_c_int64_t],size(rhs))/= &
      transfer(expected_rhs,[0_c_int64_t],size(rhs)))) error stop 'boundary state/RHS contract differs'
  end subroutine
  integer(c_size_t) function bytes(n)
    integer,intent(in) :: n
    bytes=int(n,c_size_t)*8_c_size_t
  end function
  subroutine check(status)
    integer,intent(in) :: status
    if(status==0) return
    write(*,*) 'PROFILE_BOUNDARY_STATUS',rank,backend,stage,debug,operation,trial,status
    error stop 'profile boundary command failed'
  end subroutine
end program
