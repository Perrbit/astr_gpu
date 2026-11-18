program filter_boundary_probe
  use iso_c_binding
  use astr_precision, only: real_kind,real_bytes,real_to_word
  use ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_quiet_nan
  use device_state
  use fluid_explicit
  implicit none
  integer,parameter :: hm=5
  real(real_kind) :: nan,error,maximum=0
  real(real_kind),allocatable :: seed(:,:,:,:),qafter(:,:,:,:),target(:,:,:,:)
  real(real_kind),allocatable :: work(:,:,:,:),wafter(:,:,:,:)
  real(real_kind),allocatable :: layout_reference(:,:,:,:,:),sync_reference(:,:,:,:,:)
  integer :: axis,mask,debug,nc,c,component,q_component,i,j,k,mode,position(3),last(3),cases=0,grid,im,jm,km
  integer(c_int) :: lower,upper
  character(16) :: method
  logical :: high_wall
  integer(c_int) :: adiabatic
  type(c_ptr) :: state
  call get_command_argument(1,method)
  high_wall=method=='wall41'.or.method=='wall42'
  adiabatic=int(merge(1,0,method=='wall42'),c_int)
  if(len_trim(method)>0.and..not.high_wall) error stop 'filter probe method must be wall41/wall42 or empty'
  nan=ieee_value(0._real_kind,ieee_quiet_nan)
  do grid=1,2
  last=[8,10,12]
  if(grid==2) last=5
  im=last(1);jm=last(2);km=last(3)
  allocate(seed(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),qafter(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5), &
    target(0:im,0:jm,0:km,5))
  allocate(layout_reference(0:im,0:jm,0:km,5,3),sync_reference(0:im,0:jm,0:km,5,3))
  do axis=0,2
  do mask=0,merge(3,2,grid==1)
    lower=merge(1,0,btest(mask,0));upper=merge(1,0,btest(mask,1))
    do debug=0,1
    do nc=1,5,4
      allocate(work(-hm:im+hm,-hm:jm+hm,-hm:km+hm,nc),wafter(-hm:im+hm,-hm:jm+hm,-hm:km+hm,nc))
      do component=1,merge(5,1,nc==1)
      do mode=1,merge(1,3,high_wall)
        do k=-hm,km+hm
        do j=-hm,jm+hm
        do i=-hm,im+hm
          position=[i,j,k]
          do c=1,5
            if(mode==1) then
              seed(i,j,k,c)=real(c,real_kind)+(real(i,real_kind)**5+2*real(j,real_kind)**5-3*real(k,real_kind)**5)/1048576._real_kind
            elseif(mode==2) then
              seed(i,j,k,c)=2._real_kind+c+(.125_real_kind*c)*real(1-2*modulo(position(axis+1),2),real_kind)
            else
              seed(i,j,k,c)=2._real_kind+c
            endif
          enddo
          if((lower==1.and.position(axis+1)<0).or.(upper==1.and.position(axis+1)>last(axis+1))) seed(i,j,k,:)=nan
        enddo
        enddo
        enddo
        target=seed(0:im,0:jm,0:km,:)
        if(mode==2) then
          do k=0,km
          do j=0,jm
          do i=0,im
            position=[i,j,k]
            if((lower==1.and.position(axis+1)==0).or.(upper==1.and.position(axis+1)==last(axis+1))) cycle
            do c=1,5
              target(i,j,k,c)=2._real_kind+c
            enddo
          enddo
          enddo
          enddo
        endif
        call check(device_state_create(0_c_int,im,jm,km,hm,int(debug,c_int),state))
        if(nc==1) then
          call check(fluid_filter_prepare_scalar(state))
        else
          call check(fluid_filter_prepare(state))
        endif
        call check(device_state_upload(state,device_q,seed,bytes(size(seed))))
        work=-91
        if(axis==1) then
          if(nc==1) then
            work(:,:,:,1)=seed(:,:,:,component)
          else
            work=seed
          endif
        endif
        call check(device_state_upload(state,device_filter_work,work,bytes(size(work))))
        if(high_wall) then
          call check(fluid_filter_axis_high_wall(state,int(axis,c_int),int(merge(component,0,nc==1),c_int),lower,upper,adiabatic))
        elseif(nc==1) then
          call check(fluid_filter_component_axis_basic(state,int(axis,c_int),int(component,c_int),lower,upper))
        else
          call check(fluid_filter_axis_basic(state,int(axis,c_int),lower,upper))
        endif
        call check(device_state_download(state,device_q,qafter,bytes(size(qafter))))
        call check(device_state_download(state,device_filter_work,wafter,bytes(size(wafter))))
        if(axis==1) then
          if(any(real_to_word(work)/=real_to_word(wafter))) &
            error stop 'filter modified its work source'
        else
          if(any(real_to_word(seed)/=real_to_word(qafter))) &
            error stop 'filter modified its q source'
        endif
        do c=1,nc
          if(axis==1) then
            q_component=merge(component,c,nc==1)
            if(.not.all(ieee_is_finite(qafter(0:im,0:jm,0:km,q_component)))) error stop 'filter read physical ghost'
            error=maxval(abs(qafter(0:im,0:jm,0:km,q_component)-target(:,:,:,q_component)))/ &
              max(1._real_kind,maxval(abs(target(:,:,:,q_component))))
          else
            q_component=merge(component,c,nc==1)
            if(.not.all(ieee_is_finite(wafter(0:im,0:jm,0:km,c)))) error stop 'filter read physical ghost'
            error=maxval(abs(wafter(0:im,0:jm,0:km,c)-target(:,:,:,q_component)))/ &
              max(1._real_kind,maxval(abs(target(:,:,:,q_component))))
          endif
          if(nc==1) then
            if(axis==1) then
              layout_reference(:,:,:,q_component,mode)=qafter(0:im,0:jm,0:km,q_component)
            else
              layout_reference(:,:,:,q_component,mode)=wafter(0:im,0:jm,0:km,c)
            endif
          else
            if(axis==1) then
              if(any(real_to_word(layout_reference(:,:,:,c,mode))/= &
                  real_to_word(qafter(0:im,0:jm,0:km,c)))) error stop 'filter scalar/full words differ'
            else
              if(any(real_to_word(layout_reference(:,:,:,c,mode))/= &
                  real_to_word(wafter(0:im,0:jm,0:km,c)))) error stop 'filter scalar/full words differ'
            endif
          endif
          if((real_bytes==8.and..not.(error<=2.e-10_real_kind)).or.(mode>=2.and.error/=0)) then
            write(*,*) axis,mask,debug,nc,component,mode,c,error
            error stop 'filter polynomial/Nyquist identity'
          endif
          maximum=max(maximum,error)
        enddo
        if(nc==5) then
          if(debug==0) then
            sync_reference(:,:,:,:,mode)=layout_reference(:,:,:,:,mode)
          elseif(any(real_to_word(sync_reference(:,:,:,:,mode))/= &
                     real_to_word(layout_reference(:,:,:,:,mode)))) then
            error stop 'filter synchronization words differ'
          endif
        endif
        do k=-hm,km+hm
        do j=-hm,jm+hm
        do i=-hm,im+hm
          if(i>=0.and.i<=im.and.j>=0.and.j<=jm.and.k>=0.and.k<=km) cycle
          if(any(real_to_word(qafter(i,j,k,:))/=real_to_word(seed(i,j,k,:)))) &
            error stop 'filter changed q ghosts'
          if(any(real_to_word(wafter(i,j,k,:))/=real_to_word(work(i,j,k,:)))) &
            error stop 'filter changed work ghosts'
        enddo
        enddo
        enddo
        if(axis==1.and.nc==1) then
          do c=1,5
            if(c==component) cycle
            if(any(real_to_word(qafter(:,:,:,c))/=real_to_word(seed(:,:,:,c)))) error stop 'scalar filter changed another component'
          enddo
        endif
        call check(device_state_destroy(state))
        cases=cases+1
      enddo
      enddo
      deallocate(work,wafter)
    enddo
    enddo
  enddo
  enddo
  deallocate(seed,qafter,target,layout_reference,sync_reference)
  enddo
  write(*,'(a,1x,a,1x,i0,1x,es24.16e3)') 'CPP_FILTER_BOUNDARY_PASS',trim(method),cases,maximum
  write(*,'(a,1x,i0)') 'CPP_FILTER_BOUNDARY_REAL_BYTES',real_bytes
contains
  integer(c_size_t) function bytes(n)
    integer,intent(in) :: n
    bytes=int(n,c_size_t)*int(real_bytes,c_size_t)
  end function
  subroutine check(status)
    integer,intent(in) :: status
    if(status==0) return
    print *, 'filter boundary status=',status
    error stop 1
  end subroutine
end program
