program shock_mask_halo_probe
  use iso_c_binding
  use astr_precision, only: real_kind,real_bytes
  use ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  use mpi
  use device_state
  use fluid_explicit
  use halo_transport
  use shock_sensor_cpp
  implicit none
  interface
    integer(c_int) function interfaces(handle,axis,host,count) bind(C,name='test_shock_interfaces')
      import c_ptr,c_int,c_int8_t,c_size_t
      type(c_ptr),value :: handle
      integer(c_int),value :: axis
      integer(c_int8_t),intent(out) :: host(*)
      integer(c_size_t),value :: count
    end function
  end interface
  integer,parameter :: g(3)=[24,24,24],hm=5
  type(c_ptr) :: state
  integer :: rank,ranks,local_rank,ierr,cart,axis=1,fixture,debug,iteration,cases=0
  integer :: i,j,k,a,b,c,h,side,p(3),point(3),n1,n2,pos,face_size,total_interfaces=0,changed=0,global_changed
  integer(c_int) :: dims(3),upper(3),origin(3),periodic(3),low(3),high(3),lower(3),higher(3),mode
  logical :: periods(3)
  real(real_kind),allocatable :: raw(:,:,:)
  integer(c_int8_t),allocatable :: mask(:,:,:),expected(:,:,:),observed(:),faces(:)
  integer(c_int8_t),allocatable :: interface_before(:,:),interface_after(:),interface_expected(:)
  character(80) :: text
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text,status=ierr)
  if(ierr/=0) error stop 'mask halo probe needs the matching MPI launcher'
  read(text,*) local_rank
  call check(device_bind(int(local_rank,c_int)))
  call MPI_Init(ierr);call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  call get_command_argument(1,text);read(text,*) axis
  call get_command_argument(2,text);mode=halo_pinned
  if(trim(text)=='device') mode=halo_device
  dims=1;dims(axis)=ranks;upper=g;upper(axis)=g(axis)/ranks;origin=0;origin(axis)=rank*upper(axis)
  face_size=0
  do a=1,3
    face_size=face_size+2*(upper(modulo(a,3)+1)+1)*(upper(modulo(a+1,3)+1)+1)
    total_interfaces=max(total_interfaces,(upper(a)+2)*(upper(modulo(a,3)+1)+1)*(upper(modulo(a+1,3)+1)+1))
  enddo
  allocate(raw(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm), &
    mask(0:upper(1),0:upper(2),0:upper(3)),expected(0:upper(1),0:upper(2),0:upper(3)), &
    faces(face_size),observed(face_size),interface_before(total_interfaces,3), &
    interface_after(total_interfaces),interface_expected(total_interfaces))
  do fixture=1,2
    periodic=1
    if(fixture==2) periodic=[0,0,1]
    periods=periodic/=0
    call MPI_Cart_create(MPI_COMM_WORLD,3,dims,periods,.false.,cart,ierr);call check(ierr)
    do a=1,3
      call MPI_Cart_shift(cart,a-1,1,lower(a),higher(a),ierr);call check(ierr)
    enddo
    low=merge(1,0,periodic==0.and.origin==0);high=merge(1,0,periodic==0.and.origin+upper==g)
    do debug=0,1
      call check(device_state_create(int(local_rank,c_int),upper(1),upper(2),upper(3),hm,int(debug,c_int),state))
      call check(fluid_prepare(state,1_c_int));call check(shock_prepare_cpp(state))
      call check(device_transport_prepare(state,int(cart,c_int),lower,higher,dims,1_c_int,mode,0_c_int,periodic))
      do iteration=1,2
        raw=ieee_value(0.e0_real_kind,ieee_quiet_nan)
        do k=0,upper(3)
        do j=0,upper(2)
        do i=0,upper(1)
          point=origin+[i,j,k];raw(i,j,k)=raw_value(point);expected(i,j,k)=expanded_value(point)
        enddo
        enddo
        enddo
        call check(device_state_upload(state,device_shock_sensor,raw,int(size(raw),c_size_t)*real_bytes))
        do a=0,2
          call check(device_exchange_axis(state,device_shock_sensor,int(a,c_int),0_c_int))
        enddo
        call check(shock_expand_cpp(state,low,high,.5e0_real_kind))
        if(shock_faces_download_cpp(state,observed,int(face_size,c_size_t))/=device_not_allocated) &
          error stop 'expanded-mask faces were not invalidated'
        do a=1,3
          n1=(upper(a)+2)*(upper(modulo(a,3)+1)+1)*(upper(modulo(a+1,3)+1)+1)
          call check(interfaces(state,int(a-1,c_int),interface_before(:n1,a),int(n1,c_size_t)))
        enddo
        call check(shock_mask_exchange_cpp(state))
        call check(shock_mask_exchange_cpp(state))
        call check(shock_mask_download_cpp(state,mask,int(size(mask),c_size_t)))
        if(any(mask/=expected)) error stop 'mask exchange changed physical markers'
        pos=0
        do a=1,3
          b=modulo(a,3)+1;c=modulo(a+1,3)+1;n1=upper(b)+1;n2=upper(c)+1
          do side=0,1
            do j=0,n2-1
            do i=0,n1-1
              p=0;p(b)=i;p(c)=j;p(a)=merge(-1,upper(a)+1,side==0)
              point=origin+p
              if(periodic(a)==0) point(a)=max(0,min(g(a),point(a)))
              pos=pos+1;faces(pos)=expanded_value(point)
            enddo
            enddo
          enddo
        enddo
        call check(shock_faces_download_cpp(state,observed,int(face_size,c_size_t)))
        if(any(observed/=faces)) then
          write(*,*) 'MASK_FACE_WRONG',rank,fixture,debug,iteration,count(observed/=faces)
          error stop 'expanded-mask face ordering or values differ'
        endif
        do a=1,3
          b=modulo(a,3)+1;c=modulo(a+1,3)+1;n1=upper(b)+1;n2=upper(c)+1;pos=0
          do j=0,n2-1
          do i=0,n1-1
          do h=-1,upper(a)
            p=0;p(b)=i;p(c)=j;p(a)=h;point=origin+p
            if(periodic(a)==0) point(a)=max(0,min(g(a),point(a)))
            pos=pos+1;interface_expected(pos)=expanded_value(point)
            point=origin+p;point(a)=point(a)+1
            if(periodic(a)==0) point(a)=max(0,min(g(a),point(a)))
            interface_expected(pos)=max(interface_expected(pos),expanded_value(point))
          enddo
          enddo
          enddo
          call check(interfaces(state,int(a-1,c_int),interface_after(:pos),int(pos,c_size_t)))
          if(any(interface_after(:pos)/=interface_expected(:pos))) error stop 'upwind interface mask differs'
          changed=changed+count(interface_after(:pos)/=interface_before(:pos,a))
        enddo
        cases=cases+1
      enddo
      call check(device_state_destroy(state))
    enddo
    call MPI_Comm_free(cart,ierr);call check(ierr)
  enddo
  call MPI_Allreduce(changed,global_changed,1,MPI_INTEGER,MPI_SUM,MPI_COMM_WORLD,ierr);call check(ierr)
  if(global_changed==0) error stop 'synthetic mask did not exercise the expanded interface branch'
  write(*,'(a,3(1x,i0))') 'CPP_MASK_HALO_PASS',rank,cases,changed
  write(*,'(a,2(1x,i0))') 'CPP_MASK_ELEMENT_BYTES',rank,real_bytes
  call MPI_Finalize(ierr)
contains
  real(real_kind) function raw_value(input) result(value)
    integer,intent(in) :: input(3)
    integer :: x(3),a
    x=input
    do a=1,3
      if(periodic(a)==1) then
        x(a)=modulo(x(a),g(a))
      else
        x(a)=max(0,min(g(a),x(a)))
      endif
    enddo
    value=0
    if(iteration==1) then
      if((x(1)==6.or.x(1)==19).and.x(2)>=7.and.x(2)<=11.and.modulo(x(3),5)<2) value=1
      if((x(3)==6.or.x(3)==19).and.x(1)>=9.and.x(1)<=13.and.modulo(x(2),7)<3) value=1
    else
      if((x(2)==6.or.x(2)==19).and.x(3)>=8.and.x(3)<=14.and.modulo(x(1),6)<2) value=1
    endif
  end function
  integer(c_int8_t) function expanded_value(input) result(value)
    integer,intent(in) :: input(3)
    integer :: center(3),point(3),a,h
    center=input
    do a=1,3
      if(periodic(a)==1) center(a)=modulo(center(a),g(a))
    enddo
    value=0
    do a=1,3
      do h=-hm+1,hm
        point=center;point(a)=point(a)+h
        if(raw_value(point)>.5e0_real_kind) value=1
      enddo
    enddo
  end function
  subroutine check(status)
    integer,intent(in) :: status
    if(status==0) return
    write(*,*) 'MASK_HALO_STATUS',rank,status
    error stop 'mask halo operation failed'
  end subroutine
end program
