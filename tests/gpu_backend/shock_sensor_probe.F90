program shock_sensor_probe
  use iso_c_binding
  use astr_precision, only: real_kind,real_bytes,real_to_word
  use ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_quiet_nan
  use mpi
  use device_state
  use fluid_explicit
  use halo_transport
  use shock_sensor_cpp
  implicit none
  type(c_ptr) :: state
  type(device_view) :: view,before
  integer :: rank,ranks,local_rank,ierr,cart,split_axis=1,h(10),g(3),upper(3),origin(3),hm, &
    backend,stage,debug,unit,record_unit,axis,i,j,k,a,off,mode,coords(3),trial(3),cases=0,analytic_cases=0
  integer(c_int) :: dims(3),lower(3),higher(3),periodic(3),low(3),high(3),transport=halo_pinned
  logical :: periods(3)
  logical :: analytic_only
  real(real_kind),allocatable :: gp(:,:,:),gg(:,:,:,:,:),gr(:,:,:),p(:,:,:),gradient(:,:,:,:,:), &
    raw(:,:,:),expected_raw(:,:,:),old(:,:,:),gold_sensor(:,:,:)
  integer(c_int8_t),allocatable :: gm(:,:,:),mask(:,:,:),expected_mask(:,:,:),gold_mask(:,:,:)
  real(real_kind) :: threshold,error,maximum=0,analytic_maximum=0,nan,expanded
  character(1024) :: root,path
  character(80) :: text,suffix
  character(3) :: label
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text,status=ierr)
  if(ierr/=0) error stop 'sensor probe needs matching Open MPI launcher'
  read(text,*) local_rank
  call check(device_bind(int(local_rank,c_int)))
  call MPI_Init(ierr);call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  if(ranks/=1.and.ranks/=2) error stop 'sensor probe covers NP=1/2'
  call get_command_argument(1,root)
  analytic_only=trim(root)=='analytic'
  if(real_bytes/=8.and..not.analytic_only) error stop 'frozen sensor files are FP64; select analytic for native components'
  if(ranks==2) then
    call get_command_argument(2,text);read(text,*) split_axis
    if(split_axis<1.or.split_axis>3) error stop 'split axis must be 1/2/3'
  endif
  call get_command_argument(3,text)
  if(trim(text)=='device') then
    transport=halo_device
  elseif(len_trim(text)>0.and.trim(text)/='pinned') then
    error stop 'sensor transport must be pinned/device'
  endif
  write(path,'("observations.rank",i8.8,".bin")') rank
  open(newunit=record_unit,file=trim(path),access='stream',form='unformatted',status='new')
  write(record_unit) real_bytes,rank,split_axis
  nan=ieee_value(0.e0_real_kind,ieee_quiet_nan)
  mode=-1
  do backend=1,merge(0,2,analytic_only)
    label='cpu'
    if(backend==2) label='gpu'
    do stage=1,3
      write(suffix,'(a,i2.2,a)') '.rk',stage,'.bin'
      path=trim(root)//'/'//label//trim(suffix)
      open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='old')
      read(unit) h;g=h(:3);hm=h(4);read(unit) threshold
      allocate(gp(-hm:g(1)+hm,-hm:g(2)+hm,-hm:g(3)+hm),gg(0:g(1),0:g(2),0:g(3),3,3), &
        gr(0:g(1),0:g(2),0:g(3)),gm(0:g(1),0:g(2),0:g(3)))
      read(unit) gp,gg,gr,gm;close(unit)
      call local_arrays()
      p=gp(origin(1)-hm:origin(1)+upper(1)+hm,origin(2)-hm:origin(2)+upper(2)+hm, &
        origin(3)-hm:origin(3)+upper(3)+hm)
      gradient=gg(origin(1):origin(1)+upper(1),origin(2):origin(2)+upper(2),origin(3):origin(3)+upper(3),:,:)
      expected_raw=gr(origin(1):origin(1)+upper(1),origin(2):origin(2)+upper(2),origin(3):origin(3)+upper(3))
      expected_mask=gm(origin(1):origin(1)+upper(1),origin(2):origin(2)+upper(2),origin(3):origin(3)+upper(3))
      do debug=0,1
        call setup(.true.)
        call evaluate()
        error=maxval(abs(raw(0:upper(1),0:upper(2),0:upper(3))-expected_raw))/ &
          max(1.e0_real_kind,maxval(abs(expected_raw)))
        if(.not.(error<=2.e-10_real_kind).or.any(mask/=expected_mask)) then
          write(*,*) label,stage,rank,error,count(mask/=expected_mask)
          error stop 'sensor differs from frozen original implementation'
        endif
        maximum=max(maximum,error);cases=cases+1
        call teardown()
      enddo
      deallocate(gp,gg,gr,gm);call release_arrays()
    enddo
  enddo
  g=16;hm=5;backend=0;stage=0
  call local_arrays()
  do mode=0,4
    p=nan;gradient=0
    do k=-hm,upper(3)+hm
    do j=-hm,upper(2)+hm
    do i=-hm,upper(1)+hm
      coords=origin+[i,j,k]
      if(all(coords>=0).and.all(coords<=g)) p(i,j,k)=2.e0_real_kind+real(sum(coords*coords),real_kind)
    enddo
    enddo
    enddo
    gradient(:,:,:,1,1)=merge(-1.e0_real_kind,1.e0_real_kind,mode==0)
    threshold=.03e0_real_kind
    if(mode==2) then
      gradient=0;gradient(:,:,:,1,2)=1;gradient(:,:,:,2,1)=-1
      threshold=0
    elseif(mode==3) then
      where(ieee_is_finite(p)) p=3.e0_real_kind
      threshold=0
    elseif(mode==4) then
      do a=1,3
        do axis=1,3
          gradient(:,:,:,axis,a)=real(3*(axis-1)+a,real_kind)
        enddo
      enddo
    endif
    do k=0,upper(3)
    do j=0,upper(2)
    do i=0,upper(1)
      coords=origin+[i,j,k]
      expected_raw(i,j,k)=polynomial_sensor(coords)
      expanded=0
      do a=1,3
        do off=-hm+1,hm
          trial=coords;trial(a)=min(g(a),max(0,trial(a)+off))
          expanded=max(expanded,polynomial_sensor(trial))
        enddo
      enddo
      expected_mask(i,j,k)=merge(1_c_int8_t,0_c_int8_t,expanded>threshold)
    enddo
    enddo
    enddo
    do debug=0,1
      call setup(.false.)
      call evaluate()
      error=maxval(abs(raw(0:upper(1),0:upper(2),0:upper(3))-expected_raw))
      if(.not.ieee_is_finite(error).or.any(mask/=expected_mask)) error stop 'analytic finite sensor/byte mask mismatch'
      if(real_bytes==8.and..not.(error<=2.e-10_real_kind)) error stop 'analytic FP64 sensor criterion'
      analytic_maximum=max(analytic_maximum,error);analytic_cases=analytic_cases+1
      call teardown()
    enddo
  enddo
  call threshold_ties()
  call release_arrays()
  write(*,'(a,2(1x,i0),1x,es24.16e3)') 'CPP_SENSOR_FROZEN_PASS',rank,cases,maximum
  write(*,'(a,2(1x,i0),1x,es24.16e3)') 'CPP_SENSOR_ANALYTIC_PASS',rank,analytic_cases,analytic_maximum
  write(*,'(a,2(1x,i0))') 'CPP_SENSOR_ELEMENT_BYTES',rank,real_bytes
  close(record_unit)
  call MPI_Finalize(ierr)
contains
  subroutine local_arrays()
    upper=g;origin=0
    if(ranks==2) then
      upper(split_axis)=g(split_axis)/2;origin(split_axis)=rank*upper(split_axis)
    endif
    allocate(p(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm), &
      gradient(0:upper(1),0:upper(2),0:upper(3),3,3))
    allocate(raw(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm), &
      old(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm))
    allocate(expected_raw(0:upper(1),0:upper(2),0:upper(3)), &
      mask(0:upper(1),0:upper(2),0:upper(3)),expected_mask(0:upper(1),0:upper(2),0:upper(3)))
    allocate(gold_sensor,mold=expected_raw);allocate(gold_mask,mold=mask)
  end subroutine
  subroutine release_arrays()
    deallocate(p,gradient,raw,old,expected_raw,mask,expected_mask,gold_sensor,gold_mask)
  end subroutine
  subroutine setup(is_periodic)
    logical,intent(in) :: is_periodic
    periods=is_periodic
    if(is_periodic) periods=h(5:7)==0.and.h(8:10)==0
    periodic=merge(1,0,periods);dims=1;dims(split_axis)=ranks
    call MPI_Cart_create(MPI_COMM_WORLD,3,dims,periods,.false.,cart,ierr);call check(ierr)
    do axis=1,3
      call MPI_Cart_shift(cart,axis-1,1,lower(axis),higher(axis),ierr);call check(ierr)
    enddo
    low=0;high=0
    if(is_periodic) then
      low=merge(h(5:7),0,origin==0);high=merge(h(8:10),0,origin+upper==g)
    else
      low=merge(1,0,origin==0);high=merge(1,0,origin+upper==g)
    endif
    call check(device_state_create(int(local_rank,c_int),upper(1),upper(2),upper(3),hm,int(debug,c_int),state))
    call check(fluid_prepare(state,1_c_int))
    if(device_state_view(state,device_shock_sensor,view)/=device_not_allocated) error stop 'sensor allocated without use'
    call check(shock_prepare_cpp(state));call check(device_state_view(state,device_shock_sensor,before))
    call check(shock_prepare_cpp(state));call check(device_state_view(state,device_shock_sensor,view))
    if(view%element_bytes/=real_bytes) error stop 'sensor storage width differs from the numerical kind'
    if(.not.c_associated(view%data,before%data)) error stop 'sensor preparation changed device buffer'
    call check(device_transport_prepare(state,int(cart,c_int),lower,higher,dims,1_c_int,transport,0_c_int,periodic))
    call check(device_state_upload(state,device_pressure,p,bytes(size(p))))
    call check(device_state_upload(state,device_dvelocity,gradient,bytes(size(gradient))))
    old=nan
    call check(device_state_upload(state,device_shock_sensor,old,bytes(size(old))))
  end subroutine
  subroutine evaluate()
    call check(shock_raw_cpp(state,low,high))
    call check(device_state_download(state,device_shock_sensor,old,bytes(size(old))))
    do axis=0,2
      call check(device_exchange_axis(state,device_shock_sensor,int(axis,c_int),0_c_int))
    enddo
    call check(shock_expand_cpp(state,low,high,threshold))
    call check(device_state_download(state,device_shock_sensor,raw,bytes(size(raw))))
    call check(shock_mask_download_cpp(state,mask,int(size(mask),c_size_t)))
    if(.not.all(ieee_is_finite(raw(0:upper(1),0:upper(2),0:upper(3))))) error stop 'nonfinite raw sensor'
    if(any(real_to_word(raw(0:upper(1),0:upper(2),0:upper(3)))/= &
      real_to_word(old(0:upper(1),0:upper(2),0:upper(3))))) &
      error stop 'halo-only exchange changed physical sensor values'
    if(debug==0) then
      gold_sensor=raw(0:upper(1),0:upper(2),0:upper(3));gold_mask=mask
    elseif(any(real_to_word(gold_sensor)/=real_to_word(raw(0:upper(1),0:upper(2),0:upper(3)))).or. &
      any(gold_mask/=mask)) then
      error stop 'sensor synchronization changed result words'
    endif
    write(record_unit) mode,backend,stage,debug,upper,origin,raw(0:upper(1),0:upper(2),0:upper(3)),mask
  end subroutine
  subroutine threshold_ties()
    integer :: trial
    real(real_kind) :: boundary
    do debug=0,1
      call setup(.false.)
      old=.5_real_kind
      call check(device_state_upload(state,device_shock_sensor,old,bytes(size(old))))
      do trial=-1,1
        boundary=.5_real_kind
        if(trial/=0) boundary=nearest(boundary,real(trial,real_kind))
        call check(shock_expand_cpp(state,low,high,boundary))
        call check(shock_mask_download_cpp(state,mask,int(size(mask),c_size_t)))
        if(any(mask/=merge(1_c_int8_t,0_c_int8_t,trial<0))) error stop 'sensor strict native threshold tie'
        write(record_unit) 5,0,trial,debug,upper,origin,old(0:upper(1),0:upper(2),0:upper(3)),mask
      enddo
      call teardown()
    enddo
    write(*,'(a,2(1x,i0))') 'CPP_SENSOR_THRESHOLD_EXACT',rank,6
  end subroutine
  subroutine teardown()
    call check(device_state_destroy(state));call MPI_Comm_free(cart,ierr);call check(ierr)
  end subroutine
  real(real_kind) function polynomial_sensor(position) result(value)
    integer,intent(in) :: position(3)
    real(real_kind) :: pressure,numerator,denominator
    integer :: d
    value=0
    if(mode==2.or.mode==3) return
    pressure=2.e0_real_kind+real(sum(position*position),real_kind)
    do d=1,3
      numerator=2;denominator=4*pressure+2
      if(position(d)==0) then
        numerator=1;denominator=4*pressure+1
      elseif(position(d)==g(d)) then
        numerator=2*g(d)-1;denominator=4*pressure-2*g(d)+1
      endif
      value=max(value,numerator/denominator)
    enddo
    ! The dense tensor has trace=15 and curl=(2,-4,2): 225/(225+24)=75/83.
    if(mode==4) value=value*(75.e0_real_kind/83.e0_real_kind)
  end function
  integer(c_size_t) function bytes(n)
    integer,intent(in) :: n
    bytes=int(n,c_size_t)*real_bytes
  end function
  subroutine check(status)
    integer,intent(in) :: status
    if(status==0) return
    write(*,*) 'SENSOR_STATUS',rank,backend,stage,mode,debug,status
    error stop 'sensor component operation failed'
  end subroutine
end program
