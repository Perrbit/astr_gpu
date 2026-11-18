program exterior_initialization_probe
  use iso_c_binding
  use device_state
  use fluid_explicit
  implicit none
  integer,parameter :: upper(3)=[3,4,5],hm=2
  type(c_ptr) :: state
  real(c_double),allocatable :: original(:,:,:,:),q(:,:,:,:),primitives(:,:,:,:),expected(:,:,:,:)
  integer(c_int) :: low(3),high(3)
  integer :: i,j,k,c,a,mask,debug,outside,cases=0,node(3)
  logical :: physical
  allocate(original(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,6), &
    primitives(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,6), &
    expected(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,6), &
    q(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,5))
  do c=1,6
  do k=-hm,upper(3)+hm
  do j=-hm,upper(2)+hm
  do i=-hm,upper(1)+hm
    original(i,j,k,c)=10000*c+100*k+10*j+i
  enddo
  enddo
  enddo
  enddo
  do debug=0,1
  do mask=0,63
    do a=1,3
      low(a)=merge(1,0,btest(mask,2*a-2));high(a)=merge(1,0,btest(mask,2*a-1))
    enddo
    call check(device_state_create(0_c_int,upper(1),upper(2),upper(3),hm,debug,state),'create')
    call check(fluid_prepare(state,0_c_int),'prepare')
    q=original(:,:,:,:5);primitives=original
    call check(device_state_upload(state,device_q,q,bytes(size(q))),'q seed')
    call check(device_state_upload(state,device_rho,primitives(:,:,:,1),bytes(size(q)/5)),'rho seed')
    call check(device_state_upload(state,device_velocity,primitives(:,:,:,2:4),bytes(3*size(q)/5)),'velocity seed')
    call check(device_state_upload(state,device_pressure,primitives(:,:,:,5),bytes(size(q)/5)),'pressure seed')
    call check(device_state_upload(state,device_temperature,primitives(:,:,:,6),bytes(size(q)/5)),'temperature seed')
    call check(fluid_initialize_exterior(state,low,high),'initialize')
    call check(device_state_download(state,device_q,q,bytes(size(q))),'q observation')
    call check(device_state_download(state,device_rho,primitives(:,:,:,1),bytes(size(q)/5)),'rho observation')
    call check(device_state_download(state,device_velocity,primitives(:,:,:,2:4),bytes(3*size(q)/5)),'velocity observation')
    call check(device_state_download(state,device_pressure,primitives(:,:,:,5),bytes(size(q)/5)),'pressure observation')
    call check(device_state_download(state,device_temperature,primitives(:,:,:,6),bytes(size(q)/5)),'temperature observation')
    expected=original
    do k=-hm,upper(3)+hm
    do j=-hm,upper(2)+hm
    do i=-hm,upper(1)+hm
      node=[i,j,k];outside=count(node<0.or.node>upper)
      physical=any((node<0.and.low/=0).or.(node>upper.and.high/=0))
      if(physical.or.outside>1) expected(i,j,k,:5)=0.d0
      if(any(transfer(q(i,j,k,:),[0_c_int64_t],5)/=transfer(expected(i,j,k,:5),[0_c_int64_t],5))) &
        error stop 'q physical/MPI ownership or positive-zero exterior definition differs'
      expected(i,j,k,:)=original(i,j,k,:)
      if(physical) expected(i,j,k,:)=0.d0
    enddo
    enddo
    enddo
    if(any(transfer(primitives,[0_c_int64_t],size(primitives))/= &
      transfer(expected,[0_c_int64_t],size(expected)))) &
      error stop 'primitive physical/MPI ownership or positive-zero exterior definition differs'
    call check(device_state_destroy(state),'destroy');cases=cases+1
  enddo
  enddo
  write(*,'(a,i0)') 'EXTERIOR_INITIALIZATION_PASS cases=',cases
contains
  integer(c_size_t) function bytes(n)
    integer,intent(in) :: n
    bytes=int(n,c_size_t)*c_sizeof(0.d0)
  end function
  subroutine check(status,label)
    integer(c_int),intent(in) :: status
    character(*),intent(in) :: label
    if(status==0) return
    print *, 'EXTERIOR_INITIALIZATION_FAILED ',label,status
    error stop 'device operation failed'
  end subroutine
end program
