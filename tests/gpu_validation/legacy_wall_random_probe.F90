program legacy_wall_random_probe
  use wall_blowing_random, only: sample_legacy_wall_velocity
  implicit none
  integer,parameter :: im=64,km=24,calls=3
  real(8),parameter :: pi=3.1415926535897932384626433832795d0
  real(8) :: xc(0:im,0:km),zc(0:im,0:km),value(0:im,0:km)
  real(8) :: expected(0:im,0:km,calls),theta,fx,gz,rfluc
  integer :: i,k,m,c,rank,seed_size
  integer,allocatable :: seed(:),expected_seed(:),actual_seed(:)
  character(len=16) :: argument

  call get_command_argument(1,argument)
  read(argument,*) rank
  do k=0,km
    do i=0,im
      xc(i,k)=dble(i)*80.d0/dble(im)
      zc(i,k)=dble(k)*90.d0/dble(km)
    enddo
  enddo

  ! Independent call-sequence reference, including draws at zero envelope nodes.
  call random_seed(size=seed_size)
  allocate(seed(seed_size),expected_seed(seed_size),actual_seed(seed_size))
  seed=1
  call random_seed(put=seed)
  do m=1,15
    call random_number(rfluc)
  enddo
  seed=rank+1
  call random_seed(put=seed)
  do c=1,calls
    do k=0,km
      do i=0,im
        if(xc(i,k)<=40.d0 .and. xc(i,k)>=20.d0) then
          theta=2.d0*pi*(xc(i,k)-20.d0)/20.d0
          fx=4.d0*sin(theta)*(1.d0-cos(theta))*(1.d0/sqrt(27.d0))
          gz=sin(2.d0*3*pi*(zc(i,k)/90.d0))
        elseif(xc(i,k)<=60.d0 .and. xc(i,k)>=40.d0) then
          theta=2.d0*pi*(xc(i,k)-40.d0)/20.d0
          fx=4.d0*sin(theta)*(1.d0-cos(theta))*(1.d0/sqrt(27.d0))
          gz=sin(2.d0*3*pi*(zc(i,k)/90.d0)+0.5d0*pi)
        else
          expected(i,k,c)=0.d0
          cycle
        endif
        call random_number(rfluc)
        rfluc=(rfluc*2.d0-1.d0)*0.1d0
        expected(i,k,c)=0.12d0*1.d0*fx*gz*1.d0*(1.d0+rfluc)
      enddo
    enddo
  enddo
  call random_seed(get=expected_seed)

  do c=1,calls
    call sample_legacy_wall_velocity(rank,xc,zc,90.d0,1.d0,0.12d0, &
                                    20.d0,40.d0,60.d0,3,value)
    if(any(value/=expected(:,:,c))) error stop 'legacy wall sequence mismatch'
  enddo
  call random_seed(get=actual_seed)
  if(any(expected_seed/=actual_seed)) error stop 'legacy wall RNG state mismatch'
  if(all(expected(:,:,1)==expected(:,:,2))) error stop 'legacy wall is not time-varying'
  if(maxval(abs(value(33:47,:)))==0.d0) error stop 'second wall segment is missing'
  if(any(value(0:15,:)/=0.d0) .or. any(value(49:im,:)/=0.d0)) &
    error stop 'wall forcing leaked outside its support'
  print '(A,I0,A)', 'legacy wall rank ',rank,': PASS (3 calls, exact values and RNG state)'
end program legacy_wall_random_probe
