program legacy_wall_random_probe
  use astr_precision, only: real_kind
  use wall_blowing_random, only: sample_legacy_wall_velocity,legacy_random_state_size, &
    get_legacy_random_state,put_legacy_random_state,legacy_random_runtime_signature
  use iso_fortran_env, only: int64
  implicit none
  integer,parameter :: im=64,km=24,calls=3
  real(real_kind),parameter :: pi=3.1415926535897932384626433832795e0_real_kind
  real(real_kind) :: xc(0:im,0:km),zc(0:im,0:km),value(0:im,0:km)
  real(real_kind) :: expected(0:im,0:km,calls),theta,fx,gz,rfluc
  integer :: i,k,m,c,rank,seed_size
  integer,allocatable :: seed(:),expected_seed(:),actual_seed(:)
  integer,allocatable :: initial(:),snapshot(:),after(:)
  integer(int64) :: signature(8),again(8)
  logical :: ok
  real(real_kind) :: continued(0:im,0:km)
  character(len=16) :: argument

  call get_command_argument(1,argument)
  read(argument,*) rank
  do k=0,km
    do i=0,im
      xc(i,k)=real(i,real_kind)*80.e0_real_kind/real(im,real_kind)
      zc(i,k)=real(k,real_kind)*90.e0_real_kind/real(km,real_kind)
    enddo
  enddo

  ! Independent call-sequence reference, including draws at zero envelope nodes.
  call random_seed(size=seed_size)
  allocate(seed(seed_size),expected_seed(seed_size),actual_seed(seed_size))
  allocate(initial(legacy_random_state_size()),snapshot(legacy_random_state_size()),after(legacy_random_state_size()))
  call get_legacy_random_state(initial)
  if(any(initial/=0)) error stop 'unused wall RNG state is not canonical'
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
        if(xc(i,k)<=40.e0_real_kind .and. xc(i,k)>=20.e0_real_kind) then
          theta=2.e0_real_kind*pi*(xc(i,k)-20.e0_real_kind)/20.e0_real_kind
          fx=4.e0_real_kind*sin(theta)*(1.e0_real_kind-cos(theta))*(1.e0_real_kind/sqrt(27.e0_real_kind))
          gz=sin(2.e0_real_kind*3*pi*(zc(i,k)/90.e0_real_kind))
        elseif(xc(i,k)<=60.e0_real_kind .and. xc(i,k)>=40.e0_real_kind) then
          theta=2.e0_real_kind*pi*(xc(i,k)-40.e0_real_kind)/20.e0_real_kind
          fx=4.e0_real_kind*sin(theta)*(1.e0_real_kind-cos(theta))*(1.e0_real_kind/sqrt(27.e0_real_kind))
          gz=sin(2.e0_real_kind*3*pi*(zc(i,k)/90.e0_real_kind)+0.5e0_real_kind*pi)
        else
          expected(i,k,c)=0.e0_real_kind
          cycle
        endif
        call random_number(rfluc)
        rfluc=(rfluc*2.e0_real_kind-1.e0_real_kind)*0.1e0_real_kind
        expected(i,k,c)=0.12e0_real_kind*1.e0_real_kind*fx*gz*1.e0_real_kind*(1.e0_real_kind+rfluc)
      enddo
    enddo
  enddo
  call random_seed(get=expected_seed)

  do c=1,calls
    call sample_legacy_wall_velocity(rank,xc,zc,90.e0_real_kind,1.e0_real_kind,0.12e0_real_kind, &
                                    20.e0_real_kind,40.e0_real_kind,60.e0_real_kind,3,value)
    if(any(value/=expected(:,:,c))) error stop 'legacy wall sequence mismatch'
  enddo
  call random_seed(get=actual_seed)
  if(any(expected_seed/=actual_seed)) error stop 'legacy wall RNG state mismatch'
  if(all(expected(:,:,1)==expected(:,:,2))) error stop 'legacy wall is not time-varying'
  if(maxval(abs(value(33:47,:)))==0.e0_real_kind) error stop 'second wall segment is missing'
  if(any(value(0:15,:)/=0.e0_real_kind) .or. any(value(49:im,:)/=0.e0_real_kind)) &
    error stop 'wall forcing leaked outside its support'
  call get_legacy_random_state(snapshot)
  call legacy_random_runtime_signature(signature)
  call legacy_random_runtime_signature(again)
  call get_legacy_random_state(after)
  if(any(snapshot/=after).or.any(signature/=again)) error stop 'RNG identity probe perturbed sequence'
  call sample_legacy_wall_velocity(rank,xc,zc,90.e0_real_kind,1.e0_real_kind,0.12e0_real_kind,20.e0_real_kind,40.e0_real_kind,60.e0_real_kind,3,continued)
  call get_legacy_random_state(after)
  call put_legacy_random_state(snapshot,ok)
  if(.not.ok) error stop 'RNG restore failed'
  call sample_legacy_wall_velocity(rank,xc,zc,90.e0_real_kind,1.e0_real_kind,0.12e0_real_kind,20.e0_real_kind,40.e0_real_kind,60.e0_real_kind,3,value)
  call get_legacy_random_state(snapshot)
  if(any(value/=continued).or.any(snapshot/=after)) error stop 'RNG continuation differs'
  snapshot(1)=2
  call put_legacy_random_state(snapshot,ok)
  if(ok) error stop 'invalid RNG first-call flag admitted'
  call put_legacy_random_state(initial,ok)
  if(.not.ok) error stop 'unused RNG restore failed'
  call sample_legacy_wall_velocity(rank,xc,zc,90.e0_real_kind,1.e0_real_kind,0.12e0_real_kind,20.e0_real_kind,40.e0_real_kind,60.e0_real_kind,3,value)
  if(any(value/=expected(:,:,1))) error stop 'first-call lifecycle was not restored'
  print '(A,I0,A)', 'legacy wall rank ',rank,': PASS (exact sequence, snapshot, continuation, first call)'
end program legacy_wall_random_probe
