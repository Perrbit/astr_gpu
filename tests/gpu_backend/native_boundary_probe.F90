program native_boundary_probe
  use iso_c_binding, only: c_ptr,c_int,c_size_t
  use mpi
  use astr_precision, only: real_kind,real_bytes,real_to_word
  use astr_precision_mpi, only: mpi_real_type
  use ieee_arithmetic, only: ieee_is_finite
  use device_state
  use fluid_explicit, only: fluid_prepare,fluid_rhs_prepare,fluid_check
  use boundary_state_cpp
  implicit none
  integer,parameter :: global_n=16,hm=5,codes(4)=[41,42,50,60]
  real(real_kind),parameter :: gamma=2._real_kind,mach=2._real_kind,c2=8._real_kind,c6=1._real_kind
  real(real_kind),parameter :: velocity(3)=[.125_real_kind,.25_real_kind,-.125_real_kind]
  type(c_ptr) :: state
  type(device_view) :: view
  real(real_kind),allocatable :: q(:,:,:,:),p(:,:,:,:),met(:,:,:,:,:),expected_q(:,:,:,:),expected_p(:,:,:,:), &
    actual_q(:,:,:,:),actual_p(:,:,:,:),rhs(:,:,:,:),expected_rhs(:,:,:,:),gold(:,:,:,:)
  real(real_kind) :: pressure,den,u(3),normal(3),norm,temperature,error,maximum=0,local_max,observed, &
    expected_global,observed_global,mach_error,upper_max=0,factor
  logical,allocatable :: face(:,:,:)
  logical :: owner
  integer :: ierr,rank,ranks,local_rank,split_axis,upper(3),origin(3),axis,side,kind,debug,i,j,k,c, &
    point(3),distance,last,cases=0,unit
  character(1024) :: arg,path
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',arg,status=ierr)
  if(ierr/=0) error stop 'native boundary probe requires Open MPI local rank'
  read(arg,*) local_rank
  call check(device_bind(int(local_rank,c_int)),'device binding')
  call MPI_Init(ierr);call check(ierr,'MPI initialization')
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call check(ierr,'rank')
  call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr);call check(ierr,'size')
  call get_command_argument(1,arg);read(arg,*) split_axis
  if(split_axis<1.or.split_axis>3.or.(ranks/=1.and.ranks/=2)) error stop 'native boundary decomposition'
  upper=global_n;upper(split_axis)=global_n/ranks;origin=0;origin(split_axis)=rank*upper(split_axis)
  allocate(q(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,5), &
    p(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,6), &
    met(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,3,3), &
    rhs(0:upper(1),0:upper(2),0:upper(3),5),face(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm))
  allocate(expected_q,mold=q);allocate(expected_p,mold=p);allocate(actual_q,mold=q);allocate(actual_p,mold=p)
  allocate(gold,mold=q);allocate(expected_rhs,mold=rhs)
  met=0;met(:,:,:,1,1)=1;met(:,:,:,2,1)=1;met(:,:,:,2,2)=2;met(:,:,:,2,3)=2;met(:,:,:,3,3)=1
  write(path,'("boundaries.rank",i8.8,".bin")') rank
  open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='new')
  write(unit) real_bytes,rank,split_axis,upper,origin
  do kind=1,4
  do axis=0,2
    if(codes(kind)==42.and.axis==2) cycle
  do side=0,1
    owner=merge(origin(axis+1)+upper(axis+1)==global_n,origin(axis+1)==0,side==1)
    if(.not.owner) cycle
    last=merge(upper(axis+1),0,side==1);face=.false.
    do k=-hm,upper(3)+hm
    do j=-hm,upper(2)+hm
    do i=-hm,upper(1)+hm
      point=[i,j,k];distance=abs(point(axis+1)-last)
      pressure=.25_real_kind+real(distance*distance,real_kind)/1024
      den=pressure*c2
      factor=1
      if(point(axis+1)==last.and.all(point>=0).and.all(point<=upper)) then
        face(i,j,k)=.true.;factor=2
      endif
      call make_state(den*factor,velocity,pressure*factor,q(i,j,k,:),p(i,j,k,:))
    enddo
    enddo
    enddo
    expected_q=q;expected_p=p
    normal=met(0,0,0,axis+1,:);norm=sqrt(sum(normal*normal));normal=normal/norm
    u=velocity
    if(kind<=2) u=0
    if(codes(kind)==60) u=u-sum(u*normal)*normal
    do k=0,upper(3)
    do j=0,upper(2)
    do i=0,upper(1)
      if(.not.face(i,j,k)) cycle
      call make_state(2._real_kind,u,.25_real_kind,expected_q(i,j,k,:),expected_p(i,j,k,:))
    enddo
    enddo
    enddo
    do debug=0,1
      call check(device_state_create(int(local_rank,c_int),upper(1),upper(2),upper(3),hm,int(debug,c_int),state),'context')
      call check(fluid_prepare(state,0_c_int),'primitives');call check(fluid_rhs_prepare(state,0_c_int),'RHS')
      call check(device_state_upload(state,device_q,q,bytes(size(q))),'q')
      call check(device_state_upload(state,device_rho,p(:,:,:,1),bytes(size(face))),'density')
      call check(device_state_upload(state,device_velocity,p(:,:,:,2:4),bytes(3*size(face))),'velocity')
      call check(device_state_upload(state,device_pressure,p(:,:,:,5),bytes(size(face))),'pressure')
      call check(device_state_upload(state,device_temperature,p(:,:,:,6),bytes(size(face))),'temperature')
      call check(device_state_upload(state,device_metric,met,bytes(size(met))),'metric')
      rhs=7;expected_rhs=7
      if(kind>=3) then
        do c=1,5
          where(face(0:upper(1),0:upper(2),0:upper(3))) expected_rhs(:,:,:,c)=0
        enddo
      endif
      call check(device_state_upload(state,device_rhs,rhs,bytes(size(rhs))),'RHS seed')
      select case(codes(kind))
      case(41);call check(wall41_basic_state(state,axis,side,1._real_kind,c2,c6),'isothermal wall')
      case(42);call check(wall42_basic_state(state,axis,side,c2,c6),'adiabatic wall')
      case(50);call check(zeroextrap_basic(state,axis,side,c2,c6),'zero-gradient state')
      case(60);call check(symmetry_basic(state,axis,side,c2,c6),'symmetry state')
      end select
      call check(fluid_check(state),'finite/positive status')
      call check(device_state_download(state,device_q,actual_q,bytes(size(q))),'q observation')
      call check(device_state_download(state,device_rho,actual_p(:,:,:,1),bytes(size(face))),'density observation')
      call check(device_state_download(state,device_velocity,actual_p(:,:,:,2:4),bytes(3*size(face))),'velocity observation')
      call check(device_state_download(state,device_pressure,actual_p(:,:,:,5),bytes(size(face))),'pressure observation')
      call check(device_state_download(state,device_temperature,actual_p(:,:,:,6),bytes(size(face))),'temperature observation')
      call check(device_state_download(state,device_rhs,rhs,bytes(size(rhs))),'RHS observation')
      if(.not.all(ieee_is_finite(actual_q)).or..not.all(ieee_is_finite(actual_p))) error stop 'native boundary nonfinite state'
      error=0
      do c=1,5
        error=max(error,maxval(abs(actual_q(:,:,:,c)-expected_q(:,:,:,c)))/max(1._real_kind,maxval(abs(expected_q(:,:,:,c)))))
      enddo
      do c=1,6
        error=max(error,maxval(abs(actual_p(:,:,:,c)-expected_p(:,:,:,c)))/max(1._real_kind,maxval(abs(expected_p(:,:,:,c)))))
      enddo
      if(real_bytes==8.and.error>2.e-10_real_kind) error stop 'native boundary FP64 polynomial reference'
      if(any(real_to_word(rhs)/=real_to_word(expected_rhs))) error stop 'native boundary RHS footprint'
      do k=-hm,upper(3)+hm
      do j=-hm,upper(2)+hm
      do i=-hm,upper(1)+hm
        if(face(i,j,k)) cycle
        if(any(real_to_word(actual_q(i,j,k,:))/=real_to_word(q(i,j,k,:))).or. &
          any(real_to_word(actual_p(i,j,k,:))/=real_to_word(p(i,j,k,:)))) error stop 'native boundary exterior footprint'
      enddo
      enddo
      enddo
      if(debug==0) then
        gold=actual_q
      elseif(any(real_to_word(gold)/=real_to_word(actual_q))) then
        error stop 'native boundary sync words'
      endif
      write(unit) codes(kind),axis,side,debug,actual_q,actual_p
      write(*,'(a,5(1x,i0),1x,es24.16e3)') 'CPP_NATIVE_BOUNDARY',rank,codes(kind),axis,side,debug,error
      maximum=max(maximum,error);cases=cases+1
      call check(device_state_destroy(state),'release context')
    enddo
  enddo
  enddo
  enddo
  ! A varying oblique normal velocity tests the real native maximum reduction.
  do debug=0,1
    call check(device_state_create(int(local_rank,c_int),upper(1),upper(2),upper(3),hm,int(debug,c_int),state),'Mach context')
    call check(fluid_prepare(state,0_c_int),'Mach primitives')
    do k=-hm,upper(3)+hm
    do j=-hm,upper(2)+hm
    do i=-hm,upper(1)+hm
      p(i,j,k,2:4)=velocity+real(origin(1)+i+origin(3)+k,real_kind)/128
      p(i,j,k,6)=1
    enddo
    enddo
    enddo
    call check(device_state_upload(state,device_velocity,p(:,:,:,2:4),bytes(3*size(face))),'Mach velocity')
    call check(device_state_upload(state,device_temperature,p(:,:,:,6),bytes(size(face))),'Mach temperature')
    call check(device_state_upload(state,device_metric,met,bytes(size(met))),'Mach metric')
    local_max=0;observed=0
    if(origin(2)+upper(2)==global_n) then
      do k=0,upper(3)
      do i=0,upper(1)
        local_max=max(local_max,sum(met(i,upper(2),k,2,:)*p(i,upper(2),k,2:4))**2/ &
          sum(met(i,upper(2),k,2,:)**2)*mach*mach)
      enddo
      enddo
      call check(nscbc_top_mach2(state,mach,observed),'native top normal Mach maximum')
    endif
    call MPI_Allreduce(local_max,expected_global,1,mpi_real_type,MPI_MAX,MPI_COMM_WORLD,ierr);call check(ierr,'expected maximum')
    call MPI_Allreduce(observed,observed_global,1,mpi_real_type,MPI_MAX,MPI_COMM_WORLD,ierr);call check(ierr,'observed maximum')
    mach_error=abs(observed_global-expected_global)/max(1._real_kind,expected_global)
    if(.not.ieee_is_finite(mach_error).or.observed_global<0) error stop 'nonfinite native Mach reduction'
    if(real_bytes==8.and.mach_error>2.e-10_real_kind) error stop 'FP64 native Mach reduction'
    upper_max=max(upper_max,mach_error)
    write(*,'(a,2(1x,i0),3(1x,es24.16e3))') 'CPP_NATIVE_MACH',rank,debug,observed_global,expected_global,mach_error
    call check(device_state_destroy(state),'release Mach context')
  enddo
  close(unit)
  write(*,'(a,3(1x,i0),2(1x,es24.16e3))') 'CPP_NATIVE_BOUNDARY_PASS',rank,real_bytes,cases,maximum,upper_max
  call MPI_Finalize(ierr)
contains
  subroutine make_state(rho,vel,prs,conservative,primitive)
    real(real_kind),intent(in) :: rho,vel(3),prs
    real(real_kind),intent(out) :: conservative(5),primitive(6)
    conservative=[rho,rho*vel,prs*c6+.5_real_kind*rho*sum(vel*vel)]
    primitive=[rho,vel,prs,prs/rho*c2]
  end subroutine
  integer(c_size_t) function bytes(count)
    integer,intent(in) :: count
    bytes=int(count,c_size_t)*real_bytes
  end function
  subroutine check(status,label)
    integer(c_int),intent(in) :: status
    character(*),intent(in) :: label
    if(status==0) return
    write(*,*) 'CPP_NATIVE_BOUNDARY_STATUS ',label,status
    error stop 'native boundary operation'
  end subroutine
end program
