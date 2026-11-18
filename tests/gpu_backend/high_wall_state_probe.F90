program high_wall_state_probe
  use iso_c_binding
  use mpi
  use ieee_arithmetic, only: ieee_is_finite
  use device_state
  use fluid_explicit, only: fluid_prepare,fluid_rhs_prepare
  use boundary_state_cpp, only: wall41_evolved_density_state,wall_adiabatic_evolved_state
  implicit none
  integer :: rank,ranks,ierr,unit,header(5),cache_header(4),dims(3),h,debug,reference,side,c,j,cases=0
  real(c_double),parameter :: gm=1.4d0,ma=.3d0,c2=gm*ma**2,c6=1.d0/(gm-1.d0)
  real(c_double) :: maximum=0,error,analytic=0,line(0:7),condition
  real(c_double),parameter :: weights(0:7)=[-1089.d0,2940.d0,-4410.d0,4900.d0,-3675.d0,1764.d0,-490.d0,60.d0]/420.d0
  integer :: kind,s,i,k
  real(c_double),allocatable :: initial(:,:,:,:),expected(:,:,:,:),actual(:,:,:,:),cache(:,:,:,:), &
    expected_cache(:,:,:,:),rhs(:,:,:,:)
  type(c_ptr) :: state
  character(1024) :: root,path
  character(3) :: label
  character(16) :: argument
  call check(device_bind(0_c_int))
  call MPI_Init(ierr);call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  if(ranks/=1) error stop 'high-wall state oracle is the frozen NP=1 trajectory'
  call get_command_argument(1,root)
  call get_command_argument(2,argument)
  kind=41
  if(len_trim(argument)>0) read(argument,*) kind
  if(kind/=41.and.kind/=42.and.kind/=421) error stop 'state probe kind must be 41,42,421'
  do debug=0,1
    do reference=1,2
      label='cpu'
      if(reference==2) label='gpu'
      path=trim(root)//'/'//label//'/outdat/rhs.initialized.step00000000.rk01.rank00000000.bin'
      open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='old');read(unit) header
      dims=header(1:3);h=header(4)
      if(header(5)/=5.or.any(dims/=16).or.h/=5) error stop 'high wall reference layout'
      allocate(initial(-h:dims(1)+h,-h:dims(2)+h,-h:dims(3)+h,5))
      allocate(actual,mold=initial)
      allocate(expected,mold=initial)
      allocate(cache(-h:dims(1)+h,-h:dims(2)+h,-h:dims(3)+h,6))
      allocate(expected_cache,mold=cache)
      allocate(rhs(0:dims(1),0:dims(2),0:dims(3),5))
      read(unit) initial;close(unit)
      path=trim(root)//'/'//label//'/outdat/rhs.pre_rhs.step00000000.rk01.rank00000000.bin'
      open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='old');read(unit) header
      read(unit) expected;close(unit)
      path=trim(root)//'/'//label//'/outdat/rhs.pre_rhs_primitives.step00000000.rk01.rank00000000.bin'
      open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='old');read(unit) cache_header
      if(any(cache_header/=[dims,h])) error stop 'high wall primitive layout'
      read(unit) expected_cache;close(unit)
      cache=expected_cache
      cache(0:dims(1),0,0:dims(3),:)=77
      cache(0:dims(1),dims(2),0:dims(3),:)=77
      call check(device_state_create(0_c_int,dims(1),dims(2),dims(3),h,int(debug,c_int),state))
      call check(fluid_prepare(state,0_c_int));call check(fluid_rhs_prepare(state,0_c_int))
      call check(device_state_upload(state,device_q,initial,bytes(size(initial))))
      call check(device_state_upload(state,device_rho,cache(:,:,:,1),bytes(size(cache(:,:,:,1)))))
      call check(device_state_upload(state,device_velocity,cache(:,:,:,2:4),bytes(size(cache(:,:,:,2:4)))))
      call check(device_state_upload(state,device_pressure,cache(:,:,:,5),bytes(size(cache(:,:,:,5)))))
      call check(device_state_upload(state,device_temperature,cache(:,:,:,6),bytes(size(cache(:,:,:,6)))))
      rhs=7;call check(device_state_upload(state,device_rhs,rhs,bytes(size(rhs))))
      do side=0,1
        if(kind==41) then
          call check(wall41_evolved_density_state(state,1_c_int,int(side,c_int),1.d0,c2,c6))
        else
          call check(wall_adiabatic_evolved_state(state,1_c_int,int(side,c_int),int(merge(1,0,kind==421),c_int),c2,c6))
        endif
      enddo
      call check(device_state_download(state,device_q,actual,bytes(size(actual))))
      call check(device_state_download(state,device_rho,cache(:,:,:,1),bytes(size(cache(:,:,:,1)))))
      call check(device_state_download(state,device_velocity,cache(:,:,:,2:4),bytes(size(cache(:,:,:,2:4)))))
      call check(device_state_download(state,device_pressure,cache(:,:,:,5),bytes(size(cache(:,:,:,5)))))
      call check(device_state_download(state,device_temperature,cache(:,:,:,6),bytes(size(cache(:,:,:,6)))))
      call check(device_state_download(state,device_rhs,rhs,bytes(size(rhs))))
      do c=1,5
        call compare(actual(0:dims(1),0:dims(2),0:dims(3),c),expected(0:dims(1),0:dims(2),0:dims(3),c))
      enddo
      do c=1,6
        call compare(cache(0:dims(1),0:dims(2),0:dims(3),c),expected_cache(0:dims(1),0:dims(2),0:dims(3),c))
      enddo
      if(any(rhs/=7)) error stop 'high wall state changed RHS'
      if(any(transfer(actual(:,:,:,1),[0_c_int64_t])/=transfer(initial(:,:,:,1),[0_c_int64_t]))) &
        error stop 'high wall state changed authoritative density bits'
      do j=-h,dims(2)+h
        if(j==0.or.j==dims(2)) cycle
        if(any(transfer(actual(:,j,:,:),[0_c_int64_t])/=transfer(initial(:,j,:,:),[0_c_int64_t]))) &
          error stop 'high wall state wrote outside its faces'
      enddo
      do side=0,1
        j=side*dims(2)
        error=maxval(abs(cache(0:dims(1),j,0:dims(3),1)-initial(0:dims(1),j,0:dims(3),1)))
        if(kind==421) then
          error=max(error,maxval(abs(cache(0:dims(1),j,0:dims(3),3))))
        else
          error=max(error,maxval(abs(cache(0:dims(1),j,0:dims(3),2:4))))
        endif
        if(kind==41) then
          error=max(error,maxval(abs(cache(0:dims(1),j,0:dims(3),6)-1)))
        else
          do k=0,dims(3)
          do i=0,dims(1)
            do s=0,7
              line(s)=cache(i,j+(1-2*side)*s,k,6)
            enddo
            condition=abs(sum(weights*(line-line(0))))/max(1.d0,maxval(abs(line)))
            error=max(error,condition)
            if(kind==421) then
              do c=2,4,2
                do s=0,7
                  line(s)=cache(i,j+(1-2*side)*s,k,c)
                enddo
                condition=abs(sum(weights*(line-line(0))))/max(1.d0,maxval(abs(line)))
                error=max(error,condition)
              enddo
            endif
          enddo
          enddo
        endif
        analytic=max(analytic,error)
        if(.not.(error<=2.d-10)) error stop 'high wall density/velocity/temperature identity'
      enddo
      call check(device_state_destroy(state));cases=cases+1
      deallocate(initial,actual,expected,cache,expected_cache,rhs)
    enddo
  enddo
  print *, 'CPP_HIGH_WALL_STATE_PASS ',kind,cases,maximum,analytic
  call MPI_Finalize(ierr)
contains
  integer(c_size_t) function bytes(n)
    integer,intent(in) :: n
    bytes=int(n,c_size_t)*c_sizeof(0.d0)
  end function
  subroutine compare(a,b)
    real(c_double),intent(in) :: a(:,:,:),b(:,:,:)
    error=maxval(abs(a-b))/max(1.d0,maxval(abs(b)))
    if(.not.all(ieee_is_finite(a)).or..not.(error<=2.d-10)) error stop 'high wall state reference mismatch'
    maximum=max(maximum,error)
  end subroutine
  subroutine check(status)
    integer,intent(in) :: status
    if(status/=0) then
      print *, 'HIGH_WALL_STATE_ERROR ',status
      error stop 'high wall state API'
    endif
  end subroutine
end program
