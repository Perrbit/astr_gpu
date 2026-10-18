program air5_density_padding_probe
  use iso_fortran_env, only: int64
  use commvar, only: im,jm,km,hm,numq,num_species,num_modequ,ndims,turbmode,lcomb
  use commarray, only: allocommarray,rho
  implicit none
  integer :: i,j,k,outside,edges,defined,status
  integer(int64) :: bits
  character(16) :: value

  call get_environment_variable('MALLOC_PERTURB_',value,status=status)
  if(status/=0.or.trim(value)/='165') error stop 'probe requires MALLOC_PERTURB_=165'
  call get_command_argument(1,value)
  lcomb=trim(value)=='air5'
  im=8; jm=7; km=6
  numq=11; num_species=5; num_modequ=1; ndims=3; turbmode='none'
  call allocommarray()
  edges=0; defined=0
  do k=-hm,km+hm
    do j=-hm,jm+hm
      do i=-hm,im+hm
        outside=count([i<0.or.i>im,j<0.or.j>jm,k<0.or.k>km])
        bits=transfer(rho(i,j,k),bits)
        if(lcomb.and.outside>=2) then
          if(bits/=0_int64) error stop 'AIR5 density edge/corner padding is not positive zero'
          edges=edges+1
        else
          if(bits/=int(z'5A5A5A5A5A5A5A5A',int64)) &
            error stop 'allocation changed density outside the approved padding'
          defined=defined+1
        endif
      enddo
    enddo
  enddo
  write(*,'(A,L1,2(1X,I0))') 'DENSITY_PADDING_PASS air5=',lcomb,edges,defined
end program
