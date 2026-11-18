module commvar
  use iso_fortran_env,only:real64
  implicit none
  integer,parameter :: im=0,jm=2,km=0
  integer :: rkstep=1
  real(real64) :: deltat=1e-6_real64
end module
module commarray
  use iso_fortran_env,only:real64
  implicit none
  real(real64) :: q(0:0,0:2,0:0,11),x(0:0,0:2,0:0,3),jacob(0:0,0:2,0:0),qrhs(0:0,0:2,0:0,11)
end module
module parallel
  use mpi,only:MPI_PROC_NULL
  implicit none
  integer :: mpiup=MPI_PROC_NULL,mpileft=0
end module
! Frozen constant target supplier: the actual target geometry is separately tested.
module chemistry_hbl_boundary
  use iso_fortran_env,only:real64
  implicit none
  real(real64) :: point_target(11),point_rate,point_sound
contains
  logical function air5_dynamic_top()
    air5_dynamic_top=.true.
  end function
  real(real64) function get_air5_top_rate()
    get_air5_top_rate=point_rate
  end function
  real(real64) function get_air5_top_reference_sound()
    get_air5_top_reference_sound=point_sound
  end function
  subroutine get_air5_top_target(x,state,status)
    real(real64),intent(in) :: x
    real(real64),intent(out) :: state(11)
    integer,intent(out) :: status
    state=point_target;status=0
  end subroutine
end module
#ifdef _CUDA
module commarray_gpu
  use iso_fortran_env,only:real64
  use cudafor
  implicit none
  real(real64),device :: q_d(0:0,0:2,0:0,11),x_d(0:0,0:2,0:0,3),jacob_d(0:0,0:2,0:0),qrhs_d(0:0,0:2,0:0,11)
end module
module chemistry_hbl_boundary_gpu
  use iso_fortran_env,only:real64
  use cudafor
  implicit none
  real(real64),device :: point_target_d(11)
contains
  attributes(device) subroutine get_air5_top_target_gpu(x,state,status)
    real(real64),intent(in) :: x
    real(real64),intent(out) :: state(11)
    integer,intent(out) :: status
    state=point_target_d;status=0
  end subroutine
end module
#endif
module state_reference
  use iso_fortran_env,only:real64
  use chemistry_air5_data
  use chemistry_model
  use chemistry_flow_state
  use chemistry_thermo
#ifdef _CUDA
  use cudafor
  use chemistry_flow_state_gpu
  use chemistry_thermo_gpu
  integer,device :: air5_flow_status_d
  real(real64),device :: top_normal_high_rhs_d(0:0,0:0,11)
  real(real64),device :: diagnostic_d(22)
#else
  real(real64),allocatable :: top_normal_high_rhs(:,:,:)
  integer :: budget_calls=0
#endif
  real(real64) :: diagnostic(22)
contains
#include "frozen_top_rhs.inc"
#ifndef _CUDA
  ! Assembly-only boundary: do not execute the separately blocked shared limiter.
  subroutine air5_limit_symmetric_convection(a,b,cdt,top)
    real(real64),intent(in) :: a,b,cdt
    logical,intent(in) :: top
    if(.not.top.or.a/=1.0_real64.or.b/=0.0_real64.or.cdt/=1e-6_real64)error stop 'unexpected boundary budget handoff'
    budget_calls=budget_calls+1
  end subroutine
#endif
end module
program air5_characteristic_state_reference
  use mpi
  use state_reference
  use commarray
  use chemistry_hbl_boundary
#ifdef _CUDA
  use commarray_gpu
  use chemistry_hbl_boundary_gpu
#endif
  implicit none
  real(real64),allocatable :: input(:,:),output(:,:)
  real(real64),allocatable :: diagnostics(:,:)
  real(real64) :: floor(5),plane(0:0,0:0,11)
  integer,allocatable :: statuses(:)
  integer :: n,u,row,c,status,ierr
  character(1024) :: path,result,diagnostic_path
#ifdef _CUDA
  status=cudaSetDevice(0);if(status/=0)error stop 'reference device'
#endif
  call MPI_Init(ierr);if(ierr/=MPI_SUCCESS)error stop 'reference MPI init'
  call get_command_argument(1,path);call get_command_argument(2,result)
  open(newunit=u,file=trim(path),access='stream',form='unformatted',status='old')
  read(u)n,floor;allocate(input(96,n),output(22,n),statuses(n),diagnostics(22,n));read(u)input;close(u)
  do row=1,n
    do c=1,11
      q(0,2,0,c)=input(c,row);q(0,1,0,c)=input(55+c,row);q(0,0,0,c)=input(66+c,row)
    enddo
    x=0.0_real64;x(0,2,0,2)=input(89,row);jacob=input(95,row);qrhs=-91.0_real64
    point_target=input(12:22,row);point_rate=input(92,row);point_sound=input(93,row);diagnostic=0.0_real64
#ifdef _CUDA
    call configure_air5_species_cache_gpu(input(94,row)/=0.0_real64)
    q_d=q;x_d=x;jacob_d=jacob;qrhs_d=qrhs;top_normal_high_rhs_d=-77.0_real64
    point_target_d=point_target;air5_flow_status_d=0;diagnostic_d=0.0_real64
    call air5_characteristic_top_rhs_kernel<<<1,128>>>(0,2,0,4,4,3,0,.false.,0,point_rate,point_sound)
    status=cudaDeviceSynchronize();if(status/=0)error stop 'original top kernel launch'
    statuses(row)=air5_flow_status_d;plane=top_normal_high_rhs_d;qrhs=qrhs_d
    output(1:11,row)=plane(0,0,:);diagnostic=diagnostic_d
#else
    call configure_air5_species_cache(input(94,row)/=0.0_real64)
    if(allocated(top_normal_high_rhs))top_normal_high_rhs=-77.0_real64
    call air5_characteristic_top_rhs();statuses(row)=0
    output(1:11,row)=top_normal_high_rhs(0,0,:)
    if(budget_calls/=row)error stop 'reference handoff count'
#endif
    output(12:22,row)=qrhs(0,2,0,:)
    diagnostics(:,row)=diagnostic
  enddo
  open(newunit=u,file=trim(result),access='stream',form='unformatted',status='new')
  write(u)n,statuses,output;close(u)
  if(command_argument_count()==3)then
    call get_command_argument(3,diagnostic_path)
    open(newunit=u,file=trim(diagnostic_path),access='stream',form='unformatted',status='new')
    write(u)n,diagnostics;close(u)
  endif
  call MPI_Finalize(ierr);if(ierr/=MPI_SUCCESS)error stop 'reference MPI finalize'
end program
