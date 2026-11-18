#ifdef ASTR_AIR5_REFERENCE_GPU
module commarray_gpu
  use iso_fortran_env,only:real64
  use cudafor
  implicit none
  real(real64),device,allocatable :: q_d(:,:,:,:)
end module
#endif
module air5_boundary_reference_math
  use iso_fortran_env,only:real64
  use chemistry_air5_data
  use chemistry_model
  use chemistry_state_layout
  use chemistry_flow_state
  use chemistry_thermo
#ifdef ASTR_AIR5_REFERENCE_GPU
  use cudafor
  use chemistry_flow_state_gpu
  use chemistry_thermo_gpu
  use chemistry_model_gpu
#endif
  implicit none
#ifdef ASTR_AIR5_REFERENCE_GPU
  integer,device :: hbl_status_d
  real(real64),device :: wall_temperature_d
  real(real64),device :: farfield_q_d(11),hbl_x_origin_d,incident_q_d(11,2),incident_x_d
  logical,device :: incident_top_d
#endif
contains
#include "frozen_boundary_functions.inc"
#ifdef ASTR_AIR5_REFERENCE_GPU
  attributes(global) subroutine pressure_probe(x,out,status)
    real(real64),device :: x(32),out(11)
    integer,device :: status(1)
    select case(nint(x(32)))
    case(0)
      call build_air5_pressure_outlet_state_gpu(x(1:11),x(23),out,status(1))
    case(2)
      call build_air5_hbl_outflow_state_gpu(x(1:11),x(12:22),out,status(1))
    case(3)
      call get_air5_top_target_gpu(x(23),out,status(1))
    end select
  end subroutine
#endif
end module
program air5_boundary_reference
  use air5_boundary_reference_math
#ifdef ASTR_AIR5_REFERENCE_GPU
  use commarray_gpu
#endif
  implicit none
  real(real64),allocatable :: x(:,:),out(:,:)
  integer,allocatable :: statuses(:)
  integer :: n,u,i,code
  character(1024) :: input,output
#ifdef ASTR_AIR5_REFERENCE_GPU
  real(real64),device :: xd(32),od(11)
  integer,device :: sd(1)
  real(real64) :: point(0:0,0:2,0:0,11)
#endif
  call get_command_argument(1,input);call get_command_argument(2,output)
  open(newunit=u,file=trim(input),access='stream',form='unformatted',status='old')
  read(u)n;allocate(x(32,n),out(11,n),statuses(n));read(u)x;close(u)
#ifdef ASTR_AIR5_REFERENCE_GPU
  allocate(q_d(0:0,0:2,0:0,11))
#endif
  do i=1,n
#ifdef ASTR_AIR5_REFERENCE_GPU
    call configure_air5_species_cache_gpu(x(24,i)/=0.0_real64)
    if(nint(x(32,i))/=1)then
      farfield_q_d=x(1:11,i);hbl_x_origin_d=x(25,i);incident_top_d=.false.
      xd=x(:,i);call pressure_probe<<<1,1>>>(xd,od,sd);code=cudaDeviceSynchronize()
      if(code/=0)error stop 'pressure reference launch'
      out(:,i)=od;statuses(i)=sd(1)
    else
      point=0;point(0,1,0,:)=x(1:11,i);point(0,2,0,:)=x(12:22,i);q_d=point
      wall_temperature_d=x(23,i);hbl_status_d=0
      call air5_hbl_wall_boundary_kernel<<<1,256>>>(0,2,0,0);code=cudaDeviceSynchronize()
      if(code/=0)error stop 'wall reference launch'
      point=q_d;out(:,i)=point(0,0,0,:);statuses(i)=hbl_status_d
    endif
#else
    call configure_air5_species_cache(x(24,i)/=0.0_real64)
    select case(nint(x(32,i)))
    case(0)
      call build_air5_pressure_outlet_state(x(1:11,i),x(23,i),out(:,i),statuses(i))
    case(1)
      call build_air5_hbl_wall_state(x(1:11,i),x(12:22,i),x(23,i),out(:,i),statuses(i))
    case(2)
      call build_air5_hbl_outflow_state(x(1:11,i),x(12:22,i),out(:,i),statuses(i))
    case(3)
      call build_air5_hbl_similarity_farfield_state(x(1:11,i),x(25,i),x(23,i),out(:,i),statuses(i))
    end select
#endif
  enddo
  open(newunit=u,file=trim(output),access='stream',form='unformatted',status='new');write(u)n,statuses,out;close(u)
end program
