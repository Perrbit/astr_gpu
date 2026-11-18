module air5_flux_reference_math
  use iso_fortran_env,only: real64
  use chemistry_air5_data
  use chemistry_state_layout
  use chemistry_model
  use chemistry_thermo
  use chemistry_flow_state
#ifdef ASTR_AIR5_REFERENCE_GPU
  use cudafor
  use chemistry_model_gpu
  use chemistry_thermo_gpu
  use chemistry_flow_state_gpu
#endif
  implicit none
  integer,parameter :: air5_limiter_constraint_species=1,air5_limiter_constraint_vibrational=2, &
    air5_limiter_constraint_density=3,air5_limiter_constraint_translational=4
#ifdef ASTR_AIR5_REFERENCE_GPU
  real(real64),device :: air5_vibrational_floor_energy_d(5)
#else
  real(real64) :: vibrational_floor_energy(5)
#endif
contains
#include "frozen_flux_functions.inc"
#ifdef ASTR_AIR5_REFERENCE_GPU
  attributes(device) &
#endif
  subroutine evaluate(x,out,status)
    real(real64),intent(in) :: x(80)
    real(real64),intent(out) :: out(17)
    integer,intent(out) :: status
    integer :: mask,species,diagnostic_flag,budget_flag
    real(real64) :: trial(11)
    out=0.0_real64;status=0
    select case(nint(x(80)))
    case(0)
#ifdef ASTR_AIR5_REFERENCE_GPU
      call air5_close_species_flux_gpu(x(1:5),x(6:10),x(11:15),x(16),out(1:5),status)
#else
      call air5_close_species_flux(x(1:5),x(6:10),x(11:15),x(16),out(1:5),status)
#endif
    case(1)
#ifdef ASTR_AIR5_REFERENCE_GPU
      call air5_close_species_flux_moment_gpu(x(1:5),x(6:10),x(11:15),x(16),x(17:21),x(22),out(1:5),status)
#else
      call air5_close_species_flux_moment(x(1:5),x(6:10),x(11:15),x(16),x(17:21),x(22),out(1:5),status)
#endif
    case(2)
#ifdef ASTR_AIR5_REFERENCE_GPU
      out(1)=air5_interior_ratio_gpu(x(1));out(2)=air5_transport_species_ratio_gpu(x(2),x(3),x(4))
#else
      out(1)=air5_interior_ratio(x(1));out(2)=air5_transport_species_ratio(x(2),x(3),x(4))
#endif
    case(3)
      trial=x(1:11)+6.0_real64*x(12:22)
      diagnostic_flag=nint(x(23));budget_flag=nint(x(24))
#ifdef ASTR_AIR5_REFERENCE_GPU
      out(1)=air5_admissible_face_ratio_gpu(x(1:11),x(12:22),diagnostic_flag,mask,species,budget_flag)
      out(4)=air5_translational_margin_gpu(x(1:11))
      out(5)=air5_vibrational_excess_gpu(x(11),x(6:10))
      out(6)=real(nint(x(24)),real64)
      out(7)=merge(1.0_real64,0.0_real64,air5_state_is_admissible_gpu(trial))
      out(8)=merge(1.0_real64,0.0_real64,air5_face_state_is_admissible_gpu(trial,budget_flag))
#else
      out(1)=air5_admissible_face_ratio(x(1:11),x(12:22),x(23)/=0.0_real64,mask,species,x(24)/=0.0_real64)
      out(4)=air5_translational_margin(x(1:11));out(5)=air5_vibrational_excess(x(11),x(6:10))
      out(6)=real(nint(x(24)),real64)
      out(7)=merge(1.0_real64,0.0_real64,air5_state_is_admissible(trial))
      out(8)=merge(1.0_real64,0.0_real64,air5_face_state_is_admissible(trial,x(24)/=0.0_real64))
#endif
      out(9:13)=trial(6:10)
      out(2)=real(mask,real64);out(3)=real(species,real64)
    case(4)
#ifdef ASTR_AIR5_REFERENCE_GPU
      call air5_symmetric_face_trial_gpu(x(1:11),x(12:22),x(23:33),x(34:44),x(45:55),x(56:66), &
        x(67),out(1:11),status,out(12),x(68)/=0.0_real64,x(69)/=0.0_real64)
#else
      call air5_symmetric_face_trial(x(1:11),x(12:22),x(23:33),x(34:44),x(45:55),x(56:66), &
        x(67),out(1:11),status,out(12),x(68)/=0.0_real64,x(69)/=0.0_real64)
#endif
    case default
      status=-1
    end select
  end subroutine
#ifdef ASTR_AIR5_REFERENCE_GPU
  attributes(global) subroutine batch(n,x,out,status)
    integer,value :: n
    real(real64),device :: x(80,n),out(17,n)
    integer,device :: status(n)
    integer :: i
    i=(blockidx%x-1)*blockdim%x+threadidx%x
    if(i<=n)call evaluate(x(:,i),out(:,i),status(i))
  end subroutine
#endif
end module
program air5_flux_reference
  use air5_flux_reference_math
  implicit none
  character(1024) :: input,output,cache
  integer :: n,u,i
  real(real64),allocatable :: x(:,:),out(:,:)
  integer,allocatable :: status(:)
#ifdef ASTR_AIR5_REFERENCE_GPU
  real(real64),device,allocatable :: xd(:,:),od(:,:)
  integer,device,allocatable :: sd(:)
#endif
  call get_command_argument(1,input);call get_command_argument(2,output);call get_command_argument(3,cache)
  open(newunit=u,file=trim(input),access='stream',form='unformatted',status='old')
  read(u)n;allocate(x(80,n),out(17,n),status(n));read(u)x;close(u)
#ifdef ASTR_AIR5_REFERENCE_GPU
  call configure_air5_species_cache_gpu(trim(cache)=='1')
  air5_vibrational_floor_energy_d=x(71:75,1)
  allocate(xd(80,n),od(17,n),sd(n));xd=x
  call batch<<<(n+31)/32,32>>>(n,xd,od,sd)
  out=od;status=sd
#else
  call configure_air5_species_cache(trim(cache)=='1')
  vibrational_floor_energy=x(71:75,1)
  do i=1,n
    call evaluate(x(:,i),out(:,i),status(i))
  enddo
#endif
  open(newunit=u,file=trim(output),access='stream',form='unformatted',status='new')
  write(u)n,status,out;close(u)
end program
