module air5_properties_reference_math
  use iso_fortran_env, only: real64
#ifdef ASTR_AIR5_REFERENCE_GPU
  use chemistry_flow_state_gpu, only: primitive=>air5_primitive_to_conservative_gpu, &
    conservative=>air5_conservative_to_primitive_gpu
  use chemistry_thermo_gpu, only: pressure=>air5_pressure_gpu, &
    t_derivatives=>air5_temperature_derivatives_gpu,tv_derivatives=>air5_tv_derivatives_gpu, &
    cv_tr=>air5_species_cv_tr_gpu,ev_species=>air5_species_vibrational_energy_gpu
  use chemistry_transport_gpu, only: transport=>air5_transport_properties_gpu,flux=>air5_diffusive_flux_gpu
#else
  use chemistry_flow_state, only: primitive=>air5_primitive_to_conservative, &
    conservative=>air5_conservative_to_primitive
  use chemistry_thermo, only: pressure=>air5_pressure, &
    t_derivatives=>air5_temperature_derivatives,tv_derivatives=>air5_tv_derivatives, &
    cv_tr=>air5_species_cv_tr,ev_species=>air5_species_vibrational_energy
  use chemistry_transport, only: transport=>air5_transport_properties,flux=>air5_diffusive_flux
#endif
  use chemistry_air5_data, only: air5_formation_energy,air5_ru,air5_molar_mass
  implicit none
contains
#ifdef ASTR_AIR5_REFERENCE_GPU
  attributes(device) &
#endif
  subroutine evaluate(x,out,status)
    real(real64),intent(in) :: x(41)
    real(real64),intent(out) :: out(114)
    integer,intent(out) :: status(7)
    real(real64) :: z(5),p,binary(5,5),stress(3,3),species_flux(5,3)
    real(real64) :: dv(3,3),dy(5,3)
    integer :: s
    out=0.0_real64
    z=x(1)*x(6:10)
    dv=reshape(x(12:20),[3,3]);dy=reshape(x(27:41),[5,3])
    call pressure(z,x(5),p,status(1))
    call primitive(x(1),x(2:4),x(5),x(6:10),x(11),out(1:11),status(2))
    call conservative(out(1:11),out(12),out(13:15),out(16),out(17:21),out(22),out(23),status(3))
    out(24)=p
    call t_derivatives(z,x(5),out(25:29),out(30),status(4))
    call tv_derivatives(z,x(11),out(31:35),out(36),status(5))
    call transport(x(5),x(11),p,x(6:10),out(37),out(38),out(39),out(40:44),binary,out(70:74),status(6))
    out(45:69)=reshape(binary,[25])
    call flux(x(1),x(2:4),x(5),x(11),p,x(6:10),dv,x(21:23),x(24:26),dy, &
      stress,species_flux,out(99:101),out(102:104),status(7))
    out(75:83)=reshape(stress,[9]);out(84:98)=reshape(species_flux,[15])
    if(status(7)==0) then
      do s=1,5
        out(109+s)=ev_species(s,x(11))
        out(104+s)=cv_tr(s)*x(5)+out(109+s)+air5_formation_energy(s)+air5_ru/air5_molar_mass(s)*x(5)
      enddo
    endif
  end subroutine
#ifdef ASTR_AIR5_REFERENCE_GPU
  attributes(global) subroutine evaluate_batch(n,x,out,status)
    integer,value :: n
    real(real64),device,intent(in) :: x(41,n)
    real(real64),device,intent(out) :: out(114,n)
    integer,device,intent(out) :: status(7,n)
    integer :: i
    i=(blockidx%x-1)*blockdim%x+threadidx%x
    if(i<=n) call evaluate(x(:,i),out(:,i),status(:,i))
  end subroutine
#endif
end module

program air5_properties_reference
  use iso_fortran_env, only: real64
  use air5_properties_reference_math
#ifdef ASTR_AIR5_REFERENCE_GPU
  use cudafor
  use chemistry_flow_state_gpu, only: configure_air5_species_cache_gpu
#else
  use chemistry_flow_state, only: configure_air5_species_cache
#endif
  implicit none
  character(len=1024) :: input_path,output_path,mode
  real(real64),allocatable :: x(:,:),out(:,:)
  integer,allocatable :: statuses(:,:)
  integer :: n,u,i
#ifdef ASTR_AIR5_REFERENCE_GPU
  real(real64),device,allocatable :: xd(:,:),od(:,:)
  integer,device,allocatable :: sd(:,:)
#endif
  call get_command_argument(1,input_path)
  call get_command_argument(2,output_path)
  call get_command_argument(3,mode)
  open(newunit=u,file=trim(input_path),access='stream',form='unformatted',status='old')
  read(u) n
  allocate(x(41,n),out(114,n),statuses(7,n))
  read(u) x
  close(u)
#ifdef ASTR_AIR5_REFERENCE_GPU
  call configure_air5_species_cache_gpu(trim(mode)=='all')
  allocate(xd(41,n),od(114,n),sd(7,n))
  xd=x
  call evaluate_batch<<<(n+127)/128,128>>>(n,xd,od,sd)
  i=cudaDeviceSynchronize()
  if(i/=0) error stop 'AIR5 reference device execution failed'
  out=od;statuses=sd
#else
  call configure_air5_species_cache(trim(mode)=='all')
  do i=1,n
    call evaluate(x(:,i),out(:,i),statuses(:,i))
  enddo
#endif
  open(newunit=u,file=trim(output_path),access='stream',form='unformatted',status='new')
  write(u) n,statuses,out
  close(u)
end program
