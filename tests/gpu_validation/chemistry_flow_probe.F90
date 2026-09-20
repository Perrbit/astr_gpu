program chemistry_flow_probe
  use iso_fortran_env, only: real64
  use chemistry_air5_data, only: air5_num_species
  use chemistry_flow_state, only: air5_primitive_to_conservative, &
    air5_conservative_to_primitive
  use chemistry_transport, only: air5_transport_properties, air5_diffusive_flux
  implicit none

  character(len=32) :: mode
  real(real64) :: rho,velocity(3),temperature,tv,mass_fraction(air5_num_species)
  real(real64) :: q(11),q_roundtrip(11),rho_out,velocity_out(3),temperature_out, &
    tv_out,mass_fraction_out(air5_num_species),pressure
  real(real64) :: viscosity,conductivity_tr,conductivity_v, &
    species_viscosity(air5_num_species),binary_diffusion(air5_num_species,air5_num_species), &
    mixture_diffusion(air5_num_species)
  real(real64) :: grad_velocity(3,3),grad_temperature(3),grad_tv(3), &
    grad_mass_fraction(air5_num_species,3),species_flux(air5_num_species,3), &
    momentum_flux(3,3),energy_flux(3),ev_flux(3)
  integer :: status,status2

  call get_command_argument(1,mode)
  select case(trim(mode))
  case('roundtrip')
    rho=0.42_real64
    velocity=[1200.0_real64,-37.0_real64,84.0_real64]
    temperature=4200.0_real64
    tv=3100.0_real64
    mass_fraction=[0.62_real64,0.18_real64,0.07_real64,0.05_real64,0.08_real64]
    call air5_primitive_to_conservative(rho,velocity,temperature,mass_fraction,tv,q,status)
    call air5_conservative_to_primitive(q,rho_out,velocity_out,temperature_out, &
      mass_fraction_out,tv_out,pressure,status2)
    call air5_primitive_to_conservative(rho_out,velocity_out,temperature_out, &
      mass_fraction_out,tv_out,q_roundtrip,status2)
    write(*,'(2(I0,1X),7(ES25.16E3,1X))') status,status2, &
      maxval(abs(q_roundtrip-q)),abs(rho_out-rho),maxval(abs(velocity_out-velocity)), &
      abs(temperature_out-temperature),maxval(abs(mass_fraction_out-mass_fraction)), &
      abs(tv_out-tv),pressure
  case('properties')
    mass_fraction=[0.70_real64,0.20_real64,0.04_real64,0.03_real64,0.03_real64]
    call air5_transport_properties(5000.0_real64,3600.0_real64,2.0e5_real64, &
      mass_fraction,viscosity,conductivity_tr,conductivity_v,species_viscosity, &
      binary_diffusion,mixture_diffusion,status)
    write(*,'(I0,1X,15(ES25.16E3,1X))') status,viscosity,conductivity_tr, &
      conductivity_v,species_viscosity,binary_diffusion(1,2), &
      binary_diffusion(3,5),mixture_diffusion
  case('flux')
    rho=0.42_real64
    velocity=[1200.0_real64,-37.0_real64,84.0_real64]
    temperature=4200.0_real64
    tv=3100.0_real64
    mass_fraction=[0.62_real64,0.18_real64,0.07_real64,0.05_real64,0.08_real64]
    grad_velocity=reshape([0.8_real64,-0.3_real64,0.2_real64, &
      0.1_real64,-0.4_real64,0.7_real64,-0.5_real64,0.6_real64,-0.2_real64],[3,3])
    grad_temperature=[130.0_real64,-75.0_real64,42.0_real64]
    grad_tv=[-90.0_real64,35.0_real64,11.0_real64]
    grad_mass_fraction(:,1)=[0.012_real64,-0.006_real64,-0.002_real64,-0.001_real64,-0.003_real64]
    grad_mass_fraction(:,2)=[-0.004_real64,0.008_real64,-0.001_real64,-0.002_real64,-0.001_real64]
    grad_mass_fraction(:,3)=[0.002_real64,-0.005_real64,0.001_real64,0.001_real64,0.001_real64]
    call air5_diffusive_flux(rho,velocity,temperature,tv,2.0e5_real64,mass_fraction, &
      grad_velocity,grad_temperature,grad_tv,grad_mass_fraction,momentum_flux, &
      species_flux,energy_flux,ev_flux,status)
    write(*,'(I0,1X,21(ES25.16E3,1X))') status,maxval(abs(sum(species_flux,dim=1))), &
      species_flux(:,1),energy_flux,ev_flux,momentum_flux
  case('pure')
    rho=0.8_real64
    velocity=0.0_real64
    temperature=1000.0_real64
    tv=1000.0_real64
    mass_fraction=[1.0_real64,0.0_real64,0.0_real64,0.0_real64,0.0_real64]
    grad_velocity=0.0_real64
    grad_temperature=0.0_real64
    grad_tv=0.0_real64
    grad_mass_fraction=0.0_real64
    call air5_diffusive_flux(rho,velocity,temperature,tv,1.0e5_real64,mass_fraction, &
      grad_velocity,grad_temperature,grad_tv,grad_mass_fraction,momentum_flux, &
      species_flux,energy_flux,ev_flux,status)
    write(*,'(I0,1X,4(ES25.16E3,1X))') status,maxval(abs(species_flux)), &
      maxval(abs(momentum_flux)),maxval(abs(energy_flux)),maxval(abs(ev_flux))
  case('trace')
    rho=0.8_real64
    velocity=0.0_real64
    temperature=1000.0_real64
    tv=1000.0_real64
    mass_fraction=[1.0_real64-4.0e-14_real64,1.0e-14_real64, &
      1.0e-14_real64,1.0e-14_real64,1.0e-14_real64]
    grad_velocity=0.0_real64
    grad_temperature=0.0_real64
    grad_tv=0.0_real64
    grad_mass_fraction=0.0_real64
    grad_mass_fraction(:,1)=1.0e-8_real64*[4.0_real64,-1.0_real64, &
      -1.0_real64,-1.0_real64,-1.0_real64]
    call air5_transport_properties(temperature,tv,1.0e5_real64,mass_fraction, &
      viscosity,conductivity_tr,conductivity_v,species_viscosity, &
      binary_diffusion,mixture_diffusion,status2)
    call air5_diffusive_flux(rho,velocity,temperature,tv,1.0e5_real64,mass_fraction, &
      grad_velocity,grad_temperature,grad_tv,grad_mass_fraction,momentum_flux, &
      species_flux,energy_flux,ev_flux,status)
    status=max(status,status2)
    write(*,'(I0,1X,3(ES25.16E3,1X))') status, &
      maxval(abs(species_flux(2:air5_num_species,:))),minval(mixture_diffusion), &
      maxval(abs(sum(species_flux,dim=1)))
  case('closure-roundoff')
    mass_fraction=[0.76690794303092946_real64,0.23308774709158103_real64, &
      5.1679452124184590e-11_real64,4.2992112853434266e-6_real64, &
      1.0614510385629360e-8_real64]
    call air5_transport_properties(1541.1514486_real64,1408.1376083_real64, &
      74228.928248_real64,mass_fraction,viscosity,conductivity_tr,conductivity_v, &
      species_viscosity,binary_diffusion,mixture_diffusion,status)
    write(*,'(I0,1X,2(ES25.16E3,1X))') status, &
      abs(sum(mass_fraction)-1.0_real64),minval(mixture_diffusion)
  case('closure-hbl-outflow')
    q=[1.161428770839788988e-1_real64,2.375474942785869246e2_real64, &
      1.192005396401516926e0_real64,-2.755478444608583182e-11_real64, &
      4.021483775032878038e5_real64,8.907091130587020678e-2_real64, &
      2.707115024895278069e-2_real64,9.467318581106629851e-12_real64, &
      8.134607202249798363e-7_real64,2.058965052555392159e-9_real64, &
      2.165529498430201784e4_real64]
    call air5_conservative_to_primitive(q,rho,velocity,temperature,mass_fraction, &
      tv,pressure,status2)
    call air5_transport_properties(temperature,tv,pressure, &
      mass_fraction,viscosity,conductivity_tr,conductivity_v,species_viscosity, &
      binary_diffusion,mixture_diffusion,status)
    status=max(status,status2)
    write(*,'(I0,1X,2(ES25.16E3,1X))') status, &
      sum(mass_fraction)-1.0_real64,minval(mixture_diffusion)
  case default
    error stop 'unknown chemistry flow probe mode'
  end select
end program chemistry_flow_probe
