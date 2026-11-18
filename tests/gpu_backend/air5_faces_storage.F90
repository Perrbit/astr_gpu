! Field/configuration owners for the frozen face operators, not a CFD runtime.
module commarray
  use iso_fortran_env,only:real64
  implicit none
  private :: real64
  real(real64),allocatable :: q(:,:,:,:),rho(:,:,:),vel(:,:,:,:),prs(:,:,:),spc(:,:,:,:), &
    dxi(:,:,:,:,:),jacob(:,:,:),sigma(:,:,:,:),qflux(:,:,:,:),jsflux(:,:,:,:,:),evflux(:,:,:,:),esflux(:,:,:,:)
  logical,allocatable :: lshock(:,:,:)
  real(real64),allocatable :: qsave(:,:,:,:),qrhs(:,:,:,:)
end module
module chemistry_flow_runtime
  implicit none
  logical :: test_symmetric=.false.,test_capturing=.false.
contains
  logical function air5_symmetric_species_convection()
    air5_symmetric_species_convection=test_symmetric
  end function
  logical function air5_shock_capturing_enabled()
    air5_shock_capturing_enabled=test_capturing
  end function
end module
#ifdef ASTR_AIR5_REFERENCE_GPU
module commarray_gpu
  use iso_fortran_env,only:real64
  use cudafor
  implicit none
  private :: real64
  real(real64),device,allocatable :: q_d(:,:,:,:),rho_d(:,:,:),vel_d(:,:,:,:),prs_d(:,:,:),spc_d(:,:,:,:), &
    dxi_d(:,:,:,:,:),jacob_d(:,:,:),sigma_d(:,:,:,:),qflux_d(:,:,:,:),species_flux_d(:,:,:,:,:), &
    ev_flux_d(:,:,:,:),species_energy_flux_d(:,:,:,:)
  integer(kind=1),device,allocatable :: shock_mask_d(:,:,:)
  real(real64),device,allocatable :: qsave_d(:,:,:,:),qrhs_d(:,:,:,:)
end module
#endif
