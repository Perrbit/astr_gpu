module device_state
  use iso_c_binding, only: c_ptr,c_int,c_int64_t,c_size_t,c_double,c_float,c_char,c_loc
  implicit none
  private
#ifdef ASTR_REAL32
  integer,parameter,public :: c_device_real_kind=c_float
#else
  integer,parameter,public :: c_device_real_kind=c_double
#endif
  integer(c_int),parameter,public :: device_q=0,device_rhs=1,device_save=2
  integer(c_int),parameter,public :: device_rho=3,device_velocity=4,device_pressure=5, &
    device_temperature=6,device_metric=7,device_dvelocity=8,device_dtemperature=9
  integer(c_int),parameter,public :: device_jacobian=10,device_stress=11,device_heat_flux=12
  integer(c_int),parameter,public :: device_filter_work=13
  integer(c_int),parameter,public :: device_coordinate_y=14
  integer(c_int),parameter,public :: device_shock_sensor=15,device_interface_flux=16
  integer(c_int),parameter,public :: device_inflow_target=17,device_cell_volume=18, &
    device_wall_cell_area=19,device_diagnostic_dvelocity=20,device_diagnostic_dtemperature=21
  integer(c_int),parameter,public :: device_mp_delta=22,device_mp_ratio=23
  integer(c_int),parameter,public :: device_air5_tv=24,device_air5_species=25
  integer(c_int),parameter,public :: device_air5_carry=26,device_air5_history=27
  integer(c_int),parameter,public :: device_air5_origin=28,device_air5_origin_carry=29
  integer(c_int),parameter,public :: device_air5_dtv=30,device_air5_dspecies=31
  integer(c_int),parameter,public :: device_air5_species_flux=32,device_air5_ev_flux=33
  integer(c_int),parameter,public :: device_air5_species_energy_flux=34,device_air5_convection_base=35
  integer(c_int),parameter,public :: device_air5_top_normal_high=36
  integer(c_int),parameter,public :: device_air5_mean_sum=37,device_air5_mean_endpoint=38
  integer(c_int),parameter,public :: device_air5_diffusion_ratio=39,device_air5_diffusion_energy_ratio=40
  integer(c_int),parameter,public :: device_air5_convection_fluid_ratio=41,device_air5_mean_velocity=42, &
    device_keep_w1_flux=43,device_field_count=44
  integer(c_int),parameter,public :: device_invalid_argument=-1,device_not_allocated=-2, &
    device_nonphysical=-3,device_invalid_spectrum=-4
  type,bind(C),public :: device_view
    type(c_ptr) :: data
    integer(c_int) :: rank,element_bytes
    integer(c_int) :: lower(5),extent(5)
    integer(c_int64_t) :: stride(5)
  end type
  public :: device_bind,device_state_create,device_state_view
  public :: device_count
  public :: device_state_upload,device_state_download,device_state_synchronize,device_state_destroy
  public :: device_state_upload_f32,device_state_download_f32
  public :: device_state_snapshot_axis
  public :: device_memory_info
  public :: device_pci_identity
  interface
    integer(c_int) function device_pci_identity(handle,identity,capacity) bind(C,name='astr_device_pci_identity')
      import c_int,c_ptr,c_char
      type(c_ptr),value :: handle
      character(c_char),intent(out) :: identity(*)
      integer(c_int),value :: capacity
    end function
    integer(c_int) function device_memory_info(handle,free_bytes,total_bytes) bind(C,name='astr_device_memory_info')
      import c_int,c_ptr,c_size_t
      type(c_ptr),value :: handle
      integer(c_size_t),intent(out) :: free_bytes,total_bytes
    end function
    integer(c_int) function device_state_snapshot_axis(handle,field,axis,low,high,host,bytes) &
        bind(C,name='astr_device_state_snapshot_axis')
      import c_int,c_ptr,c_double,c_size_t
      type(c_ptr),value :: handle
      integer(c_int),value :: field,axis
      integer(c_int),intent(in) :: low(3),high(3)
      real(c_double),intent(out) :: host(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function device_count(count) bind(C,name='astr_device_count')
      import c_int
      integer(c_int),intent(out) :: count
    end function
    integer(c_int) function device_bind(device) bind(C,name='astr_device_bind')
      import c_int
      integer(c_int),value :: device
    end function
    integer(c_int) function device_state_create(device,im,jm,km,halo,debug,handle) &
        bind(C,name='astr_device_state_create')
      import c_int,c_ptr
      integer(c_int),value :: device,im,jm,km,halo,debug
      type(c_ptr),intent(out) :: handle
    end function
    integer(c_int) function device_state_view(handle,field,view) bind(C,name='astr_device_state_view')
      import c_int,c_ptr,device_view
      type(c_ptr),value :: handle
      integer(c_int),value :: field
      type(device_view),intent(out) :: view
    end function
    integer(c_int) function upload_c(handle,field,host,bytes) bind(C,name='astr_device_state_upload')
      import c_int,c_ptr,c_size_t
      type(c_ptr),value :: handle,host
      integer(c_int),value :: field
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function download_c(handle,field,host,bytes) bind(C,name='astr_device_state_download')
      import c_int,c_ptr,c_size_t
      type(c_ptr),value :: handle,host
      integer(c_int),value :: field
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function device_state_synchronize(handle) bind(C,name='astr_device_state_synchronize')
      import c_int,c_ptr
      type(c_ptr),value :: handle
    end function
    integer(c_int) function device_state_destroy(handle) bind(C,name='astr_device_state_destroy')
      import c_int,c_ptr
      type(c_ptr),intent(inout) :: handle
    end function
  end interface
contains
  integer(c_int) function device_state_upload(handle,field,host,bytes) result(status)
    type(c_ptr),value :: handle
    integer(c_int),value :: field
    real(c_device_real_kind),target,intent(in) :: host(*)
    integer(c_size_t),value :: bytes
    status=upload_c(handle,field,c_loc(host),bytes)
  end function
  integer(c_int) function device_state_upload_f32(handle,field,host,bytes) result(status)
    type(c_ptr),value :: handle
    integer(c_int),value :: field
    real(c_float),target,intent(in) :: host(*)
    integer(c_size_t),value :: bytes
    status=upload_c(handle,field,c_loc(host),bytes)
  end function
  integer(c_int) function device_state_download(handle,field,host,bytes) result(status)
    type(c_ptr),value :: handle
    integer(c_int),value :: field
    real(c_device_real_kind),target,intent(out) :: host(*)
    integer(c_size_t),value :: bytes
    status=download_c(handle,field,c_loc(host),bytes)
  end function
  integer(c_int) function device_state_download_f32(handle,field,host,bytes) result(status)
    type(c_ptr),value :: handle
    integer(c_int),value :: field
    real(c_float),target,intent(out) :: host(*)
    integer(c_size_t),value :: bytes
    status=download_c(handle,field,c_loc(host),bytes)
  end function
end module
