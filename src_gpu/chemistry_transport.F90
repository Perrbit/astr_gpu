module chemistry_device_transport
  use iso_c_binding, only: c_ptr,c_int,c_double,c_size_t,c_int64_t
  implicit none
  private
  public :: air5_geometry_prepare,air5_rk_prepare,air5_rk_update
  public :: air5_velocity_gradients_prepare,air5_transport_prepare,air5_gradients,air5_diffusive_flux
  public :: air5_diffusion_prepare,air5_diffusion_begin,air5_diffusion_budget,air5_diffusion_rhs,air5_diffusion_read_report
  public :: air5_raw_convection,air5_convection_budget_prepare,air5_convection_budget_begin,air5_convection_budget
  public :: air5_convection_rhs,air5_convection_read_report
  public :: air5_faces_prepare,air5_reconstruct_faces
  public :: air5_symmetric_prepare,air5_symmetric_base,air5_symmetric_faces,air5_symmetric_apply,air5_symmetric_final
  public :: air5_symmetric_top_normal,air5_symmetric_plane,air5_symmetric_buffers,air5_symmetric_copy_local_planes
  public :: air5_symmetric_failure,air5_symmetric_exchange
  public :: air5_flux_transport_prepare,air5_flux_exchange
  public :: air5_compensation_axis
  public :: air5_filter_prepare,air5_filter_apply
  public :: air5_legacy_filter_save,air5_legacy_filter_repair
  public :: air5_legacy_linear_prepare,air5_legacy_filter_apply
  integer(c_int),parameter,public :: air5_flux_separate=0,air5_flux_packed=1,air5_flux_packed_overlap=2
  integer(c_int),parameter,public :: air5_conv_full_state=0,air5_conv_species_budget=1,air5_conv_consistent_species=2
  type,bind(C),public :: air5_transport_failure
    integer(c_int) :: failure(7)
    real(c_double) :: values(11,6),cdt
  end type
  type,bind(C),public :: air5_diffusion_report
    integer(c_int) :: claimed,location(4),limited
    integer(c_int64_t) :: minimum_bits
    real(c_double) :: values(5)
  end type
  type,bind(C),public :: air5_convection_report
    integer(c_int) :: claimed,location(4),active(10)
    integer(c_int64_t) :: minimum_bits(10)
    real(c_double) :: values(5)
  end type
  interface
    integer(c_int) function air5_raw_convection(handle,ntypes,first,last,capturing,symmetric) &
        bind(C,name='astr_device_air5_raw_convection')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),intent(in) :: ntypes(3),first(3),last(3)
      integer(c_int),value :: capturing,symmetric
    end function
    integer(c_int) function air5_convection_budget_prepare(handle,consistent) bind(C,name='astr_device_air5_convection_budget_prepare')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),value :: consistent
    end function
    integer(c_int) function air5_convection_budget_begin(handle,phase) bind(C,name='astr_device_air5_convection_budget_begin')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),value :: phase
    end function
    integer(c_int) function air5_convection_budget(handle,ntypes,first,last,floor,a,b,cdt,first_stage, &
        capturing,species_budget,phase,collect) bind(C,name='astr_device_air5_convection_budget')
      import c_ptr,c_int,c_double
      type(c_ptr),value :: handle
      integer(c_int),intent(in) :: ntypes(3),first(3),last(3)
      real(c_double),intent(in) :: floor(5)
      real(c_double),value :: a,b,cdt
      integer(c_int),value :: first_stage,capturing,species_budget,phase,collect
    end function
    integer(c_int) function air5_convection_rhs(handle,ntypes,first,last,capturing,phase) bind(C,name='astr_device_air5_convection_rhs')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),intent(in) :: ntypes(3),first(3),last(3)
      integer(c_int),value :: capturing,phase
    end function
    integer(c_int) function air5_convection_read_report(handle,report) bind(C,name='astr_device_air5_convection_read_report')
      import c_ptr,c_int,air5_convection_report
      type(c_ptr),value :: handle
      type(air5_convection_report),intent(out) :: report
    end function
    integer(c_int) function air5_diffusion_prepare(handle) bind(C,name='astr_device_air5_diffusion_prepare')
      import c_ptr,c_int
      type(c_ptr),value :: handle
    end function
    integer(c_int) function air5_diffusion_begin(handle) bind(C,name='astr_device_air5_diffusion_begin')
      import c_ptr,c_int
      type(c_ptr),value :: handle
    end function
    integer(c_int) function air5_diffusion_budget(handle,ntypes,first,last,floor,a,b,cdt,first_stage,energy,collect) &
        bind(C,name='astr_device_air5_diffusion_budget')
      import c_ptr,c_int,c_double
      type(c_ptr),value :: handle
      integer(c_int),intent(in) :: ntypes(3),first(3),last(3)
      real(c_double),intent(in) :: floor(5)
      real(c_double),value :: a,b,cdt
      integer(c_int),value :: first_stage,energy,collect
    end function
    integer(c_int) function air5_diffusion_rhs(handle,ntypes,first,last,axis) bind(C,name='astr_device_air5_diffusion_rhs')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),intent(in) :: ntypes(3),first(3),last(3)
      integer(c_int),value :: axis
    end function
    integer(c_int) function air5_diffusion_read_report(handle,report) bind(C,name='astr_device_air5_diffusion_report')
      import c_ptr,c_int,air5_diffusion_report
      type(c_ptr),value :: handle
      type(air5_diffusion_report),intent(out) :: report
    end function
    integer(c_int) function air5_legacy_linear_prepare(handle,scalar) bind(C,name='astr_device_air5_legacy_linear_prepare')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),value :: scalar
    end function
    integer(c_int) function air5_legacy_filter_apply(handle,floor,wall_axis,family,physics_status) &
        bind(C,name='astr_device_air5_legacy_filter_apply')
      import c_ptr,c_int,c_double
      type(c_ptr),value :: handle
      real(c_double),intent(in) :: floor(5)
      integer(c_int),value :: wall_axis,family
      integer(c_int),intent(out) :: physics_status
    end function
    integer(c_int) function air5_legacy_filter_save(handle) bind(C,name='astr_device_air5_legacy_filter_save')
      import c_ptr,c_int
      type(c_ptr),value :: handle
    end function
    integer(c_int) function air5_legacy_filter_repair(handle,floor) bind(C,name='astr_device_air5_legacy_filter_repair')
      import c_ptr,c_int,c_double
      type(c_ptr),value :: handle
      real(c_double),intent(in) :: floor(5)
    end function
    integer(c_int) function air5_compensation_axis(handle,axis,normalize) bind(C,name='astr_device_air5_compensation_axis')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),value :: axis,normalize
    end function
    integer(c_int) function air5_filter_prepare(handle,tr_floor,v_floor,global,offset,periodic,scalar) &
        bind(C,name='astr_device_air5_filter_prepare')
      import c_ptr,c_int,c_double
      type(c_ptr),value :: handle
      real(c_double),intent(in) :: tr_floor(5),v_floor(5)
      integer(c_int),intent(in) :: global(3),offset(3),periodic(3)
      integer(c_int),value :: scalar
    end function
    integer(c_int) function air5_filter_apply(handle,preserve,physics_status) bind(C,name='astr_device_air5_filter_apply')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),value :: preserve
      integer(c_int),intent(out) :: physics_status
    end function
    integer(c_int) function air5_flux_transport_prepare(handle,mode) bind(C,name='astr_device_air5_flux_transport_prepare')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),value :: mode
    end function
    integer(c_int) function air5_flux_exchange(handle) bind(C,name='astr_device_air5_flux_exchange')
      import c_ptr,c_int
      type(c_ptr),value :: handle
    end function
    integer(c_int) function air5_symmetric_exchange(handle,axis) bind(C,name='astr_device_air5_symmetric_exchange')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),value :: axis
    end function
    integer(c_int) function air5_symmetric_prepare(handle,floor,with_top) bind(C,name='astr_device_air5_symmetric_prepare')
      import c_ptr,c_int,c_double
      type(c_ptr),value :: handle
      real(c_double),intent(in) :: floor(5)
      integer(c_int),value :: with_top
    end function
    integer(c_int) function air5_symmetric_base(handle,ntypes,a,b,cdt,first,top) bind(C,name='astr_device_air5_symmetric_base')
      import c_ptr,c_int,c_double
      type(c_ptr),value :: handle
      integer(c_int),intent(in) :: ntypes(3)
      real(c_double),value :: a,b,cdt
      integer(c_int),value :: first,top
    end function
    integer(c_int) function air5_symmetric_faces(handle,ntypes,axis,capturing,cdt,top,preserve) &
        bind(C,name='astr_device_air5_symmetric_faces')
      import c_ptr,c_int,c_double
      type(c_ptr),value :: handle
      integer(c_int),intent(in) :: ntypes(3)
      integer(c_int),value :: axis,capturing,top,preserve
      real(c_double),value :: cdt
    end function
    integer(c_int) function air5_symmetric_apply(handle,ntypes,axis,cdt,top) bind(C,name='astr_device_air5_symmetric_apply')
      import c_ptr,c_int,c_double
      type(c_ptr),value :: handle
      integer(c_int),intent(in) :: ntypes(3)
      integer(c_int),value :: axis,top
      real(c_double),value :: cdt
    end function
    integer(c_int) function air5_symmetric_final(handle,ntypes,a,b,cdt,first,top) bind(C,name='astr_device_air5_symmetric_final')
      import c_ptr,c_int,c_double
      type(c_ptr),value :: handle
      integer(c_int),intent(in) :: ntypes(3)
      real(c_double),value :: a,b,cdt
      integer(c_int),value :: first,top
    end function
    integer(c_int) function air5_symmetric_top_normal(handle,ntypes,cdt,preserve) &
        bind(C,name='astr_device_air5_symmetric_top_normal')
      import c_ptr,c_int,c_double
      type(c_ptr),value :: handle
      integer(c_int),intent(in) :: ntypes(3)
      real(c_double),value :: cdt
      integer(c_int),value :: preserve
    end function
    integer(c_int) function air5_symmetric_plane(handle,axis,unpack,plus,minus) &
        bind(C,name='astr_device_air5_symmetric_plane')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),value :: axis,unpack,plus,minus
    end function
    integer(c_int) function air5_symmetric_buffers(handle,buffers,capacity) bind(C,name='astr_device_air5_symmetric_buffers')
      import c_ptr,c_int,c_size_t
      type(c_ptr),value :: handle
      type(c_ptr),intent(out) :: buffers(4)
      integer(c_size_t),intent(out) :: capacity
    end function
    integer(c_int) function air5_symmetric_copy_local_planes(handle,axis) &
        bind(C,name='astr_device_air5_symmetric_copy_local_planes')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),value :: axis
    end function
    integer(c_int) function air5_symmetric_failure(handle,failure) bind(C,name='astr_device_air5_symmetric_failure')
      import c_ptr,c_int,air5_transport_failure
      type(c_ptr),value :: handle
      type(air5_transport_failure),intent(out) :: failure
    end function
    integer(c_int) function air5_faces_prepare(handle) bind(C,name='astr_device_air5_faces_prepare')
      import c_ptr,c_int
      type(c_ptr),value :: handle
    end function
    integer(c_int) function air5_reconstruct_faces(handle,axis,ntype,kind,capturing,symmetric,first,last) &
        bind(C,name='astr_device_air5_reconstruct_faces')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),value :: axis,ntype,kind,capturing,symmetric
      integer(c_int),intent(in) :: first(3),last(3)
    end function
    integer(c_int) function air5_velocity_gradients_prepare(handle) bind(C,name='astr_device_air5_velocity_gradients_prepare')
      import c_ptr,c_int
      type(c_ptr),value :: handle
    end function
    integer(c_int) function air5_transport_prepare(handle,layered) bind(C,name='astr_device_air5_transport_prepare')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),value :: layered
    end function
    integer(c_int) function air5_gradients(handle,low,high,zero_lower_y) bind(C,name='astr_device_air5_gradients')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),intent(in) :: low(3),high(3)
      integer(c_int),value :: zero_lower_y
    end function
    integer(c_int) function air5_diffusive_flux(handle) bind(C,name='astr_device_air5_diffusive_flux')
      import c_ptr,c_int
      type(c_ptr),value :: handle
    end function
    integer(c_int) function air5_geometry_prepare(handle) bind(C,name='astr_device_air5_geometry_prepare')
      import c_ptr,c_int
      type(c_ptr),value :: handle
    end function
    integer(c_int) function air5_rk_prepare(handle) bind(C,name='astr_device_air5_rk_prepare')
      import c_ptr,c_int
      type(c_ptr),value :: handle
    end function
    integer(c_int) function air5_rk_update(handle,first,last,dt,a,b,c,first_stage) bind(C,name='astr_device_air5_rk_update')
      import c_ptr,c_int,c_double
      type(c_ptr),value :: handle
      integer(c_int),intent(in) :: first(3),last(3)
      real(c_double),value :: dt,a,b,c
      integer(c_int),value :: first_stage
    end function
  end interface
end module
