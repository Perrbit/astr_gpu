module insitu_moments_cpp
  use iso_c_binding
  use device_state, only: device_view,c_device_real_kind
  implicit none
  private
  public :: moments_create,moments_push,moments_view,moments_pack,moments_restore,moments_output
  public :: moments_create_air5
  public :: moments_info,moments_metrics,moments_destroy
  public :: moments_weights,moments_spatial
  public :: selected_create,selected_push,selected_scatter,selected_view,selected_download,selected_upload
  public :: selected_profile,selected_metrics,selected_destroy
  public :: selected_workspace
  interface
    integer(c_int) function selected_workspace(plan,bytes) bind(C,name='astr_insitu_selected_workspace')
      import c_int,c_ptr,c_int64_t
      type(c_ptr),value :: plan
      integer(c_int64_t),intent(out) :: bytes(2)
    end function
    integer(c_int) function selected_create(parent,nodes,indices,components,plan) bind(C,name='astr_insitu_selected_create')
      import c_int,c_ptr
      type(c_ptr),value :: parent
      integer(c_int),value :: nodes,components
      integer(c_int),intent(in) :: indices(*)
      type(c_ptr),intent(out) :: plan
    end function
    integer(c_int) function selected_push(plan,nfields,ids,const2,const6,weights,previous,covariance) &
        bind(C,name='astr_insitu_selected_push')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: plan
      integer(c_int),value :: nfields,previous,covariance
      integer(c_int),intent(in) :: ids(*)
      real(c_device_real_kind),value :: const2,const6
      real(c_device_real_kind),intent(in) :: weights(2)
    end function
    integer(c_int) function selected_scatter(plan,nodes,positions,values,nfields,weights,previous) &
        bind(C,name='astr_insitu_selected_scatter')
      import c_int,c_ptr,c_device_real_kind,device_view
      type(c_ptr),value :: plan
      integer(c_int),value :: nodes,nfields,previous
      integer(c_int),intent(in) :: positions(*)
      type(device_view),intent(in) :: values
      real(c_device_real_kind),intent(in) :: weights(2)
    end function
    integer(c_int) function selected_view(plan,field,view) bind(C,name='astr_insitu_selected_view')
      import c_int,c_ptr,device_view
      type(c_ptr),value :: plan
      integer(c_int),value :: field
      type(device_view),intent(out) :: view
    end function
    integer(c_int) function selected_download(plan,values,bytes) bind(C,name='astr_insitu_selected_download')
      import c_int,c_ptr,c_device_real_kind,c_size_t
      type(c_ptr),value :: plan
      real(c_device_real_kind),intent(out) :: values(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function selected_upload(plan,values,bytes) bind(C,name='astr_insitu_selected_upload')
      import c_int,c_ptr,c_device_real_kind,c_size_t
      type(c_ptr),value :: plan
      real(c_device_real_kind),intent(in) :: values(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function selected_profile(plan,ny,nk,weights,nfields,covariance,values,bytes) &
        bind(C,name='astr_insitu_selected_profile')
      import c_int,c_ptr,c_device_real_kind,c_size_t
      type(c_ptr),value :: plan
      integer(c_int),value :: ny,nk,nfields,covariance
      real(c_device_real_kind),intent(in) :: weights(*)
      real(c_device_real_kind),intent(out) :: values(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function selected_metrics(plan,values) bind(C,name='astr_insitu_selected_metrics')
      import c_int,c_ptr,c_int64_t
      type(c_ptr),value :: plan
      integer(c_int64_t),intent(out) :: values(9)
    end function
    integer(c_int) function selected_destroy(plan) bind(C,name='astr_insitu_selected_destroy')
      import c_int,c_ptr
      type(c_ptr),intent(inout) :: plan
    end function
    integer(c_int) function moments_create(parent,owned,window,plan) bind(C,name='astr_insitu_moments_create')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: parent
      integer(c_int),intent(in) :: owned(3)
      real(c_device_real_kind),intent(in) :: window(2)
      type(c_ptr),intent(out) :: plan
    end function
    integer(c_int) function moments_create_air5(parent,owned,window,plan) bind(C,name='astr_insitu_moments_create_air5')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: parent
      integer(c_int),intent(in) :: owned(3)
      real(c_device_real_kind),intent(in) :: window(2)
      type(c_ptr),intent(out) :: plan
    end function
    integer(c_int) function moments_push(plan,time,weights) bind(C,name='astr_insitu_moments_push')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: plan
      real(c_device_real_kind),value :: time
      real(c_device_real_kind),intent(in) :: weights(2)
    end function
    integer(c_int) function moments_view(plan,view) bind(C,name='astr_insitu_moments_view')
      import c_int,c_ptr,device_view
      type(c_ptr),value :: plan
      type(device_view),intent(out) :: view
    end function
    integer(c_int) function moments_pack(plan,values,bytes) bind(C,name='astr_insitu_moments_pack')
      import c_int,c_ptr,c_device_real_kind,c_size_t
      type(c_ptr),value :: plan
      real(c_device_real_kind),intent(out) :: values(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function moments_restore(plan,values,bytes,time) bind(C,name='astr_insitu_moments_restore')
      import c_int,c_ptr,c_device_real_kind,c_size_t
      type(c_ptr),value :: plan
      real(c_device_real_kind),intent(in) :: values(*)
      integer(c_size_t),value :: bytes
      real(c_device_real_kind),value :: time
    end function
    integer(c_int) function moments_output(plan,values,bytes) bind(C,name='astr_insitu_moments_output')
      import c_int,c_ptr,c_device_real_kind,c_size_t
      type(c_ptr),value :: plan
      real(c_device_real_kind),intent(out) :: values(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function moments_weights(plan,values,bytes) bind(C,name='astr_insitu_moments_weights')
      import c_int,c_ptr,c_device_real_kind,c_size_t
      type(c_ptr),value :: plan
      real(c_device_real_kind),intent(in) :: values(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function moments_spatial(plan,values) bind(C,name='astr_insitu_moments_spatial')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: plan
      real(c_device_real_kind),intent(out) :: values(7)
    end function
    integer(c_int) function moments_info(plan,values) bind(C,name='astr_insitu_moments_info')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: plan
      real(c_device_real_kind),intent(out) :: values(5)
    end function
    integer(c_int) function moments_metrics(plan,values) bind(C,name='astr_insitu_moments_metrics')
      import c_int,c_ptr,c_int64_t
      type(c_ptr),value :: plan
      integer(c_int64_t),intent(out) :: values(6)
    end function
    integer(c_int) function moments_destroy(plan) bind(C,name='astr_insitu_moments_destroy')
      import c_int,c_ptr
      type(c_ptr),intent(inout) :: plan
    end function
  end interface
end module
