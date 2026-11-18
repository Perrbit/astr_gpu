module insitu_sampling_cpp
  use iso_c_binding
  use device_state, only: device_view,c_device_real_kind
  implicit none
  private
  integer(c_int),parameter,public :: sample_velocity=0,sample_halo=1,sample_diagnostics=2,sample_state=3
  integer(c_int),parameter,public :: sample_coordinates=4,sample_geometry=5
  public :: sample_create,sample_capture,sample_view,sample_download,sample_metrics,sample_destroy
  public :: begin_sample,canonical_axis,make_velocity,halo_axis,finish_sample
  public :: sample_capture_vector,sample_capture_coordinates,sample_capture_owned_vector,sample_upload_coordinates
  public :: probe_create,probe_capture,probe_metrics,probe_destroy
  type,public :: sample_plan
    type(c_ptr) :: handle=c_null_ptr
    logical :: halo=.false.
  end type
  interface
    integer(c_int) function probe_create(parent,count,indices,constants,handle) bind(C,name='astr_insitu_probe_create')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: parent
      integer(c_int),value :: count
      integer(c_int),intent(in) :: indices(*)
      real(c_device_real_kind),intent(in) :: constants(6)
      type(c_ptr),intent(out) :: handle
    end function
    integer(c_int) function probe_capture(handle,values,bytes) bind(C,name='astr_insitu_probe_capture')
      import c_int,c_ptr,c_device_real_kind,c_size_t
      type(c_ptr),value :: handle
      real(c_device_real_kind),intent(out) :: values(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function probe_metrics(handle,values) bind(C,name='astr_insitu_probe_metrics')
      import c_int,c_ptr,c_int64_t
      type(c_ptr),value :: handle
      integer(c_int64_t),intent(out) :: values(5)
    end function
    integer(c_int) function probe_destroy(handle) bind(C,name='astr_insitu_probe_destroy')
      import c_int,c_ptr
      type(c_ptr),intent(inout) :: handle
    end function
    integer(c_int) function create(parent,comm,lower,upper,sizes,periodic,mode,velocity,halo,components,sixth,handle) &
        bind(C,name='astr_insitu_sample_create')
      import c_int,c_ptr
      type(c_ptr),value :: parent
      integer(c_int),value :: comm,mode,velocity,halo,components,sixth
      integer(c_int),intent(in) :: lower(3),upper(3),sizes(3),periodic(3)
      type(c_ptr),intent(out) :: handle
    end function
    integer(c_int) function begin_sample(handle) bind(C,name='astr_insitu_sample_begin')
      import c_int,c_ptr
      type(c_ptr),value :: handle
    end function
    integer(c_int) function import_vector(handle,source,slots) bind(C,name='astr_insitu_sample_vector')
      import c_int,c_ptr,device_view
      type(c_ptr),value :: handle
      type(device_view),intent(in) :: source
      integer(c_int),intent(in) :: slots(3)
    end function
    integer(c_int) function sample_capture_coordinates(handle,source,offset,global) bind(C,name='astr_insitu_sample_coordinates')
      import c_int,c_ptr,device_view
      type(c_ptr),value :: handle
      type(device_view),intent(in) :: source
      integer(c_int),intent(in) :: offset(3),global(3)
    end function
    integer(c_int) function sample_capture_owned_vector(handle,source,owned,slots) bind(C,name='astr_insitu_sample_owned_vector')
      import c_int,c_ptr,device_view
      type(c_ptr),value :: handle
      type(device_view),intent(in) :: source
      integer(c_int),intent(in) :: owned(3),slots(3)
    end function
    integer(c_int) function sample_upload_coordinates(handle,xyz,bytes,offset,global) &
        bind(C,name='astr_insitu_sample_upload_coordinates')
      import c_int,c_ptr,c_device_real_kind,c_size_t
      type(c_ptr),value :: handle
      real(c_device_real_kind),intent(in) :: xyz(*)
      integer(c_size_t),value :: bytes
      integer(c_int),intent(in) :: offset(3),global(3)
    end function
    integer(c_int) function canonical_axis(handle,axis) bind(C,name='astr_insitu_sample_canonical_axis')
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),value :: axis
    end function
    integer(c_int) function make_velocity(handle) bind(C,name='astr_insitu_sample_velocity')
      import c_int,c_ptr
      type(c_ptr),value :: handle
    end function
    integer(c_int) function halo_axis(handle,axis) bind(C,name='astr_insitu_sample_halo_axis')
      import c_int,c_ptr
      type(c_ptr),value :: handle
      integer(c_int),value :: axis
    end function
    integer(c_int) function finish_sample(handle) bind(C,name='astr_insitu_sample_finish')
      import c_int,c_ptr
      type(c_ptr),value :: handle
    end function
    integer(c_int) function sample_view(handle,field,view) bind(C,name='astr_insitu_sample_view')
      import c_int,c_ptr,device_view
      type(c_ptr),value :: handle
      integer(c_int),value :: field
      type(device_view),intent(out) :: view
    end function
    integer(c_int) function sample_download(handle,field,host,bytes) bind(C,name='astr_insitu_sample_download')
      import c_int,c_ptr,c_device_real_kind,c_size_t
      type(c_ptr),value :: handle
      integer(c_int),value :: field
      real(c_device_real_kind),intent(out) :: host(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function sample_metrics(handle,values) bind(C,name='astr_insitu_sample_metrics')
      import c_int,c_ptr,c_int64_t
      type(c_ptr),value :: handle
      integer(c_int64_t),intent(out) :: values(9)
    end function
    integer(c_int) function destroy(handle) bind(C,name='astr_insitu_sample_destroy')
      import c_int,c_ptr
      type(c_ptr),intent(inout) :: handle
    end function
  end interface
contains
  integer(c_int) function sample_create(parent,comm,lower,upper,sizes,periodic,mode,velocity,halo,components,sixth,plan) result(status)
    type(c_ptr),intent(in) :: parent
    integer,intent(in) :: comm
    integer(c_int),intent(in) :: lower(3),upper(3),sizes(3),periodic(3),mode,components
    logical,intent(in) :: velocity,halo,sixth
    type(sample_plan),intent(inout) :: plan
    plan%handle=c_null_ptr;plan%halo=halo
    status=create(parent,int(comm,c_int),lower,upper,sizes,periodic,mode,merge(1_c_int,0_c_int,velocity), &
      merge(1_c_int,0_c_int,halo),components,merge(1_c_int,0_c_int,sixth),plan%handle)
  end function
  integer(c_int) function sample_capture(plan) result(status)
    type(sample_plan),intent(in) :: plan
    integer(c_int) :: axis
    status=begin_sample(plan%handle)
    if(status/=0) return
    do axis=0,2
      status=canonical_axis(plan%handle,axis)
      if(status/=0) return
    enddo
    status=make_velocity(plan%handle)
    if(status/=0) return
    if(plan%halo) then
      do axis=0,2
        status=halo_axis(plan%handle,axis)
        if(status/=0) return
      enddo
    endif
    status=finish_sample(plan%handle)
  end function
  integer(c_int) function sample_destroy(plan) result(status)
    type(sample_plan),intent(inout) :: plan
    status=destroy(plan%handle)
  end function
  integer(c_int) function sample_capture_vector(plan,source,slots) result(status)
    type(sample_plan),intent(in) :: plan
    type(device_view),intent(in) :: source
    integer(c_int),intent(in) :: slots(3)
    integer(c_int) :: axis
    status=import_vector(plan%handle,source,slots)
    if(status/=0) return
    if(plan%halo) then
      do axis=0,2
        status=halo_axis(plan%handle,axis)
        if(status/=0) return
      enddo
    endif
    status=finish_sample(plan%handle)
  end function
end module
