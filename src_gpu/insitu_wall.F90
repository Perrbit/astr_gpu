module insitu_wall_cpp
  use iso_c_binding
  use device_state, only: device_view,c_device_real_kind
  implicit none
  private
  public :: wall_create,wall_capture,wall_view,wall_line,wall_download,wall_metrics,wall_destroy
  public :: wall_air5_span
  public :: wall_statistics_push,wall_statistics_pack,wall_statistics_restore,wall_statistics_mean
  public :: sbli_wall_create,sbli_wall_capture,sbli_wall_metrics,sbli_wall_destroy
  public :: profile_wall_create,profile_wall_capture,profile_wall_metrics,profile_wall_destroy
  interface
    integer(c_int) function profile_wall_create(parent,offset,global_x,material,reference,weights,sample) &
        bind(C,name='astr_insitu_profile_wall_create')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: parent
      integer(c_int),value :: offset,global_x
      real(c_device_real_kind),intent(in) :: material(7),reference(2),weights(*)
      type(c_ptr),intent(out) :: sample
    end function
    integer(c_int) function profile_wall_capture(sample,wall,bytes) bind(C,name='astr_insitu_profile_wall_capture')
      import c_ptr,c_int,c_device_real_kind,c_size_t
      type(c_ptr),value :: sample
      real(c_device_real_kind),intent(out) :: wall(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function profile_wall_metrics(sample,values) bind(C,name='astr_insitu_profile_wall_metrics')
      import c_ptr,c_int,c_int64_t
      type(c_ptr),value :: sample
      integer(c_int64_t),intent(out) :: values(5)
    end function
    integer(c_int) function profile_wall_destroy(sample) bind(C,name='astr_insitu_profile_wall_destroy')
      import c_ptr,c_int
      type(c_ptr),intent(inout) :: sample
    end function
    integer(c_int) function wall_create(parent,comm,lower,upper,sizes,periodic,mode,first,nw,layers,curved, &
        material,geometry,wall) bind(C,name='astr_insitu_wall_create')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: parent
      integer(c_int),value :: comm,mode,first,nw,layers,curved
      integer(c_int),intent(in) :: lower(3),upper(3),sizes(3),periodic(3)
      real(c_device_real_kind),intent(in) :: material(7),geometry(*)
      type(c_ptr),intent(out) :: wall
    end function
    integer(c_int) function wall_capture(wall) bind(C,name='astr_insitu_wall_capture')
      import c_ptr,c_int
      type(c_ptr),value :: wall
    end function
    integer(c_int) function wall_view(wall,geometry,view) bind(C,name='astr_insitu_wall_view')
      import c_ptr,c_int,device_view
      type(c_ptr),value :: wall
      integer(c_int),value :: geometry
      type(device_view),intent(out) :: view
    end function
    integer(c_int) function wall_line(wall,k,side,host,bytes) bind(C,name='astr_insitu_wall_line')
      import c_ptr,c_int,c_size_t,c_device_real_kind
      type(c_ptr),value :: wall
      integer(c_int),value :: k,side
      real(c_device_real_kind),intent(out) :: host(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function wall_download(wall,host,bytes) bind(C,name='astr_insitu_wall_download')
      import c_ptr,c_int,c_size_t,c_device_real_kind
      type(c_ptr),value :: wall
      real(c_device_real_kind),intent(out) :: host(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function wall_air5_span(wall,weights,owned_x,profile) bind(C,name='astr_insitu_wall_air5_span')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: wall
      real(c_device_real_kind),intent(in) :: weights(*)
      integer(c_int),value :: owned_x
      real(c_device_real_kind),intent(out) :: profile(*)
    end function
    integer(c_int) function wall_metrics(wall,values) bind(C,name='astr_insitu_wall_metrics')
      import c_ptr,c_int,c_int64_t
      type(c_ptr),value :: wall
      integer(c_int64_t),intent(out) :: values(10)
    end function
    integer(c_int) function wall_destroy(wall) bind(C,name='astr_insitu_wall_destroy')
      import c_ptr,c_int
      type(c_ptr),intent(inout) :: wall
    end function
    integer(c_int) function wall_statistics_push(wall,weights,previous) bind(C,name='astr_insitu_wall_statistics_push')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: wall
      real(c_device_real_kind),intent(in) :: weights(2)
      integer(c_int),value :: previous
    end function
    integer(c_int) function wall_statistics_pack(wall,host,bytes,window,time) bind(C,name='astr_insitu_wall_statistics_pack')
      import c_ptr,c_int,c_size_t,c_device_real_kind
      type(c_ptr),value :: wall
      real(c_device_real_kind),intent(out) :: host(*)
      integer(c_size_t),value :: bytes
      real(c_device_real_kind),intent(in) :: window(2)
      real(c_device_real_kind),value :: time
    end function
    integer(c_int) function wall_statistics_restore(wall,host,bytes) bind(C,name='astr_insitu_wall_statistics_restore')
      import c_ptr,c_int,c_size_t,c_device_real_kind
      type(c_ptr),value :: wall
      real(c_device_real_kind),intent(in) :: host(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function wall_statistics_mean(wall,history,duration) bind(C,name='astr_insitu_wall_statistics_mean')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: wall,history
      real(c_device_real_kind),value :: duration
    end function
    integer(c_int) function sbli_wall_create(parent,comm,lower,upper,parts,mode,first,fraction,material,reference,x,sample) &
        bind(C,name='astr_insitu_sbli_wall_create')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: parent
      integer(c_int),value :: comm,lower,upper,parts,mode,first
      real(c_device_real_kind),value :: fraction
      real(c_device_real_kind),intent(in) :: material(7),reference(3),x(*)
      type(c_ptr),intent(out) :: sample
    end function
    integer(c_int) function sbli_wall_capture(sample,wall,bytes) bind(C,name='astr_insitu_sbli_wall_capture')
      import c_ptr,c_int,c_device_real_kind,c_size_t
      type(c_ptr),value :: sample
      real(c_device_real_kind),intent(out) :: wall(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function sbli_wall_metrics(sample,values) bind(C,name='astr_insitu_sbli_wall_metrics')
      import c_ptr,c_int,c_int64_t
      type(c_ptr),value :: sample
      integer(c_int64_t),intent(out) :: values(8)
    end function
    integer(c_int) function sbli_wall_destroy(sample) bind(C,name='astr_insitu_sbli_wall_destroy')
      import c_ptr,c_int
      type(c_ptr),intent(inout) :: sample
    end function
  end interface
end module
