module insitu_index_api
  use iso_c_binding
  use device_state, only: device_view,c_device_real_kind
  implicit none
  private
  public :: index_region,index_begin,index_fill,index_sample,index_receive,index_evaluate,index_download
  public :: index_exchange_prepare,index_exchange,index_exchange_end,index_usage,index_view,index_end
  public :: index_source
  public :: index_statistics
  public :: index_air5_mean
  public :: index_selected
  type,bind(C) :: index_region
    integer(c_int) :: first(3),extent(3),lower(3),shape(3),origin(3),intervals(3),periodic(3)
    integer(c_int) :: cartesian_support,nondimen
    real(c_device_real_kind) :: tempconst,tempconst1,reynolds,prandtl,const5,cp
  end type
  interface
    integer(c_int) function index_selected(parent,state,indices,nfields,ids,cells,source,const2,const6) &
        bind(C,name='astr_insitu_index_selected')
      import c_int,c_ptr,c_device_real_kind,device_view
      type(c_ptr),value :: parent
      type(device_view),intent(in) :: state,indices
      integer(c_int),value :: nfields,source
      integer(c_int),intent(in) :: ids(*),cells(3)
      real(c_device_real_kind),value :: const2,const6
    end function
    integer(c_int) function index_statistics(parent,state,owned,source) bind(C,name='astr_insitu_index_statistics')
      import c_int,c_ptr,device_view
      type(c_ptr),value :: parent
      type(device_view),intent(in) :: state
      integer(c_int),intent(in) :: owned(3)
      integer(c_int),value :: source
    end function
    integer(c_int) function index_air5_mean(parent,duration,source) bind(C,name='astr_insitu_index_air5_mean')
      import c_ptr,c_int,c_double
      type(c_ptr),value :: parent
      real(c_double),value :: duration
      integer(c_int),value :: source
    end function
    integer(c_int) function index_begin(parent,nodes,outputs,nraw,raw,nfields,fields,weights,geometry,xyz,bytes) &
        bind(C,name='astr_insitu_index_begin')
      import c_int,c_ptr,c_device_real_kind,c_size_t
      type(c_ptr),value :: parent
      integer(c_int),value :: nodes,outputs,nraw,nfields,geometry
      integer(c_int),intent(in) :: raw(*),fields(*)
      real(c_device_real_kind),intent(in) :: weights(49),xyz(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function index_fill(parent,nodes,indices,positions) bind(C,name='astr_insitu_index_fill')
      import c_int,c_ptr
      type(c_ptr),value :: parent
      integer(c_int),value :: nodes
      integer(c_int),intent(in) :: indices(*),positions(*)
    end function
    integer(c_int) function index_sample(parent,nodes,indices,values) bind(C,name='astr_insitu_index_sample')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: parent
      integer(c_int),value :: nodes
      integer(c_int),intent(in) :: indices(*)
      real(c_device_real_kind),intent(out) :: values(*)
    end function
    integer(c_int) function index_receive(parent,nodes,positions,values) bind(C,name='astr_insitu_index_receive')
      import c_int,c_ptr,c_device_real_kind
      type(c_ptr),value :: parent
      integer(c_int),value :: nodes
      integer(c_int),intent(in) :: positions(*)
      real(c_device_real_kind),intent(in) :: values(*)
    end function
    integer(c_int) function index_exchange_prepare(parent,capacity,mode) bind(C,name='astr_insitu_index_exchange_prepare')
      import c_int,c_ptr
      type(c_ptr),value :: parent
      integer(c_int),value :: capacity,mode
    end function
    integer(c_int) function index_exchange(parent,sends,indices,receives,positions,destination,source,comm) &
        bind(C,name='astr_insitu_index_exchange')
      import c_int,c_ptr
      type(c_ptr),value :: parent
      integer(c_int),value :: sends,receives,destination,source,comm
      integer(c_int),intent(in) :: indices(*),positions(*)
    end function
    integer(c_int) function index_exchange_end(parent) bind(C,name='astr_insitu_index_exchange_end')
      import c_int,c_ptr
      type(c_ptr),value :: parent
    end function
    integer(c_int) function index_evaluate(parent,region) bind(C,name='astr_insitu_index_evaluate')
      import c_int,c_ptr,index_region
      type(c_ptr),value :: parent
      type(index_region),intent(in) :: region
    end function
    integer(c_int) function index_download(parent,neighborhood,values,bytes) bind(C,name='astr_insitu_index_download')
      import c_int,c_ptr,c_device_real_kind,c_size_t
      type(c_ptr),value :: parent
      integer(c_int),value :: neighborhood
      real(c_device_real_kind),intent(out) :: values(*)
      integer(c_size_t),value :: bytes
    end function
    integer(c_int) function index_view(parent,values,coordinates) bind(C,name='astr_insitu_index_view')
      import c_int,c_ptr
      type(c_ptr),value :: parent
      type(c_ptr),intent(out) :: values,coordinates
    end function
    integer(c_int) function index_source(parent,coordinates,view) bind(C,name='astr_insitu_index_source')
      import c_int,c_ptr,device_view
      type(c_ptr),value :: parent
      integer(c_int),value :: coordinates
      type(device_view),intent(out) :: view
    end function
    integer(c_int) function index_usage(parent,values) bind(C,name='astr_insitu_index_usage')
      import c_int,c_ptr,c_int64_t
      type(c_ptr),value :: parent
      integer(c_int64_t),intent(out) :: values(3)
    end function
    integer(c_int) function index_end(parent) bind(C,name='astr_insitu_index_end')
      import c_int,c_ptr
      type(c_ptr),value :: parent
    end function
  end interface
end module
