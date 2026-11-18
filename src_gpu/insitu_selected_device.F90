module insitu_selected_device_cpp
  use iso_c_binding
  use iso_fortran_env, only: int64
  use astr_precision, only: real_kind,real_bytes
  use device_state, only: device_view,device_not_allocated
  use fluid_runtime, only: cpp_device_context
  use insitu_moments_cpp
  use insitu_sample_moments, only: sample_profile_size
  use insitu_index_api, only: index_source
  implicit none
  private
  public :: selected_moment_device,allocate_selected_moments_cpp,release_selected_moments_cpp
  public :: selected_moments_present_cpp,selected_moments_bytes_cpp,move_selected_moments_cpp
  public :: download_selected_moments_cpp,upload_selected_moments_cpp,gather_selected_moments_cpp
  public :: selected_profile_workspace_bytes_cpp,selected_external_workspace_bytes_cpp
  public :: reduce_selected_profile_cpp,activate_selected_moments_cpp,selected_active_views_cpp
  public :: selected_moment_transfer_counts_cpp,selected_profile_transfer_counts_cpp
  public :: push_scattered_moments_cpp
  type :: selected_moment_device
    type(c_ptr) :: handle=c_null_ptr
    integer :: nodes=0,components=0
  end type
  type(c_ptr),save :: active=c_null_ptr
  integer,save :: active_fields(5)=0,active_cells(3)=0,active_count=0
  integer(int64),save :: indexed_calls=0,status_bytes=0,index_bytes=0,profile_calls=0,profile_down=0,profile_up=0
contains
  subroutine check(status,where)
    use mpi
    integer(c_int),intent(in) :: status
    character(*),intent(in) :: where
    integer :: ignored
    if(status==0) return
    print *, 'C++ selected statistics: ',where,status
    call MPI_Abort(MPI_COMM_WORLD,int(status),ignored)
  end subroutine
  logical function selected_moments_present_cpp(storage) result(present_state)
    type(selected_moment_device),intent(in) :: storage
    present_state=c_associated(storage%handle)
  end function
  subroutine usage(storage,values)
    type(selected_moment_device),intent(in) :: storage
    integer(c_int64_t),intent(out) :: values(9)
    call check(selected_metrics(storage%handle,values),'owned counters')
  end subroutine
  integer(int64) function selected_moments_bytes_cpp(storage) result(bytes)
    type(selected_moment_device),intent(in) :: storage
    integer(c_int64_t) :: values(9)
    bytes=0
    if(.not.c_associated(storage%handle)) return
    call usage(storage,values);bytes=values(1)
  end function
  integer(int64) function selected_profile_workspace_bytes_cpp(storage) result(bytes)
    type(selected_moment_device),intent(in) :: storage
    integer(c_int64_t) :: values(2)
    bytes=0
    if(.not.c_associated(storage%handle)) return
    call check(selected_workspace(storage%handle,values),'profile bytes');bytes=values(1)
  end function
  integer(int64) function selected_external_workspace_bytes_cpp(storage) result(bytes)
    type(selected_moment_device),intent(in) :: storage
    integer(c_int64_t) :: values(2)
    bytes=0
    if(.not.c_associated(storage%handle)) return
    call check(selected_workspace(storage%handle,values),'external index bytes');bytes=values(2)
  end function
  subroutine allocate_selected_moments_cpp(storage,indices,components,ok)
    type(selected_moment_device),intent(inout) :: storage
    integer,intent(in) :: indices(:,:),components
    logical,intent(out) :: ok
    integer(c_int64_t) :: values(9)
    integer(c_int) :: status
    call release_selected_moments_cpp(storage)
    status=selected_create(cpp_device_context(),int(size(indices,2),c_int),int(indices,c_int),int(components,c_int),storage%handle)
    ok=status==0
    if(.not.ok) return
    storage%nodes=size(indices,2);storage%components=components
    call usage(storage,values);index_bytes=index_bytes+values(6)
  end subroutine
  subroutine release_selected_moments_cpp(storage)
    type(selected_moment_device),intent(inout) :: storage
    if(c_associated(storage%handle)) call check(selected_destroy(storage%handle),'release group')
    storage%nodes=0;storage%components=0;active=c_null_ptr
    active_fields=0;active_cells=0;active_count=0
  end subroutine
  subroutine move_selected_moments_cpp(source,destination)
    type(selected_moment_device),intent(inout) :: source,destination
    destination%handle=source%handle;destination%nodes=source%nodes;destination%components=source%components
    source%handle=c_null_ptr;source%nodes=0;source%components=0
  end subroutine
  subroutine download_selected_moments_cpp(storage,values,ok)
    type(selected_moment_device),intent(in) :: storage
    real(real_kind),contiguous,intent(out) :: values(:,:)
    logical,intent(out) :: ok
    ok=selected_download(storage%handle,values,int(size(values),c_size_t)*real_bytes)==0
  end subroutine
  subroutine upload_selected_moments_cpp(storage,values,ok)
    type(selected_moment_device),intent(inout) :: storage
    real(real_kind),contiguous,intent(in) :: values(:,:)
    logical,intent(out) :: ok
    ok=selected_upload(storage%handle,values,int(size(values),c_size_t)*real_bytes)==0
  end subroutine
  subroutine gather_selected_moments_cpp(storage,ids,const2,const6,weights,previous,covariance,ok)
    type(selected_moment_device),intent(inout) :: storage
    integer,intent(in) :: ids(:)
    real(real_kind),intent(in) :: const2,const6,weights(2)
    logical,intent(in) :: previous,covariance
    logical,intent(out) :: ok
    integer(c_int64_t) :: before(9),after(9)
    integer(c_int) :: status
    call usage(storage,before)
    status=selected_push(storage%handle,int(size(ids),c_int),int(ids,c_int),const2,const6,weights, &
      merge(1_c_int,0_c_int,previous),merge(1_c_int,0_c_int,covariance))
    call usage(storage,after)
    indexed_calls=indexed_calls+(after(3)-before(3))/(storage_size(0_c_int)/8)
    status_bytes=status_bytes+after(3)-before(3)
    index_bytes=index_bytes+after(6)-before(6);ok=status==0
  end subroutine
  subroutine push_scattered_moments_cpp(storage,values_pointer,positions,nfields,weights,previous,ok)
    type(selected_moment_device),intent(inout) :: storage
    type(c_ptr),intent(in) :: values_pointer
    integer,intent(in) :: positions(:),nfields
    real(real_kind),intent(in) :: weights(2)
    logical,intent(in) :: previous
    logical,intent(out) :: ok
    type(device_view) :: values
    integer(c_int64_t) :: before(9),after(9)
    integer(c_int) :: status
    status=index_source(cpp_device_context(),0_c_int,values)
    ok=status==0
    if(.not.ok) return
    values%data=values_pointer
    call usage(storage,before)
    status=selected_scatter(storage%handle,int(size(positions),c_int),int(positions,c_int),values, &
      int(nfields,c_int),weights,merge(1_c_int,0_c_int,previous))
    call usage(storage,after)
    indexed_calls=indexed_calls+(after(3)-before(3))/(storage_size(0_c_int)/8)
    status_bytes=status_bytes+after(3)-before(3);index_bytes=index_bytes+after(6)-before(6)
    ok=status==0
  end subroutine
  subroutine reduce_selected_profile_cpp(storage,ny,weights,nfields,covariance,profile,ok)
    type(selected_moment_device),intent(inout) :: storage
    integer,intent(in) :: ny,nfields
    real(real_kind),intent(in) :: weights(:)
    logical,intent(in) :: covariance
    real(real_kind),allocatable,intent(out) :: profile(:,:)
    logical,intent(out) :: ok
    integer :: status,nc
    integer(c_int64_t) :: before(9),after(9)
    nc=sample_profile_size(nfields,covariance)
    allocate(profile(nc,ny),stat=status);ok=status==0
    if(.not.ok) return
    call usage(storage,before)
    status=selected_profile(storage%handle,int(ny,c_int),int(size(weights),c_int),weights,int(nfields,c_int), &
      merge(1_c_int,0_c_int,covariance),profile,int(size(profile),c_size_t)*real_bytes)
    call usage(storage,after)
    if(ny>0) profile_calls=profile_calls+1
    profile_down=profile_down+after(8)-before(8);profile_up=profile_up+after(9)-before(9)
    ok=status==0
  end subroutine
  subroutine activate_selected_moments_cpp(storage,ids,cells,nodes)
    type(selected_moment_device),intent(in) :: storage
    integer,intent(in) :: ids(:),cells(3),nodes
    active=storage%handle;active_fields=0;active_fields(:size(ids))=ids;active_cells=cells;active_count=size(ids)
  end subroutine
  integer(c_int) function selected_active_views_cpp(state,indices,ids,cells,nfields) result(status)
    type(device_view),intent(out) :: state,indices
    integer(c_int),intent(out) :: ids(5),cells(3),nfields
    status=device_not_allocated
    if(.not.c_associated(active)) return
    status=selected_view(active,0_c_int,state)
    if(status==0) status=selected_view(active,1_c_int,indices)
    ids=int(active_fields,c_int);cells=int(active_cells,c_int);nfields=int(active_count,c_int)
  end function
  subroutine selected_moment_transfer_counts_cpp(calls,status,index)
    integer(int64),intent(out) :: calls,status,index
    calls=indexed_calls;status=status_bytes;index=index_bytes
  end subroutine
  subroutine selected_profile_transfer_counts_cpp(calls,download,upload)
    integer(int64),intent(out) :: calls,download,upload
    calls=profile_calls;download=profile_down;upload=profile_up
  end subroutine
end module
