module index_slice_cpp
  use iso_c_binding
  use iso_fortran_env, only: int64
  use astr_precision, only: real_kind,real_bytes
  use commvar, only: im,jm,km,nondimen,tempconst,tempconst1,reynolds,prandtl,const5,cp,const2,const6
  use commarray, only: x
  use fluid_runtime, only: cpp_device_context
  use insitu_index_api
  use insitu_sampling_cpp, only: sample_plan,sample_create,sample_capture_vector,sample_capture_coordinates, &
    sample_destroy,sample_view,sample_halo,sample_coordinates,sample_metrics
  use device_state, only: device_view
  use insitu_statistics_cpp, only: statistics_cpp_view
  use insitu_fields, only: insitu_owned_counts
  use insitu_selected_device_cpp, only: selected_active_views_cpp
  implicit none
  private
  public :: allocate_index_slice_cpp,release_index_slice_cpp,fill_index_slice_cpp
  public :: sample_remote_slice_cpp,receive_index_slice_cpp,evaluate_index_slice_cpp,download_index_slice_cpp
  public :: index_slice_device_view_cpp,download_index_neighborhood_cpp
  public :: begin_index_slice_exchange_cpp,exchange_index_slice_cpp,end_index_slice_exchange_cpp,index_slice_exchange_usage_cpp
  public :: validate_index_slice_transport_cpp
  public :: begin_catalog_streamlines_cpp,end_catalog_streamlines_cpp,catalog_streamline_pointers_cpp
  type(sample_plan),save :: line_sample
  logical,save :: line_physical=.false.
  integer,save :: packet_first(3)=0,packet_extent(3)=0,packet_origin(3)=0
contains
  logical function select_statistics(source) result(ok)
    integer,intent(in) :: source
    type(device_view) :: view,indices
    integer(c_int) :: ids(5),cells(3),nf,status
    ok=.false.
    select case(source)
    case(0:2)
      view%data=c_null_ptr
      if(source/=0) call statistics_cpp_view(view)
      ok=index_statistics(cpp_device_context(),view,int(insitu_owned_counts(),c_int),int(source,c_int))==0
    case(5:8)
      status=selected_active_views_cpp(view,indices,ids,cells,nf)
      if(status==0) status=index_selected(cpp_device_context(),view,indices,nf,ids,cells,int(source,c_int),const2,const6)
      ok=status==0
    end select
  end function
  subroutine validate_index_slice_transport_cpp(mode,ok,message)
    character(*),intent(in) :: mode
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    ok=mode=='pinned'.or.mode=='device-aware';message=''
    if(.not.ok) message='private C++ image transport must be pinned or device-aware'
  end subroutine
  subroutine allocate_index_slice_cpp(nodes,outputs,raw,fields,weights,ok,device_geometry)
    integer,intent(in) :: nodes,outputs,raw(:),fields(:)
    real(real_kind),intent(in) :: weights(7,0:6)
    logical,intent(out) :: ok
    logical,optional,intent(in) :: device_geometry
    logical :: geometry
    integer(c_int) :: status
    geometry=.false.;if(present(device_geometry)) geometry=device_geometry
    status=index_begin(cpp_device_context(),int(nodes,c_int),int(outputs,c_int),int(size(raw),c_int),int(raw,c_int), &
      int(size(fields),c_int),int(fields,c_int),reshape(weights,[49]),merge(1_c_int,0_c_int,geometry),x, &
      int(size(x,kind=int64)*real_bytes,c_size_t))
    ok=status==0
    if(.not.ok) print *, 'C++ index allocation status=',status
  end subroutine
  subroutine release_index_slice_cpp()
    call require_transfer(index_end(cpp_device_context()))
  end subroutine
  subroutine fill_index_slice_cpp(indices,positions,source,count,ok)
    integer,intent(in) :: indices(:,:),positions(:),source
    integer(int64),intent(in) :: count
    logical,intent(out) :: ok
    ok=select_statistics(source);if(.not.ok) return
    ok=index_fill(cpp_device_context(),int(size(positions),c_int),int(indices,c_int),int(positions,c_int))==0
  end subroutine
  subroutine sample_remote_slice_cpp(indices,source,count,values,ok)
    integer,intent(in) :: indices(:,:),source
    integer(int64),intent(in) :: count
    real(real_kind),intent(out) :: values(:,:)
    logical,intent(out) :: ok
    ok=select_statistics(source);if(.not.ok) return
    ok=index_sample(cpp_device_context(),int(size(indices,2),c_int),int(indices,c_int),values)==0
  end subroutine
  subroutine receive_index_slice_cpp(positions,values,ok)
    integer,intent(in) :: positions(:)
    real(real_kind),intent(in) :: values(:,:)
    logical,intent(out) :: ok
    ok=index_receive(cpp_device_context(),int(size(positions),c_int),int(positions,c_int),values)==0
  end subroutine
  subroutine evaluate_index_slice_cpp(first,extent,lower,shape,origin,intervals,periodic,source,count,ok,cartesian_support)
    integer,intent(in) :: first(3),extent(3),lower(3),shape(3),origin(3),intervals(3),source
    logical,intent(in) :: periodic(3)
    integer(int64),intent(in) :: count
    logical,intent(out) :: ok
    logical,optional,intent(in) :: cartesian_support
    type(index_region) :: region
    ok=select_statistics(source);if(.not.ok) return
    region%first=int(first,c_int);region%extent=int(extent,c_int);region%lower=int(lower,c_int)
    region%shape=int(shape,c_int);region%origin=int(origin,c_int);region%intervals=int(intervals,c_int)
    region%periodic=merge(1_c_int,0_c_int,periodic);region%cartesian_support=0
    if(present(cartesian_support)) region%cartesian_support=merge(1_c_int,0_c_int,cartesian_support)
    region%nondimen=merge(1_c_int,0_c_int,nondimen)
    region%tempconst=tempconst;region%tempconst1=tempconst1;region%reynolds=reynolds
    region%prandtl=prandtl;region%const5=const5;region%cp=cp
    ok=index_evaluate(cpp_device_context(),region)==0
    if(ok) then
      packet_first=first;packet_extent=extent;packet_origin=origin
    endif
  end subroutine
  subroutine download_index_slice_cpp(values,ok)
    real(real_kind),intent(out) :: values(:,:)
    logical,intent(out) :: ok
    ok=index_download(cpp_device_context(),0_c_int,values,int(size(values,kind=int64)*real_bytes,c_size_t))==0
  end subroutine
  subroutine download_index_neighborhood_cpp(values,ok)
    real(real_kind),intent(out) :: values(:,:)
    logical,intent(out) :: ok
    ok=index_download(cpp_device_context(),1_c_int,values,int(size(values,kind=int64)*real_bytes,c_size_t))==0
  end subroutine
  subroutine index_slice_device_view_cpp(values,coordinates)
    type(c_ptr),intent(out) :: values
    type(c_ptr),optional,intent(out) :: coordinates
    type(c_ptr) :: xyz
    call require_transfer(index_view(cpp_device_context(),values,xyz))
    if(present(coordinates)) coordinates=xyz
  end subroutine
  subroutine begin_index_slice_exchange_cpp(capacity,mode,ok,message)
    integer,intent(in) :: capacity
    character(*),intent(in) :: mode
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    integer(c_int) :: status
    call validate_index_slice_transport_cpp(mode,ok,message)
    if(.not.ok) return
    status=index_exchange_prepare(cpp_device_context(),int(capacity,c_int),merge(1_c_int,0_c_int,mode=='device-aware'))
    ok=status==0
    if(.not.ok) write(message,'(a,i0)') 'cannot allocate C++ private image exchange: ',status
  end subroutine
  subroutine exchange_index_slice_cpp(indices,positions,source,count,destination,sender,comm)
    integer,intent(in) :: indices(:,:),positions(:),source,destination,sender,comm
    integer(int64),intent(in) :: count
    if(.not.select_statistics(source)) call require_transfer(-1_c_int)
    call require_transfer(index_exchange(cpp_device_context(),int(size(indices,2),c_int),int(indices,c_int), &
      int(size(positions),c_int),int(positions,c_int),int(destination,c_int),int(sender,c_int),int(comm,c_int)))
  end subroutine
  subroutine end_index_slice_exchange_cpp()
    call require_transfer(index_exchange_end(cpp_device_context()))
  end subroutine
  subroutine index_slice_exchange_usage_cpp(download,upload)
    integer(int64),intent(out) :: download,upload
    integer(c_int64_t) :: values(3)
    call require_transfer(index_usage(cpp_device_context(),values))
    download=values(1);upload=values(2)
  end subroutine
  subroutine begin_catalog_streamlines_cpp(mode,values_pointer,coordinates_pointer,fields,physical, &
      host_budget,device_budget,reserve,retained,ok,message)
    use mpi
    use commvar, only: ia,ja,ka,lihomo,ljhomo,lkhomo
    use parallel, only: ig0,jg0,kg0,isize,jsize,ksize,mpileft,mpiright,mpidown,mpiup,mpiback,mpifront
    character(*),intent(in) :: mode
    type(c_ptr),intent(in) :: values_pointer,coordinates_pointer
    integer,intent(in) :: fields(:)
    logical,intent(in) :: physical
    integer(int64),intent(in) :: host_budget,device_budget,reserve,retained
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    integer(c_int) :: status,slots(3)
    integer :: d
    type(device_view) :: source
    ok=.false.;message=''
    line_physical=physical
    if(any(packet_extent/=[im+1,jm+1,km+1]).or.any(packet_first/=packet_origin)) then
      message='C++ streamline supply currently needs the complete rank-local volume';return
    endif
    do d=1,3
      slots(d)=int(findloc(fields,d+1,dim=1)-1,c_int)
    enddo
    if(any(slots<0)) then
      message='streamline source does not contain all three velocity components';return
    endif
    status=index_source(cpp_device_context(),0_c_int,source)
    if(status/=0) then
      message='C++ selected streamline source is not allocated';return
    endif
    source%data=values_pointer
    status=sample_create(cpp_device_context(),MPI_COMM_WORLD, &
      int([mpileft,mpidown,mpiback],c_int),int([mpiright,mpiup,mpifront],c_int), &
      int([isize,jsize,ksize],c_int),merge(1_c_int,0_c_int,[lihomo,ljhomo,lkhomo]), &
      merge(1_c_int,0_c_int,mode=='device-aware'),.false.,.true.,0_c_int,.false.,line_sample)
    if(status==0) status=sample_capture_vector(line_sample,source,slots)
    if(status==0.and.physical) then
      status=index_source(cpp_device_context(),1_c_int,source)
      source%data=coordinates_pointer
      if(status==0) status=sample_capture_coordinates(line_sample%handle,source,int([ig0,jg0,kg0],c_int), &
        int([ia,ja,ka],c_int))
    endif
    ok=status==0
    if(.not.ok) write(message,'(a,i0)') 'C++ instantaneous streamline supply failed: ',status
  end subroutine
  subroutine catalog_streamline_pointers_cpp(velocity,coordinates)
    type(c_ptr),intent(out) :: velocity,coordinates
    type(device_view) :: view
    call require_transfer(sample_view(line_sample%handle,sample_halo,view))
    velocity=view%data;coordinates=c_null_ptr
    if(line_physical) then
      call require_transfer(sample_view(line_sample%handle,sample_coordinates,view))
      coordinates=view%data
    endif
  end subroutine
  subroutine end_catalog_streamlines_cpp(step)
    use parallel, only: mpirank
    integer(int64),intent(in) :: step
    integer(c_int64_t) :: values(9)
    if(.not.c_associated(line_sample%handle)) return
    call require_transfer(sample_metrics(line_sample%handle,values))
    write(*,'(a,i0,a,i0,a,i0,a,i0)') 'ASTR_CPP_LINE_SUPPLY rank=',mpirank,' step=',step, &
      ' private_d2h_bytes=',values(5),' private_h2d_bytes=',values(6)
    call require_transfer(sample_destroy(line_sample))
    line_physical=.false.
  end subroutine
  subroutine require_transfer(status)
    use mpi
    integer(c_int),intent(in) :: status
    integer :: ignored
    if(status==0) return
    print *, 'C++ index transfer status=',status
    call MPI_Abort(MPI_COMM_WORLD,int(status),ignored)
  end subroutine
end module
