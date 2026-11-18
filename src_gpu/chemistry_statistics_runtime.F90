module chemistry_mean_statistics_cpp
  use iso_c_binding,only:c_ptr,c_null_ptr,c_int,c_size_t,c_associated
  use iso_fortran_env,only:int64,real64
  use ieee_arithmetic,only:ieee_is_finite
  use mpi,only:MPI_COMM_WORLD
  use astr_precision,only:real_kind,real_bytes
  use device_state,only:device_view,device_state_view,device_state_upload,device_state_download, &
    device_air5_mean_sum,device_air5_mean_endpoint
  use chemistry_device_statistics
  use checkpoint_state_io,only:checkpoint_state_identity,checkpoint_state_transfer, &
    allocate_checkpoint_buffer,checkpoint_state_require
  implicit none
  private
  public :: initialize_air5_mean_statistics_cpp,release_air5_mean_statistics_cpp
  public :: set_air5_mean_sampling_cpp,accumulate_air5_mean_statistics_cpp,reset_air5_mean_statistics_cpp
  public :: complete_air5_mean_statistics_cpp_file,slice_air5_mean_info_cpp,air5_mean_statistics_cpp_device_bytes
  type(c_ptr),save :: context=c_null_ptr
  logical,save :: active=.false.,sampling_enabled=.false.
  integer(int64),save :: sample_count=0,first_step=0,last_step=-1
  real(real64),save :: first_time=0,last_time=0,time_weight=0
contains
  subroutine check(status,label)
    use mpi,only:MPI_Abort,MPI_Comm_rank
    integer,intent(in) :: status
    character(*),intent(in) :: label
    integer :: rank,ierr
    if(status==0)return
    call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
    write(*,'(a,i0,2a,i0)')'C++ AIR5 mean rank=',rank,' ',label,status
    call MPI_Abort(MPI_COMM_WORLD,status,ierr)
  end subroutine

  subroutine initialize_air5_mean_statistics_cpp(handle,device_budget)
    use commvar,only:im,jm,km,numq,num_species,num_modequ,nondimen,diffterm
    use statistic,only:nsamples
    type(c_ptr),intent(in) :: handle
    integer(int64),intent(in) :: device_budget
    integer(int64) :: bytes
    call checkpoint_state_require(.not.active,MPI_COMM_WORLD,'C++ AIR5 mean statistics already initialized')
    call checkpoint_state_require(real_bytes==8.and.numq==11.and.num_species==5.and.num_modequ==1.and. &
      .not.nondimen.and.diffterm,MPI_COMM_WORLD,'C++ AIR5 mean statistics configuration')
    bytes=int(im+1,int64)*(jm+1)*(km+1)*58*8+4
    call checkpoint_state_require(bytes<=device_budget,MPI_COMM_WORLD,'AIR5 mean statistics device budget')
    context=handle
    call check(air5_mean_prepare(context),'C++ AIR5 mean statistics allocation')
    sample_count=0;first_step=0;last_step=-1;first_time=0;last_time=0;time_weight=0
    nsamples=0;sampling_enabled=.false.;active=.true.
  end subroutine

  subroutine release_air5_mean_statistics_cpp()
    if(c_associated(context))call check(air5_mean_release(context),'release C++ AIR5 mean statistics')
    context=c_null_ptr;active=.false.;sampling_enabled=.false.
  end subroutine

  subroutine set_air5_mean_sampling_cpp(enabled)
    logical,intent(in) :: enabled
    sampling_enabled=enabled
  end subroutine

  subroutine reset_air5_mean_statistics_cpp()
    use statistic,only:nsamples
    if(active)call check(air5_mean_reset(context),'reset C++ AIR5 mean statistics')
    sample_count=0;first_step=0;last_step=-1;first_time=0;last_time=0;time_weight=0
    nsamples=0
  end subroutine

  subroutine slice_air5_mean_info_cpp(enabled,covered,count,window,duration)
    use commvar,only:lavg
    logical,intent(out) :: enabled,covered
    integer(int64),intent(out) :: count
    real(real64),intent(out) :: window(2)
    real(real64),optional,intent(out) :: duration
    enabled=active.and.lavg;covered=enabled.and.time_weight>0
    count=sample_count;window=[first_time,last_time]
    if(present(duration))duration=time_weight
  end subroutine

  integer(int64) function air5_mean_statistics_cpp_device_bytes() result(bytes)
    type(device_view) :: view
    bytes=0
    if(.not.active)return
    call check(device_state_view(context,device_air5_mean_sum,view),'AIR5 mean sum view')
    bytes=product(int(view%extent,int64))*view%element_bytes
    call check(device_state_view(context,device_air5_mean_endpoint,view),'AIR5 mean endpoint view')
    bytes=bytes+product(int(view%extent,int64))*view%element_bytes+4
  end function

  subroutine accumulate_air5_mean_statistics_cpp()
    use commvar,only:nstep,time,lavg
    use statistic,only:nsamples
    real(real64) :: interval
    integer(c_int) :: failed
    if(.not.active)return
    if(.not.sampling_enabled.or..not.lavg.or.int(nstep,int64)<=last_step)return
    interval=0
    if(sample_count>0)interval=time-last_time
    call checkpoint_state_require(ieee_is_finite(time).and.ieee_is_finite(interval).and. &
      (sample_count==0.or.interval>0),MPI_COMM_WORLD,'invalid AIR5 mean sampling clock')
    call checkpoint_state_require(sample_count<int(huge(nsamples),int64),MPI_COMM_WORLD,'AIR5 mean sample count overflow')
    call check(air5_mean_sample(context,interval,merge(1_c_int,0_c_int,sample_count>0)),'C++ AIR5 mean sample launch')
    call check(air5_mean_check(context,failed),'C++ AIR5 mean sample status')
    call checkpoint_state_require(failed==0,MPI_COMM_WORLD,'AIR5 mean sampling transport or finite-state failure')
    if(sample_count==0)then
      first_step=int(nstep,int64);first_time=time
    endif
    sample_count=sample_count+1;time_weight=time_weight+interval
    last_step=int(nstep,int64);last_time=time;nsamples=sample_count
  end subroutine

  subroutine complete_air5_mean_statistics_cpp_file(path,writing,identity,budget,allow_repartition,repartition_axes)
    use commvar,only:ia,ja,ka,im,jm,km
    use parallel,only:ig0,jg0,kg0
    use statistic,only:nsamples
    character(*),intent(in) :: path
    logical,intent(in) :: writing
    logical,optional,intent(in) :: allow_repartition,repartition_axes(3)
    type(checkpoint_state_identity),intent(inout) :: identity
    integer(int64),intent(in) :: budget
    real(real_kind),allocatable :: values(:,:,:,:)
    integer(int64) :: remaining,metadata(9)
    logical :: axes(3)
    type(checkpoint_state_identity) :: expected
    call checkpoint_state_require(active,MPI_COMM_WORLD,'AIR5 mean statistics not initialized')
    call allocate_checkpoint_buffer([im,jm,km],0,58,budget,values,remaining,MPI_COMM_WORLD)
    expected=identity
    axes=[.false.,.false.,.true.]
    if(present(repartition_axes))axes=repartition_axes
    metadata=[3_int64,sample_count,first_step,max(0_int64,last_step),transfer(first_time,0_int64), &
      merge(1_int64,0_int64,sample_count>0),merge(1_int64,0_int64,sample_count>0), &
      transfer(last_time,0_int64),transfer(time_weight,0_int64)]
    if(writing)then
      call checkpoint_state_require(sample_count==nsamples.and. &
        (last_step<=identity%step.or.(sample_count==0.and.identity%step==0)),MPI_COMM_WORLD,'AIR5 mean sampling identity')
      call check(device_state_download(context,device_air5_mean_sum,values(:,:,:,1:44), &
        int(size(values(:,:,:,1:44)),c_size_t)*8_c_size_t),'AIR5 mean sum download')
      call check(device_state_download(context,device_air5_mean_endpoint,values(:,:,:,45:58), &
        int(size(values(:,:,:,45:58)),c_size_t)*8_c_size_t),'AIR5 mean endpoint download')
    endif
    call checkpoint_state_transfer(path,writing,.true.,[ia,ja,ka]+1,[ig0,jg0,kg0],[im,jm,km],0,values, &
      identity,remaining,MPI_COMM_WORLD,role=6,metadata=metadata,allow_repartition=allow_repartition,repartition_axes=axes)
    call checkpoint_state_require(all(ieee_is_finite(values)),MPI_COMM_WORLD,'nonfinite AIR5 mean payload')
    call checkpoint_state_require(metadata(1)==3.and.metadata(2)>=0.and.metadata(2)<=huge(nsamples).and. &
      metadata(3)>=0.and.metadata(3)<=identity%step.and.metadata(4)>=0.and.metadata(4)<=identity%step.and. &
      metadata(6)==merge(1_int64,0_int64,metadata(2)>0).and.metadata(7)==metadata(6).and. &
      ieee_is_finite(transfer(metadata(5),0._real64)).and.ieee_is_finite(transfer(metadata(8),0._real64)).and. &
      transfer(metadata(5),0._real64)>=0.and.transfer(metadata(8),0._real64)>=transfer(metadata(5),0._real64).and. &
      transfer(metadata(8),0._real64)<=identity%time.and.ieee_is_finite(transfer(metadata(9),0._real64)).and. &
      transfer(metadata(9),0._real64)>=0,MPI_COMM_WORLD,'AIR5 mean checkpoint metadata')
    if(metadata(2)>0)then
      call checkpoint_state_require(metadata(3)>=0.and.metadata(3)<=metadata(4).and. &
        metadata(2)<=metadata(4)-metadata(3)+1,MPI_COMM_WORLD,'AIR5 mean sample sequence')
    else
      call checkpoint_state_require(all(metadata(3:9)==0).and.all(values==0),MPI_COMM_WORLD,'nonempty zero-sample AIR5 mean checkpoint')
    endif
    if(.not.writing)then
      call checkpoint_state_require(identity%step==expected%step.and.identity%time==expected%time.and. &
        identity%dt_used==expected%dt_used.and.identity%dt_next==expected%dt_next,MPI_COMM_WORLD,'AIR5 mean checkpoint clock mismatch')
      call check(device_state_upload(context,device_air5_mean_sum,values(:,:,:,1:44), &
        int(size(values(:,:,:,1:44)),c_size_t)*8_c_size_t),'AIR5 mean sum restore')
      call check(device_state_upload(context,device_air5_mean_endpoint,values(:,:,:,45:58), &
        int(size(values(:,:,:,45:58)),c_size_t)*8_c_size_t),'AIR5 mean endpoint restore')
      sample_count=metadata(2);first_step=metadata(3);last_step=metadata(4)
      first_time=transfer(metadata(5),0._real64);last_time=transfer(metadata(8),0._real64);time_weight=transfer(metadata(9),0._real64)
      if(sample_count==0)last_step=-1
      nsamples=sample_count
    endif
  end subroutine
end module
