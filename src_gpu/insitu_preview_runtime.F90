module insitu_preview_cpp
  use iso_c_binding
  use iso_fortran_env, only: int64
  use mpi
  use astr_precision, only: real_kind,c_real_kind,real_bytes
  use commvar, only: im,jm,km,ia,ja,ka,lihomo,ljhomo,lkhomo,flowtype
  use commarray, only: x
  use parallel, only: mpirank,mpileft,mpiright,mpidown,mpiup,mpiback,mpifront,isize,jsize,ksize,ig0,jg0,kg0
  use checkpoint_state_io, only: checkpoint_state_require
  use fluid_runtime, only: cpp_device_context
  use device_state
  use insitu_sampling_cpp
  use insitu_statistics_cpp, only: statistics_cpp_view,slice_statistics_cpp_info,pack_statistics_cpp_state
  use insitu_fields, only: insitu_owned_counts
  use insitu_run_config, only: insitu_options
  implicit none
  private
  public :: render_preview_device_cpp
  type :: saved_field
    real(real_kind),allocatable :: values(:)
  end type
contains
  subroutine check(status,label)
    integer(c_int),intent(in) :: status
    character(*),intent(in) :: label
    call checkpoint_state_require(status==0,MPI_COMM_WORLD,label)
  end subroutine
  subroutine render_preview_device_cpp(options,step,t,audit)
    type(insitu_options),intent(in) :: options
    integer(int64),intent(in) :: step
    real(real_kind),intent(in) :: t
    logical,intent(in) :: audit
    type(sample_plan) :: sample,means(2)
    type(device_view) :: view,mean_source,mean_view,geometry_view,coordinate_view
    type(c_ptr) :: mean_pointer(2),coordinates,coordinate_halo
    integer(c_int),parameter :: fields(6)=[device_q,device_rho,device_velocity,device_pressure, &
      device_temperature,device_metric]
    type(saved_field) :: before(6)
    real(real_kind),allocatable :: after(:)
    real(real_kind),allocatable :: statistics_before(:,:,:,:),statistics_after(:,:,:,:)
    real(real_kind),allocatable :: positions(:,:)
    integer,allocatable :: seed_before(:),seed_after(:)
    real(real_kind) :: window(2),duration,wstart,wend
    integer :: i,a,owned(3),j,k,node,seed_size,rng_mismatches
    integer(c_int) :: status
    integer(c_size_t) :: free,total
    integer(c_int64_t) :: usage(9)
    integer(int64) :: nodes,halo_nodes,capacity,n1,peak,mismatches,statistics_mismatches,words,retained,statistics_bytes
    logical :: active,covered,ok,wanted(2),physical
    interface
      integer(c_int) function device_render(pipeline,backend,script,profile,comm,step,t,nx,ny,nz,ox,oy,oz,cells,cellsy,cellsz, &
          velocity,halo,derived,reynolds,favre,covered,duration,wstart,wend,host,curve,xyz,xyz_halo,budget,reserve,retained, &
          seeds,temperature,plane_j,diagnostic_components) bind(C,name='astr_insitu_device_render')
        import c_int,c_real_kind,c_char,c_ptr,c_int64_t
        character(c_char),intent(in) :: pipeline(*),backend(*),script(*),profile(*),seeds(*)
        integer(c_int64_t),value :: step
        integer(c_int),value :: comm,nx,ny,nz,ox,oy,oz,cells,cellsy,cellsz,covered,curve,plane_j,diagnostic_components
        real(c_real_kind),value :: t,duration,wstart,wend
        type(c_ptr),value :: velocity,halo,derived,reynolds,favre,xyz,xyz_halo,temperature
        integer(c_int64_t),value :: host,budget,reserve,retained
      end function
    end interface
    mean_pointer=c_null_ptr;wanted=.false.;covered=.false.;duration=0;wstart=0;wend=0
    physical=trim(flowtype)=='bl';coordinates=c_null_ptr;coordinate_halo=c_null_ptr
    owned=insitu_owned_counts()
    if(options%mean_streamline_render) then
      do i=1,options%view_count
        wanted(1)=wanted(1).or.options%views(i)%velocity=='reynolds'
        wanted(2)=wanted(2).or.options%views(i)%velocity=='favre'
      enddo
      call slice_statistics_cpp_info(active,covered,window,duration)
      call checkpoint_state_require(active.and.covered,MPI_COMM_WORLD,'preview checkpoint has no covered saved means')
      wend=min(t,window(2));wstart=wend-duration
      call statistics_cpp_view(mean_source)
    endif
    nodes=product(int([im,jm,km],int64)+1);halo_nodes=product(int([im,jm,km],int64)+7)
    n1=max(int(im+7,int64)*(jm+7),int(im+7,int64)*(km+7),int(jm+7,int64)*(km+7))
    capacity=9*n1
    peak=real_bytes*(2*capacity+max(7*nodes,3*nodes+3*halo_nodes))+4
    peak=peak+count(wanted)*(real_bytes*(2*capacity+3*halo_nodes)+4)
    if(physical) peak=max(peak,real_bytes*(2*capacity+3*nodes+6*halo_nodes)+4)
    statistics_bytes=0
    if(options%statistics) statistics_bytes=30_int64*real_bytes*product(int(owned,int64))+4
    free=0;total=0
    status=device_memory_info(cpp_device_context(),free,total)
    call checkpoint_state_require(status==0.and.peak+statistics_bytes<=options%device_budget_bytes.and. &
      int(free,int64)>=options%device_reserve_bytes.and.peak<=int(free,int64)-options%device_reserve_bytes, &
      MPI_COMM_WORLD,'C++ preview sample budget/free-memory reserve')
#ifdef ASTR_BUILD_TESTING
    if(audit) then
      call random_seed(size=seed_size)
      allocate(seed_before(seed_size),seed_after(seed_size))
      call random_seed(get=seed_before)
      do i=1,size(fields)
        call check(device_state_view(cpp_device_context(),fields(i),view),'preview audit field view')
        words=product(int(view%extent,int64));allocate(before(i)%values(words))
        call check(device_state_download(cpp_device_context(),fields(i),before(i)%values, &
          int(words*real_bytes,c_size_t)),'preview audit before')
      enddo
      if(options%statistics) then
        allocate(statistics_before(owned(1),owned(2),owned(3),34))
        call pack_statistics_cpp_state(statistics_before,ok)
        call checkpoint_state_require(ok,MPI_COMM_WORLD,'preview saved-statistics before audit')
      endif
    endif
#endif
    call check(sample_create(cpp_device_context(),MPI_COMM_WORLD, &
      int([mpileft,mpidown,mpiback],c_int),int([mpiright,mpiup,mpifront],c_int),int([isize,jsize,ksize],c_int), &
      merge(1_c_int,0_c_int,[lihomo,ljhomo,lkhomo]),merge(1_c_int,0_c_int,options%postprocess_transport=='device-aware'), &
      .false.,.true.,0_c_int,.false.,sample),'create read-only preview sample')
    call check(sample_capture(sample),'capture canonical private preview velocity')
    call check(sample_view(sample%handle,sample_halo,view),'preview velocity halo')
    if(physical) then
      allocate(positions(3,nodes));node=0
      do k=0,km
        do j=0,jm
          do i=0,im
            node=node+1;positions(:,node)=x(i,j,k,:)
          enddo
        enddo
      enddo
      call check(sample_upload_coordinates(sample%handle,positions,int(size(positions,kind=int64)*real_bytes,c_size_t), &
        int([ig0,jg0,kg0],c_int),int([ia,ja,ka],c_int)),'upload saved physical preview coordinates')
      deallocate(positions)
      call check(sample_view(sample%handle,sample_geometry,geometry_view),'preview physical coordinates')
      call check(sample_view(sample%handle,sample_coordinates,coordinate_view),'preview physical coordinate halo')
      coordinates=geometry_view%data;coordinate_halo=coordinate_view%data
      write(*,'(a,i0,a,i0)') 'ASTR_CPP_PREVIEW_GEOMETRY rank=',mpirank,' coordinate_upload_bytes=',3*nodes*real_bytes
    endif
    call check(sample_metrics(sample%handle,usage),'preview retained input bytes')
    retained=usage(1)+statistics_bytes
    do a=1,2
      if(.not.wanted(a)) cycle
      call check(sample_create(cpp_device_context(),MPI_COMM_WORLD, &
        int([mpileft,mpidown,mpiback],c_int),int([mpiright,mpiup,mpifront],c_int),int([isize,jsize,ksize],c_int), &
        merge(1_c_int,0_c_int,[lihomo,ljhomo,lkhomo]),merge(1_c_int,0_c_int,options%postprocess_transport=='device-aware'), &
        .false.,.true.,0_c_int,.false.,means(a)),'create saved-mean preview halo')
      call check(sample_capture_owned_vector(means(a)%handle,mean_source,int(owned,c_int), &
        int([6,7,8]+3*(a-1),c_int)),'expand saved means without accumulation')
      call check(sample_view(means(a)%handle,sample_halo,mean_view),'saved-mean preview halo view')
      mean_pointer(a)=mean_view%data
      call check(sample_metrics(means(a)%handle,usage),'saved-mean retained bytes')
      retained=retained+usage(1)
    enddo
    status=device_render(trim(options%rendering_pipeline)//c_null_char,trim(options%implementation_path)//c_null_char, &
      trim(options%pipeline_file)//c_null_char,'streamlines'//c_null_char,int(MPI_COMM_WORLD,c_int),step,t, &
      int(im,c_int),int(jm,c_int),int(km,c_int),int(ig0,c_int),int(jg0,c_int),int(kg0,c_int), &
      int(ia,c_int),int(ja,c_int),int(ka,c_int),c_null_ptr,view%data,c_null_ptr,mean_pointer(1),mean_pointer(2), &
      merge(1_c_int,0_c_int,covered),duration,wstart,wend,options%host_budget_bytes,merge(1_c_int,0_c_int,physical), &
      coordinates,coordinate_halo, &
      options%device_budget_bytes,options%device_reserve_bytes,retained,trim(options%streamline_seeds)//c_null_char, &
      c_null_ptr,0_c_int,0_c_int)
    call check(status,'read-only C++ preview renderer')
#ifdef ASTR_BUILD_TESTING
    if(audit) then
      mismatches=0
      do i=1,size(fields)
        words=size(before(i)%values,kind=int64);allocate(after(words))
        call check(device_state_download(cpp_device_context(),fields(i),after,int(words*real_bytes,c_size_t)), &
          'preview audit after')
        mismatches=mismatches+count(transfer(before(i)%values,[0_int64],int(words))/= &
          transfer(after,[0_int64],int(words)),kind=int64)
        deallocate(after,before(i)%values)
      enddo
      statistics_mismatches=0
      call random_seed(get=seed_after)
      rng_mismatches=count(seed_before/=seed_after)
      if(options%statistics) then
        allocate(statistics_after(owned(1),owned(2),owned(3),34))
        call pack_statistics_cpp_state(statistics_after,ok)
        call checkpoint_state_require(ok,MPI_COMM_WORLD,'preview saved-statistics after audit')
        words=size(statistics_before,kind=int64)
        statistics_mismatches=count(transfer(statistics_before,[0_int64],int(words))/= &
          transfer(statistics_after,[0_int64],int(words)),kind=int64)
      endif
      write(*,'(a,i0,4(a,i0))') 'ASTR_INSITU_PREVIEW_ISOLATION rank=',mpirank,' flow_mismatches=',mismatches, &
        ' statistics_mismatches=',statistics_mismatches,' statistics_loaded=',merge(1,0,options%statistics), &
        ' rng_mismatches=',rng_mismatches
      call checkpoint_state_require(mismatches==0,MPI_COMM_WORLD,'preview changed restored flow/cache/metric bits')
      call checkpoint_state_require(statistics_mismatches==0,MPI_COMM_WORLD,'preview changed saved statistics/clock bits')
      call checkpoint_state_require(rng_mismatches==0,MPI_COMM_WORLD,'preview changed process RNG state')
    endif
#endif
    call check(sample_metrics(sample%handle,usage),'preview transfer receipt')
    write(*,'(a,i0,a,i0,3(a,i0))') 'ASTR_CPP_PREVIEW_SAMPLE rank=',mpirank,' step=',step, &
      ' private_d2h_bytes=',usage(5),' private_h2d_bytes=',usage(6),' explicit_flow_downloads=',usage(8)
    call check(sample_destroy(sample),'release preview private sample')
    do a=1,2
      if(wanted(a)) call check(sample_destroy(means(a)),'release preview saved-mean halo')
    enddo
  end subroutine
end module
