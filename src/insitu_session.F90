module insitu_session
  use mpi
  use iso_fortran_env, only: int32,int64,real64
  use insitu_schedule, only: sample_schedule,configure_schedule,poll_schedule,write_schedule_state,restore_schedule_state
  use insitu_checkpoint_batch
  use insitu_velocity_statistics
  use insitu_resource_budget, only: resource_budget,configure_budget,reserve_bytes,checked_bytes
  use insitu_spatial_statistics, only: spatial_mean
  use insitu_fields, only: capture_sample,canonicalize_sample,derive_sample,complete_periodic_endpoints
  use insitu_run_config, only: insitu_options
  use insitu_config_collective, only: read_insitu_options_collective
  use iso_c_binding, only: c_int,c_double,c_char,c_null_char,c_int64_t
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: sample_insitu_step,finish_insitu,begin_insitu
  public :: save_insitu_pair,prepare_insitu_pair
  public :: capture_insitu_cpu_checkpoint,restore_insitu_cpu_checkpoint
  real(real64),allocatable,save :: cpu_checkpoint_q(:,:,:,:)
  logical,save :: configured=.false.,enabled=.false.
  logical,save :: session_checked=.false.,formal=.false.
  integer(int64),save :: native_flow_downloads=0
  type(insitu_options),save :: options
  real(real64),allocatable,save :: final_statistics(:,:,:,:)
  real(real64),save :: final_statistics_meta(10)=0.d0
  character(1024),save :: prefix=''
  type(sample_schedule),save :: render_schedule
  logical,save :: render_configured=.false.
  logical,save :: native_bridge_configured=.false.
  logical,save :: render_final=.false.
  real(real64),allocatable,save :: last_xyz(:,:,:,:),last_fields(:,:,:,:),last_derived(:,:,:,:)
  integer,save :: last_step=-1
  real(real64),save :: last_time=0.d0
  type(velocity_statistics),allocatable,save :: point_statistics(:,:,:)
  type(velocity_statistics),save :: regional_statistics
  logical,save :: statistics_configured=.false.,statistics_enabled=.false.
  logical,save :: oracle_io=.true.
  real(real64),save :: statistics_window(2)=0.d0
  integer(int64),save :: statistics_host_limit=0
  integer,save :: statistics_step=-1,restore_step=-1
  real(real64),save :: statistics_time=0.d0,restore_time=0.d0
  character(1024),save :: restore_batch=''
#if defined(ASTR_WITH_CATALYST) && defined(_CUDA)
  interface
    integer(c_int) function resource_begin(comm,host,device,reserve,output) bind(C,name='astr_insitu_resource_begin')
      import c_int,c_int64_t,c_char
      integer(c_int),value :: comm
      integer(c_int64_t),value :: host,device,reserve
      character(c_char),intent(in) :: output(*)
    end function
    integer(c_int) function resource_finish() bind(C,name='astr_insitu_resource_finish')
      import c_int
    end function
    integer(c_int) function resource_check(stage) bind(C,name='astr_insitu_resource_check')
      import c_int,c_char
      character(c_char),intent(in) :: stage(*)
    end function
  end interface
#endif
#ifdef ASTR_WITH_CATALYST
  interface
    integer(c_int) function mesh_configure(output,comm) bind(C,name='astr_insitu_mesh_configure')
      import c_int,c_char
      character(c_char),intent(in) :: output(*)
      integer(c_int),value :: comm
    end function
    integer(c_int) function mesh_finish() bind(C,name='astr_insitu_mesh_finish')
      import c_int
    end function
    integer(c_int) function mesh_probe(backend,script,comm,nx,ny,nz,step,t,xyz,f,d,has_mean,means) &
        bind(C,name='astr_insitu_mesh_probe')
      import c_int,c_double,c_char
      character(c_char),intent(in) :: backend(*),script(*)
      integer(c_int),value :: comm,nx,ny,nz,step,has_mean
      real(c_double),value :: t
      real(c_double),intent(in) :: xyz(*),f(*),d(*),means(*)
    end function
  end interface
#endif
contains
  logical function native_device_statistics()
#ifdef _CUDA
    use commvar, only: use_gpu
    native_device_statistics=formal.and.enabled.and.statistics_enabled.and.use_gpu
#else
    native_device_statistics=.false.
#endif
  end function

  subroutine require_sample(ok,message)
    logical,intent(in) :: ok
    character(*),intent(in) :: message
    integer :: bad,any_bad,ierr,ignored
    bad=merge(0,1,ok)
    call MPI_Allreduce(bad,any_bad,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS.or.any_bad/=0) then
      write(*,'(A)') 'ASTR INSITU SAMPLE ERROR: '//message
      call MPI_Abort(MPI_COMM_WORLD,1,ignored)
    endif
  end subroutine

  subroutine configure_session()
    use commvar, only: use_gpu,lrestart,lreadgrid,flowtype,ndims
#ifdef _CUDA
    use insitu_statistics_gpu, only: configure_statistics_device_budget
#endif
    character(1024) :: filename,message
    logical :: ok
    if(session_checked) return
    call read_consistent_env('ASTR_INSITU_CONFIG',filename)
    if(len_trim(filename)>0) then
      call read_insitu_options_collective(trim(filename),MPI_COMM_WORLD,options,ok,message)
      call require_sample(ok,trim(message))
      formal=.true.
      enabled=options%enabled
      configured=.true.
      statistics_configured=.true.
      oracle_io=.false.
      if(enabled) then
        call require_sample(trim(flowtype)=='tgv'.and.ndims==3.and..not.lreadgrid, &
          'formal statistics currently require internally generated Cartesian TGV')
        if(options%render) then
#ifdef ASTR_WITH_CATALYST
          call require_sample(use_gpu,'native EGL rendering requires GPU solver binding')
#else
          call require_sample(.false.,'formal rendering requires ASTR_WITH_CATALYST=ON')
#endif
        endif
        call require_sample(lrestart.eqv.(len_trim(options%restore_batch)>0), &
          'formal restart requires both lrestart=t and restore_batch')
        if(len_trim(options%restore_batch)>0.or.len_trim(options%batch_prefix)>0) then
          call require_sample(paired_supported(),'formal pairs require supported TGV HDF exact checkpoints')
        endif
        statistics_enabled=options%statistics
        statistics_window=options%statistics_window
        statistics_host_limit=options%host_budget_bytes
        call require_sample(len_trim(options%output_directory)+7<len(prefix),'output path exceeds capacity')
        prefix=trim(options%output_directory)//'/sample'
        if(use_gpu) then
          call require_sample(options%device_budget_bytes>0,'GPU statistics require an explicit device budget')
#ifdef _CUDA
          call configure_statistics_device_budget(options%device_budget_bytes,options%device_reserve_bytes,ok)
          call require_sample(ok,'invalid statistics device budget')
#endif
        endif
      endif
    endif
    session_checked=.true.
  end subroutine

  subroutine paired_inventory(files)
    use commvar, only: use_gpu
    use parallel, only: isize,jsize,ksize
    character(96),allocatable,intent(out) :: files(:)
    integer :: r,n
    n=isize*jsize*ksize
    allocate(files(2+2*n))
    files(1)='flowfield.h5'
    files(2)='auxiliary.txt'
    do r=0,n-1
      write(files(3+2*r),'("restart_q.rank",I8.8,".bin")') r
      if(.not.use_gpu) write(files(3+2*r),'("insitu_cpu_q.rank",I8.8,".bin")') r
      write(files(4+2*r),'("statistics.rank",I8.8,".bin")') r
    enddo
  end subroutine

  logical function paired_supported() result(supported)
    use commvar, only: use_gpu,flowtype,numq,num_species,iomode,lwsequ,lavg,ia,ja,ka, &
      lreadgrid,conschm,difschm,rkscheme
    use bc, only: bctype
    supported=(use_gpu.or.formal).and.trim(flowtype)=='tgv'.and.numq==5.and.num_species==0.and. &
      iomode=='h'.and..not.lwsequ.and..not.lavg.and..not.lreadgrid.and. &
      all([ia,ja,ka]==32).and.all(bctype==1).and.trim(conschm)=='643e'.and. &
      trim(difschm)=='643e'.and.trim(rkscheme)=='rk3'
  end function

  function paired_backend() result(name)
    use commvar, only: use_gpu
    character(32) :: name
    name='tgv-insitu-paired-v1'
    if(.not.use_gpu) name='tgv-insitu-cpu-paired-v1'
  end function

  subroutine capture_insitu_cpu_checkpoint()
    use commvar, only: use_gpu,nstep,feqchkpt
    use commarray, only: q
    integer :: status
    if(use_gpu.or..not.formal.or..not.enabled) return
    if(len_trim(options%batch_prefix)==0.or.nstep==0) return
    if(mod(nstep,feqchkpt)/=0) return
    call require_sample(.not.allocated(cpu_checkpoint_q),'CPU paired snapshot already pending')
    call admit_statistics_host()
    allocate(cpu_checkpoint_q,source=q,stat=status)
    call require_sample(status==0,'cannot allocate CPU paired snapshot')
  end subroutine

  subroutine cpu_checkpoint_file(path,writing)
    use commvar, only: nstep,time,deltat,ia,ja,ka,im,jm,km,hm,numq,lfilter, &
      diffterm,nondimen,const2,const6,reynolds,prandtl,flowtype,conschm,difschm,rkscheme
    use commarray, only: q
    use parallel, only: mpirank,isize,jsize,ksize,ig0,jg0,kg0
    use bc, only: bctype,twall
    use iso_fortran_env, only: iostat_end
    character(*),intent(in) :: path
    logical,intent(in) :: writing
    integer(int32) :: header(26),saved_header(26)
    real(real64) :: parameters(12),saved_parameters(12)
    character(32) :: names(4),saved_names(4)
    character(8) :: magic
    integer(int64) :: step
    integer :: unit,status,closed
    character :: trailing
    header=[1,ia,ja,ka,im,jm,km,hm,numq,isize,jsize,ksize,mpirank,ig0,jg0,kg0, &
      merge(1,0,lfilter),merge(1,0,diffterm),merge(1,0,nondimen),bctype,8]
    parameters=[time,deltat,const2,const6,reynolds,prandtl,twall]
    names=[character(32) :: flowtype,conschm,difschm,rkscheme]
    if(writing) then
      call require_sample(allocated(cpu_checkpoint_q),'missing pre-filter CPU snapshot')
      call require_sample(all(ieee_is_finite(cpu_checkpoint_q)),'nonfinite CPU checkpoint')
      open(newunit=unit,file=path,status='new',access='stream',form='unformatted', &
        convert='little_endian',iostat=status)
      call require_sample(status==0,'cannot create CPU exact checkpoint')
      write(unit,iostat=status) 'ASTRCQ01',header,int(nstep,int64),parameters,names,cpu_checkpoint_q
      call require_sample(status==0,'cannot write CPU exact checkpoint')
    else
      open(newunit=unit,file=path,status='old',access='stream',form='unformatted', &
        convert='little_endian',iostat=status)
      call require_sample(status==0,'cannot open CPU exact checkpoint')
      read(unit,iostat=status) magic,saved_header,step,saved_parameters,saved_names
      call require_sample(status==0,'truncated CPU checkpoint header')
      call require_sample(magic=='ASTRCQ01'.and.all(saved_header==header).and.step==nstep.and. &
        all(saved_parameters==parameters).and.all(saved_names==names),'CPU exact checkpoint metadata mismatch')
      call admit_statistics_host()
      allocate(cpu_checkpoint_q, mold=q,stat=status)
      call require_sample(status==0,'cannot allocate CPU restore buffer')
      read(unit,iostat=status) cpu_checkpoint_q
      call require_sample(status==0,'truncated CPU exact checkpoint')
      read(unit,iostat=status) trailing
      call require_sample(status==iostat_end,'oversized CPU exact checkpoint')
      call require_sample(all(ieee_is_finite(cpu_checkpoint_q)),'nonfinite CPU checkpoint')
      q=cpu_checkpoint_q
    endif
    close(unit,iostat=closed)
    call require_sample(closed==0,'cannot close CPU exact checkpoint')
    deallocate(cpu_checkpoint_q)
  end subroutine

  subroutine restore_insitu_cpu_checkpoint()
    use commvar, only: use_gpu
    use fludyna, only: updatefvar
    use parallel, only: mpirank
    character(1200) :: path
    if(use_gpu.or.len_trim(restore_batch)==0) return
    write(path,'(A,"/insitu_cpu_q.rank",I8.8,".bin")') trim(restore_batch),mpirank
    call cpu_checkpoint_file(trim(path),.false.)
    call updatefvar()
  end subroutine

  subroutine paired_statistics_file(path,step,t,writing)
    use commvar, only: im,jm,km,use_gpu
#ifdef _CUDA
    use insitu_statistics_gpu, only: write_statistics_gpu_state,restore_statistics_gpu_state
#endif
    character(*),intent(in) :: path
    integer,intent(in) :: step
    real(real64),intent(in) :: t
    logical,intent(in) :: writing
    character(64) :: identity
    character(8) :: magic
    integer :: unit,status,closed,i,j,k
    integer(int32) :: schedule_present
    logical :: ok,all_ok
    call admit_statistics_host()
    write(identity,'("tgv-paired-step",I8.8)') step
    if(writing) then
      open(newunit=unit,file=path,status='new',access='stream',form='unformatted',convert='little_endian',iostat=status)
    else
      open(newunit=unit,file=path,status='old',access='stream',form='unformatted',convert='little_endian',iostat=status)
    endif
    call require_sample(status==0,'cannot open paired statistics record')
    if(writing) then
      schedule_present=merge(1,0,render_configured)
      magic='ASTRPS01'
      if(native_device_statistics()) magic='ASTRPS02'
      write(unit,iostat=status) magic,schedule_present
    else
      read(unit,iostat=status) magic,schedule_present
      call require_sample(status==0,'cannot read paired statistics header')
      call require_sample((magic=='ASTRPS01'.or.(magic=='ASTRPS02'.and.native_device_statistics())).and. &
        (schedule_present==0.or.schedule_present==1),'invalid paired record')
      if(formal) call require_sample(schedule_present==merge(1,0,options%render), &
        'paired rendering configuration mismatch')
      if(magic=='ASTRPS01') allocate(point_statistics(0:im,0:jm,0:km),stat=status)
    endif
    call require_sample(status==0,'cannot initialize paired statistics record')
    all_ok=.true.
    if(magic=='ASTRPS01') then
    do k=0,km
    do j=0,jm
    do i=0,im
      if(writing) then
        call write_velocity_state(unit,point_statistics(i,j,k),identity,int(step,int64),t,ok)
      else
        call restore_velocity_state(unit,point_statistics(i,j,k),identity,int(step,int64),t, &
          statistics_window(1),statistics_window(2),ok)
      endif
      all_ok=all_ok.and.ok
    enddo
    enddo
    enddo
    call require_sample(all_ok,'paired point statistics record failed')
    if(native_device_statistics().and..not.writing) deallocate(point_statistics)
    endif
    if(writing) then
      call write_velocity_state(unit,regional_statistics,identity,int(step,int64),t,ok)
    else
      call restore_velocity_state(unit,regional_statistics,identity,int(step,int64),t, &
        statistics_window(1),statistics_window(2),ok)
    endif
    call require_sample(ok,'paired regional statistics record failed')
#ifdef _CUDA
    if(use_gpu) then
    if(writing) then
      call write_statistics_gpu_state(unit,identity,int(step,int64),t,ok)
    else
      call restore_statistics_gpu_state(unit,identity,int(step,int64),t,statistics_window,ok)
    endif
    call require_sample(ok,'paired device statistics record failed')
    endif
#endif
    if(schedule_present==1) then
      if(writing) then
        call write_schedule_state(unit,render_schedule,identity,int(step,int64),t,ok)
      else
        call configure_render_schedule()
        call restore_schedule_state(unit,render_schedule,identity,int(step,int64),t,ok)
      endif
      call require_sample(ok,'paired schedule record failed')
    endif
    close(unit,iostat=closed)
    call require_sample(closed==0,'cannot close paired statistics record')
  end subroutine

  subroutine save_insitu_pair()
    use commvar, only: nstep,time,use_gpu
    use parallel, only: mpirank,isize,jsize,ksize
    character(1024) :: root,path
    character(64) :: identity
    character(96),allocatable :: files(:)
    integer :: i
    logical :: ok
    if(formal.and..not.enabled) return
    if(formal) then
      root=options%batch_prefix
    else
      call read_consistent_env('ASTR_INSITU_TEST_BATCH_PREFIX',root)
    endif
    if(len_trim(root)==0) return
    call require_sample(paired_supported(),'paired gate requires supported TGV, HDF nonsequence, no legacy averaging')
    call configure_statistics_validation()
    call require_sample(statistics_enabled,'paired gate requires explicit statistics window')
    call require_sample(statistics_step==nstep.and.statistics_time==time, &
                        'paired flow and statistics phase mismatch')
    write(path,'(A,".step",I8.8)') trim(root),nstep
    write(identity,'("tgv-paired-step",I8.8)') nstep
    call paired_inventory(files)
    call create_batch(trim(path),MPI_COMM_WORLD,ok)
    call require_sample(ok,'cannot create immutable paired batch')
    ok=.true.
    if(mpirank==0) then
      do i=1,2
        call copy_batch_file('outdat/'//trim(files(i)),trim(path)//'/'//trim(files(i)),ok)
        if(.not.ok) exit
      enddo
    endif
    call require_sample(ok,'cannot copy flow checkpoint')
    i=3+2*mpirank
    if(use_gpu) then
      call copy_batch_file('outdat/'//trim(files(i)),trim(path)//'/'//trim(files(i)),ok)
      call require_sample(ok,'cannot copy exact device checkpoint')
    else
      call cpu_checkpoint_file(trim(path)//'/'//trim(files(i)),.true.)
    endif
    call paired_statistics_file(trim(path)//'/'//trim(files(i+1)),nstep,time,.true.)
    call publish_batch(trim(path),trim(identity),int(nstep,int64),time,[isize,jsize,ksize], &
      trim(paired_backend()),files,.true.,MPI_COMM_WORLD,ok)
    call require_sample(ok,'cannot publish complete paired batch')
  end subroutine

  subroutine prepare_insitu_pair()
    use commvar, only: lrestart
    use parallel, only: mpirank,isize,jsize,ksize
    character(96),allocatable :: files(:)
    character(64) :: identity
    character(8) :: magic
    integer(int64) :: step
    integer :: unit,status,i
    logical :: ok
    call configure_session()
    if(formal) then
      restore_batch=options%restore_batch
    else
      call read_consistent_env('ASTR_INSITU_TEST_RESTORE_BATCH',restore_batch)
    endif
    if(len_trim(restore_batch)==0) return
    call require_sample(paired_supported(),'paired gate requires supported TGV, HDF nonsequence, no legacy averaging')
    call configure_statistics_validation()
    call require_sample(statistics_enabled,'paired gate requires explicit statistics window')
    call require_sample(lrestart,'paired restore requires lrestart=t')
    open(newunit=unit,file=trim(restore_batch)//'/manifest.bin',status='old',access='stream', &
      form='unformatted',convert='little_endian',iostat=status)
    call require_sample(status==0,'paired batch has no completion manifest')
    read(unit,iostat=status) magic,identity,step,restore_time
    close(unit)
    call require_sample(status==0,'cannot read paired batch identity')
    call require_sample(magic=='ASTRB001'.and.step>=0.and.step<=huge(restore_step),'invalid paired batch identity')
    restore_step=int(step)
    call paired_inventory(files)
    call validate_batch(trim(restore_batch),trim(identity),step,restore_time,[isize,jsize,ksize], &
      trim(paired_backend()),files,MPI_COMM_WORLD,ok)
    call require_sample(ok,'paired batch incomplete or changed')
    ok=.true.
    if(mpirank==0) then
      do i=1,2
        call copy_batch_file(trim(restore_batch)//'/'//trim(files(i)),'outdat/'//trim(files(i)),ok)
        if(.not.ok) exit
      enddo
    endif
    call require_sample(ok,'paired restore requires empty output checkpoint destinations')
    i=3+2*mpirank
    call copy_batch_file(trim(restore_batch)//'/'//trim(files(i)),'outdat/'//trim(files(i)),ok)
    call require_sample(ok,'cannot stage paired exact checkpoint')
  end subroutine

  subroutine configure_statistics_validation()
#ifdef _CUDA
    use insitu_statistics_gpu, only: configure_statistics_device_budget
#endif
    character(128) :: text
    integer :: status
    integer(int64) :: limit,headroom
    logical :: ok
    if(statistics_configured) return
    call read_consistent_env('ASTR_INSITU_TEST_ORACLE_IO',text)
    call require_sample(trim(text)==''.or.trim(text)=='0'.or.trim(text)=='1','invalid oracle I/O flag')
    oracle_io=trim(text)/='0'
    call read_consistent_env('ASTR_INSITU_TEST_STATISTICS_WINDOW',text)
    statistics_enabled=len_trim(text)>0
    if(statistics_enabled) then
      read(text,*,iostat=status) statistics_window
      call require_sample(status==0,'invalid statistics window')
      call require_sample(all(ieee_is_finite(statistics_window)), 'nonfinite statistics window')
      call require_sample(statistics_window(2)>statistics_window(1),'empty statistics window')
    endif
    call read_consistent_env('ASTR_INSITU_TEST_DEVICE_BUDGET',text)
    if(len_trim(text)>0) then
      read(text,*,iostat=status) limit,headroom
      call require_sample(status==0,'device budget requires limit and headroom bytes')
#ifdef _CUDA
      call configure_statistics_device_budget(limit,headroom,ok)
      call require_sample(ok,'invalid statistics device budget')
#else
      call require_sample(.false.,'device budget requires CUDA build')
#endif
    endif
    call read_consistent_env('ASTR_INSITU_TEST_HOST_BUDGET',text)
    if(len_trim(text)>0) then
      read(text,*,iostat=status) statistics_host_limit
      call require_sample(status==0,'invalid statistics host budget')
      call require_sample(statistics_host_limit>0,'statistics host budget must be positive')
    endif
    statistics_configured=.true.
  end subroutine

  subroutine admit_statistics_host()
    use commvar, only: im,jm,km,use_gpu
    use commarray, only: q
    type(resource_budget) :: budget
    integer(int64) :: request,extra,total,point_bytes
    integer(int64),allocatable :: requests(:)
    integer :: node,nrank,i,ierr,status
    logical :: ok
    if(statistics_host_limit==0) return
    ! Conservative envelope of explicit statistics arrays, not process RSS:
    ! point states + weights/velocity/output; GPU download + checkpoint copy + mask.
    point_bytes=int(storage_size(regional_statistics)/8,int64)+45_int64*8
    if(formal) then
      point_bytes=point_bytes+(11_int64+14_int64+41_int64)*8
      if(options%render) point_bytes=point_bytes+(35_int64+28_int64+14_int64)*8
      ! Render-only: sampled/derived fields, bridge vectors and packing temporaries.
      if(.not.statistics_enabled) point_bytes=100_int64*8
    endif
    call checked_bytes(int([im,jm,km],int64)+1_int64,point_bytes,request,ok)
    call require_sample(ok,'statistics host byte count overflow')
    if(formal.and..not.use_gpu) then
      if(len_trim(options%batch_prefix)>0.or.len_trim(options%restore_batch)>0) then
        call checked_bytes(int(shape(q),int64),8_int64,extra,ok)
        call require_sample(ok,'CPU paired snapshot byte count overflow')
        call require_sample(extra<=huge(request)-request,'CPU paired snapshot demand overflow')
        request=request+extra
      endif
    endif
    if(use_gpu.and.statistics_enabled) then
      call checked_bytes(int([im,jm,km],int64),71_int64*8+storage_size(0)/8,extra,ok)
      call require_sample(ok,'statistics host GPU workspace byte count overflow')
      call require_sample(extra<=huge(request)-request,'statistics host demand overflow')
      request=request+extra
    endif
    call MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,node,ierr)
    call require_sample(ierr==MPI_SUCCESS,'statistics host node communicator')
    call MPI_Comm_size(node,nrank,ierr)
    call require_sample(ierr==MPI_SUCCESS,'statistics host node size')
    allocate(requests(nrank),stat=status)
    call require_sample(status==0,'statistics host budget metadata allocation')
    call MPI_Allgather(request,1,MPI_INTEGER8,requests,1,MPI_INTEGER8,node,ierr)
    call require_sample(ierr==MPI_SUCCESS,'statistics host demand gather')
    total=0
    ok=.true.
    do i=1,nrank
      if(requests(i)>huge(total)-total) then
        ok=.false.
        exit
      endif
      total=total+requests(i)
    enddo
    call require_sample(ok,'statistics node demand overflow')
    call MPI_Comm_free(node,ierr)
    call require_sample(ierr==MPI_SUCCESS,'statistics host communicator release')
    call configure_budget(budget,statistics_host_limit,0_int64,ok)
    call require_sample(ok,'statistics host budget configuration')
    write(*,'(A,3(1X,I0))') 'INSITU HOST BUDGET local/node/limit:',request,total,statistics_host_limit
    call reserve_bytes(budget,1,total,huge(total),ok)
    call require_sample(ok,'statistics host allocation exceeds node budget')
  end subroutine

  subroutine accumulate_statistics_validation(step,t,fields)
    use commvar, only: im,jm,km,use_gpu
#ifdef _CUDA
    use insitu_statistics_gpu, only: accumulate_statistics_gpu,write_statistics_gpu_state, &
      restore_statistics_gpu_state,release_statistics_gpu,spatial_statistics_gpu
#endif
    use commarray, only: x
    use parallel, only: mpirank,ig0,jg0,kg0
    integer,intent(in) :: step
    real(real64),intent(in) :: t,fields(0:im,0:jm,0:km,11)
    type(velocity_statistics_result) :: result,regional
    real(real64),allocatable :: weights(:),values(:,:),output(:,:,:,:)
    real(real64),allocatable :: gpu_output(:,:,:,:)
    real(real64) :: volume,mean(3),rms_local(3),rms_regional(3),cell_volume
    real(real64) :: device_variance(3)
    integer :: i,j,k,n,status,unit,close_status
    integer :: bad_unit
    integer(int32) :: saved_header(17)
    character(8) :: saved_magic
    character(64) :: saved_identity
    logical :: ok,all_ok,covered,any_covered
    character(1200) :: filename
    character(128) :: roundtrip
    if(.not.statistics_enabled) return
    call admit_statistics_host()
    statistics_step=step
    statistics_time=t
#ifdef _CUDA
    if(use_gpu) then
      if(oracle_io) then
        allocate(gpu_output(0:im-1,0:jm-1,0:km-1,41),stat=status)
        call require_sample(status==0,'cannot allocate GPU statistics verification buffer')
        call accumulate_statistics_gpu(t,statistics_window,gpu_output)
      else
        call accumulate_statistics_gpu(t,statistics_window)
      endif
      call read_consistent_env('ASTR_INSITU_TEST_STATISTICS_ROUNDTRIP',roundtrip)
      call require_sample(trim(roundtrip)==''.or.trim(roundtrip)=='0'.or.trim(roundtrip)=='1', &
                          'invalid statistics roundtrip flag')
      if(step==1.and.trim(roundtrip)=='1') then
        write(filename,'(A,".device_state.rank",I8.8,".bin")') trim(prefix),mpirank
        open(newunit=unit,file=trim(filename),status='new',access='stream',form='unformatted', &
             convert='little_endian',iostat=status)
        call require_sample(status==0,'cannot create device statistics state')
        call write_statistics_gpu_state(unit,'statistics-record-probe',int(step,int64),t,ok)
        call require_sample(ok,'cannot save device statistics state')
        rewind(unit)
        read(unit,iostat=status) saved_magic,saved_header
        call require_sample(status==0,'cannot inspect saved device statistics header')
        open(newunit=bad_unit,status='scratch',access='stream',form='unformatted', &
             convert='little_endian',iostat=status)
        call require_sample(status==0,'cannot open malformed statistics probe')
        write(bad_unit) saved_magic
        rewind(bad_unit)
        call restore_statistics_gpu_state(bad_unit,'statistics-record-probe',int(step,int64),t,statistics_window,ok)
        call require_sample(.not.ok,'accepted truncated device statistics state')
        rewind(bad_unit)
        saved_header(5)=saved_header(5)+1
        saved_identity='statistics-record-probe'
        write(bad_unit) saved_magic,saved_header,saved_identity,int(step,int64),t,t,statistics_window
        rewind(bad_unit)
        call restore_statistics_gpu_state(bad_unit,'statistics-record-probe',int(step,int64),t,statistics_window,ok)
        call require_sample(.not.ok,'accepted wrong device statistics topology')
        close(bad_unit)
        rewind(unit)
        call restore_statistics_gpu_state(unit,'wrong-batch',int(step,int64),t,statistics_window,ok)
        call require_sample(.not.ok,'accepted wrong device statistics batch')
        rewind(unit)
        call restore_statistics_gpu_state(unit,'statistics-record-probe',int(step+1,int64),t,statistics_window,ok)
        call require_sample(.not.ok,'accepted wrong device statistics step')
        rewind(unit)
        call restore_statistics_gpu_state(unit,'statistics-record-probe',int(step,int64),t+1.d0,statistics_window,ok)
        call require_sample(.not.ok,'accepted wrong device statistics time')
        rewind(unit)
        call restore_statistics_gpu_state(unit,'statistics-record-probe',int(step,int64),t, &
          statistics_window+[0.d0,1.d0],ok)
        call require_sample(.not.ok,'accepted wrong device statistics window')
        call release_statistics_gpu()
        rewind(unit)
        call restore_statistics_gpu_state(unit,'statistics-record-probe',int(step,int64),t,statistics_window,ok)
        call require_sample(ok,'cannot restore released device statistics state')
        close(unit,iostat=status)
        call require_sample(status==0,'cannot close device statistics state')
      endif
    endif
#endif
    if(.not.allocated(point_statistics)) then
      allocate(point_statistics(0:im,0:jm,0:km),stat=status)
      call require_sample(status==0,'cannot allocate statistics')
      do k=0,km
      do j=0,jm
      do i=0,im
        call configure_velocity_statistics(point_statistics(i,j,k), &
          statistics_window(1),statistics_window(2),ok)
      enddo
      enddo
      enddo
      call configure_velocity_statistics(regional_statistics,statistics_window(1),statistics_window(2),ok)
      call require_sample(ok,'cannot configure statistics')
    endif
    allocate(weights((im+1)*(jm+1)*(km+1)),values(3,(im+1)*(jm+1)*(km+1)), &
             output(0:im,0:jm,0:km,41),stat=status)
    call require_sample(status==0,'cannot allocate statistics output')
    cell_volume=(x(1,0,0,1)-x(0,0,0,1))*(x(0,1,0,2)-x(0,0,0,2))*(x(0,0,1,3)-x(0,0,0,3))
    call require_sample(ieee_is_finite(cell_volume).and.cell_volume>0,'invalid TGV cell volume')
    n=0
    all_ok=.true.
    covered=.true.
    any_covered=.false.
    do k=0,km
    do j=0,jm
    do i=0,im
      n=n+1
      weights(n)=0.d0
      if(i<im.and.j<jm.and.k<km) weights(n)=cell_volume
      values(:,n)=fields(i,j,k,7:9)
      call push_velocity_sample(point_statistics(i,j,k),t,fields(i,j,k,6),values(:,n),ok)
      all_ok=all_ok.and.ok
      call read_velocity_statistics(point_statistics(i,j,k),result,ok)
      covered=covered.and.ok
      any_covered=any_covered.or.ok
      output(i,j,k,:)=[result%duration,result%mean_density,result%mean_r,result%mean_f, &
        result%rms_r,result%rms_f,reshape(result%covariance_r,[9]), &
        reshape(result%covariance_f,[9]),reshape(result%density_stress,[9])]
    enddo
    enddo
    enddo
    call require_sample(all_ok,'invalid point statistics update')
    call require_sample(covered.eqv.any_covered,'inconsistent point statistics coverage')
#ifdef _CUDA
    if(use_gpu) then
      call spatial_statistics_gpu(cell_volume,volume,mean,device_variance,ok)
    else
#endif
      call spatial_mean(weights,values,MPI_COMM_WORLD,volume,mean,ok)
#ifdef _CUDA
    endif
#endif
    call require_sample(ok,'invalid regional velocity reduction')
    ! Regional-signal RMS is Reynolds temporal RMS of the geometric mean velocity.
    call push_velocity_sample(regional_statistics,t,1.d0,mean,ok)
    call require_sample(ok,'invalid regional statistics update')
    call read_velocity_statistics(regional_statistics,regional,ok)
    call require_sample(ok.eqv.covered,'inconsistent regional statistics coverage')
    if(.not.covered) return
    n=0
    do k=0,km
    do j=0,jm
    do i=0,im
      n=n+1
      values(:,n)=output(i,j,k,[15,19,23])
    enddo
    enddo
    enddo
#ifdef _CUDA
    if(use_gpu) then
      mean=device_variance
      ok=.true.
    else
#endif
      call spatial_mean(weights,values,MPI_COMM_WORLD,volume,mean,ok)
#ifdef _CUDA
    endif
#endif
    call require_sample(ok.and.all(mean>=0),'invalid spatial variance reduction')
    rms_local=sqrt(mean)
    rms_regional=regional%rms_r
    if(formal) then
      final_statistics_meta=[t,statistics_window,volume,rms_local,rms_regional]
      call move_alloc(output,final_statistics)
    endif
    if(.not.oracle_io) return
    write(filename,'(A,".statistics.step",I8.8,".rank",I8.8,".bin")') trim(prefix),step,mpirank
    open(newunit=unit,file=trim(filename),status='new',access='stream',form='unformatted', &
         action='write',convert='little_endian',iostat=status)
    call require_sample(status==0,'cannot create statistics output')
    write(unit,iostat=status) 'ASTRST01',int([1,step,mpirank,im+1,jm+1,km+1,ig0,jg0,kg0],int32), &
      t,statistics_window,volume,rms_local,rms_regional,output
    close(unit,iostat=close_status)
    call require_sample(status==0.and.close_status==0,'cannot write statistics output')
    if(allocated(gpu_output)) then
      write(filename,'(A,".device_statistics.step",I8.8,".rank",I8.8,".bin")') trim(prefix),step,mpirank
      open(newunit=unit,file=trim(filename),status='new',access='stream',form='unformatted', &
           action='write',convert='little_endian',iostat=status)
      call require_sample(status==0,'cannot create device statistics output')
      write(unit,iostat=status) 'ASTRST01',int([1,step,mpirank,im,jm,km,ig0,jg0,kg0],int32), &
        t,statistics_window,volume,rms_local,rms_regional,gpu_output
      close(unit,iostat=close_status)
      call require_sample(status==0.and.close_status==0,'cannot write device statistics output')
    endif
  end subroutine

  subroutine accumulate_native_device(step,t)
#ifdef _CUDA
    use insitu_statistics_gpu, only: accumulate_statistics_gpu,spatial_statistics_gpu
    use commarray, only: x
    integer,intent(in) :: step
    real(real64),intent(in) :: t
    real(real64) :: cell_volume,volume,mean(3),variance(3)
    type(velocity_statistics_result) :: regional
    logical :: ok
    call admit_statistics_host()
    if(statistics_step<0) then
      call configure_velocity_statistics(regional_statistics,statistics_window(1),statistics_window(2),ok)
      call require_sample(ok,'cannot configure native regional statistics')
    endif
    call accumulate_statistics_gpu(t,statistics_window)
    cell_volume=(x(1,0,0,1)-x(0,0,0,1))*(x(0,1,0,2)-x(0,0,0,2))*(x(0,0,1,3)-x(0,0,0,3))
    call spatial_statistics_gpu(cell_volume,volume,mean,variance,ok)
    call require_sample(ok,'invalid native device spatial reduction')
    call push_velocity_sample(regional_statistics,t,1.d0,mean,ok)
    call require_sample(ok,'invalid native regional statistics update')
    call read_velocity_statistics(regional_statistics,regional,ok)
    statistics_step=step
    statistics_time=t
    if(ok) final_statistics_meta=[t,statistics_window,volume,sqrt(variance),regional%rms_r]
#else
    integer,intent(in) :: step
    real(real64),intent(in) :: t
    call require_sample(.false.,'device statistics require CUDA')
#endif
  end subroutine

  subroutine export_native_statistics()
#ifdef _CUDA
    use commvar, only: im,jm,km
    use commarray, only: x
    use insitu_statistics_gpu, only: download_statistics_gpu_output,spatial_statistics_gpu
    type(velocity_statistics_result) :: regional
    real(real64) :: cell_volume,volume,mean(3),variance(3)
    integer :: status
    logical :: ok
    call read_velocity_statistics(regional_statistics,regional,ok)
    if(.not.ok) return
    allocate(final_statistics(0:im,0:jm,0:km,41),stat=status)
    call require_sample(status==0,'cannot allocate requested final statistics')
    final_statistics=0.d0
    call download_statistics_gpu_output(final_statistics(0:im-1,0:jm-1,0:km-1,:))
    call complete_periodic_endpoints(final_statistics)
    cell_volume=(x(1,0,0,1)-x(0,0,0,1))*(x(0,1,0,2)-x(0,0,0,2))*(x(0,0,1,3)-x(0,0,0,3))
    call spatial_statistics_gpu(cell_volume,volume,mean,variance,ok)
    call require_sample(ok,'invalid final native spatial statistics')
    final_statistics_meta=[statistics_time,statistics_window,volume,sqrt(variance),regional%rms_r]
#ifdef ASTR_WITH_CATALYST
    status=resource_check('statistics_exported'//c_null_char)
    call require_sample(status==0,'cannot check statistics export resources')
#endif
#endif
  end subroutine

  subroutine render_native_if_due(step,t,final)
    use commvar, only: im,jm,km
    use commarray, only: x
    integer,intent(in) :: step
    real(real64),intent(in) :: t
    logical,intent(in) :: final
    type(sample_schedule) :: preview
    real(real64),allocatable :: fields(:,:,:,:),derived(:,:,:,:)
    logical :: emit,ok
    integer :: status
    integer(int64) :: crossed
    if(.not.options%render) return
    call configure_render_schedule()
    preview=render_schedule
    call poll_schedule(preview,int(step,int64),t,final,emit,crossed,ok)
    call require_sample(ok,'invalid native render preview')
    if(.not.emit) then
      render_schedule=preview
      return
    endif
    allocate(fields(0:im,0:jm,0:km,11),derived(0:im,0:jm,0:km,14),stat=status)
    call require_sample(status==0,'cannot allocate requested native frame')
    call capture_sample(fields)
    native_flow_downloads=native_flow_downloads+1
    call canonicalize_sample(fields)
    call derive_sample(fields(:,:,:,7:9),derived)
    call render_sample(step,t,x(0:im,0:jm,0:km,1:3),fields,derived,final)
  end subroutine

  subroutine begin_insitu(step,t)
    use parallel, only: mpirank
    integer,intent(in) :: step
    real(real64),intent(in) :: t
    character(128) :: flag
    character(1200) :: filename
    integer :: resource_status
    call configure_session()
    if(formal.and..not.enabled) return
#if defined(ASTR_WITH_CATALYST) && defined(_CUDA)
    if(formal.and.enabled.and.options%render) then
      resource_status=resource_begin(int(MPI_COMM_WORLD,c_int), &
        int(options%host_budget_bytes,c_int64_t),int(options%device_budget_bytes,c_int64_t), &
        int(options%device_reserve_bytes,c_int64_t),trim(options%output_directory)//c_null_char)
      call require_sample(resource_status==0,'cannot initialize native resource observer')
    endif
#endif
    call configure_statistics_validation()
    if(len_trim(restore_batch)>0) then
      call require_sample(step==restore_step.and.t==restore_time,'restored flow does not match paired clock')
      write(filename,'(A,"/statistics.rank",I8.8,".bin")') trim(restore_batch),mpirank
      call paired_statistics_file(trim(filename),step,t,.false.)
      statistics_step=step
      statistics_time=t
      return
    endif
    call read_consistent_env('ASTR_INSITU_TEST_INITIAL',flag)
    if(formal.and.options%initial_frame) flag='1'
    call require_sample(trim(flag)==''.or.trim(flag)=='0'.or.trim(flag)=='1','invalid initial sampling flag')
    if(trim(flag)/='1'.and..not.statistics_enabled) return
    call require_sample(step==0.and.t==0.d0,'initial sampling gate does not support restart')
    call sample_insitu_step(0,t,0.d0)
  end subroutine
  subroutine configure_render_schedule()
    character(128) :: step_text,time_text,initial_text,final_text
    integer :: status
    integer(int64) :: interval
    real(real64) :: period
    logical :: initial,ok
    if(render_configured) return
    if(formal) then
      render_final=options%final_frame
      call configure_schedule(render_schedule,trim(options%schedule_mode),options%step_interval, &
        options%time_interval,0_int64,0.d0,options%initial_frame,options%final_frame,ok)
      call require_sample(ok,'invalid native render schedule')
      render_configured=.true.
      return
    endif
    call read_consistent_env('ASTR_INSITU_TEST_STEP_INTERVAL',step_text)
    call read_consistent_env('ASTR_INSITU_TEST_TIME_INTERVAL',time_text)
    call read_consistent_env('ASTR_INSITU_TEST_INITIAL',initial_text)
    call read_consistent_env('ASTR_INSITU_TEST_FINAL',final_text)
    call require_sample(trim(initial_text)==''.or.trim(initial_text)=='0'.or.trim(initial_text)=='1', &
                        'invalid initial sampling flag')
    call require_sample(trim(final_text)==''.or.trim(final_text)=='0'.or.trim(final_text)=='1', &
                        'invalid final sampling flag')
    initial=trim(initial_text)=='1'
    render_final=trim(final_text)=='1'
    call require_sample(len_trim(step_text)==0.or.len_trim(time_text)==0,'choose only one render schedule mode')
    interval=1
    period=0.d0
    if(len_trim(time_text)>0) then
      read(time_text,*,iostat=status) period
      call require_sample(status==0,'invalid time interval')
      interval=0
      call configure_schedule(render_schedule,'time',interval,period,0_int64,0.d0,initial,render_final,ok)
    else
      if(len_trim(step_text)>0) then
        read(step_text,*,iostat=status) interval
        call require_sample(status==0,'invalid step interval')
      endif
      call configure_schedule(render_schedule,'steps',interval,period,0_int64,0.d0,initial,render_final,ok)
    endif
    call require_sample(ok,'invalid render schedule configuration')
    render_configured=.true.
  end subroutine
  subroutine finish_insitu()
#ifdef _CUDA
    use insitu_statistics_gpu, only: release_statistics_gpu,statistics_transfer_counts
    use commvar, only: use_gpu
    use parallel, only: mpirank
    integer(int64) :: samples,downloads
#endif
#ifdef ASTR_WITH_CATALYST
    integer :: status
    if(native_device_statistics()) call render_native_if_due(statistics_step,statistics_time,.true.)
    if(allocated(last_fields)) then
      call render_sample(last_step,last_time,last_xyz,last_fields,last_derived,.true.)
      deallocate(last_xyz,last_fields,last_derived)
    endif
    status=mesh_finish()
    call require_sample(status==0,'Catalyst mesh finalization failed')
#endif
#ifdef _CUDA
    if(native_device_statistics()) call export_native_statistics()
    if(formal.and.enabled.and.use_gpu.and.statistics_enabled) then
      call statistics_transfer_counts(samples,downloads)
      write(*,'(A,I0,A,I0,A,I0)') 'ASTR_INSITU_GPU_STATS rank=',mpirank, &
        ' samples=',samples,' full_output_downloads=',downloads
      write(*,'(A,I0,A,I0)') 'ASTR_INSITU_GPU_FLOW rank=',mpirank,' frame_downloads=',native_flow_downloads
    endif
    call release_statistics_gpu()
#endif
    call write_final_statistics()
    if(allocated(point_statistics)) deallocate(point_statistics)
    if(allocated(final_statistics)) deallocate(final_statistics)
#if defined(ASTR_WITH_CATALYST) && defined(_CUDA)
    status=resource_finish()
    call require_sample(status==0,'cannot finalize native resource observer')
#endif
  end subroutine

  subroutine write_final_statistics()
    use parallel, only: mpirank,ig0,jg0,kg0
    use commvar, only: im,jm,km
    character(1200) :: filename
    integer :: unit,status,closed
    if(.not.formal.or..not.enabled.or..not.statistics_enabled) return
    if(.not.allocated(final_statistics)) return
    write(filename,'(A,".statistics.step",I8.8,".rank",I8.8,".bin")') trim(prefix),statistics_step,mpirank
    open(newunit=unit,file=trim(filename),status='new',access='stream',form='unformatted', &
      action='write',convert='little_endian',iostat=status)
    call require_sample(status==0,'cannot create final statistics output')
    write(unit,iostat=status) 'ASTRST01',int([1,statistics_step,mpirank,im+1,jm+1,km+1,ig0,jg0,kg0],int32), &
      final_statistics_meta,final_statistics
    close(unit,iostat=closed)
    call require_sample(status==0.and.closed==0,'cannot write final statistics output')
  end subroutine
  subroutine render_sample(step,t,xyz,fields,derived,is_final)
    use commvar, only: im,jm,km,use_gpu
#ifdef _CUDA
    use insitu_statistics_gpu, only: download_statistics_gpu_means
#endif
    use parallel, only: mpirank
    integer,intent(in) :: step
    real(real64),intent(in) :: t,xyz(0:im,0:jm,0:km,3),fields(0:im,0:jm,0:km,11)
    real(real64),intent(in) :: derived(0:im,0:jm,0:km,14)
    logical,optional,intent(in) :: is_final
    type(velocity_statistics_result) :: mean_result
    real(real64),allocatable :: means(:,:,:,:)
    character(128) :: mean_flag
    integer :: i,j,k,has_mean
    logical :: all_covered,any_covered
    character(4096) :: script,backend,root
    character(1200) :: filename
    integer :: status,length,ierr,unit,close_status
    integer(int64) :: crossed
    logical :: ok,emit,final
    final=.false.
    if(present(is_final)) final=is_final
    if(formal) then
      if(.not.options%render) return
      script=options%pipeline_file
    else
      call read_consistent_env('ASTR_INSITU_TEST_PIPELINE',script)
    endif
    root=script
    call MPI_Bcast(root,len(root),MPI_CHARACTER,0,MPI_COMM_WORLD,ierr)
    call require_sample(ierr==MPI_SUCCESS.and.root==script,'test pipelines differ')
    if(len_trim(script)==0) return
    call configure_render_schedule()
    if(render_final.and..not.final.and..not.native_device_statistics()) then
      if(.not.allocated(last_fields)) then
        allocate(last_xyz(0:im,0:jm,0:km,3),last_fields(0:im,0:jm,0:km,11), &
                 last_derived(0:im,0:jm,0:km,14),stat=status)
        call require_sample(status==0,'cannot allocate final-state snapshot')
      endif
      last_xyz=xyz
      last_fields=fields
      last_derived=derived
      last_step=step
      last_time=t
    endif
    call poll_schedule(render_schedule,int(step,int64),t,final,emit,crossed,ok)
    call require_sample(ok,'invalid render schedule clock')
    if(.not.emit) return
    if(.not.formal) then
    write(filename,'(A,".schedule.step",I8.8,".rank",I8.8,".txt")') trim(prefix),step,mpirank
    open(newunit=unit,file=trim(filename),status='new',action='write',iostat=status)
    call require_sample(status==0,'cannot create schedule record')
    write(unit,*,iostat=status) step,t,crossed
    close(unit,iostat=close_status)
    call require_sample(status==0.and.close_status==0,'cannot write schedule record')
    endif
#ifdef ASTR_WITH_CATALYST
    if(formal) then
      backend=options%implementation_path
      if(.not.native_bridge_configured) then
        status=mesh_configure(trim(options%output_directory)//c_null_char,int(MPI_COMM_WORLD,c_int))
        call require_sample(status==0,'cannot configure native render device')
        native_bridge_configured=.true.
      endif
    else
    call get_environment_variable('ASTR_INSITU_TEST_BACKEND',backend,length,status)
    call require_sample(status==0.and.length>0,'missing test backend')
    root=backend
    call MPI_Bcast(root,len(root),MPI_CHARACTER,0,MPI_COMM_WORLD,ierr)
    call require_sample(ierr==MPI_SUCCESS.and.root==backend,'test backends differ')
    endif
    call read_consistent_env('ASTR_INSITU_TEST_MEAN_STREAMLINES',mean_flag)
    if(formal.and.statistics_enabled) mean_flag='1'
    call require_sample(trim(mean_flag)==''.or.trim(mean_flag)=='1','invalid mean streamline flag')
    has_mean=0
    if(trim(mean_flag)=='1') then
      call require_sample(statistics_enabled,'mean streamlines require statistics')
      allocate(means(0:im,0:jm,0:km,7),stat=status)
      call require_sample(status==0,'cannot allocate mean render fields')
#ifdef _CUDA
      if(formal.and.use_gpu) then
        means=0.d0
        call download_statistics_gpu_means(means(0:im-1,0:jm-1,0:km-1,:),all_covered)
        if(all_covered) call complete_periodic_endpoints(means)
      else
#endif
      call require_sample(allocated(point_statistics),'mean streamlines require point statistics')
      all_covered=.true.
      any_covered=.false.
      do k=0,km
      do j=0,jm
      do i=0,im
        call read_velocity_statistics(point_statistics(i,j,k),mean_result,ok)
        all_covered=all_covered.and.ok
        any_covered=any_covered.or.ok
        means(i,j,k,:)=[mean_result%mean_r,mean_result%mean_f,mean_result%duration]
      enddo
      enddo
      enddo
      call require_sample(all_covered.eqv.any_covered,'inconsistent mean render coverage')
#ifdef _CUDA
      endif
#endif
      has_mean=merge(1,0,all_covered)
    else
      allocate(means(1,1,1,1))
      means=0.d0
    endif
    status=mesh_probe(trim(backend)//c_null_char,trim(script)//c_null_char, &
                     int(MPI_COMM_WORLD,c_int),im+1,jm+1,km+1,step,t,xyz,fields,derived,has_mean,means)
    call require_sample(status==0,'Catalyst mesh probe failed')
#else
    call require_sample(.false.,'test rendering requires ASTR_WITH_CATALYST=ON')
#endif
  end subroutine
  subroutine read_consistent_env(name,value)
    character(*),intent(in) :: name
    character(*),intent(out) :: value
    character(len(value)) :: root
    integer :: status,length,ierr
    value=''
#ifndef ASTR_BUILD_TESTING
    if(name/='ASTR_INSITU_CONFIG') return
#endif
    if(formal.and.name/='ASTR_INSITU_CONFIG') return
    call get_environment_variable(name,value,length,status)
    call require_sample(status==0.or.status==1,'invalid '//name)
    root=value
    call MPI_Bcast(root,len(root),MPI_CHARACTER,0,MPI_COMM_WORLD,ierr)
    call require_sample(ierr==MPI_SUCCESS.and.root==value,'rank mismatch for '//name)
  end subroutine

  subroutine sample_insitu_step(step,time_end,step_dt)
    use commvar, only: im,jm,km,numq,num_species,flowtype,use_gpu,hm,difschm
    use bc, only: bctype
    use commarray, only: x
    use parallel, only: mpirank,ig0,jg0,kg0
    integer,intent(in) :: step
    real(real64),intent(in) :: time_end,step_dt
    real(real64),allocatable :: fields(:,:,:,:),derived(:,:,:,:),manufactured(:,:,:,:)
    character(1024) :: root_prefix
    character(1200) :: filename
    integer :: status,length,ierr,unit,close_status
    integer(int32) :: header(9)
    type(sample_schedule) :: preview
    character(4096) :: on_demand, pipeline
    logical :: emit,ok
    integer(int64) :: crossed
    if(.not.configured) then
      call read_consistent_env('ASTR_INSITU_SAMPLE_PREFIX',prefix)
      root_prefix=prefix
      call MPI_Bcast(root_prefix,len(root_prefix),MPI_CHARACTER,0,MPI_COMM_WORLD,ierr)
      call require_sample(ierr==MPI_SUCCESS.and.root_prefix==prefix,'sample prefixes differ between ranks')
      enabled=len_trim(prefix)>0
      configured=.true.
    endif
    if(.not.enabled) return
    call configure_statistics_validation()
    ! Validate the supported state even when this step will not be captured.
    call require_sample(trim(flowtype)=='tgv'.and.numq==5.and.num_species==0, &
                        'initial sampling gate supports five-variable TGV only')
    if(formal.and.options%render.and..not.statistics_enabled) then
      call require_sample(max(im,jm,km)<=256.and.min(im,jm,km)>=3, &
                          'render-only demonstration requires local extents in 3..256')
    else
      call require_sample(max(im,jm,km)<=32.and.min(im,jm,km)>=1, &
                          'initial sampling gate requires local extents <=32')
    endif
    call require_sample(all(bctype==1).and.hm>=3.and.trim(difschm)=='643e', &
                        'diagnostic gate requires periodic boundaries and explicit sixth-order derivatives')
    call require_sample(step>=0.and.ieee_is_finite(time_end).and.ieee_is_finite(step_dt), &
                        'nonfinite sample metadata')
    call require_sample((step==0.and.time_end==0.d0.and.step_dt==0.d0).or. &
                        (step>0.and.step_dt>0.d0),'invalid completed-step metadata')
    if(native_device_statistics()) then
      call accumulate_native_device(step,time_end)
      call render_native_if_due(step,time_end,.false.)
      return
    endif
    call read_consistent_env('ASTR_INSITU_TEST_ON_DEMAND',on_demand)
    call require_sample(trim(on_demand)==''.or.trim(on_demand)=='0'.or.trim(on_demand)=='1', &
                        'invalid on-demand flag')
    if(trim(on_demand)=='1') then
      call read_consistent_env('ASTR_INSITU_TEST_PIPELINE',pipeline)
      call require_sample(len_trim(pipeline)>0,'on-demand test requires a rendering pipeline')
      call configure_render_schedule()
      ! Preview without consuming a due event; render_sample commits it after capture.
      preview=render_schedule
      call poll_schedule(preview,int(step,int64),time_end,.false.,emit,crossed,ok)
      call require_sample(ok,'invalid on-demand sample clock')
      if(.not.emit.and..not.render_final.and..not.statistics_enabled) then
        render_schedule=preview
        return
      endif
    endif
    if(formal) call admit_statistics_host()
    allocate(fields(0:im,0:jm,0:km,11),stat=status)
    call require_sample(status==0,'cannot allocate sample buffer')
    call capture_sample(fields)
    header=int([1,step,mpirank,im+1,jm+1,km+1,ig0,jg0,kg0],int32)
    if(oracle_io) then
    write(filename,'(A,".step",I8.8,".rank",I8.8,".bin")') trim(prefix),step,mpirank
    open(newunit=unit,file=trim(filename),status='new',access='stream', &
         form='unformatted',action='write',convert='little_endian',iostat=status)
    call require_sample(status==0,'cannot create sample file')
    write(unit,iostat=status) 'ASTRIS01',header,time_end,step_dt, &
                             x(0:im,0:jm,0:km,1:3),fields
    close(unit,iostat=close_status)
    call require_sample(status==0.and.close_status==0,'cannot write sample file')
    endif
    call canonicalize_sample(fields)
    call accumulate_statistics_validation(step,time_end,fields)
    if(formal.and..not.options%render) return
    if(oracle_io) then
    write(filename,'(A,".canonical.step",I8.8,".rank",I8.8,".bin")') trim(prefix),step,mpirank
    open(newunit=unit,file=trim(filename),status='new',access='stream', &
         form='unformatted',action='write',convert='little_endian',iostat=status)
    call require_sample(status==0,'cannot create canonical sample file')
    write(unit,iostat=status) 'ASTRIC01',header,time_end,step_dt,x(0:im,0:jm,0:km,1:3),fields
    close(unit,iostat=close_status)
    call require_sample(status==0.and.close_status==0,'cannot write canonical sample file')
    endif
    allocate(derived(0:im,0:jm,0:km,14),stat=status)
    call require_sample(status==0,'cannot allocate diagnostic buffer')
    call derive_sample(fields(:,:,:,7:9),derived)
    write(filename,'(A,".derived.step",I8.8,".rank",I8.8,".bin")') trim(prefix),step,mpirank
    if(oracle_io) call write_derived()
    call render_sample(step,time_end,x(0:im,0:jm,0:km,1:3),fields,derived)
    if(step==1.and.oracle_io) then
      allocate(manufactured(0:im,0:jm,0:km,3),stat=status)
      call require_sample(status==0,'cannot allocate manufactured velocity')
      manufactured(:,:,:,1)=sin(x(0:im,0:jm,0:km,1))+sin(x(0:im,0:jm,0:km,2))
      manufactured(:,:,:,2)=-sin(x(0:im,0:jm,0:km,1))+sin(x(0:im,0:jm,0:km,2))
      manufactured(:,:,:,3)=sin(x(0:im,0:jm,0:km,3))
      call derive_sample(manufactured,derived)
      write(filename,'(A,".manufactured.rank",I8.8,".bin")') trim(prefix),mpirank
      call write_derived()
      deallocate(manufactured)
    endif
    deallocate(derived)
    deallocate(fields)
  contains
    subroutine write_derived()
      open(newunit=unit,file=trim(filename),status='new',access='stream', &
           form='unformatted',action='write',convert='little_endian',iostat=status)
      call require_sample(status==0,'cannot create derived sample file')
      write(unit,iostat=status) 'ASTRID01',header,time_end,step_dt,derived
      close(unit,iostat=close_status)
      call require_sample(status==0.and.close_status==0,'cannot write derived sample file')
    end subroutine
  end subroutine
end module
