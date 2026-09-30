module output_runtime
  use iso_fortran_env, only: int64,real64
  use mpi
  use commvar, only: ia,ja,ka,im,jm,km,hm,numq,num_species,num_modequ, &
    nstep,time,deltat,maxstep,use_gpu,flowtype,lcomb,lavg,lcracon,limmbou,lrestart, &
    lwsequ,lwslic,feqchkpt,feqwsequ,feqslice,feqlist,feqavg,conschm,difschm,rkscheme
  use commarray, only: q,rho,vel,prs,tmp,x,jacob,dxi
  use parallel, only: ig0,jg0,kg0,mpirank
  use bc, only: bctype
  use output_config, only: output_options
  use output_config_collective, only: read_output_options_collective
  use checkpoint_state_io
  use checkpoint_bundle
  use insitu_checkpoint_batch, only: create_batch,file_fingerprint,copy_batch_file
#ifdef _CUDA
  use checkpoint_state_gpu, only: checkpoint_perfect_gas_gpu
#endif
  implicit none
  private
  public :: configure_output_runtime,begin_output_runtime,completed_output_runtime,new_output_enabled
  logical,save :: enabled=.false.
  type(output_options),save :: options
  type(checkpoint_retention),save :: ledger
  integer(int64),save :: contract(14),last_step=-1
  real(real64),save :: next_time=0
contains
  logical function new_output_enabled()
    new_output_enabled=enabled
  end function

  subroutine check(ok,message)
    logical,intent(in) :: ok
    character(*),intent(in) :: message
    call checkpoint_state_require(ok,MPI_COMM_WORLD,message)
  end subroutine

  subroutine check_capability()
    call check(trim(flowtype)=='tgv'.and.numq==5.and.num_species==0.and.num_modequ==0.and. &
      .not.lcomb.and..not.lavg.and..not.lcracon.and..not.limmbou.and..not.lrestart.and. &
      all(bctype==1).and.trim(rkscheme)=='rk3', 'new output currently admits periodic nonreacting TGV RK3 only')
    call check(.not.options%volume%enabled.and..not.options%slices%enabled, &
      'volume and slices are not yet connected to the new output lifecycle')
    call check(.not.lwsequ.and..not.lwslic,'legacy field sequences must be disabled with new output')
    call check(options%host_budget_bytes>0,'new output requires an explicit host buffer budget')
  end subroutine

  subroutine configure_output_runtime()
    character(1024) :: config
    character(256) :: message,value
    integer :: status,length,flag,minflag,maxflag,ierr
    logical :: ok
    config=''
    call get_environment_variable('ASTR_OUTPUT_CONFIG',config,length=length,status=status)
    flag=0
    if (status==0.and.length>0) flag=1
    call MPI_Allreduce(flag,minflag,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
    call check(ierr==MPI_SUCCESS,'output activation reduce')
    call MPI_Allreduce(flag,maxflag,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    call check(ierr==MPI_SUCCESS.and.minflag==maxflag.and.status/= -1,'inconsistent output activation')
    if (maxflag==0) return
    enabled=.true.
    call read_output_options_collective(trim(config),options,MPI_COMM_WORLD,ok,message)
    call check(ok,trim(message))
    call check_capability()
    value=''
    call get_environment_variable('ASTR_INSITU_CONFIG',value,status=status)
    call check(status==1.or.(status==0.and.len_trim(value)==0), &
      'formal in situ state is not yet registered in new checkpoints')
  end subroutine

  subroutine begin_output_runtime(counter,rkfirst_pending)
    integer,intent(inout) :: counter
    logical,intent(inout) :: rkfirst_pending
    character(1024) :: path,source,executable
    character(256) :: value
    integer :: status,ierr,i
    integer(int64) :: bytes,crc
    logical :: ok
    type(checkpoint_state_identity) :: identity
    if (.not.enabled) return
    call check_capability()
    ! Match the executable and primary input; controller stop time may differ.
    call get_command_argument(0,executable,status=status)
    call check(status==0,'executable path unavailable')
    call get_command_argument(2,source,status=status)
    call check(status==0,'input path unavailable')
    contract=0
    if (mpirank==0) then
      call file_fingerprint(trim(executable),bytes,crc,ok)
      contract(1:2)=[bytes,crc]
    else
      ok=.true.
    endif
    call check(ok,'cannot fingerprint executable')
    if (mpirank==0) then
      call file_fingerprint(trim(source),bytes,crc,ok)
      contract(3:4)=[bytes,crc]
    endif
    call check(ok,'cannot fingerprint primary input')
    call MPI_Bcast(contract,14,MPI_INTEGER8,0,MPI_COMM_WORLD,ierr)
    call check(ierr==MPI_SUCCESS,'contract broadcast')
    contract(5:10)=[int(feqchkpt,int64),int(feqwsequ,int64),int(feqslice,int64), &
      int(feqlist,int64),int(feqavg,int64),int(merge(1,0,use_gpu),int64)]
    ! Numerical GPU candidates are not silently accepted by this first gate.
    do i=1,3
      value=''
      select case(i)
      case(1)
        call get_environment_variable('ASTR_GPU_PRECISION_MODE',value,status=status)
        call check(status==1.or.trim(value)==''.or.trim(value)=='fp64','new restart requires FP64')
      case(2)
        call get_environment_variable('ASTR_GPU_SYNC_MODE',value,status=status)
        call check(status==1.or.trim(value)==''.or.trim(value)=='explicit','new restart first gate requires explicit sync')
      case(3)
        call get_environment_variable('ASTR_GPU_FILTER_WORKSPACE',value,status=status)
        contract(11)=merge(1_int64,0_int64,trim(value)=='full')
      end select
    enddo
    next_time=options%checkpoint%interval_time
    if (len_trim(options%restore_directory)>0) then
      path=trim(options%restore_directory)
      call validate_checkpoint_bundle(trim(path),MPI_COMM_WORLD,ok)
      call check(ok,'invalid new checkpoint bundle')
      call control_file(trim(path)//'/control.bin',.false.,counter,rkfirst_pending,identity)
      call geometry_file(trim(path)//'/../../resources/geometry.h5',.false.)
      call flow_file(trim(path)//'/state.h5',.false.,identity)
      call check(identity%step<=huge(nstep),'restored step exceeds solver integer range')
      nstep=int(identity%step)
      time=identity%time
      deltat=identity%dt_next
      call check(nstep<=maxstep+1,'restored step lies past requested stop')
    endif
    if (.not.options%checkpoint%enabled) return
    ! The caller prepares the output root; exclusive subdirectories prevent reuse.
    call create_batch(trim(options%directory)//'/resources',MPI_COMM_WORLD,ok)
    call check(ok,'cannot create new run resources directory')
    call create_batch(trim(options%directory)//'/checkpoints',MPI_COMM_WORLD,ok)
    call check(ok,'cannot create new run checkpoints directory')
    call geometry_file(trim(options%directory)//'/resources/geometry.h5',.true.)
    if (mpirank==0) call copy_batch_file(trim(source),trim(options%directory)//'/resources/input.txt',ok)
    call check(ok,'cannot freeze primary input resource')
    if (options%checkpoint%initial_frame.and.last_step/=int(nstep,int64)) &
      call completed_output_runtime(0.0_real64,counter,rkfirst_pending,initial=.true.)
  end subroutine

  subroutine flow_file(path,writing,identity)
    character(*),intent(in) :: path
    logical,intent(in) :: writing
    type(checkpoint_state_identity),intent(inout) :: identity
    type(checkpoint_state_identity) :: expected
    expected=identity
#ifdef _CUDA
    if (use_gpu) then
      call checkpoint_perfect_gas_gpu(path,writing,[ia,ja,ka]+1,[ig0,jg0,kg0],[im,jm,km],hm, &
        identity,options%host_budget_bytes,MPI_COMM_WORLD)
    else
#endif
      call checkpoint_perfect_gas_state(path,writing,[ia,ja,ka]+1,[ig0,jg0,kg0],[im,jm,km],hm, &
        q,rho,vel,prs,tmp,identity,options%host_budget_bytes,MPI_COMM_WORLD)
#ifdef _CUDA
    endif
#endif
    if (.not.writing) call check(identity%step==expected%step.and.identity%time==expected%time.and. &
      identity%dt_used==expected%dt_used.and.identity%dt_next==expected%dt_next,'flow/control clock mismatch')
  end subroutine

  subroutine geometry_file(path,writing)
    character(*),intent(in) :: path
    logical,intent(in) :: writing
    real(real64),allocatable :: buffer(:,:,:,:)
    integer(int64) :: remaining
    integer :: a,b,m,h,i,j,k
    type(checkpoint_state_identity) :: identity
    identity=checkpoint_state_identity(0_int64,0.0_real64,0.0_real64,1.0_real64)
    call allocate_checkpoint_buffer([im,jm,km],hm,13,options%host_budget_bytes,buffer,remaining,MPI_COMM_WORLD)
    h=hm+1
    if (writing) then
      ! gridsendrecv defines physical nodes and face halos, not edge/corner padding.
      buffer(:,:,:,1:3)=0.0_real64
      buffer(:,h:h+jm,h:h+km,1:3)=x(:,0:jm,0:km,:)
      buffer(h:h+im,:,h:h+km,1:3)=x(0:im,:,0:km,:)
      buffer(h:h+im,h:h+jm,:,1:3)=x(0:im,0:jm,:,:)
      buffer(:,:,:,4)=jacob
      m=4
      do b=1,3
        do a=1,3
          m=m+1
          buffer(:,:,:,m)=dxi(:,:,:,a,b)
        enddo
      enddo
      ! Metric exchanges also leave edge/corner storage outside their stencil.
      do k=1,size(buffer,3)
        do j=1,size(buffer,2)
          do i=1,size(buffer,1)
            if (count([i<h.or.i>h+im,j<h.or.j>h+jm,k<h.or.k>h+km])>=2) &
              buffer(i,j,k,:)=0.0_real64
          enddo
        enddo
      enddo
    endif
    call checkpoint_state_transfer(path,writing,.true.,[ia,ja,ka]+1,[ig0,jg0,kg0], &
      [im,jm,km],hm,buffer,identity,remaining,MPI_COMM_WORLD)
    if (.not.writing) then
      do m=1,3
        call check_geometry_field(buffer(:,:,:,m),x(:,:,:,m))
      enddo
      call check_geometry_field(buffer(:,:,:,4),jacob)
      m=4
      do b=1,3
        do a=1,3
          m=m+1
          call check_geometry_field(buffer(:,:,:,m),dxi(:,:,:,a,b))
        enddo
      enddo
    endif
  end subroutine

  subroutine check_geometry_field(saved,actual)
    real(real64),intent(in) :: saved(:,:,:),actual(:,:,:)
    integer :: h
    h=hm+1
    call check(all(saved(:,h:h+jm,h:h+km)==actual(:,h:h+jm,h:h+km)).and. &
      all(saved(h:h+im,:,h:h+km)==actual(h:h+im,:,h:h+km)).and. &
      all(saved(h:h+im,h:h+jm,:)==actual(h:h+im,h:h+jm,:)), &
      'defined geometry restart mismatch')
  end subroutine

  subroutine control_file(path,writing,counter,pending,identity)
    character(*),intent(in) :: path
    logical,intent(in) :: writing
    integer,intent(inout) :: counter
    logical,intent(inout) :: pending
    type(checkpoint_state_identity),intent(inout) :: identity
    integer(int64) :: saved(14),interval,saved_last
    real(real64) :: saved_next,dt_interval
    integer :: unit,err,closed,saved_counter
    logical :: saved_pending
    character(8) :: magic
    character(16) :: mode
    ! Small control metadata is read by all ranks; no field data is gathered.
    if (writing) then
      err=0
      closed=0
      if (mpirank==0) then
        open(newunit=unit,file=path,status='new',access='stream',form='unformatted', &
          convert='little_endian',action='write',iostat=err)
        if (err==0) then
          write(unit,iostat=err) 'ASTROC01',contract,identity%step,identity%time,identity%dt_used,identity%dt_next, &
            counter,pending,last_step,next_time,options%checkpoint%mode,options%checkpoint%interval_steps, &
            options%checkpoint%interval_time
          close(unit,iostat=closed)
        endif
      endif
      call check(err==0.and.closed==0,'write control metadata')
    else
      open(newunit=unit,file=path,status='old',access='stream',form='unformatted', &
        convert='little_endian',action='read',iostat=err)
      call check(err==0,'open control metadata')
      read(unit,iostat=err) magic,saved,identity%step,identity%time,identity%dt_used,identity%dt_next, &
        saved_counter,saved_pending,saved_last,saved_next,mode,interval,dt_interval
      close(unit,iostat=closed)
      call check(err==0.and.closed==0,'read control metadata')
      call check(magic=='ASTROC01'.and.all(saved==contract),'numerical/executable/controller contract mismatch')
      counter=saved_counter
      pending=saved_pending
      last_step=saved_last
      if (options%restart_output=='saved') then
        call check(mode==options%checkpoint%mode.and.interval==options%checkpoint%interval_steps.and. &
          dt_interval==options%checkpoint%interval_time,'saved output schedule differs; select explicit override')
        next_time=saved_next
      else
        next_time=identity%time+options%checkpoint%interval_time
      endif
    endif
  end subroutine

  subroutine completed_output_runtime(dt_used,counter,pending,initial)
    real(real64),intent(in) :: dt_used
    integer,intent(inout) :: counter
    logical,intent(inout) :: pending
    logical,optional,intent(in) :: initial
    logical :: due,ok
    character(64) :: name
    character(1200) :: path
    character(128) :: resources(2),files(3)
    type(checkpoint_state_identity) :: identity
    if (.not.enabled) return
    call check_capability()
    if (.not.options%checkpoint%enabled) return
    due=nstep>maxstep
    if (present(initial)) due=due.or.initial
    if (options%checkpoint%mode=='steps') then
      due=due.or.mod(int(nstep,int64),options%checkpoint%interval_steps)==0
    else
      due=due.or.time>=next_time
    endif
    if (.not.due.or.last_step==int(nstep,int64)) return
    if (options%checkpoint%mode=='time'.and.time>=next_time) then
      do while(next_time<=time)
        next_time=next_time+options%checkpoint%interval_time
      enddo
    endif
    write(name,'("step",i12.12)') nstep
    path=trim(options%directory)//'/checkpoints/'//trim(name)//'.tmp'
    call create_batch(trim(path),MPI_COMM_WORLD,ok)
    call check(ok,'cannot create checkpoint candidate')
    identity=checkpoint_state_identity(int(nstep,int64),time,dt_used,deltat)
    call flow_file(trim(path)//'/state.h5',.true.,identity)
    last_step=nstep
    call control_file(trim(path)//'/control.bin',.true.,counter,pending,identity)
    resources=[character(128) :: 'geometry.h5','input.txt']
    call write_checkpoint_resource_refs(trim(path),resources,MPI_COMM_WORLD,ok)
    call check(ok,'cannot record checkpoint resources')
    files=[character(128) :: 'state.h5','control.bin','RESOURCES']
    call seal_checkpoint_bundle(trim(path),files,MPI_COMM_WORLD,ok)
    call check(ok,'cannot seal checkpoint')
    call publish_retained_checkpoint(trim(options%directory)//'/checkpoints',trim(name),options%keep, &
      ledger,MPI_COMM_WORLD,ok)
    call check(ok,'cannot publish/retain checkpoint')
    if (mpirank==0) write(*,'(a,i0,a,es24.16)') 'ASTR_OUTPUT complete_step=',nstep,' time=',time
  end subroutine
end module
