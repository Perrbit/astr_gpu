module insitu_wall_statistics
  use mpi
  use iso_fortran_env, only: real64,int32,int64
  use insitu_velocity_statistics
  use insitu_time_integral, only: clipped_trapezoid
  use insitu_fields, only: nonreacting_wall_candidate
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: accumulate_wall_statistics,wall_statistics_file,finish_wall_statistics,wall_statistics_host_bytes
  public :: wall_statistics_render_values,wall_statistics_render_clock
  type(velocity_statistics),allocatable,save :: states(:,:,:,:)
  real(real64),allocatable,save :: xyz(:,:,:,:)
  logical,allocatable,save :: owned(:,:,:)
  real(real64),save :: previous_time=0.d0,window_saved(2)=0.d0
  real(real64),save :: covered_duration=0.d0
  integer,save :: last_step=-1,nfields=0,profile=0
  integer(int64),save :: samples=0
contains
  subroutine require(condition,message)
    logical,intent(in) :: condition
    character(*),intent(in) :: message
    integer :: bad,any_bad,ierr,ignored
    bad=merge(0,1,condition)
    call MPI_Allreduce(bad,any_bad,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr==MPI_SUCCESS.and.any_bad==0) return
    write(*,'(A)') 'ASTR INSITU WALL STATISTICS ERROR: '//message
    call MPI_Abort(MPI_COMM_WORLD,1,ignored)
  end subroutine

  integer(int64) function wall_statistics_host_bytes() result(bytes)
    use commvar, only: im,km,ia,ja,flowtype
    integer :: nf,nw,groups
    type(velocity_statistics) :: dummy
    nf=3; nw=2
    if(trim(flowtype)=='air5hbl') then
      nf=18; nw=1
    endif
    groups=nf/3
    bytes=int(im+1,int64)*(km+1)*nw*(int(storage_size(dummy)/8,int64)*groups+ &
      (4_int64*34*groups+6_int64*nf+3)*8+4)
    if(nf==18) bytes=bytes+8_int64*(24_int64*(int(ia,int64)+1)+3_int64*(int(im,int64)+1)+ &
      km+11_int64*(int(ja,int64)+1))
  end function

  subroutine capture(window,host_limit,device_limit,reserve,fields,download)
    use commvar, only: flowtype
    use insitu_fields, only: capture_channel_walls,capture_air5_walls
    real(real64),intent(in) :: window(2)
    integer(int64),intent(in) :: host_limit,device_limit,reserve
    real(real64),allocatable,intent(out) :: fields(:,:,:,:)
    integer(int64),intent(out) :: download
    real(real64),allocatable :: all_fields(:,:,:,:)
    integer :: status
    if(nonreacting_wall_candidate()) then
      nfields=3; profile=5
      call capture_channel_walls(xyz,all_fields,owned,host_limit,device_limit,reserve,download)
      allocate(fields(size(owned,1),size(owned,2),size(owned,3),3),stat=status)
      call require(status==0,'cannot allocate selected channel wall fields')
      fields=all_fields(:,:,:,1:3)
    else
      call require(trim(flowtype)=='air5hbl','unsupported scalar wall solver')
      nfields=18; profile=6
      call capture_air5_walls(xyz,fields,owned,host_limit,device_limit,reserve,download)
    endif
    window_saved=window
  end subroutine

  subroutine initialize_device_metadata(window,host_limit)
    use commvar, only: im,jm,km,ia,ja,flowtype,use_gpu
    use commarray, only: x
    use parallel, only: ig0,jg0
    real(real64),intent(in) :: window(2)
    integer(int64),intent(in) :: host_limit
    integer :: walls(2),nw,w,status
    integer(int64) :: bytes
    call require(use_gpu,'resident wall statistics require GPU computation')
    nw=0
    if(nonreacting_wall_candidate()) then
      profile=5; nfields=3
      if(jg0==0) then
        nw=nw+1; walls(nw)=0
      endif
      if(jg0+jm==ja) then
        nw=nw+1; walls(nw)=jm
      endif
    else
      call require(trim(flowtype)=='air5hbl','unsupported resident scalar wall solver')
      profile=6; nfields=18
      if(jg0==0) then
        nw=1; walls(1)=0
      endif
    endif
    if(allocated(owned)) then
      call require(all(shape(owned)==[im+1,km+1,nw]).and.all(window==window_saved), &
        'resident wall metadata shape/window changed')
      return
    endif
    bytes=int(im+1,int64)*(km+1)*nw*(3*8+storage_size(.false.)/8)
    call require(bytes<=host_limit,'resident wall metadata host budget')
    allocate(xyz(0:im,0:km,nw,3),owned(0:im,0:km,nw),stat=status)
    call require(status==0,'cannot allocate static wall metadata')
    do w=1,nw
      xyz(:,:,w,:)=x(0:im,walls(w),0:km,:)
    enddo
    call require(all(ieee_is_finite(xyz)),'nonfinite static wall geometry')
    owned=.false.; owned(0:im-1,0:km-1,:)=.true.
    if(profile==6.and.ig0+im==ia) owned(im,0:km-1,:)=.true.
    window_saved=window
  end subroutine

  subroutine accumulate_wall_statistics(step,time,window,host_limit,device_limit,reserve,separation_prefix,device_transport)
    use commvar, only: use_gpu,im,ia
    use parallel, only: mpirank,ig0
#ifdef _CUDA
    use insitu_wall_statistics_gpu, only: push_wall_statistics_gpu,capture_and_push_wall_statistics_gpu
#endif
    integer,intent(in) :: step
    real(real64),intent(in) :: time,window(2)
    integer(int64),intent(in) :: host_limit,device_limit,reserve
    character(*),intent(in),optional :: separation_prefix
    character(*),intent(in),optional :: device_transport
    real(real64),allocatable :: fields(:,:,:,:),span_weights(:),span_profile(:,:),local_span(:,:)
    real(real64) :: weights(2),duration,span_length,reference_u
    integer(int64) :: download
    integer :: i,k,w,g,status,e,span_comm
    logical :: ok,all_ok,resident,separate
    call require(all(ieee_is_finite([time,window])).and.window(2)>window(1),'invalid scalar window/time')
    if(samples>0) call require(step>last_step.and.time>previous_time.and.all(window==window_saved), &
      'wall scalar sample identity/window mismatch')
    resident=.false.
    if(present(device_transport)) resident=len_trim(device_transport)>0
    download=0
    if(resident) then
      call initialize_device_metadata(window,host_limit)
    else
      call capture(window,host_limit,device_limit,reserve,fields,download)
    endif
    separate=.false.
    if(present(separation_prefix)) separate=len_trim(separation_prefix)>0
    if(separate.and..not.resident) call record_air5_separation(step,time,fields,separation_prefix)
    weights=0.d0
    if(samples>0) then
      call clipped_trapezoid(previous_time,time,[1.d0,0.d0],[0.d0,1.d0], &
        window(1),window(2),weights,duration,ok)
      call require(ok,'wall scalar time weights')
    endif
    do e=1,2
      covered_duration=covered_duration+weights(e)
    enddo
#ifdef _CUDA
    if(use_gpu) then
      if(resident) then
        if(separate) then
          call prepare_air5_separation(span_weights,span_length,reference_u,span_comm)
          allocate(span_profile(0:im,3),local_span(0:ia,3),stat=status)
          call require(status==0,'cannot allocate resident separation profile')
          call capture_and_push_wall_statistics_gpu(profile,device_transport,weights,samples>0, &
            host_limit,device_limit,reserve,span_weights,span_profile)
          local_span=0.d0; local_span(ig0:ig0+im,:)=span_profile
          call publish_air5_separation(step,time,local_span,span_length,reference_u,span_comm,separation_prefix)
        else
          call capture_and_push_wall_statistics_gpu(profile,device_transport,weights,samples>0, &
            host_limit,device_limit,reserve)
        endif
      else
        call push_wall_statistics_gpu(fields,weights,samples>0)
      endif
    else
#else
    call require(.not.use_gpu,'wall scalar device statistics require CUDA')
#endif
      if(.not.allocated(states)) then
        allocate(states(size(owned,1),size(owned,2),size(owned,3),nfields/3),stat=status)
        call require(status==0,'cannot allocate wall scalar states')
        all_ok=.true.
        do g=1,nfields/3
        do w=1,size(owned,3)
        do k=1,size(owned,2)
        do i=1,size(owned,1)
          call configure_velocity_statistics(states(i,k,w,g),window(1),window(2),ok)
          all_ok=all_ok.and.ok
        enddo
        enddo
        enddo
        enddo
        call require(all_ok,'cannot configure wall scalar states')
      endif
      all_ok=.true.
      do g=1,nfields/3
      do w=1,size(owned,3)
      do k=1,size(owned,2)
      do i=1,size(owned,1)
        call push_velocity_sample(states(i,k,w,g),time,1.d0, &
          fields(i+lbound(fields,1)-1,k+lbound(fields,2)-1,w,3*g-2:3*g),ok)
        all_ok=all_ok.and.ok
      enddo
      enddo
      enddo
      enddo
      call require(all_ok,'invalid wall scalar accumulation')
#ifdef _CUDA
    endif
#endif
    previous_time=time; last_step=step; samples=samples+1
    write(*,'(A,I0,A,I0,A,I0,A,I0)') 'ASTR_INSITU_WALL_STATS rank=',mpirank,' step=',step, &
      ' samples=',samples,' field_download_bytes=',download
  end subroutine

  subroutine packed_state(values,writing,clock)
    use commvar, only: use_gpu
#ifdef _CUDA
    use insitu_wall_statistics_gpu, only: pack_wall_statistics_gpu,restore_wall_statistics_gpu
#endif
    real(real64),contiguous,intent(inout) :: values(:,:,:,:)
    logical,intent(in) :: writing
    real(real64),intent(in) :: clock
    type(velocity_statistics) :: candidate
    integer :: i,k,w,g,base,status
    logical :: ok,all_ok
    if(writing.and.use_gpu) then
#ifdef _CUDA
      call pack_wall_statistics_gpu(values,window_saved,clock)
#endif
      return
    endif
    if(.not.writing.and..not.use_gpu) then
      call require(.not.allocated(states),'wall scalar states initialized before restore')
      allocate(states(size(values,1),size(values,2),size(values,3),nfields/3),stat=status)
      call require(status==0,'cannot allocate restored wall scalar states')
    endif
    all_ok=.true.
    do g=1,nfields/3
      base=34*(g-1)
      do w=1,size(values,3)
      do k=1,size(values,2)
      do i=1,size(values,1)
        if(writing) then
          call pack_velocity_state(states(i,k,w,g),values(i,k,w,base+1:base+34),ok)
        else
          call unpack_velocity_state(values(i,k,w,base+1:base+34),candidate,clock, &
            window_saved(1),window_saved(2),ok)
          ok=ok.and.values(i,k,w,base+3)==clock.and.values(i,k,w,base+4)==1.d0.and. &
            values(i,k,w,base+34)==1.d0
          if(ok.and..not.use_gpu) states(i,k,w,g)=candidate
        endif
        all_ok=all_ok.and.ok
      enddo
      enddo
      enddo
    enddo
    call require(all_ok,'invalid packed wall scalar state')
#ifdef _CUDA
    if(.not.writing.and.use_gpu) call restore_wall_statistics_gpu(values)
#endif
  end subroutine

  subroutine wall_statistics_file(path,writing,identity,budget,window,host_limit,device_limit,reserve,root,volume_enabled, &
      separation_enabled,device_transport)
    use commvar, only: ia,ja,ka,im,jm,km,use_gpu,flowtype
    use parallel, only: ig0,jg0,kg0
    use checkpoint_state_io, only: checkpoint_state_identity,checkpoint_state_transfer,allocate_checkpoint_buffer
    character(*),intent(in) :: path
    logical,intent(in) :: writing,root
    logical,intent(in),optional :: volume_enabled
    logical,intent(in),optional :: separation_enabled
    character(*),intent(in),optional :: device_transport
    type(checkpoint_state_identity),intent(in) :: identity
    integer(int64),intent(in) :: budget,host_limit,device_limit,reserve
    real(real64),intent(in) :: window(2)
    type(checkpoint_state_identity) :: stored
    real(real64),allocatable :: buffer(:,:,:,:),values(:,:,:,:),fields(:,:,:,:)
    integer(int64) :: remaining,download
    integer(int64),allocatable :: metadata(:),expected(:)
    integer :: w,j,ncomp,status
    logical :: resident
    ncomp=10
    if(present(volume_enabled)) ncomp=11
    if(present(separation_enabled)) ncomp=12
    allocate(metadata(ncomp),expected(ncomp))
    if(.not.writing) then
      resident=.false.
      if(present(device_transport)) resident=len_trim(device_transport)>0
      if(resident) then
        call initialize_device_metadata(window,host_limit)
      else
        call capture(window,host_limit,device_limit,reserve,fields,download)
      endif
    endif
    ncomp=34*(nfields/3)
    call require(budget>=16_int64*(im+1)*(jm+1)*(km+1)*ncomp+4096,'wall scalar checkpoint budget')
    call allocate_checkpoint_buffer([im,jm,km],0,ncomp,budget,buffer,remaining,MPI_COMM_WORLD)
    allocate(values(im+1,km+1,size(owned,3),ncomp),stat=status)
    call require(status==0,'cannot allocate compact wall state transfer')
    metadata=0; expected=0; buffer=0.d0
    expected(1:10)=[1_int64,int(profile,int64),int(nfields,int64),identity%step,transfer(identity%time,0_int64), &
      transfer(window(1),0_int64),transfer(window(2),0_int64),merge(1_int64,0_int64,use_gpu),samples, &
      transfer(covered_duration,0_int64)]
    if(present(volume_enabled)) expected(11)=merge(1_int64,0_int64,volume_enabled)
    if(present(separation_enabled)) expected(12)=merge(1_int64,0_int64,separation_enabled)
    if(writing) then
      call require(last_step==identity%step.and.previous_time==identity%time,'wall scalar/checkpoint phase mismatch')
      metadata=expected
      call packed_state(values,.true.,identity%time)
      do w=1,size(owned,3)
        j=1
        if(profile==5.and.(jg0/=0.or.w==2)) j=jm+1
        buffer(:,j,:,:)=values(:,:,w,:)
      enddo
    endif
    stored=identity
    if(root) then
      call checkpoint_state_transfer(path,writing,.true.,[ia,ja,ka]+1,[ig0,jg0,kg0], &
        [im,jm,km],0,buffer,stored,remaining,MPI_COMM_WORLD,role=9,metadata=metadata)
    else
      call checkpoint_state_transfer(path,writing,.true.,[ia,ja,ka]+1,[ig0,jg0,kg0], &
        [im,jm,km],0,buffer,stored,remaining,MPI_COMM_WORLD,role=9,metadata=metadata,group_name='wall_statistics')
    endif
    if(writing) return
    call require(all(metadata(1:8)==expected(1:8)).and.metadata(9)>0,'wall scalar metadata mismatch')
    if(present(volume_enabled)) call require(metadata(11)==expected(11),'AIR5 volume statistics selection mismatch')
    if(present(separation_enabled)) call require(metadata(12)==expected(12),'AIR5 separation selection mismatch')
    covered_duration=transfer(metadata(10),covered_duration)
    call require(ieee_is_finite(covered_duration).and.covered_duration>=0.d0,'wall scalar coverage clock')
    call require(stored%step==identity%step.and.stored%time==identity%time.and. &
      stored%dt_used==identity%dt_used.and.stored%dt_next==identity%dt_next,'wall scalar checkpoint clock mismatch')
    do w=1,size(owned,3)
      j=1
      if(profile==5.and.(jg0/=0.or.w==2)) j=jm+1
      values(:,:,w,:)=buffer(:,j,:,:)
      buffer(:,j,:,:)=0.d0
    enddo
    call require(all(buffer==0.d0),'nonzero off-wall scalar checkpoint state')
    do w=1,nfields/3
      call require(all(values(:,:,:,34*(w-1)+8)==covered_duration),'wall scalar coverage differs from moments')
    enddo
    call packed_state(values,.false.,identity%time)
    samples=metadata(9); previous_time=identity%time; last_step=int(identity%step)
  end subroutine

  subroutine read_wall_moments(moments)
    real(real64),allocatable,intent(out) :: moments(:,:,:,:)
    real(real64),allocatable :: values(:,:,:,:)
    type(velocity_statistics) :: candidate
    type(velocity_statistics_result) :: result
    integer :: i,k,w,g,base,status,a
    logical :: ok,all_ok
    call require(samples>0,'wall moments require a sample')
    allocate(values(size(owned,1),size(owned,2),size(owned,3),34*(nfields/3)), &
      moments(size(owned,1),size(owned,2),size(owned,3),3*nfields),stat=status)
    call require(status==0,'wall scalar output allocation')
    call packed_state(values,.true.,previous_time)
    moments=0.d0; all_ok=.true.
    do g=1,nfields/3
    do w=1,size(owned,3)
    do k=1,size(owned,2)
    do i=1,size(owned,1)
      base=34*(g-1)
      call unpack_velocity_state(values(i,k,w,base+1:base+34),candidate,previous_time, &
        window_saved(1),window_saved(2),ok)
      all_ok=all_ok.and.ok
      call read_velocity_statistics(candidate,result,ok)
      if(.not.ok) cycle
      moments(i,k,w,3*g-2:3*g)=result%mean_r
      do a=1,3
        moments(i,k,w,nfields+3*g-3+a)=result%covariance_r(a,a)
      enddo
      moments(i,k,w,2*nfields+3*g-2:2*nfields+3*g)=result%rms_r
    enddo
    enddo
    enddo
    enddo
    call require(all_ok.and.all(ieee_is_finite(moments)),'wall scalar output state')
  end subroutine

  subroutine prepare_air5_separation(span_weights,length,reference_u,comm)
    use commvar, only: ka,km,flowtype
    use commarray, only: x
    use parallel, only: kg0,mpiback,mpifront
#ifdef ASTR_AIR5_CHEMISTRY
    use chemistry_hbl_boundary, only: get_air5_hbl_boundary
#endif
    real(real64),allocatable,intent(out) :: span_weights(:)
    real(real64),intent(out) :: length,reference_u
    integer,intent(out) :: comm
    real(real64),allocatable :: inlet(:,:)
    real(real64) :: zmin,zmax,left,send,receive,farfield(11),wall_temp,xorigin
    integer :: k,ierr,status
    call require(trim(flowtype)=='air5hbl','separation diagnostic requires Cartesian AIR5 HBL')
#ifdef ASTR_AIR5_CHEMISTRY
    call get_air5_hbl_boundary(inlet,farfield,wall_temp,xorigin)
#else
    call require(.false.,'separation diagnostic requires AIR5 chemistry build')
    return
#endif
    call require(all(ieee_is_finite(farfield)).and.farfield(1)>0.d0,'invalid separation reference state')
    reference_u=farfield(2)/farfield(1)
    allocate(span_weights(0:km-1),stat=status)
    call require(status==0,'cannot allocate spanwise shear diagnostic')
    call MPI_Comm_dup(MPI_COMM_WORLD,comm,ierr)
    call require(ierr==MPI_SUCCESS,'separation communicator')
    call MPI_Allreduce(x(0,0,0,3),zmin,1,MPI_DOUBLE_PRECISION,MPI_MIN,comm,ierr)
    call require(ierr==MPI_SUCCESS,'separation lower span bound')
    call MPI_Allreduce(x(0,0,km,3),zmax,1,MPI_DOUBLE_PRECISION,MPI_MAX,comm,ierr)
    call require(ierr==MPI_SUCCESS,'separation upper span bound')
    length=zmax-zmin
    send=x(0,0,km-1,3); receive=send
    if(km/=ka) call MPI_Sendrecv(send,1,MPI_DOUBLE_PRECISION,mpifront,1,receive,1,MPI_DOUBLE_PRECISION, &
      mpiback,1,comm,MPI_STATUS_IGNORE,ierr)
    call require(ierr==MPI_SUCCESS.and.ieee_is_finite(length).and.length>0.d0,'separation span seam')
    if(kg0==0) receive=receive-length
    do k=0,km-1
      left=receive
      if(k>0) left=x(0,0,k-1,3)
      span_weights(k)=.5d0*(x(0,0,k+1,3)-left)
    enddo
    call require(all(ieee_is_finite(span_weights)).and.all(span_weights>0.d0),'invalid physical span weights')
  end subroutine

  subroutine record_air5_separation(step,time,fields,prefix)
    use commvar, only: ia,im,km
    use parallel, only: ig0,jg0
    integer,intent(in) :: step
    real(real64),intent(in) :: time,fields(0:,0:,:,:)
    character(*),intent(in) :: prefix
    real(real64),allocatable :: local(:,:),span_weights(:)
    real(real64) :: length,reference_u,weight
    integer :: i,k,n,comm,status
    call prepare_air5_separation(span_weights,length,reference_u,comm)
    allocate(local(0:ia,3),stat=status)
    call require(status==0,'cannot allocate spanwise shear diagnostic')
    local=0.d0
    if(jg0==0) then
      do k=0,km-1
        weight=span_weights(k)
        do i=0,im
          if(.not.owned(i,k,1)) cycle
          n=ig0+i
          local(n,1)=local(n,1)+weight*xyz(i,k,1,1)
          local(n,2)=local(n,2)+weight*fields(i,k,1,13)
          local(n,3)=local(n,3)+weight
        enddo
      enddo
    endif
    call require(all(ieee_is_finite(local)),'nonfinite spanwise shear integral')
    call publish_air5_separation(step,time,local,length,reference_u,comm,prefix)
  end subroutine

  subroutine publish_air5_separation(step,time,local,length,reference_u,comm,prefix)
    use commvar, only: ia
    use parallel, only: mpirank
    use insitu_wall_separation, only: wall_crossing,find_wall_crossings,separation,reattachment,zero_interval
    integer,intent(in) :: step
    integer,intent(inout) :: comm
    real(real64),intent(in) :: time,local(0:,:),length,reference_u
    character(*),intent(in) :: prefix
    type(wall_crossing),allocatable :: events(:)
    real(real64),allocatable :: global(:,:),positions(:),shear(:)
    character(1200) :: filename
    character(48) :: eligibility
    character(16) :: classification
    integer :: i,ierr,status,unit,closed,pairs
    logical :: ok,eligible
    call require(all(shape(local)==[ia+1,3]).and.all(ieee_is_finite(local)), &
      'invalid spanwise shear reduction')
    allocate(global(0:ia,3),positions(ia+1),shear(ia+1),stat=status)
    call require(status==0,'cannot allocate reduced wall profile')
    call MPI_Allreduce(local,global,size(local),MPI_DOUBLE_PRECISION,MPI_SUM,comm,ierr)
    call require(ierr==MPI_SUCCESS.and.all(ieee_is_finite(global)).and.all(global(:,3)>0.d0), &
      'invalid unique wall span measure')
    call MPI_Comm_free(comm,ierr)
    call require(ierr==MPI_SUCCESS,'separation communicator release')
    positions=global(:,1)/global(:,3); shear=global(:,2)/global(:,3)
    call require(all(ieee_is_finite([positions,shear,reference_u])).and. &
      all(positions(2:ia+1)>positions(1:ia)),'invalid wall diagnostic profile')
    eligible=reference_u>0.d0
    eligibility='not_applicable_no_positive_inflow'
    if(eligible) eligibility='eligible_positive_x_inflow'
    pairs=0
    if(eligible) then
      call find_wall_crossings(positions,shear,events,pairs,ok)
      call require(ok,'invalid wall shear crossing sequence')
    else
      allocate(events(0))
    endif
    status=0; closed=0
    if(mpirank==0) then
      write(filename,'(A,".wall_separation.step",I8.8,".csv")') trim(prefix),step
      open(newunit=unit,file=trim(filename),status='new',action='write',iostat=status)
      if(status==0) then
        write(unit,'(A)',iostat=status) '# Cartesian bottom wall; tangent=+x; z average=physical length; no periodic x wrap'
        if(status==0) write(unit,'(A,2(1X,ES26.17),1X,I0)',iostat=status) &
          '# reference_u/span_period/complete_pairs ',reference_u,length,pairs
        if(status==0) write(unit,'(A)',iostat=status) 'record,step,time,x_lower,x_upper,value,pair,classification,status'
        do i=1,ia+1
          if(status/=0) exit
          write(unit,'(A,",",I0,4(",",ES26.17),",0,profile,",A)',iostat=status) &
            'profile',step,time,positions(i),positions(i),shear(i),trim(eligibility)
        enddo
        do i=1,size(events)
          if(status/=0) exit
          select case(events(i)%kind)
          case(separation)
            classification='separation'
          case(reattachment)
            classification='reattachment'
          case(zero_interval)
            classification='zero_interval'
          end select
          write(unit,'(A,",",I0,4(",",ES26.17),",",I0,",",A,",",A)',iostat=status) &
            'crossing',step,time,events(i)%lower,events(i)%upper,events(i)%bubble_length,events(i)%pair, &
            trim(classification),trim(eligibility)
        enddo
        close(unit,iostat=closed)
      endif
    endif
    call require(status==0.and.closed==0,'cannot publish wall separation diagnostic')
  end subroutine

  subroutine wall_statistics_render_clock(step,time,duration,window)
    integer,intent(in) :: step
    real(real64),intent(in) :: time
    real(real64),intent(out) :: duration,window(2)
    call require(samples>0.and.last_step==step.and.previous_time==time,'wall mean render phase mismatch')
    call require(ieee_is_finite(covered_duration).and.covered_duration>=0.d0.and. &
      all(ieee_is_finite(window_saved)).and.window_saved(2)>window_saved(1),'invalid wall mean render clock')
    duration=covered_duration; window=window_saved
  end subroutine

  subroutine wall_statistics_render_values(step,time,fields)
    use commvar, only: im,km,ia,ka,flowtype
    use parallel, only: mpileft,mpiright,mpiback,mpifront
    integer,intent(in) :: step
    real(real64),intent(in) :: time
    real(real64),allocatable,intent(out) :: fields(:,:,:,:)
    real(real64),allocatable :: moments(:,:,:,:),send(:),receive(:)
    integer :: comm,ierr,axis,n,negative(2),positive(2),status
    call require(last_step==step.and.previous_time==time,'wall mean render phase mismatch')
    call read_wall_moments(moments)
    allocate(fields(size(owned,1),size(owned,2),size(owned,3),3*nfields+3),stat=status)
    call require(status==0,'cannot allocate wall mean render fields')
    fields(:,:,:,1:3*nfields)=moments
    fields(:,:,:,3*nfields+1)=covered_duration
    fields(:,:,:,3*nfields+2)=window_saved(1)
    fields(:,:,:,3*nfields+3)=window_saved(2)
    negative=[mpileft,mpiback]; positive=[mpiright,mpifront]
    call MPI_Comm_dup(MPI_COMM_WORLD,comm,ierr)
    call require(ierr==MPI_SUCCESS,'wall mean render communicator')
    do axis=1,2
      if(axis==1) then
        n=size(fields,2)*size(fields,3)*size(fields,4)
      else
        n=size(fields,1)*size(fields,3)*size(fields,4)
      endif
      allocate(send(n),receive(n),stat=status)
      call require(status==0,'cannot allocate wall mean seam buffers')
      if(axis==1) then
        send=reshape(fields(1,:,:,:),[n])
      else
        send=reshape(fields(:,1,:,:),[n])
      endif
      if((axis==1.and.im==ia.and.profile==5).or.(axis==2.and.km==ka)) then
        receive=send
      else
        call MPI_Sendrecv(send,n,MPI_DOUBLE_PRECISION,negative(axis),axis,receive,n,MPI_DOUBLE_PRECISION, &
          positive(axis),axis,comm,MPI_STATUS_IGNORE,ierr)
      endif
      call require(ierr==MPI_SUCCESS,'wall mean endpoint exchange')
      if(axis==1) then
        if(positive(axis)/=MPI_PROC_NULL.or.(im==ia.and.profile==5)) &
          fields(im+1,:,:,:)=reshape(receive,[size(fields,2),size(fields,3),size(fields,4)])
      else
        fields(:,km+1,:,:)=reshape(receive,[size(fields,1),size(fields,3),size(fields,4)])
      endif
      deallocate(send,receive)
    enddo
    call MPI_Comm_free(comm,ierr)
    call require(ierr==MPI_SUCCESS,'wall mean communicator release')
    call require(all(ieee_is_finite(fields)),'nonfinite wall render statistics')
  end subroutine

  subroutine finish_wall_statistics(prefix)
    use commvar, only: use_gpu
    use parallel, only: mpirank,ig0,jg0,kg0
#ifdef _CUDA
    use insitu_wall_statistics_gpu, only: release_wall_statistics_gpu
#endif
    character(*),intent(in) :: prefix
    real(real64),allocatable :: moments(:,:,:,:)
    character(1200) :: filename
    integer :: unit,status,closed
    if(samples==0) return
    call read_wall_moments(moments)
    write(filename,'(A,".wall_statistics.step",I8.8,".rank",I8.8,".bin")') trim(prefix),last_step,mpirank
    open(newunit=unit,file=trim(filename),status='new',access='stream',form='unformatted', &
      action='write',convert='little_endian',iostat=status)
    call require(status==0,'cannot create wall scalar output')
    write(unit,iostat=status) 'ASTRWS01',int([1,last_step,mpirank,shape(owned),ig0,jg0,kg0,nfields,profile],int32), &
      [previous_time,window_saved,covered_duration],samples,xyz,moments,int(merge(1,0,owned),int32)
    close(unit,iostat=closed)
    call require(status==0.and.closed==0,'cannot write wall scalar output')
    if(allocated(states)) deallocate(states)
    deallocate(xyz,owned)
#ifdef _CUDA
    if(use_gpu) call release_wall_statistics_gpu()
#endif
    samples=0; last_step=-1; covered_duration=0.d0
  end subroutine
end module
