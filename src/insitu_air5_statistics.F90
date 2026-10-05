module insitu_air5_statistics
  use mpi
  use iso_fortran_env, only: real64,int32,int64
  use ieee_arithmetic, only: ieee_is_finite
  use insitu_velocity_statistics
  use insitu_time_integral, only: clipped_trapezoid
  use insitu_spatial_statistics, only: spatial_mean
  implicit none
  private
  public :: accumulate_air5_statistics,air5_statistics_file,finish_air5_statistics,air5_statistics_host_bytes
  type(velocity_statistics),allocatable,save :: states(:,:,:,:)
  type(velocity_statistics),save :: regional
  real(real64),allocatable,save :: weights(:,:,:)
  real(real64),save :: window_saved(2)=0.d0,previous_time=0.d0,covered_duration=0.d0,volume_saved=0.d0
  integer,save :: extent(3)=0,last_step=-1
  integer(int64),save :: samples=0
  logical,save :: reduction_saved=.false.
contains
  subroutine require(ok,message)
    logical,intent(in) :: ok
    character(*),intent(in) :: message
    integer :: bad,any_bad,ierr,ignored
    bad=merge(0,1,ok)
    call MPI_Allreduce(bad,any_bad,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr==MPI_SUCCESS.and.any_bad==0) return
    write(*,'(A)') 'ASTR INSITU AIR5 STATISTICS ERROR: '//message
    call MPI_Abort(MPI_COMM_WORLD,1,ignored)
  end subroutine

  integer(int64) function air5_statistics_host_bytes() result(bytes)
    use commvar, only: im,jm,km
    type(velocity_statistics) :: dummy
    bytes=int(im+1,int64)*(jm+1)*(km+1)*(4_int64*storage_size(dummy)/8+ &
      (4_int64*136+62+6)*8)+4096
  end function

  subroutine initialize(window,reduction)
    use commvar, only: im,jm,km,ia,ja,ka,flowtype,numq,num_species,num_modequ,nondimen,lreadgrid,lcomb,difschm
    use commarray, only: x
    use parallel, only: ig0,jg0,kg0,mpileft,mpiright,mpidown,mpiup,mpiback,mpifront,mpisize
    use bc, only: bctype
    real(real64),intent(in) :: window(2)
    logical,intent(in) :: reduction
    real(real64),allocatable :: axis_weights(:,:),line(:)
    real(real64) :: send,receive,left,right,local_bounds(2),bounds(2),length
    integer :: starts(3),global_size(3),local_size(3),negative(3),positive(3)
    integer :: axis,n,i,j,k,status,comm,ierr
    call require(trim(flowtype)=='air5hbl'.and.numq==11.and.num_species==5.and.num_modequ==1.and.lcomb.and. &
      .not.nondimen.and..not.lreadgrid.and.all(bctype==[11,50,41,51,1,1]).and. &
      trim(difschm)=='643e'.and.max(ia,ja,ka)<=32.and.mpisize<=2, &
      'volume candidate requires dimensional Cartesian noncatalytic AIR5 HBL <=32 NP=1/2')
    call require(all(ieee_is_finite(window)).and.window(2)>window(1),'invalid volume statistics window')
    if(allocated(weights)) then
      call require(all(window==window_saved),'volume statistics window changed')
      call require(reduction.eqv.reduction_saved,'volume reduction selection changed')
      return
    endif
    starts=[ig0,jg0,kg0]; global_size=[ia,ja,ka]; local_size=[im,jm,km]
    negative=[mpileft,mpidown,mpiback]; positive=[mpiright,mpiup,mpifront]
    extent=local_size
    do axis=1,2
      if(starts(axis)+local_size(axis)==global_size(axis)) extent(axis)=extent(axis)+1
    enddo
    allocate(axis_weights(maxval(local_size)+1,3),weights(extent(1),extent(2),extent(3)), &
      line(0:maxval(local_size)),stat=status)
    call require(status==0,'cannot allocate physical volume weights')
    axis_weights=0.d0
    call MPI_Comm_dup(MPI_COMM_WORLD,comm,ierr)
    call require(ierr==MPI_SUCCESS,'volume geometry communicator')
    do axis=1,3
      n=local_size(axis)
      select case(axis)
      case(1)
        line(0:n)=x(0:im,0,0,1)
      case(2)
        line(0:n)=x(0,0:jm,0,2)
      case(3)
        line(0:n)=x(0,0,0:km,3)
      end select
      call require(all(ieee_is_finite(line(0:n))).and.all(line(1:n)>line(0:n-1)), &
        'nonmonotone Cartesian coordinate axis')
      local_bounds=[line(0),line(n)]
      call MPI_Allreduce(local_bounds(1),bounds(1),1,MPI_DOUBLE_PRECISION,MPI_MIN,comm,ierr)
      call require(ierr==MPI_SUCCESS,'volume lower bound reduction')
      call MPI_Allreduce(local_bounds(2),bounds(2),1,MPI_DOUBLE_PRECISION,MPI_MAX,comm,ierr)
      call require(ierr==MPI_SUCCESS,'volume upper bound reduction')
      length=bounds(2)-bounds(1)
      send=line(n-1); receive=line(0)
      if(axis==3.and.n==global_size(axis)) then
        receive=send; ierr=MPI_SUCCESS
      else
        call MPI_Sendrecv(send,1,MPI_DOUBLE_PRECISION,positive(axis),axis,receive,1,MPI_DOUBLE_PRECISION, &
          negative(axis),axis,comm,MPI_STATUS_IGNORE,ierr)
      endif
      call require(ierr==MPI_SUCCESS,'volume coordinate seam exchange')
      if(axis==3.and.starts(axis)==0) receive=receive-length
      do i=0,extent(axis)-1
        left=line(i); right=line(i)
        if(i>0) left=line(i-1)
        if(i==0.and.(axis==3.or.starts(axis)>0)) left=receive
        if(i<n) right=line(i+1)
        axis_weights(i+1,axis)=.5d0*(right-left)
      enddo
      call require(all(axis_weights(1:extent(axis),axis)>0.d0),'invalid physical nodal measure')
    enddo
    call MPI_Comm_free(comm,ierr)
    call require(ierr==MPI_SUCCESS,'volume geometry communicator release')
    do k=1,extent(3)
    do j=1,extent(2)
    do i=1,extent(1)
      weights(i,j,k)=axis_weights(i,1)*axis_weights(j,2)*axis_weights(k,3)
    enddo
    enddo
    enddo
    call require(all(ieee_is_finite(weights)).and.all(weights>0.d0),'nonfinite volume measure')
    window_saved=window
    reduction_saved=reduction
  end subroutine

  subroutine capture_host(i,j,k,density,triples)
    use commarray, only: rho,vel,tmp,tve,spc
    integer,intent(in) :: i,j,k
    real(real64),intent(out) :: density,triples(3,4)
    density=rho(i,j,k)
    triples(:,1)=vel(i,j,k,:)
    triples(:,2)=[tmp(i,j,k),tve(i,j,k),spc(i,j,k,1)]
    triples(:,3)=spc(i,j,k,2:4)
    triples(:,4)=[spc(i,j,k,5),0.d0,0.d0]
  end subroutine

  subroutine allocate_cpu()
    integer :: i,j,k,g,status
    logical :: ok,all_ok
    if(allocated(states)) return
    allocate(states(extent(1),extent(2),extent(3),4),stat=status)
    call require(status==0,'cannot allocate volume CPU accumulators')
    all_ok=.true.
    do g=1,4
    do k=1,extent(3)
    do j=1,extent(2)
    do i=1,extent(1)
      call configure_velocity_statistics(states(i,j,k,g),window_saved(1),window_saved(2),ok)
      all_ok=all_ok.and.ok
    enddo
    enddo
    enddo
    enddo
    call require(all_ok,'cannot configure volume CPU accumulators')
  end subroutine

  subroutine accumulate_air5_statistics(step,time,window,reduction)
    use commvar, only: use_gpu
    use parallel, only: mpirank
#ifdef _CUDA
    use insitu_air5_statistics_gpu, only: push_air5_statistics_gpu
#endif
    integer,intent(in) :: step
    real(real64),intent(in) :: time,window(2)
    logical,intent(in) :: reduction
    real(real64),allocatable :: signals(:,:)
    real(real64) :: time_weights(2),duration,density,triples(3,4),variance(3),mean(3),volume
    type(velocity_statistics_result) :: result
    integer :: i,j,k,g,n,status,e
    logical :: ok,all_ok
    call initialize(window,reduction)
    call require(ieee_is_finite(time),'nonfinite volume sample time')
    if(samples>0) call require(step>last_step.and.time>previous_time,'volume sample identity did not advance')
    time_weights=0.d0
    if(samples==0) then
      call configure_velocity_statistics(regional,window(1),window(2),ok)
      call require(ok,'cannot configure volume regional signal')
    else
      call clipped_trapezoid(previous_time,time,[1.d0,0.d0],[0.d0,1.d0],window(1),window(2), &
        time_weights,duration,ok)
      call require(ok,'volume endpoint time weights')
    endif
    do e=1,2
      covered_duration=covered_duration+time_weights(e)
    enddo
#ifdef _CUDA
    if(use_gpu) then
      call push_air5_statistics_gpu(extent,weights,time_weights,samples>0,volume,mean,variance)
    else
#else
    call require(.not.use_gpu,'volume device statistics require CUDA')
#endif
      call allocate_cpu()
      allocate(signals(6,size(weights)),stat=status)
      call require(status==0,'cannot allocate CPU regional signal input')
      n=0; all_ok=.true.
      do k=1,extent(3)
      do j=1,extent(2)
      do i=1,extent(1)
        n=n+1
        call capture_host(i-1,j-1,k-1,density,triples)
        do g=1,4
          call push_velocity_sample(states(i,j,k,g),time,merge(density,1.d0,g==1),triples(:,g),ok)
          all_ok=all_ok.and.ok
        enddo
        signals(1:3,n)=triples(:,1); signals(4:6,n)=0.d0
        call read_velocity_statistics(states(i,j,k,1),result,ok)
        if(ok) signals(4:6,n)=[result%covariance_r(1,1),result%covariance_r(2,2),result%covariance_r(3,3)]
      enddo
      enddo
      enddo
      call require(all_ok,'invalid completed volume sample')
      block
        real(real64) :: reduced(6)
        call spatial_mean(reshape(weights,[size(weights)]),signals,MPI_COMM_WORLD,volume,reduced,ok)
        call require(ok,'invalid CPU physical volume reduction')
        mean=reduced(1:3); variance=reduced(4:6)
      end block
#ifdef _CUDA
    endif
#endif
    if(reduction_saved) then
      call push_velocity_sample(regional,time,1.d0,mean,ok)
      call require(ok,'invalid regional volume velocity sample')
    endif
    volume_saved=volume; previous_time=time; last_step=step; samples=samples+1
    write(*,'(A,I0,A,I0,A,I0,A,I0,A,ES24.16)') 'ASTR_INSITU_AIR5_VOLUME rank=',mpirank,' step=',step, &
      ' samples=',samples,' owned_nodes=',product(extent),' physical_volume=',volume
  end subroutine

  subroutine pack_state(values,writing,clock)
    use commvar, only: use_gpu
#ifdef _CUDA
    use insitu_air5_statistics_gpu, only: pack_air5_statistics_gpu,restore_air5_statistics_gpu
#endif
    real(real64),contiguous,intent(inout) :: values(:,:,:,:)
    logical,intent(in) :: writing
    real(real64),intent(in) :: clock
    type(velocity_statistics) :: candidate
    integer :: i,j,k,g,base
    logical :: ok,all_ok
    call require(all(shape(values)==[extent,136]),'volume packed shape')
    if(writing.and.use_gpu) then
#ifdef _CUDA
      call pack_air5_statistics_gpu(values,window_saved,clock)
#endif
      return
    endif
    if(.not.use_gpu) call allocate_cpu()
    all_ok=.true.
    do g=1,4
      base=34*(g-1)
      do k=1,extent(3)
      do j=1,extent(2)
      do i=1,extent(1)
        if(writing) then
          call pack_velocity_state(states(i,j,k,g),values(i,j,k,base+1:base+34),ok)
        else
          call unpack_velocity_state(values(i,j,k,base+1:base+34),candidate,clock, &
            window_saved(1),window_saved(2),ok)
          ok=ok.and.values(i,j,k,base+3)==clock.and.values(i,j,k,base+34)==1.d0
          if(g>1) ok=ok.and.values(i,j,k,base+4)==1.d0
          if(ok.and..not.use_gpu) states(i,j,k,g)=candidate
        endif
        all_ok=all_ok.and.ok
      enddo
      enddo
      enddo
    enddo
    call require(all_ok,'invalid volume packed statistics state')
#ifdef _CUDA
    if(.not.writing.and.use_gpu) call restore_air5_statistics_gpu(values,weights)
#endif
  end subroutine

  subroutine air5_statistics_file(path,writing,identity,budget,window,reduction)
    use commvar, only: im,jm,km,ia,ja,ka,use_gpu
    use parallel, only: ig0,jg0,kg0
    use checkpoint_state_io, only: checkpoint_state_identity,checkpoint_state_transfer,allocate_checkpoint_buffer
    character(*),intent(in) :: path
    logical,intent(in) :: writing
    logical,intent(in) :: reduction
    type(checkpoint_state_identity),intent(in) :: identity
    integer(int64),intent(in) :: budget
    real(real64),intent(in) :: window(2)
    real(real64),allocatable :: buffer(:,:,:,:),values(:,:,:,:)
    real(real64) :: regional_values(34)
    type(checkpoint_state_identity) :: stored
    integer(int64) :: remaining,metadata(44),expected(44)
    integer :: g,status
    logical :: ok
    call initialize(window,reduction)
    call require(budget>=16_int64*(im+1)*(jm+1)*(km+1)*136+4096,'volume checkpoint buffer budget')
    call allocate_checkpoint_buffer([im,jm,km],0,136,budget,buffer,remaining,MPI_COMM_WORLD)
    allocate(values(extent(1),extent(2),extent(3),136),stat=status)
    call require(status==0,'cannot allocate volume state packing')
    expected=0
    expected(1:8)=[1_int64,identity%step,transfer(identity%time,0_int64), &
      transfer(window(1),0_int64),transfer(window(2),0_int64),merge(1_int64,0_int64,use_gpu),samples, &
      transfer(covered_duration,0_int64)]
    expected(9)=transfer(volume_saved,0_int64)
    expected(10)=merge(1_int64,0_int64,reduction_saved)
    if(writing) then
      call require(last_step==identity%step.and.previous_time==identity%time,'volume/checkpoint phase mismatch')
      call pack_state(values,.true.,identity%time)
      call pack_velocity_state(regional,regional_values,ok)
      call require(ok,'cannot pack volume regional statistics')
      expected(11:44)=transfer(regional_values,expected(11:44))
    endif
    metadata=expected; buffer=0.d0
    if(writing) buffer(1:extent(1),1:extent(2),1:extent(3),:)=values
    stored=identity
    call checkpoint_state_transfer(path,writing,.true.,[ia,ja,ka]+1,[ig0,jg0,kg0],[im,jm,km],0, &
      buffer,stored,remaining,MPI_COMM_WORLD,role=10,metadata=metadata,group_name='air5_volume_statistics')
    if(writing) return
    call require(all(metadata(1:6)==expected(1:6)).and.metadata(7)>0.and.metadata(10)==expected(10), &
      'volume statistics metadata mismatch')
    call require(stored%step==identity%step.and.stored%time==identity%time.and. &
      stored%dt_used==identity%dt_used.and.stored%dt_next==identity%dt_next,'volume checkpoint clock mismatch')
    covered_duration=transfer(metadata(8),covered_duration); volume_saved=transfer(metadata(9),volume_saved)
    call require(all(ieee_is_finite([covered_duration,volume_saved])).and.covered_duration>=0.d0.and. &
      volume_saved>0.d0,'invalid volume coverage or physical measure')
    values=buffer(1:extent(1),1:extent(2),1:extent(3),:)
    buffer(1:extent(1),1:extent(2),1:extent(3),:)=0.d0
    call require(all(buffer==0.d0),'nonzero off-owned volume statistics state')
    do g=1,4
      call require(all(values(:,:,:,34*(g-1)+8)==covered_duration),'volume moment coverage mismatch')
    enddo
    call pack_state(values,.false.,identity%time)
    regional_values=transfer(metadata(11:44),regional_values)
    call unpack_velocity_state(regional_values,regional,identity%time,window(1),window(2),ok)
    if(reduction_saved) ok=ok.and.regional_values(3)==identity%time.and.regional_values(8)==covered_duration
    if(.not.reduction_saved) ok=ok.and.regional_values(34)==0.d0.and.regional_values(8)==0.d0
    call require(ok,'invalid regional volume statistics')
    samples=metadata(7); last_step=int(identity%step); previous_time=identity%time
  end subroutine

  subroutine finish_air5_statistics(prefix)
    use commarray, only: x
    use parallel, only: mpirank,ig0,jg0,kg0
#ifdef _CUDA
    use insitu_air5_statistics_gpu, only: release_air5_statistics_gpu
#endif
    character(*),intent(in) :: prefix
    real(real64),allocatable :: packed(:,:,:,:),moments(:,:,:,:),coordinates(:,:,:,:)
    real(real64) :: regional_values(34)
    type(velocity_statistics) :: candidate
    type(velocity_statistics_result) :: result
    character(1200) :: filename
    integer :: i,j,k,g,a,base,field,unit,status,closed
    logical :: ok,all_ok
    if(samples==0) return
    allocate(packed(extent(1),extent(2),extent(3),136),moments(extent(1),extent(2),extent(3),62), &
      coordinates(extent(1),extent(2),extent(3),3),stat=status)
    call require(status==0,'volume final export allocation')
    call pack_state(packed,.true.,previous_time)
    coordinates=x(0:extent(1)-1,0:extent(2)-1,0:extent(3)-1,:)
    moments=0.d0; all_ok=.true.
    do k=1,extent(3)
    do j=1,extent(2)
    do i=1,extent(1)
      do g=1,4
        base=34*(g-1)
        call unpack_velocity_state(packed(i,j,k,base+1:base+34),candidate,previous_time, &
          window_saved(1),window_saved(2),ok)
        all_ok=all_ok.and.ok
        call read_velocity_statistics(candidate,result,ok)
        if(.not.ok) cycle
        if(g==1) then
          moments(i,j,k,1:41)=[result%duration,result%mean_density,result%mean_r,result%mean_f, &
            result%rms_r,result%rms_f,reshape(result%covariance_r,[9]),reshape(result%covariance_f,[9]), &
            reshape(result%density_stress,[9])]
        else
          do a=1,min(3,7-3*(g-2))
            field=3*(g-2)+a
            moments(i,j,k,41+field)=result%mean_r(a)
            moments(i,j,k,48+field)=result%covariance_r(a,a)
            moments(i,j,k,55+field)=result%rms_r(a)
          enddo
        endif
      enddo
    enddo
    enddo
    enddo
    call require(all_ok.and.all(ieee_is_finite(moments)),'volume final moments')
    call pack_velocity_state(regional,regional_values,ok)
    call require(ok,'volume final regional state')
    write(filename,'(A,".air5_statistics.step",I8.8,".rank",I8.8,".bin")') trim(prefix),last_step,mpirank
    open(newunit=unit,file=trim(filename),status='new',access='stream',form='unformatted',action='write', &
      convert='little_endian',iostat=status)
    call require(status==0,'cannot create volume statistics export')
    write(unit,iostat=status) 'ASTRAS01',int([1,last_step,mpirank,extent,ig0,jg0,kg0,merge(1,0,reduction_saved)],int32), &
      [previous_time,window_saved,covered_duration,volume_saved],samples,regional_values, &
      coordinates,weights,moments
    close(unit,iostat=closed)
    call require(status==0.and.closed==0,'cannot write volume statistics export')
    if(reduction_saved) call write_regional_summary(prefix,moments)
    if(allocated(states)) deallocate(states)
    deallocate(weights)
#ifdef _CUDA
    call release_air5_statistics_gpu()
#endif
    samples=0; last_step=-1; extent=0; covered_duration=0.d0
  end subroutine

  subroutine write_regional_summary(prefix,moments)
    use parallel, only: mpirank
    character(*),intent(in) :: prefix
    real(real64),intent(in) :: moments(:,:,:,:)
    type(velocity_statistics_result) :: result
    real(real64) :: local(3),averaged(3),signal_variance(3)
    character(1200) :: filename
    character(1),parameter :: components(3)=['u','v','w']
    integer :: a,ierr,unit,status,closed
    logical :: ok
    do a=1,3
      local(a)=sum(weights*moments(:,:,:,14+a+3*(a-1)))
    enddo
    call MPI_Allreduce(local,averaged,3,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr)
    call require(ierr==MPI_SUCCESS,'local temporal variance volume reduction')
    averaged=averaged/volume_saved
    call read_velocity_statistics(regional,result,ok)
    if(covered_duration==0.d0) then
      signal_variance=0.d0
      ok=.true.
    else
      signal_variance=[result%covariance_r(1,1),result%covariance_r(2,2),result%covariance_r(3,3)]
    endif
    call require(ok.and.all(ieee_is_finite([averaged,signal_variance])).and. &
      minval([averaged,signal_variance])>=0.d0,'invalid two-class volume RMS')
    status=0; closed=0
    if(mpirank==0) then
      write(filename,'(A,".air5_volume_rms.step",I8.8,".csv")') trim(prefix),last_step
      open(newunit=unit,file=trim(filename),status='new',action='write',iostat=status)
      if(status==0) then
        write(unit,'(A)',iostat=status) '# ASTR_AIR5_VOLUME_RMS 1; region=whole_physical_domain; average=Reynolds'
        if(status==0) write(unit,'(A,I0,5(1X,ES26.17))',iostat=status) '# step/time/window/duration/volume ', &
          last_step,previous_time,window_saved,covered_duration,volume_saved
        if(status==0) write(unit,'(A)',iostat=status) &
          'component,local_variance_volume_mean,local_variance_volume_rms,regional_signal_variance,regional_signal_rms'
        do a=1,3
          if(status/=0) exit
          write(unit,'(A,4(",",ES26.17))',iostat=status) components(a),averaged(a),sqrt(averaged(a)), &
            signal_variance(a),sqrt(signal_variance(a))
        enddo
        close(unit,iostat=closed)
      endif
    endif
    call require(status==0.and.closed==0,'cannot publish two-class volume RMS')
  end subroutine
end module
