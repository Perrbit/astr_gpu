program insitu_moments_probe
  use iso_c_binding
  use mpi
  use ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  use device_state
  use insitu_moments_cpp
  use insitu_velocity_statistics
  use insitu_time_integral, only: clipped_trapezoid
  implicit none
  integer,parameter :: global(3)=[16,12,8],hm=5,nt=6
  real(c_double),parameter :: times(nt)=[0.d0,.001d0,.0017d0,.003d0,.0049d0,.006d0]
  real(c_double),parameter :: constant_velocity(3)=[.25d0,-.5d0,.125d0]
  type(c_ptr) :: parent,plan,resumed,borrowed
  type(device_view) :: view
  type(velocity_statistics),allocatable :: oracle(:,:,:)
  type(velocity_statistics_result) :: answer
  real(c_double),allocatable :: q(:,:,:,:),saved_q(:,:,:,:),packed(:,:,:,:),cpu(:,:,:,:), &
    resumed_state(:,:,:,:),candidate(:,:,:,:),values(:,:,:,:),expected(:,:,:,:),measure(:,:,:)
  real(c_double) :: window(2),weights(2),duration,info(5),rho,u(3),r(41),node, &
    state_error=0,result_error=0,rms_error=0,spatial_error=0,local_error,previous_duration,local(7),spatial_expected(7),again(7)
  integer(c_int64_t) :: usage(6),before(6),owned_bytes
  integer :: rank,ranks,local_rank,status,ierr,dims(3),coords(3),extent(3),owned(3),origin(3),cart, &
    debug,fixture,physical,frame,i,j,k,a,checks=0,restores=0
  logical :: ok
  character(64) :: argument
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',argument,status=status)
  if(status/=0) error stop 'Open MPI local device identity required'
  read(argument,*) local_rank
  call check(device_bind(int(local_rank,c_int)),'bind')
  call MPI_Init(ierr);call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  call get_command_argument(1,argument);read(argument,*) dims
  if(product(dims)/=ranks.or.any(dims<1).or.any(mod(global,dims)/=0)) error stop 'moments topology mismatch'
  call MPI_Cart_create(MPI_COMM_WORLD,3,dims,[.false.,.false.,.false.],.false.,cart,ierr)
  call MPI_Cart_coords(cart,rank,3,coords,ierr)
  extent=global/dims;origin=coords*extent
  do debug=0,1
  do physical=0,1
    owned=extent
    if(physical==1) where(coords==dims-1) owned=owned+1
    allocate(q(-hm:extent(1)+hm,-hm:extent(2)+hm,-hm:extent(3)+hm,5), &
      saved_q(-hm:extent(1)+hm,-hm:extent(2)+hm,-hm:extent(3)+hm,5), &
      oracle(0:owned(1)-1,0:owned(2)-1,0:owned(3)-1), &
      packed(0:owned(1)-1,0:owned(2)-1,0:owned(3)-1,34), &
      cpu(0:owned(1)-1,0:owned(2)-1,0:owned(3)-1,34), &
      resumed_state(0:owned(1)-1,0:owned(2)-1,0:owned(3)-1,34), &
      candidate(0:owned(1)-1,0:owned(2)-1,0:owned(3)-1,34), &
      values(0:owned(1)-1,0:owned(2)-1,0:owned(3)-1,41), &
      expected(0:owned(1)-1,0:owned(2)-1,0:owned(3)-1,41),measure(0:owned(1)-1,0:owned(2)-1,0:owned(3)-1))
    owned_bytes=30_c_int64_t*product(int(owned,c_int64_t))*8+4
    do fixture=1,3
      window=[.0004d0,.0042d0]
      if(fixture==2) window=[1.d0,2.d0]
      call check(device_state_create(int(local_rank,c_int),int(extent(1),c_int),int(extent(2),c_int), &
        int(extent(3),c_int),hm,int(debug,c_int),parent),'create parent')
      call check(moments_create(parent,int(owned,c_int),window,plan),'create moments')
      status=moments_view(plan,view)
      if(status/=device_not_allocated) error stop 'uninitialized moments exposed'
      call check(moments_metrics(plan,usage),'initial bytes')
      if(usage(1)/=owned_bytes.or.any(usage(3:6)/=0)) error stop 'unexpected eager output or transfer'
      borrowed=parent
      status=device_state_destroy(parent)
      if(status/=device_invalid_argument.or..not.c_associated(borrowed,parent)) error stop 'live moments parent destroyed'
      do k=0,owned(3)-1
      do j=0,owned(2)-1
      do i=0,owned(1)-1
        call configure_velocity_statistics(oracle(i,j,k),window(1),window(2),ok)
        if(.not.ok) error stop 'CPU configure'
      enddo
      enddo
      enddo
      resumed=c_null_ptr;previous_duration=0
      do frame=1,nt
        q=ieee_value(0.d0,ieee_quiet_nan)
        do k=0,owned(3)-1
        do j=0,owned(2)-1
        do i=0,owned(1)-1
          node=real(origin(1)+i+global(1)*(origin(2)+j+global(2)*(origin(3)+k)),c_double)
          rho=1+node/4096+times(frame)/16
          u=[node/2048+times(frame),-node/8192+2*times(frame),node/4096-3*times(frame)]
          if(fixture==3) then
            rho=2;u=constant_velocity
          endif
          q(i,j,k,1)=rho;q(i,j,k,2:4)=rho*u;q(i,j,k,5)=99
          call push_velocity_sample(oracle(i,j,k),times(frame),rho,q(i,j,k,2:4)/rho,ok)
          if(.not.ok) error stop 'CPU push'
          call pack_velocity_state(oracle(i,j,k),cpu(i,j,k,:),ok)
          if(.not.ok) error stop 'CPU pack'
        enddo
        enddo
        enddo
        call check(device_state_upload(parent,device_q,q,nbytes(size(q))),'explicit fixture input')
        weights=0
        if(frame>1) then
          call clipped_trapezoid(times(frame-1),times(frame),[1.d0,0.d0],[0.d0,1.d0], &
            window(1),window(2),weights,duration,ok)
          if(.not.ok) error stop 'host clipped quadrature'
        endif
        call check(moments_metrics(plan,before),'pre-update transfers')
        call check(moments_push(plan,times(frame),weights),'push moments')
        call check(moments_metrics(plan,usage),'post-update transfers')
        if(usage(3)/=4*frame.or.usage(4)/=before(4).or.usage(5)/=0.or.usage(6)/=frame) &
          error stop 'sample performed non-scalar transfer'
        if(usage(1)/=owned_bytes) error stop 'sample retained an unnecessary output array'
        call check(moments_info(plan,info),'clock')
        previous_duration=previous_duration+weights(1);previous_duration=previous_duration+weights(2)
        if(info(1)/=times(frame).or.any(info(2:3)/=window).or.info(4)/=previous_duration.or.info(5)/=1) &
          error stop 'sample identity or duration differs'
        call check(moments_view(plan,view),'resident state view')
        if(view%rank/=2.or.view%extent(1)/=product(owned).or.view%extent(2)/=30) error stop 'state view layout'
        call check(moments_pack(plan,packed,nbytes(size(packed))),'explicit state observation')
        state_error=max(state_error,maxval(abs(packed-cpu))/max(1.d0,maxval(abs(cpu))))
        if(state_error>2.d-10) error stop 'CPU/device moment state differs'
        if(frame==3) then
          call check(moments_create(parent,int(owned,c_int),window,resumed),'create resume context')
          call check(moments_restore(resumed,packed,nbytes(size(packed)),times(frame)),'restore checkpoint payload')
          restores=restores+1
        else if(frame>3) then
          call check(moments_push(resumed,times(frame),weights),'continued moments')
          call check(moments_pack(resumed,resumed_state,nbytes(size(resumed_state))),'resumed payload')
          if(any(transfer(packed,[0_c_int64_t],size(packed))/= &
            transfer(resumed_state,[0_c_int64_t],size(resumed_state)))) error stop 'resume state bits differ'
        endif
        call check(device_state_download(parent,device_q,saved_q,nbytes(size(q))),'solver isolation observation')
        if(any(transfer(q,[0_c_int64_t],size(q))/=transfer(saved_q,[0_c_int64_t],size(q)))) &
          error stop 'statistics changed q or its unused halo'
        checks=checks+1
      enddo
      call check(moments_output(plan,values,nbytes(size(values))),'explicit derived output')
      expected=0
      do k=0,owned(3)-1
      do j=0,owned(2)-1
      do i=0,owned(1)-1
        call read_velocity_statistics(oracle(i,j,k),answer,ok)
        if(fixture==2) then
          if(ok) error stop 'uncovered window reported a mean'
          cycle
        endif
        if(.not.ok) error stop 'covered window absent'
        r=[answer%duration,answer%mean_density,answer%mean_r,answer%mean_f,answer%rms_r,answer%rms_f, &
          reshape(answer%covariance_r,[9]),reshape(answer%covariance_f,[9]),reshape(answer%density_stress,[9])]
        expected(i,j,k,:)=r
      enddo
      enddo
      enddo
      rms_error=max(rms_error,maxval(abs(values(:,:,:,9:14)-expected(:,:,:,9:14))))
      values(:,:,:,9:14)=values(:,:,:,9:14)**2;expected(:,:,:,9:14)=expected(:,:,:,9:14)**2
      local_error=maxval(abs(values-expected))/max(1.d0,maxval(abs(expected)))
      result_error=max(result_error,local_error)
      if(result_error>2.d-10) error stop 'means/covariance/squared RMS/stress differ'
      if(fixture==3) then
        if(any(values(:,:,:,9:41)/=0)) error stop 'constant velocity manufactured variance is nonzero'
        if(maxval(abs(values(:,:,:,1)-.0038d0))>2.d-18) error stop 'clipped duration analytic mismatch'
        do a=1,3
          if(any(values(:,:,:,a+2)/=constant_velocity(a))) error stop 'constant manufactured mean mismatch'
        enddo
      endif
      candidate=packed;candidate(0,0,0,16)=-1
      status=moments_restore(plan,candidate,nbytes(size(candidate)),times(nt))
      if(status/=device_nonphysical) error stop 'negative diagonal restored'
      status=moments_push(plan,times(nt),[0.d0,0.d0])
      if(status/=device_invalid_argument) error stop 'repeated sample time accepted'
      call check(moments_pack(plan,candidate,nbytes(size(candidate))),'rejection isolation')
      if(any(transfer(packed,[0_c_int64_t],size(packed))/=transfer(candidate,[0_c_int64_t],size(candidate)))) &
        error stop 'pre-update rejection modified statistics'
      call check(moments_metrics(plan,usage),'output allocation released')
      if(usage(1)/=owned_bytes.or.usage(2)/=owned_bytes+41_c_int64_t*product(int(owned,c_int64_t))*8) &
        error stop 'output buffer lifetime or peak differs'
      status=moments_spatial(plan,local)
      if(status/=device_not_allocated) error stop 'unrequested spatial weights were allocated'
      do k=0,owned(3)-1
      do j=0,owned(2)-1
      do i=0,owned(1)-1
        measure(i,j,k)=.25d0+real(i+origin(1)+j+origin(2)+k+origin(3),c_double)/1024
      enddo
      enddo
      enddo
      spatial_expected(1)=sum(measure)
      do a=1,3
        spatial_expected(1+a)=sum(measure*cpu(:,:,:,a+4))
        spatial_expected(4+a)=sum(measure*expected(:,:,:,a+8))
      enddo
      call check(moments_metrics(plan,before),'pre-spatial bytes')
      call check(moments_weights(plan,measure,nbytes(size(measure))),'static node measures')
      call check(moments_spatial(plan,local),'scalar spatial reduction')
      call check(moments_spatial(plan,again),'repeated scalar spatial reduction')
      if(any(transfer(local,[0_c_int64_t],7)/=transfer(again,[0_c_int64_t],7))) &
        error stop 'repeated spatial reduction bits differ'
      spatial_error=max(spatial_error,maxval(abs(local-spatial_expected))/max(1.d0,maxval(abs(spatial_expected))))
      if(spatial_error>2.d-10) error stop 'weighted instantaneous velocity or temporal variance differs'
      call check(moments_metrics(plan,usage),'spatial resource counters')
      if(usage(1)/=owned_bytes+8_c_int64_t*size(measure)+48.or.usage(3)/=before(3)+96.or. &
        usage(4)/=before(4).or.usage(5)/=before(5)+8_c_int64_t*size(measure)) &
        error stop 'spatial reduction downloaded a field or allocated a hidden field'
      q(0,0,0,1)=0
      call check(device_state_upload(parent,device_q,q,nbytes(size(q))),'invalid source fixture')
      status=moments_push(plan,times(nt)+.001d0,[0.d0,0.d0])
      if(status/=device_nonphysical) error stop 'nonpositive sample density accepted'
      ! A failed update is not reused. No rollback/retry is part of this provider.
      call check(moments_destroy(resumed),'destroy resumed')
      call check(moments_destroy(plan),'destroy moments')
      call check(device_state_destroy(parent),'destroy parent')
    enddo
    deallocate(q,saved_q,oracle,packed,cpu,resumed_state,candidate,values,expected,measure)
  enddo
  enddo
  write(*,'(a,3(1x,i0),4(1x,es24.16e3))') &
    'CPP_INSITU_MOMENTS_PASS',rank,checks,restores,state_error,result_error,rms_error,spatial_error
  call MPI_Comm_free(cart,ierr);call MPI_Finalize(ierr)
contains
  integer(c_size_t) function nbytes(n)
    integer,intent(in) :: n
    nbytes=int(n,c_size_t)*8
  end function
  subroutine check(code,where)
    integer(c_int),intent(in) :: code
    character(*),intent(in) :: where
    integer :: ignored
    if(code==0) return
    print *, 'moments failure: ',where,code
    call MPI_Abort(MPI_COMM_WORLD,int(code),ignored)
  end subroutine
end program
