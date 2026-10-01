program checkpoint_state_probe
  use iso_fortran_env, only: int64, real64
  use mpi
  use checkpoint_state_io
  use insitu_velocity_statistics
  implicit none
  integer :: err,rank,np,axis,ncomp,halo,origin(3),cells(3),owned(3),i,j,k,m,bad,total_bad
  integer, parameter :: global_shape(3)=[9,7,5]
  integer(int64), parameter :: budget=2_int64*1024*1024
  real(real64), allocatable :: q(:,:,:,:)
  real(real64), allocatable :: carry(:,:,:,:)
  real(real64) :: expected
  character(1024) :: path,arg,mode
  logical :: writing,exact,is_owned,cache
  type(checkpoint_state_identity) :: identity
  call MPI_Init(err)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,err)
  call MPI_Comm_size(MPI_COMM_WORLD,np,err)
  call get_command_argument(1,path)
  call get_command_argument(2,mode)
  call get_command_argument(3,arg)
  read(arg,*) axis
  call get_command_argument(4,arg)
  read(arg,*) ncomp
  call get_command_argument(5,arg)
  read(arg,*) halo
  if(trim(mode)=='context_reject'.or.trim(mode)=='context_clear') then
    call checkpoint_state_context('test-batch','test-last-complete')
    if(trim(mode)=='context_clear') call checkpoint_state_context('','')
    call checkpoint_state_require(rank/=np-1,MPI_COMM_WORLD,'injected single-rank failure')
  endif
  if ((np/=1.and.np/=2).or.axis<1.or.axis>3.or. &
      (ncomp/=5.and.ncomp/=11.and.ncomp/=34).or.halo<0.or.halo>1) &
    call MPI_Abort(MPI_COMM_WORLD,2,err)
  writing=trim(mode)=='write'.or.trim(mode)=='cache_write'.or.trim(mode)=='air5_write'
  cache=index(trim(mode),'cache_')==1
  if (cache.and.ncomp/=11) call MPI_Abort(MPI_COMM_WORLD,2,err)
  exact=trim(mode)/='repartition'
  origin=0
  cells=global_shape-1
  cells(axis)=cells(axis)/np
  origin(axis)=rank*cells(axis)
  owned=cells
  where (origin+cells==global_shape-1) owned=owned+1
  ! Account for q plus worst-case extras and metadata, not just HDF5 payload.
  if (24_int64*product(int(cells+2*halo+1,int64))*ncomp+4096>budget) &
    call MPI_Abort(MPI_COMM_WORLD,3,err)
  allocate(q(cells(1)+2*halo+1,cells(2)+2*halo+1,cells(3)+2*halo+1,ncomp))
  if(index(trim(mode),'statistics_')==1) then
    call statistics_roundtrip()
    call MPI_Finalize(err)
    stop
  endif
  if(ncomp==34.and.index(trim(mode),'air5_')/=1) call MPI_Abort(MPI_COMM_WORLD,2,err)
  q=-huge(1.0_real64)
  identity=checkpoint_state_identity(2147483650_int64,0.375_real64,0.125_real64,0.0625_real64)
  if (writing) then
    do m=1,ncomp
      do k=1,size(q,3)
        do j=1,size(q,2)
          do i=1,size(q,1)
            q(i,j,k,m)=value(i,j,k,m,.true.)
          enddo
        enddo
      enddo
    enddo
  else
    identity=checkpoint_state_identity()
  endif
  if(index(trim(mode),'air5_')==1) then
    if(ncomp/=34) call MPI_Abort(MPI_COMM_WORLD,2,err)
    allocate(carry(cells(1)+1,cells(2)+1,cells(3)+1,11))
    carry=-huge(1.0_real64)
    if(writing) carry=q(halo+1:halo+cells(1)+1,halo+1:halo+cells(2)+1,halo+1:halo+cells(3)+1,12:22)
    call checkpoint_air5_state(trim(path),writing,global_shape,origin,cells,halo, &
      q(:,:,:,1:11),carry,trim(mode)/='air5_compensation_off',q(:,:,:,23),q(:,:,:,24:26), &
      q(:,:,:,27),q(:,:,:,28),q(:,:,:,29),q(:,:,:,30:34),identity,budget,MPI_COMM_WORLD)
    q(:,:,:,12:22)=0.0_real64
    q(halo+1:halo+cells(1)+1,halo+1:halo+cells(2)+1,halo+1:halo+cells(3)+1,12:22)=carry
  else if (cache) then
    call checkpoint_perfect_gas_state(trim(path),writing,global_shape,origin,cells,halo, &
      q(:,:,:,1:5),q(:,:,:,6),q(:,:,:,7:9),q(:,:,:,10),q(:,:,:,11), &
      identity,budget,MPI_COMM_WORLD)
  else
    call checkpoint_state_transfer(trim(path),writing,exact,global_shape,origin,cells,halo,q,identity, &
                                   budget,MPI_COMM_WORLD)
  endif
  bad=0
  if (identity%step/=2147483650_int64.or.identity%time/=0.375_real64.or. &
      identity%dt_used/=0.125_real64.or.identity%dt_next/=0.0625_real64) bad=bad+1
  do m=1,ncomp
    do k=1,size(q,3)
      do j=1,size(q,2)
        do i=1,size(q,1)
          expected=value(i,j,k,m,exact)
          if(index(trim(mode),'air5_')==1.and.m>=12.and.m<=22.and. &
            (any([i,j,k]<halo+1).or.any([i,j,k]>halo+cells+1))) expected=0.0_real64
          if (transfer(q(i,j,k,m),0_int64)/=transfer(expected,0_int64)) bad=bad+1
        enddo
      enddo
    enddo
  enddo
  call MPI_Allreduce(bad,total_bad,1,MPI_INTEGER,MPI_SUM,MPI_COMM_WORLD,err)
  if (total_bad/=0) then
    if (rank==0) print *, 'MISMATCHES ',total_bad
    call MPI_Abort(MPI_COMM_WORLD,4,err)
  endif
  if (rank==0) print *, 'PASS ',trim(mode),' NP=',np,' components=',ncomp,' halo=',halo
  call MPI_Finalize(err)
contains
  subroutine statistics_roundtrip()
    type(velocity_statistics) :: original,recovered
    real(real64) :: values(34),saved(34),continued(34),t
    integer :: role
    integer(int64) :: metadata(4),expected_metadata(4)
    logical :: ok
    if(ncomp/=34) call MPI_Abort(MPI_COMM_WORLD,2,err)
    writing=trim(mode)=='statistics_write'
    if(writing) then
      do k=1,size(q,3)
        do j=1,size(q,2)
          do i=1,size(q,1)
            call seed_statistics(original)
            call pack_velocity_state(original,values,ok)
            call checkpoint_state_require(ok,MPI_COMM_WORLD,'pack synthetic statistics')
            q(i,j,k,:)=values
          enddo
        enddo
      enddo
    endif
    identity=checkpoint_state_identity(1_int64,1.d0,1.d0,1.d0)
    role=3
    expected_metadata=[1_int64,transfer(-0.d0,0_int64),transfer(0.5d0,0_int64),2147483650_int64]
    metadata=expected_metadata
    if(.not.writing) metadata=0
    if(trim(mode)=='statistics_wrong_role') role=1
    call checkpoint_state_transfer(trim(path),writing,.true.,global_shape,origin,cells,halo, &
      q,identity,budget,MPI_COMM_WORLD,role=role,metadata=metadata)
    bad=0
    if(any(metadata/=expected_metadata)) bad=bad+1
    do k=1,size(q,3)
      do j=1,size(q,2)
        do i=1,size(q,1)
          call seed_statistics(original)
          call pack_velocity_state(original,values,ok)
          if(.not.ok) bad=bad+1
          saved=q(i,j,k,:)
          if(any(transfer(saved,[0_int64],34)/=transfer(values,[0_int64],34))) bad=bad+1
          call unpack_velocity_state(saved,recovered,1.d0,0.d0,3.d0,ok)
          if(.not.ok) bad=bad+1
          t=real(i+j+k+rank,real64)/8
          call push_velocity_sample(original,2.d0,2.d0+t,[t,-t,2*t],ok)
          if(.not.ok) bad=bad+1
          call push_velocity_sample(recovered,2.d0,2.d0+t,[t,-t,2*t],ok)
          if(.not.ok) bad=bad+1
          call pack_velocity_state(original,continued,ok)
          if(.not.ok) bad=bad+1
          call pack_velocity_state(recovered,values,ok)
          if(.not.ok) bad=bad+1
          if(any(transfer(continued,[0_int64],34)/=transfer(values,[0_int64],34))) bad=bad+1
        enddo
      enddo
    enddo
    call MPI_Allreduce(bad,total_bad,1,MPI_INTEGER,MPI_SUM,MPI_COMM_WORLD,err)
    call checkpoint_state_require(total_bad==0,MPI_COMM_WORLD,'statistics exact continuation')
    if(rank==0) print *, 'PASS statistics storage/continuation NP=',np
  end subroutine

  subroutine seed_statistics(state)
    type(velocity_statistics),intent(out) :: state
    real(real64) :: v
    logical :: ok
    v=real(i+2*j+3*k+rank,real64)/8
    call configure_velocity_statistics(state,0.d0,3.d0,ok)
    if(.not.ok) error stop 'configure synthetic statistics'
    call push_velocity_sample(state,0.d0,1.d0+v,[-0.d0,v,-v],ok)
    if(.not.ok) error stop 'initial synthetic statistics'
    call push_velocity_sample(state,1.d0,2.d0+v,[v,-v,2*v],ok)
    if(.not.ok) error stop 'second synthetic statistics'
  end subroutine

  real(real64) function value(i,j,k,m,original) result(v)
    integer,intent(in) :: i,j,k,m
    logical,intent(in) :: original
    integer :: xyz(3)
    xyz=[i,j,k]-halo-1
    is_owned=all(xyz>=0).and.all(xyz<owned)
    if (original.and..not.is_owned) then
      v=-real(1000000*rank+100000*m+10000*k+100*j+i,real64)/8
    else if (.not.original.and.(any(xyz<0).or.any(xyz>cells))) then
      v=-huge(1.0_real64)
    else
      xyz=xyz+origin
      v=real(10000*m+1000*xyz(3)+100*xyz(2)+xyz(1),real64)/8
      if (m==1.and.all(xyz==0)) v=-0.0_real64
    endif
  end function
end program
