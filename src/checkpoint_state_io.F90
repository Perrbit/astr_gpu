module checkpoint_state_io
  use iso_fortran_env, only: int64, real64, error_unit
  use ieee_arithmetic, only: ieee_is_finite
  use mpi
  use hdf5
  implicit none
  private
  public :: checkpoint_state_transfer, checkpoint_state_identity
  public :: checkpoint_perfect_gas_state, allocate_checkpoint_buffer, checkpoint_state_require
  public :: checkpoint_air5_state
  public :: checkpoint_state_context
  public :: checkpoint_halo_refresh
  public :: checkpoint_state_header
  public :: checkpoint_crashfix_state,checkpoint_crashfix_words

  abstract interface
    subroutine checkpoint_halo_refresh(buffer,cells,halo,budget,comm)
      import :: real64,int64
      real(real64),contiguous,intent(inout) :: buffer(:,:,:,:)
      integer,intent(in) :: cells(3),halo,comm
      integer(int64),intent(in) :: budget
    end subroutine
  end interface

  character(1200),save :: failure_batch='',failure_last_complete=''

  ! This is a state-file primitive, not a published restart bundle.
  integer(int64), parameter :: magic=int(z'4153545251533031',int64), schema=2
  type checkpoint_state_identity
    integer(int64) :: step=0
    real(real64) :: time=0, dt_used=0, dt_next=0
  end type
contains
  subroutine checkpoint_crashfix_words(path,writing,mode,active,nodes,budget,comm,words,restored_exact,allow_inexact)
    character(*),intent(in) :: path
    logical,intent(in) :: writing,mode,restored_exact
    logical,optional,intent(in) :: allow_inexact
    logical,intent(inout) :: active
    integer(int64),intent(in) :: nodes,budget
    integer,intent(in) :: comm
    integer(int64),allocatable,intent(inout) :: words(:)
    integer(int64) :: header(4),nwords,local(2),offset,total
    integer(int64),allocatable :: counts(:),expected(:)
    integer(hid_t) :: file,access,xfer,dset,space,mem,native
    integer(hsize_t) :: dims(1),start(1),count(1),mdims(1)
    integer :: rank,np,err,mpi_i64,r
    logical :: exists,inexact
    inexact=.false.
    if(present(allow_inexact)) inexact=allow_inexact
    if(writing.and..not.active) return
    call MPI_Comm_rank(comm,rank,err)
    call MPI_Comm_size(comm,np,err)
    call MPI_Type_match_size(MPI_TYPECLASS_INTEGER,8,mpi_i64,err)
    call require(err==MPI_SUCCESS,comm,'crashfix integer type')
    call h5open_f(err)
    call require(err==0,comm,'crashfix HDF5 initialize')
    call h5pcreate_f(H5P_FILE_ACCESS_F,access,err)
    call require(err==0,comm,'crashfix file access')
    call h5pset_fapl_mpio_f(access,comm,MPI_INFO_NULL,err)
    call require(err==0,comm,'crashfix parallel file access')
    if(writing) then
      call h5fopen_f(path,H5F_ACC_RDWR_F,file,err,access_prp=access)
    else
      call h5fopen_f(path,H5F_ACC_RDONLY_F,file,err,access_prp=access)
    endif
    call require(err==0,comm,'open crashfix state')
    call h5pclose_f(access,err)
    call require(err==0,comm,'close crashfix file access')
    if(.not.writing) then
      call h5lexists_f(file,'crashfix_layout',exists,err)
      call require(err==0,comm,'read crashfix presence')
      active=exists
      if(.not.active) then
        call require(.not.mode.or.inexact,comm,'checkpoint lacks required crashfix history')
        call h5fclose_f(file,err)
        call require(err==0,comm,'close state without crashfix history')
        return
      endif
    endif
    call require(writing.or.restored_exact,comm,'crashfix history requires the same MPI topology')
    nwords=(nodes+63_int64)/64_int64
    call require(nodes>0.and.nwords<=huge(1).and. &
      nwords<=budget/8-int(np,int64)*4-4,comm,'crashfix packing budget')
    allocate(counts(2*np),expected(2*np),stat=err)
    call require(err==0,comm,'crashfix partition allocation')
    local=[nodes,nwords]
    call MPI_Allgather(local,2,mpi_i64,expected,2,mpi_i64,comm,err)
    call require(err==MPI_SUCCESS,comm,'crashfix partition gather')
    counts=expected
    header=[1_int64,int(merge(1,0,mode),int64),int(np,int64),sum(expected(2::2))]
    call h5pcreate_f(H5P_DATASET_XFER_F,xfer,err)
    call require(err==0,comm,'crashfix transfer property')
    call h5pset_dxpl_mpio_f(xfer,H5FD_MPIO_COLLECTIVE_F,err)
    call require(err==0,comm,'crashfix collective transfer')
    call integers(file,'crashfix_layout',header,writing,xfer,comm,rank)
    call require(header(1)==1.and.(header(2)==merge(1,0,mode).or.inexact).and.header(3)==np.and. &
      header(4)==sum(expected(2::2)),comm,'crashfix mode/layout mismatch')
    call integers(file,'crashfix_partitions',counts,writing,xfer,comm,rank)
    call require(all(counts==expected),comm,'crashfix partition mismatch')
    if(writing) then
      call require(allocated(words),comm,'missing packed crashfix history')
      call require(size(words,kind=int64)==nwords,comm,'crashfix word count')
    else
      allocate(words(int(nwords)),stat=err)
      call require(err==0,comm,'crashfix read buffer allocation')
    endif
    offset=0
    do r=1,rank
      offset=offset+counts(2*r)
    enddo
    total=header(4)
    dims=total; mdims=nwords; start=offset; count=nwords
    call dataset(file,'crashfix_history',dims,H5T_STD_I64LE,writing,comm,dset,space)
    call h5screate_simple_f(1,mdims,mem,err)
    call require(err==0,comm,'crashfix memory space')
    call h5sselect_hyperslab_f(space,H5S_SELECT_SET_F,start,count,err)
    call require(err==0,comm,'crashfix file selection')
    native=h5kind_to_type(int64,H5_INTEGER_KIND)
    if(writing) then
      call h5dwrite_f(dset,native,words,mdims,err,mem_space_id=mem,file_space_id=space,xfer_prp=xfer)
    else
      call h5dread_f(dset,native,words,mdims,err,mem_space_id=mem,file_space_id=space,xfer_prp=xfer)
    endif
    call require(err==0,comm,'crashfix history transfer')
    call close_dataset(dset,space,mem,comm)
    call h5pclose_f(xfer,err)
    call require(err==0,comm,'close crashfix transfer property')
    call h5fclose_f(file,err)
    call require(err==0,comm,'close crashfix state')
  end subroutine checkpoint_crashfix_words

  subroutine checkpoint_crashfix_state(path,writing,mask,mode,budget,comm,restored_exact,allow_inexact)
    character(*),intent(in) :: path
    logical,intent(in) :: writing,mode,restored_exact
    logical,optional,intent(in) :: allow_inexact
    logical,contiguous,intent(inout) :: mask(:,:,:)
    integer(int64),intent(in) :: budget
    integer,intent(in) :: comm
    integer(int64),allocatable :: words(:)
    integer(int64) :: nodes,id
    integer :: i,j,k,err
    logical :: active
    nodes=size(mask,kind=int64)
    active=mode.or.any(mask)
    call MPI_Allreduce(MPI_IN_PLACE,active,1,MPI_LOGICAL,MPI_LOR,comm,err)
    call require(err==MPI_SUCCESS,comm,'crashfix history activation')
    if(writing.and.active) then
      call require((nodes+63)/64<=budget/8,comm,'crashfix pack buffer budget')
      allocate(words(int((nodes+63)/64)),stat=err)
      call require(err==0,comm,'crashfix pack buffer allocation')
      words=0; id=0
      do k=1,size(mask,3)
      do j=1,size(mask,2)
      do i=1,size(mask,1)
        if(mask(i,j,k)) words(id/64+1)=ibset(words(id/64+1),int(mod(id,64_int64)))
        id=id+1
      enddo
      enddo
      enddo
    endif
    call checkpoint_crashfix_words(path,writing,mode,active,nodes,budget,comm,words,restored_exact,allow_inexact)
    if(.not.writing) then
      mask=.false.
      if(active) then
        id=0
        do k=1,size(mask,3)
        do j=1,size(mask,2)
        do i=1,size(mask,1)
          mask(i,j,k)=btest(words(id/64+1),int(mod(id,64_int64)))
          id=id+1
        enddo
        enddo
        enddo
      endif
    endif
  end subroutine checkpoint_crashfix_state

  subroutine checkpoint_state_header(path,header,metadata,comm)
    character(*),intent(in) :: path
    integer(int64),intent(out) :: header(13)
    integer(int64),optional,intent(out) :: metadata(:)
    integer,intent(in) :: comm
    integer(hid_t) :: file,xfer
    integer :: err,rank
    call MPI_Comm_rank(comm,rank,err)
    call require(err==MPI_SUCCESS,comm,'header rank')
    call h5open_f(err)
    call require(err==0,comm,'header HDF5 initialize')
    call h5fopen_f(path,H5F_ACC_RDONLY_F,file,err)
    call require(err==0,comm,'open state header')
    call h5pcreate_f(H5P_DATASET_XFER_F,xfer,err)
    call require(err==0,comm,'header transfer property')
    call integers(file,'identity',header,.false.,xfer,comm,rank)
    call require(header(1)==magic.and.header(2)==schema.and.header(3)==1,comm,'header identity/version/phase')
    if(present(metadata)) call integers(file,'metadata',metadata,.false.,xfer,comm,rank)
    call h5pclose_f(xfer,err)
    call require(err==0,comm,'close header transfer property')
    call h5fclose_f(file,err)
    call require(err==0,comm,'close state header')
  end subroutine

  subroutine checkpoint_state_context(batch,last_complete)
    character(*),intent(in) :: batch,last_complete
    failure_batch=batch
    failure_last_complete=last_complete
  end subroutine

  subroutine checkpoint_state_require(condition,comm,label)
    logical,intent(in) :: condition
    integer,intent(in) :: comm
    character(*),intent(in) :: label
    call require(condition,comm,label)
  end subroutine

  subroutine allocate_checkpoint_buffer(cells,halo,components,budget,buffer,remaining,comm)
    integer,intent(in) :: cells(3),halo,components,comm
    integer(int64),intent(in) :: budget
    real(real64),allocatable,intent(out) :: buffer(:,:,:,:)
    integer(int64),intent(out) :: remaining
    integer(int64) :: extents(3),nodes,bytes
    integer :: err
    call require(halo>=0.and.all(cells>0).and.components>0.and.budget>0,comm,'invalid packing parameters')
    extents=int(cells,int64)+2_int64*halo+1
    call require(all(extents<=huge(halo)),comm,'packing extent overflow')
    nodes=volume(extents,comm)
    call require(nodes<=budget/8/components,comm,'packing buffer budget')
    bytes=nodes*components*8
    remaining=budget-bytes
    allocate(buffer(extents(1),extents(2),extents(3),components),stat=err)
    call require(err==0,comm,'packing buffer allocation')
  end subroutine

  subroutine checkpoint_perfect_gas_state(path,writing,global_shape,origin,cells,halo, &
                                         q,rho,vel,prs,tmp,identity,budget,comm,allow_repartition,restored_exact,refresh_halos, &
                                         repartition_axes)
    character(*),intent(in) :: path
    logical,intent(in) :: writing
    logical,optional,intent(in) :: allow_repartition
    logical,optional,intent(in) :: repartition_axes(3)
    logical,optional,intent(out) :: restored_exact
    procedure(checkpoint_halo_refresh),optional :: refresh_halos
    integer,intent(in) :: global_shape(3),origin(3),cells(3),halo,comm
    real(real64),contiguous,intent(inout) :: q(:,:,:,:),rho(:,:,:),vel(:,:,:,:), &
      prs(:,:,:),tmp(:,:,:)
    type(checkpoint_state_identity),intent(inout) :: identity
    integer(int64),intent(in) :: budget
    real(real64),allocatable :: buffer(:,:,:,:)
    integer(int64) :: remaining
    integer :: extents(3)
    logical :: permit,matched
    permit=.false.
    if(present(allow_repartition)) permit=allow_repartition
    call allocate_checkpoint_buffer(cells,halo,11,budget,buffer,remaining,comm)
    extents=shape(buffer(:,:,:,1))
    call require(all(shape(q)==[extents,5]).and.all(shape(rho)==extents).and. &
      all(shape(vel)==[extents,3]).and.all(shape(prs)==extents).and. &
      all(shape(tmp)==extents),comm,'perfect-gas cache shape')
    if (writing) then
      buffer(:,:,:,1:5)=q
      buffer(:,:,:,6)=rho
      buffer(:,:,:,7:9)=vel
      buffer(:,:,:,10)=prs
      buffer(:,:,:,11)=tmp
    else
      buffer=0.d0
    endif
    ! Role 2 preserves q and cached primitives independently, including halos.
    call checkpoint_state_transfer(path,writing,.true.,global_shape,origin,cells,halo, &
      buffer,identity,remaining,comm,role=2,allow_repartition=permit,restored_exact=matched, &
      repartition_axes=repartition_axes)
    if(present(restored_exact)) restored_exact=matched
    if (.not.writing) then
      if(.not.matched) then
        call require(present(refresh_halos),comm,'repartition halo provider missing')
        call refresh_halos(buffer,cells,halo,remaining,comm)
      endif
      q=buffer(:,:,:,1:5)
      rho=buffer(:,:,:,6)
      vel=buffer(:,:,:,7:9)
      prs=buffer(:,:,:,10)
      tmp=buffer(:,:,:,11)
    endif
  end subroutine

  subroutine checkpoint_air5_state(path,writing,global_shape,origin,cells,halo, &
                                  q,carry,compensated,rho,vel,prs,tmp,tve,spc,identity,budget,comm, &
                                  allow_repartition,restored_exact,refresh_halos,repartition_axes,step_history)
    character(*),intent(in) :: path
    logical,intent(in) :: writing,compensated
    logical,optional,intent(in) :: allow_repartition,repartition_axes(3)
    logical,optional,intent(out) :: restored_exact
    procedure(checkpoint_halo_refresh),optional :: refresh_halos
    integer,intent(in) :: global_shape(3),origin(3),cells(3),halo,comm
    real(real64),contiguous,intent(inout) :: q(:,:,:,:),rho(:,:,:),vel(:,:,:,:), &
      prs(:,:,:),tmp(:,:,:),tve(:,:,:),spc(:,:,:,:)
    real(real64),allocatable,intent(inout) :: carry(:,:,:,:)
    real(real64),allocatable,optional,intent(inout) :: step_history(:,:,:)
    type(checkpoint_state_identity),intent(inout) :: identity
    integer(int64),intent(in) :: budget
    real(real64),allocatable :: buffer(:,:,:,:)
    integer(int64) :: remaining,metadata(2),expected(2)
    integer :: extents(3),h
    logical :: permit,matched,has_history
    permit=.false.
    if(present(allow_repartition)) permit=allow_repartition
    has_history=.false.
    if(present(step_history)) has_history=allocated(step_history)
    call allocate_checkpoint_buffer(cells,halo,34+merge(1,0,has_history),budget,buffer,remaining,comm)
    extents=shape(buffer(:,:,:,1)); h=halo+1
    call require(all(shape(q)==[extents,11]).and.all(shape(rho)==extents).and. &
      all(shape(vel)==[extents,3]).and.all(shape(prs)==extents).and. &
      all(shape(tmp)==extents).and.all(shape(tve)==extents).and. &
      all(shape(spc)==[extents,5]),comm,'AIR5 cache shape')
    if(compensated) then
      call require(allocated(carry),comm,'AIR5 carry not initialized')
      call require(all(shape(carry)==[cells+1,11]),comm,'AIR5 carry shape')
    endif
    if(has_history) call require(all(shape(step_history)==cells+1),comm,'AIR5 step history shape')
    metadata=[1_int64,int(merge(1,0,compensated),int64)]
    expected=metadata
    if(writing) then
      buffer(:,:,:,1:11)=q
      ! Carry has physical nodes only; its halo slots are canonical padding.
      buffer(:,:,:,12:22)=0.0_real64
      if(compensated) buffer(h:h+cells(1),h:h+cells(2),h:h+cells(3),12:22)=carry
      buffer(:,:,:,23)=rho
      buffer(:,:,:,24:26)=vel
      buffer(:,:,:,27)=prs
      buffer(:,:,:,28)=tmp
      buffer(:,:,:,29)=tve
      buffer(:,:,:,30:34)=spc
      if(has_history) then
        call require(all(ieee_is_finite(step_history)).and.all(step_history>=0.0_real64), &
          comm,'AIR5 invalid step history')
        buffer(:,:,:,35)=0.0_real64
        buffer(h:h+cells(1),h:h+cells(2),h:h+cells(3),35)=step_history
      endif
    else
      buffer=0.0_real64
    endif
    call checkpoint_state_transfer(path,writing,.true.,global_shape,origin,cells,halo, &
      buffer,identity,remaining,comm,role=7,metadata=metadata,allow_repartition=permit,restored_exact=matched, &
      repartition_axes=repartition_axes)
    if(present(restored_exact)) restored_exact=matched
    call require(all(metadata==expected),comm,'AIR5 compensation contract mismatch')
    if(.not.writing) then
      if(.not.matched) then
        call require(present(refresh_halos),comm,'AIR5 repartition halo provider missing')
        call refresh_halos(buffer,cells,halo,remaining,comm)
      endif
      q=buffer(:,:,:,1:11)
      if(compensated) carry=buffer(h:h+cells(1),h:h+cells(2),h:h+cells(3),12:22)
      rho=buffer(:,:,:,23)
      vel=buffer(:,:,:,24:26)
      prs=buffer(:,:,:,27)
      tmp=buffer(:,:,:,28)
      tve=buffer(:,:,:,29)
      spc=buffer(:,:,:,30:34)
      if(has_history) then
        step_history=buffer(h:h+cells(1),h:h+cells(2),h:h+cells(3),35)
        call require(all(ieee_is_finite(step_history)).and.all(step_history>=0.0_real64), &
          comm,'AIR5 invalid restored step history')
      endif
    endif
  end subroutine

  subroutine require(condition,comm,label)
    logical, intent(in) :: condition
    integer, intent(in) :: comm
    character(*), intent(in) :: label
    integer :: local_bad, bad, ierr, rank
    local_bad=0
    if (.not.condition) local_bad=1
    call MPI_Allreduce(local_bad,bad,1,MPI_INTEGER,MPI_MAX,comm,ierr)
    if (ierr/=MPI_SUCCESS) bad=1
    if (bad/=0) then
      call MPI_Comm_rank(comm,rank,ierr)
      if (rank==0) then
        write(error_unit,'(a)') 'checkpoint state rejected: '//label
        if(len_trim(failure_batch)>0) write(error_unit,'(a)') &
          'checkpoint context: batch='//trim(failure_batch)//'; last_complete='//trim(failure_last_complete)
        flush(error_unit)
      endif
      call MPI_Abort(comm,71,ierr)
      error stop 'checkpoint state rejected'
    endif
  end subroutine

  integer(int64) function volume(n,comm) result(v)
    integer(int64), intent(in) :: n(3)
    integer, intent(in) :: comm
    integer :: d
    v=1
    do d=1,3
      call require(n(d)>0,comm,'nonpositive extent')
      call require(v<=huge(v)/n(d),comm,'extent overflow')
      v=v*n(d)
    enddo
  end function

  pure subroutine validate_partitions_local(partitions,g,ncomp,lean,rank,offset,total,reason)
    integer(int64),intent(in) :: partitions(:),g(3)
    integer,intent(in) :: ncomp,rank
    logical,intent(in) :: lean
    integer(int64),intent(out) :: offset,total
    character(*),intent(out) :: reason
    integer(int64) :: lo(3),hi(3),other_lo(3),other_hi(3),count(3),owned,all_nodes,v,full_nodes,points
    integer :: r,s,a,b
    offset=0; total=0; owned=0
    reason='extent overflow'
    all_nodes=local_volume(g)
    if(all_nodes==0) return
    do r=0,size(partitions)/8-1
      a=8*r
      lo=partitions(a+1:a+3); count=partitions(a+4:a+6)
      reason='partition origin'
      if(any(lo<0).or.any(lo>=g)) return
      reason='partition cells'
      if(any(count<=0).or.any(count>g-1-lo)) return
      reason='halo width'
      if(partitions(a+7)<0.or.partitions(a+7)>huge(0)) return
      other_hi=count+2*partitions(a+7)+1
      where(lo+count==g-1) count=count+1
      hi=lo+count
      v=local_volume(count); full_nodes=local_volume(other_hi)
      reason='extent overflow'
      if(v==0.or.full_nodes==0) return
      points=full_nodes-v
      reason='extras overflow'
      if(points>huge(points)/ncomp) return
      reason='extras count'
      if(lean) then
        if(partitions(a+8)/=0) return
      else
        if(partitions(a+8)/=points*ncomp) return
      endif
      reason='ownership overflow'
      if(owned>huge(v)-v) return
      owned=owned+v
      reason='overlapping owners'
      do s=0,r-1
        b=8*s
        other_lo=partitions(b+1:b+3)
        other_hi=other_lo+partitions(b+4:b+6)
        where(other_hi==g-1) other_hi=other_hi+1
        if(all(min(hi,other_hi)>max(lo,other_lo))) return
      enddo
      if(r==rank) offset=total
      reason='extras file overflow'
      if(total>huge(total)-partitions(a+8)) return
      total=total+partitions(a+8)
    enddo
    reason='incomplete ownership'
    if(owned/=all_nodes) return
    reason=''
  contains
    pure integer(int64) function local_volume(extents) result(value)
      integer(int64),intent(in) :: extents(3)
      integer :: d
      value=1
      do d=1,3
        if(extents(d)<=0) then
          value=0
          return
        endif
        if(value>huge(value)/extents(d)) then
          value=0
          return
        endif
        value=value*extents(d)
      enddo
    end function
  end subroutine validate_partitions_local

  subroutine dataset(file,name,dims,filetype,writing,comm,dset,space)
    integer(hid_t), intent(in) :: file,filetype
    character(*), intent(in) :: name
    integer(hsize_t), intent(in) :: dims(:)
    logical, intent(in) :: writing
    integer, intent(in) :: comm
    integer(hid_t), intent(out) :: dset,space
    integer(hid_t) :: dtype
    integer(hsize_t) :: actual(size(dims)), maximum(size(dims))
    integer :: err,ndim
    logical :: equal
    if (writing) then
      call h5screate_simple_f(size(dims),dims,space,err)
      call require(err==0,comm,'create dataspace '//name)
      call h5dcreate_f(file,name,filetype,space,dset,err)
      call require(err==0,comm,'create dataset '//name)
    else
      call h5dopen_f(file,name,dset,err)
      call require(err==0,comm,'open dataset '//name)
      call h5dget_space_f(dset,space,err)
      call require(err==0,comm,'get dataspace '//name)
      call h5sget_simple_extent_ndims_f(space,ndim,err)
      call require(err==0.and.ndim==size(dims),comm,'dataset rank '//name)
      call h5sget_simple_extent_dims_f(space,actual,maximum,err)
      call require(err>=0.and.all(actual==dims),comm,'dataset shape '//name)
      call h5dget_type_f(dset,dtype,err)
      call require(err==0,comm,'get datatype '//name)
      call h5tequal_f(dtype,filetype,equal,err)
      call require(err==0.and.equal,comm,'dataset type '//name)
      call h5tclose_f(dtype,err)
      call require(err==0,comm,'close datatype')
    endif
  end subroutine

  subroutine close_dataset(dset,space,mem,comm)
    integer(hid_t), intent(in) :: dset,space,mem
    integer, intent(in) :: comm
    integer :: err
    call h5sclose_f(mem,err)
    call require(err==0,comm,'close memory space')
    call h5sclose_f(space,err)
    call require(err==0,comm,'close file space')
    call h5dclose_f(dset,err)
    call require(err==0,comm,'close dataset')
  end subroutine

  subroutine integers(file,name,data,writing,xfer,comm,rank)
    integer(hid_t), intent(in) :: file,xfer
    character(*), intent(in) :: name
    integer(int64), intent(inout) :: data(:)
    logical, intent(in) :: writing
    integer, intent(in) :: comm,rank
    integer(hid_t) :: dset,space,mem,native
    integer(hsize_t) :: dims(1)
    integer :: err
    dims=size(data,kind=hsize_t)
    native=h5kind_to_type(int64,H5_INTEGER_KIND)
    call dataset(file,name,dims,H5T_STD_I64LE,writing,comm,dset,space)
    call h5screate_simple_f(1,dims,mem,err)
    call require(err==0,comm,'integer memory space')
    if (writing) then
      ! All ranks make the collective call, but only root owns metadata writes.
      if (rank/=0) then
        call h5sselect_none_f(mem,err)
      else
        err=0
      endif
      call require(err==0,comm,'metadata memory selection')
      if (rank/=0) then
        call h5sselect_none_f(space,err)
      else
        err=0
      endif
      call require(err==0,comm,'metadata file selection')
      call h5dwrite_f(dset,native,data,dims,err,mem_space_id=mem,file_space_id=space,xfer_prp=xfer)
    else
      call h5dread_f(dset,native,data,dims,err,mem_space_id=mem,file_space_id=space,xfer_prp=xfer)
    endif
    call require(err==0,comm,'metadata transfer '//name)
    call close_dataset(dset,space,mem,comm)
  end subroutine

  subroutine checkpoint_state_transfer(path,writing,exact,global_shape,origin,cells,halo,q,identity, &
                                       buffer_limit,comm,role,metadata,allow_repartition,restored_exact,repartition_axes,group_name, &
                                       physical_only)
    character(*), intent(in) :: path
    character(*),optional,intent(in) :: group_name
    logical, intent(in) :: writing,exact
    logical,optional,intent(in) :: allow_repartition
    logical,optional,intent(in) :: physical_only
    logical,optional,intent(in) :: repartition_axes(3)
    logical,optional,intent(out) :: restored_exact
    integer, intent(in) :: global_shape(3),origin(3),cells(3),halo,comm
    integer,optional,intent(in) :: role
    integer(int64),optional,intent(inout) :: metadata(:)
    real(real64), contiguous, intent(inout) :: q(:,:,:,:)
    type(checkpoint_state_identity), intent(inout) :: identity
    integer(int64), intent(in) :: buffer_limit
    integer(int64) :: header(13),agreed(13),local(8),owned(3),full(3),g(3),points,extras_count,offset,total
    integer(int64) :: metadata_root(64)
    integer :: metadata_count,root_metadata_count
    integer(int64), allocatable :: partitions(:),current(:)
    real(real64), allocatable :: extras(:)
    integer(hid_t) :: file,container,access,xfer,dset,space,mem,native
    integer(hsize_t) :: dims3(3),mdims3(3),start3(3),count3(3),dims1(1),mdims1(1),start1(1),count1(1)
    integer :: rank,np,err,mpi_i64,ncomp,old_np,r,a,i,j,k,m,n,allocerr,mode(3),root_mode(3),state_role
    integer :: axes_mode(3),root_axes(3)
    logical :: permit,restore_exact,matched,lean
    character(16) :: name
    character(64) :: partition_error
    character(len(path)) :: root_path
    call MPI_Comm_rank(comm,rank,err)
    call require(err==MPI_SUCCESS,comm,'rank')
    call MPI_Comm_size(comm,np,err)
    call require(err==MPI_SUCCESS,comm,'size')
    call MPI_Type_match_size(MPI_TYPECLASS_INTEGER,8,mpi_i64,err)
    call require(err==MPI_SUCCESS,comm,'int64 MPI type')
    permit=.false.
    if(present(allow_repartition)) permit=allow_repartition
    restore_exact=exact
    lean=.false.
    if(present(physical_only)) lean=physical_only
    mode=[merge(1,0,writing),merge(1,0,exact),merge(1,0,permit)]
    root_mode=mode
    call MPI_Bcast(root_mode,3,MPI_INTEGER,0,comm,err)
    call require(err==MPI_SUCCESS.and.all(root_mode==mode),comm,'transfer mode mismatch')
    axes_mode=1
    if(present(repartition_axes)) axes_mode=merge(1,0,repartition_axes)
    root_axes=axes_mode
    call MPI_Bcast(root_axes,3,MPI_INTEGER,0,comm,err)
    call require(err==MPI_SUCCESS.and.all(root_axes==axes_mode),comm,'repartition axes mismatch')
    metadata_count=-1
    if(present(metadata)) metadata_count=size(metadata)
    root_metadata_count=metadata_count
    call MPI_Bcast(root_metadata_count,1,MPI_INTEGER,0,comm,err)
    call require(err==MPI_SUCCESS.and.metadata_count==root_metadata_count.and. &
      metadata_count>=-1.and.metadata_count/=0.and.metadata_count<=64,comm,'metadata layout mismatch')
    if(present(metadata)) then
      call require(metadata_count>0,comm,'empty metadata payload')
      if(writing) then
        metadata_root(1:metadata_count)=metadata
        call MPI_Bcast(metadata_root,metadata_count,mpi_i64,0,comm,err)
        call require(err==MPI_SUCCESS.and.all(metadata==metadata_root(1:metadata_count)),comm, &
          'global metadata differs between ranks')
      endif
    endif
    ! The path must be agreed before entering parallel HDF5.
    n=len(path)
    call MPI_Bcast(n,1,MPI_INTEGER,0,comm,err)
    call require(err==MPI_SUCCESS.and.n==len(path),comm,'path length mismatch')
    root_path=path
    call MPI_Bcast(root_path,n,MPI_CHARACTER,0,comm,err)
    call require(err==MPI_SUCCESS.and.path==root_path.and.len_trim(path)>0,comm,'path mismatch')
    ncomp=size(q,4)
    state_role=1
    if (present(role)) state_role=role
    call require(state_role==1.or.(state_role==2.and.ncomp==11).or. &
      ((state_role==3.or.state_role==4).and.ncomp==34).or.(state_role==5.and.ncomp==17).or. &
      (state_role==6.and.ncomp==44).or.(state_role==7.and.(ncomp==34.or.ncomp==35)).or. &
      (state_role==8.and.ncomp==26).or.(state_role==9.and.(ncomp==34.or.ncomp==204)).or. &
      (state_role==10.and.ncomp==136).or.(state_role==11.and.ncomp==5.and.lean), &
      comm,'state component role')
    g=int(global_shape,int64)
    header=[magic,schema,1_int64,g,int(ncomp,int64),int(np,int64),identity%step, &
      transfer(identity%time,0_int64),transfer(identity%dt_used,0_int64),transfer(identity%dt_next,0_int64), &
      int(state_role,int64)]
    agreed=header
    call MPI_Bcast(agreed,13,mpi_i64,0,comm,err)
    call require(err==MPI_SUCCESS,comm,'state identity broadcast')
    call require(all(header(1:8)==agreed(1:8)),comm,'inconsistent global layout')
    call require(header(13)==agreed(13),comm,'inconsistent state role')
    if (writing) call require(all(header==agreed),comm,'inconsistent state clock')
    full=int(cells,int64)+2_int64*int(halo,int64)+1
    call require(all(g>1).and.all(cells>0).and.halo>=0.and.ncomp>0.and.ncomp<=9999,comm,'invalid layout')
    call require(all(origin>=0).and.all(int(origin,int64)+cells<g),comm,'partition bounds')
    call require(all(int(shape(q(:,:,:,1)),int64)==full),comm,'local array shape')
    owned=int(cells,int64)
    where (int(origin,int64)+cells==g-1) owned=owned+1
    points=volume(full,comm)-volume(owned,comm)
    call require(points<=huge(points)/ncomp,comm,'extras overflow')
    extras_count=points*ncomp
    if(lean) extras_count=0
    call require(buffer_limit>=0.and.extras_count<=buffer_limit/8,comm,'extras buffer budget')
    call require(extras_count<=huge(n)-1,comm,'extras indexing capacity')
    call require(int(np,int64)*8<=huge(n),comm,'partition table capacity')
    call require(int(np,int64)*64<=buffer_limit,comm,'current partition budget')
    allocate(current(8*np),stat=allocerr)
    call require(allocerr==0,comm,'partition allocation')
    local=[int(origin,int64),int(cells,int64),int(halo,int64),extras_count]
    call MPI_Allgather(local,8,mpi_i64,current,8,mpi_i64,comm,err)
    call require(err==MPI_SUCCESS,comm,'partition gather')
    call validate_partitions_local(current,g,ncomp,lean,rank,offset,total,partition_error)
    call require(len_trim(partition_error)==0,comm,trim(partition_error))
    call h5open_f(err)
    call require(err==0,comm,'HDF5 initialize')
    call h5pcreate_f(H5P_FILE_ACCESS_F,access,err)
    call require(err==0,comm,'file access property')
    call h5pset_fapl_mpio_f(access,comm,MPI_INFO_NULL,err)
    call require(err==0,comm,'parallel file access')
    if(present(group_name)) then
      call require(group_name=='inherited'.or.(group_name=='wall_statistics'.and.state_role==9).or. &
        (group_name=='air5_volume_statistics'.and.state_role==10), &
        comm,'unsupported state group')
      if(writing) then
        call h5fopen_f(path,H5F_ACC_RDWR_F,container,err,access_prp=access)
      else
        call h5fopen_f(path,H5F_ACC_RDONLY_F,container,err,access_prp=access)
      endif
      call require(err==0,comm,'open grouped state file')
      if(writing) then
        call h5gcreate_f(container,group_name,file,err)
      else
        call h5gopen_f(container,group_name,file,err)
      endif
    else if (writing) then
      call h5fcreate_f(path,H5F_ACC_EXCL_F,file,err,access_prp=access)
    else
      call h5fopen_f(path,H5F_ACC_RDONLY_F,file,err,access_prp=access)
    endif
    call require(err==0,comm,'open state file')
    call h5pclose_f(access,err)
    call require(err==0,comm,'close file access property')
    call h5pcreate_f(H5P_DATASET_XFER_F,xfer,err)
    call require(err==0,comm,'transfer property')
    call h5pset_dxpl_mpio_f(xfer,H5FD_MPIO_COLLECTIVE_F,err)
    call require(err==0,comm,'collective transfer property')
    call integers(file,'identity',header,writing,xfer,comm,rank)
    call require(header(1)==magic.and.header(2)==schema.and.header(3)==1,comm,'state identity/version/phase')
    call require(header(13)==state_role,comm,'state role mismatch')
    call require(all(header(4:6)==g).and.header(7)==ncomp,comm,'global state shape')
    call require(header(8)>0.and.header(8)<=huge(n)/8,comm,'original rank count')
    old_np=int(header(8))
    identity%step=header(9)
    identity%time=transfer(header(10),0.0_real64)
    identity%dt_used=transfer(header(11),0.0_real64)
    identity%dt_next=transfer(header(12),0.0_real64)
    call require(identity%step>=0.and.ieee_is_finite(identity%time).and.identity%time>=0.and. &
      ieee_is_finite(identity%dt_used).and.identity%dt_used>=0.and. &
      ieee_is_finite(identity%dt_next).and.identity%dt_next>0,comm,'invalid state clock')
    call require(identity%step==0.or.identity%dt_used>0,comm,'missing completed-step dt')
    call require(int(old_np,int64)*64<=buffer_limit,comm,'partition buffer budget')
    call require((int(old_np,int64)+np)*64<=buffer_limit,comm,'combined partition budget')
    if (writing.or.exact) then
      call require(max(1_int64,extras_count)<=(buffer_limit-(int(old_np,int64)+np)*64)/8, &
        comm,'combined state buffer budget')
    endif
    allocate(partitions(8*old_np),stat=allocerr)
    call require(allocerr==0,comm,'saved partition allocation')
    if (writing) partitions=current
    call integers(file,'partitions',partitions,writing,xfer,comm,rank)
    call validate_partitions_local(partitions,g,ncomp,lean,rank,offset,total,partition_error)
    call require(len_trim(partition_error)==0,comm,'saved '//trim(partition_error))
    matched=old_np==np
    if(matched) matched=all(partitions==current)
    if(.not.writing.and.permit.and..not.matched) then
      do m=1,3
        if(axes_mode(m)==1) cycle
        do r=0,np-1
          a=8*r
          call require(current(a+m)==0.and.current(a+m+3)==g(m)-1,comm, &
            'repartition nonperiodic direction must stay unpartitioned')
        enddo
        do r=0,old_np-1
          a=8*r
          call require(partitions(a+m)==0.and.partitions(a+m+3)==g(m)-1,comm, &
            'repartition nonperiodic direction must stay unpartitioned')
        enddo
      enddo
    endif
    if(.not.writing.and.permit) restore_exact=matched
    if(lean) then
      call require(writing.or.matched.or.permit,comm,'lean restart topology is not registered')
      restore_exact=.false.
    endif
    if(present(restored_exact)) restored_exact=matched
    if (restore_exact.or.writing) then
      call require(old_np==np,comm,'exact restore rank count')
      call require(all(partitions==current),comm,'exact restore partition mismatch')
    endif
    if(present(metadata)) call integers(file,'metadata',metadata,writing,xfer,comm,rank)
    native=h5kind_to_type(real64,H5_REAL_KIND)
    dims3=g
    mdims3=full
    count3=owned
    if (.not.writing.and..not.restore_exact) count3=int(cells,hsize_t)+1
    do m=1,ncomp
      write(name,'("q",i4.4)') m
      call dataset(file,trim(name),dims3,H5T_IEEE_F64LE,writing,comm,dset,space)
      call h5screate_simple_f(3,mdims3,mem,err)
      call require(err==0,comm,'state memory space')
      start3=origin
      call h5sselect_hyperslab_f(space,H5S_SELECT_SET_F,start3,count3,err)
      call require(err==0,comm,'state file selection')
      start3=halo
      call h5sselect_hyperslab_f(mem,H5S_SELECT_SET_F,start3,count3,err)
      call require(err==0,comm,'state memory selection')
      if (writing) then
        call h5dwrite_f(dset,native,q(:,:,:,m),mdims3,err, &
          mem_space_id=mem,file_space_id=space,xfer_prp=xfer)
      else
        call h5dread_f(dset,native,q(:,:,:,m),mdims3,err, &
          mem_space_id=mem,file_space_id=space,xfer_prp=xfer)
      endif
      call require(err==0,comm,'state transfer')
      call close_dataset(dset,space,mem,comm)
    enddo
    block
      ! Repartition reads validate the old extras dataset but do not reuse halos.
      if (.not.writing.and..not.restore_exact) extras_count=0
      allocate(extras(max(1,int(extras_count))),stat=allocerr)
      call require(allocerr==0,comm,'extras allocation')
      if (writing.and..not.lean) call extra_values(.true.)
      dims1=total
      mdims1=size(extras,kind=hsize_t)
      call dataset(file,'rank_extras',dims1,H5T_IEEE_F64LE,writing,comm,dset,space)
      call h5screate_simple_f(1,mdims1,mem,err)
      call require(err==0,comm,'extras memory space')
      if (extras_count>0) then
        start1=offset
        count1=extras_count
        call h5sselect_hyperslab_f(space,H5S_SELECT_SET_F,start1,count1,err)
      else
        call h5sselect_none_f(space,err)
      endif
      call require(err==0,comm,'extras file selection')
      if (extras_count==0) then
        call h5sselect_none_f(mem,err)
      else
        err=0
      endif
      call require(err==0,comm,'extras memory selection')
      if (writing) then
        call h5dwrite_f(dset,native,extras,mdims1,err, &
          mem_space_id=mem,file_space_id=space,xfer_prp=xfer)
      else
        call h5dread_f(dset,native,extras,mdims1,err, &
          mem_space_id=mem,file_space_id=space,xfer_prp=xfer)
      endif
      call require(err==0,comm,'extras transfer')
      if (.not.writing.and.restore_exact) call extra_values(.false.)
      call close_dataset(dset,space,mem,comm)
    end block
    call h5pclose_f(xfer,err)
    call require(err==0,comm,'close transfer property')
    if(present(group_name)) then
      call h5gclose_f(file,err)
      call require(err==0,comm,'close state group')
      call h5fclose_f(container,err)
    else
      call h5fclose_f(file,err)
    endif
    call require(err==0,comm,'close state file')
    ! Do not globally shut down HDF5: the caller may have other files open.
  contains
    subroutine extra_values(pack)
      logical, intent(in) :: pack
      integer :: p
      p=0
      do m=1,ncomp
        do k=1,size(q,3)
          do j=1,size(q,2)
            do i=1,size(q,1)
              if (all([i,j,k]>halo).and.all(int([i,j,k],int64)<=halo+owned)) cycle
              p=p+1
              if (pack) then
                extras(p)=q(i,j,k,m)
              else
                q(i,j,k,m)=extras(p)
              endif
            enddo
          enddo
        enddo
      enddo
      call require(int(p,int64)==extras_count,comm,'extras packing count')
    end subroutine
  end subroutine
end module
