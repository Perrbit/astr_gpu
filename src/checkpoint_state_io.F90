module checkpoint_state_io
  use iso_fortran_env, only: int64, real64, error_unit
  use ieee_arithmetic, only: ieee_is_finite
  use mpi
  use hdf5
  implicit none
  private
  public :: checkpoint_state_transfer, checkpoint_state_identity
  public :: checkpoint_perfect_gas_state, allocate_checkpoint_buffer, checkpoint_state_require

  ! This is a state-file primitive, not a published restart bundle.
  integer(int64), parameter :: magic=int(z'4153545251533031',int64), schema=2
  type checkpoint_state_identity
    integer(int64) :: step=0
    real(real64) :: time=0, dt_used=0, dt_next=0
  end type
contains
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
                                         q,rho,vel,prs,tmp,identity,budget,comm)
    character(*),intent(in) :: path
    logical,intent(in) :: writing
    integer,intent(in) :: global_shape(3),origin(3),cells(3),halo,comm
    real(real64),contiguous,intent(inout) :: q(:,:,:,:),rho(:,:,:),vel(:,:,:,:), &
      prs(:,:,:),tmp(:,:,:)
    type(checkpoint_state_identity),intent(inout) :: identity
    integer(int64),intent(in) :: budget
    real(real64),allocatable :: buffer(:,:,:,:)
    integer(int64) :: remaining
    integer :: extents(3)
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
    endif
    ! Role 2 preserves q and cached primitives independently, including halos.
    call checkpoint_state_transfer(path,writing,.true.,global_shape,origin,cells,halo, &
      buffer,identity,remaining,comm,role=2)
    if (.not.writing) then
      q=buffer(:,:,:,1:5)
      rho=buffer(:,:,:,6)
      vel=buffer(:,:,:,7:9)
      prs=buffer(:,:,:,10)
      tmp=buffer(:,:,:,11)
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
      if (rank==0) write(error_unit,'(a)') 'checkpoint state rejected: '//label
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
                                       buffer_limit,comm,role)
    character(*), intent(in) :: path
    logical, intent(in) :: writing,exact
    integer, intent(in) :: global_shape(3),origin(3),cells(3),halo,comm
    integer,optional,intent(in) :: role
    real(real64), contiguous, intent(inout) :: q(:,:,:,:)
    type(checkpoint_state_identity), intent(inout) :: identity
    integer(int64), intent(in) :: buffer_limit
    integer(int64) :: header(13),agreed(13),local(8),owned(3),full(3),g(3),points,extras_count,offset,total
    integer(int64), allocatable :: partitions(:),current(:)
    real(real64), allocatable :: extras(:)
    integer(hid_t) :: file,access,xfer,dset,space,mem,native
    integer(hsize_t) :: dims3(3),mdims3(3),start3(3),count3(3),dims1(1),mdims1(1),start1(1),count1(1)
    integer :: rank,np,err,mpi_i64,ncomp,old_np,r,s,a,b,i,j,k,m,n,allocerr,mode(2),root_mode(2),state_role
    integer(int64) :: lo(3),hi(3),other_lo(3),other_hi(3),count(3),sum_owned,v
    character(16) :: name
    character(len(path)) :: root_path
    call MPI_Comm_rank(comm,rank,err)
    call require(err==MPI_SUCCESS,comm,'rank')
    call MPI_Comm_size(comm,np,err)
    call require(err==MPI_SUCCESS,comm,'size')
    call MPI_Type_match_size(MPI_TYPECLASS_INTEGER,8,mpi_i64,err)
    call require(err==MPI_SUCCESS,comm,'int64 MPI type')
    mode=[merge(1,0,writing),merge(1,0,exact)]
    root_mode=mode
    call MPI_Bcast(root_mode,2,MPI_INTEGER,0,comm,err)
    call require(err==MPI_SUCCESS.and.all(root_mode==mode),comm,'transfer mode mismatch')
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
      (state_role==3.and.ncomp==34),comm,'state component role')
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
    call require(buffer_limit>=0.and.extras_count<=buffer_limit/8,comm,'extras buffer budget')
    call require(extras_count<=huge(n)-1,comm,'extras indexing capacity')
    call require(int(np,int64)*8<=huge(n),comm,'partition table capacity')
    call require(int(np,int64)*64<=buffer_limit,comm,'current partition budget')
    allocate(current(8*np),stat=allocerr)
    call require(allocerr==0,comm,'partition allocation')
    local=[int(origin,int64),int(cells,int64),int(halo,int64),extras_count]
    call MPI_Allgather(local,8,mpi_i64,current,8,mpi_i64,comm,err)
    call require(err==MPI_SUCCESS,comm,'partition gather')
    sum_owned=0
    do r=0,np-1
      a=8*r
      lo=current(a+1:a+3)
      count=current(a+4:a+6)
      where(lo+count==g-1) count=count+1
      hi=lo+count
      v=volume(count,comm)
      call require(sum_owned<=huge(v)-v,comm,'owned count overflow')
      sum_owned=sum_owned+v
      do s=0,r-1
        b=8*s
        other_lo=current(b+1:b+3)
        other_hi=other_lo+current(b+4:b+6)
        where(other_hi==g-1) other_hi=other_hi+1
        call require(.not.all(min(hi,other_hi)>max(lo,other_lo)),comm,'overlapping owners')
      enddo
    enddo
    call require(sum_owned==volume(g,comm),comm,'incomplete ownership')
    call h5open_f(err)
    call require(err==0,comm,'HDF5 initialize')
    call h5pcreate_f(H5P_FILE_ACCESS_F,access,err)
    call require(err==0,comm,'file access property')
    call h5pset_fapl_mpio_f(access,comm,MPI_INFO_NULL,err)
    call require(err==0,comm,'parallel file access')
    if (writing) then
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
    total=0
    offset=0
    sum_owned=0
    do r=0,old_np-1
      a=8*r
      lo=partitions(a+1:a+3)
      count=partitions(a+4:a+6)
      call require(all(lo>=0).and.all(lo<g),comm,'saved partition origin')
      call require(all(count>0).and.all(count<=g-1-lo),comm,'saved partition cells')
      call require(partitions(a+7)>=0.and.partitions(a+7)<=huge(n),comm,'saved halo width')
      other_hi=count+2*partitions(a+7)+1
      where(lo+count==g-1) count=count+1
      hi=lo+count
      v=volume(count,comm)
      points=volume(other_hi,comm)-v
      call require(points<=huge(points)/ncomp,comm,'saved extras overflow')
      call require(partitions(a+8)==points*ncomp,comm,'saved extras count')
      call require(sum_owned<=huge(v)-v,comm,'saved ownership overflow')
      sum_owned=sum_owned+v
      do s=0,r-1
        b=8*s
        other_lo=partitions(b+1:b+3)
        other_hi=other_lo+partitions(b+4:b+6)
        where(other_hi==g-1) other_hi=other_hi+1
        call require(.not.all(min(hi,other_hi)>max(lo,other_lo)),comm,'saved overlapping owners')
      enddo
      if (r==rank) offset=total
      call require(total<=huge(total)-partitions(a+8),comm,'extras file overflow')
      total=total+partitions(a+8)
    enddo
    call require(sum_owned==volume(g,comm),comm,'saved incomplete ownership')
    if (exact.or.writing) then
      call require(old_np==np,comm,'exact restore rank count')
      call require(all(partitions==current),comm,'exact restore partition mismatch')
    endif
    native=h5kind_to_type(real64,H5_REAL_KIND)
    dims3=g
    mdims3=full
    count3=owned
    if (.not.writing.and..not.exact) count3=int(cells,hsize_t)+1
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
      if (.not.writing.and..not.exact) extras_count=0
      allocate(extras(max(1,int(extras_count))),stat=allocerr)
      call require(allocerr==0,comm,'extras allocation')
      if (writing) call extra_values(.true.)
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
      if (.not.writing.and.exact) call extra_values(.false.)
      call close_dataset(dset,space,mem,comm)
    end block
    call h5pclose_f(xfer,err)
    call require(err==0,comm,'close transfer property')
    call h5fclose_f(file,err)
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
