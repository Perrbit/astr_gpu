program state_probe
  use iso_c_binding
  use mpi
  use device_state
  use astr_precision, only: real_kind,c_real_kind,real_bytes,real_to_word
  implicit none
  interface
    integer(c_int) function consume(handle,reversed,count) bind(C,name='astr_mb1_consume')
      import c_int,c_ptr,c_real_kind,c_size_t
      type(c_ptr),value :: handle
      real(c_real_kind),intent(out) :: reversed(*)
      integer(c_size_t),value :: count
    end function
  end interface
  integer(c_int),parameter :: im=7,jm=5,km=3
  integer(c_int) :: halo,debug,field,rank,local_rank,code
  integer :: ierr,i,j,k,c,axis,test,status
  integer(c_int64_t) :: stride
  integer(c_size_t) :: qbytes,pbytes
  character(len=32) :: text
  type(c_ptr) :: handle
  type(device_view) :: view
  real(real_kind),allocatable :: q(:,:,:,:),back(:,:,:,:),rhs(:,:,:,:),saved(:,:,:,:),reversed(:),expected(:)

  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text,status=status)
  if (status/=0) error stop 'state probe requires Open MPI local rank'
  read(text,*) local_rank
  code=device_bind(local_rank)
  if (code/=0) error stop 'pre-MPI device binding failed'
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
  do test=1,2
    halo=2*test+1
    debug=test-1
    code=device_state_create(local_rank,im,jm,km,halo,debug,handle)
    call check(code,'create')
    allocate(q(-halo:im+halo,-halo:jm+halo,-halo:km+halo,5), &
             back(-halo:im+halo,-halo:jm+halo,-halo:km+halo,5), &
             rhs(0:im,0:jm,0:km,5),saved(0:im,0:jm,0:km,5))
    do c=1,5
      do k=-halo,km+halo
        do j=-halo,jm+halo
          do i=-halo,im+halo
            q(i,j,k,c)=real(rank,real_kind)*1048576._real_kind+real(c,real_kind)*65536._real_kind &
                +real(k+halo,real_kind)*1024._real_kind+real(j+halo,real_kind)*32._real_kind+real(i+halo,real_kind)
          end do
        end do
      end do
    end do
    q(-halo,-halo,-halo,1)=-0._real_kind
    saved=-q(0:im,0:jm,0:km,:)
    qbytes=int(size(q),c_size_t)*real_bytes
    pbytes=int(size(rhs),c_size_t)*real_bytes
    do field=device_q,device_save
      code=device_state_view(handle,field,view)
      call check(code,'view')
      if (view%rank/=4.or.view%element_bytes/=real_bytes) call fail('rank or native storage type')
      if (.not.c_associated(view%data)) call fail('missing device buffer')
      if (field==device_q) then
        if (any(view%lower/=[-halo,-halo,-halo,1,1])) call fail('q lower bounds')
        if (any(view%extent/=[im+2*halo+1,jm+2*halo+1,km+2*halo+1,5,1])) call fail('q extents')
      else
        if (any(view%lower/=[0,0,0,1,1])) call fail('physical lower bounds')
        if (any(view%extent/=[im+1,jm+1,km+1,5,1])) call fail('physical extents')
      end if
      stride=1
      do axis=1,5
        if (view%stride(axis)/=stride) call fail('Fortran element stride')
        stride=stride*view%extent(axis)
      end do
    end do
    code=device_state_upload(handle,device_q,q,qbytes)
    call check(code,'q upload')
    code=device_state_upload(handle,device_save,saved,pbytes)
    call check(code,'save upload')
    allocate(reversed(size(q)),expected(size(q)))
    expected=reshape(q(im+halo:-halo:-1,jm+halo:-halo:-1,km+halo:-halo:-1,5:1:-1),[size(q)])
    code=consume(handle,reversed,int(size(q),c_size_t))
    call check(code,'borrowed-view consumer')
    call exact(reversed,expected,'all q halo and component indices')
    code=device_state_download(handle,device_rhs,rhs,pbytes)
    call check(code,'rhs download')
    call exact(reshape(rhs,[size(rhs)]),reshape(q(0:im,0:jm,0:km,:),[size(rhs)]),'physical q-to-rhs mapping')
    code=device_state_download(handle,device_save,rhs,pbytes)
    call check(code,'save download')
    call exact(reshape(rhs,[size(rhs)]),reshape(saved,[size(saved)]),'qsave isolated from rhs')
    code=device_state_download(handle,device_q,back,qbytes)
    call check(code,'q download')
    call exact(reshape(back,[size(q)]),reshape(q,[size(q)]),'q untouched by consumer')
    code=device_state_synchronize(handle)
    call check(code,'completion')
    code=device_state_destroy(handle)
    call check(code,'destroy')
    if (c_associated(handle)) call fail('destroy did not invalidate owner handle')
    write(*,'(a,i0,a,i0,a,i0,a,i0,a)') 'MB1 rank=',rank,' halo=',halo,' debug=',debug, &
      ' element_bytes=',real_bytes,' exact layout/copy/view PASS'
    deallocate(q,back,rhs,saved,reversed,expected)
  end do
  call MPI_Finalize(ierr)
contains
  subroutine fail(message)
    character(len=*),intent(in) :: message
    write(*,'(a,i0,2a)') 'MB1 FAIL rank=',rank,': ',message
    call MPI_Abort(MPI_COMM_WORLD,1,ierr)
    error stop 1
  end subroutine
  subroutine check(status,message)
    integer(c_int),intent(in) :: status
    character(len=*),intent(in) :: message
    if (status/=0) then
      write(*,'(a,i0)') 'device status=',status
      call fail(message)
    end if
  end subroutine
  subroutine exact(actual,reference,message)
    real(real_kind),intent(in) :: actual(:),reference(:)
    character(len=*),intent(in) :: message
    if (any(real_to_word(actual)/=real_to_word(reference))) call fail(message)
  end subroutine
end program
