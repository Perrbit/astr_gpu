module me6opti_halo
  use astr_precision, only: real_kind
  use astr_precision_mpi, only: mpi_real_type
  use mpi
  implicit none
  private
  public :: extend_me6_edges_cpu,release_me6_edges_cpu,me6_edge_tags
  integer,parameter :: me6_edge_tags(2,2)=reshape([21801,21802,21803,21804],[2,2])
  type :: edge_buffers
    real(real_kind),allocatable :: sl(:,:,:,:),sh(:,:,:,:),rh(:,:,:,:),rl(:,:,:,:)
  end type
  type(edge_buffers),save :: buffers(2)
contains
  ! Physical nodes and face halos must already hold the current stage primitives.
  ! Y includes X ghosts; Z then includes X/Y ghosts. No shared node is averaged.
  subroutine extend_me6_edges_cpu(im,jm,km,hm,velocity,temperature,parts,peers)
    integer,intent(in) :: im,jm,km,hm,parts(3),peers(2,3)
    real(real_kind),intent(inout) :: velocity(-hm:,-hm:,-hm:,:),temperature(-hm:,-hm:,-hm:)
    integer :: axis,slot,nx,nt,i,j,k,h,c,lo(3),hi(3),lowghost(3),highghost(3),last(3),n,ierr
    integer :: status(MPI_STATUS_SIZE),ignored
    real(real_kind) :: lowvalue,highvalue
    last=[im,jm,km];nx=im+2*hm+1
    do axis=2,3
      slot=axis-1
      nt=km+1
      if(axis==3) nt=jm+2*hm+1
      if(parts(axis)>1 .and. .not.allocated(buffers(slot)%sl)) then
        allocate(buffers(slot)%sl(hm,nx,nt,4),buffers(slot)%sh(hm,nx,nt,4), &
          buffers(slot)%rh(hm,nx,nt,4),buffers(slot)%rl(hm,nx,nt,4))
      endif
      do c=1,4
      do j=1,nt
      do i=1,nx
      do h=1,hm
        lo=[i-hm-1,h,j-1]
        if(axis==3) lo=[i-hm-1,j-hm-1,h]
        hi=lo;hi(axis)=last(axis)-hm+h-1
        lowghost=lo;lowghost(axis)=h-hm-1
        highghost=lo;highghost(axis)=last(axis)+h
        if(c<=3) then
          lowvalue=velocity(lo(1),lo(2),lo(3),c)
          highvalue=velocity(hi(1),hi(2),hi(3),c)
        else
          lowvalue=temperature(lo(1),lo(2),lo(3))
          highvalue=temperature(hi(1),hi(2),hi(3))
        endif
        if(parts(axis)>1) then
          buffers(slot)%sl(h,i,j,c)=lowvalue;buffers(slot)%sh(h,i,j,c)=highvalue
        elseif(c<=3) then
          velocity(lowghost(1),lowghost(2),lowghost(3),c)=highvalue
          velocity(highghost(1),highghost(2),highghost(3),c)=lowvalue
        else
          temperature(lowghost(1),lowghost(2),lowghost(3))=highvalue
          temperature(highghost(1),highghost(2),highghost(3))=lowvalue
        endif
      enddo
      enddo
      enddo
      enddo
      if(parts(axis)==1) cycle
      n=size(buffers(slot)%sl)
      call mpi_sendrecv(buffers(slot)%sl,n,mpi_real_type,peers(1,axis),me6_edge_tags(1,slot), &
        buffers(slot)%rh,n,mpi_real_type,peers(2,axis),me6_edge_tags(1,slot),MPI_COMM_WORLD,status,ierr)
      if(ierr/=MPI_SUCCESS) call mpi_abort(MPI_COMM_WORLD,ierr,ignored)
      call mpi_sendrecv(buffers(slot)%sh,n,mpi_real_type,peers(2,axis),me6_edge_tags(2,slot), &
        buffers(slot)%rl,n,mpi_real_type,peers(1,axis),me6_edge_tags(2,slot),MPI_COMM_WORLD,status,ierr)
      if(ierr/=MPI_SUCCESS) call mpi_abort(MPI_COMM_WORLD,ierr,ignored)
      do c=1,4
      do j=1,nt
      do i=1,nx
      do h=1,hm
        lowghost=[i-hm-1,h-hm-1,j-1]
        if(axis==3) lowghost=[i-hm-1,j-hm-1,h-hm-1]
        highghost=lowghost;highghost(axis)=last(axis)+h
        if(c<=3) then
          velocity(lowghost(1),lowghost(2),lowghost(3),c)=buffers(slot)%rl(h,i,j,c)
          velocity(highghost(1),highghost(2),highghost(3),c)=buffers(slot)%rh(h,i,j,c)
        else
          temperature(lowghost(1),lowghost(2),lowghost(3))=buffers(slot)%rl(h,i,j,c)
          temperature(highghost(1),highghost(2),highghost(3))=buffers(slot)%rh(h,i,j,c)
        endif
      enddo
      enddo
      enddo
      enddo
    enddo
  end subroutine

  subroutine release_me6_edges_cpu()
    integer :: slot
    do slot=1,2
      if(allocated(buffers(slot)%sl)) deallocate(buffers(slot)%sl,buffers(slot)%sh,buffers(slot)%rh,buffers(slot)%rl)
    enddo
  end subroutine
end module
