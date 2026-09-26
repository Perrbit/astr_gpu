module chemistry_compensation
  use iso_fortran_env, only: real64
  use mpi
  implicit none
  private
  logical, public, save :: air5_compensated=.false.
  integer, public, save :: air5_compensation_comm=MPI_COMM_NULL
  real(real64), allocatable, public, save :: air5_carry(:,:,:,:)
  real(real64), allocatable, public, save :: air5_origin(:,:,:,:),air5_origin_carry(:,:,:,:)
  public :: compensated_add,compensated_rk,compensated_mean
  public :: configure_air5_compensation,compensation_exchange_faces
contains
  ! The represented value is high - carry, matching the ROS-2 accumulator.
  pure subroutine compensated_add(high,carry,increment)
    real(real64), intent(inout) :: high,carry
    real(real64), intent(in) :: increment
    real(real64) :: adjusted,updated
    adjusted=increment-carry
    updated=high+adjusted
    carry=(updated-high)-adjusted
    high=updated
  end subroutine

  pure subroutine compensated_rk(high,carry,origin,origin_carry,a,b,increment)
    real(real64), intent(inout) :: high,carry
    real(real64), intent(in) :: origin,origin_carry,a,b,increment
    real(real64) :: difference,virtual,error
    ! TwoDiff retains cancellation error in the affine SSP-RK combination.
    difference=origin-high
    virtual=difference-origin
    error=(origin-(difference-virtual))-(high+virtual)
    carry=a*origin_carry+b*carry
    call compensated_add(high,carry,a*difference)
    call compensated_add(high,carry,a*error)
    call compensated_add(high,carry,increment)
  end subroutine

  pure subroutine compensated_mean(left,right,cl,cr,carry)
    real(real64), intent(in) :: left,right,cl,cr
    real(real64), intent(out) :: carry
    real(real64) :: total,virtual,error
    total=left+right
    virtual=total-left
    error=(left-(total-virtual))+(right-virtual)
    carry=0.5_real64*(cl+cr)-0.5_real64*error
  end subroutine

  subroutine configure_air5_compensation(enabled,ni,nj,nk)
    logical, intent(in) :: enabled
    integer, intent(in) :: ni,nj,nk
    integer :: ierr,abort_error
    air5_compensated=enabled
    if(air5_compensation_comm/=MPI_COMM_NULL) call MPI_Comm_free(air5_compensation_comm,ierr)
    if(allocated(air5_carry)) deallocate(air5_carry,air5_origin,air5_origin_carry)
    if(.not.enabled) return
    call MPI_Comm_dup(MPI_COMM_WORLD,air5_compensation_comm,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,ierr,abort_error)
    allocate(air5_carry(0:ni,0:nj,0:nk,11),air5_origin(0:ni,0:nj,0:nk,11), &
      air5_origin_carry(0:ni,0:nj,0:nk,11))
    air5_carry=0.0_real64
    air5_origin=0.0_real64
    air5_origin_carry=0.0_real64
  end subroutine

  subroutine compensation_exchange_faces(low,high,cl,ch,lower,upper,comm)
    real(real64), intent(in) :: low(:,:,:),high(:,:,:)
    real(real64), intent(inout) :: cl(:,:,:),ch(:,:,:)
    integer, intent(in) :: lower,upper,comm
    real(real64), allocatable :: send_low(:,:,:,:),send_high(:,:,:,:)
    real(real64), allocatable :: recv_low(:,:,:,:),recv_high(:,:,:,:)
    integer :: i,j,m,n,ierr,abort_error
    allocate(send_low(size(low,1),size(low,2),size(low,3),2))
    allocate(send_high,mold=send_low)
    allocate(recv_low,mold=send_low)
    allocate(recv_high,mold=send_low)
    send_low(:,:,:,1)=low; send_low(:,:,:,2)=cl
    send_high(:,:,:,1)=high; send_high(:,:,:,2)=ch
    n=size(send_low)
    ! The caller supplies a private communicator, isolating tags from qswap.
    call MPI_Sendrecv(send_low,n,MPI_DOUBLE_PRECISION,lower,1, &
      recv_high,n,MPI_DOUBLE_PRECISION,upper,1,comm,MPI_STATUS_IGNORE,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(comm,ierr,abort_error)
    call MPI_Sendrecv(send_high,n,MPI_DOUBLE_PRECISION,upper,2, &
      recv_low,n,MPI_DOUBLE_PRECISION,lower,2,comm,MPI_STATUS_IGNORE,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(comm,ierr,abort_error)
    do m=1,size(low,3)
      do j=1,size(low,2)
        do i=1,size(low,1)
          if(lower/=MPI_PROC_NULL) call compensated_mean(low(i,j,m), &
            recv_low(i,j,m,1),send_low(i,j,m,2),recv_low(i,j,m,2),cl(i,j,m))
          if(upper/=MPI_PROC_NULL) call compensated_mean(high(i,j,m), &
            recv_high(i,j,m,1),send_high(i,j,m,2),recv_high(i,j,m,2),ch(i,j,m))
        enddo
      enddo
    enddo
  end subroutine
end module chemistry_compensation
