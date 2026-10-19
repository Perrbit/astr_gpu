module compact_operator
  use mpi
  use iso_fortran_env, only: real64
  use compact_line_solver, only: compact_matrix,compact_reference_factor, &
    factor_compact_reference,solve_compact_reference
  implicit none
  private
  public :: compact_line_partition,partition_compact_line,compact_direction_communicator
  public :: periodic_compact_core,compact_cpu_plan,prepare_compact_cpu,solve_compact_cpu,release_compact_cpu
  public :: compact_derivative_core,compact_derivative_rhs
  public :: compact_filter_core,compact_filter_rhs_global

  type :: compact_line_partition
    integer :: intervals=0,global_n=0,first=0,local_intervals=0,owned=0,rank=0,ranks=1
    logical :: periodic=.false.
  end type
  type :: compact_cpu_plan
    integer :: comm=MPI_COMM_NULL,n=0,rank=0,ranks=1,capacity=0
    type(compact_reference_factor) :: interface_factor
    real(real64),allocatable :: inverse(:),multiplier(:),upper(:),response(:,:)
    real(real64) :: edge_inner(2)=0.d0
    real(real64),allocatable :: interior(:,:),endpoints(:,:),gathered(:),global_rhs(:,:),global_solution(:,:)
  end type
contains
  subroutine compact_filter_core(partition,alpha,matrix,edge)
    type(compact_line_partition),intent(in) :: partition
    real(real64),intent(in) :: alpha
    type(compact_matrix),intent(out) :: matrix
    real(real64),intent(out) :: edge(2)
    integer :: n
    n=partition%owned
    call periodic_compact_core(n,alpha,matrix,edge)
    if(partition%periodic) return
    if(partition%first==0) then
      edge(1)=0.d0
      matrix%upper(1)=0.98d0
    endif
    if(partition%first+n==partition%global_n) then
      edge(2)=0.d0
      matrix%lower(n)=0.98d0
    endif
  end subroutine

  subroutine compact_filter_rhs_global(partition,dim,halo,f,c6,c8,c10,boundary,rhs)
    type(compact_line_partition),intent(in) :: partition
    integer,intent(in) :: dim,halo
    real(real64),intent(in) :: f(-halo:,:),c6(0:3),c8(0:4),c10(0:5),boundary(0:2,0:6)
    real(real64),intent(out) :: rhs(:,:)
    integer :: j,g,k,distance,total
    total=partition%intervals
    do j=0,partition%owned-1
      g=partition%first+j
      if(.not.partition%periodic.and.g<=2) then
        rhs(j+1,:)=0.d0
        do k=0,6
          rhs(j+1,:)=rhs(j+1,:)+boundary(g,k)*f(j-g+k,:)
        enddo
      elseif(.not.partition%periodic.and.total-g<=2) then
        distance=total-g
        rhs(j+1,:)=0.d0
        do k=0,6
          rhs(j+1,:)=rhs(j+1,:)+boundary(distance,k)*f(j+distance-k,:)
        enddo
      elseif(.not.partition%periodic.and.(g==3.or.total-g==3)) then
        rhs(j+1,:)=c6(0)*(f(j,:)+f(j,:))
        do k=1,3
          rhs(j+1,:)=rhs(j+1,:)+c6(k)*(f(j+k,:)+f(j-k,:))
        enddo
      elseif(.not.partition%periodic.and.(g==4.or.total-g==4)) then
        rhs(j+1,:)=c8(0)*(f(j,:)+f(j,:))
        do k=1,4
          rhs(j+1,:)=rhs(j+1,:)+c8(k)*(f(j+k,:)+f(j-k,:))
        enddo
      else
        rhs(j+1,:)=c10(0)*(f(j,:)+f(j,:))
        do k=1,5
          rhs(j+1,:)=rhs(j+1,:)+c10(k)*(f(j+k,:)+f(j-k,:))
        enddo
      endif
    enddo
  end subroutine

  subroutine compact_derivative_core(partition,matrix,edge)
    type(compact_line_partition),intent(in) :: partition
    type(compact_matrix),intent(out) :: matrix
    real(real64),intent(out) :: edge(2)
    integer :: row,g,n
    n=partition%owned
    call periodic_compact_core(n,1.d0/3.d0,matrix,edge)
    if(partition%periodic) return
    matrix%lower(1)=1.d0/3.d0
    matrix%upper(n)=1.d0/3.d0
    do row=1,n
      g=partition%first+row-1
      if(g==0) matrix%upper(row)=2.d0
      if(g==partition%intervals) matrix%lower(row)=2.d0
      if(g==1.or.g==partition%intervals-1) then
        matrix%lower(row)=0.25d0
        matrix%upper(row)=0.25d0
      endif
    enddo
    ! Retain every MPI coupling; only the two physical ends terminate a line.
    edge=[matrix%lower(1),matrix%upper(n)]
    if(partition%first==0) edge(1)=0.d0
    if(partition%first+n==partition%global_n) edge(2)=0.d0
    matrix%lower(1)=0.d0
    matrix%upper(n)=0.d0
  end subroutine

  subroutine compact_derivative_rhs(partition,dim,halo,f,rhs)
    type(compact_line_partition),intent(in) :: partition
    integer,intent(in) :: dim,halo
    real(real64),intent(in) :: f(-halo:,:)
    real(real64),intent(out) :: rhs(:,:)
    integer :: j,g
    real(real64),parameter :: num7d9=7.d0/9.d0,num1d36=1.d0/36.d0
    do j=0,partition%owned-1
      g=partition%first+j
      if(.not.partition%periodic.and.g==0) then
        ! Equivalent endpoint closure without cancelling a large constant state.
        rhs(j+1,:)=2.d0*(f(j+1,:)-f(j,:))+0.5d0*(f(j+2,:)-f(j,:))
      elseif(.not.partition%periodic.and.g==partition%intervals) then
        rhs(j+1,:)=2.d0*(f(j,:)-f(j-1,:))+0.5d0*(f(j,:)-f(j-2,:))
      elseif(.not.partition%periodic.and.(g==1.or.g==partition%intervals-1)) then
        rhs(j+1,:)=0.75d0*(f(j+1,:)-f(j-1,:))
      else
        rhs(j+1,:)=num7d9*(f(j+1,:)-f(j-1,:))+num1d36*(f(j+2,:)-f(j-2,:))
      endif
    enddo
  end subroutine

  subroutine periodic_compact_core(n,alpha,matrix,edge)
    integer,intent(in) :: n
    real(real64),intent(in) :: alpha
    type(compact_matrix),intent(out) :: matrix
    real(real64),intent(out) :: edge(2)
    matrix%n=n
    allocate(matrix%lower(n),matrix%diagonal(n),matrix%upper(n))
    matrix%lower=alpha
    matrix%diagonal=1.d0
    matrix%upper=alpha
    matrix%lower(1)=0.d0
    matrix%upper(n)=0.d0
    edge=alpha
  end subroutine

  subroutine interior_direct_solve(plan,values,nrhs)
    type(compact_cpu_plan),intent(in) :: plan
    integer,intent(in) :: nrhs
    real(real64),intent(inout) :: values(:,:)
    integer :: i,m
    m=plan%n-2
    do i=2,m
      values(i,1:nrhs)=values(i,1:nrhs)-plan%multiplier(i)*values(i-1,1:nrhs)
    enddo
    if(m>0) values(m,1:nrhs)=values(m,1:nrhs)*plan%inverse(m)
    do i=m-1,1,-1
      values(i,1:nrhs)=(values(i,1:nrhs)-plan%upper(i)*values(i+1,1:nrhs))*plan%inverse(i)
    enddo
  end subroutine

  subroutine prepare_compact_cpu(matrix,edge,partition,comm,capacity,plan)
    type(compact_matrix),intent(in) :: matrix
    real(real64),intent(in) :: edge(2)
    type(compact_line_partition),intent(in) :: partition
    integer,intent(in) :: comm,capacity
    type(compact_cpu_plan),intent(inout) :: plan
    type(compact_matrix) :: reduced
    real(real64),allocatable :: coefficients(:,:)
    real(real64) :: row(6),pivot
    integer :: n,m,i,p,k,ierr
    if(plan%n/=0) call release_compact_cpu(plan)
    n=matrix%n
    m=n-2
    if(n<2.or.capacity<1) error stop 'invalid compact CPU plan extent'
    call MPI_Comm_dup(comm,plan%comm,ierr)
    call MPI_Comm_rank(plan%comm,plan%rank,ierr)
    call MPI_Comm_size(plan%comm,plan%ranks,ierr)
    if(n/=partition%owned.or.plan%rank/=partition%rank.or.plan%ranks/=partition%ranks) &
      error stop 'compact CPU partition mismatch'
    plan%n=n
    plan%capacity=capacity
    allocate(plan%inverse(m),plan%multiplier(m),plan%upper(m),plan%response(m,2), &
      plan%interior(m,capacity),plan%endpoints(2,capacity),plan%gathered(2*capacity*plan%ranks), &
      plan%global_rhs(2*plan%ranks,capacity),plan%global_solution(2*plan%ranks,capacity))
    plan%response=0.d0
    do i=1,m
      plan%multiplier(i)=0.d0
      pivot=matrix%diagonal(i+1)
      if(i>1) then
        plan%multiplier(i)=matrix%lower(i+1)*plan%inverse(i-1)
        pivot=pivot-plan%multiplier(i)*plan%upper(i-1)
      endif
      if(pivot==0.d0) error stop 'singular compact CPU interior pivot'
      plan%inverse(i)=1.d0/pivot
      plan%upper(i)=matrix%upper(i+1)
    enddo
    row=[edge(1),matrix%diagonal(1),matrix%upper(1),matrix%lower(n),matrix%diagonal(n),edge(2)]
    plan%edge_inner=[matrix%upper(1),matrix%lower(n)]
    if(m>0) then
      plan%response(1,1)=matrix%lower(2)
      plan%response(m,2)=matrix%upper(n-1)
      call interior_direct_solve(plan,plan%response,2)
      row(2)=row(2)-matrix%upper(1)*plan%response(1,1)
      row(3)=-matrix%upper(1)*plan%response(1,2)
      row(4)=-matrix%lower(n)*plan%response(m,1)
      row(5)=row(5)-matrix%lower(n)*plan%response(m,2)
    endif
    allocate(coefficients(6,plan%ranks))
    call MPI_Allgather(row,6,MPI_DOUBLE_PRECISION,coefficients,6,MPI_DOUBLE_PRECISION,plan%comm,ierr)
    reduced%n=2*plan%ranks
    reduced%cyclic=partition%periodic
    allocate(reduced%lower(reduced%n),reduced%diagonal(reduced%n),reduced%upper(reduced%n))
    do p=1,plan%ranks
      k=2*p-1
      reduced%lower(k)=coefficients(1,p)
      reduced%diagonal(k)=coefficients(2,p)
      reduced%upper(k)=coefficients(3,p)
      reduced%lower(k+1)=coefficients(4,p)
      reduced%diagonal(k+1)=coefficients(5,p)
      reduced%upper(k+1)=coefficients(6,p)
    enddo
    call factor_compact_reference(reduced,plan%interface_factor)
  end subroutine

  subroutine solve_compact_cpu(plan,rhs,solution,values,nrhs)
    type(compact_cpu_plan),intent(inout) :: plan
    integer,intent(in) :: nrhs
    real(real64),intent(in) :: rhs(:,:)
    real(real64),intent(out) :: solution(:,:),values(:,:)
    integer :: m,p,j,k,offset,ierr
    m=plan%n-2
    if(nrhs<1.or.nrhs>plan%capacity) error stop 'compact CPU RHS exceeds capacity'
    plan%interior(:,1:nrhs)=rhs(2:plan%n-1,1:nrhs)
    call interior_direct_solve(plan,plan%interior,nrhs)
    plan%endpoints(1,1:nrhs)=rhs(1,1:nrhs)
    plan%endpoints(2,1:nrhs)=rhs(plan%n,1:nrhs)
    if(m>0) then
      plan%endpoints(1,1:nrhs)=plan%endpoints(1,1:nrhs)-plan%edge_inner(1)*plan%interior(1,1:nrhs)
      plan%endpoints(2,1:nrhs)=plan%endpoints(2,1:nrhs)-plan%edge_inner(2)*plan%interior(m,1:nrhs)
    endif
    ! Only the two retained endpoint RHS values are gathered, never a whole line.
    call MPI_Allgather(plan%endpoints,2*nrhs,MPI_DOUBLE_PRECISION,plan%gathered,2*nrhs, &
      MPI_DOUBLE_PRECISION,plan%comm,ierr)
    do p=1,plan%ranks
      offset=2*nrhs*(p-1)
      do j=1,nrhs
        plan%global_rhs(2*p-1:2*p,j)=plan%gathered(offset+2*j-1:offset+2*j)
      enddo
    enddo
    call solve_compact_reference(plan%interface_factor,plan%global_rhs(:,1:nrhs), &
      plan%global_solution(:,1:nrhs))
    k=2*plan%rank+1
    solution(1,1:nrhs)=plan%global_solution(k,1:nrhs)
    solution(plan%n,1:nrhs)=plan%global_solution(k+1,1:nrhs)
    do p=1,nrhs
      solution(2:plan%n-1,p)=plan%interior(:,p)-plan%response(:,1)*solution(1,p) &
        -plan%response(:,2)*solution(plan%n,p)
    enddo
    values(1:plan%n,1:nrhs)=solution(:,1:nrhs)
    if(size(values,1)>plan%n) then
      k=2*mod(plan%rank+1,plan%ranks)+1
      values(plan%n+1,1:nrhs)=plan%global_solution(k,1:nrhs)
    endif
  end subroutine

  subroutine release_compact_cpu(plan)
    type(compact_cpu_plan),intent(inout) :: plan
    integer :: ierr
    if(plan%n==0) return
    deallocate(plan%inverse,plan%multiplier,plan%upper,plan%response,plan%interior, &
      plan%endpoints,plan%gathered,plan%global_rhs,plan%global_solution)
    deallocate(plan%interface_factor%lu,plan%interface_factor%pivot)
    call MPI_Comm_free(plan%comm,ierr)
    plan%n=0
  end subroutine

  subroutine partition_compact_line(intervals,rank,ranks,periodic,part)
    integer,intent(in) :: intervals,rank,ranks
    logical,intent(in) :: periodic
    type(compact_line_partition),intent(out) :: part
    integer :: quotient,remainder
    if(ranks<1.or.rank<0.or.rank>=ranks.or.intervals<2*ranks) &
      error stop 'compact partition needs at least two owned nodes per rank'
    quotient=intervals/ranks
    remainder=mod(intervals,ranks)
    part%intervals=intervals
    part%rank=rank
    part%ranks=ranks
    part%periodic=periodic
    part%global_n=intervals+1
    if(periodic) part%global_n=intervals
    part%first=rank*quotient+min(rank,remainder)
    part%local_intervals=quotient
    if(rank<remainder) part%local_intervals=part%local_intervals+1
    part%owned=part%local_intervals
    if(.not.periodic.and.rank==ranks-1) part%owned=part%owned+1
  end subroutine

  subroutine compact_direction_communicator(cart_comm,axis,line_comm)
    integer,intent(in) :: cart_comm,axis
    integer,intent(out) :: line_comm
    integer :: subcomm,ierr,coords(3),dims(3)
    logical :: remain(3),periods(3)
    if(axis<1.or.axis>3) error stop 'invalid compact line direction'
    call MPI_Cart_get(cart_comm,3,dims,periods,coords,ierr)
    remain=.false.
    remain(axis)=.true.
    call MPI_Cart_sub(cart_comm,remain,subcomm,ierr)
    ! Explicit coordinate order also covers reordered parent communicators.
    call MPI_Comm_split(subcomm,0,coords(axis),line_comm,ierr)
    call MPI_Comm_free(subcomm,ierr)
  end subroutine
end module
