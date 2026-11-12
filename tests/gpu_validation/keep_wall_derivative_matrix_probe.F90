! Extract production derivative matrices; do not advance a flow solution.
program keep_wall_derivative_matrix_probe
  use astr_precision, only: real_kind
  use commvar, only: hm
  use derivative, only: ddfc
  use boundary_high_order, only: high_order_boundary_enabled,high_order_wall_enabled
  use compact_operator, only: compact_line_partition,compact_derivative_core,compact_derivative_rhs
  use compact_line_solver, only: compact_matrix,compact_reference_factor, &
    factor_compact_reference,solve_compact_reference,compact_residual
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  type(compact_line_partition) :: partition
  type(compact_matrix) :: matrix
  type(compact_reference_factor) :: factor
  real(real_kind),allocatable :: basis(:,:),rhs(:,:),explicit_d(:,:),compact_d(:,:),high_order_d(:,:)
  real(real_kind) :: edge(2),residual
  integer :: n,i,j,unit
  character(1024) :: argument,path
  call get_command_argument(1,argument)
  read(argument,*) n
  call get_command_argument(2,path)
  allocate(basis(-hm:n+hm,n+1),rhs(n+1,n+1),explicit_d(0:n,n+1),compact_d(n+1,n+1),high_order_d(0:n,n+1))
  high_order_boundary_enabled=.false.;high_order_wall_enabled=.false.
  basis=0.e0_real_kind
  do j=1,n+1
    basis(j-1,j)=1.e0_real_kind
    explicit_d(:,j)=ddfc(basis(:,j),'643e',4,n)
  enddo
  partition%intervals=n;partition%global_n=n+1;partition%owned=n+1
  call compact_derivative_core(partition,matrix,edge)
  call compact_derivative_rhs(partition,n,hm,basis,rhs)
  call factor_compact_reference(matrix,factor)
  call solve_compact_reference(factor,rhs,compact_d)
  residual=compact_residual(matrix,compact_d,rhs)
  high_order_boundary_enabled=.true.
  do j=1,n+1
    high_order_d(:,j)=ddfc(basis(:,j),'643e',4,n)
  enddo
  high_order_boundary_enabled=.false.
  if(.not.all(ieee_is_finite(compact_d)).or..not.all(ieee_is_finite(explicit_d))) &
    error stop 'nonfinite production derivative matrix'
  if(residual>2.e-10_real_kind) error stop 'compact matrix extraction residual'
  if(maxval(abs(sum(explicit_d,dim=2)))>2.e-10_real_kind.or. &
     maxval(abs(sum(compact_d,dim=2)))>2.e-10_real_kind) error stop 'constant derivative is nonzero'
  open(newunit=unit,file=trim(path),status='new',action='write')
  do j=1,n+1
    do i=0,n
      write(unit,'(2I6,3ES26.17)') i,j-1,explicit_d(i,j),compact_d(i+1,j),high_order_d(i,j)
    enddo
  enddo
  close(unit)
  write(*,'(A,I0,A,ES24.16)') 'KEEP_WALL_MATRIX n=',n,' residual=',residual
end program keep_wall_derivative_matrix_probe
