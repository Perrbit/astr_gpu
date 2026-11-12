program keep_split_probe
  use astr_precision, only: real_kind
  use keep_operator
  use compact_operator
  use compact_line_solver, only: compact_matrix,compact_residual,compact_reference_factor, &
    factor_compact_reference,solve_compact_reference
  use mpi
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  type(keep_split_plan) :: split
  real(real_kind),parameter :: pi=acos(-1.e0_real_kind),gamma=1.4e0_real_kind
  real(real_kind) :: error(5,3),order(5)
  integer :: ierr,level,n
  logical :: periodic
  character(32) :: mode
  call MPI_Init(ierr)
  call get_command_argument(1,mode)
  periodic=trim(mode)/='--wall'
  call build_keep_split_plan(split)
  write(*,'(A,I0,A,I0)') 'KEEP split terms=',split%terms,' derivative_bases=',split%bases
  if(periodic) call check_pair_equivalence(.false.,.true.)
  call check_pair_equivalence(.true.,periodic)
  do level=1,3
    n=16*2**level
    if(.not.periodic) n=n+1
    call check_compact_accuracy(n,periodic,error(:,level))
    write(*,'(A,I0,A,5ES16.7)') 'KEEP compact n=',n,' errors=',error(:,level)
  enddo
  do level=1,2
    order=log(error(:,level)/error(:,level+1))/log(2.e0_real_kind)
    write(*,'(A,5F12.6)') 'KEEP compact component_orders=',order
    if(periodic.and.any(order<5.5e0_real_kind)) error stop 'compact split order below sixth-order asymptote'
    if(.not.periodic.and.any(order<2.5e0_real_kind)) error stop 'compact physical split lacks third-order convergence'
  enddo
  call MPI_Finalize(ierr)
  if(periodic) then
    print '(A)', 'KEEP_SPLIT_PASS explicit_pair compact_pair sixth_order'
  else
    print '(A)', 'KEEP_SPLIT_WALL_PASS physical_core pair_equivalence reference_LU local_endpoint_closure'
  endif
contains
  subroutine fill_state(n,varying_metric,periodic,rho,velocity,p,metric,variables)
    integer,intent(in) :: n
    logical,intent(in) :: varying_metric,periodic
    real(real_kind),intent(out) :: rho(n),velocity(n,3),p(n),metric(n,3),variables(n,9)
    real(real_kind) :: x
    integer :: i
    do i=1,n
      x=real(i-1,real_kind)*2.e0_real_kind*pi/real(merge(n,n-1,periodic),real_kind)
      rho(i)=1.e0_real_kind+0.1e0_real_kind*sin(x)
      p(i)=1.e0_real_kind+0.1e0_real_kind*cos(2.e0_real_kind*x)
      velocity(i,:)=[0.2e0_real_kind+0.1e0_real_kind*sin(2.e0_real_kind*x), &
        0.1e0_real_kind*cos(x),0.05e0_real_kind*sin(3.e0_real_kind*x)]
      metric(i,:)=[1.e0_real_kind,0.e0_real_kind,0.e0_real_kind]
      if(varying_metric) metric(i,:)=[1.e0_real_kind+0.05e0_real_kind*sin(x), &
        0.1e0_real_kind*cos(2.e0_real_kind*x),0.03e0_real_kind*sin(3.e0_real_kind*x)]
      call keep_split_variables(rho(i),velocity(i,:),p(i),metric(i,:),gamma,variables(i,:))
    enddo
  end subroutine

  subroutine derivative_matrix(n,compact,periodic,d)
    integer,intent(in) :: n
    logical,intent(in) :: compact,periodic
    real(real_kind),intent(out) :: d(n,n)
    type(compact_matrix) :: matrix
    type(compact_line_partition) :: partition
    type(compact_cpu_plan) :: plan
    type(compact_reference_factor) :: reference
    real(real_kind) :: input(-3:n+2,n),rhs(n,n),owned(n,n),edge(2),h
    real(real_kind),parameter :: a(3)=[0.75e0_real_kind,-0.15e0_real_kind,1.e0_real_kind/60.e0_real_kind]
    integer :: i,j,l
    h=2.e0_real_kind*pi/real(merge(n,n-1,periodic),real_kind)
    d=0.e0_real_kind
    if(.not.compact) then
      do i=1,n
        do l=1,3
          d(i,modulo(i-1+l,n)+1)=a(l)/h
          d(i,modulo(i-1-l,n)+1)=-a(l)/h
        enddo
      enddo
      return
    endif
    partition=compact_line_partition(intervals=merge(n,n-1,periodic),global_n=n,first=0,local_intervals=n-1, &
      owned=n,rank=0,ranks=1,periodic=periodic)
    input=0.e0_real_kind
    do j=1,n
      do i=-3,n+2
        if(periodic) then
          if(modulo(i,n)+1==j) input(i,j)=1.e0_real_kind
        else
          if(i==j-1) input(i,j)=1.e0_real_kind
        endif
      enddo
    enddo
    call compact_derivative_core(partition,matrix,edge)
    call prepare_compact_cpu(matrix,edge,partition,MPI_COMM_SELF,n,plan)
    call compact_derivative_rhs(partition,n-1,3,input,rhs)
    call solve_compact_cpu(plan,rhs,owned,d,n)
    d=d/h
    if(.not.periodic) then
      if(compact_residual(matrix,d,rhs/h)>2.e-10_real_kind) error stop 'physical compact matrix residual'
      call factor_compact_reference(matrix,reference)
      call solve_compact_reference(reference,rhs/h,owned)
      if(maxval(abs(d-owned)/max(1.e0_real_kind,abs(d),abs(owned)))>2.e-10_real_kind) &
        error stop 'physical compact PCR differs from independent pivoted LU'
      call release_compact_cpu(plan)
      return
    endif
    do i=1,n
      if(maxval(abs(d(i,:)+(d(modulo(i-2,n)+1,:)+d(modulo(i,n)+1,:))/3.e0_real_kind-rhs(i,:)/h)) / &
        max(1.e0_real_kind,maxval(abs(rhs(i,:)/h)))>2.e-10_real_kind) error stop 'compact matrix residual'
    enddo
    if(maxval(abs(d+transpose(d)))>2.e-12_real_kind) error stop 'periodic compact skew symmetry'
    if(maxval(abs(sum(d,dim=2)))>2.e-12_real_kind) error stop 'periodic compact constant derivative'
    call release_compact_cpu(plan)
  end subroutine

  subroutine split_apply(n,d,variables,result)
    integer,intent(in) :: n
    real(real_kind),intent(in) :: d(n,n),variables(n,9)
    real(real_kind),intent(out) :: result(n,5)
    real(real_kind) :: products(n,split%bases),derivatives(n,split%bases)
    integer :: i,b
    do b=1,split%bases
      do i=1,n
        products(i,b)=keep_monomial(variables(i,:),split%right(:,b))
      enddo
    enddo
    derivatives=matmul(d,products)
    do i=1,n
      call keep_split_accumulate(split,variables(i,:),derivatives(i,:),result(i,:))
    enddo
    if(.not.all(ieee_is_finite(result))) error stop 'nonfinite compact split result'
  end subroutine

  subroutine check_pair_equivalence(compact,periodic)
    logical,intent(in) :: compact,periodic
    integer,parameter :: n=32
    real(real_kind) :: rho(n),velocity(n,3),p(n),metric(n,3),variables(n,9),d(n,n)
    real(real_kind) :: result(n,5),reference(n,5),flux(5),difference
    integer :: i,j
    call fill_state(n,.true.,periodic,rho,velocity,p,metric,variables)
    call derivative_matrix(n,compact,periodic,d)
    call split_apply(n,d,variables,result)
    reference=0.e0_real_kind
    do i=1,n
      do j=1,n
        call keep_two_point_flux5(rho(i),velocity(i,:),p(i),metric(i,:), &
          rho(j),velocity(j,:),p(j),metric(j,:),gamma,flux)
        reference(i,:)=reference(i,:)+2.e0_real_kind*d(i,j)*flux
      enddo
    enddo
    difference=maxval(abs(reference-result)/max(1.e0_real_kind,abs(reference),abs(result)))
    write(*,'(A,L1,A,ES16.7)') 'KEEP split compact=',compact,' pair_equivalence=',difference
    if(difference>2.e-10_real_kind) error stop 'KEEP polynomial split differs from original pair flux'
    if(periodic.and.maxval(abs(sum(result,dim=1)))>2.e-10_real_kind) error stop 'KEEP split periodic conservation'
  end subroutine

  subroutine check_compact_accuracy(n,periodic,error)
    integer,intent(in) :: n
    logical,intent(in) :: periodic
    real(real_kind),intent(out) :: error(5)
    real(real_kind) :: rho(n),velocity(n,3),p(n),metric(n,3),variables(n,9),d(n,n),result(n,5)
    real(real_kind) :: x,dr,dp,dv(3),exact(5),enthalpy,dh
    real(real_kind) :: kinetic(n),pressure_gradient(n),internal(n),internal_convection(n),expected_internal(n),balance
    integer :: i
    call fill_state(n,.false.,periodic,rho,velocity,p,metric,variables)
    call derivative_matrix(n,.true.,periodic,d)
    call split_apply(n,d,variables,result)
    kinetic=sum(velocity*result(:,2:4),dim=2)-0.5e0_real_kind*sum(velocity**2,dim=2)*result(:,1)
    pressure_gradient=matmul(d,p)
    balance=sum(kinetic-velocity(:,1)*pressure_gradient)*2.e0_real_kind*pi/real(merge(n,n-1,periodic),real_kind)
    if(periodic) then
      write(*,'(A,ES16.7)') 'KEEP compact kinetic_pressure_balance=',balance
      if(abs(balance)>2.e-10_real_kind) error stop 'compact KEEP kinetic pressure exchange'
    else
      write(*,'(A,ES16.7)') 'KEEP physical unweighted kinetic_pressure_sum (diagnostic)=',balance
    endif
    internal=variables(:,5)
    internal_convection=0.25e0_real_kind*(matmul(d,rho*internal*velocity(:,1))+ &
      velocity(:,1)*matmul(d,rho*internal)+rho*internal*matmul(d,velocity(:,1))+ &
      internal*matmul(d,rho*velocity(:,1))+rho*velocity(:,1)*matmul(d,internal)+ &
      rho*matmul(d,internal*velocity(:,1))+internal*velocity(:,1)*matmul(d,rho))
    expected_internal=internal_convection+p*matmul(d,velocity(:,1))
    if(maxval(abs(result(:,5)-kinetic-expected_internal)/max(1.e0_real_kind,abs(expected_internal))) &
      >2.e-10_real_kind) error stop 'compact KEEP internal energy product relation'
    error=0.e0_real_kind
    do i=1,n
      x=real(i-1,real_kind)*2.e0_real_kind*pi/real(merge(n,n-1,periodic),real_kind)
      dr=0.1e0_real_kind*cos(x)
      dp=-0.2e0_real_kind*sin(2.e0_real_kind*x)
      dv=[0.2e0_real_kind*cos(2.e0_real_kind*x),-0.1e0_real_kind*sin(x), &
        0.15e0_real_kind*cos(3.e0_real_kind*x)]
      exact(1)=dr*velocity(i,1)+rho(i)*dv(1)
      exact(2:4)=exact(1)*velocity(i,:)+rho(i)*velocity(i,1)*dv
      exact(2)=exact(2)+dp
      enthalpy=gamma*p(i)/(gamma-1.e0_real_kind)+0.5e0_real_kind*rho(i)*sum(velocity(i,:)**2)
      dh=gamma*dp/(gamma-1.e0_real_kind)+0.5e0_real_kind*dr*sum(velocity(i,:)**2) &
        +rho(i)*dot_product(velocity(i,:),dv)
      exact(5)=dh*velocity(i,1)+enthalpy*dv(1)
      error=max(error,abs(result(i,:)-exact))
    enddo
  end subroutine
end program
