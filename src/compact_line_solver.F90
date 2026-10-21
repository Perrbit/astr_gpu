module compact_line_solver
  use astr_precision, only: real_kind
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: compact_matrix,compact_reference_factor,compact_pcr_factors,compact_block_factors
  public :: factor_compact_reference,solve_compact_reference,compact_matvec,compact_residual
  public :: prepare_compact_blocks,compact_endpoint_responses

  type :: compact_matrix
    integer :: n=0
    logical :: cyclic=.false.
    real(real_kind),allocatable :: lower(:),diagonal(:),upper(:)
  end type

  ! Whole-matrix pivoted LU is an independent small-component reference.
  type :: compact_reference_factor
    real(real_kind),allocatable :: lu(:,:)
    integer,allocatable :: pivot(:)
  end type

  type :: compact_pcr_factors
    integer :: n=0,levels=0
    real(real_kind),allocatable :: initial_inverse(:),left(:,:),right(:,:),inverse(:,:)
  end type

  type :: compact_block_factors
    integer :: n=0,tiles=0,reduced_n=0,tile_capacity=0,max_levels=0
    integer,allocatable :: first(:),last(:),owner(:),levels(:)
    real(real_kind),allocatable :: initial_inverse(:,:),left(:,:,:),right(:,:,:),inverse(:,:,:)
    real(real_kind),allocatable :: response_left(:),response_right(:),cyclic_response(:,:)
    real(real_kind) :: correction_inverse(2,2)=0.e0_real_kind
    type(compact_pcr_factors) :: reduced
  end type
contains
  subroutine compact_endpoint_responses(matrix,edge,response)
    type(compact_matrix),intent(in) :: matrix
    real(real_kind),intent(in) :: edge(2)
    real(real_kind),intent(out) :: response(matrix%n,2)
    type(compact_pcr_factors) :: factor
    real(real_kind) :: rhs(matrix%n)
    if(matrix%cyclic) error stop 'endpoint response requires a noncyclic local core'
    call prepare_pcr(matrix%lower,matrix%diagonal,matrix%upper,factor)
    rhs=0.e0_real_kind
    rhs(1)=edge(1)
    call apply_preprocessed_pcr(factor,rhs,response(:,1))
    rhs=0.e0_real_kind
    rhs(matrix%n)=edge(2)
    call apply_preprocessed_pcr(factor,rhs,response(:,2))
  end subroutine

  subroutine compact_matvec(matrix,x,b)
    type(compact_matrix),intent(in) :: matrix
    real(real_kind),intent(in) :: x(:,:)
    real(real_kind),intent(out) :: b(:,:)
    integer :: i,n
    n=matrix%n
    do i=1,n
      b(i,:)=matrix%diagonal(i)*x(i,:)
      if(i>1) b(i,:)=b(i,:)+matrix%lower(i)*x(i-1,:)
      if(i<n) b(i,:)=b(i,:)+matrix%upper(i)*x(i+1,:)
    enddo
    if(matrix%cyclic) then
      b(1,:)=b(1,:)+matrix%lower(1)*x(n,:)
      b(n,:)=b(n,:)+matrix%upper(n)*x(1,:)
    endif
  end subroutine

  function compact_residual(matrix,x,b) result(residual)
    type(compact_matrix),intent(in) :: matrix
    real(real_kind),intent(in) :: x(:,:),b(:,:)
    real(real_kind) :: residual,anorm,denominator,rnorm
    real(real_kind),allocatable :: work(:,:),rowsum(:)
    integer :: j,n
    n=matrix%n
    allocate(work(n,size(x,2)),rowsum(n))
    call compact_matvec(matrix,x,work)
    rowsum=abs(matrix%diagonal)
    rowsum(2:n)=rowsum(2:n)+abs(matrix%lower(2:n))
    rowsum(1:n-1)=rowsum(1:n-1)+abs(matrix%upper(1:n-1))
    if(matrix%cyclic) then
      if(n==2) then
        rowsum(1)=abs(matrix%diagonal(1))+abs(matrix%upper(1)+matrix%lower(1))
        rowsum(2)=abs(matrix%diagonal(2))+abs(matrix%lower(2)+matrix%upper(2))
      else
        rowsum(1)=rowsum(1)+abs(matrix%lower(1))
        rowsum(n)=rowsum(n)+abs(matrix%upper(n))
      endif
    endif
    anorm=maxval(rowsum)
    residual=0.e0_real_kind
    do j=1,size(x,2)
      rnorm=maxval(abs(work(:,j)-b(:,j)))
      denominator=anorm*maxval(abs(x(:,j)))+maxval(abs(b(:,j)))
      if(denominator==0.e0_real_kind) then
        if(rnorm/=0.e0_real_kind) residual=huge(1.e0_real_kind)
      else
        residual=max(residual,rnorm/denominator)
      endif
    enddo
  end function

  subroutine factor_compact_reference(matrix,factor)
    type(compact_matrix),intent(in) :: matrix
    type(compact_reference_factor),intent(out) :: factor
    real(real_kind),allocatable :: row(:)
    integer :: n,i,j,k,p
    n=matrix%n
    if(n<2) error stop 'compact reference requires at least two unknowns'
    allocate(factor%lu(n,n),factor%pivot(n),row(n))
    factor%lu=0.e0_real_kind
    do i=1,n
      factor%lu(i,i)=matrix%diagonal(i)
      if(i>1) factor%lu(i,i-1)=matrix%lower(i)
      if(i<n) factor%lu(i,i+1)=matrix%upper(i)
    enddo
    if(matrix%cyclic) then
      factor%lu(1,n)=factor%lu(1,n)+matrix%lower(1)
      factor%lu(n,1)=factor%lu(n,1)+matrix%upper(n)
    endif
    do k=1,n
      p=k-1+maxloc(abs(factor%lu(k:n,k)),dim=1)
      factor%pivot(k)=p
      if(factor%lu(p,k)==0.e0_real_kind) error stop 'singular compact reference matrix'
      if(p/=k) then
        row=factor%lu(k,:)
        factor%lu(k,:)=factor%lu(p,:)
        factor%lu(p,:)=row
      endif
      factor%lu(k+1:n,k)=factor%lu(k+1:n,k)/factor%lu(k,k)
      do j=k+1,n
        do i=k+1,n
          factor%lu(i,j)=factor%lu(i,j)-factor%lu(i,k)*factor%lu(k,j)
        enddo
      enddo
    enddo
    if(.not.all(ieee_is_finite(factor%lu))) error stop 'nonfinite compact reference factor'
  end subroutine

  subroutine solve_compact_reference(factor,b,x)
    type(compact_reference_factor),intent(in) :: factor
    real(real_kind),intent(in) :: b(:,:)
    real(real_kind),intent(out) :: x(:,:)
    real(real_kind) :: value
    integer :: n,i,j,k,p
    n=size(factor%lu,1)
    x=b
    do k=1,n
      p=factor%pivot(k)
      if(p==k) cycle
      do j=1,size(b,2)
        value=x(k,j)
        x(k,j)=x(p,j)
        x(p,j)=value
      enddo
    enddo
    do k=1,n
      do i=k+1,n
        x(i,:)=x(i,:)-factor%lu(i,k)*x(k,:)
      enddo
    enddo
    do k=n,1,-1
      x(k,:)=x(k,:)/factor%lu(k,k)
      do i=1,k-1
        x(i,:)=x(i,:)-factor%lu(i,k)*x(k,:)
      enddo
    enddo
  end subroutine

  subroutine prepare_pcr(lower,diagonal,upper,factor)
    real(real_kind),intent(in) :: lower(:),diagonal(:),upper(:)
    type(compact_pcr_factors),intent(out) :: factor
    real(real_kind),allocatable :: a(:),c(:),next_a(:),next_c(:)
    real(real_kind) :: divisor,al,cu
    integer :: n,level,stride,i
    n=size(diagonal)
    factor%n=n
    factor%levels=0
    stride=1
    do while(stride<n)
      factor%levels=factor%levels+1
      stride=stride*2
    enddo
    if(any(diagonal==0.e0_real_kind)) error stop 'zero compact PCR initial diagonal'
    allocate(factor%initial_inverse(n),factor%left(n,factor%levels), &
      factor%right(n,factor%levels),factor%inverse(n,factor%levels))
    allocate(a(n),c(n),next_a(n),next_c(n))
    factor%initial_inverse=1.e0_real_kind/diagonal
    a=lower*factor%initial_inverse
    c=upper*factor%initial_inverse
    a(1)=0.e0_real_kind
    c(n)=0.e0_real_kind
    stride=1
    do level=1,factor%levels
      factor%left(:,level)=a
      factor%right(:,level)=c
      next_a=0.e0_real_kind
      next_c=0.e0_real_kind
      do i=1,n
        al=0.e0_real_kind
        cu=0.e0_real_kind
        divisor=1.e0_real_kind
        if(i>stride) then
          divisor=divisor-a(i)*c(i-stride)
          al=-a(i)*a(i-stride)
        endif
        if(i+stride<=n) then
          divisor=divisor-c(i)*a(i+stride)
          cu=-c(i)*c(i+stride)
        endif
        if(divisor==0.e0_real_kind.or..not.ieee_is_finite(divisor)) &
          error stop 'invalid compact PCR reduction pivot'
        factor%inverse(i,level)=1.e0_real_kind/divisor
        next_a(i)=al*factor%inverse(i,level)
        next_c(i)=cu*factor%inverse(i,level)
      enddo
      a=next_a
      c=next_c
      stride=stride*2
    enddo
  end subroutine

  subroutine apply_preprocessed_pcr(factor,b,x)
    type(compact_pcr_factors),intent(in) :: factor
    real(real_kind),intent(in) :: b(:)
    real(real_kind),intent(out) :: x(:)
    real(real_kind),allocatable :: next(:)
    real(real_kind) :: value
    integer :: i,level,stride,n
    n=factor%n
    allocate(next(n))
    x=b*factor%initial_inverse
    stride=1
    do level=1,factor%levels
      do i=1,n
        value=x(i)
        if(i>stride) value=value-factor%left(i,level)*x(i-stride)
        if(i+stride<=n) value=value-factor%right(i,level)*x(i+stride)
        next(i)=value*factor%inverse(i,level)
      enddo
      x=next
      stride=stride*2
    enddo
  end subroutine

  subroutine prepare_compact_blocks(matrix,tile_capacity,blocks)
    type(compact_matrix),intent(in) :: matrix
    integer,intent(in) :: tile_capacity
    type(compact_block_factors),intent(out) :: blocks
    type(compact_pcr_factors) :: local
    real(real_kind),allocatable :: a(:),b(:),c(:),v(:),w(:),rhs(:)
    real(real_kind),allocatable :: ra(:),rb(:),rc(:)
    real(real_kind) :: h(2,2),determinant
    integer :: n,t,first,last,m,length,base,extra,k,stride
    n=matrix%n
    if(n<2.or.tile_capacity<4) error stop 'invalid compact line/tile extent'
    if(.not.all(ieee_is_finite(matrix%lower)).or..not.all(ieee_is_finite(matrix%diagonal)).or. &
       .not.all(ieee_is_finite(matrix%upper))) error stop 'nonfinite compact matrix'
    if(.not.matrix%cyclic) then
      if(matrix%lower(1)/=0.e0_real_kind.or.matrix%upper(n)/=0.e0_real_kind) &
        error stop 'noncyclic compact matrix has corner coefficients'
    endif
    blocks%n=n
    blocks%tile_capacity=tile_capacity
    blocks%tiles=(n+tile_capacity-1)/tile_capacity
    blocks%reduced_n=2*blocks%tiles
    blocks%max_levels=0
    stride=1
    do while(stride<tile_capacity)
      blocks%max_levels=blocks%max_levels+1
      stride=stride*2
    enddo
    allocate(blocks%first(blocks%tiles),blocks%last(blocks%tiles),blocks%levels(blocks%tiles), &
      blocks%owner(n),blocks%initial_inverse(tile_capacity,blocks%tiles), &
      blocks%left(tile_capacity,blocks%max_levels,blocks%tiles), &
      blocks%right(tile_capacity,blocks%max_levels,blocks%tiles), &
      blocks%inverse(tile_capacity,blocks%max_levels,blocks%tiles), &
      blocks%response_left(n),blocks%response_right(n))
    blocks%initial_inverse=0.e0_real_kind
    blocks%left=0.e0_real_kind
    blocks%right=0.e0_real_kind
    blocks%inverse=0.e0_real_kind
    blocks%response_left=0.e0_real_kind
    blocks%response_right=0.e0_real_kind
    allocate(ra(blocks%reduced_n),rb(blocks%reduced_n),rc(blocks%reduced_n))
    ra=0.e0_real_kind
    rb=0.e0_real_kind
    rc=0.e0_real_kind
    base=n/blocks%tiles
    extra=mod(n,blocks%tiles)
    first=1
    do t=1,blocks%tiles
      length=base
      if(t<=extra) length=length+1
      last=first+length-1
      m=length-2
      blocks%first(t)=first
      blocks%last(t)=last
      blocks%owner(first:last)=t
      blocks%levels(t)=0
      k=2*t-1
      ra(k)=matrix%lower(first)
      rb(k)=matrix%diagonal(first)
      rc(k)=matrix%upper(first)
      ra(k+1)=matrix%lower(last)
      rb(k+1)=matrix%diagonal(last)
      rc(k+1)=matrix%upper(last)
      if(m>0) then
        allocate(a(m),b(m),c(m),v(m),w(m),rhs(m))
        a=matrix%lower(first+1:last-1)
        b=matrix%diagonal(first+1:last-1)
        c=matrix%upper(first+1:last-1)
        call prepare_pcr(a,b,c,local)
        blocks%levels(t)=local%levels
        blocks%initial_inverse(1:m,t)=local%initial_inverse
        blocks%left(1:m,1:local%levels,t)=local%left
        blocks%right(1:m,1:local%levels,t)=local%right
        blocks%inverse(1:m,1:local%levels,t)=local%inverse
        rhs=0.e0_real_kind
        rhs(1)=a(1)
        call apply_preprocessed_pcr(local,rhs,v)
        rhs=0.e0_real_kind
        rhs(m)=c(m)
        call apply_preprocessed_pcr(local,rhs,w)
        blocks%response_left(first+1:last-1)=v
        blocks%response_right(first+1:last-1)=w
        rb(k)=rb(k)-matrix%upper(first)*v(1)
        rc(k)=-matrix%upper(first)*w(1)
        ra(k+1)=-matrix%lower(last)*v(m)
        rb(k+1)=rb(k+1)-matrix%lower(last)*w(m)
        deallocate(a,b,c,v,w,rhs)
      endif
      first=last+1
    enddo
    ra(1)=0.e0_real_kind
    rc(blocks%reduced_n)=0.e0_real_kind
    call prepare_pcr(ra,rb,rc,blocks%reduced)
    allocate(blocks%cyclic_response(blocks%reduced_n,2),rhs(blocks%reduced_n))
    blocks%cyclic_response=0.e0_real_kind
    blocks%correction_inverse=0.e0_real_kind
    blocks%correction_inverse(1,1)=1.e0_real_kind
    blocks%correction_inverse(2,2)=1.e0_real_kind
    if(matrix%cyclic) then
      ! Restore both real corner entries by a rank-two Woodbury correction.
      rhs=0.e0_real_kind
      rhs(1)=matrix%lower(1)
      call apply_preprocessed_pcr(blocks%reduced,rhs,blocks%cyclic_response(:,1))
      rhs=0.e0_real_kind
      rhs(blocks%reduced_n)=matrix%upper(n)
      call apply_preprocessed_pcr(blocks%reduced,rhs,blocks%cyclic_response(:,2))
      h(1,1)=1.e0_real_kind+blocks%cyclic_response(blocks%reduced_n,1)
      h(1,2)=blocks%cyclic_response(blocks%reduced_n,2)
      h(2,1)=blocks%cyclic_response(1,1)
      h(2,2)=1.e0_real_kind+blocks%cyclic_response(1,2)
      determinant=h(1,1)*h(2,2)-h(1,2)*h(2,1)
      if(determinant==0.e0_real_kind.or..not.ieee_is_finite(determinant)) &
        error stop 'invalid compact cyclic correction'
      blocks%correction_inverse(1,1)=h(2,2)/determinant
      blocks%correction_inverse(1,2)=-h(1,2)/determinant
      blocks%correction_inverse(2,1)=-h(2,1)/determinant
      blocks%correction_inverse(2,2)=h(1,1)/determinant
    endif
  end subroutine
end module compact_line_solver
