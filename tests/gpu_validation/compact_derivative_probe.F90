program compact_derivative_probe
  use cudafor
  use ieee_arithmetic, only: ieee_is_finite
  use constdef, only: num1d3,num1d60,num7d9,num1d36
  use compact_derivative_gpu
  use compact_line_solver, only: compact_matrix,compact_reference_factor, &
    factor_compact_reference,solve_compact_reference,compact_residual
  implicit none
  type(compact_derivative_plan) :: plan
  type(compact_matrix) :: matrix
  type(compact_reference_factor) :: factor
  integer,parameter :: hm=5,nrhs=4,dims(4)=[16,32,64,128]
  real(8),allocatable :: input(:,:),rhs(:,:),actual_rhs(:,:),actual(:,:),reference(:,:),analytic(:,:)
  real(8) :: h,x,abs_error,scaled_error,residual,constant_error,linear_error,cubic_error
  real(8) :: boundary_error,interior_error,full_error,previous(4),order
  integer :: unit,di,dim,ntype,first,last,n,j,r,device,istat
  character(1024) :: path,argument

  call get_command_argument(1,path)
  if(len_trim(path)==0) error stop 'usage: compact_derivative_probe OUTPUT.csv [DEVICE]'
  device=0
  call get_command_argument(2,argument)
  if(len_trim(argument)>0) read(argument,*) device
  istat=cudaSetDevice(device)
  if(istat/=cudaSuccess) error stop 'cudaSetDevice failed'
  open(newunit=unit,file=trim(path),status='new',action='write')
  write(unit,'(A)') 'dim,ntype,abs_error,scaled_error,residual,constant_error,linear_error,cubic_error,'// &
    'boundary_error,interior_error,full_error,full_order'
  previous=0.d0
  do di=1,size(dims)
    dim=dims(di)
    h=1.d0/dim
    do ntype=1,4
      call prepare_compact_derivative_gpu(dim,hm,ntype,nrhs,plan)
      first=plan%first
      last=plan%last
      n=last-first+1
      allocate(input(-hm:dim+hm,nrhs),rhs(n,nrhs),actual_rhs(n,nrhs), &
        actual(n,nrhs),reference(n,nrhs),analytic(0:dim,nrhs))
      do j=-hm,dim+hm
        x=j*h
        input(j,:)=[1.d0,x,x**3,exp(2.d0*x)]
      enddo
      do j=0,dim
        x=j*h
        analytic(j,:)=[0.d0,1.d0,3.d0*x*x,2.d0*exp(2.d0*x)]
      enddo
      matrix%n=n
      allocate(matrix%lower(n),matrix%diagonal(n),matrix%upper(n))
      matrix%lower=num1d3; matrix%upper=num1d3; matrix%diagonal=1.d0
      matrix%lower(1)=0.d0; matrix%upper(n)=0.d0
      matrix%upper(1)=0.d0; matrix%lower(n)=0.d0
      if(ntype==1.or.ntype==4) then
        matrix%upper(1)=2.d0
        matrix%lower(2)=0.25d0; matrix%upper(2)=0.25d0
      endif
      if(ntype==2.or.ntype==4) then
        matrix%lower(n)=2.d0
        matrix%lower(n-1)=0.25d0; matrix%upper(n-1)=0.25d0
      endif
      do j=first,last
        r=j-first+1
        if((ntype==1.or.ntype==4).and.j==first) then
          rhs(r,:)=-2.5d0*input(j,:)+2.d0*input(j+1,:)+0.5d0*input(j+2,:)
        elseif((ntype==2.or.ntype==4).and.j==last) then
          rhs(r,:)=2.5d0*input(j,:)-2.d0*input(j-1,:)-0.5d0*input(j-2,:)
        elseif(((ntype==1.or.ntype==4).and.j==first+1).or. &
               ((ntype==2.or.ntype==4).and.j==last-1)) then
          rhs(r,:)=0.75d0*(input(j+1,:)-input(j-1,:))
        elseif(j==first.or.j==last) then
          rhs(r,:)=0.75d0*(input(j+1,:)-input(j-1,:))-0.15d0*(input(j+2,:)-input(j-2,:))+ &
            num1d60*(input(j+3,:)-input(j-3,:))
        else
          rhs(r,:)=num7d9*(input(j+1,:)-input(j-1,:))+num1d36*(input(j+2,:)-input(j-2,:))
        endif
      enddo
      call factor_compact_reference(matrix,factor)
      call solve_compact_reference(factor,rhs,reference)
      plan%input=input
      call apply_compact_derivative_gpu(plan,nrhs)
      actual=plan%solution
      actual_rhs=plan%rhs
      if(.not.all(ieee_is_finite(actual))) error stop 'nonfinite derivative'
      abs_error=maxval(abs(actual-reference))
      scaled_error=abs_error/max(1.d0,maxval(abs(reference)))
      residual=compact_residual(matrix,actual,rhs)
      if(maxval(abs(actual_rhs-rhs))>2.d-14.or.scaled_error>2.d-10.or.residual>1.d-12) &
        error stop 'derivative reference/residual gate failed'
      actual=actual/h
      constant_error=maxval(abs(actual(1-first:dim-first+1,1)))
      linear_error=maxval(abs(actual(1-first:dim-first+1,2)-analytic(:,2)))
      cubic_error=maxval(abs(actual(1-first:dim-first+1,3)-analytic(:,3)))
      if(max(constant_error,linear_error,cubic_error)>2.d-10) error stop 'polynomial derivative failed'
      boundary_error=max(abs(actual(1-first,4)-analytic(0,4)),abs(actual(dim-first+1,4)-analytic(dim,4)))
      interior_error=maxval(abs(actual(dim/4-first+1:3*dim/4-first+1,4)-analytic(dim/4:3*dim/4,4)))
      full_error=maxval(abs(actual(1-first:dim-first+1,4)-analytic(:,4)))
      order=0.d0
      if(di>1) order=log(previous(ntype)/full_error)/log(2.d0)
      previous(ntype)=full_error
      write(unit,'(I0,A,I0,10(A,ES24.16))') dim,',',ntype,',',abs_error,',',scaled_error,',',residual, &
        ',',constant_error,',',linear_error,',',cubic_error,',',boundary_error,',',interior_error,',',full_error,',',order
      if(di==size(dims)) then
        if(ntype==3.and.order<5.5d0) error stop 'interface sixth-order convergence failed'
        if(ntype/=3.and.order<2.8d0) error stop 'physical closure third-order convergence failed'
      endif
      call release_compact_derivative_gpu(plan)
      deallocate(input,rhs,actual_rhs,actual,reference,analytic,matrix%lower,matrix%diagonal,matrix%upper)
    enddo
  enddo
  close(unit)
  write(*,'(A,I0,A)') 'PASS: 16 compact derivative cases on GPU ',device,'; independent LU and analytic checks'
end program
