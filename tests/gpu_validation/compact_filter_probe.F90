program compact_filter_probe
  use cudafor
  use ieee_arithmetic, only: ieee_is_finite
  use commvar, only: hm,alfa_filter
  use commtype, only: compact_scheme
  use filter, only: filter_coefficient_cal,compact_filter_initiate,compact_filter_rhs,compact_filter,coef10i
  use compact_filter_gpu
  use compact_line_solver, only: compact_matrix,compact_reference_factor, &
    factor_compact_reference,solve_compact_reference,compact_residual
  implicit none
  integer,parameter :: dims(9)=[16,32,64,127,128,129,255,256,512],batches(3)=[1,5,11]
  type(compact_filter_plan) :: plan,scalar
  type(compact_scheme) :: afilter
  type(compact_matrix) :: matrix
  type(compact_reference_factor) :: factor
  integer :: unit,di,bi,dim,ntype,nrhs,first,last,n,i,j,k,device,istat,cases
  real(8),allocatable :: input(:,:),rhs(:,:),actual_rhs(:,:),actual(:,:),reference(:,:), &
    actual_filtered(:,:),cpu_filtered(:,:),scalar_filtered(:,:),expected_rhs(:)
  real(8) :: x,theta,transfer,abs_error,scaled_error,rhs_error,residual,projected_error,scalar_error
  real(8) :: constant_error,polynomial_error,fourier_error,value,device_peak
  character(1024) :: path,argument

  call get_command_argument(1,path)
  if(len_trim(path)==0) error stop 'usage: compact_filter_probe OUTPUT.csv [DEVICE [ORDER.csv]]'
  device=0
  call get_command_argument(2,argument)
  if(len_trim(argument)>0) read(argument,*) device
  istat=cudaSetDevice(device)
  if(istat/=cudaSuccess) error stop 'cudaSetDevice failed'
  alfa_filter=0.49d0
  call filter_coefficient_cal(alfa_filter,1.11d0,0.98d0)
  theta=0.5d0*acos(-1.d0)
  value=2.d0*coef10i(0)
  do k=1,5
    value=value+2.d0*coef10i(k)*cos(k*theta)
  enddo
  transfer=value/(1.d0+2.d0*alfa_filter*cos(theta))
  open(newunit=unit,file=trim(path),status='new',action='write')
  write(unit,'(A)') 'dim,ntype,nrhs,rhs_error,raw_abs,raw_scaled,residual,projected_abs,scalar_full_abs,'// &
    'constant_error,polynomial_error,fourier_interior_error,controlled_device_bytes'
  cases=0
  do di=1,size(dims)
    dim=dims(di)
    do ntype=1,4
      call compact_filter_initiate(afilter,ntype,dim,note='')
      first=afilter%first_node
      last=afilter%last_node
      n=last-first+1
      matrix%n=n
      allocate(matrix%lower(n),matrix%diagonal(n),matrix%upper(n))
      matrix%lower=afilter%a; matrix%upper=afilter%c; matrix%diagonal=1.d0
      matrix%lower(1)=0.d0; matrix%upper(n)=0.d0
      call factor_compact_reference(matrix,factor)
      do bi=1,size(batches)
        nrhs=batches(bi)
        call prepare_compact_filter_gpu(afilter,hm,nrhs,plan)
        call prepare_compact_filter_gpu(afilter,hm,1,scalar)
        allocate(input(-hm:dim+hm,nrhs),rhs(n,nrhs),actual_rhs(n,nrhs),actual(n,nrhs),reference(n,nrhs), &
          actual_filtered(0:dim,nrhs),cpu_filtered(0:dim,nrhs),scalar_filtered(0:dim,nrhs))
        do j=1,nrhs
          do i=-hm,dim+hm
            x=real(i,8)/dim
            select case(mod(j-1,5))
            case(0); input(i,j)=1.d0
            case(1); input(i,j)=x
            case(2); input(i,j)=x**3
            case(3); input(i,j)=cos(theta*i)
            case(4); input(i,j)=exp(x)
            end select
          enddo
          expected_rhs=compact_filter_rhs(afilter,input(:,j),dim,note='')
          rhs(:,j)=expected_rhs
          cpu_filtered(:,j)=compact_filter(afilter,input(:,j),dim,note='')
        enddo
        call solve_compact_reference(factor,rhs,reference)
        plan%input=input
        call apply_compact_filter_gpu(plan,nrhs)
        actual_rhs=plan%rhs
        actual=plan%solution
        actual_filtered=plan%filtered
        do j=1,nrhs
          scalar%input=input(:,j:j)
          call apply_compact_filter_gpu(scalar,1)
          scalar_filtered(:,j:j)=scalar%filtered
        enddo
        if(.not.all(ieee_is_finite(actual)).or..not.all(ieee_is_finite(actual_filtered))) &
          error stop 'nonfinite compact filter output'
        rhs_error=maxval(abs(actual_rhs-rhs))
        abs_error=maxval(abs(actual-reference))
        scaled_error=abs_error/max(1.d0,maxval(abs(reference)))
        residual=compact_residual(matrix,actual,rhs)
        projected_error=maxval(abs(actual_filtered-cpu_filtered))
        scalar_error=maxval(abs(actual_filtered-scalar_filtered))
        if(rhs_error>2.d-10.or.scaled_error>2.d-10.or.residual>1.d-12.or. &
           projected_error>2.d-10.or.scalar_error>2.d-10) error stop 'compact filter comparison failed'
        if(ntype==1.or.ntype==4) then
          if(any(actual_filtered(0,:)/=input(0,:))) error stop 'left physical endpoint changed'
        endif
        if(ntype==2.or.ntype==4) then
          if(any(actual_filtered(dim,:)/=input(dim,:))) error stop 'right physical endpoint changed'
        endif
        constant_error=maxval(abs(actual_filtered(:,1)-1.d0))
        polynomial_error=0.d0
        if(nrhs>=3) polynomial_error=maxval(abs(actual_filtered(:,2:3)-input(0:dim,2:3)))
        if(constant_error>2.d-10.or.polynomial_error>2.d-10) error stop 'constant/polynomial filter property failed'
        fourier_error=0.d0
        if(dim==512.and.nrhs>=4) then
          do i=dim/3,2*dim/3
            fourier_error=max(fourier_error,abs(actual_filtered(i,4)-transfer*cos(theta*i)))
          enddo
          if(fourier_error>2.d-10) error stop 'interior Fourier filter response failed'
        endif
        device_peak=real(plan%line%device_bytes+scalar%line%device_bytes,8)+ &
          8.d0*(nrhs+1)*(dim+2*hm+1+2*n+dim+1)+8.d0*2*(4+5+6+21+33)
        write(unit,'(3(I0,A),10(ES24.16,:,A))') dim,',',ntype,',',nrhs,',',rhs_error,',',abs_error,',', &
          scaled_error,',',residual,',',projected_error,',',scalar_error,',',constant_error,',', &
          polynomial_error,',',fourier_error,',',device_peak
        cases=cases+1
        call release_compact_filter_gpu(plan)
        call release_compact_filter_gpu(scalar)
        deallocate(input,rhs,actual_rhs,actual,reference,actual_filtered,cpu_filtered,scalar_filtered)
      enddo
      deallocate(matrix%lower,matrix%diagonal,matrix%upper,afilter%a,afilter%c,afilter%ac)
    enddo
  enddo
  close(unit)
  write(*,'(A,I0,A,I0)') 'PASS: ',cases,' compact filter component cases on GPU ',device
  call get_command_argument(3,argument)
  if(len_trim(argument)>0) call check_smooth_orders(trim(argument))
contains
  subroutine check_smooth_orders(output_path)
    character(*),intent(in) :: output_path
    integer,parameter :: refinements(4)=[8,16,32,64],types(2)=[3,4]
    type(compact_filter_plan) :: smooth_plan
    type(compact_scheme) :: smooth_filter
    real(8),allocatable :: source(:,:),filtered(:,:),cpu(:)
    real(8) :: error,previous,order,cpu_error
    integer :: handle,t,r,length,point
    open(newunit=handle,file=output_path,status='new',action='write')
    write(handle,'(A)') 'dim,ntype,scaled_smooth_error,observed_order,cpu_gpu_scaled'
    do t=1,size(types)
      previous=0.d0
      do r=1,size(refinements)
        length=refinements(r)
        call compact_filter_initiate(smooth_filter,types(t),length,note='')
        call prepare_compact_filter_gpu(smooth_filter,hm,1,smooth_plan)
        allocate(source(-hm:length+hm,1),filtered(0:length,1))
        do point=-hm,length+hm
          source(point,1)=exp(4.d0*point/length)
        enddo
        smooth_plan%input=source
        call apply_compact_filter_gpu(smooth_plan,1)
        filtered=smooth_plan%filtered
        cpu=compact_filter(smooth_filter,source(:,1),length,note='')
        cpu_error=maxval(abs(filtered(:,1)-cpu))/max(1.d0,maxval(abs(cpu)))
        error=maxval(abs(filtered(:,1)-source(0:length,1)))/maxval(abs(source(0:length,1)))
        order=0.d0
        if(r>1) order=log(previous/error)/log(2.d0)
        write(handle,'(2(I0,A),3(ES24.16,:,A))') length,',',types(t),',',error,',',order,',',cpu_error
        flush(handle)
        if(cpu_error>2.d-10.or..not.ieee_is_finite(error)) error stop 'smooth filter reference failed'
        ! The highest-resolution interface case approaches the FP64 error floor.
        if(types(t)==3.and.length==32.and.order<9.d0) error stop 'interface filter tenth-order gate failed'
        if(types(t)==4.and.length==64.and.order<5.5d0) error stop 'physical filter sixth-order gate failed'
        previous=error
        call release_compact_filter_gpu(smooth_plan)
        deallocate(source,filtered,smooth_filter%a,smooth_filter%c,smooth_filter%ac)
      enddo
    enddo
    close(handle)
    write(*,'(A)') 'PASS: smooth compact filter convergence, interface and physical closures'
  end subroutine
end program
