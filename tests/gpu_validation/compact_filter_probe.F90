program compact_filter_probe
  use astr_precision, only: real_kind,real_bytes
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
  real(real_kind),allocatable :: input(:,:),rhs(:,:),actual_rhs(:,:),actual(:,:),reference(:,:), &
    actual_filtered(:,:),cpu_filtered(:,:),scalar_filtered(:,:),expected_rhs(:)
  real(real_kind) :: x,theta,transfer,abs_error,scaled_error,rhs_error,residual,projected_error,scalar_error
  real(real_kind) :: constant_error,polynomial_error,fourier_error,value,device_peak
  character(1024) :: path,argument

  call get_command_argument(1,path)
  if(trim(path)=='--explicit-window') then
    call get_command_argument(2,path)
    call explicit_window_reference(trim(path))
    stop
  endif
  if(len_trim(path)==0) error stop 'usage: compact_filter_probe OUTPUT.csv [DEVICE [ORDER.csv]]'
  device=0
  call get_command_argument(2,argument)
  if(len_trim(argument)>0) read(argument,*) device
  istat=cudaSetDevice(device)
  if(istat/=cudaSuccess) error stop 'cudaSetDevice failed'
  alfa_filter=0.49e0_real_kind
  call filter_coefficient_cal(alfa_filter,1.11e0_real_kind,0.98e0_real_kind)
  theta=0.5e0_real_kind*acos(-1.e0_real_kind)
  value=2.e0_real_kind*coef10i(0)
  do k=1,5
    value=value+2.e0_real_kind*coef10i(k)*cos(k*theta)
  enddo
  transfer=value/(1.e0_real_kind+2.e0_real_kind*alfa_filter*cos(theta))
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
      matrix%lower=afilter%a; matrix%upper=afilter%c; matrix%diagonal=1.e0_real_kind
      matrix%lower(1)=0.e0_real_kind; matrix%upper(n)=0.e0_real_kind
      call factor_compact_reference(matrix,factor)
      do bi=1,size(batches)
        nrhs=batches(bi)
        call prepare_compact_filter_gpu(afilter,hm,nrhs,plan)
        call prepare_compact_filter_gpu(afilter,hm,1,scalar)
        allocate(input(-hm:dim+hm,nrhs),rhs(n,nrhs),actual_rhs(n,nrhs),actual(n,nrhs),reference(n,nrhs), &
          actual_filtered(0:dim,nrhs),cpu_filtered(0:dim,nrhs),scalar_filtered(0:dim,nrhs))
        do j=1,nrhs
          do i=-hm,dim+hm
            x=real(i,real_kind)/dim
            select case(mod(j-1,5))
            case(0); input(i,j)=1.e0_real_kind
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
        scaled_error=abs_error/max(1.e0_real_kind,maxval(abs(reference)))
        residual=compact_residual(matrix,actual,rhs)
        projected_error=maxval(abs(actual_filtered-cpu_filtered))
        scalar_error=maxval(abs(actual_filtered-scalar_filtered))
        if(real_bytes==8.and.(rhs_error>2.e-10_real_kind.or.scaled_error>2.e-10_real_kind.or. &
           residual>1.e-12_real_kind.or.projected_error>2.e-10_real_kind.or.scalar_error>2.e-10_real_kind)) &
          error stop 'compact filter comparison failed'
        if(ntype==1.or.ntype==4) then
          if(any(actual_filtered(0,:)/=input(0,:))) error stop 'left physical endpoint changed'
        endif
        if(ntype==2.or.ntype==4) then
          if(any(actual_filtered(dim,:)/=input(dim,:))) error stop 'right physical endpoint changed'
        endif
        constant_error=maxval(abs(actual_filtered(:,1)-1.e0_real_kind))
        polynomial_error=0.e0_real_kind
        if(nrhs>=3) polynomial_error=maxval(abs(actual_filtered(:,2:3)-input(0:dim,2:3)))
        if(real_bytes==8.and.(constant_error>2.e-10_real_kind.or.polynomial_error>2.e-10_real_kind)) &
          error stop 'constant/polynomial filter property failed'
        fourier_error=0.e0_real_kind
        if(dim==512.and.nrhs>=4) then
          do i=dim/3,2*dim/3
            fourier_error=max(fourier_error,abs(actual_filtered(i,4)-transfer*cos(theta*i)))
          enddo
          if(real_bytes==8.and.fourier_error>2.e-10_real_kind) error stop 'interior Fourier filter response failed'
        endif
        device_peak=real(plan%line%device_bytes+scalar%line%device_bytes,real_kind)+ &
          real(real_bytes,real_kind)*(nrhs+1)*(dim+2*hm+1+2*n+dim+1)+ &
          real(real_bytes,real_kind)*2*(4+5+6+21+33)
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
    real(real_kind),allocatable :: source(:,:),filtered(:,:),cpu(:)
    real(real_kind) :: error,previous,order,cpu_error
    integer :: handle,t,r,length,point
    open(newunit=handle,file=output_path,status='new',action='write')
    write(handle,'(A)') 'dim,ntype,scaled_smooth_error,observed_order,cpu_gpu_scaled'
    do t=1,size(types)
      previous=0.e0_real_kind
      do r=1,size(refinements)
        length=refinements(r)
        call compact_filter_initiate(smooth_filter,types(t),length,note='')
        call prepare_compact_filter_gpu(smooth_filter,hm,1,smooth_plan)
        allocate(source(-hm:length+hm,1),filtered(0:length,1))
        do point=-hm,length+hm
          source(point,1)=exp(4.e0_real_kind*point/length)
        enddo
        smooth_plan%input=source
        call apply_compact_filter_gpu(smooth_plan,1)
        filtered=smooth_plan%filtered
        cpu=compact_filter(smooth_filter,source(:,1),length,note='')
        cpu_error=maxval(abs(filtered(:,1)-cpu))/max(1.e0_real_kind,maxval(abs(cpu)))
        error=maxval(abs(filtered(:,1)-source(0:length,1)))/maxval(abs(source(0:length,1)))
        order=0.e0_real_kind
        if(r>1) order=log(previous/error)/log(2.e0_real_kind)
        write(handle,'(2(I0,A),3(ES24.16,:,A))') length,',',types(t),',',error,',',order,',',cpu_error
        flush(handle)
        if((real_bytes==8.and.cpu_error>2.e-10_real_kind).or..not.ieee_is_finite(error)) &
          error stop 'smooth filter reference failed'
        ! The highest-resolution interface case approaches the FP64 error floor.
        if(real_bytes==8.and.types(t)==3.and.length==32.and.order<9.e0_real_kind) &
          error stop 'interface filter tenth-order gate failed'
        if(real_bytes==8.and.types(t)==4.and.length==64.and.order<5.5e0_real_kind) &
          error stop 'physical filter sixth-order gate failed'
        previous=error
        call release_compact_filter_gpu(smooth_plan)
        deallocate(source,filtered,smooth_filter%a,smooth_filter%c,smooth_filter%ac)
      enddo
    enddo
    close(handle)
    write(*,'(A)') 'PASS: smooth compact filter convergence, interface and physical closures'
  end subroutine
  subroutine explicit_window_reference(filename)
    use filter, only: filter_coefficient_explicit,spafilter10exp,coef8e
    character(*),intent(in) :: filename
    integer :: header(8),handle,a,b,c,m,h,ii,jj,kk,phase,location_x(2),location_y(1)
    real(real_kind) :: window(11,11,11,5),observed(5),expected(5),after_x(11,11),after_y(11)
    real(real_kind) :: state_x(11,11,5),state_y(11,5),internal_x(11,11),internal_y(11),stage_state(5,4)
    real(real_kind) :: minimum_internal,internal,error,kinetic,weight
    character(6),parameter :: stage_name(4)=[character(6) :: 'before','afterX','afterY','afterZ']
    real(real_kind),allocatable :: line(:),filtered(:)
    open(newunit=handle,file=filename,access='stream',form='unformatted',status='old')
    read(handle) header,window,observed
    close(handle)
    if(hm/=5) error stop 'explicit window reference requires five halo layers'
    call filter_coefficient_explicit()
    minimum_internal=huge(1.e0_real_kind)
    do c=1,11
    do b=1,11
    do a=1,11
      if(.not.all(ieee_is_finite(window(a,b,c,:))).or.window(a,b,c,1)<=0.e0_real_kind) &
        error stop 'captured input window is not admissible'
      internal=window(a,b,c,5)-0.5e0_real_kind*sum(window(a,b,c,2:4)**2)/window(a,b,c,1)
      if(.not.ieee_is_finite(internal)) error stop 'nonfinite captured internal energy'
      minimum_internal=min(minimum_internal,internal)
    enddo
    enddo
    enddo
    if(minimum_internal<=0.e0_real_kind) error stop 'captured input has nonpositive internal energy'
    do m=1,5
      allocate(line(-hm:header(1)+hm),filtered(0:header(1)))
      do c=1,11
      do b=1,11
        line=0.e0_real_kind
        do a=1,11
          ii=max(0,min(header(1),header(6)+a-6))
          line(ii)=window(a,b,c,m)
        enddo
        filtered=spafilter10exp(line,4,header(1))
        after_x(b,c)=filtered(header(6))
      enddo
      enddo
      state_x(:,:,m)=after_x
      deallocate(line,filtered)
      allocate(line(-hm:header(2)+hm),filtered(0:header(2)))
      do c=1,11
        line=0.e0_real_kind
        do b=1,11
          jj=max(0,min(header(2),header(7)+b-6))
          line(jj)=after_x(b,c)
        enddo
        filtered=spafilter10exp(line,4,header(2))
        after_y(c)=filtered(header(7))
      enddo
      state_y(:,m)=after_y
      deallocate(line,filtered)
      allocate(line(-hm:header(3)+hm),filtered(0:header(3)))
      line=0.e0_real_kind
      do c=1,11
        kk=modulo(header(8)+c-6,header(3))
        line(kk)=after_y(c)
      enddo
      line(header(3))=line(0)
      do h=1,hm
        line(-h)=line(header(3)-h)
        line(header(3)+h)=line(h)
      enddo
      filtered=spafilter10exp(line,3,header(3))
      expected(m)=filtered(header(8))
      deallocate(line,filtered)
    enddo
    if(.not.all(ieee_is_finite(expected)).or..not.all(ieee_is_finite(observed))) &
      error stop 'nonfinite CPU/GPU filter point'
    error=maxval(abs(expected-observed))
    write(*,'(A,ES24.16)') 'WINDOW_MIN_INPUT_INTERNAL=',minimum_internal
    if(.not.all(ieee_is_finite(state_x)).or..not.all(ieee_is_finite(state_y)).or. &
       minval(state_x(:,:,1))<=0.e0_real_kind.or.minval(state_y(:,1))<=0.e0_real_kind) &
      error stop 'directional replay has nonfinite state or nonpositive density'
    internal_x=state_x(:,:,5)-0.5e0_real_kind*sum(state_x(:,:,2:4)**2,dim=3)/state_x(:,:,1)
    internal_y=state_y(:,5)-0.5e0_real_kind*sum(state_y(:,2:4)**2,dim=2)/state_y(:,1)
    location_x=minloc(internal_x)
    location_y=minloc(internal_y)
    write(*,'(A,ES24.16,A,3(I0,1X))') 'WINDOW_AFTER_X_MIN_INTERNAL=',minval(internal_x), &
      ' global_ijk=',header(6),max(0,min(header(2),header(7)+location_x(1)-6)), &
      modulo(header(8)+location_x(2)-6,header(3))
    write(*,'(A,ES24.16,A,3(I0,1X))') 'WINDOW_AFTER_Y_MIN_INTERNAL=',minval(internal_y), &
      ' global_ijk=',header(6),header(7),modulo(header(8)+location_y(1)-6,header(3))
    stage_state(:,1)=window(6,6,6,:)
    stage_state(:,2)=state_x(6,6,:)
    stage_state(:,3)=state_y(6,:)
    stage_state(:,4)=expected
    do phase=1,4
      kinetic=0.5e0_real_kind*sum(stage_state(2:4,phase)**2)/stage_state(1,phase)
      internal=stage_state(5,phase)-kinetic
      write(*,'(A,A,A,5(ES24.16,1X))') 'WINDOW_STAGE ',trim(stage_name(phase)),' q=',stage_state(:,phase)
      write(*,'(A,A,A,3(ES24.16,1X))') 'WINDOW_STAGE ',trim(stage_name(phase)), &
        ' total/kinetic/internal=',stage_state(5,phase),kinetic,internal
    enddo
    if(header(6)==4) then
      write(*,'(A)') 'WINDOW_X_STENCIL columns=i,weight,rho,total,internal,weighted_total'
      do ii=0,8
        a=ii-header(6)+6
        weight=coef8e(abs(ii-header(6)))
        if(ii==header(6)) weight=2.e0_real_kind*weight
        internal=window(a,6,6,5)-0.5e0_real_kind*sum(window(a,6,6,2:4)**2)/window(a,6,6,1)
        write(*,'(A,I0,5(1X,ES24.16))') 'WINDOW_X_STENCIL ',ii,weight,window(a,6,6,1), &
          window(a,6,6,5),internal,weight*window(a,6,6,5)
      enddo
    endif
    write(*,'(A,5(ES24.16,1X))') 'WINDOW_CPU_FILTERED_Q=',expected
    write(*,'(A,5(ES24.16,1X))') 'WINDOW_GPU_FILTERED_Q=',observed
    write(*,'(A,ES24.16)') 'WINDOW_CPU_GPU_ABS=',error
    internal=expected(5)-0.5e0_real_kind*sum(expected(2:4)**2)/expected(1)
    write(*,'(A,ES24.16)') 'WINDOW_CPU_FILTERED_INTERNAL=',internal
    if(real_bytes==8.and.error>2.e-10_real_kind) error stop 'CPU filter does not reproduce captured GPU point'
    if(internal>=0.e0_real_kind) error stop 'CPU filter did not reproduce loss of admissibility'
    print *, 'PASS: native CPU filter reproduces loss of admissibility, not a CFD acceptance'
  end subroutine explicit_window_reference
end program
