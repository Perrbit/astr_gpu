program compact_line_solver_probe
  use iso_fortran_env, only: int64
  use astr_precision, only: real_kind,real_bytes
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use compact_line_solver, only: compact_matrix,compact_reference_factor, &
    factor_compact_reference,solve_compact_reference,compact_matvec,compact_residual
#ifdef TEST_CUDA
  use cudafor
  use compact_line_solver_gpu, only: compact_line_gpu_plan,prepare_compact_gpu, &
    solve_compact_gpu,release_compact_gpu
#endif
  implicit none
  integer,parameter :: lengths(9)=[127,128,129,255,256,257,511,512,513]
  integer,parameter :: rhs_counts(3)=[1,5,11],capacity=11
  type(compact_matrix) :: matrix
  type(compact_reference_factor) :: reference
#ifdef TEST_CUDA
  type(compact_line_gpu_plan) :: plan
  real(real_kind),allocatable,device :: rhs_d(:,:),solution_d(:,:)
  real(real_kind),allocatable :: unchanged(:,:)
  integer :: device,istat
#endif
  real(real_kind),allocatable :: exact(:,:),rhs(:,:),cpu(:,:),actual(:,:)
  real(real_kind) :: abs_error,scaled_error,residual,ref_error,ref_residual,scale
  real(real_kind) :: maximum_abs=0.e0_real_kind,maximum_scaled=0.e0_real_kind,maximum_residual=0.e0_real_kind
  real(real_kind) :: maximum_reference_abs=0.e0_real_kind,maximum_reference_residual=0.e0_real_kind
  integer(int64) :: device_bytes,peak_device_bytes=0_int64
  integer :: il,kind,ntype,n,nrhs,ir,i,j,unit,cases=0
  character(1024) :: filename,arg
  character(3) :: backend
  logical :: cyclic

  call get_command_argument(1,filename)
  if(len_trim(filename)==0) error stop 'usage: compact_line_probe OUTPUT.csv [DEVICE]'
#ifdef TEST_CUDA
  backend='gpu'
  device=0
  call get_command_argument(2,arg)
  if(len_trim(arg)>0) read(arg,*) device
  istat=cudaSetDevice(device)
  if(istat/=cudaSuccess) error stop 'compact probe cudaSetDevice failed'
#else
  backend='cpu'
#endif
  open(newunit=unit,file=trim(filename),status='replace',action='write')
  write(unit,'(A)') 'backend,n,kind,ntype,cyclic,nrhs,abs_error,scaled_error,residual,'// &
    'reference_abs_error,reference_residual,device_bytes'
  do il=1,size(lengths)
    n=lengths(il)
    allocate(exact(n,capacity),rhs(n,capacity))
    do j=1,capacity
      do i=1,n
        exact(i,j)=sin(0.013e0_real_kind*i*j)+0.15e0_real_kind*cos(0.021e0_real_kind*i*(j+1))+0.001e0_real_kind*j*i/n
      enddo
    enddo
    do kind=1,2
      do ntype=1,5
        cyclic=ntype==5
        call make_matrix(matrix,n,kind,ntype)
        call factor_compact_reference(matrix,reference)
        call compact_matvec(matrix,exact,rhs)
#ifdef TEST_CUDA
        allocate(rhs_d(n,capacity),solution_d(n,capacity),unchanged(n,capacity))
        rhs_d=rhs
        solution_d=huge(1.e0_real_kind)
        call prepare_compact_gpu(matrix,128,capacity,plan)
        device_bytes=plan%device_bytes+2_int64*n*capacity*int(real_bytes,int64)
#else
        device_bytes=0_int64
#endif
        peak_device_bytes=max(peak_device_bytes,device_bytes)
        do ir=1,size(rhs_counts)
          nrhs=rhs_counts(ir)
          allocate(cpu(n,nrhs),actual(n,nrhs))
          call solve_compact_reference(reference,rhs(:,1:nrhs),cpu)
          if(.not.all(ieee_is_finite(cpu))) error stop 'nonfinite compact CPU reference'
          ref_error=maxval(abs(cpu-exact(:,1:nrhs)))
          ref_residual=compact_residual(matrix,cpu,rhs(:,1:nrhs))
          if(real_bytes==8.and.ref_residual>1.e-12_real_kind) error stop 'compact reference residual failed'
          do j=1,nrhs
            scale=max(1.e0_real_kind,maxval(abs(exact(:,j))))
            if(real_bytes==8.and.maxval(abs(cpu(:,j)-exact(:,j)))/scale>2.e-10_real_kind) &
              error stop 'compact reference manufactured solution failed'
          enddo
#ifdef TEST_CUDA
          call solve_compact_gpu(plan,rhs_d,solution_d,nrhs)
          actual=solution_d(:,1:nrhs)
#else
          actual=cpu
#endif
          if(.not.all(ieee_is_finite(actual))) error stop 'nonfinite compact solution'
          abs_error=maxval(abs(actual-cpu))
          scaled_error=0.e0_real_kind
          do j=1,nrhs
            scale=max(1.e0_real_kind,maxval(abs(cpu(:,j))))
            scaled_error=max(scaled_error,maxval(abs(actual(:,j)-cpu(:,j)))/scale)
          enddo
          residual=compact_residual(matrix,actual,rhs(:,1:nrhs))
          write(unit,'(A,3(A,I0),A,L1,A,I0,5(A,ES24.16E3),A,I0)') trim(backend), &
            ',',n,',',kind,',',ntype,',',cyclic,',',nrhs, &
            ',',abs_error,',',scaled_error,',',residual,',',ref_error,',',ref_residual,',',device_bytes
          flush(unit)
          if(.not.all(ieee_is_finite([scaled_error,residual,ref_residual]))) &
            error stop 'nonfinite compact solve diagnostics'
          if(real_bytes==8.and.(scaled_error>2.e-10_real_kind.or.residual>1.e-12_real_kind)) then
            write(*,*) 'compact failure: n,kind,ntype,nrhs=',n,kind,ntype,nrhs
            write(*,*) 'absolute,scaled,residual=',abs_error,scaled_error,residual
            error stop 'compact V0 gate failed'
          endif
          maximum_abs=max(maximum_abs,abs_error)
          maximum_scaled=max(maximum_scaled,scaled_error)
          maximum_residual=max(maximum_residual,residual)
          maximum_reference_abs=max(maximum_reference_abs,ref_error)
          maximum_reference_residual=max(maximum_reference_residual,ref_residual)
          cases=cases+1
          deallocate(cpu,actual)
        enddo
#ifdef TEST_CUDA
        unchanged=rhs_d
        if(any(unchanged/=rhs)) error stop 'compact solve modified input RHS'
        call release_compact_gpu(plan)
        deallocate(rhs_d,solution_d,unchanged)
#endif
      enddo
    enddo
    deallocate(exact,rhs)
  enddo
  close(unit)
  write(*,'(A,A,A,I0)') 'COMPACT_',trim(backend),'_PASS cases=',cases
  write(*,'(A,3(ES24.16E3,1X))') 'maximum absolute/scaled/residual: ', &
    maximum_abs,maximum_scaled,maximum_residual
  write(*,'(A,2(ES24.16E3,1X))') 'reference manufactured absolute/residual: ', &
    maximum_reference_abs,maximum_reference_residual
  write(*,'(A,I0)') 'controlled peak device bytes: ',peak_device_bytes
contains
  subroutine make_matrix(a,n,kind,ntype)
    type(compact_matrix),intent(out) :: a
    integer,intent(in) :: n,kind,ntype
    real(real_kind) :: alpha
    logical :: lower_wall,upper_wall
    a%n=n
    a%cyclic=ntype==5
    allocate(a%lower(n),a%diagonal(n),a%upper(n))
    alpha=1.e0_real_kind/3.e0_real_kind
    if(kind==2) alpha=0.49e0_real_kind
    a%lower=alpha
    a%diagonal=1.e0_real_kind
    a%upper=alpha
    if(a%cyclic) return
    lower_wall=ntype==1.or.ntype==4
    upper_wall=ntype==2.or.ntype==4
    a%lower(1)=0.e0_real_kind
    a%upper(n)=0.e0_real_kind
    if(kind==1) then
      a%upper(1)=0.e0_real_kind
      a%lower(n)=0.e0_real_kind
      if(lower_wall) then
        a%upper(1)=2.e0_real_kind
        a%lower(2)=0.25e0_real_kind
        a%upper(2)=0.25e0_real_kind
      endif
      if(upper_wall) then
        a%lower(n)=2.e0_real_kind
        a%lower(n-1)=0.25e0_real_kind
        a%upper(n-1)=0.25e0_real_kind
      endif
    else
      a%upper(1)=1.11e0_real_kind
      a%lower(n)=1.11e0_real_kind
      if(lower_wall) a%upper(1)=0.98e0_real_kind
      if(upper_wall) a%lower(n)=0.98e0_real_kind
    endif
  end subroutine make_matrix
end program compact_line_solver_probe
