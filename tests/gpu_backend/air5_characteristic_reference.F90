module characteristic_reference
  use iso_fortran_env,only:real64
  use chemistry_characteristic
#ifdef _CUDA
  use cudafor
#endif
  implicit none
contains
#ifdef _CUDA
  attributes(host,device) &
#endif
  subroutine evaluate(x,out,flags)
    real(real64),intent(in) :: x(96)
    real(real64),intent(out) :: out(66)
    integer,intent(out) :: flags(5)
    logical :: valid,high_valid,low_valid
    out=0.0_real64;flags=0
    call air5_top_normal_convection(x(1:11),x(56:66),x(67:77),x(89),x(90),x(23:33), &
      out(1:11),high_valid)
    flags(1)=merge(1,0,high_valid)
    call air5_top_normal_convection(x(1:11),x(56:66),x(67:77),x(89),x(90),x(23:33), &
      out(12:22),low_valid,.true.)
    flags(2)=merge(1,0,low_valid)
    call air5_top_transport_rhs(x(1:11),x(12:22),x(90),x(91),x(23:33),x(34:44),x(45:55), &
      x(92),out(23:33),valid,x(93))
    flags(3)=merge(1,0,valid)
    call air5_y_flux_differential(x(1:11),x(90),x(23:33),x(78:88),out(34:44))
    if(high_valid)then
      call air5_top_transport_rhs(x(1:11),x(12:22),x(90),x(91),x(23:33),out(1:11),x(45:55), &
        x(92),out(45:55),valid,x(93))
      flags(4)=merge(1,0,valid)
    endif
    if(low_valid)then
      call air5_top_transport_rhs(x(1:11),x(12:22),x(90),x(91),x(23:33),out(12:22),x(45:55), &
        x(92),out(56:66),valid,x(93))
      flags(5)=merge(1,0,valid)
    endif
  end subroutine
#ifdef _CUDA
  attributes(global) subroutine evaluate_all(x,out,flags,n)
    integer,value :: n
    real(real64),device :: x(96,n),out(66,n)
    integer,device :: flags(5,n)
    integer :: i
    i=(blockIdx%x-1)*blockDim%x+threadIdx%x
    if(i>n)return
    call evaluate(x(:,i),out(:,i),flags(:,i))
  end subroutine
#endif
end module
program air5_characteristic_reference
  use characteristic_reference
  implicit none
  real(real64),allocatable :: x(:,:),out(:,:)
  integer,allocatable :: flags(:,:)
  integer :: n,u,i,status
  character(1024) :: input,output
#ifdef _CUDA
  real(real64),device,allocatable :: xd(:,:),od(:,:)
  integer,device,allocatable :: fd(:,:)
#endif
  call get_command_argument(1,input);call get_command_argument(2,output)
  open(newunit=u,file=trim(input),access='stream',form='unformatted',status='old')
  read(u)n;allocate(x(96,n),out(66,n),flags(5,n));read(u)x;close(u)
#ifdef _CUDA
  allocate(xd(96,n),od(66,n),fd(5,n));xd=x
  call evaluate_all<<<(n+127)/128,128>>>(xd,od,fd,n)
  status=cudaDeviceSynchronize();if(status/=0)error stop 'characteristic reference launch'
  out=od;flags=fd
#else
  do i=1,n
    call evaluate(x(:,i),out(:,i),flags(:,i))
  enddo
#endif
  open(newunit=u,file=trim(output),access='stream',form='unformatted',status='new')
  write(u)n,flags,out;close(u)
end program
