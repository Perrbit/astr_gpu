module air5_filter_reference_math
  use iso_fortran_env,only:real64
  use chemistry_filter_math
  use chemistry_filter_operator
  use chemistry_model,only:air5_close_species_flux
#ifdef ASTR_AIR5_REFERENCE_GPU
  use cudafor
  use chemistry_model_gpu,only:air5_close_species_flux_gpu
#endif
  implicit none
contains
#ifdef ASTR_AIR5_REFERENCE_GPU
  attributes(device) &
#endif
  subroutine evaluate(x,out,status)
    real(real64),intent(in) :: x(256)
    real(real64),intent(out) :: out(48)
    integer,intent(out) :: status
    real(real64) :: stencil(11,-4:5),carry(11,-4:5)
    out=0.0_real64;status=0
    select case(nint(x(256)))
    case(0)
      out(1)=merge(1.0_real64,0.0_real64,filter_lower_energy_admissible(x(1:11),x(12:16),x(17:21)))
      out(2)=merge(1.0_real64,0.0_real64,filter_pressure_admissible(x(1:11)))
      out(3)=merge(1.0_real64,0.0_real64,lower_admissible(x(1:11),x(12:16),x(17:21)))
    case(1)
      out(1)=merge(1.0_real64,0.0_real64,filter_pressure_represented_admissible(x(1:11),x(12:22)))
      call filter_state_magnitude(x(1:11),x(12:22),out(2:12))
    case(2)
      out(1)=merge(1.0_real64,0.0_real64,filter_pair_box_safe(x(1:11),x(12:22),x(23:33), &
        x(34)/=0.0_real64,x(35:39),x(40:44)))
    case(3)
      call filter_feasible_mass(x(1:5),x(6:10),x(11:15),x(16),out(1))
    case(4)
      stencil=reshape(x(1:110),[11,10]);carry=reshape(x(111:220),[11,10])
      call filter_high_face(stencil,carry,nint(x(221)),nint(x(222)),x(223)/=0.0_real64,x(224)/=0.0_real64,out(1:11))
    case(5)
      call face_bounds(x(1:11),x(12:22),x(23:33),out(1:5),out(6:10),out(11))
#ifdef ASTR_AIR5_REFERENCE_GPU
      call air5_close_species_flux_gpu(x(28:32),out(1:5),out(6:10),out(11),out(12:16),status)
#else
      call air5_close_species_flux(x(28:32),out(1:5),out(6:10),out(11),out(12:16),status)
#endif
    case(6)
      call thermal_limit(x(1:11),x(12:22),x(23:33),x(34:38),x(39:43),out(1:11),out(12),status, &
        x(44:54),x(55:65),x(66)/=0.0_real64)
      if(status==0)out(13)=merge(1.0_real64,0.0_real64,pair_admissible(x(1:11),x(12:22),out(1:11), &
        x(34:38),x(39:43),x(44:54),x(55:65),x(66)/=0.0_real64))
    case default
      status=-1
    end select
  end subroutine
#ifdef ASTR_AIR5_REFERENCE_GPU
  attributes(global) subroutine batch(n,x,out,status)
    integer,value :: n
    real(real64),device :: x(256,n),out(48,n)
    integer,device :: status(n)
    integer :: i
    i=(blockidx%x-1)*blockdim%x+threadidx%x
    if(i<=n)call evaluate(x(:,i),out(:,i),status(i))
  end subroutine
#endif
end module
program air5_filter_reference
  use air5_filter_reference_math
  implicit none
  character(1024) :: input,output
  integer :: n,u,i
  real(real64),allocatable :: x(:,:),out(:,:)
  integer,allocatable :: status(:)
#ifdef ASTR_AIR5_REFERENCE_GPU
  real(real64),device,allocatable :: xd(:,:),od(:,:)
  integer,device,allocatable :: sd(:)
#endif
  call get_command_argument(1,input);call get_command_argument(2,output)
  open(newunit=u,file=trim(input),access='stream',form='unformatted',status='old')
  read(u)n;allocate(x(256,n),out(48,n),status(n));read(u)x;close(u)
#ifdef ASTR_AIR5_REFERENCE_GPU
  allocate(xd(256,n),od(48,n),sd(n));xd=x
  call batch<<<(n+31)/32,32>>>(n,xd,od,sd);out=od;status=sd
#else
  do i=1,n
    call evaluate(x(:,i),out(:,i),status(i))
  enddo
#endif
  open(newunit=u,file=trim(output),access='stream',form='unformatted',status='new')
  write(u)n,status,out;close(u)
end program
