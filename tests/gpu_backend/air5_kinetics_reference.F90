module air5_kinetics_reference_math
  use iso_fortran_env, only: real64
#ifdef ASTR_AIR5_REFERENCE_GPU
  use chemistry_source_gpu, only: joint=>air5_instantaneous_source_jacobian_gpu, &
    pure_source=>air5_pure_source_gpu,activity_source=>air5_activity_source_gpu,activity_index=>air5_activity_index_gpu
  use chemistry_linear6_gpu, only: factor=>air5_lu_factor_6_gpu,solve=>air5_lu_solve_6_gpu
  use chemistry_thermo_gpu, only: difference=>air5_species_vibrational_difference_gpu, &
    td=>air5_temperature_derivatives_gpu,tvd=>air5_tv_derivatives_gpu
  use chemistry_relaxation_gpu, only: vt=>air5_vt_relaxation_source_gpu,vtj=>air5_vt_relaxation_source_derivatives_gpu
#else
  use chemistry_source, only: joint=>air5_instantaneous_source_jacobian, &
    activity_source=>air5_activity_source,activity_index=>air5_activity_index
  use chemistry_linear6, only: factor=>air5_lu_factor_6,solve=>air5_lu_solve_6
  use chemistry_thermo, only: difference=>air5_species_vibrational_difference, &
    td=>air5_temperature_derivatives,tvd=>air5_tv_derivatives
  use chemistry_relaxation, only: vt=>air5_vt_relaxation_source,vtj=>air5_vt_relaxation_source_derivatives
#endif
  implicit none
contains
#ifdef ASTR_AIR5_REFERENCE_GPU
  attributes(device) &
#endif
  subroutine evaluate(x,mode,out,status)
    real(real64),intent(in) :: x(53)
    integer,intent(in) :: mode
    real(real64),intent(out) :: out(132)
    integer,intent(out) :: status(11)
    real(real64) :: jac(6,6),matrix(6,6),lu(6,6),a(5),b(5),e(2),t(2)
    integer :: pivots(6)
    out=0.0_real64;status=-1;pivots=0
    if(mode==99) then
      call fixed_temperature(x,out,status)
      return
    endif
#ifdef ASTR_AIR5_REFERENCE_GPU
    call joint(x(1),x(2:4),x(5),x(6:11),out(1:6),jac,out(43:54),status(1),mode)
    call pure_source(x(1),x(2:4),x(5),x(6:11),out(55:60),status(2),mode)
#else
    call joint(x(1),x(2:4),x(5),x(6:11),out(1:6),jac,status(1),mode,out(43:54))
    call activity_source(x(1),x(2:4),x(5),x(6:11),out(55:60),status(2),mode,a,b,e,t)
#endif
    out(7:42)=reshape(jac,[36])
    call activity_source(x(1),x(2:4),x(5),x(6:11),out(61:66),status(3),mode, &
      out(67:71),out(72:76),out(77:78),out(79:80))
    if(status(3)==0) call activity_index(x(1),x(6:11),1.0e-10_real64,out(67:71), &
      out(72:76),out(77:78),out(79:80),out(90),out(81:89))
    matrix=reshape(x(12:47),[6,6])
    call factor(matrix,lu,pivots,status(4))
    out(91:126)=reshape(lu,[36]);status(6:11)=pivots
    if(status(4)==0) call solve(lu,pivots,x(48:53),out(127:132),status(5))
  end subroutine
#ifdef ASTR_AIR5_REFERENCE_GPU
  attributes(device) &
#endif
  subroutine fixed_temperature(x,out,status)
    real(real64),intent(in) :: x(53)
    real(real64),intent(inout) :: out(132)
    integer,intent(inout) :: status(11)
    real(real64) :: dt(6),dtv(6)
    integer :: s
    do s=1,5
      out(s)=difference(s,x(12),x(13))
    enddo
    call td(x(6:10),x(12),dt(1:5),dt(6),status(1))
    call tvd(x(6:10),x(13),dtv(1:5),dtv(6),status(2))
    call vt(x(6:10),x(12),x(13),out(6),status(3))
    if(status(1)/=0.or.status(2)/=0) return
    call vtj(x(6:10),x(12),x(13),dt,dtv,out(7),out(8:13),status(4))
    out(14:19)=dt;out(20:25)=dtv
  end subroutine
#ifdef ASTR_AIR5_REFERENCE_GPU
  attributes(global) subroutine evaluate_batch(n,x,mode,out,status)
    integer,value :: n,mode
    real(real64),device,intent(in) :: x(53,n)
    real(real64),device,intent(out) :: out(132,n)
    integer,device,intent(out) :: status(11,n)
    integer :: i
    i=(blockidx%x-1)*blockdim%x+threadidx%x
    if(i<=n) call evaluate(x(:,i),mode,out(:,i),status(:,i))
  end subroutine
#endif
end module

program air5_kinetics_reference
  use iso_fortran_env, only: real64
  use air5_kinetics_reference_math
#ifdef ASTR_AIR5_REFERENCE_GPU
  use cudafor
#endif
  implicit none
  character(len=1024) :: input_path,output_path,text
  real(real64),allocatable :: x(:,:),out(:,:)
  integer,allocatable :: statuses(:,:)
  integer :: n,u,i,mode
#ifdef ASTR_AIR5_REFERENCE_GPU
  real(real64),device,allocatable :: xd(:,:),od(:,:)
  integer,device,allocatable :: sd(:,:)
#endif
  call get_command_argument(1,input_path)
  call get_command_argument(2,output_path)
  call get_command_argument(3,text)
  read(text,*) mode
  open(newunit=u,file=trim(input_path),access='stream',form='unformatted',status='old')
  read(u) n
  allocate(x(53,n),out(132,n),statuses(11,n))
  read(u) x
  close(u)
#ifdef ASTR_AIR5_REFERENCE_GPU
  allocate(xd(53,n),od(132,n),sd(11,n))
  xd=x
  call evaluate_batch<<<(n+63)/64,64>>>(n,xd,mode,od,sd)
  i=cudaDeviceSynchronize()
  if(i/=0) error stop 'AIR5 kinetics reference device execution failed'
  out=od;statuses=sd
#else
  do i=1,n
    call evaluate(x(:,i),mode,out(:,i),statuses(:,i))
  enddo
#endif
  open(newunit=u,file=trim(output_path),access='stream',form='unformatted',status='new')
  write(u) n,statuses,out
  close(u)
end program
