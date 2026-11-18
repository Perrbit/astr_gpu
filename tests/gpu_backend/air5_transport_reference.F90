module transport_reference_math
  use iso_fortran_env, only: real64
  use cfl_spectrum, only: spectral_rates
#ifdef ASTR_AIR5_REFERENCE_GPU
  use chemistry_thermo_gpu, only: temperature_from_q5=>air5_temperature_from_q5_gpu, &
    gas_constant=>air5_species_gas_constant_gpu,cv_tr=>air5_species_cv_tr_gpu
  use frozen_compensation_gpu, only: add=>compensated_add_gpu,rk=>compensated_rk_gpu
#else
  use chemistry_thermo, only: temperature_from_q5=>air5_temperature_from_q5, &
    gas_constant=>air5_species_gas_constant,cv_tr=>air5_species_cv_tr
  use chemistry_compensation, only: add=>compensated_add,rk=>compensated_rk
#endif
  implicit none
contains
#ifdef ASTR_AIR5_REFERENCE_GPU
  attributes(device) &
#endif
  subroutine evaluate(input,dt,compensated,out,status)
    real(real64),intent(in) :: input(65),dt
    logical,intent(in) :: compensated
    real(real64),intent(out) :: out(104)
    integer,intent(out) :: status(2)
    real(real64) :: high(11),carry(11),saved(11),origin(11),origin_carry(11),jac
    real(real64) :: a(3),b(3),c(3),t,rden,cvden,sound,velocity(3),metric(3,3),rates(4)
    integer :: stage,s
    logical :: valid
    out=0.0_real64;status=0
    a=[1.0_real64,0.75_real64,1.0_real64/3.0_real64]
    b=[0.0_real64,0.25_real64,2.0_real64/3.0_real64];c=[1.0_real64,0.25_real64,2.0_real64/3.0_real64]
    high=input(1:11);carry=input(55:65);jac=input(45)
    saved=high*jac;origin=0.0_real64;origin_carry=0.0_real64
    if(compensated) then
      origin=high;origin_carry=carry
    endif
    do stage=1,3
      do s=1,11
        if(compensated) then
          if(stage==1) then
            call add(high(s),carry(s),dt*input(11+11*(stage-1)+s)/jac)
          else
            call rk(high(s),carry(s),origin(s),origin_carry(s),a(stage),b(stage), &
              c(stage)*dt*input(11+11*(stage-1)+s)/jac)
          endif
        elseif(stage==1) then
          high(s)=(saved(s)+dt*input(11+s))/jac
        else
          high(s)=(a(stage)*saved(s)+b(stage)*high(s)*jac+c(stage)*dt*input(11+11*(stage-1)+s))/jac
        endif
      enddo
      out(11*(stage-1)+1:11*stage)=high
      out(33+11*(stage-1)+1:33+11*stage)=carry
    enddo
    out(67:77)=saved;out(78:88)=origin;out(89:99)=origin_carry
    call temperature_from_q5(input(1),input(2:4),input(6:10),input(11),input(5),t,status(1))
    if(status(1)/=0) return
    velocity=input(2:4)/input(1)
    rden=0.0_real64;cvden=0.0_real64
    do s=1,5
      rden=rden+input(5+s)*gas_constant(s);cvden=cvden+input(5+s)*cv_tr(s)
    enddo
    sound=sqrt((1.0_real64+rden/cvden)*rden*t/input(1))
    metric=reshape(input(46:54),[3,3])
    call spectral_rates(velocity,sound,metric,rates,valid)
    status(2)=merge(1,0,valid);out(100:103)=rates;out(104)=sound
  end subroutine
#ifdef ASTR_AIR5_REFERENCE_GPU
  attributes(global) subroutine batch(n,input,dt,compensated,out,status)
    integer,value :: n
    real(real64),value :: dt
    logical,value :: compensated
    real(real64),device :: input(65,n),out(104,n)
    integer,device :: status(2,n)
    integer :: i
    i=(blockidx%x-1)*blockdim%x+threadidx%x
    if(i<=n) call evaluate(input(:,i),dt,compensated,out(:,i),status(:,i))
  end subroutine
#endif
end module
program air5_transport_reference
  use iso_fortran_env, only: real64
  use transport_reference_math
#ifdef ASTR_AIR5_REFERENCE_GPU
  use cudafor
#endif
  implicit none
  real(real64),allocatable :: input(:,:),out(:,:)
  integer,allocatable :: status(:,:)
  real(real64) :: dt
  integer :: n,u,code,comp
  character(len=4096) :: source,target,text
#ifdef ASTR_AIR5_REFERENCE_GPU
  real(real64),device,allocatable :: xd(:,:),od(:,:)
  integer,device,allocatable :: sd(:,:)
#endif
  call get_command_argument(1,source);call get_command_argument(2,target);call get_command_argument(3,text)
  read(text,*) comp
  open(newunit=u,file=trim(source),access='stream',form='unformatted',status='old')
  read(u) n,dt;allocate(input(65,n),out(104,n),status(2,n));read(u) input;close(u)
#ifdef ASTR_AIR5_REFERENCE_GPU
  allocate(xd(65,n),od(104,n),sd(2,n));xd=input
  call batch<<<(n+63)/64,64>>>(n,xd,dt,comp/=0,od,sd)
  code=cudaDeviceSynchronize();if(code/=0) error stop 'reference execution'
  out=od;status=sd
#else
  do code=1,n
    call evaluate(input(:,code),dt,comp/=0,out(:,code),status(:,code))
  enddo
#endif
  open(newunit=u,file=trim(target),access='stream',form='unformatted',status='new')
  write(u) n,status,out;close(u)
end program
