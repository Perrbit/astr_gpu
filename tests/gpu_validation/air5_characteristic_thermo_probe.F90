#ifdef _CUDA
module characteristic_probe_gpu
  use cudafor
  use chemistry_characteristic
  implicit none
contains
  attributes(global) subroutine evaluate_top(q,target,p,a,g,normal,remaining,rate,rhs,df,endpoint,valid)
    real(8),device :: q(11),target(11),p,a,g(11),normal(11),remaining(11),rate,rhs(11)
    logical,device :: valid
    real(8),device :: df(11)
    real(8),device :: endpoint(11)
    real(8) :: inner1(11),inner2(11)
    logical :: endpoint_valid
    call air5_top_transport_rhs(q,target,p,a,g,normal,remaining,rate,rhs,valid)
    call air5_y_flux_differential(q,p,g,normal,df)
    inner1=q+(-1.d-3+0.25d-6)*normal
    inner2=q+(-2.d-3+1.d-6)*normal
    call air5_top_normal_convection(q,inner1,inner2,1.d-3,p,g,endpoint,endpoint_valid)
    valid=valid.and.endpoint_valid
  end subroutine evaluate_top
end module characteristic_probe_gpu
#endif
program air5_characteristic_thermo_probe
  use iso_fortran_env, only: real64
  use chemistry_air5_data, only: air5_formation_energy
  use chemistry_flow_state, only: air5_primitive_to_conservative,air5_conservative_to_primitive
  use chemistry_thermo, only: air5_species_gas_constant,air5_species_cv_tr
  use chemistry_characteristic, only: air5_top_transport_rhs,air5_y_flux_differential, &
    air5_top_normal_convection
#ifdef _CUDA
  use characteristic_probe_gpu
#endif
  implicit none
  real(real64) :: q(11),dq(11),gradient(11),y(5),u(3),rs(5),cvs(5)
  real(real64) :: rho,t,tv,p,beta,a2,a,h,fd,exact,scale,error,maxerror
  real(real64) :: coefficients(6),offsets(6),enthalpy,ev
  real(real64) :: flux_derivative(11),eigenvalue,analytic_flux_derivative(11)
  real(real64) :: maxfluxerror
  real(real64) :: acoustic(11,2),source(11),projected(11,2),convected(11)
  real(real64) :: amplitudes(2),dp_source,dmn_source,maxsourceerror
  character(len=32) :: step_argument
  integer :: s,n,state,sign,status
  integer,parameter :: independent(10)=[1,2,3,4,5,7,8,9,10,11]

  coefficients=[-1.d0,9.d0,-45.d0,45.d0,-9.d0,1.d0]/60.d0
  offsets=[-3.d0,-2.d0,-1.d0,1.d0,2.d0,3.d0]
  h=1.d-3
  if(command_argument_count()>0) then
    call get_command_argument(1,step_argument)
    read(step_argument,*,iostat=status) h
    if(status/=0) error stop 'invalid difference step'
    if(.not.(h>0.d0.and.h<=0.01d0)) error stop 'difference step outside (0,0.01]'
  endif
  maxerror=0.d0
  maxfluxerror=0.d0
  maxsourceerror=0.d0
  do s=1,5
    rs(s)=air5_species_gas_constant(s)
    cvs(s)=air5_species_cv_tr(s)
  enddo
  do state=1,3
    y=[0.70d0,0.20d0,0.03d0,0.04d0,0.03d0]
    rho=0.08d0
    t=1500.d0+1000.d0*state
    tv=1200.d0+200.d0*state
    u=[2700.d0,-600.d0,100.d0]
    if(state==2) u(2)=0.d0
    if(state==3) u(2)=600.d0
    call air5_primitive_to_conservative(rho,u,t,y,tv,q,status)
    if(status/=0) error stop 'invalid characteristic probe state'
    p=pressure(q)
    beta=sum(y*rs)/sum(y*cvs)
    a2=(1.d0+beta)*p/rho
    a=sqrt(a2)
    gradient(1)=0.5d0*beta*sum(u*u)
    gradient(2:4)=-beta*u
    gradient(5)=beta
    gradient(6:10)=(rs-beta*cvs)*t-beta*air5_formation_energy
    gradient(11)=-beta
    do n=1,10
      dq=0.d0
      s=independent(n)
      dq(s)=max(abs(q(s)),1.d-3)
      ! N2 is dependent for each tangent direction; no extra composition DOF.
      dq(6)=dq(1)-sum(dq(7:10))
      exact=sum(gradient*dq)
      fd=0.d0
      do s=1,3
        fd=fd+coefficients(7-s)*(pressure(q+h*offsets(7-s)*dq)- &
          pressure(q+h*offsets(s)*dq))/h
      enddo
      scale=max(abs(exact),p)
      error=abs(fd-exact)/scale
      maxerror=max(maxerror,error)
      if(error>2.d-10) error stop 'pressure differential mismatch'
    enddo
    enthalpy=(q(5)+p)/rho
    ev=q(11)/rho
    do sign=-1,1,2
      dq(1)=1.d0
      dq(2:4)=u
      dq(3)=u(2)+sign*a
      dq(5)=enthalpy+sign*a*u(2)
      dq(6:10)=y
      dq(11)=ev
      acoustic(:,(sign+3)/2)=dq
      if(abs(sum(gradient*dq)-a2)>1.d-12*a2) error stop 'frozen acoustic pressure'
      if(abs((dq(3)-u(2)*dq(1))-sign*a)>1.d-12*a) &
        error stop 'normal acoustic velocity'
      if(abs(sum(dq(6:10))-dq(1))>1.d-14) error stop 'acoustic mass closure'
      flux_derivative=0.d0
      do s=1,3
        flux_derivative=flux_derivative+coefficients(7-s)* &
          (normal_flux(q+h*offsets(7-s)*rho*dq)- &
           normal_flux(q+h*offsets(s)*rho*dq))/(h*rho)
      enddo
      eigenvalue=u(2)+sign*a
      call air5_y_flux_differential(q,p,gradient,dq,analytic_flux_derivative)
      if(any(abs(analytic_flux_derivative-eigenvalue*dq)> &
          2.d-10*max(abs(eigenvalue*dq),a*max(abs(dq),1.d0)))) &
        error stop 'acoustic analytic flux differential'
      call check_mode(dq,eigenvalue)
      error=maxval(abs(flux_derivative-eigenvalue*dq)/ &
        max(abs(eigenvalue*dq),a*max(abs(dq),1.d0)))
      if(error>2.d-10) error stop 'frozen acoustic flux eigenvector'
      maxfluxerror=max(maxfluxerror,error)
    enddo
    do n=1,8
      dq=0.d0
      select case(n)
      case(1)
        dq(1)=rho
        dq(2:4)=rho*u
        dq(6:10)=rho*y
        dq(11)=rho*ev
        dq(5)=q(5)-rho*sum(y*cvs)*t
      case(2,3)
        s=2
        if(n==3) s=4
        dq(s)=rho*a
        dq(5)=u(s-1)*dq(s)
      case(4:7)
        s=n+3
        dq(s)=rho
        dq(6)=-rho
        dq(5)=-(gradient(s)-gradient(6))*rho/beta
      case(8)
        dq(11)=p
        dq(5)=p
      end select
      if(abs(sum(gradient*dq))>1.d-12*max(p,maxval(abs(dq)))) &
        error stop 'convected mode changes pressure'
      if(abs(sum(dq(6:10))-dq(1))>1.d-14) error stop 'convected mass closure'
      flux_derivative=0.d0
      do s=1,3
        flux_derivative=flux_derivative+coefficients(7-s)* &
          (normal_flux(q+h*offsets(7-s)*dq)-normal_flux(q+h*offsets(s)*dq))/h
      enddo
      eigenvalue=u(2)
      call air5_y_flux_differential(q,p,gradient,dq,analytic_flux_derivative)
      if(any(abs(analytic_flux_derivative-eigenvalue*dq)> &
          2.d-10*max(abs(eigenvalue*dq),a*max(abs(dq),1.d0)))) &
        error stop 'convected analytic flux differential'
      call check_mode(dq,eigenvalue)
      error=maxval(abs(flux_derivative-eigenvalue*dq)/ &
        max(abs(eigenvalue*dq),a*max(abs(dq),1.d0)))
      if(error>2.d-10) then
        print *, 'convected mode state/mode/error=',state,n,error
        print *, 'flux derivative residual=',flux_derivative-eigenvalue*dq
        error stop 'convected flux eigenvector'
      endif
      maxfluxerror=max(maxfluxerror,error)
    enddo
    ! A pure vibrational-energy source has a nonzero pressure source at fixed E.
    if(gradient(11)/=-beta) error stop 'vibrational pressure source'
    ! Synthetic instantaneous sources test the closure algebra, not reaction rates.
    do n=1,2
      source=0.d0
      if(n==1) then
        source(11)=p
      else
        source(6)=-rho
        source(8)=rho
      endif
      dp_source=sum(gradient*source)
      dmn_source=source(3)-u(2)*source(1)
      amplitudes=[dp_source-a*dmn_source,dp_source+a*dmn_source]/(2.d0*a2)
      projected(:,1)=amplitudes(1)*acoustic(:,1)
      projected(:,2)=amplitudes(2)*acoustic(:,2)
      convected=source-projected(:,1)-projected(:,2)
      error=max(abs(sum(gradient*convected))/max(p,abs(dp_source)), &
        abs(convected(3)-u(2)*convected(1))/(rho*a))
      error=max(error,abs(sum(convected(6:10))-convected(1))/rho)
      do s=1,2
        error=max(error,abs(sum(projected(6:10,s))-projected(1,s))/rho)
      enddo
      if(error>1.d-12) error stop 'source characteristic decomposition'
      maxsourceerror=max(maxsourceerror,error)
      if(state==1) then
        if(.not.(u(2)<0.d0.and.u(2)+a>0.d0)) error stop 'source probe inflow regime'
        ! Only the plus acoustic mode leaves this subsonic normal inflow.
        if(abs(projected(1,2)-dp_source/(2.d0*a2))>1.d-12*rho) &
          error stop 'projected chemical source density'
        if(abs(projected(1,2))<1.d-6*rho) error stop 'source does not distinguish closures'
        if(any(source(1:5)/=0.d0)) error stop 'unprojected chemistry invariants'
      endif
    enddo
  enddo
  print '(A,ES24.16)', 'AIR5_CHARACTERISTIC_THERMO_PASS max_scaled_error=',maxerror
  print '(A,2ES24.16)', 'difference_step/max_flux_error=',h,maxfluxerror
  print '(A,ES24.16)', 'source_decomposition_error=',maxsourceerror
contains
  subroutine check_mode(direction,speed)
    real(real64),intent(in) :: direction(11),speed
    real(real64) :: target(11),normal(11),remaining(11),rhs(11),expected(11),rate
    real(real64) :: qtest(11),gtest(11),mode_speed,scale(11)
    real(real64) :: endpoint(11),inner1(11),inner2(11),flux_scale(11)
    integer :: trial
    logical :: valid
#ifdef _CUDA
    real(8),device :: qd(11),td(11),pd,ad,gd(11),nd(11),sd(11),kd,rd(11),dfd(11),epd(11)
    logical,device :: vd
    real(8) :: gpu_rhs(11),gpu_df(11),gpu_endpoint(11)
    logical :: gpu_valid
    integer :: ierr
#endif
    do trial=1,3
      qtest=q
      gtest=gradient
      mode_speed=speed
      if(trial>1) then
        ! Galilean shifts exercise both supersonic normal regimes.
        qtest(3)=q(3)+(-1.d0)**trial*3.d0*a*rho
        qtest(5)=q(5)+(qtest(3)**2-q(3)**2)/(2.d0*rho)
        gtest(1)=0.5d0*beta*sum((qtest(2:4)/rho)**2)
        gtest(3)=-beta*qtest(3)/rho
        mode_speed=speed+(-1.d0)**trial*3.d0*a
      endif
      normal=direction
      if(trial>1) then
        normal(3)=direction(3)+(-1.d0)**trial*3.d0*a*direction(1)
        normal(5)=direction(5)+(-1.d0)**trial*3.d0*a*direction(3)+ &
          0.5d0*(3.d0*a)**2*direction(1)
      endif
      target=qtest-1.d-3*normal
      remaining=0.d0
      rate=2.d0
      expected=-normal
      if(mode_speed<0.d0) expected=-rate*1.d-3*normal
      scale=max(1.d0,abs(normal))
      flux_scale=max(abs(mode_speed*normal),a*max(abs(normal),1.d0))
      inner1=qtest+(-1.d-3+0.25d-6)*normal
      inner2=qtest+(-2.d-3+1.d-6)*normal
      call air5_top_normal_convection(qtest,inner1,inner2,1.d-3,p,gtest,endpoint,valid)
      if(.not.valid.or.any(abs(endpoint-mode_speed*normal)>2.d-10*flux_scale)) &
        error stop 'quadratic endpoint convection'
      call air5_top_normal_convection(qtest,qtest,qtest,1.d-3,p,gtest,endpoint,valid)
      if(.not.valid.or.any(endpoint/=0.d0)) error stop 'constant endpoint convection'
      call air5_top_normal_convection(qtest,inner1,inner2,0.d0,p,gtest,endpoint,valid)
      if(valid) error stop 'zero endpoint spacing accepted'
      call air5_top_transport_rhs(qtest,target,p,a,gtest,normal,remaining,rate,rhs,valid)
      if(.not.valid.or.any(abs(rhs-expected)>2.d-10*scale)) error stop 'top mode RHS mismatch'
#ifdef _CUDA
      qd=qtest
      td=target
      pd=p
      ad=a
      gd=gtest
      nd=normal
      sd=remaining
      kd=rate
      call evaluate_top<<<1,1>>>(qd,td,pd,ad,gd,nd,sd,kd,rd,dfd,epd,vd)
      ierr=cudaDeviceSynchronize()
      if(ierr/=cudaSuccess) error stop 'top mode GPU kernel failure'
      gpu_rhs=rd
      gpu_df=dfd
      gpu_endpoint=epd
      if(any(abs(gpu_endpoint-mode_speed*normal)>2.d-10*flux_scale)) &
        error stop 'GPU quadratic endpoint convection'
      if(any(abs(gpu_df-mode_speed*normal)> &
          2.d-10*max(abs(mode_speed*normal),a*max(abs(normal),1.d0)))) &
        error stop 'GPU analytic flux differential'
      gpu_valid=vd
      if(.not.gpu_valid.or.any(abs(gpu_rhs-expected)>2.d-10*scale)) &
        error stop 'top mode GPU RHS mismatch'
      if(any(abs(gpu_rhs-rhs)>2.d-10*scale)) error stop 'top CPU/GPU RHS mismatch'
#endif
      ! Transverse/diffusive increments are retained without any projection.
      remaining=normal
      call air5_top_transport_rhs(qtest,target,p,a,gtest,normal,remaining,rate,rhs,valid)
      if(.not.valid.or.any(abs(rhs-expected-remaining)>2.d-10*scale)) &
        error stop 'top transverse source was projected'
      call air5_top_transport_rhs(qtest,qtest,p,a,gtest,0.d0*normal,0.d0*remaining,0.d0,rhs,valid)
      if(.not.valid.or.any(rhs/=0.d0)) error stop 'top uniform state mismatch'
      call air5_top_transport_rhs(qtest,target,p,a,gtest,normal,remaining,-1.d0,rhs,valid)
      if(valid) error stop 'negative top relaxation rate accepted'
    enddo
  end subroutine check_mode

  function normal_flux(state) result(flux)
    real(real64),intent(in) :: state(11)
    real(real64) :: flux(11),v,p
    v=state(3)/state(1)
    p=pressure(state)
    flux=state*v
    flux(3)=flux(3)+p
    flux(5)=flux(5)+p*v
  end function normal_flux

  function pressure(state) result(value)
    real(real64),intent(in) :: state(11)
    real(real64) :: value,density,velocity(3),temperature,fractions(5),vibration
    integer :: result_status
    call air5_conservative_to_primitive(state,density,velocity,temperature,fractions, &
      vibration,value,result_status)
    if(result_status/=0) error stop 'perturbed EOS state rejected'
  end function pressure
end program air5_characteristic_thermo_probe
