#ifdef TEST_CUDA
module nscbc_incident_probe_launch
  use astr_precision, only: real_kind
  use cudafor
  use nscbc_incident_gpu
  implicit none
contains
  attributes(global) subroutine evaluate_incident(face,external,pnor,pinv,metric,base,ref_rate, &
                                                  post,selected,rates,lambdas,rhs,masks,status)
    real(real_kind),device :: face(5,9),external(5,9),pnor(5,5,9),pinv(5,5,9),metric(3), &
                             base(5),ref_rate(5),post(5),selected(5,3),rates(5,9),lambdas(5,9),rhs(5,9)
    logical,device :: masks(5,9)
    integer,device :: status(10)
    real(real_kind) :: upstream(5),beta,position(3)
    integer :: icase
    icase=threadIdx%x
    if(icase==1) then
      beta=12.645231596933309e0_real_kind*acos(-1.e0_real_kind)/180.e0_real_kind
      upstream=[1.e0_real_kind,1.e0_real_kind,0.e0_real_kind,0.e0_real_kind, &
                0.5e0_real_kind+1.e0_real_kind/(1.4e0_real_kind*49.e0_real_kind*0.4e0_real_kind)]
      call incident_shock_states(upstream,1.4e0_real_kind,beta,post,status(10))
      position=[0.7e0_real_kind,0.03e0_real_kind,0.e0_real_kind]
      call incident_reference(position,0.7e0_real_kind,0.03e0_real_kind,beta,upstream,post,selected(:,1))
      position(1)=0.7e0_real_kind-0.0005e0_real_kind
      call incident_reference(position,0.7e0_real_kind,0.03e0_real_kind,beta,upstream,post,selected(:,2))
      position(2)=0.031e0_real_kind
      call incident_reference(position,0.7e0_real_kind,0.03e0_real_kind,beta,upstream,post,selected(:,3))
    endif
    if(icase>9) return
    call grcbc_incoming_rate(face(:,icase),external(:,icase),1.4e0_real_kind,metric, &
         0.125e0_real_kind,pinv(:,:,icase),ref_rate,rates(:,icase),lambdas(:,icase),masks(:,icase),status(icase))
    if(status(icase)/=0) return
    rhs(:,icase)=base
    call apply_incident_characteristic_rhs(rhs(:,icase),2.e0_real_kind,pnor(:,:,icase),pinv(:,:,icase), &
                                           masks(:,icase),rates(:,icase),status(icase))
  end subroutine evaluate_incident

  attributes(global) subroutine evaluate_normal(face,pnor,pinv,base,normal_result,compatible_result,free_result,status)
    real(real_kind),device :: face(5,9),pnor(5,5,9),pinv(5,5,9),base(5), &
      normal_result(5,9),compatible_result(5,9),free_result(5,9)
    integer,device :: status(10)
    real(real_kind) :: states(5,3),vel(3,3),pressure(3),jac(3),metrics(3,3),rho, &
      rate(5),lambda(5),zero(5),local_rhs(5),normal_lodi(5),free_lodi(5)
    logical :: incoming(5)
    integer :: icase,l
    icase=threadIdx%x
    if(icase>9) return
    do l=1,3
      vel(:,l)=face(2:4,icase)/face(1,icase)
      pressure(l)=0.4e0_real_kind*(face(5,icase)-0.5e0_real_kind*sum(vel(:,l)**2))
      rho=1.e0_real_kind-0.01e0_real_kind*(l-1)
      states(:,l)=[rho,rho*vel(:,l),pressure(l)/0.4e0_real_kind+0.5e0_real_kind*rho*sum(vel(:,l)**2)]
      metrics(:,l)=[0.e0_real_kind,3.e0_real_kind,0.e0_real_kind]
    enddo
    jac=2.e0_real_kind
    call incident_normal_characteristics(states,vel,pressure,jac,metrics,pinv(:,:,icase),normal_lodi,status(icase))
    if(status(icase)/=0) return
    normal_result(:,icase)=normal_lodi
    zero=0.e0_real_kind
    call grcbc_incoming_rate(face(:,icase),face(:,icase),1.4e0_real_kind,metrics(:,1),0.125e0_real_kind, &
      pinv(:,:,icase),zero,rate,lambda,incoming,status(icase))
    local_rhs=base
    call incident_convective_compatibility(local_rhs,2.e0_real_kind,pinv(:,:,icase),normal_lodi,incoming,rate)
    call apply_incident_characteristic_rhs(local_rhs,2.e0_real_kind,pnor(:,:,icase),pinv(:,:,icase),incoming,rate,status(icase))
    compatible_result(:,icase)=local_rhs
    jac=[2.e0_real_kind,1.5e0_real_kind,1.e0_real_kind]
    do l=1,3
      states(:,l)=face(:,icase)
      metrics(2,l)=real(4-l,real_kind)
    enddo
    call incident_normal_characteristics(states,vel,pressure,jac,metrics,pinv(:,:,icase),free_lodi,status(icase))
    free_result(:,icase)=free_lodi
  end subroutine evaluate_normal
end module nscbc_incident_probe_launch
#endif

program nscbc_incident_probe
  use astr_precision, only: real_kind
  use nscbc_incident
#ifdef TEST_CUDA
  use nscbc_incident_probe_launch, only: evaluate_incident,evaluate_normal
  use cudafor
#endif
  implicit none
  real(real_kind),parameter :: gamma=1.4e0_real_kind,tol=2.e-10_real_kind
  real(real_kind) :: face(5,9),external(5,9),pnor(5,5,9),pinv(5,5,9),metric(3),base(5),ref_rate(5)
  real(real_kind) :: post(5),selected(5,3),rates(5,9),lambdas(5,9),rhs(5,9),un(9)
  real(real_kind) :: pre(5),beta,pressure,velocity(3),normal(3),tangent(3),p2,flux1(5),flux2(5)
  real(real_kind) :: drho,dp,du(3),inc(5),expected(5),old_projection(5),new_projection(5), &
                     delta_rate(5),trial(5),position(3),error
  real(real_kind) :: states(5,3),vel_line(3,3),p_line(3),j_line(3),metric_line(3,3), &
    normals(5,9),compatible(5,9),free_result(5,9),rate_local(5),rho_local
  logical :: masks(5,9)
  integer,parameter :: incoming_counts(9)=[1,4,0,5,1,0,4,1,1]
  integer :: statuses(10),icase,m,status,l
#ifdef TEST_CUDA
  real(real_kind),device :: face_d(5,9),external_d(5,9),pnor_d(5,5,9),pinv_d(5,5,9),metric_d(3), &
                           base_d(5),ref_rate_d(5),post_d(5),selected_d(5,3), &
                           rates_d(5,9),lambdas_d(5,9),rhs_d(5,9)
  logical,device :: masks_d(5,9)
  integer,device :: statuses_d(10)
  real(real_kind),device :: normals_d(5,9),compatible_d(5,9),free_d(5,9)
  real(real_kind) :: actual(5,9),shock_actual(5),reference_actual(5,3)
  logical :: actual_masks(5,9)
  integer :: ierr
#endif

  beta=12.645231596933309e0_real_kind*acos(-1.e0_real_kind)/180.e0_real_kind
  pressure=1.e0_real_kind/(gamma*49.e0_real_kind)
  pre=[1.e0_real_kind,1.e0_real_kind,0.e0_real_kind,0.e0_real_kind, &
       pressure/(gamma-1.e0_real_kind)+0.5e0_real_kind]
  post=-99.e0_real_kind
  call incident_shock_states(pre,gamma,beta,post,status)
  if(status/=0) error stop 'incident shock construction failed'
  velocity=post(2:4)/post(1)
  p2=(gamma-1.e0_real_kind)*(post(5)-0.5e0_real_kind*post(1)*sum(velocity*velocity))
  if(abs(p2/pressure-2.5729409244014168e0_real_kind)>tol) error stop 'wrong HCW pressure jump'
  if(abs(post(1)-1.9173870077211825e0_real_kind)>tol) error stop 'wrong HCW density jump'
  if(abs(velocity(1)-0.9770708320058104e0_real_kind)>tol) error stop 'wrong HCW streamwise turn'
  if(abs(velocity(2)+0.10220015392135719e0_real_kind)>tol) error stop 'wrong HCW downward turn'
  normal=[sin(beta),cos(beta),0.e0_real_kind]
  tangent=[cos(beta),-sin(beta),0.e0_real_kind]
  flux1=normal_flux(pre,normal)
  flux2=normal_flux(post,normal)
  if(maxval(abs(flux2-flux1))>tol) error stop 'Rankine-Hugoniot flux mismatch'
  if(abs(sum((velocity-pre(2:4)/pre(1))*tangent))>tol) error stop 'shock tangential velocity changed'
  position=[0.7e0_real_kind,0.03e0_real_kind,0.e0_real_kind]
  call incident_reference(position,0.7e0_real_kind,0.03e0_real_kind,beta,pre,post,selected(:,1))
  position(1)=0.7e0_real_kind-0.0005e0_real_kind
  call incident_reference(position,0.7e0_real_kind,0.03e0_real_kind,beta,pre,post,selected(:,2))
  position(2)=0.031e0_real_kind
  call incident_reference(position,0.7e0_real_kind,0.03e0_real_kind,beta,pre,post,selected(:,3))
  if(any(selected(:,1)/=pre).or.any(selected(:,2)/=pre).or.any(selected(:,3)/=post)) &
    error stop 'oblique exterior geometry or interface tie rule failed'
  trial=post
  call incident_shock_states(pre,gamma,0.01e0_real_kind,trial,status)
  if(status/=2.or.any(trial/=post)) error stop 'noncompressive shock was accepted or mutated output'

  metric=[0.e0_real_kind,3.e0_real_kind,0.e0_real_kind]
  base=[3.e0_real_kind,-2.e0_real_kind,1.e0_real_kind,0.5e0_real_kind,-0.25e0_real_kind]
  ref_rate=[0.001e0_real_kind,-0.002e0_real_kind,0.003e0_real_kind,-0.004e0_real_kind,0.005e0_real_kind]
  un=[0.2e0_real_kind,-0.2e0_real_kind,2.e0_real_kind,-2.e0_real_kind, &
      0.e0_real_kind,1.e0_real_kind,-1.e0_real_kind,1.e-16_real_kind,-1.e-16_real_kind]
  do icase=1,9
    velocity=[10.e0_real_kind,un(icase),0.05e0_real_kind]
    face(:,icase)=make_state(1.e0_real_kind,velocity,1.e0_real_kind/gamma)
    external(:,icase)=make_state(1.01e0_real_kind,velocity+ &
         [0.002e0_real_kind,-0.003e0_real_kind,0.004e0_real_kind],1.02e0_real_kind/gamma)
    call y_euler_basis(velocity,pnor(:,:,icase),pinv(:,:,icase))
    call grcbc_incoming_rate(face(:,icase),external(:,icase),gamma,metric,0.125e0_real_kind,pinv(:,:,icase), &
                            ref_rate,rates(:,icase),lambdas(:,icase),masks(:,icase),statuses(icase))
    if(statuses(icase)/=0) error stop 'GRCBC rate rejected valid state'
    if(count(masks(:,icase))/=incoming_counts(icase)) &
      error stop 'boundary-normal characteristic count failed'
    drho=-0.01e0_real_kind
    du=[-0.002e0_real_kind,0.003e0_real_kind,-0.004e0_real_kind]
    dp=-0.02e0_real_kind/gamma
    inc(1)=drho
    inc(2:4)=velocity*drho+du
    inc(5)=dp/(gamma-1.e0_real_kind)+0.5e0_real_kind*sum(velocity*velocity)*drho+sum(velocity*du)
    expected=ref_rate
    do m=1,5
      if(masks(m,icase)) expected(m)=ref_rate(m)-abs(lambdas(m,icase))/0.125e0_real_kind*sum(pinv(m,:,icase)*inc)
    enddo
    if(maxval(abs(expected-rates(:,icase)))>tol) error stop 'GRCBC primitive projection or sign failed'
    rhs(:,icase)=base
    call apply_incident_characteristic_rhs(rhs(:,icase),2.e0_real_kind,pnor(:,:,icase),pinv(:,:,icase), &
                                           masks(:,icase),rates(:,icase),status)
    if(status/=0) error stop 'complete RHS projection failed'
    old_projection=matmul(pinv(:,:,icase),base)/2.e0_real_kind
    new_projection=matmul(pinv(:,:,icase),rhs(:,icase))/2.e0_real_kind
    do m=1,5
      if(masks(m,icase)) then
        if(abs(new_projection(m)-rates(m,icase))>tol) error stop 'incoming total RHS not imposed'
      else
        if(abs(new_projection(m)-old_projection(m))>tol) error stop 'outgoing total RHS changed'
      endif
    enddo
    ! A later transverse/viscous contribution must be included before closure.
    trial=base+[0.1e0_real_kind,-0.2e0_real_kind,0.3e0_real_kind,-0.4e0_real_kind,0.5e0_real_kind]
    old_projection=matmul(pinv(:,:,icase),trial)/2.e0_real_kind
    call apply_incident_characteristic_rhs(trial,2.e0_real_kind,pnor(:,:,icase),pinv(:,:,icase), &
                                           masks(:,icase),rates(:,icase),status)
    new_projection=matmul(pinv(:,:,icase),trial)/2.e0_real_kind
    do m=1,5
      if(masks(m,icase)) then
        if(abs(new_projection(m)-rates(m,icase))>tol) error stop 'late source bypassed incoming closure'
      else
        if(abs(new_projection(m)-old_projection(m))>tol) error stop 'outgoing source was removed'
      endif
    enddo
    do l=1,3
      vel_line(:,l)=velocity
      p_line(l)=1.e0_real_kind/gamma
      rho_local=1.e0_real_kind-.01e0_real_kind*(l-1)
      states(:,l)=make_state(rho_local,velocity,p_line(l))
      metric_line(:,l)=metric
    enddo
    j_line=2.e0_real_kind
    call incident_normal_characteristics(states,vel_line,p_line,j_line,metric_line,pinv(:,:,icase), &
                                         normals(:,icase),status)
    expected=0.e0_real_kind
    expected(1)=3.e0_real_kind*un(icase)*.01e0_real_kind
    if(status/=0.or.maxval(abs(normals(:,icase)-expected))>tol) error stop 'normal affine characteristic mismatch'
    rate_local=0.e0_real_kind
    compatible(:,icase)=base
    call incident_convective_compatibility(compatible(:,icase),2.e0_real_kind,pinv(:,:,icase), &
                                           normals(:,icase),masks(:,icase),rate_local)
    call apply_incident_characteristic_rhs(compatible(:,icase),2.e0_real_kind,pnor(:,:,icase), &
                                           pinv(:,:,icase),masks(:,icase),rate_local,status)
    new_projection=matmul(pinv(:,:,icase),compatible(:,icase))/2.e0_real_kind
    expected=matmul(pinv(:,:,icase),base)/2.e0_real_kind
    do m=1,5
      if(masks(m,icase)) then
        if(m<=3) then
          expected(m)=expected(m)+normals(m,icase)
        else
          expected(m)=0.e0_real_kind
        endif
      endif
    enddo
    if(maxval(abs(new_projection-expected))>tol) error stop 'convective compatibility removed a source'
    j_line=[2.e0_real_kind,1.5e0_real_kind,1.e0_real_kind]
    do l=1,3
      states(:,l)=face(:,icase)
      metric_line(2,l)=real(4-l,real_kind)
    enddo
    call incident_normal_characteristics(states,vel_line,p_line,j_line,metric_line,pinv(:,:,icase), &
                                         free_result(:,icase),status)
    if(status/=0.or.maxval(abs(free_result(:,icase)))>tol) error stop 'normal metric-rest did not preserve free stream'
  enddo
  call grcbc_incoming_rate(face(:,2),face(:,2),gamma,metric,0.125e0_real_kind,pinv(:,:,2), &
                          ref_rate,expected,delta_rate,masks(:,1),status)
  if(status/=0.or.maxval(abs(expected-ref_rate))>tol) error stop 'reference-state preservation failed'
  call grcbc_incoming_rate(face(:,2),face(:,2),gamma,metric,0.125e0_real_kind,pinv(:,:,2), &
                          0.e0_real_kind*ref_rate,expected,delta_rate,masks(:,1),status)
  if(status/=0.or.any(expected/=0.e0_real_kind)) error stop 'stationary reference-state preservation failed'
  ! Restore the first mask for the host/device comparison.
  masks(:,1)=[.false.,.false.,.false.,.false.,.true.]

#ifdef TEST_CUDA
  face_d=face; external_d=external; pnor_d=pnor; pinv_d=pinv; metric_d=metric
  base_d=base; ref_rate_d=ref_rate; post_d=-99.e0_real_kind; statuses_d=-1
  call evaluate_incident<<<1,32>>>(face_d,external_d,pnor_d,pinv_d,metric_d,base_d,ref_rate_d, &
       post_d,selected_d,rates_d,lambdas_d,rhs_d,masks_d,statuses_d)
  ierr=cudaDeviceSynchronize()
  if(ierr/=cudaSuccess) error stop 'incident GPU probe kernel failed'
  statuses=statuses_d
  if(any(statuses(1:10)/=0)) error stop 'incident GPU component rejected valid input'
  actual=rates_d
  error=maxval(abs(actual-rates))
  actual=lambdas_d
  error=max(error,maxval(abs(actual-lambdas)))
  actual=rhs_d
  error=max(error,maxval(abs(actual-rhs)))
  shock_actual=post_d
  error=max(error,maxval(abs(shock_actual-post)))
  reference_actual=selected_d
  error=max(error,maxval(abs(reference_actual-selected)))
  actual_masks=masks_d
  if(any(actual_masks.neqv.masks)) error stop 'incident CPU/GPU characteristic mask mismatch'
  if(error>tol) error stop 'incident CPU/GPU component mismatch'
  print '(a,es24.16)','NSCBC_INCIDENT_CPU_GPU_MAX_ABS ',error
  call evaluate_normal<<<1,32>>>(face_d,pnor_d,pinv_d,base_d,normals_d,compatible_d,free_d,statuses_d)
  ierr=cudaDeviceSynchronize()
  if(ierr/=cudaSuccess) error stop 'normal/source GPU probe kernel failed'
  statuses=statuses_d
  if(any(statuses(1:9)/=0)) error stop 'normal/source GPU component rejected valid input'
  actual=normals_d
  error=maxval(abs(actual-normals))
  actual=compatible_d
  error=max(error,maxval(abs(actual-compatible)))
  actual=free_d
  error=max(error,maxval(abs(actual-free_result)))
  if(error>tol) error stop 'normal/source CPU/GPU component mismatch'
  print '(a,es24.16)','NSCBC_INCIDENT_SOURCE_CPU_GPU_MAX_ABS ',error
#endif
  print '(a)','NSCBC_INCIDENT_PROBE_PASS'
contains
  subroutine y_euler_basis(vel,right,left)
    real(real_kind),intent(in) :: vel(3)
    real(real_kind),intent(out) :: right(5,5),left(5,5)
    real(real_kind) :: kinetic,h,dpdq(5),normal_momentum(5),identity(5,5)
    integer :: j
    kinetic=0.5e0_real_kind*sum(vel*vel)
    h=1.e0_real_kind/(gamma-1.e0_real_kind)+kinetic
    right(:,1)=[1.e0_real_kind,vel,kinetic]
    right(:,2)=[0.e0_real_kind,1.e0_real_kind,0.e0_real_kind,0.e0_real_kind,vel(1)]
    right(:,3)=[0.e0_real_kind,0.e0_real_kind,0.e0_real_kind,1.e0_real_kind,vel(3)]
    right(:,4)=[1.e0_real_kind,vel(1),vel(2)+1.e0_real_kind,vel(3),h+vel(2)]
    right(:,5)=[1.e0_real_kind,vel(1),vel(2)-1.e0_real_kind,vel(3),h-vel(2)]
    dpdq=(gamma-1.e0_real_kind)*[kinetic,-vel,1.e0_real_kind]
    normal_momentum=[-vel(2),0.e0_real_kind,1.e0_real_kind,0.e0_real_kind,0.e0_real_kind]
    left(1,:)=[1.e0_real_kind,0.e0_real_kind,0.e0_real_kind,0.e0_real_kind,0.e0_real_kind]-dpdq
    left(2,:)=[-vel(1),1.e0_real_kind,0.e0_real_kind,0.e0_real_kind,0.e0_real_kind]
    left(3,:)=[-vel(3),0.e0_real_kind,0.e0_real_kind,1.e0_real_kind,0.e0_real_kind]
    left(4,:)=0.5e0_real_kind*(dpdq+normal_momentum)
    left(5,:)=0.5e0_real_kind*(dpdq-normal_momentum)
    identity=matmul(left,right)
    do j=1,5
      identity(j,j)=identity(j,j)-1.e0_real_kind
    enddo
    if(maxval(abs(identity))>tol) error stop 'probe Euler characteristic bases are not inverse'
  end subroutine y_euler_basis

  function make_state(density,vel,p) result(q)
    real(real_kind),intent(in) :: density,vel(3),p
    real(real_kind) :: q(5)
    q(1)=density
    q(2:4)=density*vel
    q(5)=p/(gamma-1.e0_real_kind)+0.5e0_real_kind*density*sum(vel*vel)
  end function make_state

  function normal_flux(q,n) result(flux)
    real(real_kind),intent(in) :: q(5),n(3)
    real(real_kind) :: flux(5),vel(3),p,unormal
    vel=q(2:4)/q(1)
    p=(gamma-1.e0_real_kind)*(q(5)-0.5e0_real_kind*q(1)*sum(vel*vel))
    unormal=sum(n*vel)
    flux(1)=q(1)*unormal
    flux(2:4)=q(2:4)*unormal+p*n
    flux(5)=(q(5)+p)*unormal
  end function normal_flux
end program nscbc_incident_probe
