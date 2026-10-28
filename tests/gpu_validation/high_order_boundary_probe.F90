#ifdef TEST_CUDA
module high_order_boundary_probe_launch
  use astr_precision, only: real_kind
  use boundary_high_order_gpu, only: boundary_derivative6,boundary_neumann7, &
    boundary_mp_flux6,boundary_filter6,boundary_incoming_rate7
  use cudafor
  implicit none
contains
  attributes(global) subroutine differentiate(f,df,n,h)
    integer,value :: n
    real(real_kind),value :: h
    real(real_kind),device :: f(0:n),df(0:n)
    real(real_kind) :: line(0:7)
    integer :: i,j
    i=(blockIdx%x-1)*blockDim%x+threadIdx%x-1
    if(i>n) return
    if(i<3) then
      line=f(0:7)
      df(i)=boundary_derivative6(line,i)/h
    elseif(i>n-3) then
      do j=0,7
        line(j)=f(n-j)
      enddo
      df(i)=-boundary_derivative6(line,n-i)/h
    else
      df(i)=(0.75e0_real_kind*(f(i+1)-f(i-1))-0.15e0_real_kind*(f(i+2)-f(i-2))+ &
             (f(i+3)-f(i-3))/60.e0_real_kind)/h
    endif
  end subroutine differentiate

  attributes(global) subroutine reconstruct_neumann(f,slope,value)
    real(real_kind),device :: f(1:7,0:7),slope(0:7),value(0:7)
    integer :: p
    p=threadIdx%x-1
    if(p<=7) value(p)=boundary_neumann7(f(:,p),slope(p))
  end subroutine reconstruct_neumann

  attributes(global) subroutine evaluate_candidates(samples,primitive,lambda,rhs,lodi,flux,filtered,rates)
    real(real_kind),device :: samples(0:7,10),primitive(5,0:7,10),lambda(5,10),rhs(5,10), &
      lodi(5,10),flux(0:2,0:1,10),filtered(0:2,10),rates(5,10)
    real(real_kind) :: pinv(5,5)
    logical :: incoming(5)
    integer :: icase,row,wind,m
    icase=threadIdx%x
    if(icase>10) return
    do row=0,2
      do wind=0,1
        flux(row,wind,icase)=boundary_mp_flux6(samples(:,icase),row,wind==1)
      enddo
      filtered(row,icase)=boundary_filter6(samples(0:6,icase),row)
    enddo
    pinv=0.e0_real_kind
    do m=1,5
      pinv(m,m)=1.e0_real_kind
      incoming(m)=lambda(m,icase)<0.e0_real_kind
    enddo
    call boundary_incoming_rate7(primitive(:,:,icase),1.4e0_real_kind,0.025e0_real_kind,pinv, &
      lambda(:,icase),incoming,rhs(:,icase),2.e0_real_kind, &
      lodi(:,icase),icase==10,rates(:,icase))
  end subroutine evaluate_candidates
end module high_order_boundary_probe_launch
#endif

program high_order_boundary_probe
  use astr_precision, only: real_kind
  use boundary_high_order, only: boundary_derivative6,boundary_neumann7, &
    boundary_derivative6_weights,boundary_neumann7_weights,boundary_mp_flux6, &
    boundary_filter6,boundary_incoming_rate7
#ifdef TEST_CUDA
  use high_order_boundary_probe_launch, only: differentiate,reconstruct_neumann,evaluate_candidates
  use cudafor
#endif
  implicit none
  integer,parameter :: sizes(3)=[32,64,128]
  ! At frequency 2 the finest composite error is already at the FP64 noise floor.
  ! Frequency 4 keeps all three preselected grids in the truncation regime.
  real(real_kind),parameter :: wave_number=4.e0_real_kind
  real(real_kind) :: sample(0:7),expected,moment,error,h,z,wall_data(1:7,0:7), &
    wall_slopes(0:7),wall_values(0:7),slope,orders(2),errors(3),backend_error
  real(real_kind),allocatable :: f(:),df(:),metric(:),mu(:),flux(:),result(:),reference(:),actual(:)
  integer :: row,p,j,n,k
  real(real_kind) :: samples(0:7,10),primitive(5,0:7,10),test_lambda(5,10),rhs(5,10), &
    lodi(5,10),candidate_flux(0:2,0:1,10),filtered(0:2,10),rates(5,10),expected_rates(5,10), &
    pinv(5,5),linear(5),quadratic(5),gradient(5),dq(5),source(5),un(10),css
  integer :: icase,wind,m
#ifdef TEST_CUDA
  real(real_kind),allocatable,device :: f_d(:),df_d(:)
  real(real_kind),device :: wall_data_d(1:7,0:7),wall_slopes_d(0:7),wall_values_d(0:7)
  integer :: status
  real(real_kind),device :: samples_d(0:7,10),primitive_d(5,0:7,10),lambda_d(5,10),rhs_d(5,10), &
    lodi_d(5,10),candidate_flux_d(0:2,0:1,10),filtered_d(0:2,10),rates_d(5,10)
  real(real_kind) :: flux_actual(0:2,0:1,10),filter_actual(0:2,10),rates_actual(5,10)
#endif
  if(precision(1.e0_real_kind)<15) error stop 'HB probe requires FP64'
  backend_error=0.e0_real_kind
  do row=0,2
    write(*,'(a,i0,8(1x,es25.17))') 'HB_WEIGHTS ',row,boundary_derivative6_weights(:,row)
    do p=0,7
      moment=0.e0_real_kind
      do j=0,7
        moment=moment+boundary_derivative6_weights(j,row)*real(j-row,real_kind)**p
      enddo
      expected=0.e0_real_kind
      if(p==1) expected=1.e0_real_kind
      if(p==7) expected=36.e0_real_kind
      if(abs(moment-expected)>1.e-8_real_kind) error stop 'HB derivative moment mismatch'
    enddo
    sample=2.e0_real_kind
    if(boundary_derivative6(sample,row)/=0.e0_real_kind) error stop 'HB constant derivative changed'
  enddo
  write(*,'(a,8(1x,es25.17))') 'HB_NEUMANN_WEIGHTS',boundary_neumann7_weights
  error=0.e0_real_kind
  do p=0,7
    do j=1,7
      wall_data(j,p)=(real(j,real_kind)/7.e0_real_kind)**p
    enddo
    slope=0.e0_real_kind
    if(p==1) slope=1.e0_real_kind/7.e0_real_kind
    wall_slopes(p)=slope
    wall_values(p)=boundary_neumann7(wall_data(:,p),slope)
    expected=0.e0_real_kind
    if(p==0) expected=1.e0_real_kind
    error=max(error,abs(wall_values(p)-expected))
  enddo
  if(error>2.e-13_real_kind) error stop 'HB Neumann polynomial reconstruction failed'
#ifdef TEST_CUDA
  wall_data_d=wall_data; wall_slopes_d=wall_slopes
  call reconstruct_neumann<<<1,8>>>(wall_data_d,wall_slopes_d,wall_values_d)
  status=cudaDeviceSynchronize()
  if(status/=0) error stop 'HB Neumann CUDA launch failed'
  sample=wall_values_d
  backend_error=maxval(abs(sample-wall_values))
#endif
  write(*,'(a,es25.17)') 'HB_NEUMANN_ERROR ',error
  samples=0.e0_real_kind
  do j=0,7
    samples(j,j+1)=1.e0_real_kind
  enddo
  samples(:,9)=1.e0_real_kind
  samples(4:7,10)=1.e0_real_kind
  pinv=0.e0_real_kind
  do m=1,5
    pinv(m,m)=1.e0_real_kind
  enddo
  css=sqrt(1.4e0_real_kind*0.05e0_real_kind/1.2e0_real_kind)
  un=[-.5e0_real_kind,-.1e0_real_kind,0.e0_real_kind,.1e0_real_kind,.5e0_real_kind, &
    css-1.e-6_real_kind,css+1.e-6_real_kind,-css-1.e-6_real_kind,-css+1.e-6_real_kind,-.1e0_real_kind]
  source=[.003e0_real_kind,-.002e0_real_kind,.005e0_real_kind,.001e0_real_kind,-.004e0_real_kind]
  do icase=1,10
    primitive(:,0,icase)=[1.2e0_real_kind,.7e0_real_kind,un(icase),.03e0_real_kind,.05e0_real_kind]
    linear=[.01e0_real_kind,-.02e0_real_kind,.015e0_real_kind,.01e0_real_kind,.005e0_real_kind]
    quadratic=0.5e0_real_kind*linear
    if(icase==1) then
      linear=0.e0_real_kind; quadratic=0.e0_real_kind
    endif
    do j=1,7
      z=real(j,real_kind)*.025e0_real_kind
      primitive(:,j,icase)=primitive(:,0,icase)+linear*z+quadratic*z*z
    enddo
    test_lambda(:,icase)=[un(icase),un(icase),un(icase),un(icase)+css,un(icase)-css]
    lodi(:,icase)=test_lambda(:,icase)*[.012e0_real_kind,.02e0_real_kind,-.01e0_real_kind, &
                                      .003e0_real_kind,.004e0_real_kind]
    rhs(:,icase)=2.e0_real_kind*(source-lodi(:,icase))
    gradient=linear
    if(icase==10) gradient=linear+.025e0_real_kind*quadratic
    dq(1)=gradient(1)
    dq(2:4)=primitive(2:4,0,icase)*gradient(1)+primitive(1,0,icase)*gradient(2:4)
    dq(5)=gradient(5)/.4e0_real_kind+.5e0_real_kind*sum(primitive(2:4,0,icase)**2)*gradient(1)+ &
      primitive(1,0,icase)*sum(primitive(2:4,0,icase)*gradient(2:4))
    expected_rates(:,icase)=source-lodi(:,icase)
    do m=1,5
      if(test_lambda(m,icase)<0.e0_real_kind) expected_rates(m,icase)=source(m)-test_lambda(m,icase)*dq(m)
    enddo
    call boundary_incoming_rate7(primitive(:,:,icase),1.4e0_real_kind,.025e0_real_kind,pinv, &
      test_lambda(:,icase),test_lambda(:,icase)<0.e0_real_kind,rhs(:,icase),2.e0_real_kind, &
      lodi(:,icase),icase==10,rates(:,icase))
    do row=0,2
      do wind=0,1
        candidate_flux(row,wind,icase)=boundary_mp_flux6(samples(:,icase),row,wind==1)
      enddo
      filtered(row,icase)=boundary_filter6(samples(0:6,icase),row)
    enddo
  enddo
  error=maxval(abs(rates-expected_rates))
  if(error>2.e-12_real_kind) error stop 'HB incoming source or external derivative mismatch'
  if(any(candidate_flux(:,:,9)/=1.e0_real_kind).or.any(filtered(:,9)/=1.e0_real_kind)) &
    error stop 'HB candidate changed a constant'
  do row=0,2
    do wind=0,1
      write(*,'(a,2(i0,1x),8(es25.17,1x))') 'HB_FLUX ',row,wind,candidate_flux(row,wind,1:8)
    enddo
    write(*,'(a,i0,1x,8(es25.17,1x))') 'HB_FILTER ',row,filtered(row,1:8)
  enddo
  write(*,'(a,es25.17)') 'HB_INCOMING_RATE_ERROR ',error
#ifdef TEST_CUDA
  samples_d=samples; primitive_d=primitive; lambda_d=test_lambda; rhs_d=rhs; lodi_d=lodi
  call evaluate_candidates<<<1,16>>>(samples_d,primitive_d,lambda_d,rhs_d,lodi_d, &
    candidate_flux_d,filtered_d,rates_d)
  status=cudaDeviceSynchronize()
  if(status/=0) error stop 'HB candidate CUDA launch failed'
  flux_actual=candidate_flux_d; filter_actual=filtered_d; rates_actual=rates_d
  backend_error=max(backend_error,maxval(abs(flux_actual-candidate_flux)), &
    maxval(abs(filter_actual-filtered)),maxval(abs(rates_actual-rates)))
#endif
  do k=1,size(sizes)
    n=sizes(k); h=1.e0_real_kind/real(n,real_kind)
    allocate(f(0:n),df(0:n),metric(0:n),mu(0:n),flux(0:n),result(0:n),reference(0:n),actual(0:n))
    do j=0,n
      z=real(j,real_kind)*h
      f(j)=exp(wave_number*z)+0.1e0_real_kind*sin(3.e0_real_kind*z)
      metric(j)=1.5e0_real_kind*exp(1.5e0_real_kind*z)/(exp(1.5e0_real_kind)-1.e0_real_kind)
      mu(j)=1.e0_real_kind+0.2e0_real_kind*cos(z)
      slope=wave_number*exp(wave_number*z)+0.3e0_real_kind*cos(3.e0_real_kind*z)
      reference(j)=(-0.2e0_real_kind*sin(z)*slope+mu(j)* &
        (wave_number**2*exp(wave_number*z)-0.9e0_real_kind*sin(3.e0_real_kind*z)))/metric(j)**2 &
        -mu(j)*slope*1.5e0_real_kind/metric(j)**2
    enddo
    call host_differentiate(f,df,n,h)
    flux=mu*df/metric
    call host_differentiate(flux,result,n,h)
    result=result/metric
    errors(k)=max(maxval(abs(result(0:5)-reference(0:5))), &
                  maxval(abs(result(n-5:n)-reference(n-5:n))))
#ifdef TEST_CUDA
    allocate(f_d(0:n),df_d(0:n))
    f_d=f
    call differentiate<<<(n+128)/128,128>>>(f_d,df_d,n,h)
    status=cudaDeviceSynchronize()
    if(status/=0) error stop 'HB gradient CUDA launch failed'
    actual=df_d
    backend_error=max(backend_error,maxval(abs(actual-df)/max(1.e0_real_kind,abs(df))))
    f_d=mu*actual/metric
    call differentiate<<<(n+128)/128,128>>>(f_d,df_d,n,h)
    status=cudaDeviceSynchronize()
    if(status/=0) error stop 'HB divergence CUDA launch failed'
    actual=df_d
    actual=actual/metric
    backend_error=max(backend_error,maxval(abs(actual-result)/max(1.e0_real_kind,abs(result))))
    deallocate(f_d,df_d)
#endif
    write(*,'(a,i0,a,es25.17)') 'HB_COMPOSITE intervals=',n,' error=',errors(k)
    deallocate(f,df,metric,mu,flux,result,reference,actual)
  enddo
  orders=log(errors(1:2)/errors(2:3))/log(2.e0_real_kind)
  write(*,'(a,2(1x,f12.7))') 'HB_COMPOSITE_ORDERS',orders
  write(*,'(a,es25.17)') 'HB_BACKEND_MAX_SCALED ',backend_error
  if(any(orders<5.5e0_real_kind)) error stop 'HB composite order below 5.5'
  if(backend_error>2.e-10_real_kind) error stop 'HB CPU GPU mismatch'
  print *, 'HB_COMPONENT_ACCURACY_PASS_NOT_SOLVER_STABILITY'
contains
  subroutine host_differentiate(v,dv,n,h)
    integer,intent(in) :: n
    real(real_kind),intent(in) :: v(0:n),h
    real(real_kind),intent(out) :: dv(0:n)
    real(real_kind) :: line(0:7)
    integer :: i,j
    do i=0,n
      if(i<3) then
        line=v(0:7)
        dv(i)=boundary_derivative6(line,i)/h
      elseif(i>n-3) then
        do j=0,7
          line(j)=v(n-j)
        enddo
        dv(i)=-boundary_derivative6(line,n-i)/h
      else
        dv(i)=(0.75e0_real_kind*(v(i+1)-v(i-1))-0.15e0_real_kind*(v(i+2)-v(i-2))+ &
               (v(i+3)-v(i-3))/60.e0_real_kind)/h
      endif
    enddo
  end subroutine host_differentiate
end program high_order_boundary_probe
