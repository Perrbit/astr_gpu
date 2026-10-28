#ifndef BOUNDARY_HIGH_ORDER_PROC
#define BOUNDARY_HIGH_ORDER_PROC pure
#endif
  use astr_precision, only: real_kind
  implicit none
  private
#ifdef BOUNDARY_HIGH_ORDER_DEVICE
  logical,device,public :: high_order_boundary_enabled_d=.false.
#else
  logical,save,public :: high_order_boundary_enabled=.false.
#endif
  public :: boundary_derivative6, boundary_neumann7
  public :: boundary_derivative6_weights, boundary_neumann7_weights
  public :: boundary_mp_flux6, boundary_filter6, boundary_incoming_rate7
  public :: boundary_interface_position
  public :: boundary_normal_terms6

  ! FD numerical-flux moments, not interpolation of physical flux at a midpoint.
  real(real_kind),parameter :: boundary_flux6_weights(0:7,0:2)=reshape([ &
    60.e0_real_kind,669.e0_real_kind,-591.e0_real_kind,459.e0_real_kind, &
    -241.e0_real_kind,74.e0_real_kind,-10.e0_real_kind,0.e0_real_kind, &
    -10.e0_real_kind,130.e0_real_kind,459.e0_real_kind,-241.e0_real_kind, &
    109.e0_real_kind,-31.e0_real_kind,4.e0_real_kind,0.e0_real_kind, &
    4.e0_real_kind,-38.e0_real_kind,214.e0_real_kind,319.e0_real_kind, &
    -101.e0_real_kind,25.e0_real_kind,-3.e0_real_kind,0.e0_real_kind], [8,3])/420.e0_real_kind
  real(real_kind),parameter :: difference7(0:7)=[ &
    -1.e0_real_kind,7.e0_real_kind,-21.e0_real_kind,35.e0_real_kind, &
    -35.e0_real_kind,21.e0_real_kind,-7.e0_real_kind,1.e0_real_kind]
  real(real_kind),parameter :: incoming_flux_strength(0:2)=[ &
    0.125e0_real_kind,0.25e0_real_kind,0.125e0_real_kind]

  ! Match the centered D6 seventh moment (36), not only moments zero to six.
  ! The common leading error is needed when the gradient is differentiated again.
  real(real_kind),parameter :: boundary_derivative6_weights(0:7,0:2)=reshape([ &
    -156.e0_real_kind,423.e0_real_kind,-639.e0_real_kind,715.e0_real_kind, &
    -540.e0_real_kind,261.e0_real_kind,-73.e0_real_kind,9.e0_real_kind, &
    -9.e0_real_kind,-84.e0_real_kind,171.e0_real_kind,-135.e0_real_kind, &
    85.e0_real_kind,-36.e0_real_kind,9.e0_real_kind,-1.e0_real_kind, &
    1.e0_real_kind,-17.e0_real_kind,-56.e0_real_kind,115.e0_real_kind, &
    -65.e0_real_kind,29.e0_real_kind,-8.e0_real_kind,1.e0_real_kind], [8,3])/60.e0_real_kind

  real(real_kind),parameter :: boundary_neumann7_weights(0:7)=[ &
    -1089.e0_real_kind,2940.e0_real_kind,-4410.e0_real_kind,4900.e0_real_kind, &
    -3675.e0_real_kind,1764.e0_real_kind,-490.e0_real_kind,60.e0_real_kind]/420.e0_real_kind
contains
  BOUNDARY_HIGH_ORDER_PROC subroutine boundary_normal_terms6(state,velocity,pressure,jacobian,metric, &
      derivative,rest)
    real(real_kind),intent(in) :: state(5,0:7),velocity(3,0:7),pressure(0:7), &
      jacobian(0:7),metric(3,0:7)
    real(real_kind),intent(out) :: derivative(5),rest(5)
    real(real_kind) :: flux(5,0:7),cartesian(5,3),delta_metric(3),line(0:7),un
    integer :: s,m,d
    do s=0,7
      un=sum(metric(:,s)*velocity(:,s))
      flux(1,s)=jacobian(s)*state(1,s)*un
      flux(2:4,s)=jacobian(s)*(state(2:4,s)*un+metric(:,s)*pressure(s))
      flux(5,s)=jacobian(s)*(state(5,s)+pressure(s))*un
    enddo
    do d=1,3
      cartesian(1,d)=state(d+1,0)
      cartesian(2:4,d)=state(2:4,0)*velocity(d,0)
      cartesian(d+1,d)=cartesian(d+1,d)+pressure(0)
      cartesian(5,d)=(state(5,0)+pressure(0))*velocity(d,0)
      line=metric(d,:)*jacobian
      delta_metric(d)=-boundary_derivative6(line,0)
    enddo
    do m=1,5
      line=flux(m,:)
      derivative(m)=-boundary_derivative6(line,0)
    enddo
    rest=matmul(cartesian,delta_metric)
  end subroutine boundary_normal_terms6

  BOUNDARY_HIGH_ORDER_PROC subroutine boundary_interface_position(node,dim,ntype,row,side)
    integer,intent(in) :: node,dim,ntype
    integer,intent(out) :: row,side
    row=-1; side=0
    if((ntype==1.or.ntype==4).and.node>=0.and.node<3) then
      row=node; side=1
    elseif((ntype==2.or.ntype==4).and.node>=dim-3.and.node<dim) then
      row=dim-1-node; side=-1
    endif
  end subroutine boundary_interface_position

  BOUNDARY_HIGH_ORDER_PROC function boundary_derivative6(f,row) result(derivative)
    real(real_kind),intent(in) :: f(0:7)
    integer,intent(in) :: row
    real(real_kind) :: derivative
    integer :: j
    ! f is ordered from a physical boundary into the domain; row is 0, 1 or 2.
    ! No h factor: the caller supplies the same computational-coordinate scaling
    ! and physical metric as its interior derivative.
    derivative=0.e0_real_kind
    do j=0,7
      derivative=derivative+boundary_derivative6_weights(j,row)*(f(j)-f(row))
    enddo
  end function boundary_derivative6

  BOUNDARY_HIGH_ORDER_PROC function boundary_neumann7(f,inward_slope) result(value)
    real(real_kind),intent(in) :: f(1:7),inward_slope
    real(real_kind) :: value,residual
    integer :: j
    ! inward_slope is the derivative with respect to the inward node index.
    ! A seventh-order derivative constraint gives an eighth-order boundary value.
    residual=inward_slope
    do j=1,7
      residual=residual-boundary_neumann7_weights(j)*(f(j)-f(1))
    enddo
    value=f(1)+residual/boundary_neumann7_weights(0)
  end function boundary_neumann7

  BOUNDARY_HIGH_ORDER_PROC function boundary_mp_flux6(f,row,incoming) result(flux)
    real(real_kind),intent(in) :: f(0:7)
    integer,intent(in) :: row
    logical,intent(in) :: incoming
    real(real_kind) :: flux,correction
    integer :: j
    ! Physical nodes are ordered inward. This is the unlimited candidate only;
    ! shock/critical-node fallback belongs to the production reconstruction caller.
    flux=f(row)
    correction=0.e0_real_kind
    do j=0,7
      flux=flux+boundary_flux6_weights(j,row)*(f(j)-f(row))
      correction=correction+difference7(j)*(f(j)-f(row))
    enddo
    if(incoming) flux=flux-incoming_flux_strength(row)*correction
  end function boundary_mp_flux6

  BOUNDARY_HIGH_ORDER_PROC function boundary_filter6(f,row) result(value)
    real(real_kind),intent(in) :: f(0:6)
    integer,intent(in) :: row
    real(real_kind) :: value,residual
    real(real_kind),parameter :: difference6(0:6)=[ &
      1.e0_real_kind,-6.e0_real_kind,15.e0_real_kind,-20.e0_real_kind, &
      15.e0_real_kind,-6.e0_real_kind,1.e0_real_kind]
    integer :: j
    ! Physical endpoints remain owned by their boundary condition.
    value=f(row)
    if(row==0) return
    residual=0.e0_real_kind
    do j=0,6
      residual=residual+difference6(j)*(f(j)-f(row))
    enddo
    if(row==2) residual=-residual
    value=value+residual/64.e0_real_kind
  end function boundary_filter6

  BOUNDARY_HIGH_ORDER_PROC subroutine boundary_incoming_rate7(primitive,gamma,hn,pinv,lambda, &
      incoming,rhs,jacobian,normal_lodi,crosses_front,rate)
    real(real_kind),intent(in) :: primitive(5,0:7),gamma,hn,pinv(5,5),lambda(5),rhs(5), &
      jacobian,normal_lodi(5)
    logical,intent(in) :: incoming(5),crosses_front
    real(real_kind),intent(out) :: rate(5)
    real(real_kind) :: derivative(5),dq(5),projected_rhs(5),velocity(3),rho
    integer :: j,m
    derivative=0.e0_real_kind
    if(crosses_front) then
      derivative=primitive(:,1)-primitive(:,0)
    else
      do j=1,7
        derivative=derivative+boundary_neumann7_weights(j)*(primitive(:,j)-primitive(:,0))
      enddo
    endif
    derivative=derivative/hn
    rho=primitive(1,0)
    velocity=primitive(2:4,0)
    dq(1)=derivative(1)
    dq(2:4)=velocity*derivative(1)+rho*derivative(2:4)
    dq(5)=derivative(5)/(gamma-1.e0_real_kind)+ &
      0.5e0_real_kind*sum(velocity**2)*derivative(1)+rho*sum(velocity*derivative(2:4))
    projected_rhs=matmul(pinv,rhs)/jacobian
    rate=projected_rhs
    do m=1,5
      ! rhs/J=-R L_n+G: keep G for every incoming wave, not only entropy/shear.
      if(incoming(m)) rate(m)=projected_rhs(m)+normal_lodi(m)-lambda(m)*sum(pinv(m,:)*dq)
    enddo
  end subroutine boundary_incoming_rate7
