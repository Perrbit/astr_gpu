!+---------------------------------------------------------------------+
!| This module contains subroutines and variables related to turbulence|
!| model.                                                              |
!| ==============                                                      |
!| CHANGE RECORD                                                       |
!| -------------                                                       |
!| 09-Aug-2021  | Created by J. Fang @ STFC Daresbury Laboratory       |
!+---------------------------------------------------------------------+
module models
  !
  use commvar,only : im,jm,km,hm,Reynolds,num_species
  use parallel, only : mpirank,mpistop,irk,jrk,krk
  use commarray,only : tke,omg,miut,dtke,domg,dvel
  use constdef
  !
  implicit none
  !
  type :: komega_coef
    real(8) :: sigma_k1,sigma_k2,sigma_omega1,sigma_omega2,beta1,beta2,&
               beta_star,gamma1,gamma2,a1,b1,c1
    real(8) :: gamma,beta,prt
    real(8),allocatable,dimension(:,:,:) :: sigma_k,sigma_omega
  end type komega_coef
  real(8) :: kamma=0.41d0
  !
  type(komega_coef) :: komega
  !
  contains
  ! 
  !+-------------------------------------------------------------------+
  !| This subroutine is to allocate array for k-omega sst model.       |
  !+-------------------------------------------------------------------+
  !| ref: https://turbmodels.larc.nasa.gov/sst.html
  !|      https://www.openfoam.com/documentation/guides/latest/doc/guide-bcs-wall-turbulence-omegaWallFunction.html
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 09-Aug-2020: Created by J. Fang @ STFC Daresbury Laboratory       |
  !+-------------------------------------------------------------------+
  subroutine init_komegasst
    !
    integer :: lallo
    !
    allocate(   tke(-hm:im+hm,-hm:jm+hm,-hm:km+hm),   &
                omg(-hm:im+hm,-hm:jm+hm,-hm:km+hm),   &
               miut(0:im,0:jm,0:km),stat=lallo)
    if(lallo.ne.0) stop ' !! error at allocating tke,omg,miut'
    !
    allocate( dtke(0:im,0:jm,0:km,1:3),               &
              domg(0:im,0:jm,0:km,1:3),stat=lallo )
    if(lallo.ne.0) stop ' !! error at allocating dtke,domg'
    !
    allocate( komega%sigma_k(0:im,0:jm,0:km),               &
              komega%sigma_omega(0:im,0:jm,0:km),stat=lallo )
    if(lallo.ne.0) stop ' !! error at allocating sigma_k,sigma_omega'
    !
    !
    komega%sigma_k1     = 0.85d0
    komega%sigma_k2     = 1.d0
    komega%sigma_omega1 = 0.5d0
    komega%sigma_omega2 = 0.856d0
    komega%beta1        = 0.075d0
    komega%beta2        = 0.0828d0
    komega%beta_star    = 0.09d0
    komega%a1           = 0.31d0
    komega%b1           = 1.d0
    komega%c1           = 10.d0
    !
    komega%prt          = 0.9d0
    !
    komega%gamma1=5.d0/9.d0
    komega%gamma2=0.44d0
    !
    ! komega%gamma1=komega%beta1/komega%beta_star - &
    !               komega%sigma_omega1*kamma*kamma/sqrt(komega%beta_star)
    ! komega%gamma2=komega%beta2/komega%beta_star - &
    !               komega%sigma_omega2*kamma*kamma/sqrt(komega%beta_star)
    !
  end subroutine init_komegasst
  !+-------------------------------------------------------------------+
  !| The end of the subroutine init_komegasst.                         |
  !+-------------------------------------------------------------------+
  !
  !+-------------------------------------------------------------------+
  !| This subroutine is to obtain eddy viscousity of k-omega sst model |
  !+-------------------------------------------------------------------+
  !| ref: https://turbmodels.larc.nasa.gov/sst.html                    |
  !| Menter, F. R. 1994 Two-Equation Eddy-Viscosity Turbulence Models  |
  !|   for Engineering Applications. AIAA J. 32,1598-605.              |
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 09-08-2021: Created by J. Fang @ Warrington.                      |
  !+-------------------------------------------------------------------+
  subroutine src_komega

    use commarray,only : dis2wall,rho,tmp,vor,dvel,jacob,qrhs
    use fludyna,  only : miucal
    !
    ! local data
    integer :: i,j,k
    real(8) :: arg1,arg2,f1,f2,cdkomega,miu,sqrtk,dwall,vorti,ro,dkdomg
    real(8) :: var1,var2,var3,var4
    real(8) :: s11,s12,s13,s22,s23,s33,d11,d12,d13,d22,d23,d33,       &
               div,det,tau11,tau12,tau13,tau22,tau23,tau33,s
    real(8) :: produ_k,produ_omega,miueddy
    !
    do k=0,km
    do j=0,jm
    do i=0,im
      !
      miu=miucal(tmp(i,j,k))/Reynolds
      dwall=max(dis2wall(i,j,k),1.d-8)
      sqrtk=sqrt(tke(i,j,k))
      ro=rho(i,j,k)
      !
      dkdomg=dtke(i,j,k,1)*domg(i,j,k,1)+dtke(i,j,k,2)*domg(i,j,k,2) + &
             dtke(i,j,k,3)*domg(i,j,k,3)
      var2=2.d0*ro*komega%sigma_omega2/omg(i,j,k)*dkdomg
      cdkomega=max(var2,10.d-10)
      !
      var1=sqrtk/(komega%beta_star*omg(i,j,k)*dwall)
      var2=500.d0*miu/(ro*dwall*dwall*omg(i,j,k))
      var3=4.d0*ro*komega%sigma_omega2*tke(i,j,k)/(cdkomega*dwall*dwall)
      arg1=min(max(var1,var2),var3)
      arg2=max(2.d0*var1,var2)
      !
      f1=tanh(arg1**4)
      f2=tanh(arg2**2)
      !
      komega%sigma_k(i,j,k)    =komega%sigma_k1*f1+komega%sigma_k2*(1.d0-f1)
      komega%sigma_omega(i,j,k)=  komega%gamma1*f1+  komega%gamma2*(1.d0-f1)
      !
      komega%beta              =   komega%beta1*f1+   komega%beta2*(1.d0-f1)
      komega%gamma             =  komega%gamma1*f1+  komega%gamma2*(1.d0-f1)
      !
      vorti=sqrt(vor(i,j,k,1)**2+vor(i,j,k,2)**2+vor(i,j,k,3)**2)
      !
      s11=dvel(i,j,k,1,1)
      s12=0.5d0*(dvel(i,j,k,1,2)+dvel(i,j,k,2,1))
      s13=0.5d0*(dvel(i,j,k,1,3)+dvel(i,j,k,3,1))
      s22=dvel(i,j,k,2,2)
      s23=0.5d0*(dvel(i,j,k,2,3)+dvel(i,j,k,3,2))
      s33=dvel(i,j,k,3,3)
      !
      s=2.d0*(s11*s11+s22*s22+s33*s33+2.d0*(s12*s12+s13*s13*s23*s23))
      s=sqrt(s)
      !
      var4=max(komega%a1*omg(i,j,k),s*f2)
      !
      miut(i,j,k)=ro*komega%a1*tke(i,j,k)/var4
      miut(i,j,k)=min(miut(i,j,k),100.d0*miu)
      miut(i,j,k)=max(1.d-10,miut(i,j,k))
      !
      div=s11+s22+s33
      !
      det=num2d3*(miut(i,j,k)*div+ro*tke(i,j,k))
      tau11=2.d0*miut(i,j,k)*s11-det
      tau12=2.d0*miut(i,j,k)*s12
      tau13=2.d0*miut(i,j,k)*s13
      tau22=2.d0*miut(i,j,k)*s22-det
      tau23=2.d0*miut(i,j,k)*s23
      tau33=2.d0*miut(i,j,k)*s33-det
      !
      ! production term
      produ_k=      tau11*s11+tau22*s22+tau33*s33  +  &
              2.d0*(tau12*s12+tau13*s13+tau23*s23)
      produ_k=min(produ_k,20.d0*komega%beta_star*ro*tke(i,j,k)*omg(i,j,k))
      !
      var1=-komega%beta_star*ro*omg(i,j,k)*tke(i,j,k)
      !
      qrhs(i,j,k,6+num_species)=qrhs(i,j,k,6+num_species) +            &
                                             (produ_k+var1)*jacob(i,j,k)
      !
      miueddy=max(miut(i,j,k),1.d-10)
      produ_omega=komega%gamma*ro/miueddy*produ_k
      var1=-komega%beta*ro*omg(i,j,k)**2
      var2=2.d0*(1.d0-f1)*ro*komega%sigma_omega2/omg(i,j,k)*dkdomg
      !
      qrhs(i,j,k,7+num_species)=qrhs(i,j,k,7+num_species) +            &
                                    (produ_omega+var1+var2)*jacob(i,j,k)
      !
    enddo
    enddo
    enddo
    !
  end subroutine src_komega
  !+-------------------------------------------------------------------+
  !| The end of the subroutine src_komega.                             |
  !+-------------------------------------------------------------------+
  !
  !--------------------------------------------------------------------
  ! Baldwin-Lomax model on one wall-normal line.
  !
  ! Inputs
  !   n        : number of points on the wall-normal line
  !   y(n)     : wall distance, y(1)=0 at wall, increasing outward
  !   rho(n)   : density
  !   mu(n)    : molecular dynamic viscosity
  !   ut(n)    : tangential velocity component along the wall
  !   vt(n)    : second tangential velocity component (set to 0 for 2D)
  !   omega(n) : magnitude of local vorticity or shear measure
  !              for 2D BL, omega = abs(du_t/dy)
  !   tauw     : wall shear stress magnitude
  !
  ! Outputs
  !   mut(n)        : turbulent dynamic viscosity
  !   mut_inner(n)  : inner-layer contribution
  !   mut_outer(n)  : outer-layer contribution
  !   y_max         : location of F maximum used by the model
  !   f_max         : maximum of F(y)
  !
  ! Notes
  !   1) This routine assumes an attached boundary-layer-style BL model.
  !   2) For 3D wall-bounded flow, ut and vt should be the two velocity
  !      components tangent to the wall along the wall-normal cut.
  !   3) In ASTR, call this along each wall-normal stencil/ray.
  !--------------------------------------------------------------------
  subroutine baldwin_lomax_line

    use commarray,only : dis2wall,rho,tmp,dvel,vel,miut,bnorm_j0,x
    use parallel, only : ia,ja,ka,ig0,jg0,kg0,pmax,mpirankname,irk,jrk,krk
    use fludyna,  only : miucal
    use tecio

    real(8), parameter :: kappa  = 0.4d0
    real(8), parameter :: Aplus  = 26.d0
    real(8), parameter :: Ccp    = 1.6d0
    real(8), parameter :: Cwk    = 0.25d0
    real(8), parameter :: Ckleb  = 0.30d0
    real(8), parameter :: Kbl    = 0.0168d0
    real(8), parameter :: tinyv  = 1.0d-10

    integer :: i,j,k,imax
    real(8) :: miu,utau,nuw, yplus, Lm, fkleb, fwake
    real(8) :: umin, umax, omega,F
    real(8) :: miut_inner,miut_outer
    real(8),allocatable :: utaw(:,:),miuw(:,:),f_max(:,:),f2_max(:,:),y_max(:,:),udiff(:,:)

    logical,save :: lfirstcall=.true.

    if(lfirstcall) then
      allocate(   miut(0:im,0:jm,0:km) )
    endif

    allocate(utaw(0:ia,0:ka),miuw(0:ia,0:ka))

    utaw=0.d0
    miuw=0.d0

    if(jrk==0) then

      j=0

      do k=0,km
      do i=0,im
        miu=miucal(tmp(i,j,k))/Reynolds

        miuw(i+ig0,k+kg0)=miu

        utaw(i+ig0,k+kg0)=tau_cal(grad_u=dvel(i,j,k,:,:),normal=bnorm_j0(i,k,:),mu=miu)
        utaw(i+ig0,k+kg0)=sqrt(utaw(i+ig0,k+kg0)/rho(i,j,k))
      enddo
      enddo

    endif

    utaw=pmax(utaw)
    miuw=pmax(miuw)

    allocate(f_max(0:ia,0:ka),f2_max(0:ia,0:ka),y_max(0:ia,0:ka),udiff(0:ia,0:ka))

    f_max=0.d0
    y_max=0.d0
    udiff=0.d0
    do k=0,km
    do i=0,im
      do j=0,jm

        utau=utaw(i+ig0,k+kg0)

        nuw =miuw(i+ig0,k+kg0)

        yplus = rho(i,j,k) * dis2wall(i,j,k) * utau / max(nuw, tinyv)

        omega=abs(dvel(i,j,k,1,2)-dvel(i,j,k,2,1))

        F = dis2wall(i,j,k) * omega * (1.0d0 - exp(-yplus / Aplus))

        ! if(F>10.d0) print*,x(i,j,k,2),F

        if(F>=f_max(i+ig0,k+kg0)) then
          f_max(i+ig0,k+kg0)=F
          y_max(i+ig0,k+kg0)=dis2wall(i,j,k)
        endif

        udiff(i+ig0,k+kg0)=sqrt(vel(i,j,k,1)*vel(i,j,k,1) + vel(i,j,k,2)*vel(i,j,k,2) + vel(i,j,k,3)*vel(i,j,k,3))

      enddo
    enddo
    enddo

    udiff=pmax(udiff)

    f2_max=pmax(f_max)

    do k=0,km
    do i=0,im
      if(f_max(i+ig0,k+kg0)==f2_max(i+ig0,k+kg0)) then
      else
        y_max(i+ig0,k+kg0)=0.d0
      endif
    enddo
    enddo

    f_max=f2_max
    y_max=pmax(y_max)

      ! call tecbin('testout/tec_fymax'//mpirankname//'.plt',x(0:im,0,0:km,1),'x', &
      !                                                     x(0:im,0,0:km,2),'y', &
      !                                                     x(0:im,0,0:km,3),'z', &
      !                                                    f_max(ig0:ig0+im,kg0:kg0+km),'f_max', &
      !                                                    y_max(ig0:ig0+im,kg0:kg0+km),'y_max' )
    do k=0,km
    do j=0,jm
    do i=0,im

        utau=utaw(i+ig0,k+kg0)

        nuw =miuw(i+ig0,k+kg0)

        !--------------------------------------------------------------
        ! Outer-layer wake function:
        !   Fwake = min( y_max * F_max, Cwk * y_max * Udiff^2 / F_max )
        !--------------------------------------------------------------
        fwake = min(y_max(i+ig0,k+kg0) * f_max(i+ig0,k+kg0),  &
                          Cwk *y_max(i+ig0,k+kg0)* udiff(i+ig0,k+kg0)**2/ max(f_max(i+ig0,k+kg0), tinyv))


        yplus = rho(i,j,k) * dis2wall(i,j,k) * utau / max(nuw, tinyv)

        ! Mixing length
        Lm = kappa * dis2wall(i,j,k) * (1.0d0 - exp(-yplus / Aplus))

        ! Inner layer
        omega=abs(dvel(i,j,k,1,2)-dvel(i,j,k,2,1))
        miut_inner = rho(i,j,k) * Lm*Lm * omega

        ! Klebanoff intermittency factor
        fkleb = 1.0d0 / (1.0d0 + 5.5d0 * (Ckleb * dis2wall(i,j,k) / y_max(i+ig0,k+kg0))**6)

        ! Outer layer
        miut_outer = rho(i,j,k) * Kbl * Ccp * fwake * fkleb

        ! Baldwin-Lomax final value
        miut(i,j,k) = min(miut_inner, miut_outer)

    enddo
    enddo
    enddo

    if(lfirstcall) then
      ! call tecbin('testout/tec_miut'//mpirankname//'.plt',x(0:im,0:jm,0:km,1),'x', &
      !                                                    x(0:im,0:jm,0:km,2),'y', &
      !                                                    x(0:im,0:jm,0:km,3),'z', &
      !                                                    miut(0:im,0:jm,0:km),'miut')
      lfirstcall=.false.
    endif

    ! call mpistop

  end subroutine baldwin_lomax_line

  function tau_cal(grad_u, normal, mu) result(tau_mag)

    real(8), intent(in)  :: grad_u(3,3)
    real(8), intent(in)  :: normal(3)
    real(8), intent(in)  :: mu
    real(8) :: tau_mag

    real(8) :: n(3), nnorm
    real(8) :: sigma(3,3)
    real(8) :: traction(3)
    real(8) :: divu, tn
    real(8) :: tau_vec(3)
    integer :: i, j

    ! ---- normalize wall normal ----
    nnorm = sqrt(normal(1)**2 + normal(2)**2 + normal(3)**2)
    if (nnorm <= 1.0d-30) then
      tau_vec = 0.0d0
      tau_mag = 0.0d0
      return
    end if
    n = normal / nnorm

    ! ---- divergence ----
    divu = grad_u(1,1) + grad_u(2,2) + grad_u(3,3)

    ! ---- viscous stress tensor sigma ----
    sigma = 0.0d0
    do i = 1, 3
      do j = 1, 3
        sigma(i,j) = mu * (grad_u(i,j) + grad_u(j,i))
      end do
    end do

    sigma(1,1) = sigma(1,1) - (2.0d0/3.0d0) * mu * divu
    sigma(2,2) = sigma(2,2) - (2.0d0/3.0d0) * mu * divu
    sigma(3,3) = sigma(3,3) - (2.0d0/3.0d0) * mu * divu

    ! ---- traction = sigma . n ----
    traction = 0.0d0
    do i = 1, 3
      do j = 1, 3
        traction(i) = traction(i) + sigma(i,j) * n(j)
      end do
    end do

    ! ---- remove normal component to get tangential wall shear ----
    tn = traction(1)*n(1) + traction(2)*n(2) + traction(3)*n(3)

    tau_vec(1) = traction(1) - tn * n(1)
    tau_vec(2) = traction(2) - tn * n(2)
    tau_vec(3) = traction(3) - tn * n(3)

    tau_mag = sqrt(tau_vec(1)**2 + tau_vec(2)**2 + tau_vec(3)**2)

    return

  end function tau_cal
  !
end module models
!+---------------------------------------------------------------------+
!| The end of the module models.                                       |
!+---------------------------------------------------------------------+