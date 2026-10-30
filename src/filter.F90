!+---------------------------------------------------------------------+
!| This module contains routines related to low-pass filter.           |
!+---------------------------------------------------------------------+
!| CHANGE RECORD                                                       |
!| -------------                                                       |
!| 26-05-2025  | Created by J. Fang @ Liverpool                        |
!+---------------------------------------------------------------------+
module filter
    use astr_precision, only: real_kind

    use commtype,only : compact_scheme

    implicit none
    private :: real_kind

    type(compact_scheme) :: filter_i,filter_j,filter_k
    type(compact_scheme) :: filter_ii,filter_jj,filter_kk

    real(real_kind),allocatable :: coef2i(:),coef4i(:),coef6i(:),coef8i(:),      &
                           coef10i(:),coefb(:,:),coefh(:,:),             &
                           coef10e(:),coef8e(:),coef6e(:),coef4e(:),     &
                           coef2e(:),coef4be(:,:),coef3be(:,:),coef6be(:,:)
    ! RHS filter coefficients
    contains

    !+-------------------------------------------------------------------+
    !| This subroutine is to initiate the comapct filter.                |
    !+-------------------------------------------------------------------+
    !| CHANGE RECORD                                                     |
    !| -------------                                                     |
    !| 26-05-2025  | Rewrite by J. Fang @ Liverpool.                     |
    !+-------------------------------------------------------------------+
    subroutine compact_filter_initiate(afilter,ntype,dim,note)

        use commvar, only : hm,alfa_filter
        use commfunc,only : tridiagonal_thomas_proprocess

        type(compact_scheme) :: afilter
        integer,intent(in) :: ntype,dim
        character(len=*),intent(in),optional :: note

        integer :: i_0,i_m
        real(real_kind) :: beter_0,beter_m
        real(real_kind),allocatable :: a(:),c(:)

        select case (ntype)
        case (1)
          ! the block with boundary at i==0
          i_0=0
          i_m=dim+3
          beter_0=0.98e0_real_kind
          beter_m=1.11e0_real_kind
        case (2)
          ! the block with boundary at i==im
          i_0=-3
          i_m=dim
          beter_0=1.11e0_real_kind
          beter_m=0.98e0_real_kind
        case (3)
          ! inner block
          i_0=-3
          i_m=dim+3
          beter_0=1.11e0_real_kind
          beter_m=1.11e0_real_kind
        case (4)
          ! the block with boundary at i=0 and i=im
          i_0=0
          i_m=dim
          beter_0=0.98e0_real_kind
          beter_m=0.98e0_real_kind
        case default
          print*, ' !! error 1 in subroutine compact_filter_initiate !'
        end select
  
        afilter%first_node=i_0
        afilter%last_node =i_m
        afilter%dimension =dim
        afilter%nbctype   =ntype
        
        call afilter%init()

        afilter%a=alfa_filter
        afilter%c=alfa_filter

        if(present(note) .and. note=='boundary_no_filter') then

          afilter%a(i_0) = 0.e0_real_kind
          afilter%c(i_0) = 0.e0_real_kind

          afilter%a(i_m) = 0.e0_real_kind
          afilter%c(i_m) = 0.e0_real_kind
        else
          afilter%a(i_0) = beter_0
          afilter%c(i_0) = beter_0

          afilter%a(i_m) = beter_m
          afilter%c(i_m) = beter_m
        endif

        afilter%ac=tridiagonal_thomas_proprocess(afilter%a,afilter%c)

    end subroutine compact_filter_initiate
    !+-------------------------------------------------------------------+
    !| The end of the subroutine compact_filter_initiate.                |
    !+-------------------------------------------------------------------+

    !+-------------------------------------------------------------------+
    !| This function is to apply compact low-pass filter to a input array|
    !+-------------------------------------------------------------------+
    !| CHANGE RECORD                                                     |
    !| -------------                                                     |
    !| 25-05-2025  | Rewrite by J. Fang @ Liverpool.                     |
    !+-------------------------------------------------------------------+
    function compact_filter(afilter,f,dim,note) result(ff)

        use commvar, only : hm
        use commfunc,only : tridiagonal_thomas_solver

        ! arguments
        type(compact_scheme),intent(in) :: afilter
        integer,intent(in) :: dim
        real(real_kind),intent(in) :: f(-hm:dim+hm)
        character(len=*),intent(in),optional :: note
        real(real_kind) :: ff(0:dim)

        ! local data
        integer :: l,i_0,i_m
        real(real_kind) :: alpha,var0,var1,var2,var3,var4,var5
        real(real_kind),pointer :: ac(:,:)
        real(real_kind),allocatable :: d(:),xx(:)

        i_0=afilter%first_node
        i_m=afilter%last_node

        d=compact_filter_rhs(afilter,f,dim,note)

        allocate(xx(i_0:i_m))

        xx=tridiagonal_thomas_solver(afilter%ac,d)

        ff(0:dim)=xx(0:dim)

        if(afilter%nbctype==1 .or. afilter%nbctype==4) ff(0)=f(0)
        if(afilter%nbctype==2 .or. afilter%nbctype==4) ff(dim)=f(dim)

    end function compact_filter
    !+-------------------------------------------------------------------+
    !| The end of the function compact_filter.                           |
    !+-------------------------------------------------------------------+

    !+-------------------------------------------------------------------+
    !| This function is to calculate R.H.S of the compact filter         |
    !+-------------------------------------------------------------------+
    !| CHANGE RECORD                                                     |
    !| -------------                                                     |
    !| 26-05-2025  | Rewrite by J. Fang @ Liverpool.                     |
    !+-------------------------------------------------------------------+
    function compact_filter_rhs(afilter,f,dim,note) result(d)

        use commvar, only : hm

        type(compact_scheme),intent(in) :: afilter
        integer,intent(in) :: dim
        real(real_kind),intent(in) :: f(-hm:dim+hm)
        character(len=*),intent(in),optional :: note
        real(real_kind),allocatable :: d(:)

        integer :: i_0,i_m,j,k,i_s,i_e,ntype
        real(real_kind) :: var0,var1,var2,var3,var4,var5


        i_0=afilter%first_node
        i_m=afilter%last_node
        ntype=afilter%nbctype

        allocate(d(i_0:i_m))

        if(ntype==1 .or. ntype==4) then

          ! physical boundary

          i_s=i_0+5

          do k=i_0,i_0+2
            var0=0.e0_real_kind
            do j=0,6
                var0=var0+coefb(k-i_0,j)*f(i_0+j)
            enddo
            d(k)=var0
          enddo

          j=i_0+3
          var0=f(j)+f(j)
          var1=f(j+1)+f(j-1)
          var2=f(j+2)+f(j-2)
          var3=f(j+3)+f(j-3)
          d(j)=coef6i(0)*var0+coef6i(1)*var1+coef6i(2)*var2+coef6i(3)*var3
    
          j=i_0+4
          var0=f(j)  +f(j)
          var1=f(j+1)+f(j-1)
          var2=f(j+2)+f(j-2)
          var3=f(j+3)+f(j-3)
          var4=f(j+4)+f(j-4)
          d(j)=coef8i(0)*var0+coef8i(1)*var1+coef8i(2)*var2 +              &
               coef8i(3)*var3+coef8i(4)*var4

        else

          ! interfaces
          
          i_s=i_0+3

          do k=i_0,i_0+2
            var0=0.e0_real_kind
            do j=0,10
              var0=var0+coefh(k-i_0,j)*f(i_0-2+j)
            enddo
            d(k)=var0
          enddo

        endif

        if(ntype==2 .or. ntype==4) then

          ! physical boundary
          i_e=i_m-5

          do k=i_m-2,i_m
            var0=0.e0_real_kind
            do j=0,6
              var0=var0+coefb(i_m-k,j)*f(i_m-j)
            enddo
            d(k)=var0
          enddo

          j=i_m-3
          var0=f(j)+f(j)
          var1=f(j+1)+f(j-1)
          var2=f(j+2)+f(j-2)
          var3=f(j+3)+f(j-3)
          d(j)=coef6i(0)*var0+coef6i(1)*var1+coef6i(2)*var2+coef6i(3)*var3
    
          j=i_m-4
          var0=f(j)  +f(j)
          var1=f(j+1)+f(j-1)
          var2=f(j+2)+f(j-2)
          var3=f(j+3)+f(j-3)
          var4=f(j+4)+f(j-4)
          d(j)=coef8i(0)*var0+coef8i(1)*var1+coef8i(2)*var2 +              &
               coef8i(3)*var3+coef8i(4)*var4

        else
          ! interfaces
          i_e=i_m-3

          do k=i_m-2,i_m
            var0=0.e0_real_kind
            do j=0,10
              var0=var0+coefh(i_m-k,j)*f(i_m+2-j)
            enddo
            d(k)=var0
          enddo


        endif

        if(present(note) .and. note=='boundary_no_filter') then
            d(i_0)=f(i_0)
            d(i_m)=f(i_m)
        endif

        do j=i_s,i_e
    
          var0=f(j)+f(j)
          var1=f(j+1)+f(j-1)
          var2=f(j+2)+f(j-2)
          var3=f(j+3)+f(j-3)
          var4=f(j+4)+f(j-4)
          var5=f(j+5)+f(j-5)
          !
          d(j)=coef10i(0)*var0+coef10i(1)*var1+coef10i(2)*var2+     &
               coef10i(3)*var3+coef10i(4)*var4+coef10i(5)*var5
    
        end do
    
    end function compact_filter_rhs
    !+-------------------------------------------------------------------+
    !| The end of the function compact_filter_initiate.                  |
    !+-------------------------------------------------------------------+

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! This subroutine is used to generate the coefficient for the filter,
    ! and boundary filter is included.
    ! The boundary filter is lower order (0-6-6-6-8...) using one side
    ! filter.
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! Writen by Fang Jian, 2008-11-05.
    ! Moved to this module, 30-05-2025
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine filter_coefficient_cal(alfa,beter_halo,beter_bouond)
      !
      real(real_kind),intent(in) :: alfa,beter_halo,beter_bouond
      !
      allocate(coef2i(0:1),coef4i(0:2),coef6i(0:3),coef8i(0:4),coef10i(0:5))
      allocate(coefb(0:4,0:8),coefh(0:4,0:10))
      !
      ! inernal coefficent
      ! 2nd-order
      coef2i(0)=( 1.e0_real_kind + 2.e0_real_kind*alfa)   /4.e0_real_kind
      coef2i(1)=( 1.e0_real_kind + 2.e0_real_kind*alfa)   /4.e0_real_kind
      !
      ! 4th-order
      coef4i(0)=( 5.e0_real_kind + 6.e0_real_kind*alfa)   /16.e0_real_kind
      coef4i(1)=( 1.e0_real_kind + 2.e0_real_kind*alfa)   /4.e0_real_kind
      coef4i(2)=(-1.e0_real_kind + 2.e0_real_kind*alfa)   /16.e0_real_kind
      !
      ! 6th-order
      coef6i(0)=(11.e0_real_kind +10.e0_real_kind*alfa)   /32.e0_real_kind
      coef6i(1)=(15.e0_real_kind +34.e0_real_kind*alfa)   /64.e0_real_kind
      coef6i(2)=(-3.e0_real_kind + 6.e0_real_kind*alfa)   /32.e0_real_kind
      coef6i(3)=( 1.e0_real_kind - 2.e0_real_kind*alfa)   /64.e0_real_kind
      !
      ! 8th-order
      coef8i(0)=(93.e0_real_kind +70.e0_real_kind*alfa)   /256.e0_real_kind
      coef8i(1)=( 7.e0_real_kind +18.e0_real_kind*alfa)   /32.e0_real_kind
      coef8i(2)=(-7.e0_real_kind +14.e0_real_kind*alfa)   /64.e0_real_kind
      coef8i(3)=( 1.e0_real_kind - 2.e0_real_kind*alfa)   /32.e0_real_kind
      coef8i(4)=(-1.e0_real_kind + 2.e0_real_kind*alfa)   /256.e0_real_kind
      !
      ! 10th-order
      coef10i(0)=(193.e0_real_kind +126.e0_real_kind*alfa)/512.e0_real_kind
      coef10i(1)=(105.e0_real_kind +302.e0_real_kind*alfa)/512.e0_real_kind
      coef10i(2)=(-15.e0_real_kind + 30.e0_real_kind*alfa)/128.e0_real_kind
      coef10i(3)=( 45.e0_real_kind - 90.e0_real_kind*alfa)/1024.e0_real_kind
      coef10i(4)=( -5.e0_real_kind + 10.e0_real_kind*alfa)/512.e0_real_kind
      coef10i(5)=(  1.e0_real_kind -  2.e0_real_kind*alfa)/1024.e0_real_kind
      !
      ! boundary coef10ifficent
      !
      ! 8th-order for the point3
      coefb(3,0)=(  1.e0_real_kind - 2.e0_real_kind*alfa)  /256.e0_real_kind
      coefb(3,1)=( -1.e0_real_kind + 2.e0_real_kind*alfa)  /32.e0_real_kind
      coefb(3,2)=(  7.e0_real_kind +50.e0_real_kind*alfa)  /64.e0_real_kind
      coefb(3,3)=( 25.e0_real_kind +14.e0_real_kind*alfa)  /32.e0_real_kind
      coefb(3,4)=( 35.e0_real_kind +58.e0_real_kind*alfa)  /128.e0_real_kind
      coefb(3,5)=( -7.e0_real_kind +14.e0_real_kind*alfa)  /32.e0_real_kind
      coefb(3,6)=(  7.e0_real_kind -14.e0_real_kind*alfa)  /64.e0_real_kind
      coefb(3,7)=( -1.e0_real_kind + 2.e0_real_kind*alfa)  /32.e0_real_kind
      coefb(3,8)=(  1.e0_real_kind - 2.e0_real_kind*alfa)  /256.e0_real_kind
      !
      ! 6-order asymmetry scheme for point 2
      coefb(2,0)=( -1.e0_real_kind + 2.e0_real_kind*alfa)  /64.e0_real_kind
      coefb(2,1)=(  3.e0_real_kind +26.e0_real_kind*alfa)  /32.e0_real_kind
      coefb(2,2)=( 49.e0_real_kind +30.e0_real_kind*alfa)  /64.e0_real_kind
      coefb(2,3)=(  5.e0_real_kind + 6.e0_real_kind*alfa)  /16.e0_real_kind
      coefb(2,4)=(-15.e0_real_kind +30.e0_real_kind*alfa)  /64.e0_real_kind
      coefb(2,5)=(  3.e0_real_kind - 6.e0_real_kind*alfa)  /32.e0_real_kind
      coefb(2,6)=( -1.e0_real_kind + 2.e0_real_kind*alfa)  /64.e0_real_kind
      coefb(2,7)=0.e0_real_kind
      coefb(2,8)=0.e0_real_kind
      !
      !
      ! ! 4th-order asymmetry scheme for point 1
      ! coefb(1,0)=( 1.d0 +14.d0*alfa)  /16.d0
      ! coefb(1,1)=( 3.d0 + 2.d0*alfa)  /4.d0
      ! coefb(1,2)=( 3.d0 + 2.d0*alfa)  /8.d0
      ! coefb(1,3)=(-1.d0 + 2.d0*alfa)  /4.d0
      ! coefb(1,4)=( 1.d0 - 2.d0*alfa)  /16.d0
      ! coefb(1,5)=0.d0
      ! coefb(1,6)=0.d0
      ! coefb(1,7)=0.d0
      ! coefb(1,8)=0.d0
      ! 6-order asymmetry scheme for point 1
      coefb(1,0)=( 1.e0_real_kind +62.e0_real_kind*alfa)  /64.e0_real_kind
      coefb(1,1)=(29.e0_real_kind + 6.e0_real_kind*alfa)  /32.e0_real_kind
      coefb(1,2)=(15.e0_real_kind +34.e0_real_kind*alfa)  /64.e0_real_kind
      coefb(1,3)=(-5.e0_real_kind +10.e0_real_kind*alfa)  /16.e0_real_kind
      coefb(1,4)=(15.e0_real_kind -30.e0_real_kind*alfa)  /64.e0_real_kind
      coefb(1,5)=(-3.e0_real_kind + 6.e0_real_kind*alfa)  /32.e0_real_kind
      coefb(1,6)=( 1.e0_real_kind - 2.e0_real_kind*alfa)  /64.e0_real_kind
      coefb(1,7)=0.e0_real_kind
      coefb(1,8)=0.e0_real_kind

      ! 6-order asymmetry scheme for point 0
      coefb(0,0)=( 63.e0_real_kind + 1.e0_real_kind*beter_bouond)  /64.e0_real_kind
      coefb(0,1)=(  3.e0_real_kind +29.e0_real_kind*beter_bouond)  /32.e0_real_kind
      coefb(0,2)=(-15.e0_real_kind +15.e0_real_kind*beter_bouond)  /64.e0_real_kind
      coefb(0,3)=(  5.e0_real_kind - 5.e0_real_kind*beter_bouond)  /16.e0_real_kind
      coefb(0,4)=(-15.e0_real_kind +15.e0_real_kind*beter_bouond)  /64.e0_real_kind
      coefb(0,5)=(  3.e0_real_kind - 3.e0_real_kind*beter_bouond)  /32.e0_real_kind
      coefb(0,6)=( -1.e0_real_kind + 1.e0_real_kind*beter_bouond)  /64.e0_real_kind
      coefb(0,7)=0.e0_real_kind
      coefb(0,8)=0.e0_real_kind
      !
      ! halo nodes
      coefh(0,0)=(-1.e0_real_kind+1.e0_real_kind*beter_halo)/1024.e0_real_kind
      coefh(0,1)=(5.e0_real_kind-5.e0_real_kind*beter_halo)/512.e0_real_kind
      coefh(0,2)=(979.e0_real_kind+45.e0_real_kind*beter_halo)/1024.e0_real_kind
      coefh(0,3)=(15.e0_real_kind+113.e0_real_kind*beter_halo)/128.e0_real_kind
      coefh(0,4)=(-105.e0_real_kind+105.e0_real_kind*beter_halo)/512.e0_real_kind
      coefh(0,5)=(63.e0_real_kind-63.e0_real_kind*beter_halo)/256.e0_real_kind
      coefh(0,6)=(-105.e0_real_kind+105.e0_real_kind*beter_halo)/512.e0_real_kind
      coefh(0,7)=(15.e0_real_kind-15.e0_real_kind*beter_halo)/128.e0_real_kind
      coefh(0,8)=(-45.e0_real_kind+45.e0_real_kind*beter_halo)/1024.e0_real_kind
      coefh(0,9)=(5.e0_real_kind-5.e0_real_kind*beter_halo)/512.e0_real_kind
      coefh(0,10)=(-1.e0_real_kind+1.e0_real_kind*beter_halo)/1024.e0_real_kind
  
      coefh(1,0)=(1.e0_real_kind-2.e0_real_kind*alfa)/1024.e0_real_kind
      coefh(1,1)=(-5.e0_real_kind+10.e0_real_kind*alfa)/512.e0_real_kind
      coefh(1,2)=(45.e0_real_kind+934.e0_real_kind*alfa)/1024.
      coefh(1,3)=(113.e0_real_kind+30.e0_real_kind*alfa)/128.e0_real_kind
      coefh(1,4)=(105.e0_real_kind+302.e0_real_kind*alfa)/512.
      coefh(1,5)=(-63.e0_real_kind+126.e0_real_kind*alfa)/256.
      coefh(1,6)=(105.e0_real_kind-210.e0_real_kind*alfa)/512.
      coefh(1,7)=(-15.e0_real_kind+30.e0_real_kind*alfa)/128.e0_real_kind
      coefh(1,8)=(45.e0_real_kind-90.e0_real_kind*alfa)/1024.e0_real_kind
      coefh(1,9)=(-5.e0_real_kind+10.e0_real_kind*alfa)/512.e0_real_kind
      coefh(1,10)=(1.e0_real_kind-2.e0_real_kind*alfa)/1024.e0_real_kind
  
      coefh(2,0)=(-1.e0_real_kind+2.e0_real_kind*alfa)/1024.e0_real_kind
      coefh(2,1)=(5.e0_real_kind-10.e0_real_kind*alfa)/512.e0_real_kind
      coefh(2,2)=(-45.e0_real_kind+90.e0_real_kind*alfa)/1024.e0_real_kind
      coefh(2,3)=(15.e0_real_kind+98.e0_real_kind*alfa)/128.e0_real_kind
      coefh(2,4)=(407.e0_real_kind+210.e0_real_kind*alfa)/512.e0_real_kind
      coefh(2,5)=(63.e0_real_kind+130.e0_real_kind*alfa)/256.e0_real_kind
      coefh(2,6)=(-105.e0_real_kind+210.e0_real_kind*alfa)/512.e0_real_kind
      coefh(2,7)=(15.e0_real_kind-30.e0_real_kind*alfa)/128.e0_real_kind
      coefh(2,8)=(-45.e0_real_kind+90.e0_real_kind*alfa)/1024.e0_real_kind
      coefh(2,9)=(5.e0_real_kind-10.e0_real_kind*alfa)/512.e0_real_kind
      coefh(2,10)=(-1.e0_real_kind+2.e0_real_kind*alfa)/1024.e0_real_kind
  
  
    end subroutine filter_coefficient_cal
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! End of the Subroutine filter_coefficient_cal.
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !
    subroutine filter_coefficient_explicit

      allocate(coef10e(0:5),coef8e(0:4),coef6e(0:3),coef4e(0:2),coef2e(0:1))
      allocate(coef3be(0:1,0:3),coef4be(0:1,0:4),coef6be(0:2,0:6))

      coef3be=0.e0_real_kind
      coef4be=0.e0_real_kind
      coef6be=0.e0_real_kind

      ! explicit filter
      coef4be(0,0)=(15.e0_real_kind )/16.e0_real_kind
      coef4be(0,1)=( 1.e0_real_kind )/4.e0_real_kind
      coef4be(0,2)=(-3.e0_real_kind )/8.e0_real_kind
      coef4be(0,3)=( 1.e0_real_kind )/4.e0_real_kind
      coef4be(0,4)=(-1.e0_real_kind )/16.e0_real_kind
      !
      coef4be(1,0)=( 1.e0_real_kind )/16.e0_real_kind
      coef4be(1,1)=( 3.e0_real_kind )/4.e0_real_kind
      coef4be(1,2)=( 3.e0_real_kind )/8.e0_real_kind
      coef4be(1,3)=(-1.e0_real_kind )/4.e0_real_kind
      coef4be(1,4)=( 1.e0_real_kind )/16.e0_real_kind
      !
      coef3be(0,0)=( 7.e0_real_kind )/8.e0_real_kind
      coef3be(0,1)=( 3.e0_real_kind )/8.e0_real_kind
      coef3be(0,2)=(-3.e0_real_kind )/8.e0_real_kind
      coef3be(0,3)=( 1.e0_real_kind )/8.e0_real_kind
      !
      coef3be(1,0)=( 1.e0_real_kind )/8.e0_real_kind
      coef3be(1,1)=( 5.e0_real_kind )/8.e0_real_kind
      coef3be(1,2)=( 3.e0_real_kind )/8.e0_real_kind
      coef3be(1,3)=(-1.e0_real_kind )/8.e0_real_kind
      !
      ! 2nd-order
      coef2e(0)=0.25e0_real_kind
      coef2e(1)=0.25e0_real_kind
      !
      ! 4th-order
      coef4e(0)=( 5.e0_real_kind )/16.e0_real_kind
      coef4e(1)=( 1.e0_real_kind )/4.e0_real_kind
      coef4e(2)=(-1.e0_real_kind )/16.e0_real_kind
      !
      ! 6th-order
      coef6e(0)=(11.e0_real_kind) /32.e0_real_kind
      coef6e(1)=(15.e0_real_kind) /64.e0_real_kind
      coef6e(2)=(-3.e0_real_kind) /32.e0_real_kind
      coef6e(3)=( 1.e0_real_kind) /64.e0_real_kind
      !
      ! 8th-order
      coef8e(0)=(93.e0_real_kind) /256.e0_real_kind
      coef8e(1)=( 7.e0_real_kind) /32.e0_real_kind
      coef8e(2)=(-7.e0_real_kind) /64.e0_real_kind
      coef8e(3)=( 1.e0_real_kind) /32.e0_real_kind
      coef8e(4)=(-1.e0_real_kind) /256.e0_real_kind
      !
      ! 10th-order
      coef10e(0)=(193.e0_real_kind )/512.e0_real_kind
      coef10e(1)=(105.e0_real_kind )/512.e0_real_kind
      coef10e(2)=(-15.e0_real_kind )/128.e0_real_kind
      coef10e(3)=( 45.e0_real_kind )/1024.e0_real_kind
      coef10e(4)=( -5.e0_real_kind )/512.e0_real_kind
      coef10e(5)=(  1.e0_real_kind )/1024.e0_real_kind

      ! Sixth-order one-sided closure at the first interior point.
      coef6be(1,0)=(  1.e0_real_kind )/64.e0_real_kind
      coef6be(1,1)=( 29.e0_real_kind )/32.e0_real_kind
      coef6be(1,2)=( 15.e0_real_kind )/64.e0_real_kind
      coef6be(1,3)=( -5.e0_real_kind )/16.e0_real_kind
      coef6be(1,4)=( 15.e0_real_kind )/64.e0_real_kind
      coef6be(1,5)=( -3.e0_real_kind )/32.e0_real_kind
      coef6be(1,6)=(  1.e0_real_kind )/64.e0_real_kind
  
      ! Sixth-order one-sided closure at the second interior point.
      coef6be(2,0)=( -1.e0_real_kind )  /64.e0_real_kind
      coef6be(2,1)=(  3.e0_real_kind )  /32.e0_real_kind
      coef6be(2,2)=( 49.e0_real_kind )  /64.e0_real_kind
      coef6be(2,3)=(  5.e0_real_kind )  /16.e0_real_kind
      coef6be(2,4)=(-15.e0_real_kind )  /64.e0_real_kind
      coef6be(2,5)=(  3.e0_real_kind )  /32.e0_real_kind
      coef6be(2,6)=( -1.e0_real_kind )  /64.e0_real_kind

    end subroutine filter_coefficient_explicit
    !+-------------------------------------------------------------------+
    !| This function is to apply 6th-order low-pass filter to a  array   |
    !+-------------------------------------------------------------------+
    !| CHANGE RECORD                                                     |
    !| -------------                                                     |
    !| 10-02-2021  | Created by J. Fang @ Warrington.                    |
    !| 30-05-2025  | Moved to this module by J. Fang @ Liverpool.        |
    !+-------------------------------------------------------------------+
    real(real_kind) function filter8exp(a)
      !
      real(real_kind) :: a(-4:4)
      !
      real(real_kind) :: coef(0:4)
      integer :: n
      !
      coef(0)= 93.e0_real_kind
      coef(1)= 56.e0_real_kind
      coef(2)=-28.e0_real_kind
      coef(3)=  8.e0_real_kind
      coef(4)= -1.e0_real_kind
      !
      filter8exp=0.e0_real_kind
      do n=0,4
        filter8exp=filter8exp+coef(n)/256.e0_real_kind*(a(n)+a(-n))
      enddo 
      !
    end function filter8exp
    !+-------------------------------------------------------------------+
    !| The end of the function filter8exp.                               |
    !+-------------------------------------------------------------------+
    !
    function spafilter10exp(f,ntype,dim,wall_kind) result(ff)
      !
      use commvar, only: hm
      use boundary_high_order, only: boundary_wall_filter6

      ! arguments
      integer,intent(in) :: ntype,dim
      integer,intent(in),optional :: wall_kind
      real(real_kind),intent(in) :: f(-hm:dim+hm)
      real(real_kind) :: ff(0:dim)
      !
      ! local data
      integer :: ii,m,i_s,i_e
      logical :: wall_filter
      !
      ff=0.e0_real_kind
      i_s=0
      i_e=dim
      wall_filter=.false.
      if(present(wall_kind)) wall_filter=wall_kind==41.or.wall_kind==42.or.wall_kind==421
      !
      ! A physical face has no valid halo. Preserve the boundary node and
      ! use the documented 0-6-6-6-8-10 explicit closure toward the interior.
      if(ntype==1 .or. ntype==4) then
        ff(0)=f(0)
        if(wall_filter) then
          do ii=1,4
            ff(ii)=boundary_wall_filter6(f(0:8),ii,wall_kind/=41)
          enddo
        else
        do m=0,6
          ff(1)=ff(1)+coef6be(1,m)*f(m)
          ff(2)=ff(2)+coef6be(2,m)*f(m)
        enddo

        ii=3
        do m=0,3
          ff(ii)=ff(ii)+coef6e(m)*(f(ii-m)+f(ii+m))
        enddo

        ii=4
        do m=0,4
          ff(ii)=ff(ii)+coef8e(m)*(f(ii-m)+f(ii+m))
        enddo
        endif

        i_s=5
      endif

      if(ntype==2 .or. ntype==4) then
        ff(dim)=f(dim)
        if(wall_filter) then
          do ii=1,4
            if(ntype==4.and.dim-ii<=4) cycle
            ff(dim-ii)=boundary_wall_filter6(f(dim:dim-8:-1),ii,wall_kind/=41)
          enddo
        else
        do m=0,6
          ff(dim-1)=ff(dim-1)+coef6be(1,m)*f(dim-m)
          ff(dim-2)=ff(dim-2)+coef6be(2,m)*f(dim-m)
        enddo

        ii=dim-3
        do m=0,3
          ff(ii)=ff(ii)+coef6e(m)*(f(ii-m)+f(ii+m))
        enddo

        ii=dim-4
        ff(ii)=0.e0_real_kind
        do m=0,4
          ff(ii)=ff(ii)+coef8e(m)*(f(ii-m)+f(ii+m))
        enddo
        endif

        i_e=dim-5
      endif

      do ii=i_s,i_e
        do m=0,5
          ff(ii)=ff(ii)+coef10e(m)*(f(ii-m)+f(ii+m))
        enddo
      enddo
      !
      return
      !
    end function spafilter10exp
    !+-------------------------------------------------------------------+
    !| The end of the function spafilter10exp.                           |
    !+-------------------------------------------------------------------+
    !
    !+-------------------------------------------------------------------+
    !| This function is to apply 6th-order low-pass filter to a  array   |
    !+-------------------------------------------------------------------+
    !| CHANGE RECORD                                                     |
    !| -------------                                                     |
    !| 06-07-2021  | Created by J. Fang @ Warrington.                    |
    !| 30-05-2025  | Moved to this module by J. Fang @ Liverpool.        |
    !+-------------------------------------------------------------------+
    function spafilter6exp(f,ntype,dim) result(ff)
      use boundary_high_order, only: high_order_boundary_enabled,boundary_filter6
      !
      use commvar, only: hm
      
      ! arguments
      integer,intent(in) :: ntype,dim
      real(real_kind),intent(in) :: f(-hm:dim+hm)
      real(real_kind) :: ff(0:dim)
      !
      ! local data
      integer :: ii,m
      real(real_kind) :: line(0:6)

      if(high_order_boundary_enabled) then
        do ii=0,dim
          if((ntype==1.or.ntype==4).and.ii<3) then
            ff(ii)=boundary_filter6(f(0:6),ii)
          elseif((ntype==2.or.ntype==4).and.ii>dim-3) then
            do m=0,6
              line(m)=f(dim-m)
            enddo
            ff(ii)=boundary_filter6(line,dim-ii)
          else
            ff(ii)=11.e0_real_kind/16.e0_real_kind*f(ii)+ &
              15.e0_real_kind/64.e0_real_kind*(f(ii-1)+f(ii+1))- &
              3.e0_real_kind/32.e0_real_kind*(f(ii-2)+f(ii+2))+ &
              1.e0_real_kind/64.e0_real_kind*(f(ii-3)+f(ii+3))
          endif
        enddo
        return
      endif
      !
      ff=0.e0_real_kind
      !
      select case(ntype)
      case(1)
      
        ii=0
        do m=0,4
          ff(ii)=ff(ii)+coef4be(0,m)*f(m)
        enddo
        ! ff(ii)=f(ii)
        !
        ii=1
        do m=0,4
          ff(ii)=ff(ii)+coef4be(1,m)*f(m)
        enddo
        ! do m=0,1
        !   ff(ii)=ff(ii)+coef2e(m)*(f(ii-m)+f(ii+m))
        ! enddo
        !
        ii=2
        do m=0,2
          ff(ii)=ff(ii)+coef4e(m)*(f(ii-m)+f(ii+m))
        enddo
        do ii=3,dim
          !
          do m=0,3
            ff(ii)=ff(ii)+coef6e(m)*(f(ii-m)+f(ii+m))
          enddo
          !
        enddo
      !   !
      case(2)
        !
        do ii=0,dim-3
          !
          do m=0,3
            ff(ii)=ff(ii)+coef6e(m)*(f(ii-m)+f(ii+m))
          enddo
          !
        enddo
        ii=dim-2
        do m=0,2
          ff(ii)=ff(ii)+coef4e(m)*(f(ii-m)+f(ii+m))
        enddo
        !
        ii=dim-1
        !
        ! do m=0,1
        !   ff(ii)=ff(ii)+coef2e(m)*(f(ii-m)+f(ii+m))
        ! enddo
        do m=0,4
          ff(ii)=ff(ii)+coef4be(1,m)*f(dim-m)
        enddo
        
        ii=dim
        do m=0,4
          ff(ii)=ff(ii)+coef4be(0,m)*f(dim-m)
        enddo
        ! ff(ii)=f(ii)
        !
      case(3)
        !
        do ii=0,dim
          !
          do m=0,3
            ff(ii)=ff(ii)+coef6e(m)*(f(ii-m)+f(ii+m))
          enddo
          !
        enddo
        !
      case(4)
        !
        ii=0
        do m=0,4
          ff(ii)=ff(ii)+coef4be(0,m)*f(m)
        enddo
        !
        ii=1
        do m=0,4
          ff(ii)=ff(ii)+coef4be(1,m)*f(m)
        enddo
        !
        ii=2
        do m=0,2
          ff(ii)=ff(ii)+coef4e(m)*(f(ii-m)+f(ii+m))
        enddo
        !
        do ii=3,dim-3
          do m=0,3
            ff(ii)=ff(ii)+coef6e(m)*(f(ii-m)+f(ii+m))
          enddo
        enddo
        !
        ii=dim-2
        do m=0,2
          ff(ii)=ff(ii)+coef4e(m)*(f(ii-m)+f(ii+m))
        enddo
        !
        ii=dim-1
        do m=0,4
          ff(ii)=ff(ii)+coef4be(1,m)*f(dim-m)
        enddo
        !
        ii=dim
        do m=0,4
          ff(ii)=ff(ii)+coef4be(0,m)*f(dim-m)
        enddo
        !
      case default
        stop '!! ERROR @ spafilter6exp'
      end select
      !
      return
      !
    end function spafilter6exp
    !+-------------------------------------------------------------------+
    !| The end of the function spafilter6exp.                            |
    !+-------------------------------------------------------------------+
    
end module filter
!+---------------------------------------------------------------------+
!| The end of the module filter.                                       |
!+---------------------------------------------------------------------+
