!+---------------------------------------------------------------------+
!| This module contains subroutines to do interpolation.               |
!| ==============                                                      |
!| CHANGE RECORD                                                       |
!| -------------                                                       |
!| 22-Jul-2022  | Created by J. Fang @ Warrington                      |
!+---------------------------------------------------------------------+
module interp
  use astr_precision, only: real_kind
  !
  implicit none
  private :: real_kind
  !
  interface interlinear
    module procedure linear1d_s
    module procedure linear1d_a1
    module procedure linear1d_a2
    module procedure linear1d_arrayin
    module procedure linear1d_array2array
  end interface interlinear
  !
  contains
  !
  !+-------------------------------------------------------------------+
  !| This function is a linear interpolation function.                 |
  !+-------------------------------------------------------------------+
  function linear1d_s(xx1,xx2,yy1,yy2,xx) result(yy)
    !
    real(real_kind),intent(in) :: xx1,xx2,yy1,yy2,xx
    real(real_kind) :: yy
    !
    real(real_kind) :: var1
    !
    var1=(yy2-yy1)/(xx2-xx1)
    yy=var1*(xx-xx1)+yy1
    !
    return
    !
  end function linear1d_s
  !
  function linear1d_a1(xx1,xx2,yy1,yy2,xx) result(yy)
    !
    real(real_kind),intent(in) :: xx1,xx2,xx
    real(real_kind),intent(in) ::  yy1(:),yy2(:)
    real(real_kind) :: yy(1:size(yy1))
    !
    real(real_kind) :: var1
    !
    var1=(xx-xx1)/(xx2-xx1)
    yy=(yy2-yy1)*var1+yy1
    !
    return
    !
  end function linear1d_a1
  !
  function linear1d_a2(xx1,xx2,yy1,yy2,xx) result(yy)
    !
    real(real_kind),intent(in) :: xx1,xx2,xx
    real(real_kind),intent(in) ::  yy1(:,:),yy2(:,:)
    real(real_kind) :: yy(1:size(yy1,1),1:size(yy1,2))
    !
    real(real_kind) :: var1
    !
    var1=(xx-xx1)/(xx2-xx1)
    yy=(yy2-yy1)*var1+yy1
    !
    return
    !
  end function linear1d_a2
  !
  function linear1d_arrayin(x1,y1,xx,mode) result(yy)
    !
    real(real_kind),intent(in) :: x1(:),y1(:),xx
    real(real_kind) :: yy
    character(len=3),intent(in),optional :: mode
    !
    integer :: dim,i
    !
    dim=size(x1)
    !
    if(xx<x1(1)) then
      if(present(mode) .and. mode=='---') then
        yy=y1(1)
      else
        yy=2.e0_real_kind*y1(1)-y1(2)
      endif
    elseif(xx>=x1(dim)) then
      ! yy=linear1d_s(x1(dim-1),x1(dim),y1(dim-1),y1(dim),xx)
      if(present(mode) .and. mode=='---') then
        yy=y1(dim)
      else
        yy=2.e0_real_kind*y1(dim)-y1(dim-1)
      endif
    else
      do i=2,dim
        if(xx>=x1(i-1) .and. xx<x1(i)) then
          yy=linear1d_s(x1(i-1),x1(i),y1(i-1),y1(i),xx)
        endif
      enddo
    endif
    !
    return
    !
  end function linear1d_arrayin
  !+-------------------------------------------------------------------+
  !| The end of the function linear1d                                  |
  !+-------------------------------------------------------------------+
  !
  function linear1d_array2array(x,y,xx,mode) result(yy)

    real(real_kind),intent(in) :: x(:),y(:),xx(:)
    real(real_kind) :: yy(size(xx))
    character(len=3),intent(in),optional :: mode

    integer :: i,m,n,j
    character(len=3) :: act_mod

    m=size(x)
    n=size(xx)

    if(present(mode)) then
      act_mod=mode
    else
      act_mod='lin'
    endif

    do j=1,n

      if(xx(j)<=x(1)) then
        if(act_mod=='lin') then
          yy(j)=linear1d_s(x(1),x(2),y(1),y(2),xx(j))
        elseif(act_mod=='end') then
          yy(j)=y(1)
        elseif(act_mod=='dec') then
          yy(j)=y(1)*exp(-5.e0_real_kind*(x(1)-xx(j)))
        elseif(act_mod=='zer') then
          yy(j)=0.e0_real_kind
        else
          print*,'act_mod',act_mod
          stop ' erro 1 @ linear1d_array2array'
        endif
      elseif(xx(j)>=x(m)) then
        if(act_mod=='lin') then
          yy(j)=linear1d_s(x(m-1),x(m),y(m-1),y(m),xx(j))
        elseif(act_mod=='end') then
          yy(j)=y(m)
        elseif(act_mod=='dec') then
          yy(j)=y(m)*exp(-5.e0_real_kind*(xx(j)-x(m)))
        elseif(act_mod=='zer') then
          yy(j)=0.e0_real_kind
        else
          stop ' erro 2 @ linear1d_array2array'
        endif

      else
        do i=2,m
          if(xx(j)>=x(i-1) .and. xx(j)<=x(i)) then
            yy(j)=linear1d_s(x(i-1),x(i),y(i-1),y(i),xx(j))
            exit
          endif
        enddo
      endif

      if(abs(yy(j))<1.e-16_real_kind) yy(j)=0.e0_real_kind 

    enddo

    return

  end function linear1d_array2array
  !
end module interp
!+---------------------------------------------------------------------+
!| The end of the module interp.                                       |
!+---------------------------------------------------------------------+
