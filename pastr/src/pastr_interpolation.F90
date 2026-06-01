module pastr_interpolation

    use iso_fortran_env, only: wp => real64

    implicit none

    interface interpolat
      module procedure linear1d_s
      module procedure linear1d_search_array
      module procedure linear1d_array2array
    end interface interpolat

contains

    pure function linear1d_s(xx1,xx2,yy1,yy2,xx) result(yy)
      real(wp),intent(in) :: xx1,xx2,yy1,yy2,xx
      real(wp) :: yy
      real(wp) :: var1
      var1=(yy2-yy1)/(xx2-xx1)
      yy=var1*(xx-xx1)+yy1
      return
    end function linear1d_s

    pure function linear1d_search_array(x,y,xx) result(yy)

      real(wp),intent(in) :: x(:),y(:),xx
      real(wp) :: yy

      integer :: i,m

      m=size(x)

      if(xx<=x(1)) then
        yy=linear1d_s(x(1),x(2),y(1),y(2),xx)
      elseif(xx>=x(m)) then
        yy=linear1d_s(x(m-1),x(m),y(m-1),y(m),xx)
      else
        do i=2,m
          if(xx>=x(i-1) .and. xx<=x(i)) then
            yy=linear1d_s(x(i-1),x(i),y(i-1),y(i),xx)
            exit
          endif
        enddo
      endif

      return

   end function linear1d_search_array

   function linear1d_array2array(x,y,xx,mode) result(yy)

      real(wp),intent(in) :: x(:),y(:),xx(:)
      real(wp) :: yy(size(xx))
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
            yy(j)=y(1)*exp(-5.d0*(x(1)-xx(j)))
          elseif(act_mod=='zer') then
            yy(j)=0.d0
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
            yy(j)=y(m)*exp(-5.d0*(xx(j)-x(m)))
          elseif(act_mod=='zer') then
            yy(j)=0.d0
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

        if(abs(yy(j))<1.d-16) yy(j)=0.d0 

      enddo

      return

   end function linear1d_array2array

end module pastr_interpolation