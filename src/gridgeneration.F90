!+---------------------------------------------------------------------+
!| This module contains subroutines of generating grid.                |
!+---------------------------------------------------------------------+
!| CHANGE RECORD                                                       |
!| -------------                                                       |
!| 12-02-2021  | Created by J. Fang                                    |
!+---------------------------------------------------------------------+
module gridgeneration
  use astr_precision, only: real_kind
  !
  use constdef
  !
  implicit none
  private :: real_kind
  !
  contains
  !
  !+-------------------------------------------------------------------+
  !| This subroutine is the main entrance of the grid generation.      |
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 12-02-2021  | Created by J. Fang @ Warrington                     |
  !+-------------------------------------------------------------------+
  subroutine gridgen
    !
    use parallel, only : mpirank
    use commvar,  only : flowtype,lreadgrid,nondimen,gridfile,ref_len,ka
    use readwrite,only : readgrid,writegrid,xdmfwriter
    use userdefine,only: udf_grid
    use benchmark_runtime, only: benchmark_field_io_disabled
    use output_runtime, only: output_resource_path
#ifdef ASTR_AIR5_CHEMISTRY
    use chemistry_hbl_geometry, only: read_air5_hbl_domain
    real(real_kind) :: hbl_lengths(3)
#endif
    !
    if(lreadgrid) then
      call readgrid(trim(output_resource_path('grid.h5',trim(gridfile))))
    else
      if(flowtype(1:3)=='tgv') then
        call gridcube(ref_len*2.e0_real_kind*pi,ref_len*2.e0_real_kind*pi,ref_len*2.e0_real_kind*pi)
      elseif(trim(flowtype)=='jet') then
        call gridjet
      elseif(trim(flowtype)=='hit') then
        call gridcube(2.e0_real_kind*pi,2.e0_real_kind*pi,2.e0_real_kind*pi)
      elseif(trim(flowtype)=='2dvort') then
        call gridcube(20.e0_real_kind,10.e0_real_kind,1.e0_real_kind)
      elseif(trim(flowtype)=='accutest') then
        call gridcube(10.e0_real_kind,1.e0_real_kind,1.e0_real_kind)
      elseif(trim(flowtype)=='shuosher' .or. trim(flowtype)=='openshock') then
        if(ka==0) then
          call grid1d(-5.e0_real_kind,5.e0_real_kind)
        else
          call gridsod3d(-5.e0_real_kind,5.e0_real_kind,1.e0_real_kind,1.e0_real_kind)
        endif
      elseif(trim(flowtype)=='sod') then
        if(ka==0) then
          call grid1d(-5.e0_real_kind,5.e0_real_kind)
        else
          call gridsod3d(-5.e0_real_kind,5.e0_real_kind,1.e0_real_kind,1.e0_real_kind)
        endif
      elseif(trim(flowtype)=='riem2d') then
        call gridcube(1.e0_real_kind,1.e0_real_kind,0.e0_real_kind)
      elseif(trim(flowtype)=='windtunn') then
        call gridsandbox
      elseif(trim(flowtype)=='channel') then
        call grichan(2.e0_real_kind*pi*ref_len,2.e0_real_kind*ref_len,pi*ref_len)
      elseif(trim(flowtype)=='0dreactor') then
        call gridhitflame(mode='cuboid')
      elseif(trim(flowtype)=='air5postshock' .or. &
             trim(flowtype)=='air5normalshock') then
        call gridcube(ref_len,ref_len,ref_len)
      elseif(trim(flowtype)=='air5hbl' .or. trim(flowtype)=='air5sbli') then
#ifdef ASTR_AIR5_CHEMISTRY
        call read_air5_hbl_domain(flowtype,ref_len,hbl_lengths, &
          trim(output_resource_path('air5_hbl_domain.dat','datin/air5_hbl_domain.dat')))
        call gridcube(hbl_lengths(1),hbl_lengths(2),hbl_lengths(3))
#else
        if(trim(flowtype)=='air5hbl') then
          call gridcube(20.e0_real_kind*ref_len,8.e0_real_kind*ref_len,2.e0_real_kind*ref_len)
        else
          call gridcube(80.e0_real_kind*ref_len,8.e0_real_kind*ref_len,2.e0_real_kind*ref_len)
        endif
#endif
      elseif(trim(flowtype)=='air5wave' .or. trim(flowtype)=='air5reactor' .or. &
             trim(flowtype)=='air5tgv' .or. &
             trim(flowtype)=='air5shocktube' .or. &
             trim(flowtype)=='air5advection' .or. trim(flowtype)=='air5difflayer' .or. &
             trim(flowtype)=='air5evpulse') then
        call gridcube(ref_len*2.e0_real_kind*pi,ref_len*2.e0_real_kind*pi,ref_len*2.e0_real_kind*pi)
      elseif(trim(flowtype)=='1dflame') then
        call gridcube(1.e-2_real_kind,1.e-3_real_kind,0.e0_real_kind)
      elseif(trim(flowtype)=='h2supersonic') then
        call gridsupersonicjet
      elseif(trim(flowtype)=='rti') then
        if(ka==0) then
          call gridcube(0.25e0_real_kind,1.e0_real_kind,0.e0_real_kind)
        else
          call gridcube(0.25e0_real_kind,1.e0_real_kind,0.25e0_real_kind)
        endif
      elseif(trim(flowtype)=='ldcavity') then
        if(ka==0) then
          call gridcube(1.e0_real_kind,1.e0_real_kind,0.e0_real_kind)
        else
          call gridcube(1.e0_real_kind,1.e0_real_kind,1.e0_real_kind)
        endif
      else
        !call gridcube(0.03d0,0.03d0,0.03d0)
        call udf_grid
        ! print*,trim(flowtype),' is not defined @ gridgen'
        ! stop ' !! error at gridgen' 
      endif
      !
      if(.not.benchmark_field_io_disabled()) call writegrid(trim(gridfile))
      !
    endif
    !
    ! call xdmfwriter(gridh5file=trim(gridfile),mode='grid')
    !
  end subroutine gridgen
  !+-------------------------------------------------------------------+
  !| The end of the subroutine gridgen.                                |
  !+-------------------------------------------------------------------+
  !
  !+-------------------------------------------------------------------+
  !| This subroutine is to generate a mesh for jet flow.               |
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 24-02-2021  | Created by J. Fang @ Warrington                     |
  !+-------------------------------------------------------------------+
  subroutine gridjet
  !
    use commvar,  only : im,jm,km,gridfile,ia,ja,ka
    use parallel, only : ig0,jg0,kg0,lio,irk,irkm
    use commarray,only : x
    use hdf5io
    !
    ! local data
    real(real_kind) :: lx,ly,lz
    integer :: i,j,k,in
    real(real_kind) :: x1d(0:im)
    !
    lx=50.e0_real_kind
    ly=15.e0_real_kind
    lz=15.e0_real_kind
    !
    do i=0,im
      x1d(i)=lx/real(ia,real_kind)*real(i+ig0,real_kind)
    enddo
    !
    if(irk==irkm) then
      in=im-25
      call spongstretch(im-in,lx+10.e0_real_kind,x1d(in-2:im))
    endif
    !
    do k=0,km
    do j=0,jm
    do i=0,im
      !
      x(i,j,k,1)=x1d(i)
      x(i,j,k,2)=ly/real(ja,real_kind)*real(j+jg0,real_kind)-0.5e0_real_kind*ly
      if(ka==0) then
        x(i,j,k,3)=0.e0_real_kind !lz/real(ka,8)*real(k+kg0,8)
      else
        x(i,j,k,3)=lz/real(ka,real_kind)*real(k+kg0,real_kind)-0.5e0_real_kind*lz
      endif
      !
    enddo
    enddo
    enddo
    !
    if(lio) print*,' ** jet grid generated'
    !
  end subroutine gridjet
  !+-------------------------------------------------------------------+
  !| The end of the subroutine gridjet.                                |
  !+-------------------------------------------------------------------+
  !
  !+-------------------------------------------------------------------+
  !| This subroutine is to generate a cubic mesh.                      |
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 12-02-2021  | Created by J. Fang @ Warrington                     |
  !+-------------------------------------------------------------------+
  subroutine grid1d(xmin,xmax)
    !
    use commvar,  only : im,jm,km,ia,ja,ka
    use parallel, only : ig0,jg0,kg0,lio
    use commarray,only : x
    use hdf5io
    !
    ! arguments
    real(real_kind),intent(in) :: xmin,xmax
    !
    ! local data
    integer :: i,j,k
    !
    do k=0,km
    do j=0,jm
    do i=0,im
      x(i,j,k,1)=(xmax-xmin)/real(ia,real_kind)*real(i+ig0,real_kind)+xmin
      if(ja==0) then
        x(i,j,k,2)=0.e0_real_kind
      else
        x(i,j,k,2)=1.e0_real_kind/real(ja,real_kind)*real(j+jg0,real_kind)
      endif
      x(i,j,k,3)=0.e0_real_kind
      !
    enddo
    enddo
    enddo
    !
    if(lio) print*,' ** 1-D grid generated'
    !
  end subroutine grid1d
  !
  subroutine gridsod3d(xmin,xmax,ly,lz)
    !
    use commvar,  only : im,jm,km,ia,ja,ka
    use parallel, only : ig0,jg0,kg0,lio
    use commarray,only : x
    !
    ! arguments
    real(real_kind),intent(in) :: xmin,xmax,ly,lz
    !
    ! local data
    integer :: i,j,k
    real(real_kind) :: dx,dy,dz
    !
    dx=(xmax-xmin)/real(ia,real_kind)
    dy=ly/real(ja,real_kind)
    dz=lz/real(ka,real_kind)
    !
    do k=0,km
    do j=0,jm
    do i=0,im
      x(i,j,k,1)=dx*real(i+ig0,real_kind)+xmin
      x(i,j,k,2)=dy*real(j+jg0,real_kind)
      x(i,j,k,3)=dz*real(k+kg0,real_kind)
      !
    enddo
    enddo
    enddo
    !
    if(lio) print*,' ** forced 3-D shock-tube grid generated'
    !
  end subroutine gridsod3d
  !
  subroutine gridhitflame(mode)
    !
    use commvar,  only : im,jm,km,gridfile,ia,ja,ka
    use parallel, only : ig0,jg0,kg0,lio,bcast
    use commarray,only : x
    use hdf5io
    !
    ! arguments
    character(len=*),intent(in) :: mode
    !
    ! local data
    integer :: i,j,k
    real(real_kind) :: flamethickness,lx,ly,lz
    !
    if(mpirank==0) then
      open(12,file='datin/userinput.txt')
      read(12,*)flamethickness
      close(12)
      print*, ' ** flamethickness =',flamethickness
    endif
    !
    call bcast(flamethickness)
    !
    if(mode=='cuboid') then
      lx=12.e0_real_kind*5.3e0_real_kind*flamethickness
      ly=5.3e0_real_kind*flamethickness
      lz=5.3e0_real_kind*flamethickness
    elseif(mode=='cubic') then
      lx=5.3e0_real_kind*flamethickness
      ly=5.3e0_real_kind*flamethickness
      lz=5.3e0_real_kind*flamethickness
    else
      stop ' !! error1 @ gridhitflame'
    endif
    !
    do k=0,km
    do j=0,jm
    do i=0,im
      x(i,j,k,1)=lx/real(ia,real_kind)*real(i+ig0,real_kind)
      if(ja==0) then
        x(i,j,k,2)=ly
      else
        x(i,j,k,2)=ly/real(ja,real_kind)*real(j+jg0,real_kind)
      endif
      if(ka==0) then
        x(i,j,k,3)=0.e0_real_kind
      else
        x(i,j,k,3)=lz/real(ka,real_kind)*real(k+kg0,real_kind)
      endif
      !
    enddo
    enddo
    enddo
    !
    if(lio) print*,' ** cubic grid generated'
    !
  end subroutine gridhitflame
  !
  subroutine gridcube(lx,ly,lz)
    !
    use commvar,  only : im,jm,km,gridfile,ia,ja,ka
    use parallel, only : ig0,jg0,kg0,lio
    use commarray,only : x
    use hdf5io
    !
    ! arguments
    real(real_kind),intent(in) :: lx,ly,lz
    !
    ! local data
    integer :: i,j,k
    real(real_kind) :: dx,dy,dz
    !
    dx=lx/real(ia,real_kind)
    if(ja==0) then
      dy=0.e0_real_kind
    else
      dy=ly/real(ja,real_kind)
    endif
    if(ka==0) then
      dz=0.e0_real_kind
    else
      dz=lz/real(ka,real_kind)
    endif
    !
    do k=0,km
    do j=0,jm
    do i=0,im
      x(i,j,k,1)=dx*real(i+ig0,real_kind)
      if(ja==0) then
        x(i,j,k,2)=ly
      else
        x(i,j,k,2)=dy*real(j+jg0,real_kind)
      endif
      if(ka==0) then
        x(i,j,k,3)=0.e0_real_kind
      else
        x(i,j,k,3)=dz*real(k+kg0,real_kind)
      endif
      !
    enddo
    enddo
    enddo
    !
    if(lio) print*,' ** cubic grid generated'
    !
  end subroutine gridcube
  !+-------------------------------------------------------------------+
  !| The end of the subroutine gridcube.                               |
  !+-------------------------------------------------------------------+
  !
  subroutine grichan(lx,ly,lz)
    !
    use commvar,  only : im,jm,km,gridfile,ia,ja,ka,ref_vel,ref_len
    use parallel, only : ig0,jg0,kg0,lio,pmax,mpistop
    use commarray,only : x
    use commfunc, only : argtanh
    !
    real(real_kind),intent(in) :: lx,ly,lz
    !
    integer :: n,i,j,k
    real(real_kind) :: varc,var1,var2,Retau,dx,yy(0:jm),xmax,ymax,dymax,zmax
    !
    Retau=185.e0_real_kind/ref_len
    !
    ! varc=1.02d0
    varc=1.07e0_real_kind
    do j=0,jm
      !
      var1=argtanh(1.e0_real_kind/varc)
      var2=2.e0_real_kind*(j+jg0)/(ja)*1.e0_real_kind-1.e0_real_kind
      !
      yy(j)=0.5e0_real_kind*ly*(1.e0_real_kind+varc*tanh(var1*var2))
      !
    end do
    !
    do k=0,km
    do j=0,jm
    do i=0,im
      !
      x(i,j,k,1)=lx/real(ia,real_kind)*real(i+ig0,real_kind)
      !
      x(i,j,k,2)=yy(j)
      !
      if(ka==0) then
        x(i,j,k,3)=0.e0_real_kind
      else
        x(i,j,k,3)=lz/real(ka,real_kind)*real(k+kg0,real_kind)
      endif
      !
    end do
    end do
    end do
    !
    do j=1,jm
      dymax=max(dymax,yy(j)-yy(j-1))
    enddo
    !
    xmax=pmax(x(im,0,0,1))
    ymax=pmax(yy(jm))
    dymax=pmax(dymax)
    zmax=pmax(x(0,0,km,3))
    !
    if(lio) then
      !
      write(*,'(2X,62A)')('-',i=1,62)
      write(*,'(18X,A)')' ** channel grid resolution **'
      write(*,'(2(A,F13.7))')'  ** y1+=',yy(1)*Retau,                  &
                             '   ymax+=',dymax*Retau
      write(*,'(2(A,F13.7))')'  ** dx+=',(x(1,0,0,1)-x(0,0,0,1))*Retau,&
                               '     lx+=',xmax*Retau
      write(*,'(2(A,F13.7))')'  ** dz+=',(x(0,0,1,3)-x(0,0,0,3))*Retau,&
                               '     lz+=',zmax*Retau
      write(*,'(2X,62A)')('-',i=1,62)
      !
      write(*,'(A)')'  ** channel grid generated'
      !
    endif
    !
  end subroutine grichan
  !+-------------------------------------------------------------------+
  !| The end of the subroutine grichan.                                |
  !+-------------------------------------------------------------------+
  !
  !+-------------------------------------------------------------------+
  !| This subroutine is to generate grid for jet simulation.           |
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 04-Jan-2020: Created by J. Fang @ Warrington.                     |
  !+-------------------------------------------------------------------+
  subroutine gridsupersonicjet
    !
#ifdef COMB
    use commvar, only: dj_i,dco_i,im,jm,km,ia,ja,ka
    use parallel, only : ig0,jg0,kg0
    use commarray,only : x
    !
    ! local data
    real(real_kind),allocatable :: yn(:),xn(:),rn(:)
    real(real_kind) :: varc,var1,var2,varmin,varmax,dy,xmax,ymax,zmax,scale
    integer :: i,j,k,in,nd
    !
    !+--------+
    !| begin  |
    !+--------+
    xmax=118.e-3_real_kind
    ymax=70.8e-3_real_kind
    zmax=70.8e-3_real_kind
    ! NOTE: im, jm ,zm must be even numbers
    ! xmax=44.0d-3
    ! ymax=40.0d-3
    ! zmax=40.0d-3
    !
    allocate(yn(0:ja),xn(0:ia),rn(0:ja))
    !
    ! number of cells in a diameter
    nd=8*24
    ! dy=16.d-3/real(nd,8)
    dy=8.e0_real_kind*dj_i/real(nd,real_kind)
    in=nd/2
    do j=1,in
      rn(j)=dy*j
    enddo
    !
    call spongstretch(ja/2-in,ymax/2,rn(in-2:ja/2))
    !
    yn(:)=0.e0_real_kind
    do j=1,ja/2
      yn(j)=yn(j-1)+(rn(ja/2-j+1)-rn(ja/2-j))
    enddo
    !
    do j=ja/2+1,ja
      yn(j)=yn(j-1)+(rn(j-ja/2)-rn(j-ja/2-1))
    enddo
    !
    ! in=nd+22*24
    ! do i=0,in
    !   xn(i)=dy*i
    ! enddo
    ! call spongstretch(ia-in,xmax,xn(in-2:ia))
    do i=0,ia
      xn(i)=xmax/ia*i
    enddo
    !
    !
    do k=0,km
    do j=0,jm
    do i=0,im
      x(i,j,k,1)=xn(i+ig0) 
      x(i,j,k,2)=yn(j+jg0)
      x(i,j,k,3)=yn(k+kg0)
      ! z(i,j,k)=(zmax/km)*k
    end do
    end do
    end do
    !+--------+
    !| end    |
    !+--------+
    !
    return
    !
#endif
  end subroutine gridsupersonicjet
  !+-------------------------------------------------------------------+
  !| The end of the subroutine gridjet.                                |
  !+-------------------------------------------------------------------+
  !
  !+-------------------------------------------------------------------+
  !| This subroutine is to generate a sandbox grid for wind tunnel.    |
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 23-Jun-2022: Created by J. Fang @ Warrington.                     |
  !+-------------------------------------------------------------------+
  subroutine gridsandbox
    !
    use parallel,  only : mpistop,ig0,jg0,kg0,lio,mpirank
    use commvar,   only : immbody,nsolid,ia,ja,ka,im,jm,km,ndims
    use commarray, only : x
    use stlaio
    use tecio
    !
    integer :: imm,jmm,kmm,im4,jm4,km4,isp
    integer :: i,j,k,jbody
    real(real_kind) :: lx_solid,ly_solid,lz_solid,len_chara,x0,y0,z0
    real(real_kind) :: xmin,ymin,zmin,xmax,ymax,zmax
    real(real_kind),allocatable,dimension(:) :: xx,yy,zz
    real(real_kind),allocatable,dimension(:,:,:) :: xa,ya,za
    !
    xmin=1.e10_real_kind; xmax=-1.e10_real_kind
    ymin=1.e10_real_kind; ymax=-1.e10_real_kind
    zmin=1.e10_real_kind; zmax=-1.e10_real_kind
    do jbody=1,size(immbody)
      xmin=min(xmin,immbody(jbody)%xmin(1))
      ymin=min(ymin,immbody(jbody)%xmin(2))
      zmin=min(zmin,immbody(jbody)%xmin(3))
      !
      xmax=max(xmax,immbody(jbody)%xmax(1))
      ymax=max(ymax,immbody(jbody)%xmax(2))
      zmax=max(zmax,immbody(jbody)%xmax(3))
    enddo
    !
    lx_solid=xmax-xmin
    ly_solid=ymax-ymin
    lz_solid=zmax-zmin
    !
    len_chara=1.e0_real_kind !sqrt(lx_solid**2+ly_solid**2)
    ! lx_solid=2.091906d0
    ! ly_solid=2.d0
    ! lz_solid=8.291200d-2
    !
    if(mpirank==0) print*,' **           solid size:',lx_solid,ly_solid,lz_solid
    if(mpirank==0) print*,' **  characteristic size:',len_chara
    if(mpirank==0) print*,' **            mesh size:',ia,ja,ka
    !
    lx_solid=lx_solid
    ly_solid=ly_solid
    lz_solid=lz_solid
    !
    allocate(xx(0:ia),yy(0:ja),zz(0:ka))
    !
    isp=20
    !
    imm=(ia-isp)/2
    jmm=ja/2
    kmm=ka/2
    !
    im4=imm/2
    jm4=jmm/2
    km4=kmm/2
    !
    ! core region 
    do i=imm,imm+im4
      xx(i)=lx_solid/real(im4,real_kind)*real(i-imm,real_kind)
    enddo
    !
    ! effecive region
    call spongstretch(imm-im4,5.e0_real_kind*lx_solid,xx(imm+im4-2:imm+imm))
    !
    do i=0,imm-1
      xx(i)=-xx(imm+imm-i)
    enddo
    !
    ! sponge region
    call spongstretch(ia-2*imm,10.e0_real_kind*lx_solid,xx(imm+imm-2:ia))
    !
    x0=xx(0)
    xx=xx-x0
    !
    if(mpirank==0) then
      print*,' ** -------------------- mesh report --------------------'
      print*,'    x direction'
      print*,'       core zone '
      write(*,'(2(A,1X,F12.8))')'          x:',xx(imm-im4),' ~ ',xx(imm+im4)
      write(*,'(2(A,1X,I0))')   '          i:',imm-im4,' ~ ', imm+im4
      write(*,'(1(A,1X,F12.8))')'         dx:',xx(imm+im4)-xx(imm+im4-1)
      print*,'       effecive zone '
      write(*,'(2(A,1X,F12.8))')'          x:',xx(0),' ~ ',xx(imm+imm)
      write(*,'(2(A,1X,I0))')   '          i:',0,' ~ ', imm+imm
      write(*,'(1(A,1X,F12.8))')'         dx:',xx(imm+imm)-xx(imm+imm-1)
      print*,'       sponge zone '
      write(*,'(2(A,1X,F12.8))')'          x:',xx(imm+imm),' ~ ',xx(ia)
      write(*,'(2(A,1X,I0))')   '          i:',imm+imm,' ~ ', ia
      write(*,'(1(A,1X,F12.8))')'         dx:',xx(ia)-xx(ia-1)
      print*,' ** -----------------------------------------------------'
    endif
    !
    ! core regjon 
    do j=jmm,jmm+jm4
      yy(j)=ly_solid/real(jm4,real_kind)*real(j-jmm,real_kind)
    enddo
    !
    ! effecive regjon
    call spongstretch(jmm-jm4,5.e0_real_kind*ly_solid,yy(jmm+jm4-2:jmm+jmm))
    !
    do j=0,jmm-1
      yy(j)=-yy(jmm+jmm-j)
    enddo
    y0=yy(0)
    yy=yy-y0
    !
    if(mpirank==0) then
      print*,'    y djrectjon'
      print*,'       core zone '
      write(*,'(2(A,1x,F12.8))')'          y:',yy(jmm-jm4),' ~ ',yy(jmm+jm4)
      write(*,'(2(A,1x,i0))')   '          j:',jmm-jm4,' ~ ', jmm+jm4
      write(*,'(1(A,1x,F12.8))')'         dy:',yy(jmm+jm4)-yy(jmm+jm4-1)
      print*,'       effecive zone '
      write(*,'(2(A,1x,F12.8))')'          y:',yy(0),' ~ ',yy(jmm+jmm)
      write(*,'(2(A,1x,i0))')   '          j:',0,' ~ ', jmm+jmm
      write(*,'(1(A,1x,F12.8))')'         dy:',yy(jmm+jmm)-yy(jmm+jmm-1)
      print*,' ** -----------------------------------------------------'
    endif
    !
    if(ndims==3) then
      !
      ! core regkon 
      do k=kmm,kmm+km4
        zz(k)=lz_solid/real(km4,real_kind)*real(k-kmm,real_kind)
      enddo
      !
      ! effeckve regkon
      ! call spongstretch(kmm-km4,5.d0*lz_solid,zz(kmm+km4-2:kmm+kmm))
      call spongstretch(kmm-km4,5.e0_real_kind,zz(kmm+km4-2:kmm+kmm))
      !
      do k=0,kmm-1
        zz(k)=-zz(kmm+kmm-k)
      enddo
      if(mpirank==0) then
        print*,'    z dkrectkon'
        print*,'       core zone '
        write(*,'(2(A,1x,F12.8))')'          z:',zz(kmm-km4),' ~ ',zz(kmm+km4)
        write(*,'(2(A,1x,i0))')   '          k:',kmm-km4,' ~ ', kmm+km4
        write(*,'(1(A,1x,F12.8))')'         dz:',zz(kmm+km4)-zz(kmm+km4-1)
        print*,'       effecive zone '
        write(*,'(2(A,1x,F12.8))')'          z:',zz(0),' ~ ',zz(kmm+kmm)
        write(*,'(2(A,1x,i0))')   '          k:',0,' ~ ', kmm+kmm
        write(*,'(1(A,1x,F12.8))')'         dz:',zz(kmm+kmm)-zz(kmm+kmm-1)
        print*,' ** -----------------------------------------------------'
      endif
      !
      z0=zz(0)
      zz=zz-z0
      !
    else
      zz=0.e0_real_kind
    endif
    !
    if(mpirank==0) print*,' ** center of the domain:',xx(imm),yy(jmm),zz(kmm)
    !
    do k=0,km
    do j=0,jm
    do i=0,im
      x(i,j,k,1)=xx(i+ig0)
      x(i,j,k,2)=yy(j+jg0)
      if(ndims==3) then
        x(i,j,k,3)=zz(k+kg0)
      else
        x(i,j,k,3)=0.e0_real_kind
      endif
    enddo
    enddo
    enddo
    !
    if(mpirank==0) then
      print*,' ** sandbox mesh generated, ranging:'
      write(*,'(2(A,1x,F12.8))')'          x:',xx(0),' ~ ',xx(ia)
      write(*,'(2(A,1x,F12.8))')'          y:',yy(0),' ~ ',yy(ja)
      write(*,'(2(A,1x,F12.8))')'          z:',zz(0),' ~ ',zz(ka)
    endif
    ! allocate(xa(0:ia,0:ja,0:ka),ya(0:ia,0:ja,0:ka),za(0:ia,0:ja,0:ka))
    ! do k=0,ka
    ! do j=0,ja
    ! do i=0,ia
    !   xa(i,j,k)=xx(i)
    !   ya(i,j,k)=yy(j)
    !   za(i,j,k)=zz(k)
    ! enddo
    ! enddo
    ! enddo
    ! !
    ! call tecbin('tecgrid.plt',xa,'x',ya,'y',za,'z')
    !
  end subroutine gridsandbox
  !+-------------------------------------------------------------------+
  !| The end of the subroutine gridjet.                                |
  !+-------------------------------------------------------------------+
  !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! This subroutine is used to stretch grid for spong layers
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine spongstretch(dim,varmax,var)
    !
    integer :: dim,hh
    real(real_kind) :: var(-2:dim)
    real(real_kind) :: varmax
    real(real_kind) :: err,err2,dra,ddra,var1,var2,dde,ratio
    !
    dra=1.e0_real_kind
    !
    ddra=0.001e0_real_kind*dra
    err=1.e10_real_kind
    err2=0.e0_real_kind
    !
    do while( abs(err)>1.e-9_real_kind )
      !
      ! ratio=1.d0
      ratio=(var(0)-var(-1))/(var(-1)-var(-2))
      do hh=1,dim
        !
        dde=var(hh-1)-var(hh-2)
        var(hh)=var(hh-1)+dde*ratio
        ratio=ratio*dra
        !
      end do
      !
      err=varmax-var(dim)
      !
      if(err>0) then
        dra=dra+ddra
      else
        dra=dra-ddra
      end if
      !
      if(err*err2<0.e0_real_kind) ddra=0.5e0_real_kind*ddra
      !
      err2=err
      !
      !print*,err,ddra,dra,ratio
      !
    end do
    !
    ! print*,' the mesh is strenched to x=',varmax,' from x=',var(0),' the max ratio is:',ratio,' using ',dim,' nodes'
    !
  end subroutine spongstretch
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! End of the subroutine spongstretch
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
end module gridgeneration
!+---------------------------------------------------------------------+
!| The end of the module gridgeneration.                              |
!+---------------------------------------------------------------------+
