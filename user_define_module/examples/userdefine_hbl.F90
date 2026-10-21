!+---------------------------------------------------------------------+
!| This module contains user defined subroutines to interfere  program |
!+---------------------------------------------------------------------+
!| CHANGE RECORD                                                       |
!| -------------                                                       |
!| 18-08-2023  | Created by J. Fang                                    |
!+---------------------------------------------------------------------+
module userdefine
  use astr_precision, only: real_kind
  !
  implicit none
  private :: real_kind
  !
  contains
  !
  !+-------------------------------------------------------------------+
  !| This subroutine is to set flow environment, such as, incoming     |
  !| free stream variables.                                            |
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 18-Aug-2023: created by Jian Fang @ Daresbury                     |
  !+-------------------------------------------------------------------+
  subroutine udf_setflowenv
    !
!     use commvar,  only: roinf,uinf,vinf,winf,pinf,tinf,spcinf,num_species
!     use fludyna,  only: thermal
!     !
! #ifdef COMB
!     use thermchem,only : tranco,spcindex,mixture,convertxiyi
!     use cantera 
!     !
!     real(8) :: specr(num_species)
!     ! 
!     specr(:)=0.d0
!     specr(spcindex('H2'))=0.0173
!     specr(spcindex('O2'))=0.2289
!     specr(spcindex('N2'))=1.d0-sum(specr)
!     !
!     ! pinf=5.d0*pinf
!     uinf=0.97d0
!     vinf=0.d0
!     winf=0.d0
!     tinf=300.d0
!     spcinf(:)=specr(:)
!     roinf=thermal(pressure=pinf,temperature=tinf,species=spcinf(:))
!     !
! #endif
    !
  end subroutine udf_setflowenv
  !+-------------------------------------------------------------------+
  !| The end of the subroutine udf_setflowenv.                         |
  !+-------------------------------------------------------------------+
  !
  !+-------------------------------------------------------------------+
  !| This subroutine is to generate fluctuations for inflow            |
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 05-Oct-2023: Created by by Jian Fang @ Daresbury                  |
  !+-------------------------------------------------------------------+
  subroutine udf_inflow_fluc(umean,uinst)
    !
    use commvar, only : jm,km
    !
    real(real_kind),intent(in) ::  umean(0:jm,1:3)  ! inflow mean velocity
    real(real_kind),intent(out) :: uinst(0:jm,0:km,1:3)  ! velocity with fluctuations
    !
  end subroutine udf_inflow_fluc
  !+-------------------------------------------------------------------+
  !| The end of the subroutine udf_inflow_fluc.                        |
  !+-------------------------------------------------------------------+
  !
  !+-------------------------------------------------------------------+
  !| This subroutine is to initialise flowfield by a user              |
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 25-May-2023: Created by Yifan Xu @ Peking University              |
  !| 18-Aug-2023: Rename and relocated by Jian Fang @ Daresbury        |
  !+-------------------------------------------------------------------+
  subroutine udf_flowinit
    !
!     use commvar,  only: im,jm,km,ndims,roinf,uinf,nondimen,xmax,pinf,  &
!                         ia,num_species
!     use commarray,only: x,vel,rho,prs,spc,tmp,q
!     use parallel, only: lio
!     use fludyna,  only: thermal
!     !
! #ifdef COMB
!     !
!     use thermchem,only : tranco,spcindex,mixture,convertxiyi
!     use cantera 
!     !
!     ! local data
!     integer :: i,j,k
!     real(8) ::  xc,yc,zc,tmpr,tmpp,xloc,xwid,specr(num_species),  &
!       specp(num_species),arg,prgvar,masflx,specx(num_species)
!     real(8) :: pthick
!     !
!     tmpr=300.d0
!     xloc=3.d0*xmax/4.d0
!     xwid=xmax/(12.d0*5.3d0*2.d0)
!     !
!     !reactants
!     specr(:)=0.d0
!     specr(spcindex('H2'))=0.0173
!     specr(spcindex('O2'))=0.2289
!     specr(spcindex('N2'))=1.d0-sum(specr)
!     !
!     !products
!     tmpp=1814.32d0
!     !
!     ! pthick=1.d-4
!     !
!     do k=0,km
!     do j=0,jm
!     do i=0,im
!       !
!       xc=x(i,j,k,1)
!       !
!       !prgvar=0.5d0*(1.d0+tanh(10.d0*(xc-xloc)/xloc))
!       ! if(xc-xloc<xwid*0.5d0*1.2d0) then 
!       !   prgvar=0.d0
!       !   if(xc-xloc>xwid*0.5d0) &
!       !   prgvar=1.d0-(xc-xloc-(xwid*0.5d0))/(xwid*0.5d0*0.2d0)
!       ! else
!       !   prgvar=1.d0
!       ! endif
!       !
!       prgvar=1.d0*exp(-0.5d0*((xc-xloc)/xwid)**2)
!       !
!       spc(i,j,k,:)=specr(:)
!       !
!       vel(i,j,k,1)=uinf
!       !
!       vel(i,j,k,2)=0.d0
!       vel(i,j,k,3)=0.d0
!       !
!       tmp(i,j,k)=tmpr+prgvar*(tmpp-tmpr)
!       !
!       prs(i,j,k)=pinf
!       !
!       rho(i,j,k)=thermal(pressure=prs(i,j,k),temperature=tmp(i,j,k), &
!                           species=spc(i,j,k,:))
!     enddo
!     enddo
!     enddo
!     !
!     !
!     if(lio)  write(*,'(A,I1,A)')'  ** HIT flame initialised.'
!     !
! #endif
    !
  end subroutine udf_flowinit
  !+-------------------------------------------------------------------+
  !| The end of the subroutine udf_flowinit.                           |
  !+-------------------------------------------------------------------+
  !
  !+-------------------------------------------------------------------+
  !| This subroutine is to generate grid.                              | 
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 23-Aug-2023: created by Jian Fang @ Daresbury                     |
  !+-------------------------------------------------------------------+
  subroutine udf_grid

    use commvar,  only : im,jm,km,gridfile,ia,ja,ka,ref_vel,ref_len
    use parallel, only : ig0,jg0,kg0,lio,pmax,mpistop
    use commarray,only : x
    use commfunc, only : argtanh
    use hdf5io

    integer :: i,j,k
    real(real_kind) :: lz
    real(real_kind),allocatable :: x2d(:,:),y2d(:,:)
    !
    allocate(x2d(0:im,0:jm),y2d(0:im,0:jm))

    call h5io_init(filename='datin/grid.2d',mode='read')
    call h5read(varname='x',var=x2d,dir='k')
    call h5read(varname='y',var=y2d,dir='k')
    call h5io_end()
    
    lz=90.e0_real_kind

    do k=0,km
    do j=0,jm
    do i=0,im
      !
      x(i,j,k,1)=x2d(i,j)
      !
      x(i,j,k,2)=y2d(i,j)
      !
      x(i,j,k,3)=lz/real(ka,real_kind)*real(k+kg0,real_kind)
      !
    end do
    end do
    end do
    !
    if(lio) then
      !
      write(*,'(2X,62A)')('-',i=1,62)
      write(*,'(18X,A)')' ** channel grid resolution **'
      write(*,'(1(A,F13.7))')'  ** x1=',x(1,0,0,1)
      write(*,'(1(A,F13.7))')'  ** y1=',x(0,1,0,2)
      write(*,'(2(A,F13.7))')'  ** dz=',x(0,0,1,3),' lz=',lz
      write(*,'(2X,62A)')('-',i=1,62)
      !
      write(*,'(A)')'  ** BL grid generated'
      !
    endif
    !
  end subroutine udf_grid
  !+-------------------------------------------------------------------+
  !| The end of the subroutine udf_grid.                               |
  !+-------------------------------------------------------------------+
  
  subroutine udf_writejslice
    
    use commvar,  only : im,jm,km,fnumslic,nstep,time
    use parallel, only : ig0,jg0,kg0,lio,mpirankname
    use commarray,only : rho,vel,prs,tmp,dvel,dtmp

    integer :: i,j,k
    integer :: jslice(8),js
    character(len=3) :: jsname
    character(len=48) :: fname
    character(len=5) :: stepname

    jslice(1)=10
    jslice(2)=15
    jslice(3)=20
    jslice(4)=30
    jslice(5)=40
    jslice(6)=60
    jslice(7)=80
    jslice(8)=110

    write(stepname,'(i5.5)')fnumslic

    do js=1,8

      if( jslice(js)>=jg0 .and. jslice(js)<jg0+jm) then

        j=jslice(js)-jg0
        write(jsname,'(i3.3)')jslice(js)
        fname='jslice/j'//jsname//'_'//stepname//mpirankname

        open(17,file=trim(fname),form='unformatted')
        write(17)im,jm,km
        write(17)nstep
        write(17)time
        write(17)rho(0:im,j,0:km)
        write(17)vel(0:im,j,0:km,1:3)
        write(17)prs(0:im,j,0:km)
        write(17)tmp(0:im,j,0:km)
        write(17)dvel(0:im,j,0:km,1:3,1:3)
        write(17)dtmp(0:im,j,0:km,2)
        close(17)
        ! print*,' << '//trim(fname)
      endif

    enddo

  end subroutine udf_writejslice

  !+-------------------------------------------------------------------+
  !| This subroutine is to list something during a computation.        | 
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 18-Aug-2023: created by Jian Fang @ Daresbury                     |
  !+-------------------------------------------------------------------+
  subroutine udf_stalist
    !
!     use commvar,  only : im,jm,km,ia,ja,ka,deltat
!     use commarray,only : vel,prs,rho,tmp,spc,cell,x
!     use interp,   only : interlinear
!     use parallel, only : pmax,psum,irk,irkm,lio
!     use utility,  only : listinit,listwrite
!     !
! #ifdef COMB
!     use thermchem,only : heatrate,spcindex
! #endif
!     !
!     integer :: i,j,k
!     real(8) :: tmpmax,rhomax,umax,qdotmax,poutrt
!     real(8) :: qdot,var1,var2
!     !
!     integer,save :: hand_fs
!     real(8),save :: xflame=0.d0,vflame=0.d0
!     logical,save :: linit=.true.
!     !
!     if(lio) then
!       !
!       if(linit) then
!         call listinit(filename='flamesta.dat',handle=hand_fs, &
!                       firstline='nstep time maxT maxU maxHRR xflame vflame pout')
!         linit=.false.
!       endif
!       !
!     endif
!     !
!     tmpmax=maxval(tmp(0:im,0:jm,0:km))
!     tmpmax=pmax(tmpmax)
!     !
!     rhomax=maxval(rho(0:im,0:jm,0:km))
!     rhomax=pmax(rhomax)
!     !
!     umax=maxval(vel(0:im,0:jm,0:km,1))
!     umax=pmax(umax)
!     !
!     var1=0.d0
!     var2=0.d0
!     !
!     qdotmax=-1.d30
!     !
!     do i=0,im
!       do j=0,jm
!         do k=0,km
!           !
!           qdot=heatrate(rho(i,j,k),tmp(i,j,k),spc(i,j,k,:))
!           if(qdot>qdotmax) then
!             qdotmax=qdot
!           endif
!           !
!         enddo 
!       enddo 
!     enddo  
!     qdotmax=pmax(qdotmax)
!     !
!     ! calculate the averaged flame location, set as the T=400K
!     var1=0.d0
!     !
!     do j=1,jm
!       !
!       do i=1,im
!         !
!         if( (tmp(i-1,j,k)<=400.d0 .and. tmp(i,j,k)>=400.d0) ) then
!           var1=var1+interlinear(tmp(i-1,j,k),tmp(i,j,k),        &
!                                 x(i-1,j,k,1),x(i,j,k,1),400.d0)
!           exit
!         endif
!         !
!       enddo
!       !
!     enddo
!     !
!     var1=psum(var1)/ja
!     !
!     ! use xflame to calculate vflame
!     if(abs(xflame)<1.d-16) then
!       vflame=0.d0
!     else
!       vflame=(var1-xflame)/deltat
!     endif
!     !
!     xflame=var1
!     !
!     ! calculate mean pressure at outflow
!     i=im
!     k=0
!     poutrt=0.d0
!     !
!     if(irk==irkm) then
!       do j=1,jm
!         poutrt=poutrt+prs(i,j,k)
!       enddo
!     endif
!     !
!     poutrt=psum(poutrt)/dble(ja)
!     !
!     if(lio) call listwrite(hand_fs,tmpmax,umax,qdotmax,xflame,vflame,poutrt)
    !
  end subroutine udf_stalist
  !+-------------------------------------------------------------------+
  !| The end of the subroutine udf_stalist.                            |
  !+-------------------------------------------------------------------+
  !
  !+-------------------------------------------------------------------+
  !| This subroutine is to add vortical fluctuations to initial field  | 
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 18-Aug-2023: created by Jian Fang @ Daresbury                     |
  !+-------------------------------------------------------------------+
  subroutine addvortex(xc,yc,radius,amp)
    !
    use commvar,  only: im,jm,km,ndims,roinf,uinf
    use parallel, only: lio
    use commarray,only: x,vel,rho,prs,spc,tmp,q
    use fludyna,  only: thermal
    !
    ! local data
    real(real_kind),intent(in) :: xc,yc,radius,amp
    !
    integer :: i,j,k
    real(real_kind) :: var1,radi2,cvor
    !
    cvor=amp*uinf*radius
    !
    do k=0,km
    do j=0,jm
    do i=0,im
      radi2=((x(i,j,k,1)-xc)**2+(x(i,j,k,2)-yc)**2)/radius/radius
      var1=cvor/radius/radius*exp(-0.5e0_real_kind*radi2)
      !
      vel(i,j,k,1)=vel(i,j,k,1)-var1*(x(i,j,k,2)-yc)
      if(ndims>=2) vel(i,j,k,2)=vel(i,j,k,2)+var1*(x(i,j,k,1)-xc)
      if(ndims==3) vel(i,j,k,3)=0.e0_real_kind
      prs(i,j,k)  =prs(i,j,k)-0.5e0_real_kind*roinf*cvor*cvor/radi2/radi2*exp(-radi2)
      !
      tmp(i,j,k)=thermal(density=rho(i,j,k),pressure=prs(i,j,k),species=spc(i,j,k,:))
      !
    enddo
    enddo
    enddo
    !
  end subroutine addvortex
  !+-------------------------------------------------------------------+
  !| The end of the subroutine addvortex.                              |
  !+-------------------------------------------------------------------+
  !
  !+-------------------------------------------------------------------+
  !| This subroutine add a source term to the rsd of the equation to   |
  !| hit flame.                                                        |
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 13-06-2023: Created by Yifan Xu @ Peking University               |
  !+-------------------------------------------------------------------+
  subroutine udf_src
    !
    use commvar,  only : force,im,jm,km,ndims
    use parallel, only : psum 
    use commarray,only : q,qrhs,x,jacob
    !
    ! local data
    integer :: i,j,k,k1,k2
    !
    real(real_kind) :: dy,u1,u2,u3
    real(real_kind),allocatable,save :: forcex(:,:,:)
    logical,save :: lfirstcal=.true.
    !
    ! if(lfirstcal) then
    !   allocate(forcex(0:im,0:jm,0:km))
      
    !   lfirstcal=.false.
    ! endif

    ! if(ndims==2) then
    !   k1=0
    !   k2=0
    ! elseif(ndims==3) then
    !   k1=1
    !   k2=km
    ! else
    !   print*,' !! ndims=',ndims
    !   stop ' !! error @ massfluxchan !!'
    ! endif
    ! !
    ! do k=0,km
    ! do j=0,jm
    ! do i=0,im
    !   qrhs(i,j,k,2)=qrhs(i,j,k,2)+forcex(i,j,k)*jacob(i,j,k)
    !   qrhs(i,j,k,5)=qrhs(i,j,k,5)+forcex(i,j,k)*vel(i,j,k,1)*jacob(i,j,k)
    ! end do
    ! end do
    ! end do
    !
  end subroutine udf_src
  !+-------------------------------------------------------------------+
  !| The end of the subroutine udf_src.                                |
  !+-------------------------------------------------------------------+
  !
  !+-------------------------------------------------------------------+
  !| This subroutine is to manipulate data solver as one likes at the  |
  !| end of each loop.                                                 | 
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 30-Oct-2023: created by Jian Fang @ Daresbury                     |
  !+-------------------------------------------------------------------+
  subroutine udf_eom_set
    !
  end subroutine udf_eom_set
  !+-------------------------------------------------------------------+
  !| The end of the subroutine udf_eom_set.                            |
  !+-------------------------------------------------------------------+
  !
  !+-------------------------------------------------------------------+
  !| This subroutine is to defined an output by a user.                | 
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 18-Aug-2023: created by Jian Fang @ Daresbury                     |
  !+-------------------------------------------------------------------+
  subroutine udf_write
    !
    
    !
  end subroutine udf_write
  !+-------------------------------------------------------------------+
  !| The end of the subroutine udf_write.                              |
  !+-------------------------------------------------------------------+
  !
  !+-------------------------------------------------------------------+
  !| This subroutine is to collect statistics.                         | 
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 09-Nov-2023: created by Jian Fang @ Appleton                      |
  !+-------------------------------------------------------------------+
  subroutine udf_meanflow
    !
  end subroutine udf_meanflow
  !+-------------------------------------------------------------------+
  !| The end of the subroutine udf_meanflow.                           |
  !+-------------------------------------------------------------------+


  !+-------------------------------------------------------------------+
  !| This subroutine is a user defined boundary condition              | 
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 02-Juil-2025: created by Jian Fang @ IMech, CAS, Beijing          |
  !+-------------------------------------------------------------------+
  subroutine udf_bc(ndir)
    
    use constdef
    use commarray, only : prs,vel,tmp,rho,spc,q
    use commvar,   only : nondimen,im,km,jm,num_species
    use parallel,  only : jrk,jrkm
    use fludyna,   only : thermal,fvar2q

    ! arguments
    integer,intent(in) :: ndir

    ! local data
    integer :: i,j,k
    real(real_kind) :: pe
    !

  end subroutine udf_bc
  !+-------------------------------------------------------------------+
  !| The end of the subroutine udf_bc.                                 |
  !+-------------------------------------------------------------------+

  subroutine udf_setup_before_comp
  end subroutine udf_setup_before_comp

end module userdefine
!+---------------------------------------------------------------------+
!| The end of the module userdefine.                                   |
!+---------------------------------------------------------------------+
