!+---------------------------------------------------------------------+
!| This module contains some common calculater.                        |
!+---------------------------------------------------------------------+
!| CHANGE RECORD                                                       |
!| -------------                                                       |
!| 19-03-2021  | Created by J. Fang                                    |
!+---------------------------------------------------------------------+
module commcal
  !
  use parallel, only: mpirank,mpistop,mpirankname
  use utility,  only: timereporter
  !
  implicit none
  !
  contains
  !
  !+-------------------------------------------------------------------+
  !| This subroutine is used to calculate CFL number and the           |
  !| corresponding time step.                                          |
  !+-------------------------------------------------------------------+
  !| ref: Adams, N. A., Shariff, K. 1996, J COMPUT PHYS. 127,27-51.    |
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 21-03-2021: Created by J. Fang @ STFC Daresbury Laboratory        |
  !+-------------------------------------------------------------------+
  subroutine cflcal(deltat,device_collector)
    !
    use iso_fortran_env, only: int64
    use ieee_arithmetic, only: ieee_is_finite
    use mpi
    use commvar, only: im,jm,km,ia,ja,lcomb,time,nstep
    use commarray,only: vel,tmp,dxi,spc,q
    use fludyna,  only: sos
    use parallel, only: ig0,jg0,kg0
    use thermchem,only: aceval
    use cfl_spectrum, only: spectral_rates
#ifdef ASTR_AIR5_CHEMISTRY
    use chemistry_thermo, only: air5_temperature_from_q5, &
      air5_species_gas_constant,air5_species_cv_tr
#endif
    !
    ! arguments
    real(8),intent(in) :: deltat
    abstract interface
      subroutine cfl_provider(values,keys,yvalues,by_plane,ny)
        use iso_fortran_env, only: int64
        integer,intent(in) :: ny
        real(8),intent(out) :: values(4),yvalues(4,0:ny-1)
        integer(int64),intent(out) :: keys(4)
        logical,intent(in) :: by_plane
      end subroutine cfl_provider
    end interface
    procedure(cfl_provider),optional :: device_collector
    !
    ! local data
    real(8) :: cfl,css,temp,rden,cvden,velocity(3),metric(3,3),rates(4),local(4),global(4)
    real(8),allocatable :: ylocal(:,:),yglobal(:,:)
    integer :: i,j,k,s,axis,status,ierr,invalid,invalid_global,opts(2),omin(2),omax(2),env_status
    integer(int64) :: key,keys(4),global_keys(4),gi,gj,gk
    logical :: valid,enabled,profile
    character(len=32) :: option

    option='on'
    call get_environment_variable('ASTR_CFL_DIAGNOSTICS',option,status=env_status)
    if(env_status==1) option='on'
    enabled=trim(option)/='off'
    invalid=0
    if((env_status/=0.and.env_status/=1).or. &
      (trim(option)/='on'.and.trim(option)/='off')) invalid=1
    option='0'
    call get_environment_variable('ASTR_CFL_PROFILE_Y',option,status=env_status)
    if(env_status==1) option='0'
    profile=trim(option)=='1'
    if((env_status/=0.and.env_status/=1).or. &
      (trim(option)/='0'.and.trim(option)/='1')) invalid=1
    opts=(/merge(1,0,enabled),merge(1,0,profile)/)
    call MPI_Allreduce(opts,omin,2,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,90,status)
    call MPI_Allreduce(opts,omax,2,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,90,status)
    if(any(omin/=omax)) invalid=1
    if(.not.ieee_is_finite(deltat)) invalid=1
    if(deltat<=0.d0) invalid=1
    call MPI_Allreduce(invalid,invalid_global,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS.or.invalid_global/=0) then
      if(mpirank==0) write(*,*) 'Invalid or inconsistent CFL options/time step'
      call MPI_Abort(MPI_COMM_WORLD,90,status)
    endif
    if(.not.enabled) return
    allocate(ylocal(4,0:merge(ja,0,profile)),yglobal(4,0:merge(ja,0,profile)))
    ylocal=0.d0
    local=0.d0
    keys=huge(0_int64)
    if(present(device_collector)) then
      call device_collector(local,keys,ylocal,profile,size(ylocal,2))
    else
      do k=0,km
      do j=0,jm
      do i=0,im
        status=0
        velocity=vel(i,j,k,:)
#ifdef ASTR_AIR5_CHEMISTRY
        if(lcomb) then
          call air5_temperature_from_q5(q(i,j,k,1),q(i,j,k,2:4), &
            q(i,j,k,6:10),q(i,j,k,11),q(i,j,k,5),temp,status)
          css=-1.d0
          if(status==0) then
            velocity=q(i,j,k,2:4)/q(i,j,k,1)
            rden=0.d0
            cvden=0.d0
            do s=1,5
              rden=rden+q(i,j,k,5+s)*air5_species_gas_constant(s)
              cvden=cvden+q(i,j,k,5+s)*air5_species_cv_tr(s)
            enddo
            css=sqrt((1.d0+rden/cvden)*rden*temp/q(i,j,k,1))
          endif
        else
#endif
#ifdef COMB
          call aceval(tmp(i,j,k),spc(i,j,k,:),css)
#else
          css=sos(tmp(i,j,k))
#endif
#ifdef ASTR_AIR5_CHEMISTRY
        endif
#endif
        metric=dxi(i,j,k,:,:)
        call spectral_rates(velocity,css,metric,rates,valid)
        if(.not.valid.or.status/=0) rates=huge(1.d0)
        key=(int(k+kg0,int64)*(ja+1)+j+jg0)*(ia+1)+i+ig0
        do axis=1,4
          if(rates(axis)>local(axis).or.(rates(axis)==local(axis).and.key<keys(axis))) then
            local(axis)=rates(axis)
            keys(axis)=key
          endif
        enddo
        if(profile) ylocal(:,jg0+j)=max(ylocal(:,jg0+j),rates)
      enddo
      enddo
      enddo
    endif
    call MPI_Allreduce(local,global,4,MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,90,status)
    if(any(global==huge(1.d0)).or..not.all(ieee_is_finite(global))) then
      if(mpirank==0) write(*,*) 'Invalid state or metric in complete-step CFL diagnostic'
      call MPI_Abort(MPI_COMM_WORLD,90,status)
    endif
    where(local/=global) keys=huge(0_int64)
    call MPI_Allreduce(keys,global_keys,4,MPI_INTEGER8,MPI_MIN,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,90,status)
    cfl=deltat*sum(global(1:3))
    if(.not.ieee_is_finite(cfl)) then
      if(mpirank==0) write(*,*) 'CFL diagnostic overflow'
      call MPI_Abort(MPI_COMM_WORLD,90,status)
    endif
    !
    if(mpirank==0) then
      write(*,"(A38)")'  =========== CFL Condition==========='
      write(*,"(A24,1x,E13.5)")'     current time step: ',deltat
      write(*,"(A24,1x,F13.7)")'           current CFL: ',cfl
      if(cfl>0.d0) write(*,"(A24,1x,E13.5)")'   time step for CFL=1: ',deltat/cfl
      write(*,'(A,I0,A,ES24.16E3,A,ES24.16E3)') &
        'ASTR_CFL complete_step=',nstep,' state_time=',time+deltat,' dt=',deltat
      write(*,'(A,5(1X,ES24.16E3))') 'ASTR_CFL directional/local_sum/upper_bound=', &
        deltat*global,cfl
      do axis=1,4
        key=global_keys(axis)
        gi=mod(key,int(ia+1,int64))
        gj=mod(key/(ia+1),int(ja+1,int64))
        gk=key/(int(ia+1,int64)*(ja+1))
        write(*,'(A,I0,A,3(1X,I0))') 'ASTR_CFL maximum=',axis,' global_ijk=',gi,gj,gk
      enddo
      write(*,"(A38)")'  ===================================='
    end if
    if(profile) then
      call MPI_Allreduce(ylocal,yglobal,4*(ja+1),MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,ierr)
      if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,90,status)
      if(mpirank==0) then
        do j=0,ja
          write(*,'(A,I0,A,I0,4(1X,ES24.16E3))') 'ASTR_CFL_Y step=',nstep, &
            ' j=',j,deltat*yglobal(:,j)
        enddo
      endif
    endif
    deallocate(ylocal,yglobal)
    !
  end subroutine cflcal
  !+-------------------------------------------------------------------+
  !| The end of the subroutine cflcal.                                 |
  !+-------------------------------------------------------------------+
  !
  !+-------------------------------------------------------------------+
  !| This subroutine is used to search monitor points.                 |
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 13-07-2021: Created by J. Fang @ STFC Daresbury Laboratory        |
  !+-------------------------------------------------------------------+
  subroutine monitorsearch(ijk,xyz,ijkmon,nmon)
    !
    use commvar,   only : im,jm,km
    use commarray, only : x
    use parallel,  only : ig0,jg0,kg0,irk,jrk,krk
    !
    ! arguments 
    integer,intent(in) :: ijk(:,:)
    real(8),intent(in) :: xyz(:,:)
    integer,allocatable,intent(out) :: ijkmon(:,:)
    integer,intent(out) :: nmon
    !
    ! local data
    integer :: i,j,k,nsize,n
    integer :: is,js,ks,ig,jg,kg
    integer,allocatable :: ijktemp(:,:)
    !
    nsize=size(ijk,1)
    allocate(ijktemp(nsize,4))
    !
    if(irk==0) then
      is=0
    else
      is=1
    endif
    !
    if(jrk==0) then
      js=0
    else
      js=1
    endif
    !
    if(krk==0) then
      ks=0
    else
      ks=1
    endif
    !
    nmon=0
    !
    do k=ks,km
    do j=js,jm
    do i=is,im
      !
      ig=ig0+i
      jg=jg0+j
      kg=kg0+k
      !
      do n=1,nsize
        !
        if(ijk(n,1)==-1) then
          !
          ! xyz(n,1)
          ! xyz(n,2)
          ! xyz(n,3)
          !
        else
          !
          if(ig==ijk(n,1) .and. jg==ijk(n,2) .and. kg==ijk(n,3)) then
            !
            nmon=nmon+1
            !
            ijktemp(nmon,1)=i
            ijktemp(nmon,2)=j
            ijktemp(nmon,3)=k
            ijktemp(nmon,4)=ijk(n,4) 
            !
          endif
          !
        endif
        !
      enddo
      !
    enddo
    enddo
    enddo
    !
    allocate(ijkmon(nmon,4))
    ijkmon(1:nmon,1:4)=ijktemp(1:nmon,1:4)
    !
    deallocate(ijktemp)
    !
  end subroutine monitorsearch
  !+-------------------------------------------------------------------+
  !| The end of the subroutine monitorsearch.                          |
  !+-------------------------------------------------------------------+
  !
  !+-------------------------------------------------------------------+
  !| This subroutine is used to be a shock senor.                      |
  !+-------------------------------------------------------------------+
  !! Ref1: F. Ducros, J. Comp. Phys. 1999, 152:517-549.                |
  !! Ref2: S. C. Lo, Int. J. Num. Meth. Fluid., 2009.                  |
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 08-10-2021: Created by J. Fang @ STFC Daresbury Laboratory        |
  !+-------------------------------------------------------------------+
  logical function shock_sensor_validation_enabled()
    implicit none
    character(len=1024) :: dump_path
    integer :: status,path_length

    call get_environment_variable('ASTR_SHOCK_SENSOR_DUMP',dump_path, &
                                  length=path_length,status=status)
    shock_sensor_validation_enabled = status == 0 .and. path_length > 0
  end function shock_sensor_validation_enabled

  subroutine dump_shock_sensor_validation()
    use commvar,  only : im,jm,km
    use commarray,only : ssf,lshock
    use parallel, only : mpirank,mpisize,ig0,jg0,kg0
    implicit none
    character(len=1024) :: dump_path,output_path
    character(len=32) :: rank_suffix
    integer :: status,path_length,fh,i,j,k

    call get_environment_variable('ASTR_SHOCK_SENSOR_DUMP',dump_path, &
                                  length=path_length,status=status)
    if(status /= 0 .or. path_length <= 0) return

    output_path=trim(dump_path(1:path_length))
    if(mpisize>1) then
      write(rank_suffix,'(".rank",I0)') mpirank
      output_path=trim(output_path)//trim(rank_suffix)
    endif
    open(newunit=fh,file=trim(output_path),status='replace',action='write')
    write(fh,'(A,6(1X,I0))') '# shock_sensor',im,jm,km,ig0,jg0,kg0
    do k=0,km
    do j=0,jm
    do i=0,im
      write(fh,'(3(I0,1X),ES24.16E3,1X,I1)') i,j,k,ssf(i,j,k),merge(1,0,lshock(i,j,k))
    enddo
    enddo
    enddo
    close(fh)
  end subroutine dump_shock_sensor_validation

  subroutine ducrossensor(timerept)
    !
    use commvar,  only : im,jm,km,is,ie,js,je,ks,ke,ia,ja,ka,hm,      &
                         npdci,npdcj,npdck,shkcrt,lreport,ltimrpt
    use commarray,only : ssf,lshock,dvel,prs
    use parallel, only : dataswap,pmin,pmax,psum,lio,ptime
    use validation_io, only: write_sensor_validation_snapshot
    !
    logical,intent(in),optional :: timerept
    !
    ! local data
    logical,save :: firstcall=.true.,validation_dumped=.false.
    real(8) :: div2,vort,vortx,vorty,vortz,dpdi,dpdj,dpdk
    real(8) :: ssfmin,ssfmax,ssfavg,norm,ssfmax_local
    integer :: i,j,k,i1,j1,k1,ii,jj,kk,ip1,jp1,kp1,im1,jm1,km1,nshknod,nijka
    !
    real(8) :: time_beg
    real(8),save :: subtime=0.d0
    !
    if(present(timerept)) then

      if(timerept) time_beg=ptime()

    endif 
    !
    if(firstcall) then
      !
      allocate(ssf(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
      allocate(lshock(0:im,0:jm,0:km))
      !
      firstcall=.false.
      !
    endif
    !
    ssfmin= 1.d10
    ssfmax=-1.d10
    ssfavg=0.d0
    norm=0.d0
    do k=0,km
    do j=0,jm
    do i=0,im
      !
      div2=(dvel(i,j,k,1,1)+dvel(i,j,k,2,2)+dvel(i,j,k,3,3))**2
      !
      vortx=dvel(i,j,k,3,2)-dvel(i,j,k,2,3)
      vorty=dvel(i,j,k,1,3)-dvel(i,j,k,3,1)
      vortz=dvel(i,j,k,2,1)-dvel(i,j,k,1,2)
      vort=vortx*vortx+vorty*vorty+vortz*vortz
      !
      ip1=i+1; im1=i-1
      jp1=j+1; jm1=j-1
      kp1=k+1; km1=k-1
      !
      if((npdci==1 .or. npdci==4) .and. im1<0)  im1=0
      if((npdcj==1 .or. npdcj==4) .and. jm1<0)  jm1=0
      if((npdck==1 .or. npdck==4) .and. km1<0)  km1=0
      if((npdci==2 .or. npdci==4) .and. ip1>im) ip1=im
      if((npdcj==2 .or. npdcj==4) .and. jp1>jm) jp1=jm
      if((npdck==2 .or. npdck==4) .and. kp1>km) kp1=km
      !
      dpdi= abs(prs(ip1,j,k)-2.d0*prs(i,j,k)+prs(im1,j,k)) /  &
               (prs(ip1,j,k)+2.d0*prs(i,j,k)+prs(im1,j,k))
      dpdj= abs(prs(i,jp1,k)-2.d0*prs(i,j,k)+prs(i,jm1,k)) /  &
               (prs(i,jp1,k)+2.d0*prs(i,j,k)+prs(i,jm1,k))
      dpdk= abs(prs(i,j,kp1)-2.d0*prs(i,j,k)+prs(i,j,km1)) /  &
               (prs(i,j,kp1)+2.d0*prs(i,j,k)+prs(i,j,km1))
      !
      ssf(i,j,k)=div2/(div2+vort+1.d-30) * max(dpdi,dpdj,dpdk)
      !
      ssfmin=min(ssfmin,ssf(i,j,k))
      ssfmax=max(ssfmax,ssf(i,j,k))
      !
      ssfavg=ssfavg+ssf(i,j,k)
      norm=norm+1.d0
      !
    enddo
    enddo
    enddo
    !
    call dataswap(ssf,timerept=ltimrpt)
    !
    nshknod=0
    !
    do k=0,km
    do j=0,jm
    do i=0,im
      !
      ssfmax_local=0.d0
      do i1=-hm+1,hm
        !
        ii=i+i1
        !
        if((npdci==1 .or. npdci==4) .and. ii<0)  ii=0
        if((npdci==2 .or. npdci==4) .and. ii>im) ii=im
        !
        ssfmax_local=max(ssfmax_local,ssf(ii,j,k))
        !
      enddo
      do j1=-hm+1,hm
        !
        jj=j+j1
        !
        if((npdcj==1 .or. npdcj==4) .and. jj<0)  jj=0
        if((npdcj==2 .or. npdcj==4) .and. jj>jm) jj=jm
        !
        ssfmax_local=max(ssfmax_local,ssf(i,jj,k))
        !
      enddo
      do k1=-hm+1,hm
        !
        kk=k+k1
        !
        if((npdck==1 .or. npdck==4) .and. kk<0)  kk=0
        if((npdck==2 .or. npdck==4) .and. kk>km) kk=km
        !
        ssfmax_local=max(ssfmax_local,ssf(i,j,kk))
        !
      enddo
      !
      if(ssfmax_local>shkcrt) then
        lshock(i,j,k)=.true.
        nshknod=nshknod+1
      else
        lshock(i,j,k)=.false.
      endif
      !
    enddo
    enddo
    enddo

    call write_sensor_validation_snapshot('sensor',ssf(0:im,0:jm,0:km), &
                                          merge(1_1,0_1,lshock))
    if(shock_sensor_validation_enabled() .and. (.not.validation_dumped)) then
      call dump_shock_sensor_validation()
      validation_dumped=.true.
    endif
    !
    if(lreport) then
      !
      nshknod=psum(nshknod)
      !
      ssfmin=pmin(ssfmin)
      ssfmax=pmax(ssfmax)
      ssfavg=psum(ssfavg)/psum(norm)
      !
      if(lio) then
        nijka=(ia+1)*(ja+1)*(ka+1)
        print*,' ------------- shock sensor -------------'
        write(*,"(4x,A,11X,F12.5)")'      max ssf: ',ssfmax
        write(*,"(4x,A,11X,F12.5)")'      min ssf: ',ssfmin
        write(*,"(4x,A,11X,F12.5)")'      avg ssf: ',ssfavg
        write(*,"(4x,2(A,I0),(A,F10.5,A))")'  shock nodes:  ',       &
                 nshknod,'/',nijka,' = ',dble(100*nshknod)/dble(nijka),' %'
        print*,' ----------------------------------------'
      endif
      !
    endif
    !
    if(present(timerept)) then
      if(timerept) then
      !
      subtime=subtime+ptime()-time_beg
      !
      if(lio .and. lreport .and. ltimrpt) call timereporter(routine='ducrossensor', &
                                             timecost=subtime, &
                                              message='shock sensor')
      endif
    endif
    !
    return
    !
  end subroutine ducrossensor
  !+-------------------------------------------------------------------+
  !| The end of the subroutine ducrossensor.                           |
  !+-------------------------------------------------------------------+
  !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! This subroutine is used to be a shock senor to control the nonlinear
  ! shock-capturing schemes with Jameson's approach.
  ! Ref1: F. Ducros, J. Comp. Phys. 1999, 152:517-549.
  ! Ref2: S. C. Lo, Int. J. Num. Meth. Fluid., 2009.
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Writen by Fang Jian, 2010-04-20.
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine ShockSolid
    !
    use parallel, only : npdci,npdcj,npdck,dataswap
    use commvar,  only : im,jm,km,hm,ltimrpt
    use commarray,only : nodestat,lsolid,x
    use tecio
    !
    ! local data
    integer :: i,j,k,m,i1,j1,k1
    logical,allocatable :: lss(:,:,:)
    real(8),allocatable :: rss(:,:,:)
    !
    ! set the shock area 
    allocate(lss(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
    !
    lss=.false.
    do k=0,km
    do j=0,jm
    do i=0,im
      !
      if(nodestat(i,j,k)>0 .or. nodestat(i,j,k)==-1) then
        lss(i,j,k)=.true.
      endif
      !
    end do
    end do
    end do
    !
    call dataswap(lss,timerept=ltimrpt)
    !
    lsolid=lss
    !
    ! expand the solid area 
    lsolid=.false.
    do k=-hm,km+hm
    do j=-hm,jm+hm
    do i=-hm,im+hm
      !
      if(lss(i,j,k)) then
        !
        do m=-4,4
          !
          i1=i+m
          !
          if(i1>im+hm) i1=im+hm
          if(i1<-hm)   i1=-hm
          !
          lsolid(i1,j,k)=.true.
          !
          j1=j+m
          !
          if(j1>jm+hm) j1=jm+hm
          if(j1<-hm)   j1=-hm
          !
          lsolid(i,j1,k)=.true.
          !
          k1=k+m
          !
          if(k1>km+hm) k1=km+hm
          if(k1<-hm)   k1=-hm
          !
          lsolid(i,j,k1)=.true.
          !
        enddo
        !
      endif
      !
    end do
    end do
    end do
    !
    call dataswap(lsolid,timerept=ltimrpt)
    !
    deallocate(lss)
    !
    ! allocate(rss(0:im,0:jm,0:km))
    ! !
    ! do k=0,km
    ! do j=0,jm
    ! do i=0,im
    !   !
    !   if(lsolid(i,j,k)) then
    !     rss(i,j,k)=1.d0
    !   else
    !     rss(i,j,k)=0.d0
    !   endif
    !   !
    ! end do
    ! end do
    ! end do
    ! !
    ! call tecbin('testout/tec_solid_sensor'//mpirankname//'.plt',      &
    !                                   x(0:im,0:jm,0:km,1),'x',    &
    !                                   x(0:im,0:jm,0:km,2),'y',    &
    !                                   x(0:im,0:jm,0:km,3),'z',    &
    !                                                   rss,'ss' )
    ! !
    ! deallocate(rss)
    !
    ! call mpistop
    !
    return
    !
  end subroutine ShockSolid
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! End of the subroutine ShockSolid
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !+-------------------------------------------------------------------+
  !| This function is to determin is a i,j,k is in the domain          |
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 23-07-2021  | Created by J. Fang @ Warrington                     |
  !+-------------------------------------------------------------------+
  pure logical function ijkin(i,j,k)
    !
    use commvar, only : im,jm,km
    !
    integer,intent(in) :: i,j,k
    !
    if(i<0 .or. j<0 .or. k<0 .or. i>im .or. j>jm .or. k>km) then
      ijkin=.false.
    else
      ijkin=.true.
    endif
    !
    return
    !
  end function ijkin
  !+-------------------------------------------------------------------+
  !| The end of the function ijkin.                                    |
  !+-------------------------------------------------------------------+
  !!
  !+-------------------------------------------------------------------+
  !| This function is to determin is a i,j,k is in the domain          |
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 24-08-2021  | Created by J. Fang @ Warrington                     |
  !+-------------------------------------------------------------------+
  pure logical function ijkcellin(i,j,k)
    !
    use commvar, only : im,jm,km,ndims
    !
    integer,intent(in) :: i,j,k
    !
    if(i>=1 .and. i<=im .and. j>=1 .and. j<=jm) then
      !
      if(ndims==3) then
        if(k>=1 .and. k<=km) then
          ijkcellin=.true.
        else
          ijkcellin=.false.
        endif
      elseif(ndims==2) then
        ijkcellin=.true.
      endif
    else
      ijkcellin=.false.
    endif
    !
    return
    !
  end function ijkcellin
  !+-------------------------------------------------------------------+
  !| The end of the function ijkcellin.                                |
  !+-------------------------------------------------------------------+

  !
end module commcal
!+---------------------------------------------------------------------+
!| The end of the module commcal.                                      |
!+---------------------------------------------------------------------+
