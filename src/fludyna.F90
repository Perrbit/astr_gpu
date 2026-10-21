!+---------------------------------------------------------------------+
!| This module contains subroutines and functions related to fluid     |
!| dynamicsof testing hanmish                                          |
!| ==============                                                      |
!| CHANGE RECORD                                                       |
!| -------------                                                       |
!| 09-Jnu-2020  | Created by J. Fang @ STFC Daresbury Laboratory       |
!+---------------------------------------------------------------------+
module fludyna
  use astr_precision, only: real_kind
  !
  ! use parallel, only: mpirank,mpistop
  use commvar, only: nondimen
  use thermchem, only: rgcmix,cpeval,temperature_calc
  !
  implicit none
  private :: real_kind
  real(real_kind),parameter,private :: thermal_unit_roundoff=0.5_real_kind*epsilon(1.0_real_kind)
  ! A perfect-gas pressure/temperature round trip contains four rounded operations.
  real(real_kind),parameter :: thermal_roundtrip_rtol=max(1.0e-10_real_kind, &
    nearest(4.0_real_kind*thermal_unit_roundoff/(1.0_real_kind-4.0_real_kind*thermal_unit_roundoff),1.0_real_kind))
  !
  interface thermal
     module procedure thermal_scar
     module procedure thermal_1d
     module procedure thermal_3d
  end interface
  !
  interface fvar2q
     module procedure fvar2q_sca
     module procedure fvar2q_1da
     module procedure fvar2q_3da
  end interface
  !
  interface q2fvar
     module procedure q2fvar_sca
     module procedure q2fvar_1da
     module procedure q2fvar_3da
  end interface
  !
  contains
  !
  !+-------------------------------------------------------------------+
  !| This function is used to get thermal relation among density,      |
  !| pressure and temperature                                          |
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 13-May-2020: Created by J. Fang @ STFC Daresbury Laboratory       |
  !+-------------------------------------------------------------------+
  function thermal_scar(density,pressure,temperature,species) result(vout)
    !
    use commvar,only : const2,rgas
    !
    ! arguments
    real(real_kind) :: vout
    real(real_kind),intent(in) ,optional :: density,pressure,temperature,species(:)
    !
    real(real_kind) :: rloc

    
    if(nondimen) then
      !
      if(present(density) .and. present(temperature)) then
        vout=density*temperature/const2
      elseif(present(density) .and. present(pressure)) then
        vout=pressure/density*const2
      elseif(present(temperature) .and. present(pressure)) then
        vout=pressure/temperature*const2
      else
        stop ' !! unable to get thermal variable  @ thermal !!'
      endif
      !
    else 
      !
#ifdef COMB
      rloc=rgcmix(species)
#else
      rloc=rgas
#endif
      !
      if(present(density).and.present(temperature)) then
        vout = density*temperature*rloc
      elseif(present(density).and.present(pressure)) then
        vout = pressure/density/rloc
      elseif(present(temperature).and.present(pressure)) then
        vout = pressure/temperature/rloc
      else
        stop ' !! unable to use dimensional EoS @ thermal !!'
      endif
      !
    endif 
    
  end function thermal_scar
  !
  function thermal_1d(density,pressure,temperature,species,dim) result(vout)
    !
    use commvar,only : const2,rgas
    !
    ! arguments
    integer,intent(in) :: dim
    real(real_kind) :: vout(dim)
    real(real_kind),intent(in),optional :: density(:),pressure(:),        &
                                   temperature(:),species(:,:)
    !
    real(real_kind) :: rloc(dim)
    !
    if(nondimen) then
      !
      if(present(density) .and. present(temperature)) then
        vout=density*temperature/const2
      elseif(present(density) .and. present(pressure)) then
        vout=pressure/density*const2
      elseif(present(temperature) .and. present(pressure)) then
        vout=pressure/temperature*const2
      else
        stop ' !! unable to get thermal variable  @ thermal !!'
      endif
      !
    else 
      !
#ifdef COMB
      rloc=rgcmix(species,dim)
#else
      rloc=rgas
#endif

      if(present(density).and.present(temperature)) then
        vout = density*temperature*rloc
      elseif(present(density).and.present(pressure)) then
        vout = pressure/density/rloc
      elseif(present(temperature).and.present(pressure)) then
        vout = pressure/temperature/rloc
      else
        stop ' !! unable to use dimensional EoS @ thermal !!'
      endif
      !
    endif 
    !
  end function thermal_1d
  !
  function thermal_3d(density,pressure,temperature,species,dim) result(vout)
    !
    use commvar,only : const2,rgas
    !
    ! arguments
    integer,intent(in) :: dim(3)
    real(real_kind) :: vout(dim(1),dim(2),dim(3))
    real(real_kind),intent(in),optional :: density(:,:,:),pressure(:,:,:),    &
                                   temperature(:,:,:),species(:,:,:,:)
    !
    real(real_kind) :: rloc(dim(1),dim(2),dim(3))
    !
    if(nondimen) then
      !
      if(present(density) .and. present(temperature)) then
        vout=density*temperature/const2
      elseif(present(density) .and. present(pressure)) then
        vout=pressure/density*const2
      elseif(present(temperature) .and. present(pressure)) then
        vout=pressure/temperature*const2
      else
        stop ' !! unable to get thermal variable  @ thermal_3d !!'
      endif
      !
    else
      !
#ifdef COMB
      rloc=rgcmix(species,dim)
#else
      rloc=rgas
#endif
      if(present(density).and.present(temperature)) then
        vout = density*temperature*rloc
      elseif(present(density).and.present(pressure)) then
        vout = pressure/density/rloc
      elseif(present(temperature).and.present(pressure)) then
        vout = pressure/temperature/rloc
      else
        stop ' !! unable to use dimensional EoS @ thermal_3d !!'
      endif
      !
    endif
    !
  end function thermal_3d
  !+-------------------------------------------------------------------+
  !| The end of the subroutine thermal.                                |
  !+-------------------------------------------------------------------+
  !
  !+-------------------------------------------------------------------+
  !| This subroutine is to update flow variables from q.               |
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 04-Aug-2018: Created by J. Fang @ STFC Daresbury Laboratory       |
  !+-------------------------------------------------------------------+
  subroutine updatefvar
    !
    use commarray,only : q,rho,vel,prs,tmp,spc,tke,omg
    use commvar,  only : im,jm,km,num_species,num_modequ,turbmode,lcomb
#ifdef ASTR_AIR5_CHEMISTRY
    use commarray, only: tve
    use chemistry_flow_runtime, only: air5_field_conservative_to_primitive
#endif
    !
    ! local data
    integer :: i,j,k
#ifdef ASTR_AIR5_CHEMISTRY
    integer :: air5_status,air5_failed_index(3)

    if(lcomb) then
      call air5_field_conservative_to_primitive(q(0:im,0:jm,0:km,:), &
        rho(0:im,0:jm,0:km),vel(0:im,0:jm,0:km,:),prs(0:im,0:jm,0:km), &
        tmp(0:im,0:jm,0:km),spc(0:im,0:jm,0:km,:),tve(0:im,0:jm,0:km), &
        air5_status,air5_failed_index)
      if(air5_status/=0) then
        write(*,'(A,I0,A,3(I0,1X))') 'air5 updatefvar failed, status=', &
          air5_status,', i/j/k=',air5_failed_index
        error stop 'air5 conservative-to-primitive conversion failed'
      endif
      return
    endif
#endif
    !
    if(trim(turbmode)=='k-omega') then
      !
      call q2fvar(q=q(0:im,0:jm,0:km,:),                               &
                                     density=rho(0:im,0:jm,0:km),      &
                                    velocity=vel(0:im,0:jm,0:km,:),    &
                                    pressure=prs(0:im,0:jm,0:km),      &
                                 temperature=tmp(0:im,0:jm,0:km),      &
                                     species=spc(0:im,0:jm,0:km,:),    &
                                         tke=tke(0:im,0:jm,0:km),      &
                                       omega=omg(0:im,0:jm,0:km) )
      !
      do k=0,km
      do j=0,jm
      do i=0,im
        !
        if(tke(i,j,k)<0.e0_real_kind) then
          tke(i,j,k)=0.e0_real_kind
          q(i,j,k,6+num_species)=rho(i,j,k)*tke(i,j,k)
        endif
        !
        if(omg(i,j,k)<0.01e0_real_kind) then
          omg(i,j,k)=0.01e0_real_kind
          q(i,j,k,7+num_species)=rho(i,j,k)*omg(i,j,k)
        endif
        !
      enddo
      enddo
      enddo
      !
    else
      !
      call q2fvar(q=q(0:im,0:jm,0:km,:),                               &
                                     density=rho(0:im,0:jm,0:km),      &
                                    velocity=vel(0:im,0:jm,0:km,:),    &
                                    pressure=prs(0:im,0:jm,0:km),      &
                                 temperature=tmp(0:im,0:jm,0:km),      &
                                     species=spc(0:im,0:jm,0:km,:) )
      !
    endif
    !
  end subroutine updatefvar
  !+-------------------------------------------------------------------+
  !| The end of the subroutine updatefvar.                             |
  !+-------------------------------------------------------------------+
  !
  !+-------------------------------------------------------------------+
  !| This subroutine is to update q from flow variables.               |
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 06-Aug-2018: Created by J. Fang @ STFC Daresbury Laboratory       |
  !+-------------------------------------------------------------------+
  subroutine updateq
    !
    use commarray,only : q,rho,vel,prs,tmp,spc,tke,omg
    use commvar,  only : im,jm,km,num_species,num_modequ,turbmode,numq,lcomb
    use ieee_arithmetic, only: ieee_is_nan
#ifdef ASTR_AIR5_CHEMISTRY
    use commarray, only: tve
    use chemistry_flow_runtime, only: air5_field_primitive_to_conservative
#endif
    !
    integer :: i,j,k,n
#ifdef ASTR_AIR5_CHEMISTRY
    integer :: air5_status,air5_failed_index(3)

    if(lcomb) then
      call air5_field_primitive_to_conservative(q(0:im,0:jm,0:km,:), &
        rho(0:im,0:jm,0:km),vel(0:im,0:jm,0:km,:),tmp(0:im,0:jm,0:km), &
        spc(0:im,0:jm,0:km,:),tve(0:im,0:jm,0:km),air5_status, &
        air5_failed_index)
      if(air5_status/=0) then
        write(*,'(A,I0,A,3(I0,1X))') 'air5 updateq failed, status=', &
          air5_status,', i/j/k=',air5_failed_index
        error stop 'air5 primitive-to-conservative conversion failed'
      endif
      return
    endif
#endif
    !
    if(trim(turbmode)=='k-omega') then
      !
      call fvar2q(          q=  q(0:im,0:jm,0:km,:),                   &
                      density=rho(0:im,0:jm,0:km),                     &
                     velocity=vel(0:im,0:jm,0:km,:),                   &
                     pressure=prs(0:im,0:jm,0:km),                     &
                      species=spc(0:im,0:jm,0:km,:),                   &
                          tke=tke(0:im,0:jm,0:km),                     &
                        omega=omg(0:im,0:jm,0:km)                      )
      !
    else
      call fvar2q(          q=  q(0:im,0:jm,0:km,:),                   &
                      density=rho(0:im,0:jm,0:km),                     &
                     velocity=vel(0:im,0:jm,0:km,:),                   &
                     pressure=prs(0:im,0:jm,0:km),                     &
                     temperature=tmp(0:im,0:jm,0:km),                  &
                      species=spc(0:im,0:jm,0:km,:)                    )
      !
      do k=0,km
      do j=0,jm
      do i=0,im
        !
        do n=1,numq
          if(ieee_is_nan(q(i,j,k,n))) then
            print*,i,j,k,n
            print*,'q:',q(i,j,k,:),'@ updateq'
          endif
        enddo
        !
      enddo
      enddo
      enddo
      !
    endif
    !
  end subroutine updateq
  !+-------------------------------------------------------------------+
  !| The end of the subroutine updateq.                                |
  !+-------------------------------------------------------------------+
  !
  !+-------------------------------------------------------------------+
  !| This function is used to convert flow variables to q.             |
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 09-02-2021: Created by J. Fang @ Warrington.                      |
  !+-------------------------------------------------------------------+
  subroutine fvar2q_sca(q,density,velocity,pressure,temperature,       &
                       species,tke,omega)
    !
    use commvar, only: numq,ndims,num_species,const1,const6,cv
    !
    real(real_kind),intent(in) :: density,velocity(:)
    real(real_kind),intent(in),optional :: pressure,temperature,species(:),tke,omega
    real(real_kind),intent(out) :: q(:)
    !
    ! local data
    integer :: jspec,j
    real(real_kind) :: var1,var2,cotem
    !
    q(1)=density
    q(2)=density*velocity(1)
    q(3)=density*velocity(2)
    q(4)=density*velocity(3)
    !
#ifdef COMB
    if(.not.present(species))  &
      stop ' !! error @ fvar2q_sca - species not set for energy calculation !!'
      !
    if(present(temperature)) then
      var1=0.5e0_real_kind*sum(velocity(:)*velocity(:))
      call cpeval(tmp=temperature,spc=species,ke=var1,eng=var2)
      q(5)=var2*density
    else
      print*,' !! temperature required for energy calculation !!'
      stop ' !! error @ fvar2q_sca'
    endif 
    !
#else
    if(nondimen) then 
      cotem=const1
    else
      cotem=cv
    endif
    !
    var1=0.5e0_real_kind*sum(velocity(:)*velocity(:))
    if(present(temperature)) then
      q(5)=density*(temperature*cotem+var1)
    elseif(present(pressure)) then
      q(5)=pressure*const6+density*var1
    else
      print*,' !! pressure or temperature required for energy calculation !!'
      stop ' !! error @ fvar2q_sca'
    endif
#endif
    !
    if(num_species>0) then
      !
      do jspec=1,num_species
        q(5+jspec)=density*species(jspec)
      enddo
      !
    endif
    !
    if(present(tke) .and. present(omega)) then
      !
      q(5+num_species+1)=density*tke
      q(5+num_species+2)=density*omega
      !
    endif
    !
  end subroutine fvar2q_sca
  !
  subroutine fvar2q_1da(q,density,velocity,pressure,temperature,species)
    !
    use commvar, only: numq,ndims,num_species,const1,const6,cv
    !
    real(real_kind),intent(in) :: density(:),velocity(:,:)
    real(real_kind),intent(in),optional :: pressure(:),temperature(:), &
                                   species(:,:)
    real(real_kind),intent(out) :: q(:,:)
    !
    ! local data
    integer :: jspec,j
    real(real_kind) :: var1,var2,cotem
    !
    q(:,1)=density(:)
    q(:,2)=density(:)*velocity(:,1)
    q(:,3)=density(:)*velocity(:,2)
    q(:,4)=density(:)*velocity(:,3)
    !
#ifdef COMB

    if(.not.present(species))  &
      stop ' !! error @ fvar2q_1da - species not set for energy calculation !!'
    !
    if(present(temperature)) then
      !
      do j=1,size(density)
        var1=0.5e0_real_kind*sum(velocity(j,:)*velocity(j,:))
        call cpeval(tmp=temperature(j),spc=species(j,:),ke=var1,eng=var2)
        q(j,5)=var2*density(j)
      enddo   
      !
    else
      print*,' !! temperature required for energy calculation !!'
      stop ' !! error @ fvar2q_1da'
    endif 

#else
    
    if(nondimen) then 
      cotem=const1
    else
      cotem=cv
    endif
    !
    if(present(temperature)) then
      q(:,5)=density(:)*( temperature(:)*cotem +         &
                          0.5e0_real_kind*(velocity(:,1)**2 +      &
                                 velocity(:,2)**2 +      &
                                 velocity(:,3)**2) )
    elseif(present(pressure)) then
      q(:,5)=pressure(:)*const6+0.5e0_real_kind*density(:)*(       &
                                    velocity(:,1)**2 +   &
                                    velocity(:,2)**2 +   &
                                    velocity(:,3)**2 )
    else
      print*,' !! pressure or temperature required for energy calculation !!'
      stop ' !! error @ fvar2q_1da'
    endif

#endif
    !
    if(num_species>0) then
      !
      do jspec=1,num_species
        q(:,5+jspec)=density(:)*species(:,jspec)
      enddo
      !
    endif
    !
  end subroutine fvar2q_1da
  !
  subroutine fvar2q_3da(q,density,velocity,pressure,temperature,species,tke,omega)
    !
    use commvar, only: numq,ndims,num_species,const1,const6,cv
    !
    real(real_kind),intent(in) :: density(:,:,:),velocity(:,:,:,:)
    real(real_kind),intent(in),optional :: pressure(:,:,:),temperature(:,:,:), &
                                   species(:,:,:,:),tke(:,:,:),        &
                                   omega(:,:,:)
    real(real_kind),intent(out) :: q(:,:,:,:)
    !
    ! local data
    integer :: jspec,i,j,k
    real(real_kind) :: var1,var2,cotem
    !
    q(:,:,:,1)=density(:,:,:)
    q(:,:,:,2)=density(:,:,:)*velocity(:,:,:,1)
    q(:,:,:,3)=density(:,:,:)*velocity(:,:,:,2)
    q(:,:,:,4)=density(:,:,:)*velocity(:,:,:,3)
    !
#ifdef COMB
    
    if(.not.present(species))  &
      stop ' !! error @ fvar2q_3da - species not set for energy calculation !!'
      !
    if(present(temperature)) then
      !
      do i=1,size(density,1)
        do j=1,size(density,2)
          do k=1,size(density,3)
            !
            var1=0.5e0_real_kind*sum(velocity(i,j,k,:)*velocity(i,j,k,:))
            call cpeval(tmp=temperature(i,j,k),spc=species(i,j,k,:), &
                        ke=var1,eng=var2)
            q(i,j,k,5)=var2*density(i,j,k)
            !
          enddo   
        enddo 
      enddo 
      !
    else
      print*,' !! temperature required for energy calculation !!'
      stop ' !! error @ fvar2q_3da'
      endif

#else

    if(nondimen) then 
      cotem=const1
    else
      cotem=cv
    endif
    !
    if(present(temperature)) then
      q(:,:,:,5)=density(:,:,:)*( temperature(:,:,:)*cotem +     &
                           0.5e0_real_kind*(velocity(:,:,:,1)**2 +         &
                                  velocity(:,:,:,2)**2 +         &
                                  velocity(:,:,:,3)**2) )
    elseif(present(pressure)) then
        q(:,:,:,5)=pressure(:,:,:)*const6+0.5e0_real_kind*density(:,:,:)*(       &
                                    velocity(:,:,:,1)**2 +             &
                                    velocity(:,:,:,2)**2 +             &
                                    velocity(:,:,:,3)**2 )
    else
      print*,' !! pressure or temperature required !!'
      stop ' !! error @ fvar2q_3da'
    endif

#endif
    !
    if(num_species>0) then
      !
      do jspec=1,num_species
        q(:,:,:,5+jspec)=density(:,:,:)*species(:,:,:,jspec)
      enddo
      !
    endif
    !
    if(present(tke) .and. present(omega)) then
      !
      q(:,:,:,5+num_species+1)=tke(:,:,:)*density(:,:,:)
      q(:,:,:,5+num_species+2)=omega(:,:,:)*density(:,:,:)
      !
    endif
    !
  end subroutine fvar2q_3da
  !+-------------------------------------------------------------------+
  !| The end of the subroutine fvar2q.                                 |
  !+-------------------------------------------------------------------+
  !
  !+-------------------------------------------------------------------+
  !| This subroutine calcualtes field variables from q.                |
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 09-02-2021: Created by J. Fang @ Warrington.                      |
  !+-------------------------------------------------------------------+
  subroutine q2fvar_3da(q,density,velocity,pressure,temperature,species,tke,omega)
    !
    use commvar, only: numq,ndims,num_species,const1,const6,tinf
    !
    real(real_kind),intent(inout) :: q(:,:,:,:)
    real(real_kind),intent(out) :: density(:,:,:)
    real(real_kind),intent(out),optional :: temperature(:,:,:),pressure(:,:,:),&
                                    species(:,:,:,:),velocity(:,:,:,:),&
                                    tke(:,:,:),omega(:,:,:)
    !
    ! local data
    integer :: jspec,i,j,k
    integer :: dim(3)
    real(real_kind) :: var1
    !
    density(:,:,:)   =q(:,:,:,1)
    !
    if(present(velocity) .or. present(pressure) .or. present(temperature)) then
      velocity(:,:,:,1)=q(:,:,:,2)/density
      velocity(:,:,:,2)=q(:,:,:,3)/density
      velocity(:,:,:,3)=q(:,:,:,4)/density
    endif
    !
    if(num_species>0 .and. present(species)) then
      !
      do jspec=1,num_species
        species(:,:,:,jspec)=q(:,:,:,5+jspec)/density(:,:,:)
      enddo
      !
    endif
    !
#ifdef COMB

    if(.not.present(species))  &
      stop ' !! error @ q2fvar - species not set for T calculation !!'
      !
    if(present(pressure) .or. present(temperature)) then
      !
      do i=1,size(q,1)
        do j=1,size(q,2)
          do k=1,size(q,3)
            !
            var1=q(i,j,k,5)/density(i,j,k) &
                 -0.5e0_real_kind*sum(velocity(i,j,k,:)*velocity(i,j,k,:))
            ! temperature(i,j,k)=tinf
            call temperature_calc(tmp=temperature(i,j,k),den=density(i,j,k), &
                                  spc=species(i,j,k,:),eint=var1)
          enddo
        enddo 
      enddo          
      !
    endif 
    !
    if(present(pressure)) then
      dim(1)=size(q,1)
      dim(2)=size(q,2)
      dim(3)=size(q,3)
      !
      pressure=thermal(temperature=temperature,density=density,species=species,dim=dim)

    endif
    !  
#else

    if(present(pressure) .or. present(temperature)) then
      pressure(:,:,:)  =( q(:,:,:,5)-0.5e0_real_kind*density(:,:,:)*(              &
                                          velocity(:,:,:,1)**2+         &
                                          velocity(:,:,:,2)**2+         &
                                          velocity(:,:,:,3)**2) )/const6
    endif
    !
    if(present(temperature)) then
      dim(1)=size(q,1)
      dim(2)=size(q,2)
      dim(3)=size(q,3)
      !
      temperature=thermal(pressure=pressure,density=density,dim=dim)
    endif
    !
#endif
    !
    if(present(tke)) then
      tke(:,:,:)=q(:,:,:,5+num_species+1)/density
    endif
    !
    if(present(omega)) then
      omega(:,:,:)=q(:,:,:,5+num_species+2)/density
    endif
    !
  end subroutine q2fvar_3da
  !
  subroutine q2fvar_1da(q,density,velocity,pressure,temperature,species,tke,omega)
    !
    use commvar, only: numq,ndims,num_species,const1,const6
    !
    real(real_kind),intent(in) :: q(:,:)
    real(real_kind),intent(out) :: density(:)
    real(real_kind),intent(out),optional :: velocity(:,:),pressure(:),         &
                                    temperature(:),species(:,:),       &
                                    tke(:),omega(:)
    !
    ! local data
    integer :: jspec,j
    integer :: dim
    real(real_kind) :: var1
    !
    density(:)   =q(:,1)
    !
    if(present(velocity) .or. present(pressure) .or. present(temperature)) then
      velocity(:,1)=q(:,2)/density
      velocity(:,2)=q(:,3)/density
      velocity(:,3)=q(:,4)/density
    endif
    !
    if(num_species>0 .and. present(species)) then
      !
      do jspec=1,num_species
        species(:,jspec)=q(:,5+jspec)/density(:)
      enddo
      !
    endif
    !
#ifdef COMB

    if(.not.present(species))  &
      stop ' !! error @ q2fvar - species not set for T calculation !!'
      !
    if(present(pressure) .or. present(temperature)) then
      !
      do j=1,size(density)
        !
        var1=q(j,5)/density(j)-0.5e0_real_kind*sum(velocity(j,:)*velocity(j,:))
        call temperature_calc(tmp=temperature(j),den=density(j), &
                                  spc=species(j,:),eint=var1)
      enddo          
      !
    endif 
    !
    if(present(pressure)) then
      dim=size(q,1)
      !
      pressure=thermal(temperature=temperature,density=density, &
                        species=species,dim=dim)
    endif

#else

    if(present(pressure) .or. present(temperature)) then
      pressure(:)  =( q(:,5)-0.5e0_real_kind*density(:)*(                   &
                                          velocity(:,1)**2+      &
                                          velocity(:,2)**2+      &
                                          velocity(:,3)**2) )/const6
    endif
    !
    if(present(temperature)) then
      dim=size(q,1)
      !
      temperature=thermal(pressure=pressure,density=density,dim=dim)
    endif

#endif
    !
    if(present(tke)) then
      tke(:)=q(:,5+num_species+1)/density
    endif
    !
    if(present(omega)) then
      omega(:)=q(:,5+num_species+2)/density
    endif
    !
  end subroutine q2fvar_1da
  !
  subroutine q2fvar_sca(q,density,velocity,pressure,temperature,species,tke,omega)
    !
    use commvar, only: numq,ndims,num_species,const1,const6
    !
    real(real_kind),intent(in) :: q(:)
    real(real_kind),intent(out) :: density
    real(real_kind),intent(out),optional :: velocity(:),pressure,temperature,  &
                                    species(:),tke,omega
    !
    ! local data
    integer :: jspec
    real(real_kind) :: var1
    !
    density   =q(1)
    !
    if(present(velocity) .or. present(pressure) .or. present(temperature)) then
      velocity(1)=q(2)/density
      velocity(2)=q(3)/density
      velocity(3)=q(4)/density
    endif
    !
    if(num_species>0 .and. present(species)) then
      !
      do jspec=1,num_species
        species(jspec)=q(5+jspec)/density
      enddo
      !
    endif
    !
#ifdef COMB
    if(.not.present(species))  &
      stop ' !! error @ q2fvar - species not set for T calculation !!'
      !
    if(present(pressure) .or. present(temperature)) then
      !
      var1=q(5)/density-0.5e0_real_kind*sum(velocity(:)*velocity(:))
      call temperature_calc(tmp=temperature,den=density, &
                            spc=species(:),eint=var1)
    endif 
    !
    if(present(pressure)) then
      pressure=thermal(temperature=temperature,density=density, &
                        species=species)
    endif
#else

    if(present(pressure) .or. present(temperature)) then
      pressure  =( q(5)-0.5e0_real_kind*density*(velocity(1)**2+velocity(2)**2+    &
                                       velocity(3)**2) )/const6
    endif
    !
    if(present(temperature)) then
      temperature=thermal(pressure=pressure,density=density)
    endif

#endif
    !
    if(present(tke)) then
      tke=q(5+num_species+1)/density
    endif
    !
    if(present(omega)) then
      omega=q(5+num_species+2)/density
    endif
    !
  end subroutine q2fvar_sca
  !+-------------------------------------------------------------------+
  !| The end of the subroutine q2fvar.                                 |
  !+-------------------------------------------------------------------+
  !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! calculate dynamic viscosity coefficient at different temperature.
  ! ref: https://www.cfd-online.com/Wiki/Sutherland%27s_law
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  real(real_kind) function miucal(temper)
    !
    use commvar, only :  tempconst,tempconst1
    !
    real(real_kind),intent(in) :: temper
    ! temper represent temperature, dimensionless
    ! below calculate miucal using sutherland's law
    ! tempconst=110.4d0/ref_t
    ! tempconst1=1.d0+tempconst
    !
    real(real_kind) :: s,tmpers,tmper0,miu0,tnondim
    !
    if(nondimen) then
      miucal=temper*sqrt(temper)*tempconst1/(temper+tempconst)
    else
      ! for air, kg/(m s)
      tnondim=temper/273.15e0_real_kind
      !
      miucal=1.716e-5_real_kind*tnondim*sqrt(tnondim)*(273.15e0_real_kind+110.4e0_real_kind)/(temper+110.4e0_real_kind)
      !
      ! tmpers=110.d0
      ! tmper0=273.125d0
      ! miu0=1.845d-5
      ! miucal=miu0*(tmper0+tmpers)/(temper+tmpers) &
      !        *((temper/tmper0)**1.5d0)
    endif 
    !
    return
    !
  end function miucal
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! End of function MiuCal.
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!
  !+-------------------------------------------------------------------+
  !| This function is used to calculate speed of sound.                |
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 24-Feb-2021: Created by J. Fang @ STFC Daresbury Laboratory       |
  !+-------------------------------------------------------------------+
  real(real_kind) function sos(tmp,spc)
    !
    use commvar,   only: nondimen,mach,num_species,gamma,rgas
    use thermchem, only: aceval
    !
    real(real_kind),intent(in) :: tmp
    real(real_kind),intent(in),optional :: spc(:)
    !
    ! local data
    real(real_kind) :: cpcmix,gamrgc
    !
#ifdef COMB
    if(.not. present(spc)) then
      stop ' !! error, species needed to calculate speed of sound @ function sos'
    else
      call aceval(tmp,spc,sos)
    endif
#else
    if(nondimen) then
      sos=sqrt(tmp)/mach
    else
      sos=sqrt(gamma*rgas*tmp)
    endif
#endif
    !
    return
    !
  end function sos
  !+-------------------------------------------------------------------+
  !| The end of the function sos.                                      |
  !+-------------------------------------------------------------------+
  !
  !+-------------------------------------------------------------------+
  !| This function is used to assign the velocity profile of the jet   |
  !| flow.                                                             |
  !+-------------------------------------------------------------------+
  !| ref: Bogey, C., Bailly C., and Juve,D., Noise Investigation of a  |
  !|      High Subsonic, Moderate Reynolds Number Jet Using a          |
  !|      Compressible Large Eddy Simulation. Theoret. Comput. Fluid   |
  !|      Dynamics, 2003, 16, 273–297.                                 |
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 09-Jun-2020: Created by J. Fang @ STFC Daresbury Laboratory       |
  !+-------------------------------------------------------------------+
  function jetvel(r,time) result(u)
    !
    use constdef
    use commvar, only: uinf,ymin,ymax
    !
    ! arguments
    real(real_kind) :: u(3)
    real(real_kind),intent(in) :: r
    real(real_kind),intent(in),optional :: time
    !
    ! local data
    real(real_kind) :: r0,uc,delta,ujet,alfa,theter,Tperi,var1
    !+-----------------------------+
    !| delta: the initial momentum |
    !| thickness of the shear layer|
    !| uc: jet centerline velocity |
    !+-----------------------------+
    !
    !
    r0=1.e0_real_kind
    uc=1.67e0_real_kind*uinf    ! ref: Reichert & Biringen, Mechanics Research
                      !      Communications 34 (2007) 249–259
    delta=0.05e0_real_kind*r0   ! ref: Bogey
    !
    var1=0.5e0_real_kind+0.5e0_real_kind*tanh((r0-abs(r))/(2.e0_real_kind*delta)) ! profile 0~1
    ! ujet=var1*(uc-uinf)+uinf
    ujet=var1
    !
    if(present(time)) then
      !
      ! alfa=2.5d0/180.d0*pi
      ! theter=sin(2.d0*pi/Tperi*time)*alfa
      ! !
      ! u(1)=ujet*cos(theter)+uinf
      ! u(2)=ujet*sin(theter)
      !
    else
      u(1)=ujet !+uinf
      u(2)=0.e0_real_kind
      u(3)=0.e0_real_kind
    endif
    !
    return
    !
  end function jetvel
  !+-------------------------------------------------------------------+
  !| The end of the function jetvel.                                   |
  !+-------------------------------------------------------------------+
  !
  !+-------------------------------------------------------------------+
  !| This function is used to assign velocity profile of mixing layer. |
  !+-------------------------------------------------------------------+
  !| ref: Li, Z., Jaberi, F. 2010. Numerical Investigations of         |
  !|      Shock-Turbulence Interaction in a Planar Mixing Layer.       |
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 19-Mar-2021: Created by J. Fang @ STFC Daresbury Laboratory       |
  !+-------------------------------------------------------------------+
  subroutine mixinglayervel(y,u,t,ro,info) 
    !
    use constdef
    use commvar, only: ymin,ymax,gamma,mach
    !
    ! arguments
    real(real_kind),intent(in) :: y
    real(real_kind),intent(out) :: u(3),t,ro
    logical,intent(in) :: info
    !
    ! local data
    real(real_kind) :: uref,u1,u2,delta,mc,ustar
    logical,save :: firstcall=.true.
    !+-----------------------------------+
    !| delta: inflow vorticity thickness |
    !+-----------------------------------+
    u1=1.5e0_real_kind
    u2=0.5e0_real_kind
    uref = 0.5e0_real_kind*(u1-u2)
    !
    delta=1.e0_real_kind
    mc=(u1-u2)/2.e0_real_kind*mach
    !
    u(1)=0.5e0_real_kind*(u1+u2)+uref*tanh(2.e0_real_kind*y/delta)
    u(2)=0.e0_real_kind
    u(3)=0.e0_real_kind
    !
    ustar=u(1)-0.5e0_real_kind*(u1+u2)
    t=1.e0_real_kind*(1.e0_real_kind+mc**2*0.5e0_real_kind*(gamma-1.e0_real_kind)*(1.e0_real_kind-(ustar/uref)**2))
    !
    ro=1.e0_real_kind/t
    !
    if(info .and. firstcall) then
      print*,' ** High-speed Mach number=',u1*mach
      print*,' **  Low-speed Mach number=',u2*mach
      print*,' ** Convective Mach number=',mc
      print*,' **  Shear-layer thickness=',delta
      firstcall=.false.
    endif
    !
    return
    !
  end subroutine mixinglayervel
  !+-------------------------------------------------------------------+
  !| The end of the function mixinglayervel.                           |
  !+-------------------------------------------------------------------+
  !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! This subroutine is used to cauculate the post-shock flow paramters,
  ! by using pre-shock flow paramters and shock angle and invisid R-H
  ! Relation.
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Writen by Fang Jian, 2010-10-23.
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine postshock(ropre,upre,vpre,ppre,tpre,ropos,upos,vpos,      &
                                                          ppos,tpos,ang)
    !
    use commvar, only: const2,gamma,mach
    use constdef,only: pi
    !
    real(real_kind),intent(in)  :: ropre,upre,vpre,ppre,tpre,ang
    real(real_kind),intent(out) :: ropos,upos,vpos,ppos,tpos
    !
    real(real_kind) :: csspre,unpre,utpre,machnpre,unpos,utpos,sratio,ang2
    !
    ang2=ang/180.e0_real_kind*pi
    !
    csspre=sqrt(tpre)/mach
    !
    unpre=upre*sin(ang2)-vpre*cos(ang2)
    utpre=upre*cos(ang2)+vpre*sin(ang2)
    machnpre=unpre/csspre
    ! print*,'machnpre=',machnpre
    ! print*,'unpre=',unpre
    ! print*,'csspre=',csspre
    sratio=2.e0_real_kind*gamma/(gamma+1.e0_real_kind)*machnpre**2-(gamma-1.e0_real_kind)/(gamma+1.e0_real_kind)
    !
    ppos=ppre*sratio
    ropos=(sratio*(gamma+1.e0_real_kind)+gamma-1.e0_real_kind)/                            &
                                  (gamma+1.e0_real_kind+sratio*(gamma-1.e0_real_kind))*ropre
    tpos=ppos/ropos*const2
    !
    unpos=ropre*unpre/ropos
    utpos=utpre
    upos=unpos*sin(ang2)+utpos*cos(ang2)
    vpos=-unpos*cos(ang2)+utpos*sin(ang2)
    !
  end subroutine postshock
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! end of the subroutine postshockcal.
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !+-------------------------------------------------------------------+
  !| This function is used to assign the inflow condition for the three|
  !| stream coaxial jet flow.                                          |
  !+-------------------------------------------------------------------+
  !| ref: Bouheraoua, L., Domingo, P., Ribert, G. Large-eddy simulation|
  !| of a supersonic lifted jet flame: Analysis of the turbulent flame |
  !| base. Combust. Flame, 2017, 179, 199-218.                         |                                               |
  !+-------------------------------------------------------------------+
  !| CHANGE RECORD                                                     |
  !| -------------                                                     |
  !| 04-Feb-2022: Created by Z.X. Chen @ Hengdong                      |
  !+-------------------------------------------------------------------+
  subroutine multistream_inflow(stream,rovd,rho,vel,prs,tmp,spc,ubulk)
    !
    use commvar
#ifdef COMB
    use thermchem, only: spcindex,convertxiyi
#endif
    !
    ! arguments
    character(len=*),intent(in) :: stream
    real(real_kind),intent(in),optional :: rovd
    real(real_kind),intent(out),optional :: &
      rho,prs,tmp,spc(num_species),vel(ndims),ubulk
    !
#ifdef COMB
    !local
    real(real_kind) :: vel_ref(ndims),vel0(ndims),ub,spcx(num_species),h,delta &
              ,spc_ref(num_species),tmp_ref
    !
    vel_ref(:)=0.e0_real_kind
    spc_ref(:)=0.e0_real_kind
    spcx(:)=0.e0_real_kind
    tmp=tinf
    !
    if(trim(stream)=='fuel') then
      !
      tmp_ref=545.e0_real_kind
      spcx(spcindex('H2'))=1.e0_real_kind
      ! spcx(spcindex('N2'))=1.d0-sum(spcx(:))
      call convertxiyi(spcx(:),spc_ref(:),'X2Y')
      spc(:)=spc_ref(:)
      ! tmp=tmp_ref
      !
      ub=1.78e2_real_kind
      ! ub=680.d0!14.1d0
      if(.true. .and. present(rovd)) then 
        ! vel_ref(1)=ub*1.128d0*((1.d0-2.d0*rovd)**(1.d0/7.d0))
        h=0.5e0_real_kind
        delta=0.05e0_real_kind
        vel_ref(1)=100.e0_real_kind+0.5e0_real_kind*(ub-100.e0_real_kind)*(1.e0_real_kind-tanh(2.e0_real_kind*(rovd-h)/delta))
        ! spc(spcindex('H2'))=spcinf(spcindex('H2'))+0.5d0*(spc_ref(spcindex('H2')) &
        !                 -spcinf(spcindex('H2')))*(1.d0-tanh(2.d0*(rovd-h)/delta))
        ! spc(spcindex('N2'))=1.d0-sum(spc(:))
        ! vel_ref(1)=vel_ref(1)*1.27767d0*((1.d0-2.d0*rovd)**(1.d0/3.d0))
        ! vel_ref(2:ndims)=0.d0*vel_ref(1)
        !
        tmp=tinf+0.5e0_real_kind*(tmp_ref-tinf)*(1.e0_real_kind-tanh(2.e0_real_kind*(rovd-h)/delta))
        vel(:)=vel_ref(:)
        ! if(nstep==0) vel(:)=whitenoise(vel_ref(:),vel(:),0.1d0)
      else 
        vel(1)=ub
      endif 
      !
    elseif(trim(stream)=='hotcoflow') then
      !
      tmp_ref=1250.e0_real_kind
      spc_ref(spcindex('O2'))=0.24488e0_real_kind
      spc_ref(spcindex('H2O'))=0.17491e0_real_kind
      spc_ref(spcindex('N2'))=1.e0_real_kind-sum(spc_ref(:))
      ! call convertxiyi(spcx(:),spc_ref(:),'X2Y')
      spc(:)=spc_ref(:)
      ! tmp=tmp_ref
      ! spc(spcindex('O2'))=0.233d0
      ! spc(spcindex('N2'))=1.d0-sum(spc(:))
      !
      ub=1.42e2_real_kind
      ! ub=1.d0
      if(.true. .and. present(rovd)) then 
        h=0.5e0_real_kind
        delta=0.05e0_real_kind
        vel_ref(1)=100.e0_real_kind+0.5e0_real_kind*(ub-100.e0_real_kind)*(1.e0_real_kind-tanh(2.e0_real_kind*(rovd-h)/delta))
        ! spc(spcindex('O2'))=spcinf(spcindex('O2'))+0.5d0*(spc_ref(spcindex('H2')) &
        !                 -spcinf(spcindex('H2')))*(1.d0-tanh(2.d0*(rovd-h)/delta))
        ! vel_ref(1)=ub*1.128d0*((1.d0-2.d0*rovd)**(1.d0/7.d0))
        ! vel_ref(1)=vel_ref(1)*1.27767d0*((1.d0-2.d0*rovd)**(1.d0/3.d0))
        ! vel_ref(2:ndims)=0.d0*vel_ref(1)
        !
        tmp=tinf+0.5e0_real_kind*(tmp_ref-tinf)*(1.e0_real_kind-tanh(2.e0_real_kind*(rovd-h)/delta))
        vel(:)=vel_ref(:)
        ! if(nstep==0) vel(:)=whitenoise(vel_ref(:),vel(:),0.1d0)
      else 
        vel(1)=ub
      endif
      !
    elseif(trim(stream)=='air') then
      !
      tmp=tinf
      spc(:)=spcinf(:)
      !
      ub=100.e0_real_kind
      vel(:)=0.e0_real_kind
      vel(1)=ub
      !
    else
      stop ' !! stream type not set for the inlet boundary !!'
    endif
    !
    prs=pinf
    rho=thermal(pressure=prs,temperature=tmp,species=spc(:))
    !
    if(present(ubulk))ubulk=ub
    !
#endif
    !
  end subroutine multistream_inflow
  !+-------------------------------------------------------------------+
  !| The end of the subroutine multistream_inflow.                     |
  !+-------------------------------------------------------------------+
  !
  !!
end module fludyna
!+---------------------------------------------------------------------+
!| The end of the module fludyna.                                      |
!+---------------------------------------------------------------------+
