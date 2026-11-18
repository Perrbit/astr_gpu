program mixed_viscous_probe
  use iso_c_binding
  use ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_quiet_nan
  use mpi
  use device_state
  use halo_transport
  use fluid_explicit
  implicit none
  type(c_ptr) :: state
  type(device_view) :: view
  integer :: rank,ranks,local_rank,ierr,status,mode,stage,debug,unit,h(12),im,jm,km,hm,i,j,k,c,a,b,field,reconcile
  integer :: topology(3),axis,lo(3),hi(3),origin(3),global_upper(3),cases=0,contrasts=0
  integer(c_int) :: transport
  real(c_double) :: constants(8),maximum=0,grad_error=0,cell_error=0,error,mu,conductivity,divergence,s(3,3), &
    heat(3),rounded_heat(3),bias,coeff(3),guard_value
  real(c_double),allocatable :: base(:,:,:,:),q(:,:,:,:),primitives(:,:,:,:),metric(:,:,:,:,:),jac(:,:,:), &
    gradient(:,:,:,:,:),gref(:,:,:,:,:),thermal(:,:,:,:),tref(:,:,:,:),conv(:,:,:,:),full(:,:,:,:),next(:,:,:,:), &
    actual(:,:,:,:),raw8(:,:,:,:),expected8(:,:,:,:),guard(:),observed(:,:,:,:)
  real(c_float),allocatable :: raw4(:,:,:,:),expected4(:,:,:,:)
  real(c_float) :: nan4
  real(c_double) :: nan8
  character(1024) :: root,path
  character(80) :: text
  integer(c_int64_t) :: viscous_bytes,reference_bytes
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text,status=status)
  if(status/=0) error stop 'mixed flux component needs Open MPI local rank'
  read(text,*) local_rank
  call check(device_bind(int(local_rank,c_int)))
  call MPI_Init(ierr);call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  call get_command_argument(1,root);call get_command_argument(2,text);read(text,*) topology
  if(product(topology)/=ranks.or.(ranks/=1.and.ranks/=2)) error stop 'mixed fixture topology'
  axis=maxloc(topology,dim=1)
  lo=rank;hi=rank
  if(ranks==2) then
    lo(axis)=1-rank;hi(axis)=1-rank
  endif
  call get_command_argument(3,text)
  select case(trim(text))
  case('pinned');transport=halo_pinned
  case('device');transport=halo_device
  case default;error stop 'mixed transport must be pinned/device'
  end select
  nan4=transfer(int(z'7fc12345',c_int32_t),0._c_float)
  nan8=ieee_value(0.d0,ieee_quiet_nan)
  do mode=0,1
  do stage=1,3
    write(path,'(a,a,i1,a,i1,a,i8.8,a)') trim(root),'/mode',mode,'.stage',stage,'.rank',rank,'.bin'
    open(newunit=unit,file=trim(path),status='old',access='stream',form='unformatted')
    read(unit) h,constants
    global_upper=h(1:3);im=h(4);jm=h(5);km=h(6);hm=h(7);origin=h(8:10)
    if(h(11)/=mode.or.h(12)/=stage) error stop 'mixed record identity'
    allocate(base(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5))
    allocate(primitives(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6),metric(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3))
    allocate(jac(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
    allocate(gradient(0:im,0:jm,0:km,3,3),gref(0:im,0:jm,0:km,3,3))
    allocate(thermal(0:im,0:jm,0:km,3),tref(0:im,0:jm,0:km,3))
    allocate(conv(0:im,0:jm,0:km,5),full(0:im,0:jm,0:km,5),next(0:im,0:jm,0:km,5))
    allocate(actual,mold=q)
    read(unit) base,q,primitives,metric,jac,gref,tref,conv,full,next;close(unit)
    do debug=0,1
      call check(device_state_create(int(local_rank,c_int),int(im,c_int),int(jm,c_int),int(km,c_int), &
        int(hm,c_int),int(debug,c_int),state))
      call check(fluid_prepare(state,1_c_int))
      if(mode==1) then
        call check(fluid_rhs_prepare_mixed_viscous(state))
      else
        call check(fluid_rhs_prepare(state,1_c_int))
      endif
      viscous_bytes=0;reference_bytes=0
      do field=device_stress,device_heat_flux
        call check(device_state_view(state,int(field,c_int),view))
        if(view%element_bytes/=merge(4,8,mode==1)) error stop 'wrong viscous storage kind'
        viscous_bytes=viscous_bytes+product(int(view%extent,c_int64_t))*view%element_bytes
        reference_bytes=reference_bytes+product(int(view%extent,c_int64_t))*8
      enddo
      if(viscous_bytes*merge(2,1,mode==1)/=reference_bytes) error stop 'wrong mixed workspace byte count'
      if(stage==1.and.debug==0) print *, 'CPP_MIXED_VISCOUS_STORAGE ',rank,mode,viscous_bytes,reference_bytes
      do field=device_q,device_dtemperature
        call check(device_state_view(state,int(field,c_int),view))
        if(view%element_bytes/=8) error stop 'mixed workspace changed authoritative/gradient precision'
      enddo
      call check(device_transport_prepare(state,int(MPI_COMM_WORLD,c_int),int(lo,c_int),int(hi,c_int), &
        int(topology,c_int),6_c_int,transport,0_c_int,[1_c_int,1_c_int,1_c_int]))
      if(stage==1) then
        call constitutive_fixture()
        call transport_fixture()
      endif
      call check(device_state_upload(state,device_jacobian,jac,bytes(size(jac),8)))
      call check(device_state_upload(state,device_q,base,bytes(size(base),8)))
      call check(fluid_rk_save(state))
      call check(device_state_upload(state,device_q,q,bytes(size(q),8)))
      call check(device_state_upload(state,device_rho,primitives(:,:,:,1),bytes(size(jac),8)))
      call check(device_state_upload(state,device_velocity,primitives(:,:,:,2:4),bytes(3*size(jac),8)))
      call check(device_state_upload(state,device_pressure,primitives(:,:,:,5),bytes(size(jac),8)))
      call check(device_state_upload(state,device_temperature,primitives(:,:,:,6),bytes(size(jac),8)))
      call check(device_state_upload(state,device_metric,metric,bytes(size(metric),8)))
      call check(fluid_convection(state))
      call check(device_state_download(state,device_rhs,actual(0:im,0:jm,0:km,:),bytes(size(conv),8)))
      call compare_component(actual(0:im,0:jm,0:km,:),conv,'convection')
      call check(fluid_gradient(state))
      call check(device_state_download(state,device_dvelocity,gradient,bytes(size(gradient),8)))
      call check(device_state_download(state,device_dtemperature,thermal,bytes(size(thermal),8)))
      error=max(maxval(abs(gradient-gref))/max(1.d0,maxval(abs(gref))), &
        maxval(abs(thermal-tref))/max(1.d0,maxval(abs(tref))))
      if(.not.(error<=2.d-10)) error stop 'mixed source gradient differs'
      grad_error=max(grad_error,error)
      call poison_flux()
      call check(fluid_viscous_flux(state,constants(1),constants(2),constants(3),constants(4),constants(5)))
      do a=0,2
        call check(device_exchange_axis(state,device_stress,int(a,c_int),0_c_int))
        call check(device_exchange_axis(state,device_heat_flux,int(a,c_int),0_c_int))
      enddo
      call check(fluid_viscous_divergence(state))
      call check(device_state_download(state,device_rhs,actual(0:im,0:jm,0:km,:),bytes(size(full),8)))
      call compare_component(actual(0:im,0:jm,0:km,:),full,'full RHS')
      select case(stage)
      case(1);coeff=[1.d0,0.d0,1.d0]
      case(2);coeff=[.75d0,.25d0,.25d0]
      case(3);coeff=[1.d0/3,2.d0/3,2.d0/3]
      end select
      call check(fluid_rk_update(state,constants(6),coeff(1),coeff(2),coeff(3)))
      call check(device_state_download(state,device_q,actual,bytes(size(actual),8)))
      call compare_component(actual(0:im,0:jm,0:km,:),next,'RK state')
      if(.not.all(ieee_is_finite(actual(0:im,0:jm,0:km,:))).or.any(actual(0:im,0:jm,0:km,1)<=0)) &
        error stop 'mixed physical state'
      if(any(actual(0:im,0:jm,0:km,5)-.5d0*sum(actual(0:im,0:jm,0:km,2:4)**2,dim=4)/ &
        actual(0:im,0:jm,0:km,1)<=0)) error stop 'mixed negative internal energy'
      call check(device_state_destroy(state));cases=cases+1
    enddo
    deallocate(base,q,primitives,metric,jac,gradient,gref,thermal,tref,conv,full,next,actual)
  enddo
  enddo
  print *, 'CPP_MIXED_VISCOUS_PASS ',rank,cases,maximum,grad_error,cell_error,contrasts
  call MPI_Finalize(ierr)
contains
  integer(c_size_t) function bytes(n,width)
    integer,intent(in) :: n,width
    bytes=int(n,c_size_t)*width
  end function
  subroutine check(code)
    integer,intent(in) :: code
    if(code==0) return
    print *, 'MIXED_STATUS ',rank,mode,stage,debug,code
    call MPI_Abort(MPI_COMM_WORLD,1,ierr)
  end subroutine
  subroutine compare_component(values,reference,label)
    real(c_double),intent(in) :: values(:,:,:,:),reference(:,:,:,:)
    character(*),intent(in) :: label
    if(.not.all(ieee_is_finite(values))) error stop 'nonfinite mixed operator'
    do c=1,5
      error=maxval(abs(values(:,:,:,c)-reference(:,:,:,c)))/max(1.d0,maxval(abs(reference(:,:,:,c))))
      if(.not.(error<=2.d-10)) then
        print *, 'MIXED_DIFFERENCE ',rank,mode,stage,debug,label,c,error
        error stop 'mixed operator differs from same-mode old GPU'
      endif
      maximum=max(maximum,error)
    enddo
  end subroutine
  subroutine poison_flux()
    integer :: ncomp,field_id
    do field_id=device_stress,device_heat_flux
      ncomp=merge(6,3,field_id==device_stress)
      if(mode==1) then
        allocate(raw4(-hm:im+hm,-hm:jm+hm,-hm:km+hm,ncomp));raw4=nan4
        call check(device_state_upload_f32(state,int(field_id,c_int),raw4,bytes(size(raw4),4)));deallocate(raw4)
      else
        allocate(raw8(-hm:im+hm,-hm:jm+hm,-hm:km+hm,ncomp));raw8=nan8
        call check(device_state_upload(state,int(field_id,c_int),raw8,bytes(size(raw8),8)));deallocate(raw8)
      endif
    enddo
  end subroutine
  subroutine constitutive_fixture()
    real(c_double),allocatable :: v(:,:,:,:),t(:,:,:),expected_stress(:,:,:,:),expected_heat(:,:,:,:)
    real(c_double) :: tensor(3,3),g(3,3),gt(3),velocity(3),temperature,rounded(3,3),hc
    integer :: ii,jj,kk,cc,aa,bb,before
    allocate(v(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3),t(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
    allocate(expected_stress(0:im,0:jm,0:km,6),expected_heat(0:im,0:jm,0:km,3))
    v=nan8;t=nan8;before=contrasts
    do kk=0,km
    do jj=0,jm
    do ii=0,im
      velocity=[3.7d0+.013d0*jj,-2.9d0+.021d0*kk,1.3d0+.017d0*ii]
      temperature=1.2d0+.01d0*ii
      do aa=1,3
        gt(aa)=.02d0*sin(.9d0*aa+.01d0*ii+.03d0*jj+.02d0*kk)
        do bb=1,3
          g(aa,bb)=.2d0*sin(.3d0*aa+.7d0*bb+.013d0*ii+.017d0*jj+.019d0*kk)
        enddo
      enddo
      v(ii,jj,kk,:)=velocity;t(ii,jj,kk)=temperature
      gradient(ii,jj,kk,:,:)=g;thermal(ii,jj,kk,:)=gt
      mu=temperature*sqrt(temperature)*1.4d0/(temperature+.4d0)/8.d0
      divergence=g(1,1)+g(2,2)+g(3,3)
      tensor=mu*(g+transpose(g))
      do aa=1,3
        tensor(aa,aa)=tensor(aa,aa)-(2.d0/3)*mu*divergence
      enddo
      expected_stress(ii,jj,kk,:)=[tensor(1,1),tensor(1,2),tensor(1,3),tensor(2,2),tensor(2,3),tensor(3,3)]
      hc=(mu/.72d0)/.4d0
      expected_heat(ii,jj,kk,:)=hc*gt+matmul(tensor,velocity)
      rounded=real(real(tensor,c_float),c_double)
      rounded_heat=hc*gt+matmul(rounded,velocity)
      contrasts=contrasts+count(real(rounded_heat,c_float)/=real(expected_heat(ii,jj,kk,:),c_float))
    enddo
    enddo
    enddo
    if(contrasts==before) error stop 'constitutive fixture does not distinguish early stress rounding'
    do field=device_q,device_save
      call check(device_state_view(state,int(field,c_int),view));allocate(guard(product(view%extent)))
      guard=88.d0+field;call check(device_state_upload(state,int(field,c_int),guard,bytes(size(guard),8)))
      deallocate(guard)
    enddo
    call check(device_state_upload(state,device_velocity,v,bytes(size(v),8)))
    call check(device_state_upload(state,device_temperature,t,bytes(size(t),8)))
    call check(device_state_upload(state,device_dvelocity,gradient,bytes(size(gradient),8)))
    call check(device_state_upload(state,device_dtemperature,thermal,bytes(size(thermal),8)))
    call poison_flux()
    call check(fluid_viscous_flux(state,8.d0,.72d0,.4d0,.4d0,1.4d0))
    do field=device_stress,device_heat_flux
      cc=merge(6,3,field==device_stress)
      if(mode==1) then
        allocate(raw4(-hm:im+hm,-hm:jm+hm,-hm:km+hm,cc),expected4(0:im,0:jm,0:km,cc))
        call check(device_state_download_f32(state,int(field,c_int),raw4,bytes(size(raw4),4)))
        if(field==device_stress) then
          expected4=real(expected_stress,c_float)
        else
          expected4=real(expected_heat,c_float)
        endif
        if(any(transfer(raw4(0:im,0:jm,0:km,:),[0_c_int32_t],size(expected4))/= &
          transfer(expected4,[0_c_int32_t],size(expected4)))) error stop 'mixed final-store constitutive bits'
        deallocate(raw4,expected4)
      else
        allocate(raw8(-hm:im+hm,-hm:jm+hm,-hm:km+hm,cc))
        call check(device_state_download(state,int(field,c_int),raw8,bytes(size(raw8),8)))
        if(field==device_stress) then
          error=maxval(abs(raw8(0:im,0:jm,0:km,:)-expected_stress))
        else
          error=maxval(abs(raw8(0:im,0:jm,0:km,:)-expected_heat))
        endif
        if(.not.(error<=2.d-10)) error stop 'FP64 constitutive manufactured state'
        cell_error=max(cell_error,error);deallocate(raw8)
      endif
    enddo
    do field=device_q,device_save
      call check(device_state_view(state,int(field,c_int),view));allocate(guard(product(view%extent)))
      call check(device_state_download(state,int(field,c_int),guard,bytes(size(guard),8)))
      if(any(guard/=88.d0+field)) error stop 'constitutive kernel modified q/RHS/save'
      deallocate(guard)
    enddo
    deallocate(v,t,expected_stress,expected_heat)
  end subroutine
  real(c_float) function packet_value(position,component,owner_bias) result(value)
    integer,intent(in) :: position(3),component
    real(c_double),intent(in) :: owner_bias
    integer :: global(3)
    global=modulo(position+origin,global_upper)
    if(all(global==0).and.component==1) then
      value=transfer(int(z'80000000',c_int32_t),0._c_float)
    else
      value=real(component*4096+global(1)+global_upper(1)*(global(2)+global_upper(2)*global(3)),c_float)+ &
        real(owner_bias,c_float)
    endif
  end function
  subroutine transport_fixture()
    integer :: field_id,ncomp,average,ii,jj,kk,cc,d,source_rank,position(3),last(3),outside_count
    real(c_double) :: owner_bias
    last=[im,jm,km]
    do average=0,1
    do field_id=device_stress,device_heat_flux
      ncomp=merge(6,3,field_id==device_stress)
      allocate(expected4(-hm:im+hm,-hm:jm+hm,-hm:km+hm,ncomp));expected4=nan4
      do kk=0,km
      do jj=0,jm
      do ii=0,im
      do cc=1,ncomp
        expected4(ii,jj,kk,cc)=packet_value([ii,jj,kk],cc,.125d0*rank)
      enddo
      enddo
      enddo
      enddo
      if(mode==1) then
        allocate(raw4,mold=expected4)
        call check(device_state_upload_f32(state,int(field_id,c_int),expected4,bytes(size(expected4),4)))
      else
        allocate(raw8(-hm:im+hm,-hm:jm+hm,-hm:km+hm,ncomp));raw8=real(expected4,c_double)
        call check(device_state_upload(state,int(field_id,c_int),raw8,bytes(size(raw8),8)))
      endif
      if(average==1) then
        do d=0,2
          call check(device_exchange_begin(state,int(field_id,c_int),int(d,c_int),1_c_int))
          call check(device_exchange_end(state,int(field_id,c_int),int(d,c_int),1_c_int))
        enddo
      endif
      do d=0,2
        call check(device_exchange_axis(state,int(field_id,c_int),int(d,c_int),0_c_int))
      enddo
      do kk=-hm,km+hm
      do jj=-hm,jm+hm
      do ii=-hm,im+hm
        position=[ii,jj,kk];outside_count=count(position<0.or.position>last)
        if(outside_count>1) cycle
        source_rank=rank
        if(ranks==2.and.(position(axis)<0.or.position(axis)>last(axis))) source_rank=1-rank
        owner_bias=.125d0*source_rank
        if(ranks==2.and.average==1.and.(position(axis)==0.or.position(axis)==last(axis))) owner_bias=.0625d0
        do cc=1,ncomp
          expected4(ii,jj,kk,cc)=packet_value(position,cc,owner_bias)
        enddo
      enddo
      enddo
      enddo
      if(mode==1) then
        call check(device_state_download_f32(state,int(field_id,c_int),raw4,bytes(size(raw4),4)))
        if(any(transfer(raw4,[0_c_int32_t],size(raw4))/= &
          transfer(expected4,[0_c_int32_t],size(expected4)))) error stop 'FP32 halo/shared/corner/signed-zero bits'
        deallocate(raw4)
      else
        call check(device_state_download(state,int(field_id,c_int),raw8,bytes(size(raw8),8)))
        if(any(transfer(raw8,[0_c_int64_t],size(raw8))/= &
          transfer(real(expected4,c_double),[0_c_int64_t],size(expected4)))) error stop 'FP64 halo bits'
        deallocate(raw8)
      endif
      deallocate(expected4)
    enddo
    enddo
  end subroutine
end program
