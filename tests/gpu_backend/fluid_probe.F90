program fluid_probe
  use iso_c_binding
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use mpi
  use device_state
  use fluid_explicit
  use cfl_spectrum, only: spectral_rates
  implicit none
  real(c_double),parameter :: gamma=1.4d0,mach=.1d0
  real(c_double),parameter :: c1=1.d0/(gamma*(gamma-1.d0)*mach**2),c2=gamma*mach**2,c6=1.d0/(gamma-1.d0)
  integer :: ierr,rank,nranks,local_rank,status,debug
  character(len=1024) :: qpath,ppath,text
  type(c_ptr) :: handle
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text,status=status)
  if(status/=0) error stop 'fluid probe requires Open MPI local rank'
  read(text,*) local_rank
  status=device_bind(int(local_rank,c_int))
  if(status/=0) error stop 'fluid probe pre-MPI binding failed'
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
  call MPI_Comm_size(MPI_COMM_WORLD,nranks,ierr)
  call get_command_argument(1,qpath)
  call get_command_argument(2,ppath)
  do debug=0,1
    if(len_trim(qpath)>0) then
      call frozen_conversion()
    else
      call manufactured()
    endif
  enddo
  call MPI_Finalize(ierr)
contains
  subroutine fail(message)
    character(*),intent(in) :: message
    write(*,'(a,i0,2a)') 'FLUID FAIL rank=',rank,': ',message
    call MPI_Abort(MPI_COMM_WORLD,1,ierr)
    error stop 1
  end subroutine
  subroutine check(code,message)
    integer,intent(in) :: code
    character(*),intent(in) :: message
    if(code/=0) then
      write(*,'(a,i0)') 'device status=',code
      call fail(message)
    endif
  end subroutine
  subroutine compare(actual,reference,name)
    real(c_double),intent(in) :: actual(:),reference(:)
    character(*),intent(in) :: name
    real(c_double) :: error
    if(.not.all(ieee_is_finite(actual)).or..not.all(ieee_is_finite(reference))) call fail(name//' nonfinite')
    error=maxval(abs(actual-reference))/max(1.d0,maxval(abs(reference)))
    if(.not.(error<=2.d-10)) call fail(name)
    write(*,'(a,i0,a,i0,2a,es24.16)') 'FLUID rank=',rank,' debug=',debug,' ',name,error
  end subroutine
  integer(c_size_t) function bytes(count)
    integer,intent(in) :: count
    bytes=int(count,c_size_t)*c_sizeof(0.d0)
  end function
  subroutine frozen_conversion()
    integer(c_int) :: header(5),primitive_header(4),im,jm,km,hm,nq
    integer :: unit,ios,field,n
    character(32) :: name
    real(c_double) :: good_density
    type(device_view) :: view
    real(c_double),allocatable :: q(:,:,:,:),back(:,:,:,:),reference(:,:,:,:),scalar(:,:,:),vel(:,:,:,:)
    open(newunit=unit,file=trim(qpath),access='stream',form='unformatted',status='old',action='read')
    read(unit) header
    im=header(1);jm=header(2);km=header(3);hm=header(4);nq=header(5)
    if(nq/=5) call fail('frozen component count')
    allocate(q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),back(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5), &
      reference(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6),scalar(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
      vel(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3))
    read(unit) q
    read(unit,iostat=ios) n
    if(ios>=0) call fail('frozen q trailing payload')
    close(unit)
    open(newunit=unit,file=trim(ppath),access='stream',form='unformatted',status='old',action='read')
    read(unit) primitive_header
    if(any(primitive_header/=header(1:4))) call fail('primitive header')
    read(unit) reference
    close(unit)
    call check(device_state_create(int(local_rank,c_int),im,jm,km,hm,int(debug,c_int),handle),'create frozen')
    call check(fluid_prepare(handle,0_c_int),'prepare conversion')
    status=device_state_view(handle,device_dvelocity,view)
    if(status/=device_not_allocated) call fail('inactive gradients allocated')
    if(fluid_gradient(handle)/=device_not_allocated) call fail('inactive gradient operation accepted')
    call check(device_state_upload(handle,device_q,q,bytes(size(q))),'upload frozen q')
    scalar=0.d0;vel=0.d0
    call check(device_state_upload(handle,device_rho,scalar,bytes(size(scalar))),'initialize rho halo')
    call check(device_state_upload(handle,device_pressure,scalar,bytes(size(scalar))),'initialize pressure halo')
    call check(device_state_upload(handle,device_temperature,scalar,bytes(size(scalar))),'initialize temperature halo')
    call check(device_state_upload(handle,device_velocity,vel,bytes(size(vel))),'initialize velocity halo')
    call check(fluid_primitives(handle,c2,c6),'recover primitives')
    call check(fluid_check(handle),'finite positive primitives')
    call check(device_state_download(handle,device_rho,scalar,bytes(size(scalar))),'download rho')
    call compare(reshape(scalar(0:im,0:jm,0:km),[(im+1)*(jm+1)*(km+1)]), &
      reshape(reference(0:im,0:jm,0:km,1),[(im+1)*(jm+1)*(km+1)]),'rho ')
    call check(device_state_download(handle,device_velocity,vel,bytes(size(vel))),'download velocity')
    do n=1,3
      write(name,'(a,i0)') 'velocity_',n
      call compare(reshape(vel(0:im,0:jm,0:km,n),[(im+1)*(jm+1)*(km+1)]), &
        reshape(reference(0:im,0:jm,0:km,n+1),[(im+1)*(jm+1)*(km+1)]),trim(name)//' ')
    enddo
    do n=5,6
      field=device_pressure
      name='pressure'
      if(n==6) field=device_temperature
      if(n==6) name='temperature'
      call check(device_state_download(handle,int(field,c_int),scalar,bytes(size(scalar))),'download scalar')
      call compare(reshape(scalar(0:im,0:jm,0:km),[(im+1)*(jm+1)*(km+1)]), &
        reshape(reference(0:im,0:jm,0:km,n),[(im+1)*(jm+1)*(km+1)]),trim(name)//' ')
    enddo
    call check(fluid_conservative(handle,c1),'conservative from recovered temperature')
    call check(fluid_check(handle),'conservative admissibility')
    call check(device_state_download(handle,device_q,back,bytes(size(back))),'download conservative')
    do n=1,5
      write(name,'(a,i0)') 'q_',n
      call compare(reshape(back(0:im,0:jm,0:km,n),[(im+1)*(jm+1)*(km+1)]), &
        reshape(q(0:im,0:jm,0:km,n),[(im+1)*(jm+1)*(km+1)]),trim(name)//' ')
    enddo
    good_density=q(0,0,0,1)
    q(0,0,0,1)=-1.d0
    call check(device_state_upload(handle,device_q,q,bytes(size(q))),'upload nonphysical diagnostic')
    call check(fluid_primitives(handle,c2,c6),'launch nonphysical diagnostic')
    if(fluid_check(handle)/=device_nonphysical) call fail('nonpositive density was accepted')
    write(*,'(a)') 'FLUID nonpositive_density_rejected'
    q(0,0,0,1)=good_density
    call check(device_state_upload(handle,device_q,q,bytes(size(q))),'restore diagnostic input')
    call check(fluid_primitives(handle,c2,c6),'revisit valid diagnostic state')
    if(fluid_check(handle)/=device_nonphysical) call fail('later valid state erased numerical failure')
    write(*,'(a)') 'FLUID numerical_failure_remains_latched'
    call check(device_state_destroy(handle),'destroy frozen')
  end subroutine
  subroutine manufactured()
    integer(c_int),parameter :: im=16,jm=12,km=8,hm=5
    real(c_double),allocatable :: vel(:,:,:,:),temperature(:,:,:),metric(:,:,:,:,:), &
      dvel(:,:,:,:,:),dtmp(:,:,:,:),expected(:,:,:,:,:),expected_t(:,:,:,:),density(:,:,:)
    real(c_double) :: pi,h(3),symbol(3),symbol2,projection(3,3),gradient(3,3),dt(3)
    real(c_double) :: x,y,z,oracle(3),rates(3),one(4),u(3),m(3,3),sound
    real(c_double) :: totals(13),stats(13),qnode(5),omega(3),strain(3,3),mu,div,heat
    real(c_double) :: full_oracle(4),diagnostic(4),planes(4,0:jm),plane_reference(4,0:jm)
    integer(c_int64_t) :: keys(4),key
    integer :: i,j,k,a,b,n,ii,jj,kk
    character(32) :: name
    logical :: valid
    pi=acos(-1.d0);h=2*pi/real([im,jm,km],c_double)
    symbol=1.5d0*sin(h)-.3d0*sin(2*h)+sin(3*h)/30.d0
    symbol2=1.5d0*sin(2*h(2))-.3d0*sin(4*h(2))+sin(6*h(2))/30.d0
    projection=reshape([1.2d0,.1d0,-.2d0, .2d0,.9d0,.1d0, -.1d0,.3d0,1.1d0],[3,3])
    allocate(vel(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3),temperature(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
      metric(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3),dvel(0:im,0:jm,0:km,3,3), &
      dtmp(0:im,0:jm,0:km,3),expected(0:im,0:jm,0:km,3,3),expected_t(0:im,0:jm,0:km,3), &
      density(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
    do k=-hm,km+hm
    do j=-hm,jm+hm
    do i=-hm,im+hm
      x=i*h(1);y=j*h(2);z=k*h(3)
      vel(i,j,k,:)=[sin(x)+.25d0*cos(2*y)+.05d0*rank,cos(y)+.2d0*sin(z),sin(z)+.1d0*cos(x)]
      temperature(i,j,k)=2.d0+.1d0*cos(x+y+z)
      do a=1,3
        metric(i,j,k,a,:)=projection(a,:)/h(a)
      enddo
    enddo
    enddo
    enddo
    do k=0,km
    do j=0,jm
    do i=0,im
      x=i*h(1);y=j*h(2);z=k*h(3)
      gradient=0.d0
      gradient(1,1)=symbol(1)*cos(x);gradient(2,1)=-.25d0*symbol2*sin(2*y)
      gradient(2,2)=-symbol(2)*sin(y);gradient(3,2)=.2d0*symbol(3)*cos(z)
      gradient(1,3)=-.1d0*symbol(1)*sin(x);gradient(3,3)=symbol(3)*cos(z)
      dt=-.1d0*symbol*sin(x+y+z)
      do a=1,3
      do b=1,3
        expected(i,j,k,a,b)=dot_product(gradient(:,a),metric(i,j,k,:,b))
      enddo
      enddo
      do b=1,3
        expected_t(i,j,k,b)=dot_product(dt,metric(i,j,k,:,b))
      enddo
    enddo
    enddo
    enddo
    call check(device_state_create(int(local_rank,c_int),im,jm,km,hm,int(debug,c_int),handle),'create analytic')
    call check(fluid_prepare(handle,1_c_int),'prepare gradients')
    call check(device_state_upload(handle,device_velocity,vel,bytes(size(vel))),'upload analytic velocity')
    call check(device_state_upload(handle,device_temperature,temperature,bytes(size(temperature))),'upload analytic T')
    call check(device_state_upload(handle,device_metric,metric,bytes(size(metric))),'upload full metric')
    call check(fluid_gradient(handle),'gradient launch')
    call check(device_state_download(handle,device_dvelocity,dvel,bytes(size(dvel))),'gradient download')
    call check(device_state_download(handle,device_dtemperature,dtmp,bytes(size(dtmp))),'temperature gradient download')
    do a=1,3
    do b=1,3
      n=(im+1)*(jm+1)*(km+1)
      write(name,'(a,i0,a,i0)') 'dvelocity_',a,'_',b
      call compare(reshape(dvel(:,:,:,a,b),[n]),reshape(expected(:,:,:,a,b),[n]),trim(name)//' ')
    enddo
    enddo
    do b=1,3
      write(name,'(a,i0)') 'dtemperature_',b
      call compare(reshape(dtmp(:,:,:,b),[n]),reshape(expected_t(:,:,:,b),[n]),trim(name)//' ')
    enddo
    density=1.d0
    call check(device_state_upload(handle,device_rho,density,bytes(size(density))),'statistic density upload')
    call check(fluid_conservative(handle,c1),'statistic conservative field')
    call check(fluid_statistics(handle,1600.d0,110.3d0/273.15d0,1.d0+110.3d0/273.15d0,stats),'statistics')
    totals(1:3)=0.d0;totals(4:8)=-huge(1.d0);totals(9:13)=huge(1.d0)
    do k=0,km
    do j=0,jm
    do i=0,im
      u=vel(i,j,k,:)
      qnode=[1.d0,u,c1*temperature(i,j,k)+.5d0*sum(u*u)]
      totals(4:8)=max(totals(4:8),qnode)
      totals(9:13)=min(totals(9:13),qnode)
      if(i==0.or.j==0.or.k==0) cycle
      gradient=expected(i,j,k,:,:)
      omega=[gradient(3,2)-gradient(2,3),gradient(1,3)-gradient(3,1),gradient(2,1)-gradient(1,2)]
      strain=.5d0*(gradient+transpose(gradient))
      div=strain(1,1)+strain(2,2)+strain(3,3)
      heat=temperature(i,j,k)
      mu=heat*sqrt(heat)*(1.d0+110.3d0/273.15d0)/(heat+110.3d0/273.15d0)/1600.d0
      totals(1)=totals(1)+sum(u*u)
      totals(2)=totals(2)+sum(omega*omega)
      totals(3)=totals(3)+2.d0*mu*(sum(strain*strain)-div**2/3.d0)
    enddo
    enddo
    enddo
    do n=1,13
      write(name,'(a,i0)') 'statistics_',n
      call compare([stats(n)],[totals(n)],trim(name)//' ')
    enddo
    oracle=0.d0
    full_oracle=0.d0;plane_reference=0.d0
    do k=0,km
    do j=0,jm
    do i=0,im
      sound=sqrt(temperature(i,j,k))/mach
      u=vel(i,j,k,:);m=metric(i,j,k,:,:)
      call spectral_rates(u,sound,m,one,valid)
      if(.not.valid) call fail('original CPU spectral_rates rejected analytic field')
      oracle=max(oracle,one(1:3))
      full_oracle=max(full_oracle,one)
      plane_reference(:,j)=max(plane_reference(:,j),one)
    enddo
    enddo
    enddo
    planes=0.d0
    call check(fluid_cfl_diagnostics(handle,1_c_int,mach,gamma,287.d0, &
      int([im*nranks,jm,km],c_int),int([rank*im,0,0],c_int),jm+1,diagnostic,keys,planes),'CFL diagnostics')
    do n=1,4
      write(name,'(a,i0)') 'diagnostic_maximum_',n
      call compare([diagnostic(n)],[full_oracle(n)],trim(name)//' ')
      key=keys(n)
      ii=int(modulo(key,int(im*nranks+1,c_int64_t)))-rank*im
      jj=int(modulo(key/(im*nranks+1),int(jm+1,c_int64_t)))
      kk=int(key/(int(im*nranks+1,c_int64_t)*(jm+1)))
      if(ii<0.or.ii>im.or.jj<0.or.jj>jm.or.kk<0.or.kk>km) call fail('CFL maximum key outside owned grid')
      call spectral_rates(vel(ii,jj,kk,:),sqrt(temperature(ii,jj,kk))/mach,metric(ii,jj,kk,:,:),one,valid)
      write(name,'(a,i0)') 'maximum_at_key_',n
      call compare([diagnostic(n)],[one(n)],trim(name)//' ')
    enddo
    call compare(reshape(planes,[4*(jm+1)]),reshape(plane_reference,[4*(jm+1)]),'CFL y profile ')
    call check(fluid_cfl(handle,1_c_int,mach,gamma,287.d0,rates),'CFL completion')
    do a=1,3
      write(name,'(a,i0)') 'spectral_maximum_',a
      call compare([rates(a)],[oracle(a)],trim(name)//' ')
    enddo
    call compare([sum(rates)],[sum(oracle)],'C_upper before MPI ')
    call MPI_Allreduce(MPI_IN_PLACE,rates,3,MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,ierr)
    call MPI_Allreduce(MPI_IN_PLACE,oracle,3,MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,ierr)
    call compare([sum(rates)],[sum(oracle)],'C_upper after MPI ')
    vel=1.d0;temperature=1.d0;metric=0.d0
    do n=1,3
      metric(:,:,:,n,n)=1.d0
    enddo
    call check(device_state_upload(handle,device_velocity,vel,bytes(size(vel))),'constant velocity upload')
    call check(device_state_upload(handle,device_temperature,temperature,bytes(size(temperature))),'constant T upload')
    call check(device_state_upload(handle,device_metric,metric,bytes(size(metric))),'constant metric upload')
    planes=-1.d0
    call check(fluid_cfl_diagnostics(handle,1_c_int,mach,gamma,287.d0, &
      int([im*nranks,jm,km],c_int),int([rank*im,0,0],c_int),0_c_int,diagnostic,keys,planes),'tied CFL diagnostics')
    call compare(diagnostic,[11.d0,11.d0,11.d0,33.d0],'constant CFL ')
    if(any(keys/=int(rank*im,c_int64_t))) call fail('CFL tie did not select smallest global node key')
    if(any(planes/=-1.d0)) call fail('disabled CFL y profile was touched')
    write(*,'(a)') 'FLUID CFL tie keys exact; disabled profile unchanged'
    temperature(0,0,0)=-1.d0
    call check(device_state_upload(handle,device_temperature,temperature,bytes(size(temperature))),'invalid T upload')
    if(fluid_cfl(handle,1_c_int,mach,gamma,287.d0,rates)/=device_invalid_spectrum) &
      call fail('invalid acoustic spectrum was accepted')
    write(*,'(a)') 'FLUID invalid_acoustic_spectrum_rejected'
    call check(device_state_destroy(handle),'destroy analytic')
  end subroutine
end program
