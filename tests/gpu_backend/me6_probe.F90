program me6_probe_cpp
  use iso_c_binding, only: c_ptr,c_int,c_size_t,c_int64_t
  use astr_precision, only: real_kind,real_bytes,real_to_word
  use ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_quiet_nan
  use me6opti_operator, only: me6_mapped_direction,me6_coefficients,me6_scalar_direction
  use device_state
  use fluid_explicit, only: fluid_prepare,fluid_rhs_prepare,prepare=>fluid_me6_prepare,diffuse=>fluid_me6_diffusion, &
    fluid_me6_metrics
  implicit none
  integer,parameter :: im=7,jm=5,km=3,hm=5
  real(real_kind),parameter :: re=1600.e0_real_kind,pr=.72e0_real_kind,c5=.004e0_real_kind,tc=110.3e0_real_kind/273.15e0_real_kind,tc1=1+tc
  real(real_kind) :: v(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3),t(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
    mu(-hm:im+hm,-hm:jm+hm,-hm:km+hm),kappa(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
    metric(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3),jac(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
    reference(0:im,0:jm,0:km,5),actual(0:im,0:jm,0:km,5),gold(0:im,0:jm,0:km,5), &
    ci(9,6),cd(9,6),cp(9,6),mapping(3,3),inverse(3,3),spacing(3),wave(3),x(3),theta,part(4), &
    difference,maximum=0
  type(c_ptr) :: state
  integer :: i,j,k,a,b,h,side,p,column,half,sample,debug,status
  character(1024) :: argument
  call get_command_argument(1,argument)
  if(trim(argument)=='periodic') then
    call periodic_patch()
    stop
  endif
  if(len_trim(argument)>0.and.trim(argument)/='paper'.and.trim(argument)/='paper-cell') &
    error stop 'ME6 probe mode must be periodic, paper, paper-cell or empty'
  do h=1,3
  do side=1,2
    column=2*(h-1)+side;half=(2*side-3)*(2*h-1)
    do p=-4,4
      call me6_coefficients(half,p,ci(p+5,column),cd(p+5,column),cp(p+5,column))
    enddo
  enddo
  enddo
  if(trim(argument)=='paper'.or.trim(argument)=='paper-cell') then
    call paper_tests(trim(argument)=='paper-cell')
    stop
  endif
  do sample=1,4
    spacing=[.13e0_real_kind,.17e0_real_kind,.11e0_real_kind];wave=[1.e0_real_kind,2.e0_real_kind,-1.e0_real_kind]
    mapping=0;inverse=0
    do a=1,3
      mapping(a,a)=spacing(a);inverse(a,a)=1/spacing(a)
    enddo
    if(sample==3) then
      wave=[-2.e0_real_kind,1.e0_real_kind,3.e0_real_kind]
      mapping(1,2)=.2e0_real_kind*spacing(2);mapping(2,3)=.1e0_real_kind*spacing(3)
      inverse(1,2)=-.2e0_real_kind/spacing(1);inverse(2,3)=-.1e0_real_kind/spacing(2);inverse(1,3)=.02e0_real_kind/spacing(1)
    endif
    do k=-hm,km+hm
    do j=-hm,jm+hm
    do i=-hm,im+hm
      x=matmul(mapping,real([i,j,k],real_kind));theta=.37e0_real_kind+sum(wave*x)
      v(i,j,k,:)=[.2e0_real_kind*sin(theta),.13e0_real_kind*cos(theta),-.17e0_real_kind*sin(2*theta)]
      t(i,j,k)=1+.1e0_real_kind*cos(theta+.2e0_real_kind)
      if(sample==1) t(i,j,k)=1
      if(sample==4) then
        v(i,j,k,:)=[.2e0_real_kind,.13e0_real_kind,-.17e0_real_kind]*real((-1)**(i+j+k),real_kind)
        t(i,j,k)=1+.01e0_real_kind*real((-1)**(i+j+k),real_kind)
      endif
      mu(i,j,k)=t(i,j,k)*sqrt(t(i,j,k))*tc1/(t(i,j,k)+tc)
      mu(i,j,k)=mu(i,j,k)/re;kappa(i,j,k)=(mu(i,j,k)/pr)/c5
      metric(i,j,k,:,:)=inverse;jac(i,j,k)=product(spacing)
    enddo
    enddo
    enddo
    reference=0
    do k=0,km
    do j=0,jm
    do i=0,im
      do a=1,3
        call me6_mapped_direction(a,[i,j,k],hm,v,t,mu,kappa,metric,jac,part)
        reference(i,j,k,2:5)=reference(i,j,k,2:5)+part
      enddo
    enddo
    enddo
    enddo
    do debug=0,1
      call check(device_state_create(0_c_int,im,jm,km,hm,int(debug,c_int),state),'create')
      call check(fluid_prepare(state,0_c_int),'primitives')
      call check(fluid_rhs_prepare(state,0_c_int),'RHS')
      ! Exact prescribed patch halos isolate the mapped kernel from boundary/MPI algorithms.
      call check(prepare(state,0_c_int,[0_c_int,0_c_int,0_c_int],[0_c_int,0_c_int,0_c_int], &
        [1_c_int,1_c_int,1_c_int],[0_c_int,0_c_int,0_c_int],[-hm,-hm,-hm],[im+hm,jm+hm,km+hm], &
        [0_c_int,0_c_int,0_c_int],[im,jm,km],0_c_int,ci,cd,cp),'ME6 setup')
      call check(device_state_upload(state,device_velocity,v,bytes(size(v))),'velocity')
      call check(device_state_upload(state,device_temperature,t,bytes(size(t))),'temperature')
      call check(device_state_upload(state,device_metric,metric,bytes(size(metric))),'metric')
      call check(device_state_upload(state,device_jacobian,jac,bytes(size(jac))),'Jacobian')
      actual=0
      call check(device_state_upload(state,device_rhs,actual,bytes(size(actual))),'RHS origin')
      call check(diffuse(state,re,pr,c5,tc,tc1),'ME6 diffusion')
      call check(device_state_download(state,device_rhs,actual,bytes(size(actual))),'RHS observation')
      if(.not.all(ieee_is_finite(actual))) error stop 'ME6 nonfinite output'
      if(any(actual(:,:,:,1)/=0)) error stop 'ME6 changed density RHS'
      difference=0
      do a=1,5
        difference=max(difference,maxval(abs(actual(:,:,:,a)-reference(:,:,:,a)))/ &
          max(1.e0_real_kind,maxval(abs(reference(:,:,:,a)))))
      enddo
      if(real_bytes==8.and.difference>2.e-10_real_kind) error stop 'ME6 same-state mapped parity'
      maximum=max(maximum,difference)
      if(debug==0) then
        gold=actual
      elseif(any(real_to_word(actual)/=real_to_word(gold))) then
        error stop 'ME6 sync changed output words'
      endif
      call check(device_state_destroy(state),'destroy')
      write(*,'(a,2(1x,i0),1x,es24.16e3)') 'CPP_ME6',sample,debug,difference
    enddo
  enddo
  write(*,'(a,1x,es24.16e3)') 'CPP_ME6_PASS',maximum
  write(*,'(a,1x,i0)') 'CPP_ME6_ELEMENT_BYTES',real_bytes
contains
  subroutine paper_tests(cell_centered)
    logical,intent(in) :: cell_centered
    integer :: n,level,kind,intervals,previous_intervals
    real(real_kind) :: errors(2),prior(2),parity,order(2),ratio
    real(real_kind),parameter :: published(5,2)=reshape([ &
      1.51e-3_real_kind,2.56e-5_real_kind,4.07e-7_real_kind,6.38e-9_real_kind,9.30e-11_real_kind, &
      2.43e-2_real_kind,4.13e-4_real_kind,6.65e-6_real_kind,1.05e-7_real_kind,1.65e-9_real_kind],[5,2])
    if(real_bytes/=8) error stop 'ME6 paper probe uses FP64'
    write(*,'(a,l1)') 'ME6_PAPER cell-centered=',cell_centered
    do kind=1,2
      do level=1,5
        n=20*2**(level-1)
        intervals=merge(n,n-1,cell_centered)
        call paper_patch(n,kind==2,cell_centered,errors,parity)
        order=0.e0_real_kind
        if(level>1) then
          ratio=real(intervals,real_kind)/real(previous_intervals,real_kind)
          order=log(prior/errors)/log(ratio)
        endif
        write(*,'(a,2(1x,i0),6(1x,es24.16e3))') 'ME6_PAPER',kind,n,published(level,kind),errors,order,parity
        prior=errors
        previous_intervals=intervals
      enddo
    enddo
  end subroutine

  subroutine paper_patch(n,mixed,cell_centered,errors,parity)
    integer,intent(in) :: n
    logical,intent(in) :: mixed,cell_centered
    real(real_kind),intent(out) :: errors(2),parity
    integer :: ix,iy,jmax,kk,axis,dir
    integer(c_int) :: dims(3)
    type(c_ptr) :: context
    real(real_kind) :: s,delta,vol,offset
    real(real_kind),allocatable :: u(:,:,:,:),temperature(:,:,:),viscosity(:,:,:),g(:,:,:,:,:),jdet(:,:,:), &
      rhs(:,:,:,:),expected(:,:),on_cpu(:,:),on_device(:,:)
    delta=1.e0_real_kind/real(merge(n,n-1,cell_centered),real_kind)
    offset=0.e0_real_kind
    if(cell_centered) offset=merge(1.e0_real_kind,.5e0_real_kind,mixed)
    jmax=merge(n-1,0,mixed)
    dims=int([n-1,jmax,0],c_int)
    vol=delta**3
    allocate(u(-hm:n-1+hm,-hm:jmax+hm,-hm:hm,3),temperature(-hm:n-1+hm,-hm:jmax+hm,-hm:hm), &
      viscosity(-hm:n-1+hm,-hm:jmax+hm,-hm:hm),source=0.e0_real_kind)
    allocate(g(-hm:n-1+hm,-hm:jmax+hm,-hm:hm,3,3),jdet(-hm:n-1+hm,-hm:jmax+hm,-hm:hm),source=0.e0_real_kind)
    allocate(rhs(0:n-1,0:jmax,0:0,5),expected(0:n-1,0:jmax),on_cpu(0:n-1,0:jmax),on_device(0:n-1,0:jmax))
    do kk=-hm,hm
    do iy=-hm,jmax+hm
    do ix=-hm,n-1+hm
      s=(real(ix+merge(iy,0,mixed),real_kind)+offset)*delta
      u(ix,iy,kk,2)=sin(10*s)
      viscosity(ix,iy,kk)=.1e0_real_kind*exp(2*s)
    enddo
    enddo
    enddo
    temperature=viscosity**2
    jdet=vol
    do axis=1,3
      g(:,:,:,axis,axis)=1.e0_real_kind/delta
    enddo
    dir=merge(2,1,mixed)
    do iy=0,jmax
    do ix=0,n-1
      s=(real(ix+merge(iy,0,mixed),real_kind)+offset)*delta
      expected(ix,iy)=viscosity(ix,iy,0)*(20*cos(10*s)-100*sin(10*s))
      on_cpu(ix,iy)=me6_scalar_direction(1,dir,[ix,iy,0],hm,u(:,:,:,2),viscosity,[delta,delta,delta])
    enddo
    enddo
    call check(device_state_create(0_c_int,dims(1),dims(2),dims(3),hm,1_c_int,context),'paper create')
    call check(fluid_prepare(context,0_c_int),'paper primitives')
    call check(fluid_rhs_prepare(context,0_c_int),'paper RHS')
    call check(prepare(context,0_c_int,[0_c_int,0_c_int,0_c_int],[0_c_int,0_c_int,0_c_int], &
      [1_c_int,1_c_int,1_c_int],[0_c_int,0_c_int,0_c_int],-hm+[0_c_int,0_c_int,0_c_int],dims+hm, &
      [0_c_int,0_c_int,0_c_int],dims,0_c_int,ci,cd,cp),'paper setup')
    call check(device_state_upload(context,device_velocity,u,bytes(size(u))),'paper velocity')
    call check(device_state_upload(context,device_temperature,temperature,bytes(size(temperature))),'paper temperature')
    call check(device_state_upload(context,device_metric,g,bytes(size(g))),'paper metric')
    call check(device_state_upload(context,device_jacobian,jdet,bytes(size(jdet))),'paper Jacobian')
    rhs=0
    call check(device_state_upload(context,device_rhs,rhs,bytes(size(rhs))),'paper RHS origin')
    ! With tc=0 and Re=tc1=1 the device constitutive law gives mu=sqrt(T).
    call check(diffuse(context,1.e0_real_kind,1.e0_real_kind,1.e0_real_kind,0.e0_real_kind,1.e0_real_kind),'paper diffusion')
    call check(device_state_download(context,device_rhs,rhs,bytes(size(rhs))),'paper RHS observation')
    call check(device_state_destroy(context),'paper destroy')
    if(mixed) then
      ! Here Dx(mu*Dy u2) = Dy(mu*Dx u2) by x/y symmetry. The x momentum
      ! component is their sum weighted by -2/3 and 1, respectively.
      on_device=3*rhs(:,:,0,2)/vol
    else
      on_device=rhs(:,:,0,3)/vol
    endif
    parity=maxval(abs(on_cpu-on_device))/max(1.e0_real_kind,maxval(abs(on_cpu)))
    if(.not.all(ieee_is_finite(on_device)).or.parity>2.e-10_real_kind) error stop 'ME6 paper CPU/GPU scalar parity'
    errors=[sum(abs(on_cpu-expected)),sum(abs(on_device-expected))]/real(size(expected),real_kind)
  end subroutine
  subroutine periodic_patch()
    use mpi
    use astr_precision_mpi, only: mpi_real_type
    integer,parameter :: global_n=16
    type(c_ptr) :: handle
    type(device_view) :: view
    integer :: rank,ranks,local_rank,ierr,axis,cart,dims(3),coords(3),u(3),origin(3),lo(3),hi(3), &
      ii,jj,kk,d,component,g(3),case_debug,unit,header(11),outside
    integer(c_int) :: mode
    integer(c_int64_t) :: metric_values(4),wire(5)
    real(real_kind),parameter :: seed(5)=[7._real_kind,.25_real_kind,-.5_real_kind,.125_real_kind,.0625_real_kind]
    real(real_kind),allocatable :: ve(:,:,:,:),te(:,:,:),vw(:,:,:,:),tw(:,:,:),mue(:,:,:),ke(:,:,:), &
      met(:,:,:,:,:),vol(:,:,:),ref(:,:,:,:),out(:,:,:,:),saved(:,:,:,:),vg(:,:,:,:),tg(:,:,:)
    real(real_kind) :: table_i(9,6),table_d(9,6),table_p(9,6),inverse_map(3,3),dx,phi,nan,contribution(4),error
    character(1024) :: directory,path,arg
    interface
      subroutine wire_counts(values) bind(C,name='astr_halo_wire_counts')
        import c_int64_t
        integer(c_int64_t),intent(out) :: values(5)
      end subroutine
    end interface
    call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',arg,status=ierr)
    if(ierr/=0) error stop 'ME6 periodic probe requires Open MPI local rank'
    read(arg,*) local_rank
    call check(device_bind(int(local_rank,c_int)),'pre-MPI binding')
    call MPI_Init(ierr);call check(ierr,'MPI initialization')
    call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call check(ierr,'MPI rank')
    call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr);call check(ierr,'MPI size')
    call get_command_argument(2,arg);read(arg,*) axis
    if(axis<1.or.axis>3.or.mod(global_n,ranks)/=0) error stop 'ME6 periodic fixture decomposition'
    call get_command_argument(3,arg)
    select case(trim(arg))
    case('pinned');mode=0_c_int
    case('device-aware');mode=1_c_int
    case default;error stop 'ME6 periodic transport'
    end select
    call get_command_argument(4,directory)
    if(len_trim(directory)==0) error stop 'ME6 periodic output directory required'
    dims=1;dims(axis)=ranks;u=global_n/dims
    if(any(u<hm)) error stop 'ME6 periodic fixture needs a full halo within each local interval range'
    call MPI_Cart_create(MPI_COMM_WORLD,3,dims,[.true.,.true.,.true.],.false.,cart,ierr)
    call check(ierr,'Cartesian communicator')
    call MPI_Cart_coords(cart,rank,3,coords,ierr);call check(ierr,'Cartesian coordinates')
    origin=coords*u
    do d=1,3
      call MPI_Cart_shift(cart,d-1,1,lo(d),hi(d),ierr);call check(ierr,'Cartesian peers')
    enddo
    allocate(ve(-hm:u(1)+hm,-hm:u(2)+hm,-hm:u(3)+hm,3),te(-hm:u(1)+hm,-hm:u(2)+hm,-hm:u(3)+hm), &
      met(-hm:u(1)+hm,-hm:u(2)+hm,-hm:u(3)+hm,3,3),vol(-hm:u(1)+hm,-hm:u(2)+hm,-hm:u(3)+hm), &
      ref(0:u(1),0:u(2),0:u(3),5),out(0:u(1),0:u(2),0:u(3),5),saved(0:u(1),0:u(2),0:u(3),5))
    allocate(vw,mold=ve);allocate(tw,mold=te);allocate(mue,mold=te);allocate(ke,mold=te)
    allocate(vg(0:global_n-1,0:global_n-1,0:global_n-1,3),tg(0:global_n-1,0:global_n-1,0:global_n-1))
    dx=2*acos(-1._real_kind)/global_n
    if(rank==0) then
      do kk=0,global_n-1
      do jj=0,global_n-1
      do ii=0,global_n-1
        phi=.37_real_kind+dx*real(ii+2*jj-kk,real_kind)
        vg(ii,jj,kk,:)=[.2_real_kind*sin(phi),.13_real_kind*cos(phi),-.17_real_kind*sin(2*phi)]
        tg(ii,jj,kk)=1+.1_real_kind*cos(phi+.2_real_kind)
      enddo
      enddo
      enddo
    endif
    call MPI_Bcast(vg,size(vg),mpi_real_type,0,cart,ierr);call check(ierr,'canonical velocity fixture')
    call MPI_Bcast(tg,size(tg),mpi_real_type,0,cart,ierr);call check(ierr,'canonical temperature fixture')
    inverse_map=0
    do d=1,3
      inverse_map(d,d)=1/dx
    enddo
    inverse_map(1,2)=-.2_real_kind/dx;inverse_map(2,3)=-.1_real_kind/dx
    inverse_map(1,3)=.02_real_kind/dx
    do kk=-hm,u(3)+hm
    do jj=-hm,u(2)+hm
    do ii=-hm,u(1)+hm
      g=modulo(origin+[ii,jj,kk],global_n)
      ve(ii,jj,kk,:)=vg(g(1),g(2),g(3),:)
      te(ii,jj,kk)=tg(g(1),g(2),g(3))
      mue(ii,jj,kk)=te(ii,jj,kk)*sqrt(te(ii,jj,kk))*tc1/(te(ii,jj,kk)+tc)
      mue(ii,jj,kk)=mue(ii,jj,kk)/re;ke(ii,jj,kk)=(mue(ii,jj,kk)/pr)/c5
      met(ii,jj,kk,:,:)=inverse_map;vol(ii,jj,kk)=dx**3
    enddo
    enddo
    enddo
    do component=1,5
      ref(:,:,:,component)=seed(component)
    enddo
    do kk=0,u(3)
    do jj=0,u(2)
    do ii=0,u(1)
      do d=1,3
        call me6_mapped_direction(d,[ii,jj,kk],hm,ve,te,mue,ke,met,vol,contribution)
        ref(ii,jj,kk,2:5)=ref(ii,jj,kk,2:5)+contribution
      enddo
    enddo
    enddo
    enddo
    do h=1,3
    do side=1,2
      column=2*(h-1)+side;half=(2*side-3)*(2*h-1)
      do p=-4,4
        call me6_coefficients(half,p,table_i(p+5,column),table_d(p+5,column),table_p(p+5,column))
      enddo
    enddo
    enddo
    nan=ieee_value(0._real_kind,ieee_quiet_nan)
    do case_debug=0,1
      vw=ve;tw=te
      ! Face values represent an already completed public halo exchange.
      do kk=-hm,u(3)+hm
      do jj=-hm,u(2)+hm
      do ii=-hm,u(1)+hm
        outside=count([ii,jj,kk]<0.or.[ii,jj,kk]>u)
        if(outside>1) then
          vw(ii,jj,kk,:)=nan;tw(ii,jj,kk)=nan
        endif
      enddo
      enddo
      enddo
      call check(device_state_create(int(local_rank,c_int),u(1),u(2),u(3),hm,int(case_debug,c_int),handle),'periodic create')
      call check(fluid_prepare(handle,0_c_int),'periodic primitives')
      call check(fluid_rhs_prepare(handle,0_c_int),'periodic RHS')
      call check(device_state_view(handle,device_velocity,view),'periodic view')
      if(view%element_bytes/=real_bytes) error stop 'ME6 periodic native element width'
      call check(prepare(handle,int(cart,c_int),int(lo,c_int),int(hi,c_int),int(dims,c_int), &
        [1_c_int,1_c_int,1_c_int],[-hm,-hm,-hm],u+hm,[0_c_int,0_c_int,0_c_int],u,mode, &
        table_i,table_d,table_p),'periodic ME6 setup')
      call check(device_state_upload(handle,device_velocity,vw,bytes(size(vw))),'periodic velocity')
      call check(device_state_upload(handle,device_temperature,tw,bytes(size(tw))),'periodic temperature')
      call check(device_state_upload(handle,device_metric,met,bytes(size(met))),'periodic metric')
      call check(device_state_upload(handle,device_jacobian,vol,bytes(size(vol))),'periodic Jacobian')
      do component=1,5
        out(:,:,:,component)=seed(component)
      enddo
      call check(device_state_upload(handle,device_rhs,out,bytes(size(out))),'periodic initial RHS')
      call check(diffuse(handle,re,pr,c5,tc,tc1),'periodic ME6 diffusion')
      call check(device_state_download(handle,device_velocity,vw,bytes(size(vw))),'periodic velocity halo')
      call check(device_state_download(handle,device_temperature,tw,bytes(size(tw))),'periodic temperature halo')
      if(any(real_to_word(vw)/=real_to_word(ve)).or.any(real_to_word(tw)/=real_to_word(te))) &
        error stop 'ME6 private edges differ from exact periodic source halo'
      call check(device_state_download(handle,device_rhs,out,bytes(size(out))),'periodic ME6 observation')
      if(.not.all(ieee_is_finite(out)).or.any(real_to_word(out(:,:,:,1))/=real_to_word(seed(1)))) &
        error stop 'ME6 periodic finiteness/density invariant'
      error=0
      do component=1,5
        error=max(error,maxval(abs(out(:,:,:,component)-ref(:,:,:,component)))/ &
          max(1._real_kind,maxval(abs(ref(:,:,:,component)))))
      enddo
      if(real_bytes==8.and.error>2.e-10_real_kind) error stop 'ME6 periodic mapped reference'
      if(case_debug==0) then
        saved=out
      elseif(any(real_to_word(out)/=real_to_word(saved))) then
        error stop 'ME6 periodic synchronization words'
      endif
      call check(fluid_me6_metrics(handle,metric_values),'periodic ME6 metrics')
      write(*,'(a,3(1x,i0),1x,es24.16e3,4(1x,i0))') 'CPP_ME6_PERIODIC',rank,axis,case_debug,error,metric_values
      write(path,'(a,a,i8.8,a,i0,a)') trim(directory),'/rank',rank,'.debug',case_debug,'.bin'
      open(newunit=unit,file=trim(path),status='new',access='stream',form='unformatted')
      header=[real_bytes,rank,axis,case_debug,u,origin,global_n]
      write(unit) header,out
      close(unit)
      call check(device_state_destroy(handle),'periodic destroy')
    enddo
    call wire_counts(wire)
    if(wire(5)/=0) error stop 'ME6 private MPI packet type'
    write(*,'(a,2(1x,i0),5(1x,i0))') 'CPP_ME6_WIRE',rank,real_bytes,wire
    write(*,'(a,1x,i0)') 'CPP_ME6_PERIODIC_PASS',rank
    call MPI_Comm_free(cart,ierr);call check(ierr,'Cartesian release')
    call MPI_Finalize(ierr);call check(ierr,'MPI finalization')
  end subroutine

  integer(c_size_t) function bytes(count)
    integer,intent(in) :: count
    bytes=int(count,c_size_t)*real_bytes
  end function
  subroutine check(code,label)
    integer,intent(in) :: code
    character(*),intent(in) :: label
    if(code==0) return
    write(*,'(a,1x,a,1x,i0)') 'CPP_ME6_FAILURE',label,code
    error stop 1
  end subroutine
end program
