program nscbc_outlet_probe
  use iso_c_binding
  use astr_precision, only: real_kind,real_bytes,real_to_word
  use mpi
#if defined(REFERENCE_CPU) || defined(REFERENCE_CUDA)
  use commvar
  use commarray, only: q,rho,vel,prs,tmp,spc,jacob,dxi,qrhs
  use bc, only: bctype,pout,outflow_nscbc
  use parallel, only: mpirank,mpisize,lio,irk,irkm,jrk,jrkm,krk,krkm,isize,jsize,ksize
#else
  use device_state
  use fluid_explicit, only: fluid_prepare,fluid_rhs_prepare
#ifdef TEST_INLET
  use boundary_state_cpp, only: nscbc_inlet_x
#else
  use boundary_state_cpp, only: nscbc_outlet_x
#endif
#endif
#ifdef REFERENCE_CUDA
  use cudafor
  use commarray_gpu, only: q_d,rho_d,vel_d,prs_d,tmp_d,qrhs_d,jacob_d,dxi_d
  use boundary_gpu, only: nscbc_outflow_x_rhs_kernel,nscbc_inflow_x_rhs_kernel
#endif
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  integer,parameter :: global_upper(3)=[16,8,8],halo=5
#ifdef TEST_INLET
  integer,parameter :: trials=24,reference_count=1
#ifdef REFERENCE_CPU
#error The unrepaired generic CPU bc12 is not an inlet reference
#endif
#else
  integer,parameter :: trials=20,reference_count=2
#endif
  integer :: rank,ranks,axis=1,upper(3),origin(3),unit,reference_units(2),ierr,local_rank, &
    i,j,k,c,trial,debug,reference,case_count=0,header(7),expected_header(7),n
  logical :: owner
  real(real_kind),parameter :: gm=1.4e0_real_kind,ma=1.e0_real_kind,length=16.e0_real_kind
  real(real_kind) :: ux,uy,uz,pfar,poutlet,error,maximum=0,offset_x
  real(real_kind),allocatable :: sq(:,:,:,:),srho(:,:,:),sv(:,:,:,:),sp(:,:,:),st(:,:,:), &
    sj(:,:,:),sm(:,:,:,:,:),seed(:,:,:,:),actual(:,:,:,:),expected(:,:,:,:),observed(:,:,:,:),sync_rhs(:,:,:,:,:)
  character(1024) :: directory,path
  character(80) :: text,suffix
#ifdef REFERENCE_CUDA
  type(dim3) :: block,grid
#endif
#if !defined(REFERENCE_CPU) && !defined(REFERENCE_CUDA)
  type(c_ptr) :: state
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text,status=ierr)
  if(ierr/=0) error stop 'outlet probe requires Open MPI local rank'
  read(text,*) local_rank
  call check(device_bind(int(local_rank,c_int)))
#endif
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  call get_command_argument(1,directory);call get_command_argument(2,text);read(text,*) axis
  if((ranks/=1.and.ranks/=2).or.axis<1.or.axis>3) error stop 'outlet probe topology'
  upper=global_upper;origin=0
  if(ranks==2) then
    upper(axis)=global_upper(axis)/2;origin(axis)=rank*upper(axis)
  endif
#ifdef TEST_INLET
  owner=origin(1)==0
#else
  owner=origin(1)+upper(1)==global_upper(1)
#endif
  allocate(sq(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,5), &
    srho(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo), &
    sv(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,3), &
    sp(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo), &
    st(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo), &
    sj(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo), &
    sm(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,3,3), &
    seed(0:upper(1),0:upper(2),0:upper(3),5),actual(0:upper(1),0:upper(2),0:upper(3),5), &
    expected(0:upper(1),0:upper(2),0:upper(3),5), &
    observed(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,5))
  allocate(sync_rhs(0:upper(1),0:upper(2),0:upper(3),5,trials))
  write(suffix,'(a,i8.8,a)') '.rank',rank,'.bin'
  expected_header=[upper,origin,trials]
#if defined(REFERENCE_CPU) || defined(REFERENCE_CUDA)
  im=upper(1);jm=upper(2);km=upper(3);ia=global_upper(1);ja=global_upper(2);ka=global_upper(3)
  mpirank=rank;mpisize=ranks;lio=rank==0
  irk=0;jrk=0;krk=0;irkm=0;jrkm=0;krkm=0;isize=1;jsize=1;ksize=1
  if(axis==1) then
    irk=rank;irkm=ranks-1;isize=ranks
  elseif(axis==2) then
    jrk=rank;jrkm=ranks-1;jsize=ranks
  else
    krk=rank;krkm=ranks-1;ksize=ranks
  endif
  is=0;ie=im;js=0;je=jm;ks=0;ke=km
  ndims=3;numq=5;num_species=0;num_modequ=0;nondimen=.true.;lcomb=.false.;lfilter=.false.
  flowtype='tgv';conschm='643e';difschm='643e';turbmode='none'
  lihomo=.false.;ljhomo=.true.;lkhomo=.true.;npdci=3;npdcj=3;npdck=3
  bctype=[11,22,1,1,1,1];gamma=gm;mach=ma;const2=gm;const6=1.e0_real_kind/(gm-1.e0_real_kind)
  xmin=0;xmax=length;roinf=1;uinf=.4e0_real_kind;vinf=.03e0_real_kind;winf=.02e0_real_kind
  allocate(q,mold=sq);allocate(rho,mold=srho);allocate(vel,mold=sv);allocate(prs,mold=sp)
  allocate(tmp,mold=st);allocate(jacob,mold=sj);allocate(dxi,mold=sm);allocate(qrhs,mold=seed)
  allocate(spc(-halo:im+halo,-halo:jm+halo,-halo:km+halo,0))
#ifdef REFERENCE_CUDA
  ierr=cudaSetDevice(mod(rank,2));if(ierr/=cudaSuccess) error stop 'reference GPU bind'
  allocate(q_d,mold=sq);allocate(rho_d,mold=srho);allocate(vel_d,mold=sv);allocate(prs_d,mold=sp)
  allocate(tmp_d,mold=st);allocate(jacob_d,mold=sj);allocate(dxi_d,mold=sm);allocate(qrhs_d,mold=seed)
  path=trim(directory)//'/gpu'//trim(suffix)
#else
  path=trim(directory)//'/cpu'//trim(suffix)
#endif
  open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='new')
  write(unit) expected_header
  n=1
#else
  do reference=1,reference_count
#ifdef TEST_INLET
    path=trim(directory)//'/gpu'//trim(suffix)
#else
    path=trim(directory)//'/cpu'//trim(suffix)
    if(reference==2) path=trim(directory)//'/gpu'//trim(suffix)
#endif
    open(newunit=reference_units(reference),file=trim(path),access='stream',form='unformatted',status='old')
    read(reference_units(reference)) header
    if(any(header/=expected_header)) error stop 'outlet reference layout'
  enddo
  n=2
#endif
  do debug=0,n-1
#if !defined(REFERENCE_CPU) && !defined(REFERENCE_CUDA)
    call check(device_state_create(int(local_rank,c_int),upper(1),upper(2),upper(3),halo,int(debug,c_int),state))
    call check(fluid_prepare(state,0_c_int));call check(fluid_rhs_prepare(state,0_c_int))
    do reference=1,reference_count
      rewind(reference_units(reference));read(reference_units(reference)) header
    enddo
#endif
    do trial=1,trials
      call initialize(trial)
#if defined(REFERENCE_CPU) || defined(REFERENCE_CUDA)
      q=sq;rho=srho;vel=sv;prs=sp;tmp=st;jacob=sj;dxi=sm
      pinf=pfar;pout=poutlet
#ifdef REFERENCE_CUDA
      q_d=sq;rho_d=srho;vel_d=sv;prs_d=sp;tmp_d=st;jacob_d=sj;dxi_d=sm;qrhs_d=seed
      block=dim3(256,1,1);grid=dim3(((jm+1)*(km+1)+255)/256,1,1)
#ifdef TEST_INLET
      call nscbc_inflow_x_rhs_kernel<<<grid,block>>>(im,jm,km,halo,3,3,gm,ma,0.e0_real_kind,length, &
        1.e0_real_kind,.4e0_real_kind,.03e0_real_kind,.02e0_real_kind,pfar,100.e0_real_kind,merge(1,0,owner))
#else
      call nscbc_outflow_x_rhs_kernel<<<grid,block>>>(im,jm,km,halo,3,3,gm,ma,0.e0_real_kind,length, &
        1.e0_real_kind,.4e0_real_kind,.03e0_real_kind,.02e0_real_kind,pfar,poutlet,100.e0_real_kind,merge(1,0,owner),1)
#endif
      ierr=cudaGetLastError();if(ierr/=cudaSuccess) error stop 'reference launch'
      ierr=cudaDeviceSynchronize();if(ierr/=cudaSuccess) error stop 'reference execution'
      actual=qrhs_d
#else
      qrhs=-seed
      call outflow_nscbc(2,rhs_only=.true.)
      actual=-qrhs
#endif
      write(unit) actual
#else
      call check(device_state_upload(state,device_q,sq,bytes(size(sq))))
      call check(device_state_upload(state,device_rho,srho,bytes(size(srho))))
      call check(device_state_upload(state,device_velocity,sv,bytes(size(sv))))
      call check(device_state_upload(state,device_pressure,sp,bytes(size(sp))))
      call check(device_state_upload(state,device_temperature,st,bytes(size(st))))
      call check(device_state_upload(state,device_jacobian,sj,bytes(size(sj))))
      call check(device_state_upload(state,device_metric,sm,bytes(size(sm))))
      call check(device_state_upload(state,device_rhs,seed,bytes(size(seed))))
#ifdef TEST_INLET
      if(owner) call check(nscbc_inlet_x(state,3_c_int,3_c_int,gm,ma,length, &
        1.e0_real_kind,.4e0_real_kind,.03e0_real_kind,.02e0_real_kind,pfar))
#else
      if(owner) call check(nscbc_outlet_x(state,3_c_int,3_c_int,1_c_int,gm,ma,length, &
        1.e0_real_kind,.4e0_real_kind,.03e0_real_kind,.02e0_real_kind,pfar,poutlet))
#endif
      call check(device_state_download(state,device_rhs,actual,bytes(size(actual))))
      do reference=1,reference_count
        read(reference_units(reference)) expected
        do c=1,5
          error=maxval(abs(actual(:,:,:,c)-expected(:,:,:,c)))/max(1.e0_real_kind,maxval(abs(expected(:,:,:,c))))
          if(real_bytes==8.and..not.(error<=2.e-10_real_kind)) then
            print *, 'OUTLET_MISMATCH ',rank,trial,reference,c,error
            error stop 'outlet differs from frozen reference'
          endif
          maximum=max(maximum,error)
          write(*,'(a,5(1x,i0),1x,es24.16e3)') 'CPP_NSCBC_NATIVE_DIFFERENCE', &
            rank,debug,trial,reference,c,error
        enddo
      enddo
      call check(device_state_download(state,device_q,observed,bytes(size(observed))))
      if(any(real_to_word(observed)/=real_to_word(sq))) &
        error stop 'outlet RHS changed q'
      if(.not.owner) then
        if(any(real_to_word(actual)/=real_to_word(seed))) &
          error stop 'outlet nonowner changed'
#ifdef TEST_INLET
      elseif(any(actual(1:upper(1),:,:,:)/=seed(1:upper(1),:,:,:))) then
#else
      elseif(any(actual(0:upper(1)-1,:,:,:)/=seed(0:upper(1)-1,:,:,:))) then
#endif
        error stop 'outlet changed interior RHS'
      endif
      if(debug==0) then
        sync_rhs(:,:,:,:,trial)=actual
      elseif(any(real_to_word(actual)/=real_to_word(sync_rhs(:,:,:,:,trial)))) then
        error stop 'NSCBC synchronization words differ'
      endif
#ifdef TEST_INLET
      if(owner.and.(trial==5.or.trial==17)) then
        expected=actual
        call check(device_state_upload(state,device_rhs,seed,bytes(size(seed))))
        call check(nscbc_inlet_x(state,3_c_int,3_c_int,gm,ma,length, &
          2._real_kind,-.4_real_kind,.09_real_kind,.12_real_kind,pfar*.5_real_kind))
#else
      if(owner.and.(trial==2.or.trial==12)) then
        expected=actual
        call check(device_state_upload(state,device_rhs,seed,bytes(size(seed))))
        call check(nscbc_outlet_x(state,3_c_int,3_c_int,1_c_int,gm,ma,length, &
          2._real_kind,-.4_real_kind,.09_real_kind,.12_real_kind,pfar*.5_real_kind,poutlet*.8_real_kind))
#endif
        call check(device_state_download(state,device_rhs,actual,bytes(size(actual))))
        if(any(real_to_word(actual)/=real_to_word(expected))) error stop 'outgoing NSCBC RHS depends on external target'
        call check(device_state_download(state,device_q,observed,bytes(size(observed))))
        if(any(real_to_word(observed)/=real_to_word(sq))) error stop 'NSCBC target test changed q'
        print *, 'CPP_NSCBC_OUTGOING_TARGET_INDEPENDENT ',rank,debug,trial
      endif
      case_count=case_count+1
#endif
      if(.not.all(ieee_is_finite(actual))) error stop 'nonfinite outlet RHS'
    enddo
#if !defined(REFERENCE_CPU) && !defined(REFERENCE_CUDA)
    call check(device_state_destroy(state))
#endif
  enddo
#if defined(REFERENCE_CPU) || defined(REFERENCE_CUDA)
  close(unit)
  print *, 'NSCBC_OUTLET_FROZEN ',rank,trials
#else
  do reference=1,reference_count
    close(reference_units(reference))
  enddo
#ifdef TEST_INLET
  print *, 'CPP_NSCBC_INLET_PASS ',rank,case_count,maximum
#else
  print *, 'CPP_NSCBC_OUTLET_PASS ',rank,case_count,maximum
#endif
#endif
  print *, 'NSCBC_PROBE_REAL_BYTES ',rank,real_bytes
  call MPI_Finalize(ierr)
contains
  subroutine initialize(number)
    integer,intent(in) :: number
#ifdef TEST_INLET
    real(real_kind) :: speeds(12)=[.2e0_real_kind,2.e0_real_kind,0.e0_real_kind,-.2e0_real_kind,-2.e0_real_kind,1.e0_real_kind, &
      1.e0_real_kind-1.e-9_real_kind,1.e0_real_kind+1.e-9_real_kind,-1.e0_real_kind,-1.e0_real_kind-1.e-9_real_kind,-1.e0_real_kind+1.e-9_real_kind,.2e0_real_kind]
#else
    real(real_kind) :: speeds(10)=[.2e0_real_kind,2.e0_real_kind,0.e0_real_kind,-.2e0_real_kind,-2.e0_real_kind,-1.e0_real_kind, &
      -1.e0_real_kind-1.e-9_real_kind,-1.e0_real_kind+1.e-9_real_kind,-1.e-9_real_kind,.2e0_real_kind]
#endif
    if(real_bytes==4) then
#ifdef TEST_INLET
      speeds(7)=1-128*epsilon(1._real_kind);speeds(8)=1+128*epsilon(1._real_kind)
      speeds(10)=-1-128*epsilon(1._real_kind);speeds(11)=-1+128*epsilon(1._real_kind)
#else
      speeds(7)=-1-128*epsilon(1._real_kind);speeds(8)=-1+128*epsilon(1._real_kind)
      speeds(9)=-128*epsilon(1._real_kind)
#endif
    endif
    ux=speeds(mod(number-1,size(speeds))+1);uy=.1e0_real_kind;uz=-.05e0_real_kind
    if(mod(number,size(speeds))==0) uy=10.e0_real_kind
    write(*,'(a,2(1x,i0),3(1x,es24.16e3))') 'NSCBC_PROBE_VELOCITY',rank,number,ux,uy,uz
    pfar=1.e0_real_kind/gm-.01e0_real_kind;poutlet=1
    if(mod(number,10)==4) poutlet=.8e0_real_kind
    sm=0;sj=1
    do c=1,3
      sm(:,:,:,c,c)=1
    enddo
    do k=-halo,upper(3)+halo
    do j=-halo,upper(2)+halo
    do i=-halo,upper(1)+halo
      offset_x=real(i+origin(1),real_kind)
      srho(i,j,k)=1;sp(i,j,k)=1.e0_real_kind/gm;sv(i,j,k,:)=[ux,uy,uz]
      if(number>size(speeds)) then
        srho(i,j,k)=1+.001e0_real_kind*offset_x+.0002e0_real_kind*(j+origin(2))+.0003e0_real_kind*(k+origin(3))
        sp(i,j,k)=sp(i,j,k)+.0007e0_real_kind*offset_x+.0001e0_real_kind*(j+origin(2))**2+.0002e0_real_kind*(k+origin(3))
        sv(i,j,k,2)=uy+.0001e0_real_kind*offset_x
        sm(i,j,k,1,1)=1+.001e0_real_kind*offset_x;sm(i,j,k,1,2)=.03e0_real_kind;sm(i,j,k,1,3)=-.02e0_real_kind
        sm(i,j,k,2,1)=.02e0_real_kind;sm(i,j,k,3,2)=.01e0_real_kind;sj(i,j,k)=1/sm(i,j,k,1,1)
      endif
      st(i,j,k)=gm*sp(i,j,k)/srho(i,j,k)
      sq(i,j,k,1)=srho(i,j,k);sq(i,j,k,2:4)=srho(i,j,k)*sv(i,j,k,:)
      sq(i,j,k,5)=sp(i,j,k)/(gm-1)+.5e0_real_kind*srho(i,j,k)*sum(sv(i,j,k,:)**2)
    enddo
    enddo
    enddo
    do c=1,5
      seed(:,:,:,c)=.125e0_real_kind+.001e0_real_kind*c
    enddo
  end subroutine
#if !defined(REFERENCE_CPU) && !defined(REFERENCE_CUDA)
  subroutine check(code)
    integer(c_int),intent(in) :: code
    if(code/=0) then
      print *, 'OUTLET_DEVICE_STATUS ',code
      error stop 'outlet device command'
    endif
  end subroutine
  integer(c_size_t) function bytes(count)
    integer,intent(in) :: count
    bytes=int(count,c_size_t)*c_sizeof(0.e0_real_kind)
  end function
#endif
end program
