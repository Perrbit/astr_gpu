program nscbc_top_probe
  use iso_c_binding
  use astr_precision, only: real_kind,real_bytes,real_to_word
  use astr_precision_mpi, only: mpi_real_type
  use ieee_arithmetic, only: ieee_is_finite
  use mpi
#if defined(REFERENCE_CPU) || defined(REFERENCE_CUDA)
  use commvar
  use commarray, only: q,rho,vel,prs,tmp,spc,jacob,dxi,x,qrhs
  use bc, only: bctype,farfield_nscbc,nscbc_farfield_configured,nscbc_farfield_policy, &
    nscbc_farfield_incoming_only,nscbc_farfield_sbli_shock,nscbc_farfield_rho_target, &
    nscbc_farfield_u_target,nscbc_farfield_v_target,nscbc_farfield_w_target,nscbc_farfield_t_target, &
    nscbc_farfield_shock_x,nscbc_farfield_shock_angle,nscbc_farfield_target_states
  use parallel, only: mpirank,mpisize,lio,irk,irkm,jrk,jrkm,krk,krkm,isize,jsize,ksize
#ifdef TEST_TOP_VISCOUS
  use bc, only: capture_nscbc_farfield_y_upper_prediff_rhs,apply_nscbc_farfield_y_upper_viscous_source,turbinf
  use sponge_layer, only: spg_i0,spg_im,spg_j0,spg_jm,spg_k0,spg_km
#endif
#else
  use device_state
  use fluid_explicit, only: fluid_prepare,fluid_rhs_prepare
  use boundary_state_cpp, only: nscbc_top_rhs,nscbc_top_target_upload,nscbc_top_mach2
#ifdef TEST_TOP_VISCOUS
  use boundary_state_cpp, only: nscbc_top_prediff_capture,nscbc_top_viscous_source
#endif
#endif
#ifdef REFERENCE_CUDA
  use cudafor
  use commarray_gpu, only: q_d,rho_d,vel_d,prs_d,tmp_d,qrhs_d,jacob_d,dxi_d,x_d
  use boundary_gpu, only: nscbc_farfield_y_upper_rhs_kernel,nscbc_farfield_y_upper_nonreflecting_rhs_kernel
#ifdef TEST_TOP_VISCOUS
  use boundary_gpu, only: nscbc_farfield_y_upper_prediff_capture_kernel,nscbc_farfield_y_upper_viscous_source_kernel
#endif
#endif
  implicit none
  integer,parameter :: halo=5,trials=12
  integer :: global_upper(3)=[16,16,16]
  real(real_kind),parameter :: gm=1.4e0_real_kind,ma=3.e0_real_kind,length=16.e0_real_kind,c2=gm*ma*ma,c6=1.e0_real_kind/(gm-1)
  integer :: rank,ranks,axis=1,upper(3),origin(3),unit,refs(2),ierr,local_rank,header(8),expected_header(8), &
    i,j,k,c,fixture,selection,trial,debug,reference,policy,shock,n,ntype,first,last,cases=0
  logical :: owner,physical_x,use_shock
  real(real_kind) :: error,maximum=0,local_mach,global_mach,observed_local,observed_global, &
    den,u,v,w,p,t,normal,metric_norm,css,rh,ut,vt,wt,tt,shock_x,fx,fy,fz
  real(real_kind) :: target(5),post(5),speeds(9)=[.2e0_real_kind,-.2e0_real_kind,2.e0_real_kind,-2.e0_real_kind,0.e0_real_kind,1.e-20_real_kind,-1.e-20_real_kind,1.e0_real_kind,-1.e0_real_kind]
  real(real_kind),allocatable :: sq(:,:,:,:),sr(:,:,:),sv(:,:,:,:),sp(:,:,:),st(:,:,:), &
    sj(:,:,:),sm(:,:,:,:,:),sx(:,:,:,:),seed(:,:,:,:),actual(:,:,:,:),expected(:,:,:,:), &
    plane(:,:,:),observed(:,:,:,:)
#if !defined(REFERENCE_CPU) && !defined(REFERENCE_CUDA)
  real(real_kind),allocatable :: sync_rhs(:,:,:,:,:,:)
#endif
#ifdef TEST_TOP_VISCOUS
  real(real_kind),allocatable :: source_before(:,:,:,:)
#ifdef REFERENCE_CUDA
  real(real_kind),allocatable,device :: source_before_d(:,:,:)
#endif
#endif
  character(1024) :: directory,path
  character(80) :: text,suffix
#ifdef REFERENCE_CUDA
  type(dim3) :: block,grid
#endif
#if !defined(REFERENCE_CPU) && !defined(REFERENCE_CUDA)
  type(c_ptr) :: state
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text,status=ierr)
  if(ierr/=0) error stop 'top probe requires matching MPI local rank'
  read(text,*) local_rank;call check(device_bind(int(local_rank,c_int)))
#endif
  call MPI_Init(ierr);call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  call get_command_argument(1,directory);call get_command_argument(2,text);read(text,*) axis
  call get_command_argument(3,text)
  if(len_trim(text)>0) then
    read(text,*) global_upper(1);global_upper(3)=global_upper(1)
  endif
  if((ranks/=1.and.ranks/=2).or.axis<1.or.axis>3) error stop 'top probe topology'
  upper=global_upper;origin=0
  if(ranks==2) then
    upper(axis)=global_upper(axis)/2;origin(axis)=rank*upper(axis)
  endif
  owner=origin(2)+upper(2)==global_upper(2)
  expected_header=[upper,origin,trials,4]
#ifdef TEST_TOP_VISCOUS
  expected_header(8)=8
#endif
  allocate(sq(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,5), &
    sr(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo), &
    sv(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,3), &
    sp(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo), &
    st(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo), &
    sj(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo), &
    sm(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,3,3), &
    sx(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,3), &
    seed(0:upper(1),0:upper(2),0:upper(3),5),actual(0:upper(1),0:upper(2),0:upper(3),5), &
    expected(0:upper(1),0:upper(2),0:upper(3),5),plane(0:upper(1),0:upper(3),5), &
    observed(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,5))
#ifdef TEST_TOP_VISCOUS
  allocate(source_before(0:upper(1),0:upper(2),0:upper(3),5))
#endif
  write(suffix,'(a,i8.8,a)') '.rank',rank,'.bin'
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
  ndims=3;numq=5;num_species=0;num_modequ=0;lcomb=.false.;nondimen=.true.
  flowtype='tgv';conschm='643e';difschm='643e';turbmode='none'
  lfilter=.true.;diffterm=.false.;lkhomo=.true.;ljhomo=.false.
  js=0;je=jm;ks=0;ke=km;npdck=3;npdcj=3
  if(jrk==0) npdcj=1
  if(jrk==jrkm) npdcj=2
  if(jrkm==0) npdcj=4
  gamma=gm;mach=ma;const2=c2;const6=c6;ymin=0;ymax=length
  roinf=1;uinf=1;vinf=0;winf=.02e0_real_kind;pinf=1.e0_real_kind/c2
  nscbc_farfield_configured=.true.;nscbc_farfield_rho_target=1;nscbc_farfield_u_target=1
  nscbc_farfield_v_target=0;nscbc_farfield_w_target=.02e0_real_kind;nscbc_farfield_t_target=1
  nscbc_farfield_shock_x=0;nscbc_farfield_shock_angle=35
  allocate(q,mold=sq);allocate(rho,mold=sr);allocate(vel,mold=sv);allocate(prs,mold=sp);allocate(tmp,mold=st)
  allocate(jacob,mold=sj);allocate(dxi,mold=sm);allocate(x,mold=sx);allocate(qrhs,mold=seed)
  allocate(spc(-halo:im+halo,-halo:jm+halo,-halo:km+halo,0))
#ifdef REFERENCE_CUDA
  ierr=cudaSetDevice(mod(rank,2));if(ierr/=cudaSuccess) error stop 'top reference GPU binding'
  allocate(q_d,mold=sq);allocate(rho_d,mold=sr);allocate(vel_d,mold=sv);allocate(prs_d,mold=sp);allocate(tmp_d,mold=st)
  allocate(jacob_d,mold=sj);allocate(dxi_d,mold=sm);allocate(x_d,mold=sx);allocate(qrhs_d,mold=seed)
#ifdef TEST_TOP_VISCOUS
  allocate(source_before_d(0:im,0:km,5))
#endif
  path=trim(directory)//'/gpu'//trim(suffix)
#else
  path=trim(directory)//'/cpu'//trim(suffix)
#endif
  open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='new')
  write(unit) expected_header
  n=1
#else
  call get_command_argument(4,text)
  if(trim(text)=='mach-only') then
    do debug=0,1
      call check(device_state_create(int(local_rank,c_int),upper(1),upper(2),upper(3),halo,int(debug,c_int),state))
      call check(fluid_prepare(state,0_c_int))
      st=1;sv=0;sm=0
      do c=1,3
        sm(:,:,:,c,c)=1
      enddo
      do k=-halo,upper(3)+halo
      do j=-halo,upper(2)+halo
      do i=-halo,upper(1)+halo
        normal=real(i+origin(1)+(global_upper(1)+1)*(k+origin(3)),real_kind)/ &
          real((global_upper(1)+1)*(global_upper(3)+1)-1,real_kind)
        sv(i,j,k,2)=2.e0_real_kind*normal/ma
      enddo
      enddo
      enddo
      call check(device_state_upload(state,device_velocity,sv,bytes(size(sv))))
      call check(device_state_upload(state,device_temperature,st,bytes(size(st))))
      call check(device_state_upload(state,device_metric,sm,bytes(size(sm))))
      observed_local=0
      if(owner) call check(nscbc_top_mach2(state,ma,observed_local))
      local_mach=0
      if(owner) then
        normal=real(upper(1)+origin(1)+(global_upper(1)+1)*(upper(3)+origin(3)),real_kind)/ &
          real((global_upper(1)+1)*(global_upper(3)+1)-1,real_kind)
        local_mach=4.e0_real_kind*normal*normal
      endif
      call MPI_Allreduce(observed_local,observed_global,1,mpi_real_type,MPI_MAX,MPI_COMM_WORLD,ierr)
      error=max(abs(observed_local-local_mach),abs(observed_global-4.e0_real_kind))
      if(.not.ieee_is_finite(observed_global).or.observed_global<0) error stop 'invalid top Mach2 maximum'
      if(real_bytes==8.and..not.(error<=2.e-10_real_kind)) error stop 'multi-block top Mach2 analytic maximum'
      maximum=max(maximum,error);cases=cases+1
      call check(device_state_destroy(state))
    enddo
    print *, 'CPP_NSCBC_TOP_MACH_PASS ',rank,cases,maximum
    print *, 'NSCBC_TOP_PROBE_REAL_BYTES ',rank,real_bytes
    call MPI_Finalize(ierr)
    stop
  endif
  do reference=1,2
    path=trim(directory)//'/cpu'//trim(suffix)
    if(reference==2) path=trim(directory)//'/gpu'//trim(suffix)
    open(newunit=refs(reference),file=trim(path),access='stream',form='unformatted',status='old')
    read(refs(reference)) header
    if(any(header/=expected_header)) error stop 'top reference layout'
  enddo
  allocate(sync_rhs(0:upper(1),0:upper(2),0:upper(3),5,trials,8))
  n=2
#endif
  do debug=0,n-1
#if !defined(REFERENCE_CPU) && !defined(REFERENCE_CUDA)
    call check(device_state_create(int(local_rank,c_int),upper(1),upper(2),upper(3),halo,int(debug,c_int),state))
    call check(fluid_prepare(state,0_c_int));call check(fluid_rhs_prepare(state,0_c_int))
    do reference=1,2
      rewind(refs(reference));read(refs(reference)) header
    enddo
#endif
    do fixture=0,1
      physical_x=fixture==1;ntype=3;first=0;last=upper(1)
      if(physical_x) then
        if(origin(1)==0) then
          ntype=1;first=1
        endif
        if(origin(1)+upper(1)==global_upper(1)) then
          ntype=2;last=upper(1)-1
        endif
        if(origin(1)==0.and.upper(1)==global_upper(1)) ntype=4
      endif
      do selection=0,3
#ifdef TEST_TOP_VISCOUS
        if(selection/=3) cycle
#endif
        policy=selection;shock=0
        if(selection==2) then
          policy=1;shock=1
        elseif(selection==3) then
          policy=2
        endif
        do trial=1,trials
          call initialize(trial)
          local_mach=0
          if(owner) then
            do k=0,upper(3)
            do i=0,upper(1)
              metric_norm=1.e0_real_kind/sum(sm(i,upper(2),k,2,:)**2)
              normal=sum(sm(i,upper(2),k,2,:)*sv(i,upper(2),k,:))
              css=sqrt(st(i,upper(2),k))/ma
              local_mach=max(local_mach,normal*normal*metric_norm/css/css)
            enddo
            enddo
          endif
          call MPI_Allreduce(local_mach,global_mach,1,mpi_real_type,MPI_MAX,MPI_COMM_WORLD,ierr)
#if defined(REFERENCE_CPU) || defined(REFERENCE_CUDA)
          q=sq;rho=sr;vel=sv;prs=sp;tmp=st;jacob=sj;dxi=sm;x=sx
          flowtype='tgv';lfilter=.true.;diffterm=.false.
          bctype=[1,1,50,52,1,1]
          if(physical_x) bctype(1:2)=[11,21]
          lihomo=.not.physical_x;npdci=ntype;is=first;ie=last
          nscbc_farfield_policy=policy;nscbc_farfield_incoming_only=policy==1;nscbc_farfield_sbli_shock=shock==1
          call nscbc_farfield_target_states(-huge(1.e0_real_kind),rh,ut,vt,wt,tt,shock_x,use_shock)
          target=[rh,rh*ut,rh*vt,rh*wt,rh*tt/c2*c6+.5e0_real_kind*rh*(ut*ut+vt*vt+wt*wt)]
          call nscbc_farfield_target_states(huge(1.e0_real_kind),rh,ut,vt,wt,tt,shock_x,use_shock)
          post=[rh,rh*ut,rh*vt,rh*wt,rh*tt/c2*c6+.5e0_real_kind*rh*(ut*ut+vt*vt+wt*wt)]
          do k=0,upper(3)
          do i=0,upper(1)
            plane(i,k,:)=target
            if(shock==1.and.sx(i,upper(2),k,1)>shock_x) plane(i,k,:)=post
          enddo
          enddo
#ifdef REFERENCE_CUDA
          q_d=sq;rho_d=sr;vel_d=sv;prs_d=sp;tmp_d=st;jacob_d=sj;dxi_d=sm;x_d=sx;qrhs_d=seed
          block=dim3(256,1,1);grid=dim3(((im+1)*(km+1)+255)/256,1,1)
          if(policy==2) then
            call nscbc_farfield_y_upper_nonreflecting_rhs_kernel<<<grid,block>>>( &
              im,jm,km,halo,ntype,3,gm,ma,merge(1,0,owner),0)
          else
            call nscbc_farfield_y_upper_rhs_kernel<<<grid,block>>>(im,jm,km,halo,ntype,3,gm,ma,0.e0_real_kind,length,1.e0_real_kind/c2, &
              target(1),target(2)/target(1),target(3)/target(1),target(4)/target(1),target(5), &
              post(1),post(2)/post(1),post(3)/post(1),post(4)/post(1),post(5),shock_x,shock,policy,global_mach,merge(1,0,owner))
          endif
          ierr=cudaGetLastError();if(ierr/=cudaSuccess) error stop 'top reference launch'
          ierr=cudaDeviceSynchronize();if(ierr/=cudaSuccess) error stop 'top reference execution'
          actual=qrhs_d
#else
          qrhs=-seed
          call farfield_nscbc(4)
          actual=-qrhs
#endif
#ifdef TEST_TOP_VISCOUS
#ifdef REFERENCE_CUDA
          call nscbc_farfield_y_upper_prediff_capture_kernel<<<grid,block>>>(source_before_d,im,jm,km,merge(1,0,owner))
          call add_viscous_source(actual,1.e0_real_kind)
          qrhs_d=actual
          call nscbc_farfield_y_upper_viscous_source_kernel<<<grid,block>>>( &
            source_before_d,im,jm,km,gm,ma,merge(1,0,owner))
          ierr=cudaGetLastError();if(ierr/=cudaSuccess) error stop 'reference viscous projection launch'
          ierr=cudaDeviceSynchronize();if(ierr/=cudaSuccess) error stop 'reference viscous projection execution'
          actual=qrhs_d
#else
          ! Select the existing source-projection contract without running a
          ! second farfield call or enabling either transverse filter.
          lfilter=.false.;diffterm=.true.;bctype(3)=41
          if(physical_x) then
            flowtype='bl';turbinf='prof'
          endif
          spg_i0=0;spg_im=0;spg_j0=0;spg_jm=0;spg_k0=0;spg_km=0
          call capture_nscbc_farfield_y_upper_prediff_rhs()
          call add_viscous_source(qrhs,-1.e0_real_kind)
          call apply_nscbc_farfield_y_upper_viscous_source()
          actual=-qrhs
#endif
#endif
          write(unit) target,plane,global_mach,actual
#else
          do reference=1,2
            read(refs(reference)) target,plane,global_mach,expected
            call upload()
            observed_local=0
            if(owner) call check(nscbc_top_mach2(state,ma,observed_local))
            call MPI_Allreduce(observed_local,observed_global,1,mpi_real_type,MPI_MAX,MPI_COMM_WORLD,ierr)
            error=abs(observed_global-global_mach)/max(1.e0_real_kind,abs(global_mach))
            if(.not.ieee_is_finite(observed_global).or.observed_global<0) error stop 'invalid top owner Mach2'
            if(real_bytes==8.and..not.(error<=2.e-10_real_kind)) error stop 'top Mach2 physical-owner reduction'
            maximum=max(maximum,error)
            if(owner) then
              if(shock==1) call check(nscbc_top_target_upload(state,plane,bytes(size(plane))))
              call check(nscbc_top_rhs(state,int(ntype,c_int),3_c_int,int(policy,c_int),int(shock,c_int),0_c_int, &
                gm,ma,length,1.e0_real_kind/c2,target,observed_global))
            endif
#ifdef TEST_TOP_VISCOUS
            if(owner) then
              if(nscbc_top_viscous_source(state,gm,ma)/=device_invalid_argument) &
                error stop 'viscous projection accepted a missing/consumed snapshot'
              call check(nscbc_top_prediff_capture(state))
            endif
            call check(device_state_download(state,device_rhs,actual,bytes(size(actual))))
            source_before=actual
            call add_viscous_source(actual,1.e0_real_kind)
            call check(device_state_upload(state,device_rhs,actual,bytes(size(actual))))
            if(owner) then
              call check(nscbc_top_viscous_source(state,gm,ma))
              if(nscbc_top_viscous_source(state,gm,ma)/=device_invalid_argument) &
                error stop 'viscous snapshot was consumed twice'
            endif
#endif
            call check(device_state_download(state,device_rhs,actual,bytes(size(actual))))
            if(any(.not.ieee_is_finite(actual)).or.any(.not.ieee_is_finite(expected))) &
              error stop 'nonfinite top RHS observation'
            if(debug==0.and.reference==1) then
              sync_rhs(:,:,:,:,trial,4*fixture+selection+1)=actual
            elseif(any(real_to_word(actual)/=real_to_word(sync_rhs(:,:,:,:,trial,4*fixture+selection+1)))) then
              error stop 'top repeat/synchronization RHS words differ'
            endif
            do c=1,5
              error=maxval(abs(actual(:,:,:,c)-expected(:,:,:,c)))/max(1.e0_real_kind,maxval(abs(expected(:,:,:,c))))
              if(real_bytes==8.and..not.(error<=2.e-10_real_kind)) then
                print *, 'TOP_RHS_MISMATCH ',rank,fixture,selection,trial,reference,c,error
                error stop 'top RHS differs from original'
              endif
              maximum=max(maximum,error)
              write(*,'(a,7(1x,i0),1x,es24.16e3)') 'CPP_NSCBC_TOP_NATIVE_DIFFERENCE', &
                rank,debug,fixture,selection,trial,reference,c,error
            enddo
            call check(device_state_download(state,device_q,observed,bytes(size(observed))))
            if(any(real_to_word(observed)/=real_to_word(sq))) &
              error stop 'top RHS changed q'
#ifdef TEST_TOP_VISCOUS
            if(trial==5) then
              if(any(real_to_word(actual)/=real_to_word(source_before))) error stop 'zero-source projection changed RHS'
            endif
            call add_viscous_source(source_before,1.e0_real_kind)
            if(.not.owner) then
              if(any(actual/=source_before)) error stop 'nonowner viscous source changed'
            elseif(any(actual(:,0:upper(2)-1,:,:)/=source_before(:,0:upper(2)-1,:,:))) then
              error stop 'viscous projection changed interior RHS'
            endif
#else
            if(.not.owner) then
              if(any(actual/=seed)) error stop 'top nonowner RHS changed'
            elseif(any(actual(:,0:upper(2)-1,:,:)/=seed(:,0:upper(2)-1,:,:))) then
              error stop 'top changed interior RHS'
            endif
#endif
            cases=cases+1
          enddo
#endif
          if(.not.all(ieee_is_finite(actual))) error stop 'nonfinite top RHS'
        enddo
      enddo
    enddo
#if !defined(REFERENCE_CPU) && !defined(REFERENCE_CUDA)
    call check(device_state_destroy(state))
#endif
  enddo
#if defined(REFERENCE_CPU) || defined(REFERENCE_CUDA)
#ifdef TEST_TOP_VISCOUS
  close(unit);print *, 'NSCBC_TOP_VISCOUS_FROZEN ',rank,24
#else
  close(unit);print *, 'NSCBC_TOP_FROZEN ',rank,96
#endif
#else
  do reference=1,2
    close(refs(reference))
  enddo
  print *, 'NSCBC_TOP_SYNC_WORDS_PASS ',rank,cases
#ifdef TEST_TOP_VISCOUS
  print *, 'CPP_NSCBC_TOP_VISCOUS_PASS ',rank,cases,maximum
#else
  print *, 'CPP_NSCBC_TOP_PASS ',rank,cases,maximum
#endif
#endif
  print *, 'NSCBC_TOP_PROBE_REAL_BYTES ',rank,real_bytes
  call MPI_Finalize(ierr)
contains
#ifdef TEST_TOP_VISCOUS
  subroutine add_viscous_source(values,sign)
    real(real_kind),intent(inout) :: values(0:,0:,0:,:)
    real(real_kind),intent(in) :: sign
    real(real_kind) :: strength
    integer :: ii,jj,kk,component
    do kk=0,upper(3)
    do jj=0,upper(2)
    do ii=0,upper(1)
      strength=.001e0_real_kind*(1+.01e0_real_kind*(ii+origin(1))+.02e0_real_kind*(jj+origin(2))+.03e0_real_kind*(kk+origin(3)))
      if(trial==5) strength=0
      do component=2,5
        values(ii,jj,kk,component)=values(ii,jj,kk,component)+sign*((-1.e0_real_kind)**component)*component*strength
      enddo
    enddo
    enddo
    enddo
  end subroutine
#endif
  subroutine initialize(number)
    integer,intent(in) :: number
    sm=0;sj=1
    do c=1,3
      sm(:,:,:,c,c)=1
    enddo
    do k=-halo,upper(3)+halo
    do j=-halo,upper(2)+halo
    do i=-halo,upper(1)+halo
      fx=real(i+origin(1),real_kind);fy=real(j+origin(2),real_kind);fz=real(k+origin(3),real_kind)
      den=1.03e0_real_kind;t=1.1e0_real_kind;u=.7e0_real_kind;w=.03e0_real_kind
      v=sqrt(t)/ma*speeds(mod(number-1,size(speeds))+1)
      if(number==1) u=10
      if(number>9) then
        den=den+.0003e0_real_kind*fx+.0002e0_real_kind*fy+.0001e0_real_kind*fz
        t=t+.0004e0_real_kind*fy;u=u+.0002e0_real_kind*fx;w=w+.0001e0_real_kind*fy
        sm(i,j,k,2,:)=[.03e0_real_kind,1+.001e0_real_kind*fy,-.02e0_real_kind]
        sm(i,j,k,1,2)=.02e0_real_kind;sm(i,j,k,3,1)=.01e0_real_kind;sj(i,j,k)=1/(1+.001e0_real_kind*fy)
      endif
      sr(i,j,k)=den;sv(i,j,k,:)=[u,v,w];st(i,j,k)=t;sp(i,j,k)=den*t/c2
      sq(i,j,k,:)=[den,den*u,den*v,den*w,sp(i,j,k)*c6+.5e0_real_kind*den*(u*u+v*v+w*w)]
      sx(i,j,k,:)=[(fx-8)/8+.001e0_real_kind*fz,fy,fz]
    enddo
    enddo
    enddo
    do c=1,5
      seed(:,:,:,c)=.125e0_real_kind+.001e0_real_kind*c
    enddo
  end subroutine
#if !defined(REFERENCE_CPU) && !defined(REFERENCE_CUDA)
  subroutine upload()
    call check(device_state_upload(state,device_q,sq,bytes(size(sq))))
    call check(device_state_upload(state,device_rho,sr,bytes(size(sr))))
    call check(device_state_upload(state,device_velocity,sv,bytes(size(sv))))
    call check(device_state_upload(state,device_pressure,sp,bytes(size(sp))))
    call check(device_state_upload(state,device_temperature,st,bytes(size(st))))
    call check(device_state_upload(state,device_jacobian,sj,bytes(size(sj))))
    call check(device_state_upload(state,device_metric,sm,bytes(size(sm))))
    call check(device_state_upload(state,device_rhs,seed,bytes(size(seed))))
  end subroutine
  subroutine check(code)
    integer(c_int),intent(in) :: code
    if(code/=0) then
      print *, 'TOP_DEVICE_STATUS ',code
      error stop 'top device command'
    endif
  end subroutine
  integer(c_size_t) function bytes(count)
    integer,intent(in) :: count
    bytes=int(count,c_size_t)*c_sizeof(0.e0_real_kind)
  end function
#endif
end program
