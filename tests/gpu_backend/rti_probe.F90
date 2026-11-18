program rti_probe
  use iso_c_binding
  use astr_precision, only: real_kind,real_bytes,real_to_word
  use mpi
  use ieee_arithmetic, only: ieee_is_finite
#if defined(REFERENCE_CPU) || defined(REFERENCE_CUDA)
  use commvar
  use commarray, only: q,rho,vel,prs,tmp,qrhs,jacob
  use bc, only: fixbc
  use solver, only: src_rti
  use parallel, only: mpirank,mpisize,irk,irkm,jrk,jrkm,krk,krkm
#else
  use device_state
  use fluid_explicit, only: fluid_prepare,fluid_rhs_prepare,fluid_rti_source
  use boundary_state_cpp, only: fixed_rti_y
#endif
#ifdef REFERENCE_CUDA
  use cudafor
  use commarray_gpu, only: q_d,rho_d,vel_d,prs_d,tmp_d,qrhs_d,jacob_d
  use boundary_gpu, only: fixed_rti_y_kernel
  use solver_gpu, only: rti_source_kernel
#endif
  implicit none
  integer,parameter :: global_upper(3)=[16,16,16],halo=5
  integer :: rank,ranks,axis,upper(3),origin(3),ierr,local_rank,i,j,k,c,unit,refs(2), &
    trial,thermo,debug,reference,n,header(6),cases=0
  real(real_kind) :: gm,ma,c2,c6,density,pressure,temperature,error,maximum=0,analytic_error=0,den,pw
  real(real_kind),allocatable :: sq(:,:,:,:),sr(:,:,:),sv(:,:,:,:),sp(:,:,:),st(:,:,:),sj(:,:,:), &
    base_rhs(:,:,:,:),actual(:,:,:,:),cached(:,:,:,:),rhs(:,:,:,:), &
    expected_q(:,:,:,:),expected_cache(:,:,:,:),expected_rhs(:,:,:,:)
#if !defined(REFERENCE_CPU) && !defined(REFERENCE_CUDA)
  real(real_kind),allocatable :: sync_q(:,:,:,:,:),sync_cache(:,:,:,:,:),sync_rhs(:,:,:,:,:)
#endif
  logical :: owner
  character(1024) :: directory,path
  character(80) :: text,suffix
#ifdef REFERENCE_CUDA
  type(dim3) :: block,grid
#endif
#if !defined(REFERENCE_CPU) && !defined(REFERENCE_CUDA)
  type(c_ptr) :: state
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text,status=ierr)
  if(ierr/=0) error stop 'RTI probe local device rank'
  read(text,*) local_rank;call check(device_bind(int(local_rank,c_int)))
#endif
  call MPI_Init(ierr);call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  call get_command_argument(1,directory);call get_command_argument(2,text);read(text,*) axis
  if((ranks/=1.and.ranks/=2).or.axis<1.or.axis>3) error stop 'RTI probe topology'
  upper=global_upper;origin=0
  if(ranks==2) then
    upper(axis)=global_upper(axis)/2;origin(axis)=rank*upper(axis)
  endif
  allocate(sq(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,5), &
    sr(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo), &
    sv(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,3), &
    sp(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo), &
    st(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo), &
    sj(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo), &
    base_rhs(0:upper(1),0:upper(2),0:upper(3),5))
  allocate(actual,mold=sq);allocate(rhs,mold=base_rhs)
  allocate(cached(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,6))
  allocate(expected_q,mold=sq);allocate(expected_cache,mold=cached);allocate(expected_rhs,mold=rhs)
#if !defined(REFERENCE_CPU) && !defined(REFERENCE_CUDA)
  allocate(sync_q(size(sq,1),size(sq,2),size(sq,3),5,6), &
    sync_cache(size(cached,1),size(cached,2),size(cached,3),6,6), &
    sync_rhs(size(rhs,1),size(rhs,2),size(rhs,3),5,6))
#endif
  write(suffix,'(a,i8.8,a)') '.rank',rank,'.bin'
#if defined(REFERENCE_CPU) || defined(REFERENCE_CUDA)
  im=upper(1);jm=upper(2);km=upper(3);ia=16;ja=16;ka=16;ndims=3;numq=5
  num_species=0;num_modequ=0;nondimen=.true.;lcomb=.false.
  mpirank=rank;mpisize=ranks;irk=0;irkm=0;jrk=0;jrkm=0;krk=0;krkm=0
  if(axis==1) then
    irk=rank;irkm=ranks-1
  elseif(axis==2) then
    jrk=rank;jrkm=ranks-1
  else
    krk=rank;krkm=ranks-1
  endif
  allocate(q,mold=sq);allocate(rho,mold=sr);allocate(vel,mold=sv);allocate(prs,mold=sp)
  allocate(tmp,mold=st);allocate(qrhs,mold=rhs);allocate(jacob,mold=sj)
#ifdef REFERENCE_CUDA
  ierr=cudaSetDevice(mod(rank,2));if(ierr/=cudaSuccess) error stop 'RTI reference GPU binding'
  allocate(q_d,mold=sq);allocate(rho_d,mold=sr);allocate(vel_d,mold=sv);allocate(prs_d,mold=sp)
  allocate(tmp_d,mold=st);allocate(qrhs_d,mold=rhs);allocate(jacob_d,mold=sj)
  path=trim(directory)//'/gpu'//trim(suffix)
#else
  path=trim(directory)//'/cpu'//trim(suffix)
#endif
  open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='new');write(unit) [upper,origin]
  n=1
#else
  do reference=1,2
    path=trim(directory)//'/cpu'//trim(suffix)
    if(reference==2) path=trim(directory)//'/gpu'//trim(suffix)
    open(newunit=refs(reference),file=trim(path),access='stream',form='unformatted',status='old')
    read(refs(reference)) header
    if(any(header/=[upper,origin])) error stop 'RTI reference layout'
  enddo
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
    do thermo=1,2
      gm=1.4e0_real_kind;ma=.3e0_real_kind
      if(thermo==2) then
        gm=1.67e0_real_kind;ma=2.5e0_real_kind
      endif
      c2=gm*ma*ma;c6=1/(gm-1)
      call initialize()
      do trial=0,2
        owner=(trial==0.and.origin(2)==0).or. &
          (trial==1.and.origin(2)+upper(2)==global_upper(2))
#if defined(REFERENCE_CPU) || defined(REFERENCE_CUDA)
        gamma=gm;mach=ma;const2=c2;const6=c6
        q=sq;rho=sr;vel=sv;prs=sp;tmp=st;jacob=sj;qrhs=base_rhs
        if(trial<2) qrhs=0
#ifdef REFERENCE_CUDA
        q_d=q;rho_d=rho;vel_d=vel;prs_d=prs;tmp_d=tmp;jacob_d=jacob;qrhs_d=qrhs
        block=dim3(256,1,1)
        if(trial<2) then
          grid=dim3(((im+1)*(km+1)+255)/256,1,1)
          call fixed_rti_y_kernel<<<grid,block>>>(im,jm,km,halo,c2,c6, &
            merge(1,0,owner.and.trial==0),merge(1,0,owner.and.trial==1))
        else
          grid=dim3(((im+1)*(jm+1)*(km+1)+255)/256,1,1)
          call rti_source_kernel<<<grid,block>>>(qrhs_d,rho_d,vel_d,jacob_d,im,jm,km,halo)
        endif
        ierr=cudaGetLastError();if(ierr/=cudaSuccess) error stop 'RTI reference launch'
        ierr=cudaDeviceSynchronize();if(ierr/=cudaSuccess) error stop 'RTI reference execution'
        actual=q_d;cached(:,:,:,1)=rho_d;cached(:,:,:,2:4)=vel_d;cached(:,:,:,5)=prs_d;cached(:,:,:,6)=tmp_d;rhs=qrhs_d
#else
        if(trial<2) then
          call fixbc(3+trial)
        else
          call src_rti()
        endif
        actual=q;cached(:,:,:,1)=rho;cached(:,:,:,2:4)=vel;cached(:,:,:,5)=prs;cached(:,:,:,6)=tmp;rhs=qrhs
#endif
        write(unit) actual,cached,rhs
#else
        do reference=1,2
          read(refs(reference)) expected_q,expected_cache,expected_rhs
          call upload()
          if(trial<2) then
            if(owner) call check(fixed_rti_y(state,int(trial,c_int),c2,c6))
          else
            call check(fluid_rti_source(state))
          endif
          call observe()
          if(debug==0.and.reference==1) then
            sync_q(:,:,:,:,3*(thermo-1)+trial+1)=actual
            sync_cache(:,:,:,:,3*(thermo-1)+trial+1)=cached
            sync_rhs(:,:,:,:,3*(thermo-1)+trial+1)=rhs
          else
            if(any(real_to_word(actual)/=real_to_word(sync_q(:,:,:,:,3*(thermo-1)+trial+1))).or. &
               any(real_to_word(cached)/=real_to_word(sync_cache(:,:,:,:,3*(thermo-1)+trial+1))).or. &
               any(real_to_word(rhs)/=real_to_word(sync_rhs(:,:,:,:,3*(thermo-1)+trial+1)))) &
              error stop 'RTI repeated/synchronization words differ'
          endif
          do c=1,5
            call compare(actual(:,:,:,c),expected_q(:,:,:,c))
            if(trial==2) then
              call compare(rhs(:,:,:,c),expected_rhs(:,:,:,c))
            else
              if(any(rhs(:,:,:,c)/=base_rhs(:,:,:,c))) error stop 'RTI state command changed RHS'
            endif
          enddo
          do c=1,6
            call compare(cached(:,:,:,c),expected_cache(:,:,:,c))
          enddo
          if(trial==2) then
            if(any(rhs(:,:,:,1)/=base_rhs(:,:,:,1)).or.any(rhs(:,:,:,2)/=base_rhs(:,:,:,2)).or. &
               any(rhs(:,:,:,4)/=base_rhs(:,:,:,4))) error stop 'RTI source changed unrelated equations'
            if(any(cached(:,:,:,1)/=sr).or.any(cached(:,:,:,2:4)/=sv).or. &
               any(cached(:,:,:,5)/=sp).or.any(cached(:,:,:,6)/=st)) error stop 'RTI source changed primitive caches'
            expected_rhs=base_rhs
            expected_rhs(:,:,:,3)=base_rhs(:,:,:,3)+sr(0:upper(1),0:upper(2),0:upper(3))*sj(0:upper(1),0:upper(2),0:upper(3))
            expected_rhs(:,:,:,5)=base_rhs(:,:,:,5)+sr(0:upper(1),0:upper(2),0:upper(3))* &
              sv(0:upper(1),0:upper(2),0:upper(3),2)*sj(0:upper(1),0:upper(2),0:upper(3))
            error=maxval(abs(rhs-expected_rhs));analytic_error=max(analytic_error,error)
            if(real_bytes==8.and..not.(error<=2.e-10_real_kind)) error stop 'RTI gravity identity'
            if(any(actual/=sq)) error stop 'RTI source changed q'
          elseif(owner) then
            j=trial*upper(2);den=2;pw=1
            if(trial==1) then
              den=1;pw=2.5e0_real_kind
            endif
            if(any(actual(0:upper(1),j,0:upper(3),1)/=den).or. &
               any(actual(0:upper(1),j,0:upper(3),2:4)/=0)) error stop 'RTI fixed density/velocity'
            error=maxval(abs(cached(0:upper(1),j,0:upper(3),6)-pw/den*c2))
            analytic_error=max(analytic_error,error)
            if(real_bytes==8.and..not.(error<=2.e-10_real_kind)) error stop 'RTI wall temperature'
          endif
          cases=cases+1
        enddo
#endif
      enddo
    enddo
#if !defined(REFERENCE_CPU) && !defined(REFERENCE_CUDA)
    call check(device_state_destroy(state))
#endif
  enddo
#if defined(REFERENCE_CPU) || defined(REFERENCE_CUDA)
  close(unit);print *, 'RTI_FROZEN ',rank,6
#else
  do reference=1,2
    close(refs(reference))
  enddo
  print *, 'CPP_RTI_PASS ',rank,cases,maximum,analytic_error
#endif
  call MPI_Finalize(ierr)
contains
  subroutine initialize()
    do k=-halo,upper(3)+halo
    do j=-halo,upper(2)+halo
    do i=-halo,upper(1)+halo
      density=1.2e0_real_kind+.001e0_real_kind*(i+origin(1));temperature=1+.002e0_real_kind*(j+origin(2))
      pressure=density*temperature/c2
      sr(i,j,k)=density;sp(i,j,k)=pressure;st(i,j,k)=temperature
      sv(i,j,k,:)=[.1e0_real_kind,.01e0_real_kind*(j+origin(2)-8),-.05e0_real_kind]
      sj(i,j,k)=.2e0_real_kind+.001e0_real_kind*(i+origin(1))+.002e0_real_kind*(k+origin(3))
      sq(i,j,k,:)=[density,density*sv(i,j,k,:),pressure*c6+.5e0_real_kind*density*sum(sv(i,j,k,:)**2)]
    enddo
    enddo
    enddo
    do c=1,5
      base_rhs(:,:,:,c)=.125e0_real_kind+.01e0_real_kind*c
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
    call check(device_state_upload(state,device_rhs,base_rhs,bytes(size(base_rhs))))
  end subroutine
  subroutine observe()
    call check(device_state_download(state,device_q,actual,bytes(size(actual))))
    call check(device_state_download(state,device_rho,cached(:,:,:,1),bytes(size(sr))))
    call check(device_state_download(state,device_velocity,cached(:,:,:,2:4),bytes(size(sv))))
    call check(device_state_download(state,device_pressure,cached(:,:,:,5),bytes(size(sp))))
    call check(device_state_download(state,device_temperature,cached(:,:,:,6),bytes(size(st))))
    call check(device_state_download(state,device_rhs,rhs,bytes(size(rhs))))
  end subroutine
  subroutine compare(a,b)
    real(real_kind),intent(in) :: a(:,:,:),b(:,:,:)
    error=maxval(abs(a-b))/max(1.e0_real_kind,maxval(abs(b)))
    if(.not.all(ieee_is_finite(a)).or..not.all(ieee_is_finite(b))) error stop 'RTI nonfinite component'
    if(real_bytes==8.and..not.(error<=2.e-10_real_kind)) error stop 'RTI component mismatch'
    maximum=max(maximum,error)
  end subroutine
  integer(c_size_t) function bytes(count)
    integer,intent(in) :: count
    bytes=int(count,c_size_t)*c_sizeof(0.e0_real_kind)
  end function
  subroutine check(status)
    integer,intent(in) :: status
    if(status/=0) then
      print *, 'RTI_DEVICE_ERROR ',rank,status
      call MPI_Abort(MPI_COMM_WORLD,status,ierr)
    endif
  end subroutine
#endif
end program
