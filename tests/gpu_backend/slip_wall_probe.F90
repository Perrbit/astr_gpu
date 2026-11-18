program slip_wall_probe
  use iso_c_binding
  use mpi
  use ieee_arithmetic, only: ieee_is_finite
#if defined(REFERENCE_CPU) || defined(REFERENCE_CUDA)
  use commvar
  use commarray, only: q,rho,vel,prs,tmp,spc,x,qrhs,bnorm_j0,bnorm_jm
  use bc, only: slipisotwall,slipadibwall,xslip,wall_blowing_configured,wall_blowing_amplitude
  use parallel, only: mpirank,mpisize,lio,irk,irkm,jrk,jrkm,krk,krkm
#else
  use device_state
  use fluid_explicit, only: fluid_prepare,fluid_rhs_prepare
  use boundary_state_cpp, only: slip_wall_y,slip_mask_upload
#endif
#ifdef REFERENCE_CUDA
  use cudafor
  use commarray_gpu, only: q_d,rho_d,vel_d,prs_d,tmp_d,x_d,qrhs_d,bnorm_j0_d,bnorm_jm_d
  use boundary_gpu, only: wall_slip_isothermal_y_kernel,wall_slip_adiabatic_y_kernel, &
    wall_blowing_enabled_d,wall_blowing_amplitude_d,legacy_wall_forcing_d
#endif
  implicit none
  integer,parameter :: g(3)=[16,16,16],halo=5
  real(c_double),parameter :: gm=1.4d0,c2=gm,c6=1.d0/(gm-1),cutoff=8.d0
  integer :: rank,ranks,axis=1,upper(3),origin(3),ierr,local_rank,unit,refs(2),header(6), &
    i,j,k,c,geometry,kind,code,side,temperature_case,debug,reference,n,cases=0,analytic=0,wall_j
  real(c_double) :: tw,error,maximum=0,analytic_max=0,normal(3),u(3),flux,expected_t,expected_p,rr,t,p,fx,fy,fz
  real(c_double),allocatable :: sq(:,:,:,:),sr(:,:,:),sv(:,:,:,:),sp(:,:,:),st(:,:,:), &
    metric(:,:,:,:,:),coords(:,:,:,:),bn(:,:,:,:),actual(:,:,:,:),cached(:,:,:,:), &
    expected_q(:,:,:,:),expected_pv(:,:,:,:),rhs(:,:,:,:)
  integer(c_int8_t),allocatable :: mask(:,:)
  logical :: owner
  character(1024) :: directory,path
  character(80) :: text,suffix
#ifdef REFERENCE_CUDA
  type(dim3) :: block,grid
#endif
#if !defined(REFERENCE_CPU) && !defined(REFERENCE_CUDA)
  type(c_ptr) :: state
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text,status=ierr)
  if(ierr/=0) error stop 'slip probe requires matching MPI local rank'
  read(text,*) local_rank;call check(device_bind(int(local_rank,c_int)))
#endif
  call MPI_Init(ierr);call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  call get_command_argument(1,directory);call get_command_argument(2,text);read(text,*) axis
  if((ranks/=1.and.ranks/=2).or.axis<1.or.axis>3) error stop 'slip probe topology'
  upper=g;origin=0
  if(ranks==2) then
    upper(axis)=g(axis)/2;origin(axis)=rank*upper(axis)
  endif
  allocate(sq(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,5), &
    sr(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo), &
    sv(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,3), &
    sp(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo), &
    st(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo), &
    metric(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,3,3), &
    coords(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,3),bn(0:upper(1),0:upper(3),3,2), &
    actual(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,5), &
    cached(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,6), &
    expected_q(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,5), &
    expected_pv(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,6), &
    rhs(0:upper(1),0:upper(2),0:upper(3),5),mask(0:upper(1),0:upper(3)))
  write(suffix,'(a,i8.8,a)') '.rank',rank,'.bin'
#if defined(REFERENCE_CPU) || defined(REFERENCE_CUDA)
  im=upper(1);jm=upper(2);km=upper(3);ia=g(1);ja=g(2);ka=g(3);ndims=3;numq=5;num_species=0;num_modequ=0
  mpirank=rank;mpisize=ranks;lio=rank==0;irk=0;irkm=0;jrk=0;jrkm=0;krk=0;krkm=0
  if(axis==1) then
    irk=rank;irkm=ranks-1
  elseif(axis==2) then
    jrk=rank;jrkm=ranks-1
  else
    krk=rank;krkm=ranks-1
  endif
  nondimen=.true.;lcomb=.false.;gamma=gm;mach=1;const2=c2;const6=c6;turbmode='none'
  wall_blowing_configured=.true.;wall_blowing_amplitude=0;xslip=cutoff
  allocate(q,mold=sq);allocate(rho,mold=sr);allocate(vel,mold=sv);allocate(prs,mold=sp);allocate(tmp,mold=st)
  allocate(x,mold=coords);allocate(qrhs,mold=rhs)
  allocate(spc(-halo:im+halo,-halo:jm+halo,-halo:km+halo,0),bnorm_j0(0:im,0:km,3),bnorm_jm(0:im,0:km,3))
#ifdef REFERENCE_CUDA
  ierr=cudaSetDevice(mod(rank,2));if(ierr/=cudaSuccess) error stop 'slip reference GPU binding'
  allocate(q_d,mold=sq);allocate(rho_d,mold=sr);allocate(vel_d,mold=sv);allocate(prs_d,mold=sp);allocate(tmp_d,mold=st)
  allocate(x_d,mold=coords);allocate(qrhs_d,mold=rhs)
  allocate(bnorm_j0_d(0:im,0:km,3),bnorm_jm_d(0:im,0:km,3))
  wall_blowing_enabled_d=0;wall_blowing_amplitude_d=0;legacy_wall_forcing_d=0
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
    open(newunit=refs(reference),file=trim(path),access='stream',form='unformatted',status='old');read(refs(reference)) header
    if(any(header/=[upper,origin])) error stop 'slip reference layout'
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
    do geometry=0,2
      call initialize()
      do kind=1,2
        code=421
        if(kind==2) code=411
        do side=0,1
          wall_j=side*upper(2)
          owner=(side==0.and.origin(2)==0).or.(side==1.and.origin(2)+upper(2)==g(2))
          do temperature_case=1,2
            tw=1.2d0
            if(temperature_case==2) tw=.9d0
#if defined(REFERENCE_CPU) || defined(REFERENCE_CUDA)
            q=sq;rho=sr;vel=sv;prs=sp;tmp=st;x=coords;qrhs=0
            bnorm_j0=bn(:,:,:,1);bnorm_jm=bn(:,:,:,2)
#ifdef REFERENCE_CUDA
            q_d=sq;rho_d=sr;vel_d=sv;prs_d=sp;tmp_d=st;x_d=coords;qrhs_d=0
            bnorm_j0_d=bn(:,:,:,1);bnorm_jm_d=bn(:,:,:,2)
            block=dim3(256,1,1);grid=dim3(((im+1)*(km+1)+255)/256,1,1)
            if(code==411) then
              call wall_slip_isothermal_y_kernel<<<grid,block>>>(im,jm,km,halo,c2,c6,cutoff,tw,tw, &
                merge(1,0,owner.and.side==0),merge(1,0,owner.and.side==1),0.d0)
            else
              call wall_slip_adiabatic_y_kernel<<<grid,block>>>(im,jm,km,halo,c2,c6, &
                merge(1,0,owner.and.side==0),merge(1,0,owner.and.side==1),0.d0)
            endif
            ierr=cudaGetLastError();if(ierr/=cudaSuccess) error stop 'slip reference launch'
            ierr=cudaDeviceSynchronize();if(ierr/=cudaSuccess) error stop 'slip reference execution'
            actual=q_d;cached(:,:,:,1)=rho_d;cached(:,:,:,2:4)=vel_d;cached(:,:,:,5)=prs_d;cached(:,:,:,6)=tmp_d
#else
            if(code==411) then
              call slipisotwall(3+side,tw)
            else
              call slipadibwall(3+side)
            endif
            actual=q;cached(:,:,:,1)=rho;cached(:,:,:,2:4)=vel;cached(:,:,:,5)=prs;cached(:,:,:,6)=tmp
#endif
            write(unit) actual,cached
#else
            do reference=1,2
              read(refs(reference)) expected_q,expected_pv
              call upload()
              if(owner) then
                if(code==411.and.side==0) call check(slip_mask_upload(state,mask,int(size(mask),c_size_t)))
                call check(slip_wall_y(state,int(code,c_int),int(side,c_int),tw,c2,c6))
              endif
              call observe()
              do c=1,5
                call compare(actual(:,:,:,c),expected_q(:,:,:,c))
              enddo
              do c=1,6
                call compare(cached(:,:,:,c),expected_pv(:,:,:,c))
              enddo
              if(any(rhs/=7)) error stop 'state-only slip command changed RHS'
              if(owner) then
                do k=0,upper(3)
                do i=0,upper(1)
                  normal=bn(i,k,:,side+1)
                  u=cached(i,wall_j,k,2:4);flux=sum(normal*u)
                  error=abs(flux)
                  if(.not.(error<=2.d-10)) error stop 'slip/solid wall has nonzero normal velocity'
                  analytic_max=max(analytic_max,error)
                  if(code==421.or.(side==0.and.mask(i,k)==1)) then
                    j=merge(1,upper(2)-1,side==0)
                    expected_t=(4*st(i,j,k)-st(i,merge(2,upper(2)-2,side==0),k))/3
                  else
                    expected_t=tw
                    if(maxval(abs(u))>2.d-10) error stop 'no-slip branch retained tangential velocity'
                  endif
                  error=abs(cached(i,wall_j,k,6)-expected_t)
                  if(.not.(error<=2.d-10)) error stop 'slip thermal branch'
                  analytic_max=max(analytic_max,error);analytic=analytic+1
                enddo
                enddo
              endif
              cases=cases+1
            enddo
#endif
          enddo
        enddo
      enddo
    enddo
#if !defined(REFERENCE_CPU) && !defined(REFERENCE_CUDA)
    call check(device_state_destroy(state))
#endif
  enddo
#if defined(REFERENCE_CPU) || defined(REFERENCE_CUDA)
  close(unit);print *, 'SLIP_WALL_FROZEN ',rank,24
#else
  do reference=1,2
    close(refs(reference))
  enddo
  print *, 'CPP_SLIP_WALL_PASS ',rank,cases,maximum,analytic,analytic_max
#endif
  call MPI_Finalize(ierr)
contains
  subroutine initialize()
    do k=-halo,upper(3)+halo
    do j=-halo,upper(2)+halo
    do i=-halo,upper(1)+halo
      fx=real(i+origin(1),c_double);fy=real(j+origin(2),c_double);fz=real(k+origin(3),c_double)
      t=1+.002d0*fy;p=(1+.001d0*fx+.0001d0*fz)/c2;rr=p/t*c2
      sv(i,j,k,:)=[.4d0+.002d0*fy,.1d0+.001d0*fx,.03d0+.001d0*fz]
      sr(i,j,k)=rr;sp(i,j,k)=p;st(i,j,k)=t
      sq(i,j,k,:)=[rr,rr*sv(i,j,k,:),p*c6+.5d0*rr*sum(sv(i,j,k,:)**2)]
      coords(i,j,k,:)=[fx,fy,fz]
      normal=[0.d0,1.d0,0.d0]
      if(geometry==1) normal=[.2d0,1.d0,-.1d0]
      if(geometry==2) normal=[.2d0*sin(fx*.1d0),1.d0,.1d0*cos(fz*.1d0)]
      metric(i,j,k,:,:)=0
      metric(i,j,k,1,1)=1;metric(i,j,k,3,3)=1;metric(i,j,k,2,:)=normal
    enddo
    enddo
    enddo
    do k=0,upper(3)
    do i=0,upper(1)
      normal=metric(i,0,k,2,:);normal=normal/sqrt(sum(normal*normal))
      bn(i,k,:,1)=normal;bn(i,k,:,2)=-normal
      mask(i,k)=merge(1_c_int8_t,0_c_int8_t,coords(i,0,k,1)<=cutoff)
    enddo
    enddo
  end subroutine
#if !defined(REFERENCE_CPU) && !defined(REFERENCE_CUDA)
  subroutine upload()
    call check(device_state_upload(state,device_q,sq,bytes(size(sq))))
    call check(device_state_upload(state,device_rho,sr,bytes(size(sr))))
    call check(device_state_upload(state,device_velocity,sv,bytes(size(sv))))
    call check(device_state_upload(state,device_pressure,sp,bytes(size(sp))))
    call check(device_state_upload(state,device_temperature,st,bytes(size(st))))
    call check(device_state_upload(state,device_metric,metric,bytes(size(metric))))
    rhs=7;call check(device_state_upload(state,device_rhs,rhs,bytes(size(rhs))))
  end subroutine
  subroutine observe()
    call check(device_state_download(state,device_q,actual,bytes(size(actual))))
    call check(device_state_download(state,device_rho,cached(:,:,:,1),bytes(size(sr))))
    call check(device_state_download(state,device_velocity,cached(:,:,:,2:4),bytes(size(sv))))
    call check(device_state_download(state,device_pressure,cached(:,:,:,5),bytes(size(sp))))
    call check(device_state_download(state,device_temperature,cached(:,:,:,6),bytes(size(st))))
    call check(device_state_download(state,device_rhs,rhs,bytes(size(rhs))))
  end subroutine
  subroutine compare(value,expected)
    real(c_double),intent(in) :: value(:,:,:),expected(:,:,:)
    error=maxval(abs(value-expected))/max(1.d0,maxval(abs(expected)))
    if(.not.all(ieee_is_finite(value)).or..not.(error<=2.d-10)) then
      print *, 'SLIP_WALL_MISMATCH ',rank,geometry,code,side,temperature_case,reference,error
      error stop 'slip state differs from original'
    endif
    maximum=max(maximum,error)
  end subroutine
  subroutine check(status)
    integer(c_int),intent(in) :: status
    if(status/=0) then
      print *, 'SLIP_DEVICE_STATUS ',status
      error stop 'slip device command'
    endif
  end subroutine
  integer(c_size_t) function bytes(count)
    integer,intent(in) :: count
    bytes=int(count,c_size_t)*8
  end function
#endif
end program
