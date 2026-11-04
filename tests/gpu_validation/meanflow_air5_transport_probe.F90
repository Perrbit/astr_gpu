program meanflow_air5_transport_probe
  use iso_fortran_env, only: real64
  use mpi
  use commvar, only: im,jm,km,nstep,time,nondimen,lcomb,numq,num_species, &
    num_modequ,reynolds,tempconst,tempconst1,lreport,ltimrpt,ia,ja,ka,lavg
  use commarray, only: rho,vel,prs,tmp,tve,spc,dvel
  use parallel, only: mpirank,lio,ig0,jg0,kg0
  use statistic, only: meanflowcal,nsamples,mean_time_weight,rom,u1m,u2m,u3m,pm,tm, &
    u11,u22,u33,u12,u13,u23,pp,tt,tu1,tu2,tu3,u111,u222,u333,u112,u113,u122,u133, &
    u223,u233,u123,u1rem,u2rem,u3rem,pu1,pu2,pu3,sgmam11,sgmam22,sgmam33, &
    sgmam12,sgmam13,sgmam23,visdif1,visdif2,visdif3,disspa,predil
  use statistic, only: complete_mean_statistics_file,reset_mean_statistics_window
  use checkpoint_state_io, only: checkpoint_state_identity
  use index_slice_source, only: slice_source_value_cpu
  use readwrite, only: readmeanflow,writemeanflow
  use commvar, only: iomode
#ifdef ASTR_AIR5_CHEMISTRY
  use chemistry_transport, only: air5_diffusive_flux
#endif
  use fludyna, only: miucal
#ifdef TEST_MEANFLOW_GPU
  use cudafor, only: cudaGetDeviceCount,cudaSetDevice,cudaSuccess
  use commvar, only: hm,feqavg,diffterm
  use commarray_gpu, only: rho_d,vel_d,prs_d,tmp_d,tve_d,spc_d,dvel_d
  use chemistry_mean_statistics_gpu, only: initialize_air5_mean_statistics_gpu, &
    release_air5_mean_statistics_gpu,set_air5_mean_sampling_gpu, &
    accumulate_air5_mean_statistics_gpu,complete_air5_mean_statistics_file,slice_air5_mean_info_gpu, &
    reset_air5_mean_statistics_gpu
#endif
  implicit none
  character(32) :: mode
  character(32) :: legacy_action
  integer :: ierr,sample,status
  real(real64) :: gradient(3,3),base_gradient(3,3),stress(3,3),expected(44),actual(44), &
    reference(44),previous_reference(44),mu,divergence,r,p,t,u,v,w,times(3)
  real(real64) :: species_flux(5,3),energy_flux(3),ev_flux(3),velocity(3)
  real(real64) :: zero_species_gradient(5,3)
  character(1024) :: output_prefix
  integer :: ranks
  type(checkpoint_state_identity) :: identity
#ifdef TEST_MEANFLOW_GPU
  integer :: devices
  logical :: mean_enabled,mean_covered
  integer(8) :: mean_count
  real(8) :: mean_window(2),mean_duration
#endif
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,mpirank,ierr)
  lio=.false.
  call get_command_argument(1,mode)
  call get_command_argument(2,output_prefix)
  call get_command_argument(3,legacy_action)
  call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  im=1; jm=1; km=1; numq=11; num_species=5; num_modequ=1
  ia=ranks;ja=1;ka=1;ig0=mpirank;jg0=0;kg0=0
  nondimen=trim(mode)=='perfect_nondim'.or.trim(mode)=='air5_nondim'
  lcomb=index(trim(mode),'air5')==1
#ifndef ASTR_AIR5_CHEMISTRY
  if(lcomb) error stop 'AIR5 meanflow probe requires AIR5 build'
#endif
  reynolds=7.d0; tempconst=0.4d0; tempconst1=1.4d0
  lreport=.false.; ltimrpt=.false.; nsamples=0
  lavg=.true.
  if(trim(legacy_action)=='read') then
    nstep=3;nsamples=4;iomode='h'
    call readmeanflow()
    if(any(tu2/=17.d0).or.any(tu3/=-23.d0)) error stop 'legacy tu2/tu3 read binding mismatch'
    if(mean_time_weight/=1.d0) error stop 'legacy mean time weight read mismatch'
    print*, 'MEANFLOW_LEGACY_READ_PASS'
    call MPI_Finalize(ierr)
    stop
  endif
  allocate(rho(0:1,0:1,0:1),vel(0:1,0:1,0:1,3),prs(0:1,0:1,0:1), &
    tmp(0:1,0:1,0:1),tve(0:1,0:1,0:1),spc(0:1,0:1,0:1,5),dvel(0:1,0:1,0:1,3,3))
  gradient=reshape([1.d0,2.d0,-3.d0,4.d0,-5.d0,6.d0,7.d0,-8.d0,9.d0],[3,3])
  base_gradient=gradient
  times=[0.d0,0.3d0,1.d0]
  velocity=[20.d0,-3.d0,4.d0]
  rho=0.05d0; prs=5.d4; tve=1500.d0
  vel(:,:,:,1)=velocity(1); vel(:,:,:,2)=velocity(2); vel(:,:,:,3)=velocity(3)
  dvel(0,0,0,:,:)=gradient
  dvel(1,0,0,:,:)=gradient; dvel(0,1,0,:,:)=gradient; dvel(1,1,0,:,:)=gradient
  dvel(0,0,1,:,:)=gradient; dvel(1,0,1,:,:)=gradient
  dvel(0,1,1,:,:)=gradient; dvel(1,1,1,:,:)=gradient
  expected=0.d0
  zero_species_gradient=0.d0
#ifdef TEST_MEANFLOW_GPU
  if(trim(mode)=='air5') then
    if(len_trim(output_prefix)==0) error stop 'missing probe output prefix'
    status=cudaGetDeviceCount(devices)
    if(status/=cudaSuccess.or.devices<1) error stop 'no GPU for mean statistics probe'
    status=cudaSetDevice(mod(mpirank,devices))
    if(status/=cudaSuccess) error stop 'cannot bind mean statistics probe GPU'
    feqavg=0;diffterm=.true.;lavg=.true.
    allocate(rho_d(-hm:1+hm,-hm:1+hm,-hm:1+hm),vel_d(-hm:1+hm,-hm:1+hm,-hm:1+hm,3), &
      prs_d(-hm:1+hm,-hm:1+hm,-hm:1+hm),tmp_d(-hm:1+hm,-hm:1+hm,-hm:1+hm), &
      tve_d(-hm:1+hm,-hm:1+hm,-hm:1+hm),spc_d(-hm:1+hm,-hm:1+hm,-hm:1+hm,5), &
      dvel_d(0:1,0:1,0:1,3,3))
    call initialize_air5_mean_statistics_gpu(huge(0_8)/4096_8)
    call set_air5_mean_sampling_gpu(.true.)
  endif
#endif
  if(len_trim(output_prefix)>0) then
    identity=checkpoint_state_identity(0_8,0.d0,0.d0,0.1d0)
    call complete_mean_statistics_file(trim(output_prefix)//'_cpu_empty.h5',.true.,identity,huge(0_8)/4096_8)
    call reset_mean_statistics_window()
    call complete_mean_statistics_file(trim(output_prefix)//'_cpu_empty.h5',.false.,identity,huge(0_8)/4096_8)
#ifdef TEST_MEANFLOW_GPU
    if(trim(mode)=='air5') then
      call complete_air5_mean_statistics_file(trim(output_prefix)//'_gpu_empty.h5',.true.,identity,huge(0_8)/4096_8)
    endif
#endif
  endif
  do sample=1,3
    r=0.05d0+0.005d0*sample;p=5.d4+100.d0*sample
    velocity=[20.d0+sample,-3.d0+0.5d0*sample,4.d0-0.25d0*sample]
    u=velocity(1);v=velocity(2);w=velocity(3)
    rho=r;prs=p
    vel(:,:,:,1)=u;vel(:,:,:,2)=v;vel(:,:,:,3)=w
    gradient=base_gradient*(0.75d0+0.25d0*sample)
    dvel(:,:,:,1,1)=gradient(1,1);dvel(:,:,:,1,2)=gradient(1,2);dvel(:,:,:,1,3)=gradient(1,3)
    dvel(:,:,:,2,1)=gradient(2,1);dvel(:,:,:,2,2)=gradient(2,2);dvel(:,:,:,2,3)=gradient(2,3)
    dvel(:,:,:,3,1)=gradient(3,1);dvel(:,:,:,3,2)=gradient(3,2);dvel(:,:,:,3,3)=gradient(3,3)
    tmp=3000.d0+500.d0*sample
    if(nondimen) tmp=1.d0+0.25d0*sample
    spc(:,:,:,1)=0.75d0-0.1d0*sample
    spc(:,:,:,2)=0.20d0; spc(:,:,:,3)=0.03d0
    spc(:,:,:,4)=0.01d0; spc(:,:,:,5)=0.01d0+0.1d0*sample
    if(trim(mode)=='air5_invalid') spc(:,:,:,3)=-1.d0
#ifdef ASTR_AIR5_CHEMISTRY
    if(lcomb.and.trim(mode)=='air5') then
      call air5_diffusive_flux(rho(0,0,0),velocity,tmp(0,0,0),tve(0,0,0), &
        prs(0,0,0),spc(0,0,0,:),gradient,[0.d0,0.d0,0.d0], &
        [0.d0,0.d0,0.d0],zero_species_gradient, &
        stress,species_flux,energy_flux,ev_flux,status)
      if(status/=0) error stop 'reference transport failed'
    else
#endif
      mu=miucal(tmp(0,0,0))
      if(nondimen) mu=mu/reynolds
      divergence=(gradient(1,1)+gradient(2,2)+gradient(3,3))/3.d0
      stress=mu*(gradient+transpose(gradient))
      stress(1,1)=2.d0*mu*(gradient(1,1)-divergence)
      stress(2,2)=2.d0*mu*(gradient(2,2)-divergence)
      stress(3,3)=2.d0*mu*(gradient(3,3)-divergence)
#ifdef ASTR_AIR5_CHEMISTRY
    endif
#endif
    t=tmp(0,0,0)
    reference=[r,r*u,r*v,r*w,p,r*t,r*u*u,r*v*v,r*w*w,r*u*v,r*u*w,r*v*w, &
      p*p,r*t*t,r*u*t,r*v*t,r*w*t,r*u**3,r*v**3,r*w**3,r*u*u*v,r*u*u*w, &
      r*u*v*v,r*u*w*w,r*v*v*w,r*v*w*w,r*u*v*w,u,v,w,p*u,p*v,p*w, &
      stress(1,1),stress(2,2),stress(3,3),stress(1,2),stress(1,3),stress(2,3), &
      sum(stress*gradient),p*(gradient(1,1)+gradient(2,2)+gradient(3,3)),matmul(stress,velocity)]
    if(sample>1) expected=expected+0.5d0*(times(sample)-times(sample-1))*(previous_reference+reference)
    previous_reference=reference
    nstep=sample-1;time=times(sample)
    call meanflowcal()
    call meanflowcal()
#ifdef TEST_MEANFLOW_GPU
    if(trim(mode)=='air5') then
      rho_d=rho(0,0,0); prs_d=prs(0,0,0); tmp_d=tmp(0,0,0); tve_d=tve(0,0,0); dvel_d=dvel
      vel_d(:,:,:,1)=velocity(1); vel_d(:,:,:,2)=velocity(2); vel_d(:,:,:,3)=velocity(3)
      spc_d(:,:,:,1)=spc(0,0,0,1); spc_d(:,:,:,2)=spc(0,0,0,2); spc_d(:,:,:,3)=spc(0,0,0,3)
      spc_d(:,:,:,4)=spc(0,0,0,4); spc_d(:,:,:,5)=spc(0,0,0,5)
      call accumulate_air5_mean_statistics_gpu()
      call accumulate_air5_mean_statistics_gpu()
    endif
#endif
    if(nsamples/=sample) error stop 'duplicate sampling'
    call read_accumulated(actual)
    if(mean_time_weight/=times(sample)) error stop 'mean time coverage mismatch'
    if(maxval(abs(actual-expected)/max(1.d0,abs(expected)))>1.d-14) then
      write(*,'(A,ES24.16)') 'meanflow transport mismatch: ',maxval(abs(actual-expected))
      error stop 'meanflow transport mismatch'
    endif
    if(mean_time_weight>0.d0) call check_normalized()
    if(sample==1.and.len_trim(output_prefix)>0) then
      identity=checkpoint_state_identity(0_8,0.d0,0.d0,0.1d0)
      call complete_mean_statistics_file(trim(output_prefix)//'_cpu_seed.h5',.true.,identity,huge(0_8)/4096_8)
      call reset_mean_statistics_window()
      call complete_mean_statistics_file(trim(output_prefix)//'_cpu_seed.h5',.false.,identity,huge(0_8)/4096_8)
#ifdef TEST_MEANFLOW_GPU
      if(trim(mode)=='air5') then
        call complete_air5_mean_statistics_file(trim(output_prefix)//'_gpu_seed.h5',.true.,identity,huge(0_8)/4096_8)
      endif
#endif
    endif
  enddo
#ifdef TEST_MEANFLOW_GPU
  if(trim(mode)=='air5') then
    lavg=.false.
    call slice_air5_mean_info_gpu(mean_enabled,mean_covered,mean_count,mean_window,mean_duration)
    if(mean_enabled.or.mean_covered) error stop 'disabled AIR5 statistics exposed as active'
    lavg=.true.
    call slice_air5_mean_info_gpu(mean_enabled,mean_covered,mean_count,mean_window,mean_duration)
    if(.not.mean_enabled.or..not.mean_covered.or.mean_count/=3.or.mean_duration/=1.d0) &
      error stop 'AIR5 mean consumer clock mismatch'
  endif
#endif
  if(len_trim(output_prefix)>0) then
    identity=checkpoint_state_identity(3_8,1.d0,0.7d0,0.1d0)
    call complete_mean_statistics_file(trim(output_prefix)//'_cpu.h5',.true.,identity,huge(0_8)/4096_8)
#ifdef TEST_MEANFLOW_GPU
    if(trim(mode)=='air5') then
    call complete_air5_mean_statistics_file(trim(output_prefix)//'_gpu.h5',.true.,identity,huge(0_8)/4096_8)
    call release_air5_mean_statistics_gpu()
    call initialize_air5_mean_statistics_gpu(huge(0_8)/4096_8)
    call complete_air5_mean_statistics_file(trim(output_prefix)//'_gpu.h5',.false.,identity,huge(0_8)/4096_8)
    call set_air5_mean_sampling_gpu(.true.)
    endif
#endif
    call reset_mean_statistics_window()
    call complete_mean_statistics_file(trim(output_prefix)//'_cpu.h5',.false.,identity,huge(0_8)/4096_8)
    nstep=3;time=1.8d0
    expected=expected+(time-times(3))*reference
    call meanflowcal()
    call read_accumulated(actual)
    if(nsamples/=4.or.mean_time_weight/=time) error stop 'restored mean clock mismatch'
    if(maxval(abs(actual-expected)/max(1.d0,abs(expected)))>1.d-14) error stop 'restored mean endpoint mismatch'
    call check_normalized()
    identity=checkpoint_state_identity(4_8,time,0.8d0,0.1d0)
    call complete_mean_statistics_file(trim(output_prefix)//'_cpu_resumed.h5',.true.,identity,huge(0_8)/4096_8)
#ifdef TEST_MEANFLOW_GPU
    if(trim(mode)=='air5') then
      call accumulate_air5_mean_statistics_gpu()
      call complete_air5_mean_statistics_file(trim(output_prefix)//'_gpu_resumed.h5',.true.,identity,huge(0_8)/4096_8)
      call reset_air5_mean_statistics_gpu()
      call reset_mean_statistics_window()
      nstep=4;time=2.4d0
      call meanflowcal()
      call accumulate_air5_mean_statistics_gpu()
      call read_accumulated(actual)
      if(nsamples/=1.or.mean_time_weight/=0.d0.or.any(actual/=0.d0)) &
        error stop 'reopened mean window retained an old integral'
      identity=checkpoint_state_identity(4_8,time,0.6d0,0.1d0)
      call complete_mean_statistics_file(trim(output_prefix)//'_cpu_reset_seed.h5',.true.,identity,huge(0_8)/4096_8)
      call complete_air5_mean_statistics_file(trim(output_prefix)//'_gpu_reset_seed.h5',.true.,identity,huge(0_8)/4096_8)
      nstep=5;time=3.2d0
      expected=(time-2.4d0)*reference
      call meanflowcal()
      call accumulate_air5_mean_statistics_gpu()
      call read_accumulated(actual)
      if(maxval(abs(actual-expected)/max(1.d0,abs(expected)))>1.d-14) error stop 'reopened mean interval mismatch'
      call check_normalized()
      identity=checkpoint_state_identity(5_8,time,0.8d0,0.1d0)
      call complete_mean_statistics_file(trim(output_prefix)//'_cpu_reset_resumed.h5',.true.,identity,huge(0_8)/4096_8)
      call complete_air5_mean_statistics_file(trim(output_prefix)//'_gpu_reset_resumed.h5',.true.,identity,huge(0_8)/4096_8)
      call release_air5_mean_statistics_gpu()
    endif
#endif
  endif
  if(trim(legacy_action)=='write') then
    lio=mpirank==0;iomode='h'
    nstep=3;nsamples=4
    tu2=17.d0;tu3=-23.d0
    call writemeanflow()
  endif
  print*, 'MEANFLOW_TRANSPORT_PASS ',trim(mode)
  call MPI_Finalize(ierr)
contains
  subroutine check_normalized()
    real(real64) :: observed(5),wanted(5)
    observed=[slice_source_value_cpu(0,0,0,1,3),slice_source_value_cpu(0,0,0,2,3), &
      slice_source_value_cpu(0,0,0,2,4),slice_source_value_cpu(0,0,0,5,3),slice_source_value_cpu(0,0,0,6,4)]
    wanted=[expected(1)/mean_time_weight,expected(28)/mean_time_weight,expected(2)/expected(1), &
      expected(5)/mean_time_weight,expected(6)/expected(1)]
    if(maxval(abs(observed-wanted)/max(1.d0,abs(wanted)))>1.d-14) error stop 'mean source normalization mismatch'
  end subroutine
  subroutine read_accumulated(values)
    real(real64),intent(out) :: values(44)
    values=[rom(0,0,0),u1m(0,0,0),u2m(0,0,0),u3m(0,0,0),pm(0,0,0),tm(0,0,0), &
      u11(0,0,0),u22(0,0,0),u33(0,0,0),u12(0,0,0),u13(0,0,0),u23(0,0,0),pp(0,0,0),tt(0,0,0), &
      tu1(0,0,0),tu2(0,0,0),tu3(0,0,0),u111(0,0,0),u222(0,0,0),u333(0,0,0), &
      u112(0,0,0),u113(0,0,0),u122(0,0,0),u133(0,0,0),u223(0,0,0),u233(0,0,0),u123(0,0,0), &
      u1rem(0,0,0),u2rem(0,0,0),u3rem(0,0,0),pu1(0,0,0),pu2(0,0,0),pu3(0,0,0), &
      sgmam11(0,0,0),sgmam22(0,0,0),sgmam33(0,0,0),sgmam12(0,0,0),sgmam13(0,0,0),sgmam23(0,0,0), &
      disspa(0,0,0),predil(0,0,0),visdif1(0,0,0),visdif2(0,0,0),visdif3(0,0,0)]
  end subroutine
end program
