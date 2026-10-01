program meanflow_air5_transport_probe
  use iso_fortran_env, only: real64
  use mpi
  use commvar, only: im,jm,km,nstep,time,nondimen,lcomb,numq,num_species, &
    num_modequ,reynolds,tempconst,tempconst1,lreport,ltimrpt
  use commarray, only: rho,vel,prs,tmp,tve,spc,dvel
  use parallel, only: mpirank,lio
  use statistic, only: meanflowcal,nsamples,sgmam11,sgmam22,sgmam33, &
    sgmam12,sgmam13,sgmam23,visdif1,visdif2,visdif3,disspa
  use chemistry_transport, only: air5_diffusive_flux
  use fludyna, only: miucal
#ifdef TEST_MEANFLOW_GPU
  use commvar, only: ia,ja,ka,hm,feqavg,diffterm,lavg
  use parallel, only: ig0,jg0,kg0
  use commarray_gpu, only: rho_d,vel_d,prs_d,tmp_d,tve_d,spc_d,dvel_d
  use chemistry_mean_statistics_gpu, only: initialize_air5_mean_statistics_gpu, &
    release_air5_mean_statistics_gpu,set_air5_mean_sampling_gpu, &
    accumulate_air5_mean_statistics_gpu,complete_air5_mean_statistics_file
  use statistic, only: complete_mean_statistics_file
  use checkpoint_state_io, only: checkpoint_state_identity
#endif
  implicit none
  character(32) :: mode
  integer :: ierr,sample,status
  real(real64) :: gradient(3,3),stress(3,3),expected(10),actual(10),mu,divergence
  real(real64) :: species_flux(5,3),energy_flux(3),ev_flux(3),velocity(3)
  real(real64) :: zero_species_gradient(5,3)
#ifdef TEST_MEANFLOW_GPU
  character(1024) :: output_prefix
  integer :: ranks
  type(checkpoint_state_identity) :: identity
#endif
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,mpirank,ierr)
  lio=.false.
  call get_command_argument(1,mode)
  im=1; jm=1; km=1; numq=11; num_species=5; num_modequ=1
  nondimen=trim(mode)=='perfect_nondim'.or.trim(mode)=='air5_nondim'
  lcomb=index(trim(mode),'air5')==1
  reynolds=7.d0; tempconst=0.4d0; tempconst1=1.4d0
  lreport=.false.; ltimrpt=.false.; nsamples=0
  allocate(rho(0:1,0:1,0:1),vel(0:1,0:1,0:1,3),prs(0:1,0:1,0:1), &
    tmp(0:1,0:1,0:1),tve(0:1,0:1,0:1),spc(0:1,0:1,0:1,5),dvel(0:1,0:1,0:1,3,3))
  gradient=reshape([1.d0,2.d0,-3.d0,4.d0,-5.d0,6.d0,7.d0,-8.d0,9.d0],[3,3])
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
    call get_command_argument(2,output_prefix)
    if(len_trim(output_prefix)==0) error stop 'missing probe output prefix'
    call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
    ia=ranks; ja=1; ka=1; feqavg=1; diffterm=.true.; lavg=.true.
    ig0=mpirank; jg0=0; kg0=0
    allocate(rho_d(-hm:1+hm,-hm:1+hm,-hm:1+hm),vel_d(-hm:1+hm,-hm:1+hm,-hm:1+hm,3), &
      prs_d(-hm:1+hm,-hm:1+hm,-hm:1+hm),tmp_d(-hm:1+hm,-hm:1+hm,-hm:1+hm), &
      tve_d(-hm:1+hm,-hm:1+hm,-hm:1+hm),spc_d(-hm:1+hm,-hm:1+hm,-hm:1+hm,5), &
      dvel_d(0:1,0:1,0:1,3,3))
    call initialize_air5_mean_statistics_gpu(67108864_8)
    call set_air5_mean_sampling_gpu(.true.)
  endif
#endif
  do sample=1,2
    tmp=3000.d0+500.d0*sample
    if(nondimen) tmp=1.d0+0.25d0*sample
    spc(:,:,:,1)=0.75d0-0.1d0*sample
    spc(:,:,:,2)=0.20d0; spc(:,:,:,3)=0.03d0
    spc(:,:,:,4)=0.01d0; spc(:,:,:,5)=0.01d0+0.1d0*sample
    if(trim(mode)=='air5_invalid') spc(:,:,:,3)=-1.d0
    if(lcomb.and.trim(mode)=='air5') then
      call air5_diffusive_flux(rho(0,0,0),velocity,tmp(0,0,0),tve(0,0,0), &
        prs(0,0,0),spc(0,0,0,:),gradient,[0.d0,0.d0,0.d0], &
        [0.d0,0.d0,0.d0],zero_species_gradient, &
        stress,species_flux,energy_flux,ev_flux,status)
      if(status/=0) error stop 'reference transport failed'
    else
      mu=miucal(tmp(0,0,0))
      if(nondimen) mu=mu/reynolds
      divergence=(gradient(1,1)+gradient(2,2)+gradient(3,3))/3.d0
      stress=mu*(gradient+transpose(gradient))
      stress(1,1)=2.d0*mu*(gradient(1,1)-divergence)
      stress(2,2)=2.d0*mu*(gradient(2,2)-divergence)
      stress(3,3)=2.d0*mu*(gradient(3,3)-divergence)
    endif
    expected=expected+[stress(1,1),stress(2,2),stress(3,3), &
      stress(1,2),stress(1,3),stress(2,3),matmul(stress,velocity),sum(stress*gradient)]
    nstep=sample; time=sample*1.d-10
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
    actual=[sgmam11(0,0,0),sgmam22(0,0,0),sgmam33(0,0,0), &
      sgmam12(0,0,0),sgmam13(0,0,0),sgmam23(0,0,0), &
      visdif1(0,0,0),visdif2(0,0,0),visdif3(0,0,0),disspa(0,0,0)]
    if(maxval(abs(actual-expected))>1.d-14*max(1.d0,maxval(abs(expected)))) then
      write(*,'(A,ES24.16)') 'meanflow transport mismatch: ',maxval(abs(actual-expected))
      error stop 'meanflow transport mismatch'
    endif
  enddo
#ifdef TEST_MEANFLOW_GPU
  if(trim(mode)=='air5') then
    identity=checkpoint_state_identity(3_8,3.d-10,1.d-10,1.d-10)
    call complete_mean_statistics_file(trim(output_prefix)//'_cpu.h5',.true.,identity,67108864_8)
    call complete_air5_mean_statistics_file(trim(output_prefix)//'_gpu.h5',.true.,identity,67108864_8)
    call release_air5_mean_statistics_gpu()
  endif
#endif
  print*, 'MEANFLOW_TRANSPORT_PASS ',trim(mode)
  call MPI_Finalize(ierr)
end program
