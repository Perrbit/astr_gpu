program air5_mean_runtime_probe
  use iso_fortran_env,only:int64,real64
  use iso_c_binding,only:c_ptr,c_int,c_size_t
  use mpi
  use commvar
  use parallel
  use statistic,only:nsamples
  use checkpoint_state_io,only:checkpoint_state_identity
#ifdef ASTR_MEAN_RUNTIME_REFERENCE
  use cudafor
  use commarray_gpu
  use chemistry_mean_statistics_gpu,only:initialize_mean=>initialize_air5_mean_statistics_gpu, &
    release_mean=>release_air5_mean_statistics_gpu,set_sampling=>set_air5_mean_sampling_gpu, &
    accumulate_mean=>accumulate_air5_mean_statistics_gpu,reset_mean=>reset_air5_mean_statistics_gpu, &
    complete_file=>complete_air5_mean_statistics_file,mean_info=>slice_air5_mean_info_gpu, &
    mean_bytes=>air5_mean_statistics_device_bytes
#else
  use device_state
  use chemistry_device_properties,only:air5_device_prepare
  use chemistry_device_transport,only:air5_geometry_prepare,air5_transport_prepare
  use chemistry_mean_statistics_cpp,only:initialize_mean=>initialize_air5_mean_statistics_cpp, &
    release_mean=>release_air5_mean_statistics_cpp,set_sampling=>set_air5_mean_sampling_cpp, &
    accumulate_mean=>accumulate_air5_mean_statistics_cpp,reset_mean=>reset_air5_mean_statistics_cpp, &
    complete_file=>complete_air5_mean_statistics_cpp_file,mean_info=>slice_air5_mean_info_cpp, &
    mean_bytes=>air5_mean_statistics_cpp_device_bytes
#endif
  implicit none
  type(checkpoint_state_identity) :: identity
  real(real64),allocatable :: input(:,:,:,:,:),times(:)
  integer(int64),allocatable :: steps(:)
  integer(int64) :: count,old_count
  real(real64) :: window(2),duration,old_window(2),old_duration
  integer :: rank,ranks,topology(3),coord(3),cart,i,j,k,c,d,n,index,field,frame,frames,unit,ierr,status,device
  logical :: periods(3),enabled,covered
  character(1024) :: inputs,outputs,path,text,mode,source
#ifndef ASTR_MEAN_RUNTIME_REFERENCE
  type(c_ptr) :: context
  type(device_view) :: view
  real(real64),allocatable :: values(:)
  integer,parameter :: caches(6)=[device_rho,device_velocity,device_pressure,device_temperature,device_air5_tv,device_air5_species]
  integer,parameter :: offsets(6)=[1,2,5,6,7,8]
#endif
  call get_command_argument(1,inputs);call get_command_argument(2,outputs)
  do i=1,3
    call get_command_argument(i+2,text);read(text,*)topology(i)
  enddo
  call get_command_argument(6,mode);call get_command_argument(7,source)
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text,status=status)
  if(status/=0)error stop 'local rank unavailable'
  read(text,*)rank;device=modulo(rank,2)
#ifdef ASTR_MEAN_RUNTIME_REFERENCE
  status=cudaSetDevice(device);if(status/=0)error stop 'reference GPU binding'
#else
  call check(device_bind(int(device,c_int)))
#endif
  call MPI_Init(ierr);call check(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call check(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr);call check(ierr)
  if(product(topology)/=ranks)error stop 'statistics topology'
  periods=.false.
  call MPI_Cart_create(MPI_COMM_WORLD,3,topology,periods,.false.,cart,ierr);call check(ierr)
  call MPI_Cart_coords(cart,rank,3,coord,ierr);call check(ierr)
  write(path,'(a,"/input.rank",i4.4,".bin")')trim(inputs),rank
  open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='old')
  read(unit)im,jm,km,frames
  allocate(input(0:im,0:jm,0:km,32,frames),times(frames),steps(frames))
  read(unit)times,steps,input;close(unit)
  ia=im*topology(1);ja=jm*topology(2);ka=km*topology(3)
  ig0=im*coord(1);jg0=jm*coord(2);kg0=km*coord(3)
  nstep=0;time=0;lavg=.true.
  call start()
  if(mean_bytes()/=int(im+1,int64)*(jm+1)*(km+1)*464+4)error stop 'active statistics bytes'
  identity=checkpoint_state_identity(0_int64,0.d0,0.d0,1d-8)
  call complete_file(trim(outputs)//'/empty.h5',.true.,identity,huge(0_int64)/4096)
  call complete_file(trim(outputs)//'/empty.h5',.false.,identity,huge(0_int64)/4096)
  call inspect(0_int64,0.d0,.false.)
  call set_sampling(.true.)
  frame=1
  if(trim(mode)=='restore-old')then
    nstep=steps(3);time=times(3)
    identity=checkpoint_state_identity(nstep,time,1d-8,1.1d-8)
    call complete_file(trim(source)//'/seed.h5',.false.,identity,huge(0_int64)/4096)
    call inspect(3_int64,times(3)-times(1),.true.)
    call accumulate_mean()
    if(nsamples/=3)error stop 'restored duplicate step was sampled'
    frame=4
  endif
  do while(frame<=frames)
    nstep=steps(frame);time=times(frame)
    call upload_input()
    call accumulate_mean();call accumulate_mean()
    if(nsamples/=frame)error stop 'sample duplicated or missed'
    call mean_info(enabled,covered,count,window,duration)
    if(count/=frame.or.any(window/=[times(1),times(frame)]))error stop 'sample clock identity'
    if(covered.neqv.(frame>1))error stop 'sample coverage flag'
    identity=checkpoint_state_identity(nstep,time,1d-8,1.1d-8)
    if(frame==1)call complete_file(trim(outputs)//'/first.h5',.true.,identity,huge(0_int64)/4096)
    if(frame==2)then
      old_count=count;old_window=window;old_duration=duration
      nstep=steps(2)+1;time=.5d0*(times(2)+times(3))
      call set_sampling(.false.);call accumulate_mean()
      call set_sampling(.true.);lavg=.false.;call accumulate_mean();lavg=.true.
      call mean_info(enabled,covered,count,window,duration)
      if(count/=old_count.or.any(window/=old_window).or.duration/=old_duration)error stop 'disabled sampling advanced clock'
    endif
    if(frame==3)then
      call complete_file(trim(outputs)//'/seed.h5',.true.,identity,huge(0_int64)/4096)
      if(trim(mode)=='resume')then
        call finish();call start();call set_sampling(.true.)
        call complete_file(trim(outputs)//'/seed.h5',.false.,identity,huge(0_int64)/4096)
        call inspect(3_int64,times(3)-times(1),.true.)
        call accumulate_mean()
        if(nsamples/=3)error stop 'same-backend restored duplicate step'
      endif
    endif
    frame=frame+1
  enddo
  call complete_file(trim(outputs)//'/final.h5',.true.,identity,huge(0_int64)/4096)
  call reset_mean()
  if(nsamples/=0)error stop 'new window did not reset sample count'
  call complete_file(trim(outputs)//'/reset_empty.h5',.true.,identity,huge(0_int64)/4096)
  nstep=steps(frames)+2;time=times(frames)+2.5d-7
  call accumulate_mean();call accumulate_mean()
  call inspect(1_int64,0.d0,.false.)
  identity=checkpoint_state_identity(nstep,time,1d-8,1.1d-8)
  call complete_file(trim(outputs)//'/reset_first.h5',.true.,identity,huge(0_int64)/4096)
  nstep=nstep+1;time=time+1.7d-7;call accumulate_mean()
  identity=checkpoint_state_identity(nstep,time,1d-8,1.1d-8)
  call complete_file(trim(outputs)//'/reset_final.h5',.true.,identity,huge(0_int64)/4096)
  call finish()
  if(mean_bytes()/=0)error stop 'released statistics bytes'
  write(*,'(a,i0,a)')'AIR5_MEAN_RUNTIME_PASS rank=',rank,' samples=6 clock=checked duplicate=skipped reset=checked'
  call MPI_Comm_free(cart,ierr);call check(ierr)
  call MPI_Finalize(ierr)
contains
  subroutine check(code)
    integer,intent(in) :: code
    if(code/=0)then
      write(*,*)'statistics runtime device/MPI error',code
      error stop
    endif
  end subroutine
  subroutine inspect(expected_count,expected_duration,expected_coverage)
    integer(int64),intent(in) :: expected_count
    real(real64),intent(in) :: expected_duration
    logical,intent(in) :: expected_coverage
    call mean_info(enabled,covered,count,window,duration)
    if(.not.enabled.or.(covered.neqv.expected_coverage).or.count/=expected_count.or.duration/=expected_duration) &
      error stop 'statistics info/coverage mismatch'
  end subroutine
  subroutine start()
#ifdef ASTR_MEAN_RUNTIME_REFERENCE
    if(.not.allocated(rho_d))then
      allocate(rho_d(0:im,0:jm,0:km),vel_d(0:im,0:jm,0:km,3),prs_d(0:im,0:jm,0:km), &
        tmp_d(0:im,0:jm,0:km),tve_d(0:im,0:jm,0:km),spc_d(0:im,0:jm,0:km,5),dvel_d(0:im,0:jm,0:km,3,3))
    endif
    call initialize_mean(huge(0_int64)/4096)
#else
    call check(device_state_create(int(device,c_int),int(im,c_int),int(jm,c_int),int(km,c_int),3_c_int,1_c_int,context))
    call check(air5_device_prepare(context));call check(air5_geometry_prepare(context))
    call check(air5_transport_prepare(context,0_c_int));call initialize_mean(context,huge(0_int64)/4096)
#endif
  end subroutine
  subroutine finish()
    call release_mean()
#ifndef ASTR_MEAN_RUNTIME_REFERENCE
    status=device_state_view(context,device_air5_mean_sum,view)
    if(status/=device_not_allocated)error stop 'release left a mean allocation'
    call check(device_state_destroy(context))
#endif
  end subroutine
  subroutine upload_input()
#ifdef ASTR_MEAN_RUNTIME_REFERENCE
    rho_d=input(:,:,:,1,frame);vel_d=input(:,:,:,2:4,frame);prs_d=input(:,:,:,5,frame)
    tmp_d=input(:,:,:,6,frame);tve_d=input(:,:,:,7,frame);spc_d=input(:,:,:,8:12,frame)
    do d=1,3;do c=1,3
      dvel_d(:,:,:,c,d)=input(:,:,:,12+c+3*(d-1),frame)
    enddo;enddo
#else
    do field=1,size(caches)
      call check(device_state_view(context,int(caches(field),c_int),view));allocate(values(product(view%extent)));values=-777
      do c=1,view%extent(4);do k=0,km;do j=0,jm;do i=0,im
        index=1+int(sum(([i,j,k]-view%lower(1:3))*view%stride(1:3))+(c-1)*view%stride(4))
        values(index)=input(i,j,k,offsets(field)+c-1,frame)
      enddo;enddo;enddo;enddo
      call check(device_state_upload(context,int(caches(field),c_int),values,int(size(values),c_size_t)*8_c_size_t));deallocate(values)
    enddo
    call check(device_state_view(context,device_dvelocity,view));allocate(values(product(view%extent)))
    do d=1,3;do c=1,3;do k=0,km;do j=0,jm;do i=0,im
      index=1+int(sum(([i,j,k]-view%lower(1:3))*view%stride(1:3))+(c-1)*view%stride(4)+(d-1)*view%stride(5))
      values(index)=input(i,j,k,12+c+3*(d-1),frame)
    enddo;enddo;enddo;enddo;enddo
    call check(device_state_upload(context,device_dvelocity,values,int(size(values),c_size_t)*8_c_size_t));deallocate(values)
#endif
  end subroutine
end program
