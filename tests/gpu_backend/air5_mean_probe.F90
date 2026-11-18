#ifdef ASTR_MEAN_REFERENCE_GPU
module commarray_gpu
  use cudafor
  implicit none
  real(8),allocatable,device :: rho_d(:,:,:),vel_d(:,:,:,:),prs_d(:,:,:),tmp_d(:,:,:), &
    tve_d(:,:,:),spc_d(:,:,:,:),dvel_d(:,:,:,:,:)
end module
module chemistry_mean_statistics_gpu
  use cudafor
  use iso_fortran_env,only:real64
  use mean_statistics_math,only:mean_point_moments,mean_endpoint_count,mean_moment_count
  implicit none
  real(8),allocatable,device :: accumulated_d(:,:,:,:),endpoint_d(:,:,:,:)
  integer,device :: failure_d(1)
contains
#include "frozen_mean_kernel.inc"
end module
#endif
program air5_mean_probe
  use iso_c_binding
  use, intrinsic :: ieee_arithmetic,only:ieee_is_finite
#ifdef ASTR_MEAN_REFERENCE_GPU
  use commarray_gpu
  use chemistry_mean_statistics_gpu
#elif defined(ASTR_MEAN_REFERENCE_CPU)
  use chemistry_transport,only:air5_transport_properties
  use mean_statistics_math,only:mean_point_moments
#else
  use device_state
  use chemistry_device_properties,only:air5_device_prepare,air5_device_check
  use chemistry_device_transport,only:air5_geometry_prepare,air5_transport_prepare
  use chemistry_device_coupling,only:air5_device_options,air5_chemistry_prepare
  use chemistry_device_statistics
#endif
  implicit none
  integer :: im,jm,km,frames,sample,unit,output,debug,resumed,status,i,j,k,c,d,field,n,index,physics
  real(c_double) :: interval
  real(c_double),allocatable :: input(:,:,:,:,:),intervals(:),sums(:,:,:,:),endpoint(:,:,:,:)
  character(1024) :: path,text
#ifndef ASTR_MEAN_REFERENCE_CPU
#ifndef ASTR_MEAN_REFERENCE_GPU
  type saved_field
    real(c_double),allocatable :: values(:)
  end type
  type(saved_field) :: before(10)
  integer,parameter :: observed(10)=[device_q,device_air5_carry,device_air5_history, &
    device_rho,device_velocity,device_pressure,device_temperature,device_air5_tv,device_air5_species,device_dvelocity]
  integer,parameter :: caches(6)=[device_rho,device_velocity,device_pressure,device_temperature,device_air5_tv,device_air5_species]
  integer,parameter :: offsets(6)=[1,2,5,6,7,8]
  type(c_ptr) :: handle
  type(device_view) :: view
  type(air5_device_options) :: options
  real(c_double),allocatable :: values(:)
  integer(c_int) :: code,solver_status
#endif
#endif
  call get_command_argument(1,path)
  open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='old')
  read(unit)im,jm,km,frames
  allocate(input(0:im,0:jm,0:km,32,frames),intervals(frames), &
    sums(0:im,0:jm,0:km,44),endpoint(0:im,0:jm,0:km,14))
  read(unit)intervals,input;close(unit)
  call get_command_argument(2,path)
  open(newunit=output,file=trim(path),access='stream',form='unformatted',status='new')
  write(output)im,jm,km,frames
  call get_command_argument(3,text);read(text,*)debug
  call get_command_argument(4,text);read(text,*)resumed
  sums=0;endpoint=0
#ifdef ASTR_MEAN_REFERENCE_GPU
  allocate(rho_d(0:im,0:jm,0:km),vel_d(0:im,0:jm,0:km,3),prs_d(0:im,0:jm,0:km), &
    tmp_d(0:im,0:jm,0:km),tve_d(0:im,0:jm,0:km),spc_d(0:im,0:jm,0:km,5), &
    dvel_d(0:im,0:jm,0:km,3,3),accumulated_d(0:im,0:jm,0:km,44),endpoint_d(0:im,0:jm,0:km,14))
  accumulated_d=0;endpoint_d=0
#elif !defined(ASTR_MEAN_REFERENCE_CPU)
  call create_state()
#endif
  do sample=1,frames
    interval=intervals(sample)
#ifdef ASTR_MEAN_REFERENCE_GPU
    rho_d=input(:,:,:,1,sample);vel_d=input(:,:,:,2:4,sample);prs_d=input(:,:,:,5,sample)
    tmp_d=input(:,:,:,6,sample);tve_d=input(:,:,:,7,sample);spc_d=input(:,:,:,8:12,sample)
    do d=1,3;do c=1,3
      dvel_d(:,:,:,c,d)=input(:,:,:,12+c+3*(d-1),sample)
    enddo;enddo
    failure_d=0
    call accumulate_mean_kernel<<<((im+1)*(jm+1)*(km+1)+255)/256,256>>>(im,jm,km,interval,sample>1)
    status=cudaDeviceSynchronize();if(status/=0)error stop 'original mean kernel CUDA error'
    physics=failure_d(1);sums=accumulated_d;endpoint=endpoint_d
#elif defined(ASTR_MEAN_REFERENCE_CPU)
    call sample_cpu()
#else
    if(sample==4.and.resumed==1)then
      call check(device_state_destroy(handle));call create_state()
      call check(device_state_upload(handle,device_air5_mean_sum,sums,int(size(sums),c_size_t)*8_c_size_t))
      call check(device_state_upload(handle,device_air5_mean_endpoint,endpoint,int(size(endpoint),c_size_t)*8_c_size_t))
    endif
    call upload_input()
    do field=1,size(observed)
      call check(device_state_view(handle,int(observed(field),c_int),view));n=product(view%extent)
      if(.not.allocated(before(field)%values))allocate(before(field)%values(n))
      call check(device_state_download(handle,int(observed(field),c_int),before(field)%values,int(n,c_size_t)*8_c_size_t))
    enddo
    call check(air5_mean_sample(handle,interval,merge(1_c_int,0_c_int,sample>1)))
    call check(air5_mean_check(handle,code));physics=code
    call check(air5_device_check(handle,solver_status))
    if(solver_status/=0)error stop 'mean sampling changed flow status'
    do field=1,size(observed)
      n=size(before(field)%values);allocate(values(n))
      call check(device_state_download(handle,int(observed(field),c_int),values,int(n,c_size_t)*8_c_size_t))
      if(any(transfer(values,[0_c_int64_t],n)/=transfer(before(field)%values,[0_c_int64_t],n))) &
        error stop 'mean sampling changed solver field words'
      deallocate(values)
    enddo
    call check(device_state_download(handle,device_air5_mean_sum,sums,int(size(sums),c_size_t)*8_c_size_t))
    call check(device_state_download(handle,device_air5_mean_endpoint,endpoint,int(size(endpoint),c_size_t)*8_c_size_t))
#endif
    write(output)physics,sums,endpoint
  enddo
  close(output)
#if !defined(ASTR_MEAN_REFERENCE_CPU) && !defined(ASTR_MEAN_REFERENCE_GPU)
  call check(air5_mean_reset(handle))
  call check(device_state_download(handle,device_air5_mean_sum,sums,int(size(sums),c_size_t)*8_c_size_t))
  call check(device_state_download(handle,device_air5_mean_endpoint,endpoint,int(size(endpoint),c_size_t)*8_c_size_t))
  if(any(transfer(sums,[0_c_int64_t],size(sums))/=0).or.any(transfer(endpoint,[0_c_int64_t],size(endpoint))/=0)) &
    error stop 'mean reset did not produce positive zero buffers'
  call check(air5_mean_check(handle,code));if(code/=0)error stop 'mean reset status'
  call check(device_state_destroy(handle))
  write(*,'(a)')'AIR5_MEAN_PASS fields_unchanged=10 reset=exact optional_allocation=checked'
#endif
#ifndef ASTR_MEAN_REFERENCE_GPU
contains
#endif
#if !defined(ASTR_MEAN_REFERENCE_CPU) && !defined(ASTR_MEAN_REFERENCE_GPU)
  subroutine check(result)
    integer,intent(in) :: result
    if(result/=0)then
      write(*,*)'mean device API failure',result
      error stop
    endif
  end subroutine
  subroutine create_state()
    call check(device_state_create(0_c_int,int(im,c_int),int(jm,c_int),int(km,c_int),3_c_int,int(debug,c_int),handle))
    call check(air5_device_prepare(handle))
    code=device_state_view(handle,device_air5_mean_sum,view)
    if(code/=device_not_allocated)error stop 'mean array allocated without request'
    options=air5_device_options(1_c_int,1_c_int,0_c_int,0_c_int,1_c_int,32_c_int,0_c_int, &
      1e-9_c_double,1e-13_c_double,0._c_double)
    call check(air5_chemistry_prepare(handle,options))
    call check(air5_geometry_prepare(handle));call check(air5_transport_prepare(handle,0_c_int))
    call check(air5_mean_prepare(handle))
    call check(device_state_view(handle,device_air5_mean_sum,view))
    if(any(view%extent/=[im+1,jm+1,km+1,44,1]).or.any(view%lower/=[0,0,0,1,1]).or.view%element_bytes/=8) &
      error stop 'mean SUM44 descriptor'
    call check(device_state_view(handle,device_air5_mean_endpoint,view))
    if(any(view%extent/=[im+1,jm+1,km+1,14,1]).or.view%element_bytes/=8)error stop 'mean ENDPOINT14 descriptor'
    call check(device_state_view(handle,device_air5_carry,view));allocate(values(product(view%extent)));values=1e-30_c_double
    call check(device_state_upload(handle,device_air5_carry,values,int(size(values),c_size_t)*8_c_size_t));deallocate(values)
    call check(device_state_view(handle,device_air5_history,view));allocate(values(product(view%extent)));values=1e-7_c_double
    call check(device_state_upload(handle,device_air5_history,values,int(size(values),c_size_t)*8_c_size_t));deallocate(values)
  end subroutine
  subroutine upload_input()
    do field=1,size(caches)
      call check(device_state_view(handle,int(caches(field),c_int),view));allocate(values(product(view%extent)));values=-777
      do c=1,view%extent(4);do k=0,km;do j=0,jm;do i=0,im
        index=1+int(sum(([i,j,k]-view%lower(1:3))*view%stride(1:3))+(c-1)*view%stride(4))
        values(index)=input(i,j,k,offsets(field)+c-1,sample)
      enddo;enddo;enddo;enddo
      call check(device_state_upload(handle,int(caches(field),c_int),values,int(size(values),c_size_t)*8_c_size_t));deallocate(values)
    enddo
    call check(device_state_view(handle,device_q,view));allocate(values(product(view%extent)));values=-321
    do c=1,11;do k=0,km;do j=0,jm;do i=0,im
      index=1+int(sum(([i,j,k]-view%lower(1:3))*view%stride(1:3))+(c-1)*view%stride(4))
      values(index)=input(i,j,k,21+c,sample)
    enddo;enddo;enddo;enddo
    call check(device_state_upload(handle,device_q,values,int(size(values),c_size_t)*8_c_size_t));deallocate(values)
    call check(device_state_view(handle,device_dvelocity,view));allocate(values(product(view%extent)))
    do d=1,3;do c=1,3;do k=0,km;do j=0,jm;do i=0,im
      index=1+int(sum(([i,j,k]-view%lower(1:3))*view%stride(1:3))+(c-1)*view%stride(4)+(d-1)*view%stride(5))
      values(index)=input(i,j,k,12+c+3*(d-1),sample)
    enddo;enddo;enddo;enddo;enddo
    call check(device_state_upload(handle,device_dvelocity,values,int(size(values),c_size_t)*8_c_size_t));deallocate(values)
  end subroutine
#elif defined(ASTR_MEAN_REFERENCE_CPU)
  subroutine sample_cpu()
    real(c_double) :: mu,ktr,kv,mus(5),binary(5,5),mixture(5),g(3,3),s(3,3),div,point(14),current(44),old(44),increment(44)
    integer :: result
    physics=0
    do k=0,km;do j=0,jm;do i=0,im
      call air5_transport_properties(input(i,j,k,6,sample),input(i,j,k,7,sample),input(i,j,k,5,sample), &
        input(i,j,k,8:12,sample),mu,ktr,kv,mus,binary,mixture,result)
      if(result/=0)then
        physics=max(physics,result);cycle
      endif
      g=reshape(input(i,j,k,13:21,sample),[3,3]);div=(1._c_double/3._c_double)*(g(1,1)+g(2,2)+g(3,3))
      s=mu*(g+transpose(g))
      do c=1,3
        s(c,c)=2._c_double*mu*(g(c,c)-div)
      enddo
      point=[input(i,j,k,1:6,sample),s(1,1),s(2,2),s(3,3),s(1,2),s(1,3),s(2,3), &
        sum(s*g),input(i,j,k,5,sample)*(g(1,1)+g(2,2)+g(3,3))]
      call mean_point_moments(point,current);increment=0
      if(sample>1)then
        call mean_point_moments(endpoint(i,j,k,:),old)
        increment=(.5_c_double*interval)*(old+current)
      endif
      if(any(.not.ieee_is_finite(current)).or.any(.not.ieee_is_finite(sums(i,j,k,:)+increment)))then
        physics=max(physics,1);cycle
      endif
      sums(i,j,k,:)=sums(i,j,k,:)+increment;endpoint(i,j,k,:)=point
    enddo;enddo;enddo
  end subroutine
#endif
end program
