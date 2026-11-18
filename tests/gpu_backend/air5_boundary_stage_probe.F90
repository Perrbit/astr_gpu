#ifdef ASTR_AIR5_STAGE_REFERENCE
module commvar
  implicit none
  integer :: im,jm,km,hm
  character(32) :: flowtype='air5hbl'
end module
module parallel
  implicit none
  integer :: mpileft,mpiright,mpidown,mpiup
end module
module gpu_check
  use cudafor
  implicit none
contains
  subroutine sync_after_kernel(label,force)
    character(*),intent(in) :: label
    logical,intent(in) :: force
    integer :: status
    status=cudaDeviceSynchronize()
    if(status/=0)then
      write(*,*)trim(label),status
      error stop 'original boundary synchronization'
    endif
  end subroutine
end module
module chemistry_compensation_gpu
  use cudafor
  use iso_fortran_env,only:real64
  implicit none
  logical :: air5_compensated
  real(real64),device,allocatable :: air5_carry_d(:,:,:,:)
end module
module chemistry_hbl_boundary
  implicit none
  logical :: dynamic_top
contains
  logical function air5_dynamic_top()
    air5_dynamic_top=dynamic_top
  end function
end module
module commarray_gpu
  use cudafor
  use iso_fortran_env,only:real64
  implicit none
  real(real64),device,allocatable :: q_d(:,:,:,:),x_d(:,:,:,:)
end module
module chemistry_hbl_boundary_gpu
  use cudafor
  use iso_fortran_env,only:real64
  use chemistry_air5_data
  use chemistry_model
  use chemistry_state_layout
  use chemistry_flow_state_gpu
  implicit none
  real(real64),device,allocatable :: inlet_q_d(:,:),cache_d(:,:,:,:)
  real(real64),device :: farfield_q_d(11),wall_temperature_d,hbl_x_origin_d,incident_x_d,incident_q_d(11,2)
  logical,device :: incident_top_d,characteristic_top_d
  integer,device :: hbl_status_d,cache_status_d
  logical :: boundary_configured_gpu=.true.
contains
#include "frozen_hbl_functions.inc"
  attributes(global) subroutine extract_cache(im,jm,km)
    use commarray_gpu,only:q_d
    integer,value :: im,jm,km
    integer :: id,i,j,k,status,ignored
    real(real64) :: state(11),rho,u(3),t,y(5),tv,p
    id=(blockIdx%x-1)*blockDim%x+threadIdx%x-1
    if(id>=(im+1)*(jm+1)*(km+1))return
    i=mod(id,im+1);j=mod(id/(im+1),jm+1);k=id/((im+1)*(jm+1))
    state=q_d(i,j,k,:)
    call air5_conservative_to_primitive_gpu(state,rho,u,t,y,tv,p,status)
    if(status/=0)then
      ignored=atomicMax(cache_status_d,status)
      return
    endif
    cache_d(i,j,k,:)=[rho,u,t,tv,p,y]
  end subroutine
end module
module chemistry_postshock_boundary_gpu
  use cudafor
  use iso_fortran_env,only:real64
  use chemistry_state_layout,only:air5_num_conservative
  implicit none
  real(real64),device :: left_q_d(11),right_q_d(11)
  integer,device :: outlet_status_d
  real(real64) :: outlet_pressure
  logical :: boundary_configured_gpu=.true.
contains
#include "frozen_postshock_functions.inc"
end module
#endif
program air5_boundary_stage_probe
  use iso_c_binding
  use iso_fortran_env,only:real64,int64
  use mpi
#ifdef ASTR_AIR5_STAGE_REFERENCE
  use cudafor
  use commvar
  use parallel
  use commarray_gpu
  use chemistry_compensation_gpu
  use chemistry_hbl_boundary
  use chemistry_hbl_boundary_gpu,only:configure_air5_species_cache_gpu,inlet_q_d,cache_d,farfield_q_d,wall_temperature_d, &
    hbl_x_origin_d,incident_top_d,characteristic_top_d,incident_x_d,incident_q_d,cache_status_d,extract_cache,apply_air5_hbl_boundary_gpu
  use chemistry_postshock_boundary_gpu,only:left_q_d,right_q_d,outlet_pressure,apply_air5_postshock_boundary_gpu
#else
  use device_state
  use chemistry_device_properties,only:air5_device_prepare,air5_device_primitives,air5_device_check
  use chemistry_device_coupling,only:air5_device_options,air5_chemistry_prepare
  use chemistry_device_boundary
#endif
  implicit none
#ifndef ASTR_AIR5_STAGE_REFERENCE
  integer :: im,jm,km,hm
  type(c_ptr) :: handle
  type(device_view) :: view
  type(air5_device_options) :: options
  real(real64),allocatable :: flat(:),q_check(:,:,:,:)
  integer,parameter :: cache_fields(6)=[device_rho,device_velocity,device_temperature,device_air5_tv,device_pressure,device_air5_species]
  integer,parameter :: components(6)=[1,3,1,1,1,5]
  integer :: field,first_component,index
  integer(c_int) :: physics,failed_stage
#endif
  real(real64),allocatable :: initial(:,:,:,:),coords(:,:,:,:),profile(:,:),initial_carry(:,:,:,:), &
    after(:,:,:,:),carry(:,:,:,:),cache(:,:,:,:)
  real(real64) :: farfield(11),incident(11,2),wall_temperature,origin,incident_x,pressure_target,outlet_target
  integer :: ncases,unit,output,case_id,i,j,k,c,status,ierr,rank,ranks,cart,topology(3),coord(3), &
    low_peer(3),high_peer(3),local_rank,devices,device,mode,preserve,compensated,debug
  integer,allocatable :: cases(:,:)
  integer(c_int) :: lower(3),upper(3)
  logical :: periods(3)
  character(1024) :: input_dir,output_dir,path,text

  call get_command_argument(1,input_dir);call get_command_argument(2,output_dir)
  do i=1,3
    call get_command_argument(i+2,text);read(text,*)topology(i)
  enddo
  call get_command_argument(6,text);read(text,*)devices
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text,status=status)
  if(status/=0)error stop 'local rank missing'
  read(text,*)local_rank
  device=modulo(local_rank,devices)
#ifdef ASTR_AIR5_STAGE_REFERENCE
  status=cudaSetDevice(device);if(status/=0)error stop 'original stage device binding'
#else
  call check(device_bind(int(device,c_int)))
#endif
  call MPI_Init(ierr);call check(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call check(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr);call check(ierr)
  call require(product(topology)==ranks,'stage topology')
  periods=[.false.,.false.,.true.]
  call MPI_Cart_create(MPI_COMM_WORLD,3,topology,periods,.false.,cart,ierr);call check(ierr)
  call MPI_Cart_coords(cart,rank,3,coord,ierr);call check(ierr)
  do i=1,3
    call MPI_Cart_shift(cart,i-1,1,low_peer(i),high_peer(i),ierr);call check(ierr)
  enddo
  lower=int(merge(1,0,low_peer==MPI_PROC_NULL),c_int);upper=int(merge(1,0,high_peer==MPI_PROC_NULL),c_int)
#ifdef ASTR_AIR5_STAGE_REFERENCE
  mpileft=low_peer(1);mpiright=high_peer(1);mpidown=low_peer(2);mpiup=high_peer(2)
#endif
  write(path,'(A,A,I4.4,A)')trim(input_dir),'/input.rank',rank,'.bin'
  open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='old')
  read(unit)im,jm,km,hm,ncases
  allocate(initial(-hm:im+hm,-hm:jm+hm,-hm:km+hm,11),after(-hm:im+hm,-hm:jm+hm,-hm:km+hm,11), &
    coords(0:im,0:jm,0:km,3),profile(0:jm,11),initial_carry(0:im,0:jm,0:km,11), &
    carry(0:im,0:jm,0:km,11),cache(0:im,0:jm,0:km,12),cases(4,ncases))
  read(unit)initial,coords,profile,farfield,incident,wall_temperature,origin,incident_x,pressure_target,initial_carry,cases;close(unit)
#ifndef ASTR_AIR5_STAGE_REFERENCE
  allocate(q_check(-hm:im+hm,-hm:jm+hm,-hm:km+hm,11))
#endif
  write(path,'(A,A,I4.4,A)')trim(output_dir),'/rank',rank,'.bin'
  open(newunit=output,file=trim(path),access='stream',form='unformatted',status='new')
  write(output)im,jm,km,hm,ncases,lower,upper
  do case_id=1,ncases
    mode=cases(1,case_id);preserve=cases(2,case_id);compensated=cases(3,case_id);debug=cases(4,case_id)
#ifdef ASTR_AIR5_STAGE_REFERENCE
    call configure_air5_species_cache_gpu(preserve/=0)
    air5_compensated=compensated/=0;dynamic_top=mode==2
    allocate(q_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,11),x_d(0:im,0:jm,0:km,3),cache_d(0:im,0:jm,0:km,12))
    q_d=initial;x_d=coords
    if(mode<=2)then
      allocate(inlet_q_d(0:jm,11));inlet_q_d=profile
      farfield_q_d=farfield;wall_temperature_d=wall_temperature;hbl_x_origin_d=origin
      incident_top_d=mode==1;characteristic_top_d=mode==2;incident_x_d=incident_x;incident_q_d=incident
    else
      left_q_d=farfield;right_q_d=incident(:,2);outlet_pressure=merge(pressure_target,0.0_real64,mode==5)
    endif
    if(air5_compensated)then
      allocate(air5_carry_d(0:im,0:jm,0:km,11));air5_carry_d=initial_carry
    endif
    if(mode<=2)then
      flowtype='air5hbl';call apply_air5_hbl_boundary_gpu()
    else
      flowtype=merge('air5normalshock','air5postshock  ',mode/=3)
      call apply_air5_postshock_boundary_gpu()
    endif
    after=q_d
    if(air5_compensated)carry=air5_carry_d
    cache_status_d=0
    call extract_cache<<<((im+1)*(jm+1)*(km+1)+127)/128,128>>>(im,jm,km)
    status=cudaDeviceSynchronize();call check(status);status=cache_status_d;call check(status);cache=cache_d
    deallocate(q_d,x_d,cache_d)
    if(mode<=2)deallocate(inlet_q_d)
    if(air5_compensated)deallocate(air5_carry_d)
#else
    call prepare_case(mode,preserve,compensated,debug)
    call check(device_state_upload(handle,device_q,initial,int(size(initial),c_size_t)*8_c_size_t))
    if(compensated/=0)call check(device_state_upload(handle,device_air5_carry,initial_carry,int(size(initial_carry),c_size_t)*8_c_size_t))
    failed_stage=0
    if(mode<=2)then
      physics=air5_hbl_boundary_apply(handle,lower,upper,wall_temperature,mode==2,int(preserve,c_int), &
        compensated/=0,int(cart,c_int),failed_stage)
    else
      outlet_target=merge(pressure_target,0.0_real64,mode==5)
      physics=air5_postshock_boundary_apply(handle,lower(1),upper(1),mode/=3,outlet_target,int(preserve,c_int),int(cart,c_int))
    endif
    call require(physics==0.and.failed_stage==0,'successful stage sequence')
    call check(device_state_download(handle,device_q,after,int(size(after),c_size_t)*8_c_size_t))
    if(compensated/=0)call check(device_state_download(handle,device_air5_carry,carry,int(size(carry),c_size_t)*8_c_size_t))
    call check(air5_device_primitives(handle,[0_c_int,0_c_int,0_c_int],int([im,jm,km],c_int),int(preserve,c_int)))
    call check(air5_device_check(handle,physics));call require(physics==0,'same-phase boundary cache')
    first_component=1
    do field=1,6
      call check(device_state_view(handle,int(cache_fields(field),c_int),view));allocate(flat(product(view%extent)))
      call check(device_state_download(handle,int(cache_fields(field),c_int),flat,int(size(flat),c_size_t)*8_c_size_t))
      do c=1,components(field);do k=0,km;do j=0,jm;do i=0,im
        index=1+int((i-view%lower(1))*view%stride(1)+(j-view%lower(2))*view%stride(2)+ &
          (k-view%lower(3))*view%stride(3)+(c-1)*view%stride(4))
        cache(i,j,k,first_component+c-1)=flat(index)
      enddo;enddo;enddo;enddo
      deallocate(flat);first_component=first_component+components(field)
    enddo
    call check(device_state_download(handle,device_q,q_check,int(size(q_check),c_size_t)*8_c_size_t))
    call require(all(transfer(after,[0_int64],size(after))==transfer(q_check,[0_int64],size(q_check))),'cache refresh changed q')
    call check(device_state_destroy(handle))
#endif
    write(output)cases(:,case_id),after
    if(compensated/=0)write(output)carry
    write(output)cache
  enddo
  close(output)
#ifndef ASTR_AIR5_STAGE_REFERENCE
  do case_id=1,4
    call prepare_case(merge(5,0,case_id==4),1,1,0)
    q_check=initial
    if(case_id==1.and.rank==ranks-1)q_check(im-1,3,0,1)=-1.0_real64
    if(case_id==2.and.rank==0)q_check(2,1,0,1)=-1.0_real64
    if(case_id==3.and.rank==ranks-1)then
      call check(air5_top_boundary_prepare(handle,farfield,incident,0.0_c_double,incident_x,0_c_int, &
        coords(0:im,jm,0:km,1),int((im+1)*(km+1),c_size_t)))
    endif
    if(case_id==4.and.rank==ranks-1)q_check(im-1,3,0,1)=-1.0_real64
    call check(device_state_upload(handle,device_q,q_check,int(size(q_check),c_size_t)*8_c_size_t))
    call check(device_state_upload(handle,device_air5_carry,initial_carry,int(size(initial_carry),c_size_t)*8_c_size_t))
    if(case_id==4)then
      physics=air5_postshock_boundary_apply(handle,lower(1),upper(1),.true.,pressure_target,1_c_int,int(cart,c_int))
      failed_stage=0_c_int
    else
      physics=air5_hbl_boundary_apply(handle,lower,upper,wall_temperature,.false.,1_c_int,.true.,int(cart,c_int),failed_stage)
    endif
    call require(physics==2.and.failed_stage==merge(0,case_id,case_id==4),'global failed-stage release')
    call check(device_state_download(handle,device_q,after,int(size(after),c_size_t)*8_c_size_t))
    call check(device_state_download(handle,device_air5_carry,carry,int(size(carry),c_size_t)*8_c_size_t))
    call require(all(transfer(carry,[0_int64],size(carry))==transfer(initial_carry,[0_int64],size(initial_carry))), &
      'failed boundary reset carry')
    do c=1,11;do k=-hm,km+hm;do j=-hm,jm+hm;do i=-hm,im+hm
      if(upper(1)/=0.and.i>=im.and.j>=0.and.j<=jm.and.k>=0.and.k<=km)cycle
      if(case_id==4.and.lower(1)/=0.and.i<=0.and.j>=0.and.j<=jm.and.k>=0.and.k<=km)cycle
      if((case_id==2.or.case_id==3).and.lower(2)/=0.and.j<=0.and.i>=0.and.i<=im.and.k>=0.and.k<=km)cycle
      if(case_id==3.and.upper(2)/=0.and.j>=jm.and.i>=0.and.i<=im.and.k>=0.and.k<=km.and.rank/=ranks-1)cycle
      call require(transfer(after(i,j,k,c),0_int64)==transfer(q_check(i,j,k,c),0_int64),'later/unowned face wrote after global failure')
    enddo;enddo;enddo;enddo
    write(*,'(A,I0,A,I0,A,I0,A,I0)')'AIR5_BOUNDARY_STAGE_FAILURE_PASS rank=',rank,' test=',case_id,' stage=',failed_stage,' status=',physics
    call check(device_state_destroy(handle))
  enddo
#endif
  write(*,'(A,I0,A,I0)')'AIR5_BOUNDARY_STAGE_PASS rank=',rank,' cases=',ncases
  call MPI_Comm_free(cart,ierr);call check(ierr)
  call MPI_Finalize(ierr)
contains
  subroutine check(code)
    integer,intent(in) :: code
    integer :: ignored
    if(code==0)return
    write(*,*)'stage probe rank/code=',rank,code
    call MPI_Abort(MPI_COMM_WORLD,abs(code),ignored)
  end subroutine
  subroutine require(valid,label)
    logical,intent(in) :: valid
    character(*),intent(in) :: label
    integer :: ignored
    if(valid)return
    write(*,*)'stage probe rank=',rank,trim(label)
    call MPI_Abort(MPI_COMM_WORLD,1,ignored)
  end subroutine
#ifndef ASTR_AIR5_STAGE_REFERENCE
  subroutine prepare_case(mode,preserve,compensated,debug)
    integer,intent(in) :: mode,preserve,compensated,debug
    call check(device_state_create(int(device,c_int),int(im,c_int),int(jm,c_int),int(km,c_int),int(hm,c_int),int(debug,c_int),handle))
    call check(air5_device_prepare(handle))
    if(compensated/=0)then
      options=air5_device_options(1_c_int,0_c_int,0_c_int,1_c_int,int(preserve,c_int),32_c_int,0_c_int, &
        1e-9_c_double,1e-13_c_double,0.0_c_double)
      call check(air5_chemistry_prepare(handle,options))
    else
      call require(device_state_view(handle,device_air5_carry,view)==device_not_allocated,'unused carry allocated')
    endif
    if(mode<=2)then
      if(lower(1)/=0)call check(air5_profile_boundary_prepare(handle,profile,int(size(profile),c_size_t)))
      if(upper(2)/=0)call check(air5_top_boundary_prepare(handle,farfield,incident,origin,incident_x, &
        int(merge(1,0,mode==1),c_int),coords(0:im,jm,0:km,1),int((im+1)*(km+1),c_size_t)))
    else
      call check(air5_fixed_boundary_prepare(handle,farfield,incident(:,2)))
    endif
  end subroutine
#endif
end program
