! Configuration/storage owners for frozen production filter procedures.
module commvar
  implicit none
  integer :: im,jm,km,hm=5
end module
module parallel
  implicit none
  integer :: mpirank
end module
module commarray
  use iso_fortran_env,only:real64
  implicit none
  real(real64),allocatable :: q(:,:,:,:),qrhs(:,:,:,:)
end module
module chemistry_compensation
  use iso_fortran_env,only:real64
  implicit none
  logical :: air5_compensated=.false.
  real(real64),allocatable :: air5_carry(:,:,:,:)
contains
#ifdef ASTR_AIR5_REFERENCE_GPU
  attributes(host,device) &
#endif
#include "frozen_compensated_add.inc"
end module
module chemistry_flow_solver
  use iso_fortran_env,only:real64
  use mpi
  use chemistry_flow_state
  use chemistry_filter_math
  implicit none
  integer :: filter_comm,filter_local(3),filter_global(3),filter_offset(3),filter_plane(2,3),filter_area,filter_width
  integer :: filter_lower(3),filter_upper(3)
  logical :: filter_periodic(3)
  real(real64) :: filter_tr_floor(5),filter_v_floor(5)
  real(real64),allocatable :: filter_send(:,:,:,:),filter_receive(:,:,:,:)
contains
  subroutine configure_air5_filter_exchange()
    ! Uniform geometry and these exact parameters are supplied by the fixture.
  end subroutine
#include "frozen_filter_cpu.inc"
end module
#ifdef ASTR_AIR5_REFERENCE_GPU
module commarray_gpu
  use iso_fortran_env,only:real64
  use cudafor
  implicit none
  logical :: test_scalar
  real(real64),device,allocatable :: q_d(:,:,:,:),qrhs_d(:,:,:,:),qwork_d(:,:,:,:),filter_work_d(:,:,:)
contains
  logical function gpu_scalar_filter_workspace_enabled()
    gpu_scalar_filter_workspace_enabled=test_scalar
  end function
end module
module chemistry_compensation_gpu
  use iso_fortran_env,only:real64
  use cudafor
  implicit none
  logical,device :: air5_compensated_d
  real(real64),device,allocatable :: air5_carry_d(:,:,:,:)
end module
module halo_transport_gpu
  implicit none
  logical :: device_aware_transport_enabled=.false.
end module
module gpu_check
  use cudafor
  use mpi
  implicit none
contains
  subroutine sync_after_kernel(label,force_sync)
    character(*),intent(in) :: label
    logical,intent(in),optional :: force_sync
    integer :: code,ignored
    code=cudaGetLastError()
    if(code==0)code=cudaDeviceSynchronize()
    if(code/=0)then
      write(*,*)label,code
      call MPI_Abort(MPI_COMM_WORLD,code,ignored)
    endif
  end subroutine
end module
module air5_filter_reference_gpu
  use iso_fortran_env,only:real64
  use cudafor
  use chemistry_flow_state_gpu
  use chemistry_model,only:chemistry_status_ok
  implicit none
  integer,device :: air5_flow_status_d,air5_face_failure_d(7)
  real(real64),device :: filter_tr_floor_d(5),filter_v_floor_d(5)
  real(real64),device,allocatable :: filter_send_d(:,:,:,:),filter_receive_d(:,:,:,:),filter_face_d(:,:,:),filter_alias_d(:,:,:)
  real(real64),pinned,allocatable :: filter_send_h(:,:,:,:),filter_receive_h(:,:,:,:),filter_face_h(:,:,:),filter_alias_h(:,:,:)
contains
#include "frozen_filter_gpu.inc"
  subroutine check_air5_status(label)
    use chemistry_flow_solver,only:check_air5_filter_status
    character(*),intent(in) :: label
    integer :: status
    status=air5_flow_status_d
    call check_air5_filter_status(status,label)
  end subroutine
end module
#endif
program air5_filter_field_reference
  use iso_fortran_env,only:real64
  use mpi
  use commvar
  use commarray
  use parallel
  use chemistry_compensation
  use chemistry_flow_solver
  use chemistry_flow_state,only:configure_air5_species_cache
#ifdef ASTR_AIR5_REFERENCE_GPU
  use cudafor
  use commarray_gpu
  use chemistry_compensation_gpu
  use chemistry_flow_state_gpu,only:configure_air5_species_cache_gpu
  use air5_filter_reference_gpu,only:air5_filter_conservative_gpu
#endif
  implicit none
  integer :: sizes(3),coordinates(3),cart,ierr,u,i,j,k,c,step,scalar,compensated,device,devices
  integer :: global(3),periods(3),head(4),local_rank
  logical :: periodic(3)
  real(real64) :: tr(5),v(5)
  real(real64),allocatable :: table(:,:,:,:),physical(:,:,:,:),low(:,:,:,:)
  character(2048) :: input,output,text,prefix
  call get_command_argument(1,input);call get_command_argument(2,prefix)
  do i=1,3
    call get_command_argument(i+2,text);read(text,*)sizes(i)
  enddo
  call get_command_argument(6,text);read(text,*)scalar
  call get_command_argument(7,text);read(text,*)compensated
  call get_command_argument(8,text);read(text,*)devices
#ifdef ASTR_AIR5_REFERENCE_GPU
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text);read(text,*)local_rank
  device=modulo(local_rank,devices);ierr=cudaSetDevice(device)
  if(ierr/=0)error stop 'reference device binding failed'
#endif
  call MPI_Init(ierr);call MPI_Comm_rank(MPI_COMM_WORLD,mpirank,ierr)
  open(newunit=u,file=trim(input),access='stream',form='unformatted',status='old')
  read(u)global,periods,tr,v;allocate(table(22,0:global(1),0:global(2),0:global(3)));read(u)table;close(u)
  periodic=periods/=0;call MPI_Cart_create(MPI_COMM_WORLD,3,sizes,periodic,.false.,cart,ierr)
  call MPI_Cart_coords(cart,mpirank,3,coordinates,ierr)
  filter_global=global;filter_local=global/sizes;filter_offset=filter_local*coordinates;filter_periodic=periodic
  im=filter_local(1);jm=filter_local(2);km=filter_local(3)
  do i=1,3
    call MPI_Cart_shift(cart,i-1,1,filter_lower(i),filter_upper(i),ierr)
  enddo
  filter_plane(:,1)=[jm,km];filter_plane(:,2)=[im,km];filter_plane(:,3)=[im,jm]
  filter_area=maxval((filter_plane(1,:)+1)*(filter_plane(2,:)+1));filter_width=hm+1
  filter_tr_floor=tr;filter_v_floor=v
  call MPI_Comm_dup(cart,filter_comm,ierr)
  allocate(filter_send(22,0:hm,0:filter_area-1,2),filter_receive(22,0:hm,0:filter_area-1,2))
  filter_send=0;filter_receive=0
  allocate(q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,11),qrhs(0:im,0:jm,0:km,11))
  allocate(physical(0:im,0:jm,0:km,11),low(0:im,0:jm,0:km,11));q=-777;qrhs=0
  do k=0,km;do j=0,jm;do i=0,im
    physical(i,j,k,:)=table(1:11,filter_offset(1)+i,filter_offset(2)+j,filter_offset(3)+k)
    low(i,j,k,:)=table(12:22,filter_offset(1)+i,filter_offset(2)+j,filter_offset(3)+k)
  enddo;enddo;enddo
  q(0:im,0:jm,0:km,:)=physical;air5_compensated=compensated/=0
  allocate(air5_carry(0:im,0:jm,0:km,11));air5_carry=low
  call configure_air5_species_cache(.true.)
#ifdef ASTR_AIR5_REFERENCE_GPU
  allocate(q_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,11),qrhs_d(0:im,0:jm,0:km,11))
  if(scalar/=0)then
    allocate(filter_work_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
  else
    allocate(qwork_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,11))
  endif
  allocate(air5_carry_d(0:im,0:jm,0:km,11));air5_carry_d=air5_carry;air5_compensated_d=air5_compensated
  q_d=q;qrhs_d=0;test_scalar=scalar/=0
  call configure_air5_species_cache_gpu(.true.)
#endif
  do step=1,2
#ifdef ASTR_AIR5_REFERENCE_GPU
    call air5_filter_conservative_gpu()
#else
    call air5_filter_conservative()
#endif
  enddo
#ifdef ASTR_AIR5_REFERENCE_GPU
  q=q_d
  air5_carry=air5_carry_d
#endif
  physical=q(0:im,0:jm,0:km,:);low=0
  if(air5_compensated)low=air5_carry
  write(output,'(A,A,I0,A)')trim(prefix),'.rank',mpirank,'.bin'
  head=[im,jm,km,0]
  open(newunit=u,file=trim(output),access='stream',form='unformatted',status='new');write(u)head,physical,low;close(u)
  call MPI_Comm_free(filter_comm,ierr);call MPI_Comm_free(cart,ierr);call MPI_Finalize(ierr)
end program
