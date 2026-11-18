module air5_transport_probe_boundary
  use iso_c_binding, only: c_ptr,c_int,c_double
  use chemistry_device_boundary, only: air5_postshock_boundary_apply
  implicit none
  integer(c_int) :: normal,boundary_ends(2),boundary_comm
contains
  integer(c_int) function state_boundary(context) result(status)
    type(c_ptr),intent(in) :: context
    status=0_c_int
    if(normal/=0)status=air5_postshock_boundary_apply(context,boundary_ends(1),boundary_ends(2), &
      .true.,0.0_c_double,0_c_int,boundary_comm)
  end function

  integer(c_int) function no_boundary_rhs(context) result(status)
    type(c_ptr),intent(in) :: context
    status=0_c_int
  end function
end module

program air5_transport_rk3_probe
  use iso_c_binding, only: c_ptr,c_int,c_double,c_size_t,c_int8_t
  use mpi
  use device_state
  use halo_transport, only: device_transport_prepare
  use chemistry_device_properties, only: air5_device_prepare
  use chemistry_device_transport, only: air5_geometry_prepare,air5_rk_prepare,air5_convection_budget_prepare, &
    air5_transport_prepare,air5_flux_transport_prepare,air5_diffusion_prepare,air5_flux_packed,air5_conv_full_state, &
    air5_velocity_gradients_prepare
  use chemistry_device_step, only: air5_rk_transport_stage,air5_rk_options,air5_rk_diagnostics
  use chemistry_device_boundary, only: air5_fixed_boundary_prepare
  use shock_sensor_cpp, only: shock_prepare_cpp,shock_mask_download_cpp
  use air5_transport_probe_boundary, only: normal,boundary_ends,boundary_comm,state_boundary,no_boundary_rhs
  implicit none
  type(c_ptr) :: handle
  type(air5_rk_options) :: options
  type(air5_rk_diagnostics) :: diagnostics
  integer(c_int) :: sizes(3),low(3),high(3),periodic(3),first(3,1),last(3,1),mode,viscous,stage,comm
  integer :: local,rank,axis,ierr,u,im,jm,km,h
  real(c_double),allocatable :: q(:,:,:,:),rhs(:,:,:,:),jac(:,:,:),metric(:,:,:,:,:),cache(:,:,:,:)
  integer(c_int8_t),allocatable :: mask(:,:,:)
  character(2048) :: input,prefix,path
  character(32) :: text
  character(8) :: rank_tag
  character(1) :: stage_tag

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text)
  read(text,*)local
  call get_command_argument(1,input)
  call get_command_argument(2,prefix)
  do axis=1,3
    call get_command_argument(axis+2,text)
    read(text,*)sizes(axis)
  enddo
  call get_command_argument(6,text)
  read(text,*)viscous
  call get_command_argument(7,text)
  read(text,*)mode
  call get_command_argument(8,text)
  read(text,*)normal
  periodic=[1-normal,1,1]
  call MPI_Cart_create(MPI_COMM_WORLD,3,sizes,periodic/=0,.false.,comm,ierr)
  do axis=1,3
    call MPI_Cart_shift(comm,axis-1,1,low(axis),high(axis),ierr)
  enddo
  boundary_ends=merge(1,0,[low(1),high(1)]==MPI_PROC_NULL);boundary_comm=comm
  write(rank_tag,'(I8.8)')rank
  path=trim(input)//'.rank'//rank_tag//'.bin'
  open(newunit=u,file=trim(path),access='stream',form='unformatted',status='old')
  read(u) im,jm,km,h,options%dt,options%floor
  allocate(q(-h:im+h,-h:jm+h,-h:km+h,11),jac(-h:im+h,-h:jm+h,-h:km+h), &
    metric(-h:im+h,-h:jm+h,-h:km+h,3,3),rhs(0:im,0:jm,0:km,11))
  read(u) q,jac,metric
  close(u)
  options%upper=[im,jm,km];options%ntypes=3;options%halo=h;options%comm=comm
  where(low==MPI_PROC_NULL)options%ntypes=1
  where(high==MPI_PROC_NULL)options%ntypes=2
  where(low==MPI_PROC_NULL.and.high==MPI_PROC_NULL)options%ntypes=4
  options%convection_mode=air5_conv_full_state;options%preserve=0
  options%wall_axis=-1;options%wall_family=0;options%shock_threshold=1.0e-3_c_double
  options%capturing=normal/=0;options%viscous=viscous/=0;options%layered=.false.;options%zero_lower_y=.false.
  options%collect=.false.;options%filtering=.false.;options%conservative_filter=.false.;options%compensated=.false.
  first(:,1)=merge(1,0,low==MPI_PROC_NULL);last(:,1)=options%upper-merge(1,0,high==MPI_PROC_NULL)
  call check(device_bind(local))
  call check(device_state_create(local,im,jm,km,h,1_c_int,handle))
  call check(air5_device_prepare(handle))
  call check(air5_geometry_prepare(handle))
  call check(air5_rk_prepare(handle))
  call check(device_transport_prepare(handle,comm,low,high,sizes,11_c_int,mode,0_c_int,periodic))
  call check(air5_convection_budget_prepare(handle,0_c_int))
  if(options%viscous)then
    call check(air5_transport_prepare(handle,0_c_int))
    call check(air5_diffusion_prepare(handle))
    call check(air5_flux_transport_prepare(handle,air5_flux_packed))
  endif
  if(options%capturing)then
    if(.not.options%viscous)call check(air5_velocity_gradients_prepare(handle))
    call check(shock_prepare_cpp(handle))
    allocate(mask(0:im,0:jm,0:km))
  endif
  if(normal/=0)call check(air5_fixed_boundary_prepare(handle,q(0,0,0,:),q(im,0,0,:)))
  call check(device_state_upload(handle,device_q,q,8_c_size_t*size(q,kind=c_size_t)))
  call check(device_state_upload(handle,device_jacobian,jac,8_c_size_t*size(jac,kind=c_size_t)))
  call check(device_state_upload(handle,device_metric,metric,8_c_size_t*size(metric,kind=c_size_t)))
  do stage=1,3
    call check(air5_rk_transport_stage(handle,stage,.false.,options,first,last,diagnostics,state_boundary,no_boundary_rhs))
    call check(device_state_download(handle,device_q,q,8_c_size_t*size(q,kind=c_size_t)))
    call check(device_state_download(handle,device_rhs,rhs,8_c_size_t*size(rhs,kind=c_size_t)))
    write(stage_tag,'(I1)')stage
    path=trim(prefix)//'.rk'//stage_tag//'.rank'//rank_tag//'.bin'
    open(newunit=u,file=trim(path),access='stream',form='unformatted',status='new')
    write(u) q(0:im,0:jm,0:km,:),rhs
    if(allocated(mask))then
      call check(shock_mask_download_cpp(handle,mask,size(mask,kind=c_size_t)))
      write(u) mask
    endif
    if(stage==3)then
      allocate(cache(-h:im+h,-h:jm+h,-h:km+h,7))
      call check(device_state_download(handle,device_rho,cache(:,:,:,1),8_c_size_t*size(jac,kind=c_size_t)))
      call check(device_state_download(handle,device_velocity,cache(:,:,:,2:4),24_c_size_t*size(jac,kind=c_size_t)))
      call check(device_state_download(handle,device_pressure,cache(:,:,:,5),8_c_size_t*size(jac,kind=c_size_t)))
      call check(device_state_download(handle,device_temperature,cache(:,:,:,6),8_c_size_t*size(jac,kind=c_size_t)))
      call check(device_state_download(handle,device_air5_tv,cache(:,:,:,7),8_c_size_t*size(jac,kind=c_size_t)))
      write(u) cache(0:im,0:jm,0:km,:)
    endif
    close(u)
  enddo
  call check(device_state_destroy(handle))
  call MPI_Comm_free(comm,ierr)
  call MPI_Finalize(ierr)
contains
  subroutine check(status)
    integer(c_int),intent(in) :: status
    integer :: ignored
    if(status==0)return
    write(*,'(A,I0,A,I0)')'AIR5 RK transport probe status ',status,' on rank ',rank
    call MPI_Abort(comm,abs(status),ignored)
  end subroutine
end program
