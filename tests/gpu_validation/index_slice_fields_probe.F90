program index_slice_fields_probe
  use mpi
  use iso_fortran_env, only: int64
  use astr_precision, only: real_kind
  use commvar, only: im,jm,km,hm,numq
  use commarray, only: rho,vel,prs,tmp,x
  use output_fields, only: write_basic_output,pack_basic_output_cpu,select_output_fields_cpu,pack_selected_output_cpu
  use checkpoint_state_io, only: checkpoint_state_identity
#ifdef TEST_INDEX_FIELDS_GPU
  use commarray_gpu, only: rho_d,vel_d,prs_d,tmp_d
  use output_fields_gpu, only: select_output_fields_gpu,pack_selected_output_gpu, &
    begin_basic_output_gpu,end_basic_output_gpu,pack_basic_output_gpu
  use cudafor, only: cudaSetDevice,cudaGetDeviceCount,cudaSuccess
#endif
  implicit none
  type(checkpoint_state_identity) :: identity
  character(4096) :: path
  character(32) :: argument,mode
  integer :: rank,ranks,err,axis,global_shape(3),origin(3),cells(3),i,j,k,m,devices
  integer(int64) :: peak,bytes,total,download
  logical :: selected
  real(real_kind) :: point(3),value
  call MPI_Init(err)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,err)
  call MPI_Comm_size(MPI_COMM_WORLD,ranks,err)
  call get_command_argument(1,path)
  call get_command_argument(2,argument)
  read(argument,*) axis
  call get_command_argument(3,mode)
  global_shape=[9,7,5];origin=0;cells=global_shape-1
  cells(axis)=cells(axis)/ranks
  origin(axis)=rank*cells(axis)
  im=cells(1);jm=cells(2);km=cells(3);numq=5
  allocate(rho(-hm:im+hm,-hm:jm+hm,-hm:km+hm),vel(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3), &
    prs(-hm:im+hm,-hm:jm+hm,-hm:km+hm),tmp(-hm:im+hm,-hm:jm+hm,-hm:km+hm),x(0:im,0:jm,0:km,3))
  do k=-hm,km+hm
    do j=-hm,jm+hm
      do i=-hm,im+hm
        point=real(origin+[i,j,k],real_kind)
        value=100*point(1)+10*point(2)+point(3)+real(rank,real_kind)/1024
        rho(i,j,k)=value+1._real_kind/16
        do m=1,3
          vel(i,j,k,m)=value+real(m+1,real_kind)/16
        enddo
        prs(i,j,k)=value+5._real_kind/16
        tmp(i,j,k)=value+6._real_kind/16
      enddo
    enddo
  enddo
  do k=0,km
    do j=0,jm
      do i=0,im
        point=real(origin+[i,j,k],real_kind)
        x(i,j,k,:)=[point(1)+point(2)/32,point(2)+point(1)/16,point(3)+point(1)*point(2)/64]
      enddo
    enddo
  enddo
  identity%step=12;identity%time=.012_real_kind;identity%dt_used=.001_real_kind;identity%dt_next=.001_real_kind
#ifdef TEST_INDEX_FIELDS_GPU
  err=cudaGetDeviceCount(devices)
  if(err/=cudaSuccess.or.devices<1) error stop 'GPU field probe needs an available device'
  err=cudaSetDevice(mod(rank,devices))
  if(err/=cudaSuccess) error stop 'GPU field probe device binding failed'
  allocate(rho_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm),vel_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3), &
    prs_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm),tmp_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
  rho_d=rho;vel_d=vel;prs_d=prs;tmp_d=tmp
  selected=select_output_fields_gpu([2,5])
  call begin_basic_output_gpu(1024_int64,67108864_int64,MPI_COMM_WORLD)
#else
  selected=select_output_fields_cpu([2,5])
#endif
  if(.not.selected) error stop 'selected primitive fields are unavailable'
  if(mode=='legacy-volume') then
    call write_basic_output(trim(path),global_shape,origin,cells,0,0,6,x,pack_legacy,identity, &
      4096_int64,67108864_int64,MPI_COMM_WORLD,'dimensionless',peak,bytes)
  elseif(mode=='shared-geometry') then
    call write_basic_output(trim(path),global_shape,origin,cells,2,2,6,x,pack_selected,identity, &
      4096_int64,67108864_int64,MPI_COMM_WORLD,'dimensionless',peak,bytes, &
      field_selection=[2,5],index_ranges=reshape([1,7,2,2,1,3],[2,3]),geometry_only=.true.,file_name='geometry.h5')
    call write_basic_output(trim(path),global_shape,origin,cells,2,2,6,x,pack_selected,identity, &
      4096_int64,67108864_int64,MPI_COMM_WORLD,'dimensionless',peak,bytes, &
      field_selection=[2,5],index_ranges=reshape([1,7,2,2,1,3],[2,3]),geometry_file='geometry.h5',file_name='step12.h5')
  else
    call write_basic_output(trim(path),global_shape,origin,cells,2,2,6,x,pack_selected,identity, &
      4096_int64,67108864_int64,MPI_COMM_WORLD,'dimensionless',peak,bytes, &
      field_selection=[2,5],index_ranges=reshape([1,7,2,2,1,3],[2,3]))
  endif
  call MPI_Reduce(bytes,total,1,MPI_INTEGER8,MPI_SUM,0,MPI_COMM_WORLD,err)
  if(rank==0) print '(a,i0)','SLICE_FIELD_BYTES=',total
#ifdef TEST_INDEX_FIELDS_GPU
  call end_basic_output_gpu(download)
  call MPI_Reduce(download,total,1,MPI_INTEGER8,MPI_SUM,0,MPI_COMM_WORLD,err)
  if(rank==0) print '(a,i0)','SLICE_DOWNLOAD_BYTES=',total
#endif
  call MPI_Finalize(err)
contains
  subroutine pack_legacy(first,extent,components,values)
    integer,intent(in) :: first(3),extent(3),components
    real(real_kind),intent(out) :: values(:)
#ifdef TEST_INDEX_FIELDS_GPU
    call pack_basic_output_gpu(first,extent,components,values)
#else
    call pack_basic_output_cpu(first,extent,components,values)
#endif
  end subroutine

  subroutine pack_selected(first,extent,components,values)
    integer,intent(in) :: first(3),extent(3),components
    real(real_kind),intent(out) :: values(:)
#ifdef TEST_INDEX_FIELDS_GPU
    call pack_selected_output_gpu(first,extent,components,values)
#else
    call pack_selected_output_cpu(first,extent,components,values)
#endif
  end subroutine
end program
