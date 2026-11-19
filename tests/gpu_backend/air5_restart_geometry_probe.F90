program air5_restart_geometry_probe
  use iso_fortran_env, only: int64
  use iso_c_binding, only: c_ptr,c_int,c_size_t
  use mpi
  use astr_precision, only: real_kind,real_bytes
  use checkpoint_state_io, only: checkpoint_state_header,checkpoint_state_identity,checkpoint_state_require, &
    checkpoint_state_transfer,allocate_checkpoint_buffer
  use device_state, only: device_state_create,device_state_upload,device_state_download,device_state_destroy, &
    device_count,device_metric,device_jacobian
  use chemistry_device_properties, only: air5_device_prepare
  use chemistry_device_transport, only: air5_geometry_prepare
  implicit none
  type(c_ptr) :: handle
  type(checkpoint_state_identity) :: geometry
  real(real_kind),allocatable :: data(:,:,:,:)
  integer(int64) :: header(13),remaining
  integer(c_size_t) :: field_bytes
  integer(c_int) :: devices
  integer :: err,rank,np,topo(3),global(3),coords(3),cells(3),origin(3),halo,producer_bytes,out
  logical :: matched
  character(1024) :: archive,prefix,arg,path

  call MPI_Init(err)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,err)
  call MPI_Comm_size(MPI_COMM_WORLD,np,err)
  call get_command_argument(1,archive)
  call get_command_argument(2,prefix)
  call get_command_argument(3,arg)
  read(arg,*) topo
  call get_command_argument(4,arg)
  read(arg,*) halo
  call checkpoint_state_require(all(topo>0).and.product(topo)==np,MPI_COMM_WORLD,'geometry probe topology')
  call checkpoint_state_header(trim(archive),header,comm=MPI_COMM_WORLD,producer_bytes=producer_bytes)
  call checkpoint_state_require(real_bytes==8.and.producer_bytes==8.and.header(7)==13.and.header(13)==1, &
    MPI_COMM_WORLD,'geometry probe requires a full FP64 archive')
  global=int(header(4:6))
  call checkpoint_state_require(all(mod(global-1,topo)==0),MPI_COMM_WORLD,'geometry probe rank extents')
  coords=[rank/(topo(2)*topo(3)),mod(rank/topo(3),topo(2)),mod(rank,topo(3))]
  cells=(global-1)/topo
  origin=cells*coords
  call allocate_checkpoint_buffer(cells,halo,13,huge(0_int64),data,remaining,MPI_COMM_WORLD)
  call checkpoint_state_transfer(trim(archive),.false.,.true.,global,origin,cells,halo,data,geometry,remaining, &
    MPI_COMM_WORLD,allow_repartition=.false.,restored_exact=matched)
  call checkpoint_state_require(matched.and.geometry%step==0.and.geometry%time==0.0_real_kind.and. &
    geometry%dt_used==0.0_real_kind.and.geometry%dt_next==1.0_real_kind,MPI_COMM_WORLD,'geometry archive identity')

  call check(device_count(devices),'count devices')
  call checkpoint_state_require(devices>=np,MPI_COMM_WORLD,'one physical GPU per probe rank')
  call check(device_state_create(int(rank,c_int),int(cells(1),c_int),int(cells(2),c_int), &
    int(cells(3),c_int),int(halo,c_int),1_c_int,handle),'create context')
  call check(air5_device_prepare(handle),'allocate AIR5 fields')
  call check(air5_geometry_prepare(handle),'allocate AIR5 geometry')
  field_bytes=8_c_size_t*size(data(:,:,:,4),kind=c_size_t)
  call check(device_state_upload(handle,device_jacobian,data(:,:,:,4),field_bytes),'upload Jacobian')
  call check(device_state_upload(handle,device_metric,data(:,:,:,5:13),9*field_bytes),'upload metrics')
  data(:,:,:,4:13)=huge(0.0_real_kind)
  call check(device_state_download(handle,device_jacobian,data(:,:,:,4),field_bytes),'download Jacobian')
  call check(device_state_download(handle,device_metric,data(:,:,:,5:13),9*field_bytes),'download metrics')
  call check(device_state_destroy(handle),'destroy context')
  write(path,'(a,".rank",i0,".bin")')trim(prefix),rank
  open(newunit=out,file=trim(path),access='stream',form='unformatted',status='new',action='write')
  write(out) data
  close(out)
  call MPI_Finalize(err)
contains
  subroutine check(status,operation)
    integer(c_int),intent(in) :: status
    character(*),intent(in) :: operation
    call checkpoint_state_require(status==0,MPI_COMM_WORLD,operation)
  end subroutine check
end program air5_restart_geometry_probe
