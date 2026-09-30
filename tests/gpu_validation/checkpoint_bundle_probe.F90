program checkpoint_bundle_probe
  use mpi
  use checkpoint_bundle
  implicit none
  integer :: err,rank,keep,i
  logical :: ok
  character(1024) :: path,mode,name
  character(128) :: names(1)
  character(128) :: resource_names(2),with_resources(2)
  type(checkpoint_retention) :: ledger
  call MPI_Init(err)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,err)
  call get_command_argument(1,path)
  call get_command_argument(2,mode)
  names='state.h5'
  if (trim(mode)=='resources') then
    resource_names=[character(128) :: 'mesh.h5','model.nml']
    call write_checkpoint_resource_refs(trim(path),resource_names,MPI_COMM_WORLD,ok)
  else if (trim(mode)=='seal_resources') then
    with_resources=[character(128) :: 'state.h5','RESOURCES']
    call seal_checkpoint_bundle(trim(path),with_resources,MPI_COMM_WORLD,ok)
  else if (trim(mode)=='retain') then
    call get_command_argument(3,name)
    read(name,*) keep
    do i=1,4
      write(name,'("batch",i0)') i
      call publish_retained_checkpoint(trim(path),trim(name),keep,ledger,MPI_COMM_WORLD,ok)
      if (.not.ok) exit
    enddo
  else if (trim(mode)=='publish') then
    call get_command_argument(3,name)
    call publish_checkpoint_bundle(trim(path),trim(name),MPI_COMM_WORLD,ok)
  else if (trim(mode)=='seal') then
    call seal_checkpoint_bundle(trim(path),names,MPI_COMM_WORLD,ok)
  else
    call validate_checkpoint_bundle(trim(path),MPI_COMM_WORLD,ok)
  endif
  if (.not.ok) then
    if (rank==0) print *, 'REJECT bundle'
    call MPI_Abort(MPI_COMM_WORLD,1,err)
  endif
  if (rank==0) print *, 'PASS bundle'
  call MPI_Finalize(err)
end program
