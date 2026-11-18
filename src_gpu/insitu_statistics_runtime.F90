module insitu_statistics_cpp
  use iso_c_binding
  use iso_fortran_env, only: int64
  use astr_precision, only: real_kind,real_bytes
  use astr_precision_mpi, only: mpi_real_type
  use mpi
  use ieee_arithmetic, only: ieee_is_finite
  use device_state, only: device_memory_info,device_pci_identity,device_view
  use insitu_moments_cpp
  use fluid_runtime, only: cpp_device_context
  use insitu_time_integral, only: clipped_trapezoid
  use insitu_fields, only: insitu_owned_counts,insitu_node_weights,insitu_volume_weights
  use insitu_resource_budget, only: resource_budget,checked_bytes,configure_budget,reserve_bytes
  implicit none
  private
  public :: configure_statistics_cpp_budget,accumulate_statistics_cpp,spatial_statistics_cpp
  public :: pack_statistics_cpp_state,unpack_statistics_cpp_state,download_statistics_cpp_output
  public :: release_statistics_cpp,statistics_cpp_transfer_counts,slice_statistics_cpp_info
  public :: statistics_cpp_view
  type(c_ptr),save :: plan=c_null_ptr
  logical,save :: weights_ready=.false.
  integer(int64),save :: device_budget=0,device_headroom=0,sample_calls=0,output_downloads=0
contains
  subroutine guard(valid,message)
    logical,intent(in) :: valid
    character(*),intent(in) :: message
    integer :: ignored
    if(valid) return
    print *, 'C++ in-situ statistics: ',message
    call MPI_Abort(MPI_COMM_WORLD,1,ignored)
  end subroutine

  subroutine configure_statistics_cpp_budget(limit,headroom,ok)
    integer(int64),intent(in) :: limit,headroom
    logical,intent(out) :: ok
    ok=.not.c_associated(plan).and.limit>0.and.headroom>=0
    if(.not.ok) return
    device_budget=limit;device_headroom=headroom
  end subroutine

  subroutine admit_allocation(extra)
    integer(int64),intent(in) :: extra
    integer :: node,nrank,i,ierr,status
    integer(c_size_t) :: free_memory,total_memory
    integer(int64) :: request,total,available
    integer(int64),allocatable :: requests(:),free_values(:)
    character(kind=c_char,len=32) :: pci
    character(kind=c_char,len=32),allocatable :: identities(:)
    type(resource_budget) :: budget
    logical :: ok
    if(device_budget==0) return
    call checked_bytes(int(insitu_owned_counts(),int64),31_int64*real_bytes,request,ok)
    call guard(ok,'statistics allocation byte count')
    call guard(extra>=0.and.extra<=huge(request)-request-52,'statistics output allocation byte count')
    request=request+52+extra
    pci=repeat(achar(0,kind=c_char),32)
    status=device_pci_identity(cpp_device_context(),pci,32_c_int)
    call guard(status==0,'statistics device identity')
    status=device_memory_info(cpp_device_context(),free_memory,total_memory)
    call guard(status==0,'statistics free device memory')
    call MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,node,ierr)
    call guard(ierr==MPI_SUCCESS,'statistics node communicator')
    call MPI_Comm_size(node,nrank,ierr)
    call guard(ierr==MPI_SUCCESS,'statistics node size')
    allocate(identities(nrank),requests(nrank),free_values(nrank),stat=status)
    call guard(status==0,'statistics resource metadata allocation')
    call MPI_Allgather(pci,32,MPI_CHARACTER,identities,32,MPI_CHARACTER,node,ierr)
    call guard(ierr==MPI_SUCCESS,'statistics GPU identity gather')
    call MPI_Allgather(request,1,MPI_INTEGER8,requests,1,MPI_INTEGER8,node,ierr)
    call guard(ierr==MPI_SUCCESS,'statistics demand gather')
    available=int(free_memory,int64)
    call MPI_Allgather(available,1,MPI_INTEGER8,free_values,1,MPI_INTEGER8,node,ierr)
    call guard(ierr==MPI_SUCCESS,'statistics free memory gather')
    total=0;ok=.true.
    do i=1,nrank
      if(identities(i)/=pci) cycle
      if(requests(i)>huge(total)-total) then
        ok=.false.;exit
      endif
      total=total+requests(i);available=min(available,free_values(i))
    enddo
    call guard(ok,'statistics shared device demand overflow')
    call MPI_Comm_free(node,ierr)
    call guard(ierr==MPI_SUCCESS,'statistics node communicator release')
    call configure_budget(budget,device_budget,device_headroom,ok)
    call guard(ok,'statistics budget configuration')
    call reserve_bytes(budget,1,total,available,ok)
    call guard(ok,'statistics allocation exceeds configured device budget or reserve')
  end subroutine

  subroutine prepare(window)
    real(real_kind),intent(in) :: window(2)
    real(real_kind) :: info(5)
    integer :: status
    if(.not.c_associated(plan)) then
      call admit_allocation(0_int64)
      status=moments_create(cpp_device_context(),int(insitu_owned_counts(),c_int),window,plan)
      call guard(status==0,'statistics state allocation')
    else
      status=moments_info(plan,info)
      call guard(status==0,'statistics state clock')
      call guard(all(info(2:3)==window),'statistics window changed during accumulation')
    endif
  end subroutine

  subroutine accumulate_statistics_cpp(time,window)
    real(real_kind),intent(in) :: time,window(2)
    real(real_kind) :: weights(2),duration,info(5)
    integer :: status
    logical :: ok
    call prepare(window)
    status=moments_info(plan,info)
    call guard(status==0,'statistics previous sample clock')
    weights=0
    if(info(5)==1) then
      call clipped_trapezoid(info(1),time,[1.e0_real_kind,0.e0_real_kind],[0.e0_real_kind,1.e0_real_kind], &
        window(1),window(2),weights,duration,ok)
      call guard(ok,'statistics sample interval')
    endif
    status=moments_push(plan,time,weights)
    call guard(status==0,'statistics sample update')
    sample_calls=sample_calls+1
  end subroutine

  subroutine spatial_statistics_cpp(cell_volume,volume,velocity,variance,ok)
    use commvar, only: im,jm,km,flowtype,lreadgrid
    real(real_kind),intent(in) :: cell_volume
    real(real_kind),intent(out) :: volume,velocity(3),variance(3)
    logical,intent(out) :: ok
    real(real_kind),allocatable :: measures(:,:,:),physical(:,:,:)
    real(real_kind) :: yweights(0:jm),local(7),global(7)
    integer :: extent(3),j,status,ierr
    ok=.false.;volume=0;velocity=0;variance=0
    call guard(c_associated(plan),'statistics spatial reduction needs accumulated state')
    if(.not.weights_ready) then
      call guard(ieee_is_finite(cell_volume).and.cell_volume>0,'statistics cell volume')
      extent=insitu_owned_counts()
      allocate(measures(extent(1),extent(2),extent(3)),stat=status)
      call guard(status==0,'statistics node measure allocation')
      if(lreadgrid) then
        allocate(physical(0:im,0:jm,0:km),stat=status)
        call guard(status==0,'statistics physical node measure allocation')
        call insitu_volume_weights(physical)
        measures=physical(0:extent(1)-1,0:extent(2)-1,0:extent(3)-1)
      elseif(trim(flowtype)=='channel') then
        call insitu_node_weights(yweights)
        do j=1,extent(2)
          measures(:,j,:)=yweights(j-1)
        enddo
      else
        measures=cell_volume
      endif
      status=moments_weights(plan,measures,int(size(measures),c_size_t)*real_bytes)
      call guard(status==0,'statistics static node measures upload')
      weights_ready=.true.
    endif
    status=moments_spatial(plan,local)
    call guard(status==0,'statistics device spatial reduction')
    call MPI_Allreduce(local,global,7,mpi_real_type,MPI_SUM,MPI_COMM_WORLD,ierr)
    call guard(ierr==MPI_SUCCESS,'statistics global spatial reduction')
    if(.not.all(ieee_is_finite(global)).or.global(1)<=0) return
    volume=global(1);velocity=global(2:4)/volume;variance=global(5:7)/volume
    ok=all(ieee_is_finite(velocity)).and.all(ieee_is_finite(variance)).and.all(variance>=0)
  end subroutine

  subroutine pack_statistics_cpp_state(values,ok)
    real(real_kind),contiguous,intent(out) :: values(:,:,:,:)
    logical,intent(out) :: ok
    integer :: status
    ok=.false.
    if(.not.c_associated(plan).or.any(shape(values)/=[insitu_owned_counts(),34])) return
    status=moments_pack(plan,values,int(size(values),c_size_t)*real_bytes)
    ok=status==0
  end subroutine

  subroutine unpack_statistics_cpp_state(values,time,window,ok)
    real(real_kind),contiguous,intent(in) :: values(:,:,:,:)
    real(real_kind),intent(in) :: time,window(2)
    logical,intent(out) :: ok
    integer :: status
    ok=.false.
    if(any(shape(values)/=[insitu_owned_counts(),34])) return
    call prepare(window)
    status=moments_restore(plan,values,int(size(values),c_size_t)*real_bytes,time)
    ok=status==0
  end subroutine

  subroutine download_statistics_cpp_output(values)
    real(real_kind),contiguous,intent(out) :: values(:,:,:,:)
    integer :: status
    call guard(c_associated(plan).and.all(shape(values)==[insitu_owned_counts(),41]),'statistics output shape/state')
    call admit_allocation(41_int64*real_bytes*product(int(insitu_owned_counts(),int64)))
    status=moments_output(plan,values,int(size(values),c_size_t)*real_bytes)
    call guard(status==0,'statistics explicit output download')
    output_downloads=output_downloads+1
  end subroutine

  subroutine slice_statistics_cpp_info(active,covered,window,duration)
    logical,intent(out) :: active,covered
    real(real_kind),intent(out) :: window(2),duration
    real(real_kind) :: info(5)
    integer :: status
    active=.false.;covered=.false.;window=0;duration=0
    if(.not.c_associated(plan)) return
    status=moments_info(plan,info)
    call guard(status==0,'statistics source metadata')
    active=info(5)==1;duration=info(4);covered=active.and.duration>0;window=info(2:3)
  end subroutine

  subroutine statistics_cpp_view(view)
    type(device_view),intent(out) :: view
    integer :: status
    call guard(c_associated(plan),'statistics device source is not allocated')
    status=moments_view(plan,view)
    call guard(status==0,'statistics borrowed device view')
  end subroutine

  subroutine statistics_cpp_transfer_counts(samples,downloads)
    integer(int64),intent(out) :: samples,downloads
    samples=sample_calls;downloads=output_downloads
  end subroutine

  subroutine release_statistics_cpp()
    integer :: status
    if(.not.c_associated(plan)) return
    status=moments_destroy(plan)
    call guard(status==0,'statistics release')
    weights_ready=.false.
  end subroutine
end module
