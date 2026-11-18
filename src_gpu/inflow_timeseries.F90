module inflow_timeseries_cpp
  use iso_c_binding, only: c_ptr,c_null_ptr,c_int,c_size_t,c_sizeof
  use iso_fortran_env, only: int64
  use ieee_arithmetic, only: ieee_is_finite
  use astr_precision, only: real_kind,real_to_word,word_to_real
  use commvar, only: jm,km,ja,ka
  use bc, only: bctype,turbinf,ninflowslice
  use parallel, only: irk,jg0,kg0,mpi_imin
  use boundary_state_cpp, only: inflow_slices_prepare,inflow_slice_upload,inflow_slice_download,dynamic_inflow_x
  use mpi
  implicit none
  private
  public :: init_dynamic_inflow_cpp,prepare_dynamic_inflow_cpp,apply_dynamic_inflow_cpp
  public :: complete_inflow_cpp_file,release_dynamic_inflow_cpp,dynamic_inflow_requested_cpp
  type(c_ptr),save :: context=c_null_ptr
  real(real_kind),allocatable,save :: slice(:,:,:)
  real(real_kind),save :: times(0:3)=0
  integer(c_int),save :: order(0:3)=[0,1,2,3]
  logical,save :: ready=.false.
contains
  logical function dynamic_inflow_requested_cpp()
    dynamic_inflow_requested_cpp=bctype(1)==11.and.trim(turbinf)=='intp'
  end function

  subroutine require(condition,message)
    logical,intent(in) :: condition
    character(*),intent(in) :: message
    integer :: ignored
    if(condition) return
    write(*,'(2a)') 'ASTR_GPU_CPP inflow: ',message
    call MPI_Abort(MPI_COMM_WORLD,1,ignored)
  end subroutine

  subroutine check(status,message)
    integer(c_int),intent(in) :: status
    character(*),intent(in) :: message
    call require(status==0,message)
  end subroutine

  subroutine init_dynamic_inflow_cpp(handle)
    type(c_ptr),intent(in) :: handle
    integer :: slot,first
    if(.not.dynamic_inflow_requested_cpp().or.irk/=0) return
    call require(ninflowslice>=3,'requires at least four input slices')
    context=handle
    allocate(slice(0:jm,0:km,5))
    call check(inflow_slices_prepare(context),'prepare four device slots')
    first=ninflowslice-3
    do slot=0,3
      call load_slot(first+slot,slot)
    enddo
    call sort_and_validate()
    ready=.true.
    write(*,'(a,i0)') 'ASTR_GPU_CPP dynamic_inflow device_bytes=',4_c_size_t*slice_bytes()
  end subroutine

  integer(c_size_t) function slice_bytes()
    slice_bytes=size(slice,kind=c_size_t)*c_sizeof(0.0_real_kind)
  end function

  subroutine load_slot(number,slot)
    use hdf5io
    use output_input_resources, only: inflow_source_path
    integer,intent(in) :: number,slot
    character(1400) :: path
    call require(number>=0.and.number<=99999,'slice index outside existing source range')
    path=inflow_source_path(number)
    call h5io_init(filename=trim(path),mode='read',comm=mpi_imin)
    call h5read(varname='time',var=times(slot))
    call h5read(varname='ro',var=slice(:,:,1),dir='i',display=.false.)
    call h5read(varname='u1',var=slice(:,:,2),dir='i',display=.false.)
    call h5read(varname='u2',var=slice(:,:,3),dir='i',display=.false.)
    call h5read(varname='u3',var=slice(:,:,4),dir='i',display=.false.)
    call h5read(varname='t',var=slice(:,:,5),dir='i',display=.false.)
    call h5io_end
    call require(ieee_is_finite(times(slot)).and.all(ieee_is_finite(slice)),'nonfinite input slice')
    call check(inflow_slice_upload(context,int(slot,c_int),slice,slice_bytes()),'upload replacement slice')
  end subroutine

  subroutine sort_and_validate()
    integer :: a,b,tmp
    real(real_kind) :: delta,scale,other
    order=[0,1,2,3]
    do a=0,2
      do b=a+1,3
        if(times(order(a))>times(order(b))) then
          tmp=order(a);order(a)=order(b);order(b)=tmp
        endif
      enddo
    enddo
    delta=times(order(1))-times(order(0));scale=max(1.e0_real_kind,maxval(abs(times)))
    call require(all(ieee_is_finite(times)).and.delta>0,'times must be finite and increasing')
    do a=1,2
      other=times(order(a+1))-times(order(a))
      call require(other>0.and.abs(other-delta)<=1.e-12_real_kind*scale,'requires original uniform slice interval')
    enddo
  end subroutine

  subroutine prepare_dynamic_inflow_cpp(time)
    real(real_kind),intent(in) :: time
    integer :: slot
    if(.not.dynamic_inflow_requested_cpp().or.irk/=0) return
    call require(ready,'device cache is not initialized')
    do while(time>times(order(2)))
      slot=order(0);ninflowslice=ninflowslice+1
      call load_slot(ninflowslice,slot)
      call sort_and_validate()
    enddo
  end subroutine

  integer(c_int) function apply_dynamic_inflow_cpp(time,mach_pressure,mach,c2,c6) result(status)
    real(real_kind),intent(in) :: time,mach,c2,c6
    logical,intent(in) :: mach_pressure
    call require(ready,'boundary requested before cache initialization')
    status=dynamic_inflow_x(context,order,times(order(0)),times(order(1))-times(order(0)),time, &
      int(merge(1,0,mach_pressure),c_int),mach,c2,c6)
  end function

  subroutine complete_inflow_cpp_file(path,writing,identity,budget,allow_repartition)
    use checkpoint_state_io
    character(*),intent(in) :: path
    logical,intent(in) :: writing
    logical,optional,intent(in) :: allow_repartition
    type(checkpoint_state_identity),intent(inout) :: identity
    integer(int64),intent(in) :: budget
    real(real_kind),allocatable :: buffer(:,:,:,:)
    integer(int64) :: metadata(15),remaining
    integer :: slot,m
    if(irk/=0) return
    call checkpoint_state_require(ready,mpi_imin,'C++ inflow cache not initialized')
    call allocate_checkpoint_buffer([jm,km,1],0,26,budget,buffer,remaining,mpi_imin)
    metadata=0
    if(writing) then
      metadata(1:6)=[1_int64,1_int64,int(ninflowslice,int64),-1_int64,0_int64,1_int64]
      metadata(7:10)=real_to_word(times);metadata(11:14)=int(order,int64)
      buffer=0
      do slot=0,3
        call check(inflow_slice_download(context,int(slot,c_int),slice,slice_bytes()),'checkpoint slot download')
        do m=1,5
          buffer(:,:,1,slot*5+m)=slice(:,:,m)
        enddo
      enddo
      call checkpoint_state_require(all(ieee_is_finite(buffer)),mpi_imin,'C++ inflow cache nonfinite')
    endif
    call checkpoint_state_transfer(path,writing,.true.,[ja+1,ka+1,2],[jg0,kg0,0], &
      [jm,km,1],0,buffer,identity,remaining,mpi_imin,role=8,metadata=metadata, &
      allow_repartition=allow_repartition,repartition_axes=[.false.,.true.,.false.])
    call checkpoint_state_require(metadata(1)==1.and.metadata(2)==1.and. &
      metadata(3)>=3.and.metadata(3)<=99999.and.metadata(4)==-1.and.metadata(5)==0.and.metadata(6)==1.and. &
      all(metadata(11:14)>=0).and.all(metadata(11:14)<=3).and.metadata(15)==0,mpi_imin,'C++ inflow metadata invalid')
    do m=0,3
      call checkpoint_state_require(count(metadata(11:14)==m)==1,mpi_imin,'C++ inflow slot permutation invalid')
    enddo
    if(.not.writing) then
      call checkpoint_state_require(all(ieee_is_finite(buffer)),mpi_imin,'restored C++ inflow cache nonfinite')
      ninflowslice=int(metadata(3));times=word_to_real(metadata(7:10));order=int(metadata(11:14),c_int)
      call validate_restored_times()
      do slot=0,3
        do m=1,5
          slice(:,:,m)=buffer(:,:,1,slot*5+m)
        enddo
        call check(inflow_slice_upload(context,int(slot,c_int),slice,slice_bytes()),'checkpoint slot restore')
      enddo
    endif
  end subroutine

  subroutine validate_restored_times()
    integer(c_int) :: saved(0:3)
    saved=order
    call sort_and_validate()
    call require(all(order==saved),'restored slot order differs from sorted times')
  end subroutine

  subroutine release_dynamic_inflow_cpp()
    if(allocated(slice)) deallocate(slice)
    context=c_null_ptr;ready=.false.
  end subroutine
end module
