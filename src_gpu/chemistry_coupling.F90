module chemistry_device_coupling
  use iso_c_binding, only: c_ptr,c_int,c_double,c_int64_t
  implicit none
  private
  type,bind(C),public :: air5_device_options
    integer(c_int) :: compensated,history_enabled,source_mode,source_only,preserve_all
    integer(c_int) :: block_size,specialized_dispatch
    real(c_double) :: rtol,atol_factor,activity_threshold
  end type
  type,bind(C),public :: air5_device_report
    integer(c_int) :: status,max_counts(4),failure(6)
    integer(c_int64_t) :: skipped,total,minimum_bits(3),maximum_bits(2)
    real(c_double) :: failure_q(11),failure_carry(6)
#ifdef ASTR_AIR5_WORKLOAD_PROBE
    integer(c_int64_t) :: sum_counts(4)
#endif
  end type
  type,public :: air5_chemistry_global_report
    logical :: collected
    integer(c_int) :: status,max_counts(4)
    integer(c_int64_t) :: skipped,total
    real(c_double) :: minimums(3),maximums(2)
  end type
  public :: air5_chemistry_prepare,air5_chemistry_begin,air5_chemistry_region
  public :: air5_chemistry_status,air5_chemistry_report
  public :: air5_chemistry_apply_regions
  interface
    integer(c_int) function air5_chemistry_prepare(handle,options) bind(C,name='astr_device_air5_chemistry_prepare')
      import c_ptr,c_int,air5_device_options
      type(c_ptr),value :: handle
      type(air5_device_options),intent(in) :: options
    end function
    integer(c_int) function air5_chemistry_begin(handle) bind(C,name='astr_device_air5_chemistry_begin')
      import c_ptr,c_int
      type(c_ptr),value :: handle
    end function
    integer(c_int) function air5_chemistry_region(handle,first,last,duration,collect) &
        bind(C,name='astr_device_air5_chemistry_region')
      import c_ptr,c_int,c_double
      type(c_ptr),value :: handle
      integer(c_int),intent(in) :: first(3),last(3)
      real(c_double),value :: duration
      integer(c_int),value :: collect
    end function
    integer(c_int) function air5_chemistry_status(handle,status) bind(C,name='astr_device_air5_chemistry_status')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),intent(out) :: status
    end function
    integer(c_int) function air5_chemistry_report(handle,report) bind(C,name='astr_device_air5_chemistry_report')
      import c_ptr,c_int,air5_device_report
      type(c_ptr),value :: handle
      type(air5_device_report),intent(out) :: report
    end function
  end interface
contains
  ! Region ownership comes from the ordinary solver. This phase stops before
  ! boundary/halo work; a nonzero global status must not enter that next phase.
  integer(c_int) function air5_chemistry_apply_regions(handle,regions,first,last,duration, &
      collect,packed,activity,comm,local,global) result(status)
    use mpi,only:MPI_Allreduce,MPI_INTEGER,MPI_INTEGER8,MPI_DOUBLE_PRECISION, &
      MPI_MAX,MPI_MIN,MPI_SUM
    type(c_ptr),intent(in) :: handle
    integer(c_int),intent(in) :: regions,first(3,regions),last(3,regions),comm
    real(c_double),intent(in) :: duration
    logical,intent(in) :: collect,packed,activity
    type(air5_device_report),intent(out) :: local
    type(air5_chemistry_global_report),intent(out) :: global
    integer(c_int) :: local_status,local_status_counts(5),global_status_counts(5),region
    integer(c_int64_t) :: local_activity(2),global_activity(2)
    real(c_double) :: local_minimums(3),local_maximums(2),local_extrema(5),global_extrema(5)
    integer :: ierr

    local%status=0;local%max_counts=0;local%failure=0;local%skipped=0;local%total=0
    local%minimum_bits=0;local%maximum_bits=0;local%failure_q=0;local%failure_carry=0
#ifdef ASTR_AIR5_WORKLOAD_PROBE
    local%sum_counts=0
#endif
    global%collected=.false.;global%status=0;global%max_counts=0
    global%skipped=0;global%total=0;global%minimums=0;global%maximums=0
    call check_phase_api(air5_chemistry_begin(handle),comm)
    do region=1,regions
      call check_phase_api(air5_chemistry_region(handle,first(:,region),last(:,region),duration, &
        merge(1_c_int,0_c_int,collect)),comm)
    enddo
    if(collect)then
      call check_phase_api(air5_chemistry_report(handle,local),comm)
      local_status=local%status
    else
      call check_phase_api(air5_chemistry_status(handle,local_status),comm)
      local%status=local_status
    endif
    if(packed.and.collect)then
      local_status_counts=[local_status,local%max_counts]
      call MPI_Allreduce(local_status_counts,global_status_counts,5,MPI_INTEGER,MPI_MAX,int(comm),ierr)
      call check_phase_api(int(ierr,c_int),comm)
      status=global_status_counts(1);global%max_counts=global_status_counts(2:5)
    else
      call MPI_Allreduce(local_status,status,1,MPI_INTEGER,MPI_MAX,int(comm),ierr)
      call check_phase_api(int(ierr,c_int),comm)
    endif
    global%status=status
    if(status/=0)then
      if(.not.collect)call check_phase_api(air5_chemistry_report(handle,local),comm)
      return
    endif
    if(.not.collect)return
    if(activity)then
      local_activity=[local%skipped,local%total]
      call MPI_Allreduce(local_activity,global_activity,2,MPI_INTEGER8,MPI_SUM,int(comm),ierr)
      call check_phase_api(int(ierr,c_int),comm)
      global%skipped=global_activity(1);global%total=global_activity(2)
    endif
    local_minimums=transfer(local%minimum_bits,local_minimums)
    local_maximums=transfer(local%maximum_bits,local_maximums)
    if(packed)then
      local_extrema=[-local_minimums,local_maximums]
      call MPI_Allreduce(local_extrema,global_extrema,5,MPI_DOUBLE_PRECISION,MPI_MAX,int(comm),ierr)
      call check_phase_api(int(ierr,c_int),comm)
      global%minimums=-global_extrema(1:3);global%maximums=global_extrema(4:5)
    else
      call MPI_Allreduce(local_minimums,global%minimums,3,MPI_DOUBLE_PRECISION,MPI_MIN,int(comm),ierr)
      call check_phase_api(int(ierr,c_int),comm)
      call MPI_Allreduce(local_maximums,global%maximums,2,MPI_DOUBLE_PRECISION,MPI_MAX,int(comm),ierr)
      call check_phase_api(int(ierr,c_int),comm)
      call MPI_Allreduce(local%max_counts,global%max_counts,4,MPI_INTEGER,MPI_MAX,int(comm),ierr)
      call check_phase_api(int(ierr,c_int),comm)
    endif
    global%collected=.true.
  end function

  subroutine check_phase_api(code,comm)
    use mpi,only:MPI_Abort
    integer(c_int),intent(in) :: code,comm
    integer :: ignored
    if(code/=0)call MPI_Abort(int(comm),int(code),ignored)
  end subroutine
end module
