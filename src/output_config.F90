module output_config
  use iso_fortran_env, only: int64,real64,iostat_end
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  integer,parameter,public :: output_format_version=1,output_slice_capacity=256
  public :: output_product_options,output_options,read_output_options,validate_output_grid
  public :: output_product_derived_indices

  type :: output_product_options
    logical :: enabled=.false.,initial_frame=.false.,final_frame=.false.
    logical :: velocity_gradient=.false.,vorticity=.false.,qcriterion=.false.
    character(16) :: mode='steps',fields='basic'
    integer(int64) :: interval_steps=0
    real(real64) :: interval_time=0
  end type

  type :: output_options
    integer :: format_version=output_format_version,keep=2
    character(1024) :: directory='outdat/output',restore_directory=''
    character(16) :: restart_output='saved'
    integer(int64) :: buffer_bytes=67108864_int64,host_budget_bytes=0,device_budget_bytes=0
    integer(int64) :: device_reserve_bytes=0
    type(output_product_options) :: checkpoint=output_product_options(final_frame=.true.)
    type(output_product_options) :: volume,slices
    integer(int64) :: i_indices(output_slice_capacity)=-1
    integer(int64) :: j_indices(output_slice_capacity)=-1
    integer(int64) :: k_indices(output_slice_capacity)=-1
  end type
contains
  pure subroutine output_product_derived_indices(options,indices,n)
    type(output_product_options),intent(in) :: options
    integer,intent(out) :: indices(14),n
    logical :: chosen(14)
    integer :: m
    indices=0; n=0; chosen=.false.
    if(.not.options%enabled) return
    if(options%velocity_gradient) chosen(:9)=.true.
    ! Q uses the established full-strain definition; record its divergence companion.
    if(options%qcriterion) chosen(10:11)=.true.
    if(options%vorticity) chosen(12:14)=.true.
    do m=1,14
      if(.not.chosen(m)) cycle
      n=n+1; indices(n)=m
    enddo
  end subroutine

  function syntax_line(line) result(clean)
    character(*),intent(in) :: line
    character(len(line)) :: clean
    integer :: n,i,c
    clean=adjustl(line)
    n=index(clean,'!')
    if(n>0) clean(n:)=' '
    do i=1,len_trim(clean)
      c=iachar(clean(i:i))
      if(c==13.or.c==9) clean(i:i)=' '
      if(c>=iachar('A').and.c<=iachar('Z')) clean(i:i)=achar(c+32)
    enddo
    clean=adjustl(clean)
  end function

  subroutine expect_group(unit,name,ok,message)
    integer,intent(in) :: unit
    character(*),intent(in) :: name
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    character(4096) :: line
    integer :: status
    ok=.false.
    message='expected &'//name//' on its own line'
    do
      read(unit,'(A)',iostat=status) line
      if(status/=0) return
      line=syntax_line(line)
      if(len_trim(line)>0) exit
    enddo
    if(trim(line)/='&'//name) return
    backspace(unit,iostat=status)
    ok=status==0
  end subroutine

  subroutine check_group_end(unit,ok,message)
    integer,intent(in) :: unit
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    integer :: status
    character(4096) :: line
    ok=.false.
    message='namelist terminator / must be on its own line'
    backspace(unit,iostat=status)
    if(status/=0) return
    read(unit,'(A)',iostat=status) line
    if(status/=0) return
    ok=trim(syntax_line(line))=='/'
  end subroutine

  subroutine read_output_options(filename,options,ok,message)
    character(*),intent(in) :: filename
    type(output_options),intent(inout) :: options
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    type(output_options) :: candidate
    integer :: format_version,unit,status,closed,kind,keep
    integer(int64) :: buffer_bytes,host_budget_bytes,device_budget_bytes,device_reserve_bytes
    character(1024) :: directory,restore_directory
    character(16) :: restart_output
    character(4096) :: line
    character(10),parameter :: groups(3)=[character(10)::'checkpoint','volume','slices']
    type(output_product_options) :: product_options
    integer(int64) :: indices(output_slice_capacity,3)
    logical :: opened,active
    namelist /output/ format_version,directory,restore_directory,restart_output, &
      buffer_bytes,host_budget_bytes,device_budget_bytes,device_reserve_bytes

    ok=.false.; opened=.false.; message=''
    format_version=candidate%format_version
    directory=candidate%directory; restore_directory=candidate%restore_directory
    restart_output=candidate%restart_output
    buffer_bytes=candidate%buffer_bytes
    host_budget_bytes=candidate%host_budget_bytes; device_budget_bytes=candidate%device_budget_bytes
    device_reserve_bytes=candidate%device_reserve_bytes
    open(newunit=unit,file=filename,status='old',action='read',iostat=status,iomsg=message)
    if(status/=0) return
    opened=.true.
    call expect_group(unit,'output',ok,message)
    if(.not.ok) goto 900
    read(unit,nml=output,iostat=status,iomsg=message)
    ok=status==0
    if(.not.ok) goto 900
    call check_group_end(unit,ok,message)
    if(.not.ok) goto 900
    candidate%format_version=format_version
    candidate%directory=directory; candidate%restore_directory=restore_directory
    candidate%restart_output=restart_output
    candidate%buffer_bytes=buffer_bytes
    candidate%host_budget_bytes=host_budget_bytes; candidate%device_budget_bytes=device_budget_bytes
    candidate%device_reserve_bytes=device_reserve_bytes
    do kind=1,3
      call read_product(unit,trim(groups(kind)),product_options,keep,indices,ok,message)
      if(.not.ok) goto 900
      select case(kind)
      case(1)
        candidate%checkpoint=product_options; candidate%keep=keep
      case(2)
        candidate%volume=product_options
      case(3)
        candidate%slices=product_options
        candidate%i_indices=indices(:,1); candidate%j_indices=indices(:,2); candidate%k_indices=indices(:,3)
      end select
    enddo
    do
      read(unit,'(A)',iostat=status) line
      if(status==iostat_end) exit
      ok=status==0
      message='cannot read output configuration tail'
      if(.not.ok) goto 900
      ok=len_trim(syntax_line(line))==0
      message='unexpected content after &slices'
      if(.not.ok) goto 900
    enddo
    ok=.false.
    message='unsupported output format_version'
    if(format_version/=output_format_version) goto 900
    message='output paths exceed character capacity'
    if(len_trim(directory)>=len(directory).or.len_trim(restore_directory)>=len(restore_directory)) goto 900
    message='restart_output must be saved or override'
    if(restart_output/='saved'.and.restart_output/='override') goto 900
    message='restart output override requires restore_directory'
    if(restart_output=='override'.and.len_trim(restore_directory)==0) goto 900
    message='buffer must be positive and budgets/reserve must not be negative'
    if(buffer_bytes<=0.or.min(host_budget_bytes,device_budget_bytes,device_reserve_bytes)<0) goto 900
    active=candidate%checkpoint%enabled.or.candidate%volume%enabled.or.candidate%slices%enabled
    if(active) then
      message='enabled output requires directory and host budget covering one buffer'
      if(len_trim(directory)==0.or.host_budget_bytes<buffer_bytes) goto 900
    endif
    ok=.true.
900 continue
    if(opened) then
      close(unit,iostat=closed)
      if(closed/=0) then
        ok=.false.; message='cannot close output configuration'
      endif
    endif
    if(ok) then
      options=candidate
      message=''
    endif
  end subroutine

  subroutine read_product(unit,name,options,keep,indices,ok,message)
    integer,intent(in) :: unit
    character(*),intent(in) :: name
    type(output_product_options),intent(out) :: options
    integer,intent(out) :: keep
    integer(int64),intent(out) :: indices(output_slice_capacity,3)
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    logical :: enabled,initial_frame,final_frame,velocity_gradient,vorticity,qcriterion
    character(16) :: mode,fields
    integer(int64) :: interval_steps,i_indices(output_slice_capacity), &
      j_indices(output_slice_capacity),k_indices(output_slice_capacity)
    real(real64) :: interval_time
    integer :: status,axis
    namelist /checkpoint/ enabled,mode,interval_steps,interval_time,initial_frame,keep
    namelist /volume/ enabled,mode,interval_steps,interval_time,initial_frame,final_frame, &
      fields,velocity_gradient,vorticity,qcriterion
    namelist /slices/ enabled,mode,interval_steps,interval_time,initial_frame,final_frame, &
      fields,velocity_gradient,vorticity,qcriterion,i_indices,j_indices,k_indices

    options=output_product_options()
    enabled=.false.; initial_frame=.false.; final_frame=name=='checkpoint'
    velocity_gradient=.false.; vorticity=.false.; qcriterion=.false.
    mode='steps'; fields='basic'; interval_steps=0; interval_time=0; keep=2
    i_indices=-1; j_indices=-1; k_indices=-1; indices=-1
    call expect_group(unit,name,ok,message)
    if(.not.ok) return
    select case(name)
    case('checkpoint')
      read(unit,nml=checkpoint,iostat=status,iomsg=message)
    case('volume')
      read(unit,nml=volume,iostat=status,iomsg=message)
    case('slices')
      read(unit,nml=slices,iostat=status,iomsg=message)
    case default
      ok=.false.; message='unknown output product'; return
    end select
    ok=status==0
    if(.not.ok) return
    call check_group_end(unit,ok,message)
    if(.not.ok) return
    ok=.false.
    message=name//': invalid interval'
    if(.not.ieee_is_finite(interval_time)) return
    if(interval_steps<0.or.interval_time<0) return
    message=name//': mode must be steps or time, with only its interval set'
    select case(mode)
    case('steps')
      if(interval_time/=0.or.(enabled.and.interval_steps==0)) return
    case('time')
      if(interval_steps/=0.or.(enabled.and.interval_time==0)) return
    case default
      return
    end select
    message='checkpoint: keep must be 1 or 2'
    if(keep/=1.and.keep/=2) return
    message=name//': fields must be basic; select derived quantities with named flags'
    if(fields/='basic') return
    indices(:,1)=i_indices; indices(:,2)=j_indices; indices(:,3)=k_indices
    message='slices: indices must be nonnegative; -1 means unused'
    if(any(indices < -1_int64)) return
    do axis=1,3
      call unique_indices(indices(:,axis))
    enddo
    message='slices: enabled output requires at least one global index'
    if(name=='slices'.and.enabled.and.all(indices==-1)) return
    options%enabled=enabled; options%initial_frame=initial_frame; options%final_frame=final_frame
    options%mode=mode; options%interval_steps=interval_steps; options%interval_time=interval_time
    options%fields=fields; options%velocity_gradient=velocity_gradient
    options%vorticity=vorticity; options%qcriterion=qcriterion
    ok=.true.; message=''
  end subroutine

  subroutine unique_indices(indices)
    integer(int64),intent(inout) :: indices(:)
    integer(int64) :: result(size(indices))
    integer :: i,n
    result=-1; n=0
    do i=1,size(indices)
      if(indices(i)<0) cycle
      if(any(result(:n)==indices(i))) cycle
      n=n+1; result(n)=indices(i)
    enddo
    indices=result
  end subroutine

  subroutine validate_output_grid(options,global_upper,ok,message)
    type(output_options),intent(in) :: options
    integer(int64),intent(in) :: global_upper(3)
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    ok=.false.
    message='global upper indices must be nonnegative'
    if(any(global_upper<0)) return
    message='slice global index lies outside the grid'
    if(any(options%i_indices>global_upper(1)).or.any(options%j_indices>global_upper(2)).or. &
       any(options%k_indices>global_upper(3))) return
    ok=.true.; message=''
  end subroutine
end module

module output_config_collective
  use mpi
  use iso_fortran_env, only: int64,real64
  use output_config
  implicit none
  private
  public :: read_output_options_collective
contains
  subroutine require_mpi(status,comm)
    integer,intent(in) :: status,comm
    integer :: abort_status
    if(status==MPI_SUCCESS) return
    call MPI_Abort(comm,status,abort_status)
    error stop 'output configuration MPI operation failed'
  end subroutine

  subroutine read_output_options_collective(filename,options,comm,ok,message)
    character(*),intent(in) :: filename
    type(output_options),intent(inout) :: options
    integer,intent(in) :: comm
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    type(output_options) :: candidate
    type(output_product_options) :: products(3)
    character(1024) :: root_file,wire_message,paths(2)
    character(16) :: labels(7)
    integer(int64) :: integers(9),indices(output_slice_capacity,3)
    real(real64) :: intervals(3)
    logical :: flags(6,3)
    integer :: rank,status,bad,total,integer_type,real_type,i

    call MPI_Comm_rank(comm,rank,status)
    call require_mpi(status,comm)
    root_file=filename
    call MPI_Bcast(root_file,len(root_file),MPI_CHARACTER,0,comm,status)
    call require_mpi(status,comm)
    bad=merge(0,1,len_trim(filename)>0.and.len_trim(filename)<len(root_file).and.filename==root_file)
    call MPI_Allreduce(bad,total,1,MPI_INTEGER,MPI_MAX,comm,status)
    call require_mpi(status,comm)
    ok=total==0
    message='output configuration filenames differ or exceed character capacity'
    if(.not.ok) return
    wire_message=''
    if(rank==0) call read_output_options(trim(filename),candidate,ok,wire_message)
    call MPI_Bcast(ok,1,MPI_LOGICAL,0,comm,status)
    call require_mpi(status,comm)
    call MPI_Bcast(wire_message,len(wire_message),MPI_CHARACTER,0,comm,status)
    call require_mpi(status,comm)
    message=wire_message
    if(.not.ok) return

    if(rank==0) then
      products=[candidate%checkpoint,candidate%volume,candidate%slices]
      integers=[int(candidate%format_version,int64),int(candidate%keep,int64),candidate%buffer_bytes, &
        candidate%host_budget_bytes,candidate%device_budget_bytes,products%interval_steps,candidate%device_reserve_bytes]
      intervals=products%interval_time
      labels(1)=candidate%restart_output
      paths=[candidate%directory,candidate%restore_directory]
      indices(:,1)=candidate%i_indices; indices(:,2)=candidate%j_indices; indices(:,3)=candidate%k_indices
      do i=1,3
        flags(:,i)=[products(i)%enabled,products(i)%initial_frame,products(i)%final_frame, &
          products(i)%velocity_gradient,products(i)%vorticity,products(i)%qcriterion]
        labels(2*i)=products(i)%mode; labels(2*i+1)=products(i)%fields
      enddo
    endif
    ! Typed wire fields avoid compiler padding and derived-type byte layout.
    call MPI_Type_match_size(MPI_TYPECLASS_INTEGER,storage_size(integers(1))/8,integer_type,status)
    call require_mpi(status,comm)
    call MPI_Type_match_size(MPI_TYPECLASS_REAL,storage_size(intervals(1))/8,real_type,status)
    call require_mpi(status,comm)
    call MPI_Bcast(integers,size(integers),integer_type,0,comm,status)
    call require_mpi(status,comm)
    call MPI_Bcast(intervals,size(intervals),real_type,0,comm,status)
    call require_mpi(status,comm)
    call MPI_Bcast(indices,size(indices),integer_type,0,comm,status)
    call require_mpi(status,comm)
    call MPI_Bcast(flags,size(flags),MPI_LOGICAL,0,comm,status)
    call require_mpi(status,comm)
    call MPI_Bcast(labels,size(labels)*len(labels),MPI_CHARACTER,0,comm,status)
    call require_mpi(status,comm)
    call MPI_Bcast(paths,size(paths)*len(paths),MPI_CHARACTER,0,comm,status)
    call require_mpi(status,comm)

    candidate%format_version=int(integers(1)); candidate%keep=int(integers(2))
    candidate%buffer_bytes=integers(3); candidate%host_budget_bytes=integers(4)
    candidate%device_budget_bytes=integers(5)
    candidate%device_reserve_bytes=integers(9)
    candidate%restart_output=labels(1)
    candidate%directory=paths(1); candidate%restore_directory=paths(2)
    candidate%i_indices=indices(:,1); candidate%j_indices=indices(:,2); candidate%k_indices=indices(:,3)
    do i=1,3
      products(i)%interval_steps=integers(5+i); products(i)%interval_time=intervals(i)
      products(i)%mode=labels(2*i); products(i)%fields=labels(2*i+1)
      products(i)%enabled=flags(1,i); products(i)%initial_frame=flags(2,i); products(i)%final_frame=flags(3,i)
      products(i)%velocity_gradient=flags(4,i); products(i)%vorticity=flags(5,i); products(i)%qcriterion=flags(6,i)
    enddo
    candidate%checkpoint=products(1); candidate%volume=products(2); candidate%slices=products(3)
    options=candidate
    ok=.true.; message=''
  end subroutine
end module
