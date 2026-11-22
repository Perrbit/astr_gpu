module compact_line_cpp
  use iso_c_binding, only: c_ptr,c_double,c_int,c_int64_t,c_size_t,c_null_ptr,c_associated,c_loc
  use mpi
  use ieee_arithmetic, only: ieee_is_finite
  use astr_precision_mpi, only: mpi_real_type
  use compact_line_solver, only: compact_matrix,compact_block_factors,prepare_compact_blocks,compact_endpoint_responses
  use compact_operator, only: compact_line_partition
  use device_state, only: device_view,c_device_real_kind
  implicit none
  private
  public :: compact_cpp_plan,prepare_compact_cpp,solve_compact_cpp,release_compact_cpp
  public :: upload_compact_rhs_cpp,download_compact_solution_cpp,download_compact_rhs_cpp,compact_views_cpp
  public :: compact_distributed_cpp_plan,prepare_distributed_compact_cpp,solve_distributed_compact_cpp
  public :: restore_distributed_compact_cpp,release_distributed_compact_cpp,compact_distributed_transfer_cpp
  public :: compact_distributed_views_cpp,compact_distributed_metrics_cpp
  public :: compact_distributed_phase_enable_cpp,compact_distributed_phase_seconds_cpp
  public :: compact_distributed_local_cpp,compact_distributed_begin_cpp,compact_distributed_finish_cpp
  public :: compact_stencil_cpp,compact_field_rhs_cpp,compact_field_commit_cpp
  public :: compact_keep_prepare_cpp,compact_keep_rhs_cpp,compact_keep_commit_cpp
  public :: compact_field_commit_wall_band_cpp

  type,bind(C) :: compact_stencil_cpp
    integer(c_int) :: axis,first,total,periodic,kind,start,component_offset
    ! Output ownership, not a restriction of the globally coupled system.
    integer(c_int) :: active_first(3),active_last(3)
    real(c_device_real_kind) :: c6(0:3)=0,c8(0:4)=0,c10(0:5)=0,boundary(0:2,0:6)=0
  end type

  type,bind(C) :: compact_factor_view
    integer(c_int) :: n,tile_capacity,tiles,levels,reduced_n,reduced_levels,cyclic
    type(c_ptr) :: first,last,owner,local_levels
    type(c_ptr) :: initial,left,right,inverse,edge_lower,edge_upper,response_left,response_right
    type(c_ptr) :: reduced_initial,reduced_left,reduced_right,reduced_inverse,cyclic_response
    real(c_device_real_kind) :: correction(4)
  end type

  type :: compact_cpp_plan
    type(c_ptr) :: handle=c_null_ptr
    integer :: n=0,capacity=0
    integer(c_size_t) :: device_bytes=0
  end type

  type,bind(C) :: distributed_factor_view
    integer(c_int) :: communicator,levels,cyclic,mode,nodes
    type(c_ptr) :: response,coefficient
    real(c_device_real_kind) :: edges(2)
  end type

  type :: compact_distributed_cpp_plan
    type(c_ptr) :: handle=c_null_ptr
    type(compact_line_partition) :: partition
    integer :: comm=MPI_COMM_NULL,n=0,capacity=0,nodes=0
    integer(c_size_t) :: device_bytes=0,host_bytes=0
  end type

  interface
    integer(c_int) function compact_field_commit_wall_band_cpp(parent,line,stencil,nrhs) &
        bind(C,name='astr_compact_field_commit_wall_band')
      import c_ptr,c_int,compact_stencil_cpp
      type(c_ptr),value :: parent,line
      type(compact_stencil_cpp),intent(in) :: stencil
      integer(c_int),value :: nrhs
    end function
    integer(c_int) function compact_field_rhs_cpp(parent,line,stencil,nrhs) bind(C,name='astr_compact_field_rhs')
      import c_ptr,c_int,compact_stencil_cpp
      type(c_ptr),value :: parent,line
      type(compact_stencil_cpp),intent(in) :: stencil
      integer(c_int),value :: nrhs
    end function
    integer(c_int) function compact_field_commit_cpp(parent,line,stencil,nrhs) bind(C,name='astr_compact_field_commit')
      import c_ptr,c_int,compact_stencil_cpp
      type(c_ptr),value :: parent,line
      type(compact_stencil_cpp),intent(in) :: stencil
      integer(c_int),value :: nrhs
    end function
    integer(c_int) function compact_keep_prepare_cpp(parent,terms,bases,left,right,component,basis,weight) &
        bind(C,name='astr_device_keep_prepare')
      import c_ptr,c_int,c_device_real_kind
      type(c_ptr),value :: parent
      integer(c_int),value :: terms,bases
      integer(c_int),intent(in) :: left(*),right(*),component(*),basis(*)
      real(c_device_real_kind),intent(in) :: weight(*)
    end function
    integer(c_int) function compact_keep_rhs_cpp(parent,line,stencil,nrhs,gamma) bind(C,name='astr_device_keep_compact_rhs')
      import c_ptr,c_int,c_device_real_kind,compact_stencil_cpp
      type(c_ptr),value :: parent,line
      type(compact_stencil_cpp),intent(in) :: stencil
      integer(c_int),value :: nrhs
      real(c_device_real_kind),value :: gamma
    end function
    integer(c_int) function compact_keep_commit_cpp(parent,line,stencil,nrhs,gamma) bind(C,name='astr_device_keep_compact_commit')
      import c_ptr,c_int,c_device_real_kind,compact_stencil_cpp
      type(c_ptr),value :: parent,line
      type(compact_stencil_cpp),intent(in) :: stencil
      integer(c_int),value :: nrhs
      real(c_device_real_kind),value :: gamma
    end function
    integer(c_int) function create_line(parent,factors,capacity,handle,device_bytes) bind(C,name='astr_compact_line_create')
      import c_ptr,c_int,c_size_t,compact_factor_view
      type(c_ptr),value :: parent
      type(compact_factor_view),intent(in) :: factors
      integer(c_int),value :: capacity
      type(c_ptr),intent(out) :: handle
      integer(c_size_t),intent(out) :: device_bytes
    end function
    integer(c_int) function solve_line(handle,nrhs) bind(C,name='astr_compact_line_solve')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),value :: nrhs
    end function
    integer(c_int) function transfer_line(handle,mode,host,nrhs) bind(C,name='astr_compact_line_transfer')
      import c_ptr,c_int
      type(c_ptr),value :: handle,host
      integer(c_int),value :: mode,nrhs
    end function
    integer(c_int) function line_views(handle,rhs,solution) bind(C,name='astr_compact_line_views')
      import c_ptr,c_int,device_view
      type(c_ptr),value :: handle
      type(device_view),intent(out) :: rhs,solution
    end function
    integer(c_int) function destroy_line(handle) bind(C,name='astr_compact_line_destroy')
      import c_ptr,c_int
      type(c_ptr),intent(inout) :: handle
    end function
    integer(c_int) function create_distributed(local,factors,handle,device_bytes,host_bytes) &
        bind(C,name='astr_compact_distributed_create')
      import c_ptr,c_int,c_size_t,distributed_factor_view
      type(c_ptr),intent(inout) :: local
      type(distributed_factor_view),intent(in) :: factors
      type(c_ptr),intent(out) :: handle
      integer(c_size_t),intent(out) :: device_bytes,host_bytes
    end function
    integer(c_int) function solve_distributed(handle,nrhs) bind(C,name='astr_compact_distributed_solve')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),value :: nrhs
    end function
    integer(c_int) function compact_distributed_local_cpp(handle,nrhs) bind(C,name='astr_compact_distributed_local')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),value :: nrhs
    end function
    integer(c_int) function compact_distributed_begin_cpp(handle,nrhs) bind(C,name='astr_compact_distributed_begin')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),value :: nrhs
    end function
    integer(c_int) function compact_distributed_finish_cpp(handle,nrhs) bind(C,name='astr_compact_distributed_finish')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),value :: nrhs
    end function
    integer(c_int) function restore_distributed(handle,nrhs) bind(C,name='astr_compact_distributed_restore')
      import c_ptr,c_int
      type(c_ptr),value :: handle
      integer(c_int),value :: nrhs
    end function
    integer(c_int) function transfer_distributed(handle,mode,host,nrhs) bind(C,name='astr_compact_distributed_transfer')
      import c_ptr,c_int
      type(c_ptr),value :: handle,host
      integer(c_int),value :: mode,nrhs
    end function
    integer(c_int) function distributed_views(handle,rhs,solution,nodes) bind(C,name='astr_compact_distributed_views')
      import c_ptr,c_int,device_view
      type(c_ptr),value :: handle
      type(device_view),intent(out) :: rhs,solution,nodes
    end function
    integer(c_int) function distributed_metrics(handle,values,reset) bind(C,name='astr_compact_distributed_metrics')
      import c_ptr,c_int,c_int64_t
      type(c_ptr),value :: handle
      integer(c_int64_t),intent(out) :: values(2)
      integer(c_int),value :: reset
    end function
    integer(c_int) function distributed_phase_enable(handle) bind(C,name='astr_compact_distributed_phase_enable')
      import c_ptr,c_int
      type(c_ptr),value :: handle
    end function
    subroutine compact_distributed_phase_seconds_cpp(handle,values) bind(C,name='astr_compact_distributed_phase_seconds')
      import c_ptr,c_double
      type(c_ptr),value :: handle
      real(c_double),intent(out) :: values(5)
    end subroutine
    integer(c_int) function destroy_distributed(handle) bind(C,name='astr_compact_distributed_destroy')
      import c_ptr,c_int
      type(c_ptr),intent(inout) :: handle
    end function
  end interface
contains
  subroutine check(status,operation)
    integer(c_int),intent(in) :: status
    character(*),intent(in) :: operation
    if(status==0) return
    print '(a,a,a,i0)', 'compact C++ ',operation,' status=',status
    error stop 'compact C++ device operation failed'
  end subroutine

  subroutine prepare_compact_cpp(parent,matrix,tile_capacity,capacity,plan)
    type(c_ptr),intent(in) :: parent
    type(compact_matrix),intent(in) :: matrix
    integer,intent(in) :: tile_capacity,capacity
    type(compact_cpp_plan),intent(inout) :: plan
    type(compact_block_factors),target :: blocks
    type(compact_factor_view) :: view
    real(c_device_real_kind),allocatable,target :: lower(:),upper(:)
    if(c_associated(plan%handle)) error stop 'release compact C++ plan before preparation'
    call prepare_compact_blocks(matrix,tile_capacity,blocks)
    allocate(lower(blocks%tiles),upper(blocks%tiles))
    lower=matrix%lower(blocks%last);upper=matrix%upper(blocks%first)
    view%n=blocks%n;view%tile_capacity=tile_capacity;view%tiles=blocks%tiles
    view%levels=blocks%max_levels;view%reduced_n=blocks%reduced_n
    view%reduced_levels=blocks%reduced%levels;view%cyclic=merge(1,0,matrix%cyclic)
    view%first=c_loc(blocks%first);view%last=c_loc(blocks%last)
    view%owner=c_loc(blocks%owner);view%local_levels=c_loc(blocks%levels)
    view%initial=c_loc(blocks%initial_inverse);view%left=c_loc(blocks%left)
    view%right=c_loc(blocks%right);view%inverse=c_loc(blocks%inverse)
    view%edge_lower=c_loc(lower);view%edge_upper=c_loc(upper)
    view%response_left=c_loc(blocks%response_left);view%response_right=c_loc(blocks%response_right)
    view%reduced_initial=c_loc(blocks%reduced%initial_inverse)
    view%reduced_left=c_loc(blocks%reduced%left);view%reduced_right=c_loc(blocks%reduced%right)
    view%reduced_inverse=c_loc(blocks%reduced%inverse);view%cyclic_response=c_loc(blocks%cyclic_response)
    view%correction=[blocks%correction_inverse(1,1),blocks%correction_inverse(1,2), &
                     blocks%correction_inverse(2,1),blocks%correction_inverse(2,2)]
    call check(create_line(parent,view,int(capacity,c_int),plan%handle,plan%device_bytes),'create')
    plan%n=matrix%n;plan%capacity=capacity
  end subroutine

  subroutine compact_views_cpp(plan,rhs,solution)
    type(compact_cpp_plan),intent(in) :: plan
    type(device_view),intent(out) :: rhs,solution
    call check(line_views(plan%handle,rhs,solution),'views')
  end subroutine

  subroutine upload_compact_rhs_cpp(plan,rhs,nrhs)
    type(compact_cpp_plan),intent(in) :: plan
    real(c_device_real_kind),contiguous,target,intent(in) :: rhs(:,:)
    integer,intent(in) :: nrhs
    if(size(rhs,1)/=plan%n.or.size(rhs,2)<nrhs.or.nrhs<1) error stop 'compact RHS host shape'
    call check(transfer_line(plan%handle,0_c_int,c_loc(rhs),int(nrhs,c_int)),'upload RHS')
  end subroutine

  subroutine solve_compact_cpp(plan,nrhs)
    type(compact_cpp_plan),intent(in) :: plan
    integer,intent(in) :: nrhs
    call check(solve_line(plan%handle,int(nrhs,c_int)),'solve')
  end subroutine

  subroutine download_compact_solution_cpp(plan,values,nrhs)
    type(compact_cpp_plan),intent(in) :: plan
    real(c_device_real_kind),contiguous,target,intent(out) :: values(:,:)
    integer,intent(in) :: nrhs
    if(size(values,1)/=plan%n.or.size(values,2)<nrhs.or.nrhs<1) error stop 'compact solution host shape'
    call check(transfer_line(plan%handle,2_c_int,c_loc(values),int(nrhs,c_int)),'download solution')
  end subroutine

  subroutine download_compact_rhs_cpp(plan,values,nrhs)
    type(compact_cpp_plan),intent(in) :: plan
    real(c_device_real_kind),contiguous,target,intent(out) :: values(:,:)
    integer,intent(in) :: nrhs
    if(size(values,1)/=plan%n.or.size(values,2)<nrhs.or.nrhs<1) error stop 'compact RHS observation shape'
    call check(transfer_line(plan%handle,1_c_int,c_loc(values),int(nrhs,c_int)),'read RHS')
  end subroutine

  subroutine release_compact_cpp(plan)
    type(compact_cpp_plan),intent(inout) :: plan
    call check(destroy_line(plan%handle),'destroy')
    plan%n=0;plan%capacity=0;plan%device_bytes=0
  end subroutine

  function inverse_interface(a) result(inverse)
    real(c_device_real_kind),intent(in) :: a(2,2)
    real(c_device_real_kind) :: inverse(2,2),det
    det=a(1,1)*a(2,2)-a(1,2)*a(2,1)
    if(det==0.or..not.ieee_is_finite(det)) error stop 'singular compact interface block'
    inverse(1,1)=a(2,2)/det;inverse(1,2)=-a(1,2)/det
    inverse(2,1)=-a(2,1)/det;inverse(2,2)=a(1,1)/det
    if(.not.all(ieee_is_finite(inverse))) error stop 'nonfinite compact interface inverse'
  end function

  subroutine prepare_distributed_compact_cpp(parent,matrix,edge,partition,comm,capacity,mode,plan)
    type(c_ptr),intent(in) :: parent
    type(compact_matrix),intent(in) :: matrix
    real(c_device_real_kind),intent(in) :: edge(2)
    type(compact_line_partition),intent(in) :: partition
    integer,intent(in) :: comm,capacity
    integer(c_int),intent(in) :: mode
    type(compact_distributed_cpp_plan),intent(inout) :: plan
    type(compact_cpp_plan) :: local
    type(distributed_factor_view) :: view
    real(c_device_real_kind),allocatable,target :: response(:,:),coefficient(:,:,:,:)
    real(c_device_real_kind) :: base_edge(2),left(2,2),right(2,2),identity(2,2),inv(2,2),wl(2,2),wr(2,2)
    real(c_device_real_kind) :: send(2,2,2),lo(2,2,2),hi(2,2,2),diagonal(2,2)
    integer :: rank,ranks,n,stride,levels,level,left_rank,right_rank,ierr,mode_min,mode_max
    if(c_associated(plan%handle)) error stop 'release distributed compact plan before preparation'
    call MPI_Comm_dup(comm,plan%comm,ierr);call check(int(ierr,c_int),'duplicate interface communicator')
    call MPI_Comm_rank(plan%comm,rank,ierr);call check(int(ierr,c_int),'interface rank')
    call MPI_Comm_size(plan%comm,ranks,ierr);call check(int(ierr,c_int),'interface size')
    call MPI_Allreduce(mode,mode_min,1,MPI_INTEGER,MPI_MIN,plan%comm,ierr)
    call check(int(ierr,c_int),'interface mode minimum')
    call MPI_Allreduce(mode,mode_max,1,MPI_INTEGER,MPI_MAX,plan%comm,ierr)
    call check(int(ierr,c_int),'interface mode maximum')
    if(mode_min/=mode_max.or.mode_min<0.or.mode_max>1) error stop 'inconsistent compact interface transport'
    n=matrix%n
    if(n<2.or.n/=partition%owned.or.capacity<1.or.rank/=partition%rank.or.ranks/=partition%ranks) &
      error stop 'invalid distributed compact core/partition'
    if(matrix%cyclic.or.matrix%lower(1)/=0.or.matrix%upper(n)/=0) &
      error stop 'distributed compact local core includes off-rank corners'
    levels=0;stride=1
    do while(stride<ranks)
      levels=levels+1;stride=stride*2
    enddo
    allocate(response(n,2),coefficient(2,2,3,levels))
    base_edge=edge
    if(rank==0) base_edge(1)=0
    if(rank==ranks-1) base_edge(2)=0
    call compact_endpoint_responses(matrix,base_edge,response)
    identity=0;identity(1,1)=1;identity(2,2)=1
    left=0;right=0
    left(:,2)=response([1,n],1);right(:,1)=response([1,n],2)
    stride=1
    do level=1,levels
      left_rank=rank-stride;right_rank=rank+stride
      if(left_rank<0) left_rank=MPI_PROC_NULL
      if(right_rank>=ranks) right_rank=MPI_PROC_NULL
      send(:,:,1)=left;send(:,:,2)=right;lo=0;hi=0
      call MPI_Sendrecv(send,8,mpi_real_type,left_rank,24011,hi,8,mpi_real_type,right_rank,24011, &
        plan%comm,MPI_STATUS_IGNORE,ierr)
      call check(int(ierr,c_int),'upper interface coefficients')
      call MPI_Sendrecv(send,8,mpi_real_type,right_rank,24012,lo,8,mpi_real_type,left_rank,24012, &
        plan%comm,MPI_STATUS_IGNORE,ierr)
      call check(int(ierr,c_int),'lower interface coefficients')
      diagonal=identity-matmul(left,lo(:,:,2))-matmul(right,hi(:,:,1))
      inv=inverse_interface(diagonal);wl=matmul(inv,left);wr=matmul(inv,right)
      coefficient(:,:,1,level)=inv;coefficient(:,:,2,level)=wl;coefficient(:,:,3,level)=wr
      left=-matmul(wl,lo(:,:,1));right=-matmul(wr,hi(:,:,2));stride=stride*2
    enddo
    call prepare_compact_cpp(parent,matrix,128,max(2,capacity),local)
    view%communicator=int(plan%comm,c_int);view%levels=levels
    view%cyclic=merge(1,0,partition%periodic);view%mode=mode
    view%nodes=partition%local_intervals+1
    view%response=c_loc(response);view%coefficient=c_null_ptr
    if(levels>0) view%coefficient=c_loc(coefficient)
    view%edges=edge
    call check(create_distributed(local%handle,view,plan%handle,plan%device_bytes,plan%host_bytes),'create distributed')
    plan%partition=partition;plan%n=n;plan%capacity=capacity;plan%nodes=view%nodes
  end subroutine

  subroutine compact_distributed_views_cpp(plan,rhs,solution,nodes)
    type(compact_distributed_cpp_plan),intent(in) :: plan
    type(device_view),intent(out) :: rhs,solution,nodes
    call check(distributed_views(plan%handle,rhs,solution,nodes),'distributed views')
  end subroutine

  subroutine solve_distributed_compact_cpp(plan,nrhs)
    type(compact_distributed_cpp_plan),intent(in) :: plan
    integer,intent(in) :: nrhs
    if(nrhs<1.or.nrhs>plan%capacity) error stop 'distributed compact solve capacity'
    call check(solve_distributed(plan%handle,int(nrhs,c_int)),'distributed solve')
  end subroutine

  subroutine restore_distributed_compact_cpp(plan,nrhs)
    type(compact_distributed_cpp_plan),intent(in) :: plan
    integer,intent(in) :: nrhs
    if(nrhs<1.or.nrhs>plan%capacity) error stop 'distributed compact restore capacity'
    call check(restore_distributed(plan%handle,int(nrhs,c_int)),'distributed restore')
  end subroutine

  subroutine compact_distributed_transfer_cpp(plan,values,nrhs,mode)
    type(compact_distributed_cpp_plan),intent(in) :: plan
    real(c_device_real_kind),contiguous,target,intent(inout) :: values(:,:)
    integer,intent(in) :: nrhs,mode
    integer :: rows
    rows=merge(plan%nodes,plan%n,mode==3)
    if(nrhs<1.or.nrhs>plan%capacity.or.size(values,1)/=rows.or.size(values,2)<nrhs) &
      error stop 'distributed compact host observation shape'
    call check(transfer_distributed(plan%handle,int(mode,c_int),c_loc(values),int(nrhs,c_int)),'distributed transfer')
  end subroutine

  subroutine compact_distributed_metrics_cpp(plan,values,reset)
    type(compact_distributed_cpp_plan),intent(in) :: plan
    integer(c_int64_t),intent(out) :: values(2)
    logical,intent(in) :: reset
    call check(distributed_metrics(plan%handle,values,int(merge(1,0,reset),c_int)),'distributed metrics')
  end subroutine

  subroutine compact_distributed_phase_enable_cpp(plan)
    type(compact_distributed_cpp_plan),intent(in) :: plan
    call check(distributed_phase_enable(plan%handle),'enable phase timing')
  end subroutine

  subroutine release_distributed_compact_cpp(plan)
    type(compact_distributed_cpp_plan),intent(inout) :: plan
    integer :: ierr
    call check(destroy_distributed(plan%handle),'destroy distributed')
    call MPI_Comm_free(plan%comm,ierr);call check(int(ierr,c_int),'free interface communicator')
    plan%n=0;plan%capacity=0;plan%nodes=0;plan%device_bytes=0;plan%host_bytes=0
  end subroutine
end module
