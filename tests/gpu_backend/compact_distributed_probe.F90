program compact_distributed_probe_cpp
  use iso_c_binding, only: c_ptr,c_int,c_int64_t,c_associated
  use mpi
  use ieee_arithmetic, only: ieee_is_finite
  use astr_precision, only: real_kind,real_bytes,real_to_word
  use astr_precision_mpi, only: mpi_real_type
  use compact_operator
  use compact_line_solver
  use compact_line_cpp
  use device_state
  implicit none
  integer,parameter :: lengths(3)=[7,129,257],batches(3)=[1,5,11],capacity=11
  type(compact_matrix) :: full,core,constructed
  type(compact_reference_factor) :: factor
  type(compact_line_partition) :: part
  type(compact_distributed_cpp_plan) :: plan,other
  type(device_view) :: input_view,solution_view,nodes_view,again_input,again_solution,again_nodes
  type(c_ptr) :: state
  real(real_kind),allocatable :: exact(:,:),rhs(:,:),reference(:,:),local_rhs(:,:),actual(:,:),gathered(:,:), &
    restored(:,:),gold(:,:),gold_nodes(:,:),unchanged(:,:)
  integer,allocatable :: counts(:),displacements(:)
  real(real_kind) :: alpha,edge(2),constructed_edge(2),scaled,residual,restored_error,manufactured,scale
  real(real_kind) :: maximum_error=0,maximum_residual=0,maximum_restore=0
  integer(c_int64_t) :: metrics(2),wire(5)
  integer(c_int) :: mode,status
  integer :: ierr,world_rank,world_size,local_rank,parent,cart,line,rank,ranks,axis,coords(3),dims(3), &
    plane,il,kind,ip,n,m,nodes,i,j,g,batch,nrhs,debug,cases=0,csv,binary,header(9)
  logical :: periods(3),periodic,split
  integer :: split_cases=0,shared_devices
  character(1024) :: output,path
  character(40) :: argument,transport
  interface
    subroutine wire_counts(values) bind(C,name='astr_halo_wire_counts')
      import c_int64_t
      integer(c_int64_t),intent(out) :: values(5)
    end subroutine
  end interface
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',argument,status=ierr)
  if(ierr/=0) error stop 'compact distributed probe needs MPI local rank'
  read(argument,*) local_rank
  call get_environment_variable('ASTR_TEST_COMPACT_LOCAL_DEVICES',argument,status=ierr)
  if(ierr==0) then
    shared_devices=0
    read(argument,*,iostat=ierr) shared_devices
    if(ierr/=0.or.shared_devices<1) error stop 'invalid test device sharing count'
    local_rank=mod(local_rank,shared_devices)
  endif
  status=device_bind(int(local_rank,c_int))
  if(status/=0) error stop 'compact distributed pre-MPI device binding'
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,world_rank,ierr)
  call MPI_Comm_size(MPI_COMM_WORLD,world_size,ierr)
  call get_command_argument(1,output)
  do i=1,3
    call get_command_argument(i+1,argument);read(argument,*) dims(i)
  enddo
  call get_command_argument(5,transport)
  select case(trim(transport))
  case('pinned');mode=0_c_int
  case('device-aware');mode=1_c_int
  case default;call fail('transport must be pinned or device-aware')
  end select
  call get_command_argument(6,argument)
  if(argument/=''.and.argument/='split') call fail('expected optional split mode')
  split=argument=='split'
  if(product(dims)/=world_size) call fail('topology does not match MPI size')
  call MPI_Comm_split(MPI_COMM_WORLD,0,world_size-1-world_rank,parent,ierr)
  periods=.false.
  call MPI_Cart_create(parent,3,dims,periods,.true.,cart,ierr)
  call MPI_Comm_free(parent,ierr)
  call MPI_Comm_rank(cart,rank,ierr)
  call MPI_Cart_coords(cart,rank,3,coords,ierr)
  write(path,'(a,"/rank",i4.4,".csv")') trim(output),world_rank
  open(newunit=csv,file=trim(path),status='new',action='write')
  write(csv,'(a)') 'axis,plane,kind,periodic,intervals,nrhs,first,owned,line_ranks,sync,'// &
    'scaled_error,residual,restore_error,device_bytes,pinned_bytes,sent_bytes,host_stage_bytes'
  write(path,'(a,"/rank",i4.4,".bin")') trim(output),world_rank
  open(newunit=binary,file=trim(path),status='new',access='stream',form='unformatted')
  do axis=1,3
    call compact_direction_communicator(cart,axis,line)
    call MPI_Comm_rank(line,rank,ierr);call MPI_Comm_size(line,ranks,ierr)
    allocate(counts(ranks),displacements(ranks))
    select case(axis)
    case(1);plane=coords(2)+dims(2)*coords(3)
    case(2);plane=coords(1)+dims(1)*coords(3)
    case(3);plane=coords(1)+dims(1)*coords(2)
    end select
    do il=1,size(lengths)
      do kind=1,2
        do ip=0,1
          periodic=ip==1
          ! Every local PCR core needs two unknowns, including a four-rank line.
          call partition_compact_line(max(lengths(il),2*ranks+1),rank,ranks,periodic,part)
          n=part%global_n;m=part%owned;nodes=part%local_intervals+1
          call MPI_Allgather(m,1,MPI_INTEGER,counts,1,MPI_INTEGER,line,ierr)
          call MPI_Allgather(part%first,1,MPI_INTEGER,displacements,1,MPI_INTEGER,line,ierr)
          if(sum(counts)/=n) call fail('unique compact node ownership')
          full%n=n;full%cyclic=periodic
          allocate(full%lower(n),full%diagonal(n),full%upper(n))
          alpha=1.e0_real_kind/3.e0_real_kind
          if(kind==2) alpha=.49e0_real_kind
          full%lower=alpha;full%diagonal=1;full%upper=alpha
          if(.not.periodic) then
            full%lower(1)=0;full%upper(n)=0
            if(kind==1) then
              full%upper(1)=2;full%lower(n)=2
              full%lower(2)=.25e0_real_kind;full%upper(2)=.25e0_real_kind
              full%lower(n-1)=.25e0_real_kind;full%upper(n-1)=.25e0_real_kind
            else
              full%upper(1)=.98e0_real_kind;full%lower(n)=.98e0_real_kind
            endif
          endif
          core%n=m;core%cyclic=.false.
          allocate(core%lower(m),core%diagonal(m),core%upper(m))
          core%lower=full%lower(part%first+1:part%first+m)
          core%diagonal=full%diagonal(part%first+1:part%first+m)
          core%upper=full%upper(part%first+1:part%first+m)
          edge=[core%lower(1),core%upper(m)];core%lower(1)=0;core%upper(m)=0
          if(kind==1) then
            call compact_derivative_core(part,constructed,constructed_edge)
          else
            call compact_filter_core(part,alpha,constructed,constructed_edge)
          endif
          if(any(constructed_edge/=edge).or.any(constructed%lower/=core%lower).or. &
            any(constructed%diagonal/=core%diagonal).or.any(constructed%upper/=core%upper)) &
            call fail('runtime core differs from full matrix slice')
          deallocate(constructed%lower,constructed%diagonal,constructed%upper)
          allocate(exact(n,capacity),rhs(n,capacity),reference(n,capacity),local_rhs(m,capacity), &
            actual(m,capacity),gathered(n,capacity),restored(nodes,capacity),gold(m,capacity), &
            gold_nodes(nodes,capacity),unchanged(m,capacity))
          do j=1,capacity
            do i=1,n
              exact(i,j)=(sin(.13e0_real_kind*i+.19e0_real_kind*j)+cos(.07e0_real_kind*i*j)+ &
                .01e0_real_kind*plane)*10.e0_real_kind**(mod(j,7)-3)
            enddo
          enddo
          call compact_matvec(full,exact,rhs)
          call factor_compact_reference(full,factor);call solve_compact_reference(factor,rhs,reference)
          if(.not.all(ieee_is_finite(reference))) call fail('nonfinite independent LU')
          if(real_bytes==8.and.compact_residual(full,reference,rhs)>1.e-12_real_kind) call fail('independent LU residual')
          local_rhs=rhs(part%first+1:part%first+m,:)
          do debug=0,1
            status=device_state_create(int(local_rank,c_int),3_c_int,3_c_int,3_c_int,1_c_int,int(debug,c_int),state)
            if(status/=0) call fail('distributed context creation')
            call prepare_distributed_compact_cpp(state,core,edge,part,line,capacity,mode,plan)
            status=device_state_destroy(state)
            if(status/=device_invalid_argument.or..not.c_associated(state)) call fail('distributed parent lifetime')
            call compact_distributed_views_cpp(plan,input_view,solution_view,nodes_view)
            if(input_view%element_bytes/=real_bytes.or.solution_view%element_bytes/=real_bytes.or. &
              nodes_view%element_bytes/=real_bytes.or. &
              any(input_view%extent(:2)/=[m,capacity]).or.any(nodes_view%extent(:2)/=[nodes,capacity])) &
              call fail('distributed view layout')
            if(mode==1.and.plan%host_bytes/=0) call fail('device transport allocated pinned slots')
            if(mode==0.and.plan%host_bytes/=6*capacity*real_bytes) call fail('pinned prefix slots')
            call compact_distributed_transfer_cpp(plan,local_rhs,capacity,0)
            if(split.and.ranks>1) then
              call prepare_distributed_compact_cpp(state,core,edge,part,line,capacity,mode,other)
              unchanged=-2*local_rhs
              call compact_distributed_transfer_cpp(other,unchanged,capacity,0)
            endif
            do batch=1,size(batches)
              nrhs=batches(batch)
              if(split.and.ranks>1) then
                call check_split(compact_distributed_local_cpp(plan%handle,int(nrhs,c_int)), 'first local solve')
                call check_split(compact_distributed_begin_cpp(plan%handle,int(nrhs,c_int)), 'first PCR post')
                call check_split(compact_distributed_local_cpp(other%handle,int(nrhs,c_int)), 'next local solve')
                call check_split(compact_distributed_finish_cpp(plan%handle,int(nrhs,c_int)), 'first finish')
              else
                call solve_distributed_compact_cpp(plan,nrhs)
              endif
              call restore_distributed_compact_cpp(plan,nrhs)
              call compact_distributed_transfer_cpp(plan,actual,nrhs,2)
              call compact_distributed_transfer_cpp(plan,restored,nrhs,3)
              if(.not.all(ieee_is_finite(actual(:,:nrhs))).or. &
                .not.all(ieee_is_finite(restored(:,:nrhs)))) call fail('nonfinite distributed result')
              if(split.and.ranks>1) then
                call check_split(compact_distributed_begin_cpp(other%handle,int(nrhs,c_int)), 'next PCR post')
                call check_split(compact_distributed_finish_cpp(other%handle,int(nrhs,c_int)), 'next finish')
                call compact_distributed_transfer_cpp(other,unchanged,nrhs,2)
                if(any(unchanged(:,:nrhs)/=-2*actual(:,:nrhs))) call fail('second batch solution or buffer isolation')
                split_cases=split_cases+1
              endif
              do j=1,nrhs
                call MPI_Allgatherv(actual(:,j),m,mpi_real_type,gathered(:,j),counts,displacements,mpi_real_type,line,ierr)
              enddo
              scaled=0;manufactured=0;restored_error=0
              do j=1,nrhs
                scale=max(1.e0_real_kind,maxval(abs(reference(:,j))))
                scaled=max(scaled,maxval(abs(gathered(:,j)-reference(:,j)))/scale)
                manufactured=max(manufactured,maxval(abs(gathered(:,j)-exact(:,j)))/scale)
                do i=1,nodes
                  g=part%first+i
                  if(periodic) g=modulo(g-1,n)+1
                  restored_error=max(restored_error,abs(restored(i,j)-reference(g,j))/scale)
                enddo
              enddo
              residual=compact_residual(full,gathered(:,:nrhs),rhs(:,:nrhs))
              if(.not.all(ieee_is_finite([scaled,residual,restored_error,manufactured]))) call fail('nonfinite diagnostics')
              if(real_bytes==8.and.(max(scaled,restored_error,manufactured)>2.e-10_real_kind.or.residual>1.e-12_real_kind)) &
                call fail('global LU/manufactured/residual gate')
              if(debug==0.and.nrhs==capacity) then
                gold=actual;gold_nodes=restored
              elseif(debug==1) then
                if(any(real_to_word(actual(:,:nrhs))/=real_to_word(gold(:,:nrhs))).or. &
                  any(real_to_word(restored(:,:nrhs))/=real_to_word(gold_nodes(:,:nrhs)))) &
                  call fail('distributed sync isolation')
              endif
              call compact_distributed_metrics_cpp(plan,metrics,.true.)
              if(mode==1.and.metrics(2)/=0) call fail('device path staged interface on host')
              if(mode==0.and.metrics(2)<=0) call fail('pinned prefix copies not recorded')
              write(csv,'(10(i0,","),3(es24.16e3,","),3(i0,","),i0)') axis,plane,kind,ip,part%intervals,nrhs, &
                part%first,m,ranks,debug,scaled,residual,restored_error,plan%device_bytes,plan%host_bytes,metrics
              header=[axis,kind,ip,part%intervals,nrhs,part%first,m,plane,debug]
              write(binary) header,actual(:,:nrhs),restored(:,:nrhs)
              cases=cases+1;maximum_error=max(maximum_error,scaled)
              maximum_residual=max(maximum_residual,residual);maximum_restore=max(maximum_restore,restored_error)
            enddo
            call compact_distributed_transfer_cpp(plan,unchanged,capacity,1)
            if(any(real_to_word(unchanged)/=real_to_word(local_rhs))) call fail('distributed RHS changed')
            actual=0
            call compact_distributed_transfer_cpp(plan,actual,capacity,0)
            call solve_distributed_compact_cpp(plan,capacity)
            call restore_distributed_compact_cpp(plan,capacity)
            call compact_distributed_transfer_cpp(plan,actual,capacity,2)
            call compact_distributed_transfer_cpp(plan,restored,capacity,3)
            if(any(actual/=0).or.any(restored/=0)) call fail('distributed zero RHS invariant')
            call compact_distributed_views_cpp(plan,again_input,again_solution,again_nodes)
            if(.not.c_associated(input_view%data,again_input%data).or. &
              .not.c_associated(solution_view%data,again_solution%data).or. &
              .not.c_associated(nodes_view%data,again_nodes%data)) call fail('distributed buffers reallocated')
            if(split.and.ranks>1) call release_distributed_compact_cpp(other)
            call release_distributed_compact_cpp(plan)
            status=device_state_destroy(state)
            if(status/=0.or.c_associated(state)) call fail('distributed release')
          enddo
          deallocate(full%lower,full%diagonal,full%upper,core%lower,core%diagonal,core%upper, &
            exact,rhs,reference,local_rhs,actual,gathered,restored,gold,gold_nodes,unchanged)
        enddo
      enddo
    enddo
    deallocate(counts,displacements)
    call MPI_Comm_free(line,ierr)
  enddo
  write(*,'(a,2(1x,i0),3(1x,es24.16e3))') 'CPP_DISTRIBUTED_PASS',world_rank,cases, &
    maximum_error,maximum_residual,maximum_restore
  if(split) write(*,'(a,2(1x,i0))') 'CPP_DISTRIBUTED_SPLIT_CASES',world_rank,split_cases
  call wire_counts(wire)
  if(wire(5)/=0) call fail('compact MPI packet type differs from build precision')
  write(*,'(a,2(1x,i0),5(1x,i0))') 'CPP_COMPACT_WIRE',world_rank,real_bytes,wire
  close(csv);close(binary)
  call MPI_Comm_free(cart,ierr)
  call MPI_Finalize(ierr)
contains
  subroutine check_split(status,message)
    integer(c_int),intent(in) :: status
    character(*),intent(in) :: message
    if(status/=0) call fail(message)
  end subroutine

  subroutine fail(message)
    character(*),intent(in) :: message
    print '(a,i0,2a)', 'CPP_DISTRIBUTED_FAILURE rank=',world_rank,' ',message
    call MPI_Abort(MPI_COMM_WORLD,1,ierr)
  end subroutine
end program
