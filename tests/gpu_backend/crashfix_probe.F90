program crashfix_probe
  use iso_c_binding
  use mpi
  use ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_quiet_nan
  use device_state
  use fluid_explicit, only: fluid_prepare
  use crashfix_operator_cpp
  use crashfix_runtime_cpp, only: repair_crashfix_cpp,repair_reconciled_cpp
  use halo_transport, only: device_transport_prepare,halo_pinned,halo_device
  implicit none
  type(c_ptr) :: state
  type(crashfix_report) :: report
  integer :: rank,ranks,local_rank,status,unit,h(13),im,jm,km,hm,scenario,debug,c,cases=0, &
    final_scenario,rounds,expanded_count,diagnostics(2),axis,step
  integer(c_int) :: periodic(3)=[0,0,1],dims(3),lower(3),higher(3),origin(3),indices(3),transport=halo_pinned
  integer(c_size_t) :: reference_bytes,expected_bytes,full_nodes
  logical :: expanded_mode,no_donor,reconciled_mode
  integer,parameter :: state_bits_kind=merge(c_int32_t,c_int64_t,c_device_real_kind==c_float)
  real(c_device_real_kind),parameter :: c2=1.4_c_device_real_kind*.3_c_device_real_kind**2, &
    c6=1._c_device_real_kind/(1.4_c_device_real_kind-1._c_device_real_kind)
  real(c_device_real_kind),parameter :: epsp=1.e-10_c_device_real_kind/c2
  real(c_double) :: maximum=0,global_error
  real(c_device_real_kind),allocatable :: initial(:,:,:,:),initial_p(:,:,:,:),cq(:,:,:,:),cp(:,:,:,:),gq(:,:,:,:),gp(:,:,:,:), &
    actual(:,:,:,:),pv(:,:,:,:),scratch(:,:,:,:)
  integer(c_int8_t),allocatable :: fluid(:,:,:),seed(:,:,:),cm(:,:,:),gm(:,:,:),observed(:,:,:)
  character(1024) :: root,path
  character(80) :: text,selection
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text,status=status)
  if(status/=0) error stop 'repair probe requires MPI local rank'
  read(text,*) local_rank;call check(device_bind(int(local_rank,c_int)))
  call MPI_Init(status);call MPI_Comm_rank(MPI_COMM_WORLD,rank,status);call MPI_Comm_size(MPI_COMM_WORLD,ranks,status)
  call get_command_argument(1,root)
  call get_command_argument(2,selection);no_donor=trim(selection)=='no-donor'
  reconciled_mode=trim(selection)=='reconciled'
  expanded_mode=trim(selection)=='expanded'.or.no_donor.or.reconciled_mode
  call get_command_argument(3,text)
  if(trim(text)=='device') transport=halo_device
  final_scenario=4
  if(expanded_mode) final_scenario=7
  if(no_donor) final_scenario=1
  do scenario=1,final_scenario
    if(scenario==6.and.ranks==1) cycle
    if(reconciled_mode.and.scenario==4) cycle
    write(path,'(a,a,i1,a,i8.8,a)') trim(root),'/reference.case',scenario,'.rank',rank,'.bin'
    open(newunit=unit,file=trim(path),status='old',access='stream',form='unformatted');read(unit) h
    im=h(1);jm=h(2);km=h(3);hm=h(4)
    if(h(5)/=5.or.h(12)/=scenario.or.product(h(9:11)/h(1:3))/=ranks) error stop 'repair reference identity'
    full_nodes=product(int([im,jm,km]+2*hm+1,c_size_t))
    expected_bytes=13*c_sizeof(0_c_int)+product(int([im,jm,km]+1,c_size_t))+3*full_nodes+33*full_nodes*c_sizeof(0._c_device_real_kind)
    inquire(unit=unit,size=reference_bytes)
    if(reference_bytes/=expected_bytes) error stop 'repair reference storage width/length'
    allocate(fluid(0:im,0:jm,0:km),seed(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
    allocate(cm,mold=seed);allocate(gm,mold=seed);allocate(observed,mold=seed)
    allocate(initial(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5));allocate(cq,mold=initial);allocate(gq,mold=initial)
    allocate(actual,mold=initial);allocate(initial_p(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6))
    allocate(cp,mold=initial_p);allocate(gp,mold=initial_p);allocate(pv,mold=initial_p)
    allocate(scratch(0:im,0:jm,0:km,5))
    read(unit) fluid,seed,initial,initial_p,cm,cq,cp,gm,gq,gp;close(unit)
    if(h(13)==1.and.any(cm/=gm)) error stop 'original repair mask references differ'
    if(expanded_mode) call reconciled_history_reference(gm)
    dims=int(h(9:11)/h(1:3),c_int);origin=int(h(6:8),c_int);indices=origin/h(1:3)
    lower=MPI_PROC_NULL;higher=MPI_PROC_NULL;step=1
    do axis=1,3
      if(indices(axis)>0) lower(axis)=rank-step
      if(indices(axis)<dims(axis)-1) higher(axis)=rank+step
      if(periodic(axis)/=0) then
        lower(axis)=rank+(modulo(indices(axis)-1,dims(axis))-indices(axis))*step
        higher(axis)=rank+(modulo(indices(axis)+1,dims(axis))-indices(axis))*step
      endif
      step=step*dims(axis)
    enddo
    do debug=0,1
      call check(device_state_create(int(local_rank,c_int),im,jm,km,hm,int(debug,c_int),state))
      call check(fluid_prepare(state,0_c_int))
      call check(crashfix_prepare_cpp(state,fluid,int(size(fluid),c_size_t)))
      call check(crashfix_history_upload_cpp(state,seed,int(size(seed),c_size_t)))
      if(expanded_mode) call check(device_transport_prepare(state,int(MPI_COMM_WORLD,c_int),lower,higher,dims, &
        5_c_int,transport,0_c_int,periodic))
      if(no_donor) then
        initial=ieee_value(0.d0,ieee_quiet_nan);initial_p=0
        initial_p(:,:,:,1)=ieee_value(0.d0,ieee_quiet_nan);initial_p(:,:,:,5:6)=-1
      endif
      call check(device_state_upload(state,device_q,initial,bytes(size(initial))))
      call check(device_state_upload(state,device_rho,initial_p(:,:,:,1),bytes(size(seed))))
      call check(device_state_upload(state,device_velocity,initial_p(:,:,:,2:4),bytes(3*size(seed))))
      call check(device_state_upload(state,device_pressure,initial_p(:,:,:,5),bytes(size(seed))))
      call check(device_state_upload(state,device_temperature,initial_p(:,:,:,6),bytes(size(seed))))
      scratch=777;call check(device_state_upload(state,device_save,scratch,bytes(size(scratch))))
      scratch=7;call check(device_state_upload(state,device_rhs,scratch,bytes(size(scratch))))
      if(expanded_mode) then
        if(reconciled_mode) then
          status=repair_reconciled_cpp(state,int(h(6:8),c_int),int(h(9:11),c_int),periodic,c2,c6,epsp, &
            MPI_COMM_WORLD,report)
        else
          status=repair_crashfix_cpp(state,int(h(6:8),c_int),int(h(9:11),c_int),periodic,c2,c6,epsp, &
            MPI_COMM_WORLD,report,rounds,expanded_count,diagnostics)
        endif
        if(no_donor) then
          if(status/=device_nonphysical.or.report%invalid/=1.or.expanded_count/=0.or.rounds/=1) &
            error stop 'global no-donor state was not rejected without repeated attempts'
          call check(device_state_download(state,device_q,actual,bytes(size(actual))))
          if(any(transfer(actual,[0_state_bits_kind],size(actual))/=transfer(initial,[0_state_bits_kind],size(initial)))) &
            error stop 'no-donor repair invented a state'
          call check(device_state_destroy(state));cases=cases+1
          cycle
        endif
        call check(int(status,c_int))
        if(.not.reconciled_mode) then
          if(scenario>=5.and.(rounds<1.or.expanded_count<1)) error stop 'expanded scene did not exercise recovery'
        endif
      else
        call check(crashfix_ordered_pass_cpp(state,int(h(6:8),c_int),int(h(9:11),c_int),c2,c6,epsp,report))
      endif
      if(report%invalid/=0) error stop 'unresolved repair'
      if(.not.reconciled_mode.and.scenario<=4.and.report%repaired/=merge(4,5,scenario==4)) error stop 'wrong ordered repair count'
      if(expanded_mode.and..not.reconciled_mode.and.debug==1.and.rank==0) write(*,'(a,5(1x,i0))') &
        'CPP_CRASHFIX_EXPANDED_CASE',scenario,rounds,expanded_count,diagnostics
      if(.not.all(ieee_is_finite(report%max_change)).or.any(report%minima<=0)) error stop 'repair report quantities'
      call check(device_state_download(state,device_q,actual,bytes(size(actual))))
      call check(device_state_download(state,device_rho,pv(:,:,:,1),bytes(size(seed))))
      call check(device_state_download(state,device_velocity,pv(:,:,:,2:4),bytes(3*size(seed))))
      call check(device_state_download(state,device_pressure,pv(:,:,:,5),bytes(size(seed))))
      call check(device_state_download(state,device_temperature,pv(:,:,:,6),bytes(size(seed))))
      if(.not.all(ieee_is_finite(actual(0:im,0:jm,0:km,:))).or. &
        .not.all(ieee_is_finite(pv(0:im,0:jm,0:km,:)))) error stop 'nonfinite physical repair result'
      do c=1,6
        if(c/=1.and.c/=5.and.c/=6) cycle
        if(any((pv(0:im,0:jm,0:km,c)<=0).and.(fluid/=0))) error stop 'nonpositive repaired fluid cache'
      enddo
      if(.not.expanded_mode) then
        if(.not.all(ieee_is_finite(actual)).or..not.all(ieee_is_finite(pv))) error stop 'nonfinite ordered result'
      endif
      do c=1,5
        maximum=max(maximum,defined_error(actual(:,:,:,c),gq(:,:,:,c)))
        if(h(13)==1) maximum=max(maximum,defined_error(actual(:,:,:,c),cq(:,:,:,c)))
      enddo
      do c=1,6
        maximum=max(maximum,defined_error(pv(:,:,:,c),gp(:,:,:,c)))
        if(h(13)==1) maximum=max(maximum,defined_error(pv(:,:,:,c),cp(:,:,:,c)))
      enddo
      call check(crashfix_history_download_cpp(state,observed,int(size(observed),c_size_t)))
      if(any(observed/=gm)) error stop 'repair history differs'
      call check(device_state_download(state,device_rhs,scratch,bytes(size(scratch))))
      if(any(scratch/=7)) error stop 'repair changed RHS'
      call check(device_state_download(state,device_save,scratch,bytes(size(scratch))))
      if(any(scratch/=777)) error stop 'repair changed RK origin'
      if(reconciled_mode) then
        call check(repair_reconciled_cpp(state,int(h(6:8),c_int),int(h(9:11),c_int),periodic,c2,c6,epsp, &
          MPI_COMM_WORLD,report))
      elseif(expanded_mode) then
        call check(repair_crashfix_cpp(state,int(h(6:8),c_int),int(h(9:11),c_int),periodic,c2,c6,epsp, &
          MPI_COMM_WORLD,report,rounds,expanded_count,diagnostics))
      else
        call check(crashfix_ordered_pass_cpp(state,int(h(6:8),c_int),int(h(9:11),c_int),c2,c6,epsp,report))
      endif
      if(report%candidates/=0.or.report%repaired/=0.or.report%invalid/=0) error stop 'healthy state still needs repair'
      call check(device_state_download(state,device_q,initial,bytes(size(initial))))
      if(any(transfer(initial,[0_state_bits_kind],size(initial))/=transfer(actual,[0_state_bits_kind],size(actual)))) &
        error stop 'healthy repeated pass changed q'
      call check(crashfix_history_download_cpp(state,observed,int(size(observed),c_size_t)))
      if(any(observed/=gm)) error stop 'healthy repeated pass changed history'
      ! Restore the captured seed for the second independent synchronization mode.
      open(newunit=unit,file=trim(path),status='old',access='stream',form='unformatted')
      read(unit) h;read(unit) fluid,seed,initial,initial_p;close(unit)
      call check(device_state_destroy(state));cases=cases+1
    enddo
    deallocate(fluid,seed,cm,gm,observed,initial,initial_p,cq,cp,gq,gp,actual,pv,scratch)
  enddo
  call MPI_Allreduce(maximum,global_error,1,MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,status)
  if(c_device_real_kind==c_double.and.global_error>2.d-10) error stop 'repair differs from frozen reference'
  selection='CPP_ORDERED_REPAIR_PASS'
  if(expanded_mode) selection='CPP_EXPANDED_REPAIR_PASS'
  if(no_donor) selection='CPP_NO_DONOR_REPAIR_PASS'
  if(reconciled_mode) selection='CPP_RECONCILED_REPAIR_PASS'
  if(rank==0) write(*,'(a,i0,a,i0,a,es24.16e3)') trim(selection)//' ranks=',ranks,' local_cases=',cases,' max_scaled=',global_error
  if(rank==0) write(*,'(a,i0)') 'CPP_REPAIR_REAL_BYTES=',c_sizeof(0._c_device_real_kind)
  call MPI_Finalize(status)
contains
  subroutine reconciled_history_reference(mask)
    integer(c_int8_t),intent(inout) :: mask(-hm:,-hm:,-hm:)
    integer(c_int8_t),allocatable :: physical(:,:,:,:),expected(:,:,:)
    integer :: peer,coords(3),point(3),global_point(3),candidate(3),sx,sy,sz,axis,ii,jj,kk,err
    logical :: allowed,marked
    allocate(physical(0:im,0:jm,0:km,0:ranks-1));allocate(expected,mold=mask);expected=mask
    call MPI_Allgather(mask(0:im,0:jm,0:km),(im+1)*(jm+1)*(km+1),MPI_INTEGER1, &
      physical,(im+1)*(jm+1)*(km+1),MPI_INTEGER1,MPI_COMM_WORLD,err)
    call check(err)
    ! The established N10 contract ORs shared replicas, then refreshes face halos.
    ! Original repair packets predate that wrapper; their q/cache oracle is unchanged.
    do kk=-hm,km+hm
    do jj=-hm,jm+hm
    do ii=-hm,im+hm
      point=[ii,jj,kk]
      if(count(point<0.or.point>[im,jm,km])>1) cycle
      global_point=h(6:8)+point;allowed=.true.
      do axis=1,3
        if(periodic(axis)/=0) then
          global_point(axis)=modulo(global_point(axis),h(axis+8))
        elseif(global_point(axis)<0.or.global_point(axis)>h(axis+8)) then
          allowed=.false.
        endif
      enddo
      if(.not.allowed) cycle
      marked=.false.
      do peer=0,ranks-1
        coords=[modulo(peer,h(9)/im),modulo(peer/(h(9)/im),h(10)/jm),peer/((h(9)/im)*(h(10)/jm))]
        do sz=0,periodic(3)
        do sy=0,periodic(2)
        do sx=0,periodic(1)
          candidate=global_point+[sx,sy,sz]*h(9:11)-coords*[im,jm,km]
          if(all(candidate>=0.and.candidate<=[im,jm,km])) &
            marked=marked.or.physical(candidate(1),candidate(2),candidate(3),peer)/=0
        enddo
        enddo
        enddo
      enddo
      expected(ii,jj,kk)=merge(1_c_int8_t,0_c_int8_t,marked)
    enddo
    enddo
    enddo
    mask=expected
  end subroutine
  real(c_double) function defined_error(actual,reference) result(value)
    real(c_device_real_kind),intent(in) :: actual(:,:,:),reference(:,:,:)
    real(c_double),allocatable :: observed(:),expected(:)
    if(any(ieee_is_finite(actual).neqv.ieee_is_finite(reference)).or. &
      any(ieee_is_nan(actual).neqv.ieee_is_nan(reference))) error stop 'repair halo/cache phase differs'
    observed=pack(actual,ieee_is_finite(reference));expected=pack(reference,ieee_is_finite(reference))
    value=maxval(abs(observed-expected))/max(1.d0,maxval(abs(expected)))
  end function
  integer(c_size_t) function bytes(count)
    integer,intent(in) :: count
    bytes=int(count,c_size_t)*c_sizeof(0._c_device_real_kind)
  end function
  subroutine check(value)
    integer(c_int),intent(in) :: value
    if(value/=0) then
      write(*,'(a,i0)') 'repair device status=',value
      error stop 'repair device operation'
    endif
  end subroutine
end program
