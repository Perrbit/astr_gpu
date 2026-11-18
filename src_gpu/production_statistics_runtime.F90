module production_statistics_runtime
  use iso_c_binding, only: c_int,c_size_t,c_associated
  use iso_fortran_env, only: int64
  use ieee_arithmetic, only: ieee_is_finite
  use astr_precision, only: real_kind,real_bytes,real_to_word,word_to_real
  use astr_precision_mpi, only: mpi_real_type
  use mpi
  use production_statistics_cpp
  use fluid_runtime, only: cpp_device_context,cpp_refresh_mean_fields
  use checkpoint_state_io, only: checkpoint_state_identity,checkpoint_state_transfer, &
    allocate_checkpoint_buffer,checkpoint_state_require,checkpoint_state_header
  implicit none
  private
  public :: compact_statistics_requested,compact_statistics_supported,compact_statistics_host_bytes
  public :: initialize_compact_statistics_cpp,reset_compact_statistics_cpp,accumulate_compact_statistics_cpp
  public :: complete_compact_statistics_file
  integer,parameter :: compact_payload_count=32
  logical,save :: active=.false.
  integer(int64),save :: sample_count=0,sampling_start_step=0,last_sample_step=-1
  real(real_kind),save :: sampling_start_time=0,last_sample_time=0,time_weight=0
  ! As in the original GPU protocol, repartitioned history is replicated on host.
  real(real_kind),allocatable,save :: inherited_statistics(:,:,:,:)
contains
  subroutine check(status,label)
    integer,intent(in) :: status
    character(*),intent(in) :: label
    call checkpoint_state_require(status==0,MPI_COMM_WORLD,label)
  end subroutine

  logical function compact_statistics_requested()
    use commvar, only: use_gpu,lavg,flowtype
    compact_statistics_requested=use_gpu.and.lavg.and.(trim(flowtype)=='bl'.or.trim(flowtype)=='swbli')
  end function

  logical function compact_statistics_supported()
    use commvar, only: flowtype,ndims,numq,num_species,turbmode,lkhomo,ljhomo
    use bc, only: bctype
    compact_statistics_supported=(trim(flowtype)=='bl'.or.trim(flowtype)=='swbli').and.ndims==3.and.numq==5.and. &
      num_species==0.and.trim(turbmode)=='none'.and..not.ljhomo.and.lkhomo.and. &
      any(bctype(3)==[41,42,411,421]).and.all(bctype(5:6)==1)
  end function

  subroutine initialize_compact_statistics_cpp()
    use commvar, only: im,jm,km,hm,ia
    use commarray, only: x,bnorm_j0
    use parallel, only: ig0,jrk,lio
    real(real_kind),allocatable :: normal(:,:,:),measure(:,:)
    if(active.or..not.compact_statistics_requested()) return
    call checkpoint_state_require(compact_statistics_supported(),MPI_COMM_WORLD,'C++ compact statistics case')
    call checkpoint_state_require(c_associated(cpp_device_context()),MPI_COMM_WORLD,'C++ statistics needs device context')
    allocate(normal(0:im,0:km,3));normal=0
    if(jrk==0) normal=bnorm_j0
    call check(production_geometry_cpp(cpp_device_context(),x,normal,im,jm,km,hm,ia,ig0,jrk==0), &
      'initialize compact device geometry')
    allocate(measure(0:im,0:jm))
    call transfer(production_measure,.false.,measure)
    call checkpoint_state_require(all(ieee_is_finite(measure)).and.all(measure>0),MPI_COMM_WORLD, &
      'compact statistics needs positive physical z measure')
    active=.true.
    if(lio) write(*,'(a)') ' ** C++ compact production statistics: enabled'
  end subroutine

  subroutine reset_compact_statistics_cpp()
    use statistic, only: nsamples
    if(active) call local_check(production_reset_cpp(cpp_device_context()),'reset compact device statistics')
    if(allocated(inherited_statistics)) inherited_statistics=0
    sample_count=0;sampling_start_step=0;last_sample_step=-1
    sampling_start_time=0;last_sample_time=0;time_weight=0
    nsamples=0
  end subroutine

  subroutine accumulate_compact_statistics_cpp()
    use commvar, only: lavg,nstep,time,nondimen,reynolds,prandtl,const5,cp,tempconst,tempconst1
    use parallel, only: jrk
    use statistic, only: nsamples
    real(real_kind) :: interval,constants(6)
    if(.not.active.or..not.lavg) return
    if(int(nstep,int64)==last_sample_step) return
    interval=0
    if(sample_count>0) interval=time-last_sample_time
    call local_check(merge(0,90,ieee_is_finite(time).and.ieee_is_finite(interval).and. &
      (sample_count==0.or.interval>0)),'compact sampling time must advance')
    call cpp_refresh_mean_fields()
    call local_check(production_plane_cpp(cpp_device_context(),interval,int(merge(1,0,sample_count>0),c_int)), &
      'accumulate compact plane')
    if(jrk==0) then
      constants=[reynolds,prandtl,const5,cp,tempconst,tempconst1]
      ! Device failure aborts the communicator; do not make a collective check
      ! conditional on the lower-wall owner.
      call local_check(production_wall_cpp(cpp_device_context(),interval,int(merge(1,0,sample_count>0),c_int), &
        int(merge(1,0,nondimen),c_int),constants),'accumulate compact wall')
    endif
    if(sample_count==0) then
      sampling_start_step=int(nstep,int64);sampling_start_time=time
    endif
    sample_count=sample_count+1
    nsamples=sample_count;last_sample_step=int(nstep,int64);last_sample_time=time;time_weight=time_weight+interval
  end subroutine

  subroutine local_check(status,label)
    use parallel, only: mpirank
    integer,intent(in) :: status
    character(*),intent(in) :: label
    integer :: ignored
    if(status==0) return
    write(*,'(a,i0,2a,i0)') 'C++ compact statistics rank=',mpirank,' ',label,status
    call MPI_Abort(MPI_COMM_WORLD,status,ignored)
  end subroutine

  subroutine complete_compact_statistics_file(path,writing,identity,budget,allow_repartition)
    use commvar, only: ia,ja,ka,im,jm,km
    use parallel, only: ig0,jg0,krk,isize,jsize,ksize,jrk,mpirank
    use statistic, only: nsamples
    character(*),intent(in) :: path
    logical,intent(in) :: writing
    logical,optional,intent(in) :: allow_repartition
    type(checkpoint_state_identity),intent(inout) :: identity
    integer(int64),intent(in) :: budget
    real(real_kind),allocatable :: values(:,:,:,:),plane_measure_h(:,:),plane_sum_h(:,:,:), &
      wall_measure_h(:),wall_sum_h(:,:),old_values(:,:,:,:),measure(:,:,:,:), &
      plane_endpoint_h(:,:,:),wall_endpoint_h(:,:)
    integer(int64) :: remaining,metadata(14),header(13),base_metadata(14),old_remaining,base_bytes,working_budget
    type(checkpoint_state_identity) :: expected
    integer :: status,ierr,old_shape(3)
    logical :: permit,migrating
    call checkpoint_state_require(active,MPI_COMM_WORLD,'compact statistics not initialized')
    permit=.false.
    if(present(allow_repartition)) permit=allow_repartition
    migrating=.false.;working_budget=budget
    base_bytes=int(ia+1,int64)*(ja+1)*2*compact_payload_count*real_bytes
    if(permit.and..not.allocated(inherited_statistics)) then
      call checkpoint_state_require(budget>=2*base_bytes+4096,MPI_COMM_WORLD,'inherited statistics host budget')
      allocate(inherited_statistics(ia+1,ja+1,2,compact_payload_count),stat=status)
      call check(status,'inherited statistics allocation')
      inherited_statistics=0;working_budget=budget-base_bytes
    endif
    ! The third index is the source z partition, not a physical z coordinate.
    call allocate_checkpoint_buffer([im,jm,1],0,compact_payload_count,working_budget,values,remaining,MPI_COMM_WORLD)
    call checkpoint_state_require(remaining>=int(im+1,int64)*(jm+1)*25*real_bytes+ &
      int(im+1,int64)*7*real_bytes+4096,MPI_COMM_WORLD,'compact statistics staging budget')
    values=0
    allocate(plane_measure_h(0:im,0:jm),plane_sum_h(0:im,0:jm,12),plane_endpoint_h(0:im,0:jm,12),stat=status)
    call check(status,'compact plane staging allocation')
    if(jrk==0) then
      allocate(wall_measure_h(0:im),wall_sum_h(0:im,3),wall_endpoint_h(0:im,3),stat=status)
    else
      status=0
    endif
    call check(status,'compact wall staging allocation')
    metadata=[3_int64,int([ia,ja,ka,isize,jsize,ksize],int64),sample_count,sampling_start_step, &
      last_sample_step,real_to_word(sampling_start_time),int(compact_payload_count,int64), &
      real_to_word(last_sample_time),real_to_word(time_weight)]
    if(permit) metadata(1)=4
    expected=identity
    if(writing) then
      call checkpoint_state_require(sample_count==int(nsamples,int64).and.last_sample_step<=identity%step, &
        MPI_COMM_WORLD,'compact sampling identity')
      call transfer(production_measure,.false.,plane_measure_h)
      call transfer(production_sum,.false.,plane_sum_h)
      call transfer(production_endpoint,.false.,plane_endpoint_h)
      values(:,:,1,1)=plane_measure_h;values(:,:,1,2:13)=plane_sum_h;values(:,:,1,18:29)=plane_endpoint_h
      if(jrk==0) then
        call transfer(production_wall_measure,.false.,wall_measure_h)
        call transfer(production_wall_sum,.false.,wall_sum_h)
        call transfer(production_wall_endpoint,.false.,wall_endpoint_h)
        values(:,1,1,14)=wall_measure_h;values(:,1,1,15:17)=wall_sum_h;values(:,1,1,30:32)=wall_endpoint_h
      endif
    endif
    remaining=remaining-int(im+1,int64)*(jm+1)*25*real_bytes
    if(jrk==0) remaining=remaining-int(im+1,int64)*7*real_bytes
    if(permit) then
      call checkpoint_state_require(remaining>=2*base_bytes+4096,MPI_COMM_WORLD,'inherited staging budget')
      if(.not.writing) then
        call checkpoint_state_header(path,header,metadata,MPI_COMM_SELF)
        call checkpoint_state_require(metadata(1)==4.and.all(metadata(2:4)==int([ia,ja,ka],int64)).and. &
          all(metadata(5:7)>=1).and.all(metadata(5:7)<=2).and.metadata(6)==1.and. &
          product(metadata(5:7))<=2.and.header(8)==product(metadata(5:7)).and.header(13)==5.and. &
          header(7)==compact_payload_count.and. &
          all(header(4:6)==[int(ia+1,int64),int(ja+1,int64),metadata(7)+1]), &
          MPI_COMM_WORLD,'compact migration source layout')
        migrating=any(metadata(5:7)/=int([isize,jsize,ksize],int64))
      endif
    endif
    if(migrating) then
      old_shape=int(header(4:6))
      call allocate_checkpoint_buffer(old_shape-1,0,compact_payload_count,remaining,old_values,old_remaining,MPI_COMM_WORLD)
      old_values=0
      call checkpoint_state_transfer(path,.false.,.true.,old_shape,[0,0,0],old_shape-1,0, &
        old_values,identity,old_remaining,MPI_COMM_SELF,role=5,metadata=metadata, &
        allow_repartition=.true.,repartition_axes=[.true.,.false.,.true.])
      call require_compact_clock(identity,expected)
      base_metadata=metadata
      call checkpoint_state_transfer(path,.false.,.true.,[ia+1,ja+1,2],[0,0,0],[ia,ja,1],0, &
        inherited_statistics,identity,old_remaining,MPI_COMM_SELF,role=5,metadata=base_metadata,group_name='inherited')
      call checkpoint_state_require(all(base_metadata==metadata),MPI_COMM_WORLD,'inherited sampling identity')
      call checkpoint_state_require(all(ieee_is_finite(old_values)).and.all(ieee_is_finite(inherited_statistics)), &
        MPI_COMM_WORLD,'nonfinite inherited statistics')
      call compact_global_measure(measure,old_remaining)
      call checkpoint_state_require(all(abs(sum(old_values(:,:,1:old_shape(3)-1,1),dim=3)-measure(:,:,1,1))<=2.e-10_real_kind).and. &
        all(abs(sum(old_values(:,:,1:old_shape(3)-1,14),dim=3)-measure(:,:,1,14))<=2.e-10_real_kind).and. &
        all(abs(inherited_statistics(:,:,:,1)-measure(:,:,:,1))<=2.e-10_real_kind).and. &
        all(abs(inherited_statistics(:,:,:,14)-measure(:,:,:,14))<=2.e-10_real_kind), &
        MPI_COMM_WORLD,'migrated compact geometry mismatch')
      inherited_statistics(:,:,1,2:13)=inherited_statistics(:,:,1,2:13)+sum(old_values(:,:,1:old_shape(3)-1,2:13),dim=3)
      inherited_statistics(:,:,1,15:17)=inherited_statistics(:,:,1,15:17)+sum(old_values(:,:,1:old_shape(3)-1,15:17),dim=3)
      call checkpoint_state_require(all(ieee_is_finite(inherited_statistics)).and. &
        all(inherited_statistics(:,:,2,:)==0),MPI_COMM_WORLD,'invalid inherited cumulative history')
      values=0
      call transfer(production_measure,.false.,plane_measure_h)
      values(:,:,1,1)=plane_measure_h
      values(:,:,1,18:29)=sum(old_values(ig0+1:ig0+im+1,jg0+1:jg0+jm+1, &
        1:old_shape(3)-1,18:29),dim=3)/real(ksize,real_kind)
      if(jrk==0) then
        call transfer(production_wall_measure,.false.,wall_measure_h)
        values(:,1,1,14)=wall_measure_h
        values(:,1,1,30:32)=sum(old_values(ig0+1:ig0+im+1,1,1:old_shape(3)-1,30:32),dim=2)/real(ksize,real_kind)
      endif
      deallocate(old_values,measure)
    else
      call checkpoint_state_transfer(path,writing,.true.,[ia+1,ja+1,ksize+1],[ig0,jg0,krk], &
        [im,jm,1],0,values,identity,remaining,MPI_COMM_WORLD,role=5,metadata=metadata)
      if(.not.writing) then
        call require_compact_clock(identity,expected)
        call checkpoint_state_require(all(metadata(5:7)==int([isize,jsize,ksize],int64)), &
          MPI_COMM_WORLD,'compact checkpoint topology metadata')
      endif
      if(permit) then
        if(writing) then
          call compact_global_measure(measure,remaining)
          inherited_statistics(:,:,:,1)=measure(:,:,:,1);inherited_statistics(:,:,:,14)=measure(:,:,:,14)
          deallocate(measure)
        endif
        base_metadata=metadata
        if(.not.writing.or.mpirank==0) call checkpoint_state_transfer(path,writing,.true., &
          [ia+1,ja+1,2],[0,0,0],[ia,ja,1],0,inherited_statistics,identity,remaining,MPI_COMM_SELF, &
          role=5,metadata=base_metadata,group_name='inherited')
        call MPI_Barrier(MPI_COMM_WORLD,ierr)
        call checkpoint_state_require(ierr==MPI_SUCCESS.and.all(base_metadata==metadata), &
          MPI_COMM_WORLD,'inherited state publication')
        call checkpoint_state_require(all(ieee_is_finite(inherited_statistics)).and. &
          all(inherited_statistics(:,:,2,:)==0),MPI_COMM_WORLD,'invalid inherited cumulative history')
        if(.not.writing) then
          call compact_global_measure(measure,remaining)
          call checkpoint_state_require(all(inherited_statistics(:,:,:,1)==measure(:,:,:,1)).and. &
            all(inherited_statistics(:,:,:,14)==measure(:,:,:,14)),MPI_COMM_WORLD,'inherited compact geometry mismatch')
          deallocate(measure)
        endif
      endif
    endif
    if(.not.writing) then
      call require_compact_clock(identity,expected)
      call checkpoint_state_require(metadata(1)==merge(4_int64,3_int64,permit).and. &
        all(metadata(2:4)==int([ia,ja,ka],int64)).and.metadata(12)==compact_payload_count.and. &
        metadata(8)>=0.and.metadata(8)<=huge(nsamples).and.metadata(9)>=0.and.metadata(9)<=identity%step.and. &
        metadata(10)<=identity%step.and.ieee_is_finite(word_to_real(metadata(11))).and. &
        ieee_is_finite(word_to_real(metadata(13))).and.ieee_is_finite(word_to_real(metadata(14))).and. &
        word_to_real(metadata(14))>=0,MPI_COMM_WORLD,'compact checkpoint metadata')
      call checkpoint_state_require(all(ieee_is_finite(values)),MPI_COMM_WORLD,'nonfinite compact payload')
      call transfer(production_measure,.false.,plane_measure_h)
      call checkpoint_state_require(all(plane_measure_h==values(:,:,1,1)).and.all(plane_measure_h>0), &
        MPI_COMM_WORLD,'compact plane geometry mismatch')
      plane_sum_h=values(:,:,1,2:13);plane_endpoint_h=values(:,:,1,18:29)
      call transfer(production_sum,.true.,plane_sum_h);call transfer(production_endpoint,.true.,plane_endpoint_h)
      if(jrk==0) then
        call transfer(production_wall_measure,.false.,wall_measure_h)
        call checkpoint_state_require(all(wall_measure_h==values(:,1,1,14)).and.all(wall_measure_h>0), &
          MPI_COMM_WORLD,'compact wall geometry mismatch')
        wall_sum_h=values(:,1,1,15:17);wall_endpoint_h=values(:,1,1,30:32)
        call transfer(production_wall_sum,.true.,wall_sum_h);call transfer(production_wall_endpoint,.true.,wall_endpoint_h)
      else
        call checkpoint_state_require(.true.,MPI_COMM_WORLD,'compact wall geometry mismatch')
      endif
      sample_count=metadata(8);sampling_start_step=metadata(9);last_sample_step=metadata(10)
      sampling_start_time=word_to_real(metadata(11));last_sample_time=word_to_real(metadata(13))
      time_weight=word_to_real(metadata(14));nsamples=sample_count
    endif
  end subroutine

  subroutine require_compact_clock(actual,expected)
    type(checkpoint_state_identity),intent(in) :: actual,expected
    call checkpoint_state_require(actual%step==expected%step.and.actual%time==expected%time.and. &
      actual%dt_used==expected%dt_used.and.actual%dt_next==expected%dt_next, &
      MPI_COMM_WORLD,'compact state group clock mismatch')
  end subroutine

  subroutine compact_global_measure(values,budget)
    use commvar, only: ia,ja,im,jm
    use parallel, only: ig0,jg0,jrk
    real(real_kind),allocatable,intent(out) :: values(:,:,:,:)
    integer(int64),intent(in) :: budget
    real(real_kind),allocatable :: plane(:,:),wall(:)
    integer :: ni,nj,ierr,status
    call checkpoint_state_require(budget>=int(ia+1,int64)*(ja+1)*2*compact_payload_count*real_bytes+ &
      int(im+1,int64)*(jm+2)*real_bytes+4096,MPI_COMM_WORLD,'global measure staging budget')
    allocate(values(ia+1,ja+1,2,compact_payload_count),plane(0:im,0:jm),stat=status)
    call check(status,'global measure allocation')
    values=0
    call transfer(production_measure,.false.,plane)
    ni=im;nj=jm
    if(ig0+im==ia) ni=ni+1
    if(jg0+jm==ja) nj=nj+1
    values(ig0+1:ig0+ni,jg0+1:jg0+nj,1,1)=plane(0:ni-1,0:nj-1)
    if(jrk==0) then
      allocate(wall(0:im))
      call transfer(production_wall_measure,.false.,wall)
      values(ig0+1:ig0+ni,1,1,14)=wall(0:ni-1)
    endif
    call MPI_Allreduce(MPI_IN_PLACE,values,size(values),mpi_real_type,MPI_SUM,MPI_COMM_WORLD,ierr)
    call check(ierr,'global compact geometry reduction')
  end subroutine

  integer(int64) function compact_statistics_host_bytes() result(bytes)
    bytes=0
    if(allocated(inherited_statistics)) bytes=real_bytes*size(inherited_statistics,kind=int64)
  end function

  subroutine transfer(buffer,writing,values)
    use commvar, only: im,jm
    integer(c_int),intent(in) :: buffer
    logical,intent(in) :: writing
    real(real_kind),intent(inout) :: values(*)
    integer(c_size_t) :: count
    select case(buffer)
    case(production_measure);count=int(im+1,c_size_t)*(jm+1)
    case(production_sum,production_endpoint);count=int(im+1,c_size_t)*(jm+1)*12
    case(production_wall_measure);count=int(im+1,c_size_t)
    case(production_wall_sum,production_wall_endpoint);count=int(im+1,c_size_t)*3
    case default
      call local_check(-1,'invalid compact statistics buffer');return
    end select
    call local_check(production_transfer_cpp(cpp_device_context(),buffer,int(merge(1,0,writing),c_int), &
      values,count*real_bytes),'compact statistics output/restore transfer')
  end subroutine
end module
