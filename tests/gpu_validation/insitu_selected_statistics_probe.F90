program selected_statistics_probe
  use mpi
  use iso_fortran_env, only: int64
  use astr_precision, only: real_kind
  use insitu_run_config, only: catalog_product
  use insitu_selected_statistics
  use ieee_arithmetic, only: ieee_value,ieee_quiet_nan
#ifdef _CUDA
  use cudafor
#endif
  implicit none
  type(selected_statistics) :: series,limited,resumed
  type(catalog_product) :: products(3)
  integer :: ierr,rank,ranks,topology(3),origin(3),cells(3),intervals(3),coords(3),d,p,g,n,local(3),step
  integer :: global_count,local_count,read_status
  integer(int64) :: host_bytes,device_bytes,temporary
  integer(int64),allocatable :: words(:),restored_words(:)
  real(real_kind),allocatable :: q(:,:,:,:)
  real(real_kind) :: value,expected,worst,nan,time
  logical :: ok,active(3),periodic(3)
  character(32) :: argument
#ifdef _CUDA
  type(selected_statistics) :: device_series,device_resumed
  real(real_kind),device,allocatable :: q_d(:,:,:,:)
  real(real_kind),allocatable :: observed(:,:)
  integer :: device_count,status
#endif
  call MPI_Init(ierr);call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  call get_command_argument(1,argument)
  read(argument,*,iostat=read_status) topology
  call require(read_status==0.and.product(topology)==ranks,'requested topology')
  intervals=[12,12,12];cells=intervals/topology
  call require(all(cells*topology==intervals),'integral local cells')
  coords=[mod(rank,topology(1)),mod(rank/topology(1),topology(2)),rank/(topology(1)*topology(2))]
  origin=coords*cells;periodic=[.false.,.false.,.true.]
  nan=ieee_value(0.e0_real_kind,ieee_quiet_nan);worst=0
  allocate(q(0:cells(1),0:cells(2),0:cells(3),5))
  do p=1,3
    products(p)%id='request'//achar(iachar('0')+p)
    products(p)%source='reynolds';products(p)%statistics_source='selected-time'
    products(p)%view%kind='slice';products(p)%axis=1;products(p)%index=6
    products(p)%ranges(1,:)=0;products(p)%ranges(2,:)=intervals
    products(p)%fields=[5,6];products(p)%valid=.true.
  enddo
  products(2)%source='favre';products(2)%view%camera=[1.e0_real_kind,2.e0_real_kind,3.e0_real_kind]
  products(3)%fields=[2];active=[.true.,.true.,.false.]
  call series%configure([.25e0_real_kind,9.75e0_real_kind],intervals,origin,cells,periodic,.false.,ok)
  call require(ok,'host configuration')
  call series%reconcile(products,active,ok);call require(ok,'catalog support reconciliation')
  call require(size(series%groups)==1.and.series%find(products(1)%id)==series%find(products(2)%id),'exact support reuse')
  local_count=size(series%groups(1)%keys)
  call MPI_Allreduce(local_count,global_count,1,MPI_INTEGER,MPI_SUM,MPI_COMM_WORLD,ierr)
  call require(global_count==13*12,'unique nonperiodic/periodic ownership')
#ifdef _CUDA
  status=cudaGetDeviceCount(device_count);call require(status==0.and.device_count>0,'GPU availability')
  status=cudaSetDevice(mod(rank,device_count));call require(status==0,'GPU binding')
  allocate(q_d(0:cells(1),0:cells(2),0:cells(3),5))
  call device_series%configure(series%window,intervals,origin,cells,periodic,.true.,ok)
  call require(ok,'device configuration')
  call device_series%reconcile(products,active,ok);call require(ok,'device support allocation')
#endif
  do step=0,10
    time=real(step,real_kind)
    if(step==3) then
      products(1)%view%camera=99;products(1)%interval_steps=100;active(3)=.true.
    endif
    if(step==5) active=.false.
    if(step==7) active=.true.
    call series%reconcile(products,active,ok);call require(ok,'runtime catalog update')
#ifdef _CUDA
    call device_series%reconcile(products,active,ok);call require(ok,'device runtime catalog update')
#endif
    q=nan
    do g=1,size(series%groups)
      if(.not.series%groups(g)%active) cycle
      do n=1,size(series%groups(g)%keys)
        local=series%groups(g)%indices(:,n)
        q(local(1),local(2),local(3),:)=[2.e0_real_kind,2*time,0.e0_real_kind,0.e0_real_kind, &
          2.5e0_real_kind*(100+time)+time*time]
      enddo
    enddo
    call series%sample_host(int(step,int64),time,0,q,3.e0_real_kind,2.5e0_real_kind,ok)
    call require(ok,'completed-state host sample')
    call series%sample_host(int(step,int64),time,0,q,3.e0_real_kind,2.5e0_real_kind,ok)
    call require(ok,'idempotent repeated observation')
#ifdef _CUDA
    q_d=q
    call device_series%sample_device(int(step,int64),time,0,q_d,3.e0_real_kind,2.5e0_real_kind,ok)
    call require(ok,'completed-state device sample')
    call device_series%sample_device(int(step,int64),time,0,q_d,3.e0_real_kind,2.5e0_real_kind,ok)
    call require(ok,'idempotent device observation')
    do g=1,size(series%groups)
      n=size(series%groups(g)%keys)
      allocate(observed(max(1,n),size(series%groups(g)%states,2)))
      observed=device_series%groups(g)%device%states
      if(n>0) worst=max(worst,maxval(abs(observed(:n,:)-series%groups(g)%states)/ &
        max(1.e0_real_kind,abs(observed(:n,:)),abs(series%groups(g)%states))))
      deallocate(observed)
    enddo
#endif
    if(step==4) then
      call series%pack(int(step,int64),time,2097152_int64,words,ok)
      call require(ok,'host checkpoint packing')
      call resumed%configure(series%window,intervals,origin,cells,periodic,.false.,ok)
      call require(ok,'host restore configuration')
      call resumed%restore_bytes(int(step,int64),time,2097152_int64,words,host_bytes,device_bytes,temporary,ok)
      call require(ok.and.device_bytes==0,'host restore requirements')
      call resumed%restore(int(step,int64),time,2097152_int64,words,ok)
      call require(ok,'host checkpoint restore')
#ifdef _CUDA
      call device_series%pack(int(step,int64),time,2097152_int64,words,ok)
      call require(ok,'device checkpoint packing')
      call device_resumed%configure(series%window,intervals,origin,cells,periodic,.true.,ok)
      call require(ok,'device restore configuration')
      call device_resumed%restore_bytes(int(step,int64),time,2097152_int64,words,host_bytes,device_bytes,temporary,ok)
      call require(ok.and.device_bytes>0,'device restore requirements')
      call device_resumed%restore(int(step,int64),time,2097152_int64,words,ok)
      call require(ok,'device checkpoint restore')
#endif
    endif
    if(step>=4) then
      call resumed%reconcile(products,active,ok);call require(ok,'restored catalog reconciliation')
      call resumed%sample_host(int(step,int64),time,0,q,3.e0_real_kind,2.5e0_real_kind,ok)
      call require(ok,'restored host completed step or initial identity')
      call series%pack(int(step,int64),time,2097152_int64,words,ok);call require(ok,'continuous host state')
      call resumed%pack(int(step,int64),time,2097152_int64,restored_words,ok)
      call require(ok.and.size(words)==size(restored_words),'restored host shape')
      call require(all(words==restored_words),'bitwise host continuation through missing samples')
#ifdef _CUDA
      call device_resumed%reconcile(products,active,ok);call require(ok,'restored device catalog reconciliation')
      call device_resumed%sample_device(int(step,int64),time,0,q_d,3.e0_real_kind,2.5e0_real_kind,ok)
      call require(ok,'restored device completed step or initial identity')
      call device_series%pack(int(step,int64),time,2097152_int64,words,ok);call require(ok,'continuous device state')
      call device_resumed%pack(int(step,int64),time,2097152_int64,restored_words,ok)
      call require(ok.and.size(words)==size(restored_words),'restored device shape')
      call require(all(words==restored_words),'bitwise device continuation through missing samples')
#endif
    endif
  enddo
  call require(size(series%groups)==2,'late field gets separate history')
  call require(series%groups(1)%covered==6.5e0_real_kind,'existing coverage excludes real gap')
  call require(series%groups(2)%covered==3.75e0_real_kind,'late coverage starts at introduction')
  call require(all(series%groups%segments==2),'gap produces a distinct segment')
  do g=1,2
    do n=1,size(series%groups(g)%keys)
      local=series%groups(g)%indices(:,n);d=merge(5,2,g==1)
      call series%value(g,local,d,.false.,value,ok);call require(ok,'selected mean access')
      expected=merge(100+(7.96875e0_real_kind+23.03125e0_real_kind)/6.5e0_real_kind, &
        (3.5e0_real_kind+23.03125e0_real_kind)/3.75e0_real_kind,g==1)
      worst=max(worst,abs(value-expected)/max(1.e0_real_kind,abs(value),abs(expected)))
      call require(abs(value-expected)/max(1.e0_real_kind,abs(value),abs(expected))<=2.e-10_real_kind,'analytic gap mean')
    enddo
  enddo
  call series%bytes(host_bytes,device_bytes);call require(device_bytes==0,'host storage accounting')
  call require(host_bytes==88_int64+int(local_count,int64)*((20+9*8)+(20+6*8)), &
    'selected storage scales with support plus fixed coverage metadata')
  ! An invalid energy cannot invalidate a velocity-only request at the same node.
  q(:,:,:,5)=nan
  call series%sample_host(11_int64,11.e0_real_kind,0,q,3.e0_real_kind,2.5e0_real_kind,ok)
  if(size(series%groups(1)%keys)>0) then
    call require(.not.ok.and.series%groups(1)%failed.and..not.series%groups(2)%failed,'isolate invalid selected fields')
  else
    call require(ok,'empty owner does not invent a local sample failure')
  endif
  call require(series%groups(2)%last_step==11,'healthy selected history still advances')
#ifdef _CUDA
  q_d=q
  call device_series%sample_device(11_int64,11.e0_real_kind,0,q_d,3.e0_real_kind,2.5e0_real_kind,ok)
  call require(device_series%groups(1)%failed.eqv.series%groups(1)%failed,'device point failure scope')
  call require(.not.device_series%groups(2)%failed.and.device_series%groups(2)%last_step==11,'device healthy history advances')
#endif
  call limited%configure(series%window,intervals,origin,cells,periodic,.false.,ok)
  call require(ok,'bounded allocation configuration')
  call limited%reconcile(products,active,ok,approve=deny_allocation)
  call require(ok.and.size(limited%groups)==0.and.all(limited%requests%resource_failed),'resource refusal before state allocation')
  call limited%release()
  products(1)%axis=3;products(1)%index=0;products(1)%fields=[13];active=[.true.,.false.,.false.]
  call series%reconcile(products,active,ok);call require(ok,'private stencil support')
  g=series%find(products(1)%id);local_count=size(series%groups(g)%keys)
  call MPI_Allreduce(local_count,global_count,1,MPI_INTEGER,MPI_SUM,MPI_COMM_WORLD,ierr)
  call require(global_count==13*13*7,'wrapped seven-node diagnostic support')
  call check_joint_history(.false.)
#ifdef _CUDA
  call check_joint_history(.true.)
#endif
  call require(worst<=2.e-10_real_kind,'same-state scaled gate')
  print '(a,i0,a,es24.16,a,i0)', 'PASS selected statistics rank=',rank,' max_scaled_error=',worst,' unique_support=',global_count
  call series%release()
  call resumed%release()
#ifdef _CUDA
  call device_series%release();call device_resumed%release();deallocate(q_d)
#endif
  deallocate(q);call MPI_Finalize(ierr)
contains
  subroutine check_joint_history(on_device)
    logical,intent(in) :: on_device
    type(selected_statistics) :: histories,restored
    type(catalog_product) :: definitions(2)
    integer(int64),allocatable :: packed(:),after(:)
    integer :: scenario,s
    logical :: enabled(2),valid
    definitions=products(:2)
    definitions%axis=1;definitions%index=6
    definitions(1)%fields=[5,6];definitions(2)%fields=[2]
    definitions(1)%view%kind='slice';definitions(2)%view%kind='slice'
    q(:,:,:,1)=2;q(:,:,:,2:4)=0;q(:,:,:,5)=250
#ifdef _CUDA
    if(on_device) q_d=q
#endif
    do scenario=1,2
      call histories%configure([0.e0_real_kind,10.e0_real_kind],intervals,origin,cells,periodic,on_device,valid)
      call require(valid,'joint-history setup')
      do s=0,10
        enabled(1)=s/=3.and.s/=4
        enabled(2)=s/=4.and.s/=5
        if(scenario==2) enabled(2)=enabled(1)
        call histories%reconcile(definitions,enabled,valid);call require(valid,'joint-history reconciliation')
#ifdef _CUDA
        if(on_device) then
          call histories%sample_device(int(s,int64),real(s,real_kind),0,q_d,3.e0_real_kind,2.5e0_real_kind,valid)
        else
#endif
          call histories%sample_host(int(s,int64),real(s,real_kind),0,q,3.e0_real_kind,2.5e0_real_kind,valid)
#ifdef _CUDA
        endif
#endif
        call require(valid,'joint-history real sample')
      enddo
      call require(histories%groups(1)%first_time==histories%groups(2)%first_time.and. &
        histories%groups(1)%last_time==histories%groups(2)%last_time.and. &
        histories%groups(1)%covered==histories%groups(2)%covered.and. &
        histories%groups(1)%segments==histories%groups(2)%segments,'ambiguous scalar coverage counterexample')
      call require(histories%matching_coverage(1,2).eqv.(scenario==2),'joint identity distinguishes gap locations')
      call histories%pack(10_int64,10.e0_real_kind,2097152_int64,packed,valid)
      call require(valid,'joint-history pack')
      call restored%configure(histories%window,intervals,origin,cells,periodic,on_device,valid)
      call require(valid,'joint-history restore setup')
      call restored%restore(10_int64,10.e0_real_kind,2097152_int64,packed,valid)
      call require(valid,'joint-history restore')
      call require(restored%matching_coverage(1,2).eqv.(scenario==2),'joint identity survives restart')
      call restored%pack(10_int64,10.e0_real_kind,2097152_int64,after,valid)
      call require(valid.and.size(packed)==size(after),'joint-history roundtrip size')
      call require(all(packed==after),'joint-history roundtrip bits')
      call histories%release();call restored%release()
    enddo
  end subroutine

  subroutine deny_allocation(host_bytes,device_bytes,additional_device,ok)
    integer(int64),intent(in) :: host_bytes,device_bytes,additional_device
    logical,intent(out) :: ok
    ok=.false.
  end subroutine

  subroutine require(condition,message)
    logical,intent(in) :: condition
    character(*),intent(in) :: message
    integer :: err
    if(condition) return
    print '(a,i0,2a)','FAIL rank=',rank,' ',message
    call MPI_Abort(MPI_COMM_WORLD,1,err)
  end subroutine
end program
