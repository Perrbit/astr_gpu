module insitu_temporal_history
  use iso_fortran_env, only: int64
  use astr_precision, only: real_kind,real_bytes,real_to_word,word_to_real
  use ieee_arithmetic, only: ieee_is_finite
  use insitu_run_config, only: catalog_product
  use insitu_product_schedule, only: runtime_product_state,runtime_catalog_count,runtime_catalog_state,runtime_active,runtime_waiting
  use insitu_temporal_analysis
  implicit none
  private
  public :: temporal_history,histories,reset_temporal_histories,temporal_catalog_present
  public :: temporal_key,temporal_sampling_same,initialize_temporal_history,observe_temporal_clock,push_temporal_history
  public :: pack_temporal_histories,unpack_temporal_histories,resize_temporal_block
  public :: discard_temporal_history
  public :: temporal_array_bytes
  type :: temporal_history
    logical :: configured=.false.,interval_uniform=.true.,was_active=.true.
    type(catalog_product) :: definition
    integer(int64) :: generation=0,samples=0,last_step=-1,last_seen_step=-1,first_step=-1,archived_step=-1
    integer(int64) :: welch_samples=0,welch_first_step=-1,welch_last_step=-1
    real(real_kind) :: last_time=0,last_dt=0,first_time=0
    real(real_kind) :: welch_first_time=0,welch_last_time=0
    real(real_kind),allocatable :: latest(:,:)
    type(time_block) :: block
    type(welch_state) :: spectrum
  end type
  ! Only rank zero retains line/probe history; other ranks keep empty collections.
  type(temporal_history),allocatable,save :: histories(:)
contains
  subroutine reset_temporal_histories()
    if(allocated(histories)) deallocate(histories)
    allocate(histories(0))
  end subroutine

  subroutine discard_temporal_history(index)
    integer,intent(in) :: index
    type(temporal_history) :: empty
    empty%generation=histories(index)%generation
    histories(index)=empty
  end subroutine

  integer(int64) function temporal_array_bytes(index) result(bytes)
    integer,intent(in) :: index
    integer(int64) :: segments,generation,spectral_bytes
    integer :: filled
    real(real_kind) :: spacing
    logical :: uniform
    bytes=0
    associate(s=>histories(index))
    if(allocated(s%latest)) bytes=bytes+size(s%latest,kind=int64)*real_bytes
    if(allocated(s%block%coordinates)) bytes=bytes+size(s%block%coordinates,kind=int64)*real_bytes
    if(allocated(s%block%values)) bytes=bytes+size(s%block%values,kind=int64)*real_bytes
    if(allocated(s%block%times)) bytes=bytes+size(s%block%times,kind=int64)*real_bytes
    if(allocated(s%block%steps)) bytes=bytes+size(s%block%steps,kind=int64)*8
    if(allocated(s%block%gaps)) bytes=bytes+size(s%block%gaps,kind=int64)*storage_size(.true.)/8
    if(allocated(s%block%mean_coverage)) bytes=bytes+size(s%block%mean_coverage,kind=int64)*real_bytes
    call welch_info(s%spectrum,segments,filled,generation,spacing,uniform,spectral_bytes)
    bytes=bytes+spectral_bytes
    end associate
  end function

  logical function temporal_catalog_present() result(present)
    type(runtime_product_state) :: state
    integer :: p
    present=.false.
    do p=1,runtime_catalog_count()
      call runtime_catalog_state(p,state)
      if(state%version==0) cycle
      if(state%definition%view%kind=='time_space'.or.state%definition%view%kind=='probe_spectrum') present=.true.
    enddo
  end function

  function temporal_key(definition) result(key)
    type(catalog_product),intent(in) :: definition
    integer(int64),allocatable :: key(:)
    integer :: source,provider,kind
    source=0;provider=0;kind=1
    if(definition%source=='reynolds') source=1
    if(definition%source=='favre') source=2
    if(definition%statistics_source=='selected-time') provider=1
    if(definition%statistics_source=='velocity-time') provider=2
    if(definition%statistics_source=='solver-samples') provider=3
    if(definition%view%kind=='probe_spectrum') kind=2
    key=[int(kind,int64),int(source,int64),int(provider,int64),int(definition%axis,int64),definition%index, &
      reshape(definition%ranges,[6]),definition%temporal%sample_steps,definition%temporal%segment_length, &
      int(merge(1,0,definition%fluctuation),int64),int(size(definition%fields),int64),int(definition%fields,int64)]
  end function

  logical function temporal_sampling_same(s,definition) result(same)
    type(temporal_history),intent(in) :: s
    type(catalog_product),intent(in) :: definition
    integer(int64),allocatable :: a(:),b(:)
    same=.false.
    if(.not.s%configured) return
    a=temporal_key(s%definition);b=temporal_key(definition)
    if(size(a)/=size(b)) return
    same=all(a==b)
  end function

  subroutine initialize_temporal_history(index,definition,coordinates,start_step,ok,coverage)
    integer,intent(in) :: index
    integer(int64),intent(in) :: start_step
    type(catalog_product),intent(in) :: definition
    real(real_kind),intent(in) :: coordinates(:,:)
    logical,intent(out) :: ok
    real(real_kind),optional,intent(in) :: coverage(3)
    type(temporal_history),allocatable :: expanded(:)
    type(temporal_history) :: fresh
    integer :: capacity,status,n,q,donor
    integer(int64) :: generation
    type(runtime_product_state) :: state
    ok=.false.
    if(.not.allocated(histories)) call reset_temporal_histories()
    if(index>size(histories)) then
      allocate(expanded(index),stat=status)
      if(status/=0) return
      expanded(:size(histories))=histories;call move_alloc(expanded,histories)
    endif
    generation=max(histories(index)%generation+1,start_step+1)
    n=size(definition%fields);capacity=int(definition%temporal%block_samples)
    if(capacity==0) capacity=int(definition%temporal%segment_length)
    call configure_time_block(fresh%block,coordinates,n,capacity,ok,with_coverage=definition%source/='instantaneous')
    if(.not.ok) return
    allocate(fresh%latest(size(coordinates,2),n),stat=status)
    ok=status==0
    if(.not.ok) return
    fresh%latest=0
    if(definition%view%kind=='probe_spectrum') then
      call configure_welch(fresh%spectrum,int(definition%temporal%segment_length),n,ok)
      if(.not.ok) return
    endif
    donor=0
    do q=1,size(histories)
      if(q==index) cycle
      if(.not.temporal_sampling_same(histories(q),definition)) cycle
      call runtime_catalog_state(q,state)
      if(state%status/=runtime_active.and.state%status/=runtime_waiting) cycle
      if(histories(q)%last_step<0.or.histories(q)%last_step>start_step) cycle
      if(allocated(fresh%block%mean_coverage)) then
        if(.not.present(coverage)) cycle
        if(histories(q)%last_step/=start_step) cycle
        if(any(histories(q)%block%last_coverage/=coverage)) cycle
      endif
      if(modulo(start_step-histories(q)%last_step,definition%temporal%sample_steps)/=0) cycle
      if(start_step-histories(q)%last_step>definition%temporal%sample_steps) cycle
      if(donor>0) then
        if(histories(q)%last_step<=histories(donor)%last_step) cycle
      endif
      donor=q
    enddo
    if(donor>0) then
      fresh%spectrum=histories(donor)%spectrum;fresh%latest=histories(donor)%latest
      fresh%samples=histories(donor)%samples;fresh%last_step=histories(donor)%last_step
      fresh%last_seen_step=histories(donor)%last_seen_step;fresh%last_time=histories(donor)%last_time
      fresh%last_dt=histories(donor)%last_dt;fresh%interval_uniform=histories(donor)%interval_uniform
      fresh%first_step=histories(donor)%first_step;fresh%first_time=histories(donor)%first_time
      fresh%welch_samples=histories(donor)%welch_samples
      fresh%welch_first_step=histories(donor)%welch_first_step;fresh%welch_last_step=histories(donor)%welch_last_step
      fresh%welch_first_time=histories(donor)%welch_first_time;fresh%welch_last_time=histories(donor)%welch_last_time
      fresh%block%samples=fresh%samples;fresh%block%last_step=fresh%last_step;fresh%block%last_time=fresh%last_time
      fresh%block%last_coverage=histories(donor)%block%last_coverage
      if(fresh%last_step==start_step) then
        fresh%block%filled=1;fresh%block%steps(1)=fresh%last_step;fresh%block%times(1)=fresh%last_time
        fresh%block%values(:,:,1)=fresh%latest
        if(allocated(fresh%block%mean_coverage)) fresh%block%mean_coverage(:,1)=fresh%block%last_coverage
      endif
    endif
    fresh%definition=definition;fresh%configured=.true.;fresh%generation=generation
    histories(index)=fresh
  end subroutine

  subroutine resize_temporal_block(s,capacity,ok)
    type(temporal_history),intent(inout) :: s
    integer,intent(in) :: capacity
    logical,intent(out) :: ok
    type(time_block) :: replacement
    ok=.false.
    if(s%block%filled/=0) return
    call configure_time_block(replacement,s%block%coordinates,s%block%fields,capacity,ok, &
      with_coverage=allocated(s%block%mean_coverage))
    if(.not.ok) return
    replacement%number=s%block%number;replacement%samples=s%block%samples
    replacement%last_step=s%block%last_step;replacement%last_time=s%block%last_time
    replacement%last_coverage=s%block%last_coverage
    s%block=replacement
  end subroutine

  subroutine observe_temporal_clock(step,dt,fixed)
    integer(int64),intent(in) :: step
    real(real_kind),intent(in) :: dt
    logical,intent(in) :: fixed
    integer :: p
    if(.not.allocated(histories)) return
    do p=1,size(histories)
      if(.not.histories(p)%configured.or.step<=histories(p)%last_seen_step) cycle
      if(histories(p)%last_step>=0) histories(p)%interval_uniform=histories(p)%interval_uniform.and.fixed.and. &
        dt==histories(p)%last_dt
      histories(p)%last_seen_step=step
    enddo
  end subroutine

  subroutine push_temporal_history(s,step,time,dt,fixed,values,spectral_sample,ok,coverage)
    type(temporal_history),intent(inout) :: s
    integer(int64),intent(in) :: step
    real(real_kind),intent(in) :: time,dt,values(:,:)
    logical,intent(in) :: fixed,spectral_sample
    logical,intent(out) :: ok
    real(real_kind),optional,intent(in) :: coverage(3)
    logical :: continuous,uniform
    continuous=s%last_step<0.or.step-s%last_step==s%definition%temporal%sample_steps
    uniform=fixed.and.s%interval_uniform
    call push_time_block(s%block,step,time,values,continuous,ok,coverage)
    if(.not.ok) return
    if(s%definition%view%kind=='probe_spectrum'.and.spectral_sample) then
      if(s%welch_samples==0) uniform=fixed
      call push_welch(s%spectrum,values(1,:),dt*real(s%definition%temporal%sample_steps,real_kind), &
        continuous,uniform,ok)
      if(.not.ok) return
      if(s%welch_samples==0) then
        s%welch_first_step=step;s%welch_first_time=time
      endif
      s%welch_samples=s%welch_samples+1;s%welch_last_step=step;s%welch_last_time=time
    endif
    s%latest=values;s%last_step=step;s%last_time=time;s%last_dt=dt;s%last_seen_step=step
    if(s%samples==0) then
      s%first_step=step;s%first_time=time
    endif
    s%samples=s%samples+1;s%interval_uniform=.true.
  end subroutine

  subroutine pack_temporal_histories(step,time,words,ok)
    integer(int64),intent(in) :: step
    real(real_kind),intent(in) :: time
    integer(int64),allocatable,intent(out) :: words(:)
    logical,intent(out) :: ok
    type :: record
      integer(int64),allocatable :: words(:)
    end type
    type(record),allocatable :: records(:)
    integer(int64),allocatable :: key(:),block_words(:),spectrum_words(:)
    integer(int64) :: total
    integer :: n,p,j,k,cursor,status
    ok=.false.;n=0
    if(allocated(histories)) n=size(histories)
    allocate(records(n),stat=status)
    if(status/=0) return
    total=6
    do p=1,n
      if(.not.histories(p)%configured) cycle
      associate(s=>histories(p))
      key=temporal_key(s%definition)
      call pack_time_block(s%block,block_words,ok)
      if(.not.ok) return
      if(s%definition%view%kind=='probe_spectrum') then
        call pack_welch(s%spectrum,spectrum_words,ok)
        if(.not.ok) return
      else
        allocate(spectrum_words(0))
      endif
      allocate(records(p)%words(21_int64+size(key,kind=int64)+size(s%latest,kind=int64)+ &
        size(block_words,kind=int64)+size(spectrum_words,kind=int64)),stat=status)
      if(status/=0) then
        ok=.false.;return
      endif
      records(p)%words(:21)=[int(p,int64),size(key,kind=int64),size(block_words,kind=int64), &
        size(spectrum_words,kind=int64),s%generation,s%samples,s%last_step,s%last_seen_step, &
        real_to_word(s%last_time),real_to_word(s%last_dt),int(merge(1,0,s%interval_uniform),int64),size(s%latest,kind=int64), &
        s%first_step,real_to_word(s%first_time),s%archived_step,int(merge(1,0,s%was_active),int64), &
        s%welch_samples,s%welch_first_step,s%welch_last_step,real_to_word(s%welch_first_time),real_to_word(s%welch_last_time)]
      j=22;k=size(key);records(p)%words(j:j+k-1)=key;j=j+k
      k=size(s%latest);records(p)%words(j:j+k-1)=reshape(real_to_word(s%latest),[k]);j=j+k
      k=size(block_words);records(p)%words(j:j+k-1)=block_words;j=j+k
      records(p)%words(j:)=spectrum_words
      total=total+1+size(records(p)%words,kind=int64)
      deallocate(spectrum_words,block_words)
      end associate
    enddo
    allocate(words(total),stat=status)
    if(status/=0) then
      ok=.false.;return
    endif
    k=0
    do p=1,n
      if(allocated(records(p)%words)) k=k+1
    enddo
    words(:6)=[int(z'4153545254483033',int64),3_int64,int(real_bytes,int64),step,real_to_word(time),int(k,int64)]
    cursor=7
    do p=1,n
      if(.not.allocated(records(p)%words)) cycle
      words(cursor)=size(records(p)%words,kind=int64);cursor=cursor+1
      k=size(records(p)%words);words(cursor:cursor+k-1)=records(p)%words;cursor=cursor+k
    enddo
    ok=.true.
  end subroutine

  subroutine unpack_temporal_histories(words,step,time,ok)
    integer(int64),intent(in) :: words(:),step
    real(real_kind),intent(in) :: time
    logical,intent(out) :: ok
    type(runtime_product_state) :: state
    integer(int64),allocatable :: key(:)
    integer :: p,n,record_index,cursor,last,j,k,m,status
    ok=.false.
    if(size(words)<6) return
    if(any(words(:5)/=[int(z'4153545254483033',int64),3_int64,int(real_bytes,int64),step,real_to_word(time)])) return
    if(words(6)<0.or.words(6)>runtime_catalog_count()) return
    call reset_temporal_histories();deallocate(histories)
    allocate(histories(runtime_catalog_count()),stat=status)
    if(status/=0) return
    cursor=7;n=int(words(6))
    do record_index=1,n
      if(cursor>size(words)) return
      if(words(cursor)<21.or.words(cursor)>size(words)-cursor) return
      last=cursor+int(words(cursor));cursor=cursor+1
      associate(header=>words(cursor:cursor+20))
      if(header(1)<1.or.header(1)>runtime_catalog_count().or.any(header(2:4)<0).or.header(12)<1) return
      if(header(2)+header(3)+header(4)+header(12)/=last-cursor-20) return
      p=int(header(1));call runtime_catalog_state(p,state)
      if(histories(p)%configured) return
      if(state%definition%view%kind/='time_space'.and.state%definition%view%kind/='probe_spectrum') return
      key=temporal_key(state%definition)
      if(size(key)/=header(2)) return
      j=cursor+21;k=size(key)
      if(any(words(j:j+k-1)/=key)) return
      j=j+k;k=int(header(12));m=j+k
      call unpack_time_block(words(m:m+int(header(3))-1),histories(p)%block,ok)
      if(.not.ok) return
      ok=.false.
      if(k/=histories(p)%block%nodes*histories(p)%block%fields) return
      allocate(histories(p)%latest(histories(p)%block%nodes,histories(p)%block%fields),stat=status)
      if(status/=0) return
      histories(p)%latest=reshape(word_to_real(words(j:j+k-1)),shape(histories(p)%latest))
      m=m+int(header(3))
      if(state%definition%view%kind=='probe_spectrum') then
        call unpack_welch(words(m:last),histories(p)%spectrum,ok)
        if(.not.ok) return
      elseif(header(4)/=0) then
        return
      endif
      ok=.false.
      if(header(5)<1.or.header(6)<0.or.header(7)<-1.or.header(8)>step.or.header(11)<0.or.header(11)>1) return
      if(header(7)/=histories(p)%block%last_step.or.header(6)/=histories(p)%block%samples) return
      if(word_to_real(header(9))/=histories(p)%block%last_time) return
      if(.not.all(ieee_is_finite(word_to_real(header(9:10)))).or..not.all(ieee_is_finite(histories(p)%latest))) return
      histories(p)%definition=state%definition;histories(p)%generation=header(5);histories(p)%samples=header(6)
      histories(p)%last_step=header(7);histories(p)%last_seen_step=header(8)
      histories(p)%last_time=word_to_real(header(9));histories(p)%last_dt=word_to_real(header(10))
      histories(p)%interval_uniform=header(11)==1;histories(p)%configured=.true.
      if(header(13)<-1.or.header(13)>header(7).or.header(15)<-1.or.header(15)>header(7).or. &
         header(16)<0.or.header(16)>1) return
      if(.not.ieee_is_finite(word_to_real(header(14)))) return
      if(word_to_real(header(14))<0.or.word_to_real(header(14))>histories(p)%last_time) return
      histories(p)%first_step=header(13);histories(p)%first_time=word_to_real(header(14))
      histories(p)%archived_step=header(15);histories(p)%was_active=header(16)==1
      if(header(17)<0.or.header(17)>header(6).or.header(18)<-1.or.header(19)>header(7)) return
      if(.not.all(ieee_is_finite(word_to_real(header(20:21))))) return
      if(header(17)==0) then
        if(header(18)/=-1.or.header(19)/=-1.or.any(word_to_real(header(20:21))/=0)) return
      else
        if(header(18)<0.or.header(19)<header(18)) return
        if(word_to_real(header(20))<0.or.word_to_real(header(21))<word_to_real(header(20)).or. &
          word_to_real(header(21))>histories(p)%last_time) return
      endif
      histories(p)%welch_samples=header(17);histories(p)%welch_first_step=header(18);histories(p)%welch_last_step=header(19)
      histories(p)%welch_first_time=word_to_real(header(20));histories(p)%welch_last_time=word_to_real(header(21))
      end associate
      cursor=last+1
    enddo
    ok=cursor==size(words)+1
  end subroutine
end module
