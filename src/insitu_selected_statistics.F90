module insitu_selected_statistics
  use iso_fortran_env, only: int64
  use astr_precision, only: real_kind,real_bytes,real_to_word,word_to_real
  use insitu_run_config, only: catalog_product,catalog_wall_mean
  use index_slice_stencil, only: slice_stencil_start
  use insitu_sample_moments, only: sample_moment_size,push_sample_moments,decode_sample_primitives
  use insitu_time_integral, only: clipped_trapezoid
  use ieee_arithmetic, only: ieee_is_finite
#ifdef _CUDA
  use insitu_sample_moments_gpu, only: selected_moment_device,allocate_selected_moments_gpu, &
    release_selected_moments_gpu,gather_selected_moments_gpu,selected_profile_workspace_bytes_gpu, &
    selected_external_workspace_bytes_gpu
#endif
  implicit none
  private
  public :: selected_statistics,selected_statistics_request,selected_statistics_group
  public :: build_selected_support,selected_request_fields
  public :: node_key
  public :: selected_external_sampler
  abstract interface
    subroutine selected_allocation_check(host_bytes,device_bytes,additional_device,ok)
      import int64
      integer(int64),intent(in) :: host_bytes,device_bytes,additional_device
      logical,intent(out) :: ok
    end subroutine
    logical function selected_validity_check(ok) result(valid)
      logical,intent(in) :: ok
    end function
  end interface
  type :: selected_statistics_request
    type(catalog_product) :: definition
    integer :: group=0
    logical :: active=.false.,resource_failed=.false.
  end type
  type :: selected_statistics_group
    integer :: bounds(2,3)=0,field_ids(5)=0,nfields=0,velocity_slots(3)=0
    integer,allocatable :: indices(:,:)
    integer(int64),allocatable :: keys(:)
    real(real_kind),allocatable :: states(:,:)
#ifdef _CUDA
    type(selected_moment_device) :: device
#endif
    integer(int64) :: last_step=-1,segments=0,coverage_id=0,coverage_parent=0,coverage_step=-1
    real(real_kind) :: last_time=0,first_time=0,covered=0
    real(real_kind) :: coverage_weights(2)=0
    logical :: active=.false.,previous=.false.,continuous=.false.,covariance=.false.,failed=.false.
  end type
  type :: selected_statistics
    real(real_kind) :: window(2)=0
    integer :: origin(3)=0,cells(3)=0,intervals(3)=0
    logical :: periodic(3)=.false.,on_device=.false.,configured=.false.
    integer(int64) :: coverage_counter=0
    type(selected_statistics_request),allocatable :: requests(:)
    type(selected_statistics_group),allocatable :: groups(:)
  contains
    procedure :: configure=>configure_selected_statistics
    procedure :: reconcile=>reconcile_selected_statistics
    procedure :: set_active=>set_selected_statistics_active
    procedure :: sample_host=>sample_selected_statistics_host
    procedure :: sample_external=>sample_selected_statistics_external
#ifdef _CUDA
    procedure :: sample_device=>sample_selected_statistics_device
#endif
    procedure :: find=>find_selected_statistics
    procedure :: matching_coverage=>matching_selected_coverage
    procedure :: value=>selected_statistics_value
    procedure :: bytes=>selected_statistics_bytes
    procedure :: pack=>pack_selected_statistics
    procedure :: restore_bytes=>selected_restore_bytes
    procedure :: restore=>restore_selected_statistics
    procedure :: release=>release_selected_statistics
  end type
  abstract interface
    subroutine selected_external_sampler(group,weights,previous,ok)
      import selected_statistics_group,real_kind
      type(selected_statistics_group),intent(inout) :: group
      real(real_kind),intent(in) :: weights(2)
      logical,intent(in) :: previous
      logical,intent(out) :: ok
    end subroutine
  end interface
contains
  subroutine set_selected_statistics_active(this,active,ok)
    class(selected_statistics),intent(inout) :: this
    logical,intent(in) :: active(:)
    logical,intent(out) :: ok
    integer :: p,g
    ok=.false.
    if(.not.this%configured.or.size(active)/=size(this%requests)) return
    this%groups%active=.false.
    do p=1,size(this%requests)
      g=this%requests(p)%group;this%requests(p)%active=.false.
      if(g==0.or..not.active(p)) cycle
      if(this%groups(g)%failed) cycle
      this%requests(p)%active=.true.;this%groups(g)%active=.true.
    enddo
    do g=1,size(this%groups)
      if(.not.this%groups(g)%active) this%groups(g)%continuous=.false.
    enddo
    ok=.true.
  end subroutine

  logical function group_has_state(this,g) result(present_state)
    class(selected_statistics),intent(in) :: this
    integer,intent(in) :: g
    present_state=allocated(this%groups(g)%states)
#ifdef _CUDA
    if(this%on_device) present_state=associated(this%groups(g)%device%states)
#endif
  end function

  subroutine pack_selected_statistics(this,step,time,budget,words,ok)
    class(selected_statistics),intent(in) :: this
    integer(int64),intent(in) :: step,budget
    real(real_kind),intent(in) :: time
    integer(int64),allocatable,intent(out) :: words(:)
    logical,intent(out) :: ok
    integer(int64) :: count,temporary,extra,flags
    real(real_kind),allocatable :: downloaded(:,:)
    integer :: g,n,c,m,pos,status
    logical :: present_state
    ok=.false.
    if(.not.this%configured.or..not.ieee_is_finite(time).or.step<0) return
    count=22;temporary=0
    do g=1,size(this%groups)
      n=size(this%groups(g)%keys);c=sample_moment_size(this%groups(g)%nfields,this%groups(g)%covariance)
      present_state=group_has_state(this,g)
      count=count+25_int64+3_int64*n
      if(present_state) count=count+int(n,int64)*c
      extra=int(n,int64)*8
      if(this%on_device.and.present_state) extra=extra+int(n,int64)*c*real_bytes
      temporary=max(temporary,extra)
      if(this%groups(g)%last_step>step.or.this%groups(g)%last_time>time) return
    enddo
    if(count>huge(pos).or.temporary>budget) return
    if(count>(budget-temporary)/8) return
    allocate(words(int(count)),stat=status)
    if(status/=0) return
    words(:22)=[int(z'4153545253533031',int64),2_int64,int(real_bytes,int64), &
      int(merge(1,0,this%on_device),int64),int(size(this%groups),int64),real_to_word(this%window), &
      int(this%intervals,int64),int(this%origin,int64),int(this%cells,int64), &
      int(merge(1,0,this%periodic),int64),step,real_to_word(time),this%coverage_counter]
    pos=23
    do g=1,size(this%groups)
      n=size(this%groups(g)%keys);c=sample_moment_size(this%groups(g)%nfields,this%groups(g)%covariance)
      present_state=group_has_state(this,g)
      flags=int(merge(1,0,this%groups(g)%active)+merge(2,0,this%groups(g)%previous)+ &
        merge(4,0,this%groups(g)%continuous)+merge(8,0,this%groups(g)%failed)+merge(16,0,present_state),int64)
      words(pos:pos+24)=[int(this%groups(g)%nfields,int64),int(merge(1,0,this%groups(g)%covariance),int64), &
        flags,this%groups(g)%last_step,this%groups(g)%segments,real_to_word(this%groups(g)%last_time), &
        real_to_word(this%groups(g)%first_time),real_to_word(this%groups(g)%covered), &
        int(reshape(this%groups(g)%bounds,[6]),int64),int(this%groups(g)%field_ids,int64), &
        int(this%groups(g)%velocity_slots,int64),int(n,int64),int(c,int64),this%groups(g)%coverage_id]
      pos=pos+25
      words(pos:pos+3*n-1)=int(reshape(this%groups(g)%indices,[3*n]),int64);pos=pos+3*n
      if(.not.present_state) cycle
#ifdef _CUDA
      if(this%on_device) then
        allocate(downloaded(n,c),stat=status)
        if(status/=0) return
        if(n>0) downloaded=this%groups(g)%device%states(:n,:)
      endif
#endif
      do m=1,c
        if(this%on_device) then
          words(pos:pos+n-1)=real_to_word(downloaded(:,m))
        else
          words(pos:pos+n-1)=real_to_word(this%groups(g)%states(:,m))
        endif
        pos=pos+n
      enddo
      if(allocated(downloaded)) deallocate(downloaded)
    enddo
    ok=pos==size(words)+1
  end subroutine

  subroutine selected_restore_bytes(this,step,time,budget,words,host_bytes,device_bytes,temporary,ok)
    class(selected_statistics),intent(in) :: this
    integer(int64),intent(in) :: step,budget,words(:)
    real(real_kind),intent(in) :: time
    integer(int64),intent(out) :: host_bytes,device_bytes,temporary
    logical,intent(out) :: ok
    integer(int64) :: expected(22),header(25),n,components,count,scratch
    integer :: g,pos
    logical :: present_state
    ok=.false.;host_bytes=0;device_bytes=0;temporary=0
    if(.not.this%configured.or.size(words)<22) return
    expected=[int(z'4153545253533031',int64),2_int64,int(real_bytes,int64), &
      int(merge(1,0,this%on_device),int64),words(5),real_to_word(this%window),int(this%intervals,int64), &
      int(this%origin,int64),int(this%cells,int64),int(merge(1,0,this%periodic),int64),step,real_to_word(time),words(22)]
    if(any(words(:22)/=expected).or.words(5)<0.or.words(5)>(size(words,kind=int64)-22)/25.or.words(22)<0) return
    if(size(words,kind=int64)>budget/8) return
    pos=23;host_bytes=8
    do g=1,int(words(5))
      if(size(words)-pos+1<25) return
      header=words(pos:pos+24);pos=pos+25
      if(header(1)<0.or.header(1)>5.or.header(2)<0.or.header(2)>1.or.header(3)<0.or.header(3)>31) return
      n=header(23);components=header(24);present_state=btest(header(3),4)
      if(n<0.or.n>huge(0).or.components<1.or.components>30) return
      if(components/=sample_moment_size(int(header(1)),header(2)==1)) return
      count=n*(3+merge(components,0_int64,present_state))
      if(count>size(words,kind=int64)-pos+1) return
      if(header(25)<0.or.header(25)>words(22)) return
      host_bytes=host_bytes+40+n*(3*storage_size(0)/8+8)
      scratch=n*(components*real_bytes+real_bytes)
      if(present_state) then
        if(this%on_device) then
          device_bytes=device_bytes+max(1_int64,n)*(3*storage_size(0)/8+components*real_bytes)+storage_size(0)/8
        else
          host_bytes=host_bytes+n*components*real_bytes
        endif
        temporary=max(temporary,scratch)
      endif
      pos=pos+int(count)
    enddo
    temporary=temporary+size(words,kind=int64)*8
    ok=pos==size(words)+1.and.temporary<=budget
  end subroutine

  subroutine restore_selected_statistics(this,step,time,budget,words,ok)
    class(selected_statistics),intent(inout) :: this
    integer(int64),intent(in) :: step,budget,words(:)
    real(real_kind),intent(in) :: time
    logical,intent(out) :: ok
    type(selected_statistics) :: restored
    real(real_kind),allocatable :: values(:,:)
    integer(int64) :: header(25),expected(22),flags,remaining
    integer :: groups,g,n,nf,c,m,pos,status,upper(3),row
    logical :: valid,present_state
    ok=.false.
    if(.not.this%configured.or.size(this%groups)/=0.or.size(words)<22) return
    expected=[int(z'4153545253533031',int64),2_int64,int(real_bytes,int64), &
      int(merge(1,0,this%on_device),int64),words(5),real_to_word(this%window),int(this%intervals,int64), &
      int(this%origin,int64),int(this%cells,int64),int(merge(1,0,this%periodic),int64),step,real_to_word(time),words(22)]
    if(any(words(:22)/=expected).or.words(5)<0.or.words(5)>(size(words,kind=int64)-22)/25.or.words(22)<0) return
    if(size(words,kind=int64)>budget/8) return
    groups=int(words(5))
    call restored%configure(this%window,this%intervals,this%origin,this%cells,this%periodic,this%on_device,valid)
    if(.not.valid) return
    deallocate(restored%groups);allocate(restored%groups(groups));pos=23
    restored%coverage_counter=words(22)
    upper=this%cells-1
    where(this%origin+this%cells==this%intervals.and..not.this%periodic) upper=upper+1
    do g=1,groups
      if(size(words)-pos+1<25) goto 900
      header=words(pos:pos+24);pos=pos+25
      if(header(1)<0.or.header(1)>5.or.header(2)<0.or.header(2)>1.or.header(3)<0.or.header(3)>31) goto 900
      nf=int(header(1));flags=header(3);present_state=btest(flags,4)
      restored%groups(g)%nfields=nf;restored%groups(g)%covariance=header(2)==1
      c=sample_moment_size(nf,restored%groups(g)%covariance)
      if(c==0.or.header(24)/=c.or.header(23)<0.or.header(23)>huge(n)) goto 900
      n=int(header(23));remaining=size(words,kind=int64)-pos+1
      if(int(n,int64)*(3+merge(c,0,present_state))>remaining) goto 900
      if(any(abs(header(9:22))>huge(0))) goto 900
      restored%groups(g)%bounds=reshape(int(header(9:14)),[2,3]);restored%groups(g)%field_ids=int(header(15:19))
      restored%groups(g)%velocity_slots=int(header(20:22))
      if(any(restored%groups(g)%field_ids(:nf)>6)) then
        if(restored%groups(g)%covariance.or.any(restored%groups(g)%velocity_slots/=0)) goto 900
        if(any(restored%groups(g)%field_ids(:nf)/=6.and. &
          (restored%groups(g)%field_ids(:nf)<31.or.restored%groups(g)%field_ids(:nf)>34))) goto 900
      else
        if(any(restored%groups(g)%field_ids(:nf)<2)) goto 900
      endif
      do m=2,nf
        if(restored%groups(g)%field_ids(m)<=restored%groups(g)%field_ids(m-1)) goto 900
      enddo
      if(restored%groups(g)%covariance) then
        if(any(restored%groups(g)%velocity_slots<1).or.any(restored%groups(g)%velocity_slots>nf)) goto 900
        if(any(restored%groups(g)%field_ids(restored%groups(g)%velocity_slots)/=[2,3,4])) goto 900
      elseif(any(restored%groups(g)%velocity_slots/=0)) then
        goto 900
      endif
      restored%groups(g)%active=btest(flags,0);restored%groups(g)%previous=btest(flags,1)
      restored%groups(g)%continuous=btest(flags,2);restored%groups(g)%failed=btest(flags,3)
      if(.not.present_state.and..not.restored%groups(g)%failed) goto 900
      restored%groups(g)%last_step=header(4);restored%groups(g)%segments=header(5)
      restored%groups(g)%last_time=word_to_real(header(6));restored%groups(g)%first_time=word_to_real(header(7))
      restored%groups(g)%covered=word_to_real(header(8))
      restored%groups(g)%coverage_id=header(25)
      if(header(25)<0.or.header(25)>restored%coverage_counter) goto 900
      if((restored%groups(g)%covered>0).neqv.(header(25)>0)) goto 900
      if(.not.all(ieee_is_finite(word_to_real(header(6:8))))) goto 900
      if(restored%groups(g)%covered<0.or.header(5)<0.or.header(4)>step.or.header(4)<-1) goto 900
      if(restored%groups(g)%first_time>restored%groups(g)%last_time.or.restored%groups(g)%last_time>time) goto 900
      if(restored%groups(g)%continuous.and..not.restored%groups(g)%previous) goto 900
      if(restored%groups(g)%previous.and.(header(4)<0.or.header(5)==0)) goto 900
      if(.not.restored%groups(g)%previous.and.(header(4)/=-1.or.header(5)/=0.or.header(8)/=real_to_word(0.e0_real_kind))) goto 900
      if(int(n,int64)*c*real_bytes>budget-size(words,kind=int64)*8) goto 900
      allocate(restored%groups(g)%indices(3,n),restored%groups(g)%keys(n),stat=status)
      if(status/=0) goto 900
      if(any(words(pos:pos+3*n-1)<0).or.any(words(pos:pos+3*n-1)>maxval(upper))) goto 900
      restored%groups(g)%indices=reshape(int(words(pos:pos+3*n-1)),[3,n]);pos=pos+3*n
      do row=1,n
        if(any(restored%groups(g)%indices(:,row)>upper)) goto 900
        restored%groups(g)%keys(row)=node_key(restored%groups(g)%indices(:,row),this%cells)
      enddo
      if(n>1) then
        if(any(restored%groups(g)%keys(2:)<=restored%groups(g)%keys(:n-1))) goto 900
      endif
      if(.not.present_state) cycle
      allocate(values(n,c),stat=status)
      if(status/=0) goto 900
      do m=1,c
        values(:,m)=word_to_real(words(pos:pos+n-1));pos=pos+n
      enddo
      if(this%on_device) then
#ifdef _CUDA
        call allocate_selected_moments_gpu(restored%groups(g)%device,restored%groups(g)%indices,c,valid)
        if(.not.valid) goto 900
        if(n>0) restored%groups(g)%device%states(:n,:)=values
#endif
      else
        call move_alloc(values,restored%groups(g)%states)
      endif
      if(allocated(values)) deallocate(values)
    enddo
    if(pos/=size(words)+1) goto 900
    call move_alloc(restored%groups,this%groups)
    this%coverage_counter=restored%coverage_counter
    ok=.true.
900 continue
    call restored%release()
  end subroutine

  subroutine configure_selected_statistics(this,window,intervals,origin,cells,periodic,on_device,ok)
    class(selected_statistics),intent(inout) :: this
    real(real_kind),intent(in) :: window(2)
    integer,intent(in) :: intervals(3),origin(3),cells(3)
    logical,intent(in) :: periodic(3),on_device
    logical,intent(out) :: ok
    ok=.false.
    if(.not.all(ieee_is_finite(window)).or.window(2)<=window(1)) return
    if(any(intervals<1).or.any(cells<1).or.any(origin<0).or.any(origin+cells>intervals)) return
#ifndef _CUDA
    if(on_device) return
#endif
    call this%release()
    this%window=window;this%intervals=intervals;this%origin=origin;this%cells=cells
    this%periodic=periodic;this%on_device=on_device;this%configured=.true.
    allocate(this%requests(0),this%groups(0));ok=.true.
  end subroutine

  subroutine selected_request_fields(definition,ids,covariance,ok)
    type(catalog_product),intent(in) :: definition
    integer,allocatable,intent(out) :: ids(:)
    logical,intent(out) :: covariance,ok
    logical :: selected(6)
    integer :: field,m,n
    selected=.false.;covariance=.false.;ok=.false.
    allocate(ids(0))
    if(definition%source/='reynolds'.and.definition%source/='favre') return
    if(definition%statistics_source/='selected-time') return
    if(definition%view%kind=='wall_profile'.or.catalog_wall_mean(definition)) then
      if(definition%source/='reynolds') return
      if(any(definition%fields/=1.and.definition%fields/=6.and. &
        (definition%fields<31.or.definition%fields>34))) return
      if(definition%wall_reference%flags(1)==1) then
        ids=[6,31,32,33,34]
      else
        ids=pack(definition%fields,definition%fields/=1)
      endif
      ok=.true.;return
    endif
    if(definition%view%kind/='slice'.and.definition%view%kind/='probe'.and. &
       definition%view%kind/='station_profile') return
    if(definition%view%kind=='station_profile'.and.any(definition%fields>6)) return
    do m=1,size(definition%fields)
      field=definition%fields(m)
      select case(field)
      case(1:6)
        selected(field)=.true.
      case(13:21)
        selected(2+mod(field-13,3))=.true.
      case(22:23)
        selected(2:4)=.true.
      case(24)
        selected(3:4)=.true.
      case(25)
        selected([2,4])=.true.
      case(26)
        selected(2:3)=.true.
      case(27)
        selected(1)=.true.
      case(28:30)
        selected(6)=.true.
      case default
        return
      end select
    enddo
    ! rho_bar is not the Favre mean of rho; the latter needs an additional moment.
    if(definition%source=='favre'.and.selected(1)) return
    if(definition%view%kind=='station_profile'.and.any(selected(2:4))) then
      selected(2:4)=.true.;covariance=.true.
    endif
    deallocate(ids);allocate(ids(count(selected(2:6))))
    n=0
    do m=2,6
      if(.not.selected(m)) cycle
      n=n+1;ids(n)=m
    enddo
    ok=.true.
  end subroutine

  subroutine build_selected_support(definition,intervals,origin,cells,periodic,indices,keys,bounds,ok)
    type(catalog_product),intent(in) :: definition
    integer,intent(in) :: intervals(3),origin(3),cells(3)
    logical,intent(in) :: periodic(3)
    integer,allocatable,intent(out) :: indices(:,:)
    integer(int64),allocatable,intent(out) :: keys(:)
    integer,intent(out) :: bounds(2,3)
    logical,intent(out) :: ok
    integer,allocatable :: axis_nodes(:,:)
    integer :: upper(3),counts(3),d,g,i,j,k,n,local(3),status
    integer(int64) :: nodes
    logical :: derived,included
    ok=.false.;bounds=int(definition%ranges)
    if(definition%axis>0) bounds(:,definition%axis)=int(definition%index)
    if(any(bounds(1,:)<0).or.any(bounds(2,:)>intervals).or.any(bounds(2,:)<bounds(1,:))) return
    derived=any(definition%fields>=13).and.definition%view%kind/='wall_profile'.and..not.catalog_wall_mean(definition)
    if(derived.and.any(intervals<6)) return
    if(derived) then
      do d=1,3
        bounds(1,d)=slice_stencil_start(bounds(1,d),intervals(d),periodic(d))
        bounds(2,d)=slice_stencil_start(bounds(2,d),intervals(d),periodic(d))+6
      enddo
    endif
    upper=origin+cells-1
    where(origin+cells==intervals.and..not.periodic) upper=upper+1
    allocate(axis_nodes(maxval(cells)+1,3));counts=0
    do d=1,3
      do g=origin(d),upper(d)
        included=g>=bounds(1,d).and.g<=bounds(2,d)
        if(periodic(d)) included=modulo(g-bounds(1,d),intervals(d))<=bounds(2,d)-bounds(1,d)
        if(.not.included) cycle
        counts(d)=counts(d)+1;axis_nodes(counts(d),d)=g-origin(d)
      enddo
    enddo
    nodes=product(int(counts,int64))
    if(nodes>huge(n)) return
    allocate(indices(3,int(nodes)),keys(int(nodes)),stat=status)
    if(status/=0) return
    n=0
    do k=1,counts(3)
    do j=1,counts(2)
    do i=1,counts(1)
      n=n+1;local=[axis_nodes(i,1),axis_nodes(j,2),axis_nodes(k,3)]
      indices(:,n)=local
      keys(n)=node_key(local,cells)
    enddo
    enddo
    enddo
    ok=.true.
  end subroutine

  pure integer(int64) function node_key(local,cells) result(key)
    integer,intent(in) :: local(3),cells(3)
    key=int(local(1),int64)+int(cells(1)+1,int64)*(int(local(2),int64)+int(cells(2)+1,int64)*local(3))
  end function

  subroutine reconcile_selected_statistics(this,definitions,active,ok,approve,agree)
    class(selected_statistics),intent(inout) :: this
    type(catalog_product),intent(in) :: definitions(:)
    logical,intent(in) :: active(:)
    logical,intent(out) :: ok
    procedure(selected_allocation_check),optional :: approve
    procedure(selected_validity_check),optional :: agree
    type(selected_statistics_group),allocatable :: expanded(:)
    type(selected_statistics_request),allocatable :: requests(:)
    integer,allocatable :: ids(:),indices(:,:)
    integer(int64),allocatable :: keys(:)
    integer :: p,g,matched,bounds(2,3),n,c,status,old
    integer(int64) :: host_bytes,device_bytes,extra_host,extra_device
    logical :: covariance,valid
    ok=.false.
    if(.not.this%configured.or.size(active)/=size(definitions)) return
    allocate(requests(size(definitions)))
    do g=1,size(this%groups)
      this%groups(g)%active=.false.
    enddo
    do p=1,size(definitions)
      requests(p)%definition=definitions(p)
      if(.not.active(p)) cycle
      call selected_request_fields(definitions(p),ids,covariance,valid)
      if(.not.valid) cycle
      call build_selected_support(definitions(p),this%intervals,this%origin,this%cells,this%periodic, &
        indices,keys,bounds,valid)
      if(.not.valid) return
      matched=0
      do g=1,size(this%groups)
        if(this%groups(g)%failed.or.this%groups(g)%nfields/=size(ids)) cycle
        if(this%groups(g)%covariance.neqv.covariance) cycle
        if(any(this%groups(g)%bounds/=bounds)) cycle
        if(any(this%groups(g)%field_ids(:size(ids))/=ids)) cycle
        matched=g;exit
      enddo
      if(matched==0) then
        n=size(keys);c=sample_moment_size(size(ids),covariance)
        if(present(approve)) then
          call this%bytes(host_bytes,device_bytes)
          extra_host=40+int(n,int64)*(3*storage_size(0)/8+8);extra_device=0
          if(this%on_device) then
            extra_device=int(max(1,n),int64)*(3*storage_size(0)/8+int(c,int64)*real_bytes)+storage_size(0)/8
          else
            extra_host=extra_host+int(n,int64)*c*real_bytes
          endif
          call approve(host_bytes+extra_host,device_bytes+extra_device,extra_device,valid)
          if(.not.valid) then
            requests(p)%resource_failed=.true.;cycle
          endif
        endif
        g=size(this%groups)+1
        allocate(expanded(g))
        do old=1,g-1
          call move_selected_group(this%groups(old),expanded(old))
        enddo
        call move_alloc(expanded,this%groups)
        this%groups(g)%bounds=bounds;this%groups(g)%nfields=size(ids)
        this%groups(g)%field_ids(:size(ids))=ids;this%groups(g)%covariance=covariance
        if(covariance) this%groups(g)%velocity_slots=[1,2,3]
        call move_alloc(indices,this%groups(g)%indices);call move_alloc(keys,this%groups(g)%keys)
        if(this%on_device) then
#ifdef _CUDA
          call allocate_selected_moments_gpu(this%groups(g)%device,this%groups(g)%indices,c,valid)
#endif
        else
          allocate(this%groups(g)%states(n,c),stat=status)
          valid=status==0
          if(valid) this%groups(g)%states=0.e0_real_kind
        endif
        if(present(agree)) valid=agree(valid)
        if(.not.valid) then
          this%groups(g)%failed=.true.;requests(p)%resource_failed=.true.;cycle
        endif
        matched=g
      endif
      requests(p)%active=.true.;requests(p)%group=matched;this%groups(matched)%active=.true.
    enddo
    do g=1,size(this%groups)
      if(.not.this%groups(g)%active) this%groups(g)%continuous=.false.
    enddo
    call move_alloc(requests,this%requests);ok=.true.
  end subroutine

  subroutine move_selected_group(source,destination)
    type(selected_statistics_group),intent(inout) :: source,destination
    destination%bounds=source%bounds;destination%field_ids=source%field_ids
    destination%nfields=source%nfields;destination%velocity_slots=source%velocity_slots
    destination%last_step=source%last_step;destination%segments=source%segments
    destination%last_time=source%last_time;destination%first_time=source%first_time;destination%covered=source%covered
    destination%coverage_id=source%coverage_id;destination%coverage_parent=source%coverage_parent
    destination%coverage_step=source%coverage_step;destination%coverage_weights=source%coverage_weights
    destination%active=source%active;destination%previous=source%previous;destination%continuous=source%continuous
    destination%covariance=source%covariance;destination%failed=source%failed
    call move_alloc(source%indices,destination%indices);call move_alloc(source%keys,destination%keys)
    call move_alloc(source%states,destination%states)
#ifdef _CUDA
    destination%device%states=>source%device%states;destination%device%indices=>source%device%indices
    destination%device%invalid=>source%device%invalid
    destination%device%profile_weights=>source%device%profile_weights
    destination%device%profile_sums=>source%device%profile_sums
    destination%device%external_positions=>source%device%external_positions
    nullify(source%device%states,source%device%indices,source%device%invalid, &
      source%device%profile_weights,source%device%profile_sums,source%device%external_positions)
#endif
  end subroutine

  subroutine sample_weights(this,g,step,time,weights,seed,repeat,ok)
    class(selected_statistics),intent(in) :: this
    integer,intent(in) :: g
    integer(int64),intent(in) :: step
    real(real_kind),intent(in) :: time
    real(real_kind),intent(out) :: weights(2)
    logical,intent(out) :: seed,repeat,ok
    real(real_kind) :: duration
    weights=0;seed=.false.;repeat=.false.;ok=.false.
    if(.not.ieee_is_finite(time).or.step<0) return
    if(this%groups(g)%previous) then
      if(step==this%groups(g)%last_step.and.time==this%groups(g)%last_time) then
        repeat=.true.;ok=.true.;return
      endif
      if(step<=this%groups(g)%last_step.or.time<=this%groups(g)%last_time) return
    endif
    seed=.not.this%groups(g)%previous.or..not.this%groups(g)%continuous
    ! A missing completed step is a sampling gap, not a long trapezoid interval.
    if(this%groups(g)%previous) seed=seed.or.step/=this%groups(g)%last_step+1
    if(seed) then
      ok=.true.;return
    endif
    call clipped_trapezoid(this%groups(g)%last_time,time,[1.e0_real_kind,0.e0_real_kind], &
      [0.e0_real_kind,1.e0_real_kind],this%window(1),this%window(2),weights,duration,ok)
  end subroutine

  subroutine accept_sample(this,g,step,time,weights,seed)
    class(selected_statistics),intent(inout) :: this
    integer,intent(in) :: g
    integer(int64),intent(in) :: step
    real(real_kind),intent(in) :: time,weights(2)
    logical,intent(in) :: seed
    integer :: other
    logical :: matched
    if(sum(weights)>0) then
      ! Intern exact transitions only within this completed step; no hash or history log.
      this%groups(g)%coverage_parent=this%groups(g)%coverage_id;matched=.false.
      do other=1,size(this%groups)
        if(other==g.or.this%groups(other)%coverage_step/=step) cycle
        if(this%groups(other)%last_time/=time) cycle
        if(this%groups(other)%coverage_parent/=this%groups(g)%coverage_parent) cycle
        if(any(real_to_word(this%groups(other)%coverage_weights)/=real_to_word(weights))) cycle
        this%groups(g)%coverage_id=this%groups(other)%coverage_id;matched=.true.;exit
      enddo
      if(.not.matched) then
        this%coverage_counter=this%coverage_counter+1
        this%groups(g)%coverage_id=this%coverage_counter
      endif
      this%groups(g)%coverage_step=step;this%groups(g)%coverage_weights=weights
    endif
    if(.not.this%groups(g)%previous) this%groups(g)%first_time=time
    if(seed) this%groups(g)%segments=this%groups(g)%segments+1
    this%groups(g)%covered=this%groups(g)%covered+sum(weights)
    this%groups(g)%last_step=step;this%groups(g)%last_time=time
    this%groups(g)%continuous=.true.;this%groups(g)%previous=.true.
  end subroutine

  logical function matching_selected_coverage(this,left,right) result(matches)
    class(selected_statistics),intent(in) :: this
    integer,intent(in) :: left,right
    matches=.false.
    if(.not.allocated(this%groups)) return
    if(min(left,right)<1.or.max(left,right)>size(this%groups)) return
    if(this%groups(left)%failed.or.this%groups(right)%failed) return
    if(this%groups(left)%covered<=0.or.this%groups(right)%covered<=0) return
    matches=this%groups(left)%coverage_id==this%groups(right)%coverage_id.and. &
      this%groups(left)%covered==this%groups(right)%covered
  end function

  subroutine sample_selected_statistics_host(this,step,time,hm,q,const2,const6,ok)
    class(selected_statistics),intent(inout) :: this
    integer,intent(in) :: hm
    integer(int64),intent(in) :: step
    real(real_kind),intent(in) :: time,q(-hm:,-hm:,-hm:,:),const2,const6
    logical,intent(out) :: ok
    real(real_kind) :: weights(2),fields(5),density,local_q(5)
    integer :: g,n,nf,local(3)
    logical :: seed,repeat,valid,all_valid,success
    ok=.false.
    if(.not.this%configured.or.this%on_device.or.size(q,4)/=5) return
    success=.true.
    do g=1,size(this%groups)
      if(.not.this%groups(g)%active.or.this%groups(g)%failed) cycle
      if(any(this%groups(g)%field_ids>6)) cycle
      call sample_weights(this,g,step,time,weights,seed,repeat,valid)
      if(.not.valid) then
        this%groups(g)%failed=.true.;success=.false.;cycle
      endif
      if(repeat) cycle
      nf=this%groups(g)%nfields;all_valid=.true.
      do n=1,size(this%groups(g)%keys)
        local=this%groups(g)%indices(:,n)
        local_q=q(local(1),local(2),local(3),:)
        call decode_sample_primitives(local_q,nf,this%groups(g)%field_ids(:nf), &
          const2,const6,fields(:nf),density,valid)
        if(valid) call push_sample_moments(this%groups(g)%states(n,:),fields(:nf),density,weights, &
          this%groups(g)%previous,this%groups(g)%velocity_slots,valid)
        all_valid=all_valid.and.valid
      enddo
      if(.not.all_valid) then
        this%groups(g)%failed=.true.;this%groups(g)%continuous=.false.;success=.false.;cycle
      endif
      call accept_sample(this,g,step,time,weights,seed)
    enddo
    ok=success
  end subroutine

#ifdef _CUDA
  subroutine sample_selected_statistics_device(this,step,time,hm,q,const2,const6,ok)
    use cudafor
    class(selected_statistics),intent(inout) :: this
    integer,intent(in) :: hm
    integer(int64),intent(in) :: step
    real(real_kind),intent(in) :: time,const2,const6
    real(real_kind),device,intent(in) :: q(-hm:,-hm:,-hm:,:)
    logical,intent(out) :: ok
    real(real_kind) :: weights(2)
    integer :: g,nf
    logical :: seed,repeat,valid,success
    ok=.false.
    if(.not.this%configured.or..not.this%on_device.or.size(q,4)/=5) return
    success=.true.
    do g=1,size(this%groups)
      if(.not.this%groups(g)%active.or.this%groups(g)%failed) cycle
      if(any(this%groups(g)%field_ids>6)) cycle
      call sample_weights(this,g,step,time,weights,seed,repeat,valid)
      if(.not.valid) then
        this%groups(g)%failed=.true.;success=.false.;cycle
      endif
      if(repeat) cycle
      nf=this%groups(g)%nfields
      call gather_selected_moments_gpu(q,this%cells(1),this%cells(2),this%cells(3),hm, &
        this%groups(g)%device%indices,this%groups(g)%field_ids(:nf),const2,const6,this%groups(g)%device%states, &
        weights,this%groups(g)%previous,this%groups(g)%covariance,this%groups(g)%device%invalid,valid, &
        sample_count=size(this%groups(g)%keys))
      if(.not.valid) then
        this%groups(g)%failed=.true.;this%groups(g)%continuous=.false.;success=.false.;cycle
      endif
      call accept_sample(this,g,step,time,weights,seed)
    enddo
    ok=success
  end subroutine
#endif

  ! The clock advances once even if a sampler assembles several periodic pieces.
  subroutine sample_selected_statistics_external(this,step,time,sampler,ok)
    class(selected_statistics),intent(inout) :: this
    integer(int64),intent(in) :: step
    real(real_kind),intent(in) :: time
    procedure(selected_external_sampler) :: sampler
    logical,intent(out) :: ok
    real(real_kind) :: weights(2)
    integer :: g
    logical :: seed,repeat,valid
    ok=.true.
    do g=1,size(this%groups)
      if(.not.this%groups(g)%active.or.this%groups(g)%failed) cycle
      if(.not.any(this%groups(g)%field_ids>6)) cycle
      call sample_weights(this,g,step,time,weights,seed,repeat,valid)
      if(valid.and..not.repeat) call sampler(this%groups(g),weights,this%groups(g)%previous,valid)
      if(.not.valid) then
        this%groups(g)%failed=.true.;this%groups(g)%continuous=.false.;ok=.false.;cycle
      endif
      if(.not.repeat) call accept_sample(this,g,step,time,weights,seed)
    enddo
  end subroutine

  integer function find_selected_statistics(this,id) result(group)
    class(selected_statistics),intent(in) :: this
    character(*),intent(in) :: id
    integer :: p
    group=0
    if(.not.allocated(this%requests)) return
    do p=1,size(this%requests)
      if(.not.this%requests(p)%active) cycle
      if(this%requests(p)%definition%id/=id) cycle
      group=this%requests(p)%group;return
    enddo
  end function

  subroutine selected_statistics_value(this,group,local,field,favre,value,ok)
    class(selected_statistics),intent(in) :: this
    integer,intent(in) :: group,local(3),field
    logical,intent(in) :: favre
    real(real_kind),intent(out) :: value
    logical,intent(out) :: ok
    integer :: lo,hi,mid,nf,c
    integer(int64) :: key
    ok=.false.;value=0
    if(this%on_device.or.group<1.or.group>size(this%groups)) return
    if(this%groups(group)%failed.or.this%groups(group)%covered<=0) return
    key=node_key(local,this%cells);lo=1;hi=size(this%groups(group)%keys)
    do while(lo<=hi)
      mid=lo+(hi-lo)/2
      if(this%groups(group)%keys(mid)<key) then
        lo=mid+1
      elseif(this%groups(group)%keys(mid)>key) then
        hi=mid-1
      else
        nf=this%groups(group)%nfields
        if(field==1.and..not.favre) then
          value=this%groups(group)%states(mid,3)/this%groups(group)%states(mid,2);ok=.true.;return
        endif
        do c=1,nf
          if(this%groups(group)%field_ids(c)/=field) cycle
          value=this%groups(group)%states(mid,3+nf*merge(2,1,favre)+c);ok=.true.;return
        enddo
        return
      endif
    enddo
  end subroutine

  subroutine selected_statistics_bytes(this,host_bytes,device_bytes)
    class(selected_statistics),intent(in) :: this
    integer(int64),intent(out) :: host_bytes,device_bytes
    integer :: g,n
    host_bytes=8;device_bytes=0
    if(.not.allocated(this%groups)) return
    do g=1,size(this%groups)
      if(.not.allocated(this%groups(g)%keys)) cycle
      n=size(this%groups(g)%keys)
      host_bytes=host_bytes+40+int(n,int64)*(3*storage_size(0)/8+8)
      if(.not.group_has_state(this,g)) cycle
      if(this%on_device) then
#ifdef _CUDA
        device_bytes=device_bytes+size(this%groups(g)%device%indices,kind=int64)*storage_size(0)/8+ &
          size(this%groups(g)%device%states,kind=int64)*real_bytes+storage_size(0)/8
        device_bytes=device_bytes+selected_profile_workspace_bytes_gpu(this%groups(g)%device)
        device_bytes=device_bytes+selected_external_workspace_bytes_gpu(this%groups(g)%device)
#endif
      else
        host_bytes=host_bytes+size(this%groups(g)%states,kind=int64)*real_bytes
      endif
    enddo
  end subroutine

  subroutine release_selected_statistics(this)
    class(selected_statistics),intent(inout) :: this
#ifdef _CUDA
    integer :: g
    if(allocated(this%groups)) then
      do g=1,size(this%groups)
        call release_selected_moments_gpu(this%groups(g)%device)
      enddo
    endif
#endif
    if(allocated(this%requests)) deallocate(this%requests)
    if(allocated(this%groups)) deallocate(this%groups)
    this%configured=.false.;this%coverage_counter=0
  end subroutine
end module
