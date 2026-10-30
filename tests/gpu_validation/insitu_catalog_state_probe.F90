program insitu_catalog_state_probe
  use iso_fortran_env, only: int64,real64
  use iso_c_binding, only: c_int,c_char
  use insitu_catalog, only: read_product_catalog
  use insitu_run_config, only: catalog_product,catalog_definition_equal,catalog_field_names
  use insitu_product_schedule
  use adaptive_output, only: adaptive_config,adaptive_shared_config,configure_adaptive,observe_adaptive
  use astr_precision, only: real_kind
  implicit none
  type(catalog_product),allocatable :: definitions(:),empty(:),expanded(:)
  type(runtime_product_state) :: state,saved
  type(adaptive_config) :: adaptive
  character(:),allocatable :: path,message
  character(64) :: scenario
  logical :: ok
  integer :: n,unit,i,j
  integer(c_int),allocatable :: field_ids(:)
  character(c_char),allocatable :: field_names(:)
  integer(c_int) :: fields,queried,dummy_id(1)
  character(c_char) :: dummy_name(1)
  interface
    integer(c_int) function native_fields(index,ids,names,capacity) bind(C,name='astr_insitu_runtime_product_fields')
      import c_int,c_char
      integer(c_int),value :: index,capacity
      integer(c_int),intent(out) :: ids(*)
      character(c_char),intent(out) :: names(*)
    end function
  end interface
  call get_command_argument(1,length=n)
  allocate(character(n)::path)
  call get_command_argument(1,path)
  call get_command_argument(2,scenario)
  if(scenario=='checkpoint-empty') then
    open(newunit=unit,status='scratch',access='stream',form='unformatted')
    call runtime_catalog_state_file(unit,.true.,0_int64,0._real64,ok)
    call require(ok,'cannot save valid empty control state')
    rewind(unit)
    call runtime_catalog_state_file(unit,.false.,0_int64,0._real64,ok)
    close(unit)
    call require(ok.and.runtime_catalog_count()==0,'empty control state did not restore')
    print '(a)','PASS: runtime catalog '//trim(scenario)
    stop
  endif
  call read_product_catalog(path,definitions,ok,message)
  call require(ok,'cannot parse initial definitions')
  call require(size(definitions)==1,'one initial definition required')
  if(scenario=='adaptive') then
    adaptive%enabled=.true.;adaptive%window_ids(1)='burst'
    adaptive%window_modes(1)='steps';adaptive%window_steps(:,1)=[12_int64,15_int64]
    call configure_adaptive(adaptive,0_int64,0._real_kind,ok)
    call require(ok,'cannot initialize shared adaptive window')
    definitions(1)%window_ids=[character(5)::'burst'];definitions(1)%dense_interval_steps=1
  endif
  call runtime_catalog_apply(definitions,10_int64,.01_real64)
  call runtime_catalog_state(1,state)
  call require(state%version==1,'first accepted definition must have version 1')
  call require(runtime_catalog_due(1,10_int64,.01_real64),'first frame must be due')
  call runtime_catalog_success(1,10_int64,.01_real64)
  select case(trim(scenario))
  case('typed-fields')
    fields=native_fields(1_c_int,dummy_id,dummy_name,0_c_int)
    call require(fields==size(definitions(1)%fields),'field query count differs')
    allocate(field_ids(fields),field_names(len(catalog_field_names)*fields))
    queried=native_fields(1_c_int,field_ids,field_names,fields)
    call require(queried==fields.and.all(field_ids==definitions(1)%fields),'typed field IDs differ')
    do i=1,fields
      do j=1,len(catalog_field_names)
        call require(field_names((i-1)*len(catalog_field_names)+j)==catalog_field_names(field_ids(i))(j:j), &
          'typed field name differs')
      enddo
    enddo
  case('visual-checkpoint')
    call runtime_catalog_state(1,saved)
    open(newunit=unit,status='scratch',access='stream',form='unformatted')
    call runtime_catalog_state_file(unit,.true.,10_int64,.01_real64,ok)
    call require(ok,'cannot save visual definition')
    if(definitions(1)%view%channel=='geometry') then
      definitions(1)%fields(1)=1
    else
      definitions(1)%view%camera(1)=definitions(1)%view%camera(1)+17
    endif
    call runtime_catalog_apply(definitions,11_int64,.011_real64)
    call runtime_catalog_state(1,state)
    call require(state%version==2,'display change must create a new definition version')
    rewind(unit)
    call runtime_catalog_state_file(unit,.false.,10_int64,.01_real64,ok)
    close(unit)
    call require(ok,'cannot restore visual definition')
    call runtime_catalog_state(1,state)
    call require(catalog_definition_equal(state%definition,saved%definition),'visual parameters lost in checkpoint')
    call require(state%version==1.and.state%last_step==10,'visual clock/version changed on restore')
    definitions(1)=saved%definition
    call runtime_catalog_apply(definitions,11_int64,.011_real64)
    call require(.not.runtime_catalog_due(1,11_int64,.011_real64),'unchanged restored view repeated its first frame')
  case('adaptive')
    call observe_adaptive(12_int64,.012_real_kind,ok)
    call require(ok.and.runtime_catalog_due(1,12_int64,.012_real64),'adaptive entry must emit')
    call runtime_catalog_success(1,12_int64,.012_real64)
    call observe_adaptive(15_int64,.015_real_kind,ok)
    call require(ok.and..not.runtime_catalog_due(1,15_int64,.015_real64),'normal cadence must anchor at dense output')
  case('first-frame')
    call require(.not.runtime_catalog_due(1,10_int64,.01_real64),'duplicate frame')
    call require(.not.runtime_catalog_due(1,109_int64,.109_real64),'early normal frame')
    call require(runtime_catalog_due(1,110_int64,.110_real64),'normal frame missing')
  case('cadence')
    definitions(1)%interval_steps=20
    call runtime_catalog_apply(definitions,50_int64,.05_real64)
    call runtime_catalog_state(1,state)
    call require(state%version==1.and.state%last_step==10,'cadence must preserve version and success')
    call require(runtime_catalog_due(1,50_int64,.05_real64),'overdue update must emit current frame')
    call runtime_catalog_success(1,50_int64,.05_real64)
    call require(.not.runtime_catalog_due(1,69_int64,.069_real64),'incorrect cadence anchor')
    call require(runtime_catalog_due(1,70_int64,.07_real64),'next cadence frame missing')
  case('definition')
    definitions(1)%index=21; definitions(1)%ranges(:,2)=21
    call runtime_catalog_apply(definitions,11_int64,.011_real64)
    call runtime_catalog_state(1,state)
    call require(state%version==2,'changed plane must create version 2')
    call require(runtime_catalog_due(1,11_int64,.011_real64),'changed definition first frame missing')
    call runtime_catalog_success(1,11_int64,.011_real64)
    call runtime_catalog_apply(definitions,12_int64,.012_real64)
    call require(.not.runtime_catalog_due(1,12_int64,.012_real64),'identical definition must not repeat first frame')
  case('invalid-recovery')
    definitions(1)%valid=.false.; definitions(1)%error='invalid field'
    call runtime_catalog_apply(definitions,11_int64,.011_real64)
    call runtime_catalog_state(1,state)
    call require(state%status==runtime_bad_config,'invalid update must suspend existing product')
    call require(.not.runtime_catalog_due(1,110_int64,.11_real64),'old definition must not keep producing')
    definitions(1)%valid=.true.; definitions(1)%error=''
    call runtime_catalog_apply(definitions,111_int64,.111_real64)
    call runtime_catalog_state(1,state)
    call require(state%version==1,'corrected unchanged definition must retain version')
    call require(runtime_catalog_due(1,111_int64,.111_real64),'valid correction must resume without request bump')
  case('remove-readd')
    allocate(empty(0))
    call runtime_catalog_apply(empty,11_int64,.011_real64)
    call runtime_catalog_state(1,state)
    call require(state%status==runtime_absent,'omitted product must stop')
    call require(.not.runtime_catalog_due(1,110_int64,.11_real64),'omitted product emitted')
    call runtime_catalog_apply(definitions,111_int64,.111_real64)
    call runtime_catalog_state(1,state)
    call require(state%version==1,'membership change must not roll version back')
    call require(runtime_catalog_due(1,111_int64,.111_real64),'re-added product needs current first frame')
  case('resource-retry')
    call runtime_catalog_disable(1,11_int64,.011_real64,'injected resource failure')
    definitions(1)%interval_steps=1
    call runtime_catalog_apply(definitions,12_int64,.012_real64)
    call require(.not.runtime_catalog_due(1,12_int64,.012_real64),'cadence must not retry resource failure')
    definitions(1)%enable_request=2
    call runtime_catalog_apply(definitions,13_int64,.013_real64)
    call require(runtime_catalog_due(1,13_int64,.013_real64),'new request must enable product')
    call runtime_catalog_disable(1,13_int64,.013_real64,'injected second failure')
    call runtime_catalog_apply(definitions,14_int64,.014_real64)
    call require(.not.runtime_catalog_due(1,14_int64,.014_real64),'consumed request must not repeat')
    definitions(1)%enable_request=1
    call runtime_catalog_apply(definitions,15_int64,.015_real64)
    call require(.not.runtime_catalog_due(1,15_int64,.015_real64),'older request must not enable product')
  case('waiting')
    call runtime_catalog_wait(1,.true.)
    call runtime_catalog_state(1,state)
    call require(state%status==runtime_waiting,'missing coverage must wait')
    call require(.not.runtime_catalog_due(1,110_int64,.11_real64),'waiting product must not emit')
    call runtime_catalog_wait(1,.false.)
    call require(runtime_catalog_due(1,110_int64,.11_real64),'ready data should permit due output')
  case('file-rejection')
    call read_product_catalog(path//'.missing',empty,ok,message)
    if(ok) call runtime_catalog_apply(empty,11_int64,.011_real64)
    call runtime_catalog_state(1,state)
    call require(state%status==runtime_active.and.state%last_step==10,'file rejection changed active state')
  case('checkpoint')
    call runtime_catalog_disable(1,10_int64,.01_real64,'injected first failure')
    definitions(1)%enable_request=2
    call runtime_catalog_apply(definitions,11_int64,.011_real64)
    call require(runtime_catalog_due(1,11_int64,.011_real64),'explicit retry missing')
    call runtime_catalog_success(1,11_int64,.011_real64)
    definitions(1)%index=21;definitions(1)%ranges(:,2)=21
    call runtime_catalog_apply(definitions,12_int64,.012_real64)
    call runtime_catalog_disable(1,12_int64,.012_real64,'injected checkpoint failure')
    allocate(expanded(2))
    expanded(1)=definitions(1)
    expanded(2)=definitions(1)
    call move_alloc(expanded,definitions)
    definitions(2)%id='other'
    call runtime_catalog_apply(definitions,13_int64,.013_real64)
    call runtime_catalog_state(1,state)
    call require(state%version==2.and.all(state%definition%fields==[2,27]),'adding a product changed existing fields')
    call runtime_catalog_success(2,13_int64,.013_real64)
    open(newunit=unit,file=path//'.state',status='replace',access='stream',form='unformatted')
    call runtime_catalog_state_file(unit,.true.,15_int64,.015_real64,ok)
    call require(ok,'cannot save catalog control state')
    definitions(1)%index=22;definitions(1)%ranges(:,2)=22;definitions(1)%enable_request=3
    call runtime_catalog_apply(definitions,16_int64,.016_real64)
    rewind(unit)
    call runtime_catalog_state_file(unit,.false.,15_int64,.015_real64,ok)
    close(unit)
    call require(ok,'cannot restore catalog control state')
    call runtime_catalog_state(1,state)
    call require(state%version==2.and.state%consumed_request==2,'version/request lost on restore')
    call require(state%status==runtime_resource_disabled.and.state%pending_first,'disabled first frame lost')
    call require(state%definition%index==21.and.state%attempt_step==12,'definition/attempt history lost')
    definitions(1)%index=21;definitions(1)%ranges(:,2)=21;definitions(1)%enable_request=2
    call runtime_catalog_apply(definitions,16_int64,.016_real64)
    call require(.not.runtime_catalog_due(1,16_int64,.016_real64),'restored consumed request was replayed')
    call require(.not.runtime_catalog_due(2,112_int64,.112_real64),'restored active clock ran early')
    call require(runtime_catalog_due(2,113_int64,.113_real64),'restored active clock lost its anchor')
  case('checkpoint-bad-format')
    open(newunit=unit,status='scratch',access='stream',form='unformatted')
    write(unit) 'ASTRRC00',0_int64
    rewind(unit)
    call runtime_catalog_state_file(unit,.false.,15_int64,.015_real64,ok)
    close(unit)
    call require(.not.ok,'old control format was accepted')
    call runtime_catalog_state(1,state)
    call require(state%version==1.and.state%last_step==10,'rejected control replaced effective state')
  case default
    error stop 'unknown scenario'
  end select
  print '(a)','PASS: runtime catalog '//trim(scenario)
contains
  subroutine require(condition,message)
    logical,intent(in) :: condition
    character(*),intent(in) :: message
    if(condition) return
    print '(a)','FAIL: '//message
    stop 1
  end subroutine
end program
