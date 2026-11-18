program air5_coupling_probe
  use iso_c_binding
  use device_state
  use chemistry_device_properties
  use chemistry_device_coupling
  implicit none
  type(c_ptr) :: handle
  type(air5_device_options) :: options
  type(air5_device_report) :: report
  real(c_double) :: input(18),duration,interval
  real(c_double) :: q(-2:2,-2:2,-2:2,11),carry(0:0,0:0,0:0,11),history(0:0,0:0,0:0)
  integer(c_int) :: first(3),last(3),chem_status
  integer :: n,mode,segments,unit,step,counts(4),status
  character(len=4096) :: text,input_path,output_path
  call get_command_argument(1,text)
  if(trim(text)/='batch') error stop 'expected batch input/output rtol atol segments history'
  call get_command_argument(2,input_path)
  call get_command_argument(3,output_path)
  open(newunit=unit,file=trim(input_path),status='old')
  read(unit,*) n,duration,mode
  if(n/=1) error stop 'this G1 consumer observes one physical cell per context'
  input=0.0_c_double
  call get_command_argument(7,text)
  options%history_enabled=merge(0,1,trim(text)=='reset')
  if(trim(text)=='history-from-input') then
    read(unit,*) input
  else
    read(unit,*) input(1:17)
  endif
  close(unit)
  call get_command_argument(4,text);read(text,*) options%rtol
  call get_command_argument(5,text);read(text,*) options%atol_factor
  call get_command_argument(6,text);read(text,*) segments
  options%source_mode=mode;options%compensated=1;options%preserve_all=1
  options%block_size=32;options%specialized_dispatch=0
  options%source_only=0
  call get_environment_variable('ASTR_AIR5_SOURCE_EVALUATION',text,status=status)
  if(status==0) then
    select case(trim(text))
    case('joint');options%source_only=0
    case('source-only');options%source_only=1
    case default;error stop 'invalid source evaluation'
    end select
  endif
  options%activity_threshold=0.0_c_double
  call get_environment_variable('ASTR_AIR5_LOW_ACTIVITY',text,status=status)
  if(status==0.and.trim(text)/='off') read(text,*) options%activity_threshold
  call check(device_state_create(0,0,0,0,2,1,handle))
  call check(air5_device_prepare(handle))
  call check(air5_chemistry_prepare(handle,options))
  q=-321.0_c_double;q(0,0,0,:)=input(1:11)
  carry=0.0_c_double;carry(0,0,0,6:11)=input(12:17);history=input(18)
  call check(device_state_upload(handle,device_q,q,int(size(q),c_size_t)*8_c_size_t))
  call check(device_state_upload(handle,device_air5_carry,carry,int(size(carry),c_size_t)*8_c_size_t))
  if(options%history_enabled/=0) &
    call check(device_state_upload(handle,device_air5_history,history,8_c_size_t))
  first=0;last=0;counts=0
  interval=duration/segments
  do step=1,segments
    call check(air5_chemistry_begin(handle))
    call check(air5_chemistry_region(handle,first,last,interval,1))
    call check(air5_chemistry_status(handle,chem_status))
    call check(air5_chemistry_report(handle,report))
    if(report%status/=chem_status.or.report%total/=1) error stop 'chemistry scalar/report region accounting'
    counts=counts+report%max_counts
    if(chem_status/=0) exit
  enddo
  call check(device_state_download(handle,device_q,q,int(size(q),c_size_t)*8_c_size_t))
  call check(device_state_download(handle,device_air5_carry,carry,int(size(carry),c_size_t)*8_c_size_t))
  if(options%history_enabled/=0) call check(device_state_download(handle,device_air5_history,history,8_c_size_t))
  open(newunit=unit,file=trim(output_path),status='new')
  write(unit,'(A,I0,A,I0)') '# AIR5_CONTEXT_ABI options_bytes=',c_sizeof(options),' report_bytes=',c_sizeof(report)
  write(unit,'(6(I0,1X),18(ES25.17,1X))') 1,chem_status,counts,q(0,0,0,:),carry(0,0,0,6:11),history
  close(unit)
  call check(device_state_destroy(handle))
  if(chem_status/=0) error stop 'context chemistry failed; actual failure was recorded'
contains
  subroutine check(code)
    integer(c_int),intent(in) :: code
    if(code/=0) then
      write(*,*) 'C ABI device operation failed',code
      error stop
    endif
  end subroutine
end program
