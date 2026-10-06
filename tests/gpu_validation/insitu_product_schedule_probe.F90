program insitu_product_schedule_probe
  use insitu_run_config
  use insitu_product_schedule
  use iso_fortran_env, only: int64,real64
  use iso_c_binding, only: c_int,c_char,c_null_char
  implicit none
  type(insitu_options) :: options
  character(1024) :: file,mode,state_file,message
  logical :: ok,emit,same
  integer :: step,i,unit,status,start_step
  interface
    integer(c_int) function report(id,code,phase) bind(C,name='astr_insitu_product_report')
      import c_int,c_char
      character(c_char),intent(in) :: id(*)
      integer(c_int),value :: code,phase
    end function
  end interface
  call get_command_argument(1,file)
  call get_command_argument(2,mode)
  call get_command_argument(3,state_file)
  call read_insitu_options(trim(file),options,ok,message)
  if(.not.ok) then
    print *,trim(message)
    stop 1
  endif
  call configure_product_schedules(options,0_int64,0.d0,ok)
  call require(ok,'configure')
  start_step=0
  if(mode=='resumed') then
    open(newunit=unit,file=trim(state_file),access='stream',form='unformatted',status='old')
    call product_state_file(unit,.false.,5_int64,0.005d0,same,ok)
    close(unit)
    call require(ok.and.same,'restore')
    start_step=6
  endif
  do step=start_step,12
    call poll_product_schedules(int(step,int64),real(step,real64)*0.001d0,step==12,emit,ok)
    call require(ok,'poll')
    do i=1,options%product_count
      if(.not.product_due(options%product_ids(i))) cycle
      print '(i0,1x,a)',step,trim(options%product_ids(i))
      status=0
      if(step==4.and.index(options%product_ids(i),'.image')>0) status=28
      status=report(trim(options%product_ids(i))//c_null_char,int(status,c_int), &
        merge(1_c_int,0_c_int,status>0))
      call require(status==0,'report')
    enddo
    call require(product_results_complete(),'publication results')
    if(step==5.and.mode=='continuous') then
      open(newunit=unit,file=trim(state_file),access='stream',form='unformatted',status='replace')
      call product_state_file(unit,.true.,5_int64,0.005d0,same,ok)
      close(unit)
      call require(ok,'save at split')
    endif
    if(step<12) then
      call poll_product_schedules(int(step,int64),real(step,real64)*0.001d0,.false.,emit,ok)
      call require(ok.and..not.emit,'complete-step deduplication')
    endif
  enddo
  open(newunit=unit,file=trim(state_file)//'.'//trim(mode),access='stream',form='unformatted',status='replace')
  call product_state_file(unit,.true.,12_int64,0.012d0,same,ok)
  close(unit)
  call require(ok,'save final')
contains
  subroutine require(condition,context)
    logical,intent(in) :: condition
    character(*),intent(in) :: context
    if(.not.condition) then
      print *, 'FAIL ',context
      stop 1
    endif
  end subroutine
end program
