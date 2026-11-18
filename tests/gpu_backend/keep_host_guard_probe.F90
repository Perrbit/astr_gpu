program keep_host_guard_probe
  use keep_operator, only: keep_split_plan,build_keep_split_plan
  implicit none
  type :: guarded_plan
    integer :: before(8)
    type(keep_split_plan) :: plan
    integer :: after(8)
  end type
  type(guarded_plan) :: sample
  integer :: i,unit
  character(1024) :: path
  sample%before=[(100+i,i=1,8)];sample%after=[(200+i,i=1,8)]
  sample%plan%terms=-1;sample%plan%bases=-1
  sample%plan%left=-1;sample%plan%right=-1
  sample%plan%component=-1;sample%plan%derivative=-1
  sample%plan%weight=-1
  call build_keep_split_plan(sample%plan)
  write(*,'(a,8(1x,i0))') 'KEEP_HOST_GUARD_BEFORE',sample%before
  write(*,'(a,8(1x,i0))') 'KEEP_HOST_GUARD_AFTER',sample%after
  write(*,'(a,2(1x,i0))') 'KEEP_HOST_PLAN',sample%plan%terms,sample%plan%bases
  call get_command_argument(1,path)
  if(len_trim(path)>0) then
    open(newunit=unit,file=trim(path),status='new',access='stream',form='unformatted')
    write(unit) sample%plan%terms,sample%plan%bases,sample%plan%left,sample%plan%right, &
      sample%plan%component,sample%plan%derivative,sample%plan%weight
    close(unit)
  endif
  if(any(sample%before/=[(100+i,i=1,8)]).or.any(sample%after/=[(200+i,i=1,8)])) &
    error stop 'KEEP host plan initialization wrote outside its object'
  if(any(sample%plan%right(:,sample%plan%bases+1:)/=0)) error stop 'KEEP unused basis storage was not cleared'
  print '(a)', 'KEEP_HOST_GUARD_PASS'
end program
