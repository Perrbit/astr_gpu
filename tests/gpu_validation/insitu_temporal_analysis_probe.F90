program insitu_temporal_analysis_probe
  use iso_fortran_env, only: int64
  use astr_precision, only: real_kind
  use insitu_temporal_analysis
  implicit none
  type(welch_state) :: state,restored
  real(real_kind),allocatable :: values(:),frequency(:),psd(:,:)
  real(real_kind) :: spacing
  integer(int64) :: segments,generation,bytes
  integer :: length,components,n,split,i,c,status,unit,filled,continuous,uniform
  logical :: ok,available,equal_time
  character(2048) :: path
  character(32) :: scenario
  call get_command_argument(1,path)
  call get_command_argument(2,scenario)
  if(scenario=='blocks') then
    call run_blocks(trim(path))
    stop
  endif
  open(newunit=unit,file=trim(path),status='old',action='read')
  read(unit,*) length,components,n,split
  call configure_welch(state,length,components,ok)
  call require(ok,'invalid Welch configuration')
  allocate(values(components))
  do i=1,n
    read(unit,*,iostat=status) spacing,continuous,uniform,values
    call require(status==0,'missing sample')
    call push_welch(state,values,spacing,continuous==1,uniform==1,ok)
    call require(ok,'Welch sample failed')
    if(i==split) then
      block
        integer :: checkpoint
        open(newunit=checkpoint,status='scratch',form='unformatted',access='stream')
        call welch_state_file(checkpoint,.true.,state,ok)
        call require(ok,'cannot save Welch overlap/cumulative state')
        rewind(checkpoint)
        call welch_state_file(checkpoint,.false.,restored,ok)
        call require(ok,'cannot restore Welch overlap/cumulative state')
        close(checkpoint)
        state=restored
      end block
    endif
  enddo
  close(unit)
  call welch_info(state,segments,filled,generation,spacing,equal_time,bytes)
  write(*,'(a,4(i0,1x),es26.17e3,1x,i0)') 'META ',segments,filled,generation,merge(1,0,equal_time),spacing,bytes
  call read_welch(state,frequency,psd,available)
  if(available) then
    do i=1,size(frequency)
      write(*,'(*(es26.17e3,1x))') frequency(i),(psd(i,c),c=1,components)
    enddo
  endif
contains
  subroutine run_blocks(path)
    character(*),intent(in) :: path
    type(time_block) :: block,restored
    real(real_kind),allocatable :: coordinates(:,:),sample(:,:)
    integer(int64),allocatable :: words(:),again(:)
    integer(int64) :: step
    real(real_kind) :: time
    integer :: unit,capacity,nodes,fields,n,split,i,j,status,continuous
    logical :: ok
    open(newunit=unit,file=path,status='old',action='read')
    read(unit,*) capacity,nodes,fields,n,split
    allocate(coordinates(3,nodes),sample(nodes,fields))
    read(unit,*) coordinates
    call configure_time_block(block,coordinates,fields,capacity,ok)
    call require(ok,'cannot configure time block')
    do j=1,nodes
      write(*,'(a,i0,3(1x,es26.17e3))') 'COORD ',j,coordinates(:,j)
    enddo
    do i=1,n
      read(unit,*,iostat=status) step,time,continuous,sample
      call require(status==0,'missing line sample')
      call push_time_block(block,step,time,sample,continuous==1,ok)
      call require(ok,'time-block sample failed')
      if(i==split) then
        call pack_time_block(block,words,ok)
        call require(ok,'cannot pack partial time block')
        call unpack_time_block(words,restored,ok)
        call require(ok,'cannot restore partial time block')
        call pack_time_block(restored,again,ok)
        call require(ok.and.size(words)==size(again),'time-block layout changed')
        call require(all(words==again),'time-block words changed on restore')
        block=restored;deallocate(words,again)
      endif
      if(block%filled==block%capacity) then
        call emit_block(block)
        call clear_time_block(block)
      endif
    enddo
    close(unit)
    if(block%filled>0) call emit_block(block)
    write(*,'(a,i0)') 'SAMPLES ',block%samples
  end subroutine

  subroutine emit_block(block)
    type(time_block),intent(in) :: block
    integer :: i,j,c
    write(*,'(a,2(i0,1x))') 'BLOCK ',block%number,block%filled
    do i=1,block%filled
      write(*,'(a,i0,1x,es26.17e3,1x,i0,1x,*(es26.17e3,1x))') &
        'DATA ',block%steps(i),block%times(i),merge(1,0,block%gaps(i)), &
        ((block%values(j,c,i),j=1,block%nodes),c=1,block%fields)
    enddo
  end subroutine

  subroutine require(valid,message)
    logical,intent(in) :: valid
    character(*),intent(in) :: message
    if(valid) return
    print '(a)','FAIL: '//message
    stop 1
  end subroutine
end program
