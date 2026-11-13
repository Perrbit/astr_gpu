program checkpoint_logical_probe
  use iso_fortran_env, only: int32,int64
  use astr_precision_io, only: pack_logical32,unpack_logical32
  implicit none
  integer(int32) :: words(3),source
  integer :: unit,status
  integer(int64) :: position
  logical :: flags(3)
  character(2048) :: path
  words=[-1_int32,0_int32,1_int32]
  flags=unpack_logical32(words)
  if(.not.all(flags.eqv.[.true.,.false.,.true.])) error stop 'foreign true representation'
  if(any(pack_logical32(flags)/=[1_int32,0_int32,1_int32])) error stop 'canonical logical words'
  open(newunit=unit,status='scratch',access='stream',form='unformatted')
  write(unit) pack_logical32(flags)
  inquire(unit,pos=position)
  if(position/=13) error stop 'three logical words must occupy 12 bytes'
  read(unit,pos=1) words
  close(unit)
  if(any(words/=[1_int32,0_int32,1_int32])) error stop 'stream logical encoding'
  call get_command_argument(1,path)
  if(len_trim(path)>0) then
    open(newunit=unit,file=trim(path),status='old',access='stream',form='unformatted', &
      convert='little_endian',action='read',iostat=status)
    if(status/=0) error stop 'open frozen archive control'
    read(unit,pos=25,iostat=status) source
    close(unit)
    if(status/=0.or..not.unpack_logical32(source)) error stop 'frozen checkpoint enabled flag'
    print '(a,i0)', 'frozen_true_word=',source
  endif
  print '(a,i0,a)', 'native_true_word=',transfer(.true.,0_int32),'; canonical_true_word=1; width=4 bytes'
end program
