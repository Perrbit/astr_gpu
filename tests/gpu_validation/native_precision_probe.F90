#ifdef _CUDA
module native_precision_device_probe
  use cudafor
  use astr_precision, only: real_kind
  implicit none
contains
  attributes(global) subroutine affine_kernel(values)
    real(real_kind),device,intent(inout) :: values(4)
    integer :: i
    i=threadIdx%x
    if(i<=4) values(i)=3.0_real_kind*values(i)+2.0_real_kind
  end subroutine affine_kernel
end module native_precision_device_probe
#endif

program native_precision_probe
  use iso_c_binding, only: c_int,c_int32_t,c_int64_t,c_float,c_double
  use astr_precision, only: real_kind,c_real_kind,clock_kind,real_bytes,numerical_precision,real_to_word,word_to_real
  use astr_precision_io, only: read_source_reals,source_real_matches
  use commarray, only: q,qrhs,rho,prs,tmp,vel,dxi,jacob
  use commvar, only: physical_time=>time,deltat,ctime
  use constdef, only: pi,num1d60
#ifdef _CUDA
  use cudafor
  use native_precision_device_probe
#endif
  implicit none
  interface
    function precision_affine(x) bind(C,name='astr_precision_affine') result(y)
      import c_real_kind
      real(c_real_kind),value :: x
      real(c_real_kind) :: y
    end function precision_affine
    function precision_bytes() bind(C,name='astr_precision_bytes') result(n)
      import c_int
      integer(c_int) :: n
    end function precision_bytes
  end interface
  real(real_kind) :: encoded_values(6),decoded_values(6)
  integer(c_int64_t) :: words(6)
  character(16) :: mode
#ifdef _CUDA
  real(real_kind),device :: values_d(4)
  real(real_kind) :: values(4)
  integer :: status
#endif

  call get_command_argument(1,mode)
  if(trim(mode)=='cast') then
    call cast_file()
    stop
  endif

  if(real_kind/=c_real_kind) error stop 'Numerical and C kinds differ'
  if(storage_size(0.0_real_kind)/8/=real_bytes) error stop 'Fortran byte size differs'
  if(precision_bytes()/=real_bytes) error stop 'C++ byte size differs'
  if(storage_size(0.0_clock_kind)/=64) error stop 'Performance clock is not FP64'
  if(precision_affine(4.0_real_kind)/=14.0_real_kind) error stop 'Fortran/C++ value ABI differs'
  if(kind(q)/=real_kind.or.kind(qrhs)/=real_kind) error stop 'Authoritative state kind differs'
  if(kind(rho)/=real_kind.or.kind(prs)/=real_kind.or.kind(tmp)/=real_kind.or.kind(vel)/=real_kind) &
    error stop 'Primitive state kind differs'
  if(kind(dxi)/=real_kind.or.kind(jacob)/=real_kind) error stop 'Geometry state kind differs'
  if(kind(physical_time)/=real_kind.or.kind(deltat)/=real_kind) error stop 'Physical time kind differs'
  if(kind(pi)/=real_kind.or.kind(num1d60)/=real_kind) error stop 'Coefficient kind differs'
  if(kind(ctime)/=clock_kind) error stop 'Performance clock state kind differs'
  encoded_values=[0.0_real_kind,-0.0_real_kind,-1.0_real_kind,1.0_real_kind,tiny(0.0_real_kind),huge(0.0_real_kind)]
  words=real_to_word(encoded_values)
  decoded_values=word_to_real(words)
  if(any(real_to_word(decoded_values)/=words)) error stop 'Real metadata bits changed'
  if(word_to_real(int(transfer(0.1_c_float,0_c_int32_t),c_int64_t),4)/=real(0.1_c_float,real_kind)) &
    error stop 'FP32 source clock conversion differs'
  if(word_to_real(transfer(0.1_c_double,0_c_int64_t),8)/=real(0.1_c_double,real_kind)) &
    error stop 'FP64 source clock conversion differs'
  if(real_to_word(word_to_real(int(-2147483648_c_int64_t,c_int64_t),4))/=real_to_word(-0.0_real_kind)) &
    error stop 'FP32 source negative zero changed'
  if(real_to_word(word_to_real(not(huge(0_c_int64_t)),8))/=real_to_word(-0.0_real_kind)) &
    error stop 'FP64 source negative zero changed'
  if(real_bytes==4) then
    if(any(words/=int(transfer(encoded_values,[0_c_int32_t],size(words)),c_int64_t))) &
      error stop 'FP32 metadata has undefined padding'
  else
    if(any(words/=transfer(encoded_values,words))) error stop 'FP64 metadata layout changed'
  endif
#ifdef _CUDA
  values_d=[-1.0_real_kind,0.0_real_kind,1.0_real_kind,2.0_real_kind]
  call affine_kernel<<<1,32>>>(values_d)
  status=cudaDeviceSynchronize()
  if(status/=cudaSuccess) error stop 'Precision device probe failed'
  values=values_d
  if(any(values/=[-1.0_real_kind,2.0_real_kind,5.0_real_kind,8.0_real_kind])) error stop 'Device arithmetic differs'
#endif
  call control_read_probe()
  write(*,'(A,A,A,I0,A,I0)') 'PASS: numerical_precision=',numerical_precision, &
    ' bytes=',real_bytes,' clock_bytes=',storage_size(0.0_clock_kind)/8
contains
  subroutine cast_file()
    real(c_float) :: source32(1024)
    real(c_double) :: source64(1024)
    real(real_kind) :: target(1024)
    character(1024) :: input,output,argument
    integer(c_int64_t) :: bytes,position
    integer :: source_bytes,input_unit,output_unit,n
    call get_command_argument(2,argument)
    read(argument,*) source_bytes
    if(source_bytes/=4.and.source_bytes/=8) error stop 'Invalid cast source width'
    call get_command_argument(3,input)
    call get_command_argument(4,output)
    open(newunit=input_unit,file=trim(input),status='old',access='stream',form='unformatted',action='read')
    inquire(unit=input_unit,size=bytes)
    if(mod(bytes,int(source_bytes,c_int64_t))/=0) error stop 'Partial cast input value'
    open(newunit=output_unit,file=trim(output),status='new',access='stream',form='unformatted',action='write')
    do position=0,bytes/source_bytes-1,1024
      n=int(min(1024_c_int64_t,bytes/source_bytes-position))
      if(source_bytes==4) then
        read(input_unit) source32(:n)
        target(:n)=real(source32(:n),real_kind)
      else
        read(input_unit) source64(:n)
        target(:n)=real(source64(:n),real_kind)
      endif
      write(output_unit) target(:n)
    enddo
    close(input_unit)
    close(output_unit)
    write(*,'(a,i0,a,i0)') 'PASS: native cast source_bytes=',source_bytes,' target_bytes=',real_bytes
  end subroutine

  subroutine control_read_probe()
    real(c_float) :: source32(2,3),value32
    real(c_double) :: source64(2,3),value64
    real(real_kind) :: values(2,3),value,expected(2,3),expected_value
    integer :: unit,bytes,status,prefix
    integer(c_int64_t) :: tail
    logical :: flag
    source32=reshape([0.1_c_float,-0.0_c_float,1.0_c_float,-2.0_c_float,0.5_c_float,3.0_c_float],[2,3])
    source64=reshape([0.1_c_double,-0.0_c_double,1.0_c_double,-2.0_c_double,0.5_c_double,3.0_c_double],[2,3])
    value32=0.1_c_float; value64=0.1_c_double
    if(.not.source_real_matches(real(value32,real_kind),0.1_real_kind,4)) &
      error stop 'Equivalent source precision configuration rejected'
    if(source_real_matches(real(value32,real_kind),0.11_real_kind,4)) &
      error stop 'Changed source precision configuration accepted'
    do bytes=4,8,4
      open(newunit=unit,status='scratch',access='stream',form='unformatted')
      if(bytes==4) then
        write(unit) 27,value32,.true.,source32,123_c_int64_t
        expected=real(source32,real_kind); expected_value=real(value32,real_kind)
      else
        write(unit) 27,value64,.true.,source64,123_c_int64_t
        expected=real(source64,real_kind); expected_value=real(value64,real_kind)
      endif
      rewind(unit)
      read(unit) prefix
      call read_source_reals(unit,value,bytes,status)
      if(status/=0.or.value/=expected_value) error stop 'Source scalar record conversion differs'
      read(unit) flag
      call read_source_reals(unit,values,bytes,status)
      if(status/=0) error stop 'Source matrix record conversion failed'
      if(any(real_to_word(values)/=real_to_word(expected))) error stop 'Source matrix values changed'
      read(unit) tail
      if(prefix/=27.or..not.flag.or.tail/=123) error stop 'Source record stream position changed'
      close(unit)
    enddo
    open(newunit=unit,status='scratch',access='stream',form='unformatted')
    write(unit) value32
    rewind(unit)
    call read_source_reals(unit,values,4,status)
    if(status==0) error stop 'Truncated source record accepted'
    rewind(unit)
    call read_source_reals(unit,value,16,status)
    if(status==0) error stop 'Unsupported source precision accepted'
    close(unit)
  end subroutine
end program native_precision_probe
