module astr_precision
  use iso_c_binding, only: c_float,c_double,c_int32_t,c_int64_t
  implicit none
  private
  public :: real_kind,c_real_kind,clock_kind,real_bytes,numerical_precision
  public :: real_to_word,word_to_real

#ifdef ASTR_REAL32
  integer,parameter :: real_kind=c_float
  character(len=4),parameter :: numerical_precision='fp32'
#else
  integer,parameter :: real_kind=c_double
  character(len=4),parameter :: numerical_precision='fp64'
#endif
  integer,parameter :: c_real_kind=real_kind
  integer,parameter :: clock_kind=c_double
  integer,parameter :: real_bytes=storage_size(0.0_real_kind)/8
contains
  elemental function real_to_word(value) result(word)
    real(real_kind),intent(in) :: value
    integer(c_int64_t) :: word
    if(real_bytes==4) then
      word=int(transfer(value,0_c_int32_t),c_int64_t)
    else
      word=transfer(value,word)
    endif
  end function

  elemental function word_to_real(word,source_bytes) result(value)
    integer(c_int64_t),intent(in) :: word
    integer,optional,intent(in) :: source_bytes
    real(real_kind) :: value
    integer :: bytes
    bytes=real_bytes
    if(present(source_bytes)) bytes=source_bytes
    if(bytes==4) then
      value=real(transfer(int(word,c_int32_t),0.0_c_float),real_kind)
    else
      value=real(transfer(word,0.0_c_double),real_kind)
    endif
  end function
end module astr_precision
