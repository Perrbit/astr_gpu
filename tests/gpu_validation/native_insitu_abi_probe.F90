program native_insitu_abi_probe
  use iso_c_binding, only: c_int,c_int64_t
  use astr_precision, only: real_kind,c_real_kind,real_bytes,real_to_word
  implicit none
  interface
    integer(c_int) function color_state(restore,mask,ranges) bind(C,name='astr_insitu_m12_color_state')
      import c_int,c_int64_t,c_real_kind
      integer(c_int),value :: restore
      integer(c_int64_t),intent(inout) :: mask
      real(c_real_kind),intent(inout) :: ranges(6)
    end function
    integer(c_int) function color_range(component,store,range) bind(C,name='astr_insitu_m12_color_range')
      import c_int,c_real_kind
      integer(c_int),value :: component,store
      real(c_real_kind),intent(inout) :: range(2)
    end function
  end interface
  integer(c_int64_t) :: mask
  real(real_kind) :: expected(6),actual(6),pair(2)
  integer :: component,status
  expected=[0.1_real_kind,0.4_real_kind,1.0_real_kind,5.0_real_kind,-2.0_real_kind,-0.5_real_kind]
  mask=7; actual=expected
  if(color_state(1_c_int,mask,actual)/=1) error stop 'Native color-state ABI rejected'
  actual=0; mask=0
  status=color_state(0_c_int,mask,actual)
  if(status/=1.or.mask/=7) error stop 'Native color-state identity changed'
  if(any(real_to_word(actual)/=real_to_word(expected))) error stop 'Native color-state values changed'
  do component=0,2
    pair=0
    if(color_range(int(component,c_int),0_c_int,pair)/=1) error stop 'Native color-range lookup failed'
    if(any(real_to_word(pair)/=real_to_word(expected(2*component+1:2*component+2)))) &
      error stop 'Native color-range ABI differs'
  enddo
  write(*,'(a,i0)') 'PASS native in-situ Fortran/C++ ABI bytes=',real_bytes
end program
