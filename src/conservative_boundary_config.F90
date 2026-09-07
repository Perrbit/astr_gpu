module conservative_boundary_config
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_is_finite
  use perfect_gas_boundary, only: valid_state=>admissible
  implicit none
  private
  type,public :: conservative_boundary_settings
    logical :: enabled=.false.
    real(8) :: split_x=0.d0,q_left(5)=0.d0,q_right(5)=0.d0
  end type conservative_boundary_settings
  public :: read_conservative_boundary_config
contains
  subroutine read_conservative_boundary_config(filename,settings,status)
    character(len=*),intent(in) :: filename
    type(conservative_boundary_settings),intent(out) :: settings
    integer,intent(out) :: status
    integer :: schema,unit,ios
    real(8) :: split_x,q_left(5),q_right(5)
    character(len=1024) :: line
    namelist /conservative_boundary/ schema,split_x,q_left,q_right

    settings=conservative_boundary_settings()
    status=0
    if(len_trim(filename)==0) return
    status=1
    schema=0
    split_x=ieee_value(0.d0,ieee_quiet_nan)
    q_left=split_x
    q_right=split_x
    open(newunit=unit,file=filename,status='old',action='read',iostat=ios)
    if(ios/=0) return
    read(unit,nml=conservative_boundary,iostat=ios)
    if(ios/=0) then
      close(unit)
      return
    endif
    ! A second namelist or stray content must not be silently ignored.
    do
      read(unit,'(a)',iostat=ios)line
      if(ios<0) exit
      if(ios>0) then
        close(unit)
        return
      endif
      line=adjustl(line)
      if(len_trim(line)>0.and.line(1:1)/='!') then
        close(unit)
        return
      endif
    enddo
    close(unit)
    if(schema/=1) return
    if(.not.ieee_is_finite(split_x)) return
    if(.not.valid_state(q_left)) return
    if(.not.valid_state(q_right)) return
    settings%split_x=split_x
    settings%q_left=q_left
    settings%q_right=q_right
    settings%enabled=.true.
    status=0
  end subroutine read_conservative_boundary_config
end module conservative_boundary_config
