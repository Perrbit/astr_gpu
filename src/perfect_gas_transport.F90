module perfect_gas_transport
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: read_transport_environment
contains
  subroutine read_transport_environment(prandtl,sutherland_k,changed,status)
    real(8),intent(out) :: prandtl,sutherland_k
    logical,intent(out) :: changed
    integer,intent(out) :: status
    logical :: supplied
    prandtl=0.72d0
    sutherland_k=110.3d0
    changed=.false.
    call read_positive_real('ASTR_PERFECT_GAS_PRANDTL',prandtl,supplied,status)
    changed=changed.or.supplied
    if(status/=0) return
    call read_positive_real('ASTR_SUTHERLAND_TEMPERATURE_K',sutherland_k,supplied,status)
    changed=changed.or.supplied
  end subroutine read_transport_environment

  subroutine read_positive_real(name,value,supplied,status)
    character(len=*),intent(in) :: name
    real(8),intent(inout) :: value
    logical,intent(out) :: supplied
    integer,intent(out) :: status
    character(len=256) :: token,extra
    integer :: length,env_status,ios
    real(8) :: parsed
    call get_environment_variable(name,token,length=length,status=env_status)
    supplied=env_status/=1
    status=0
    if(.not.supplied) return
    status=1
    if(env_status/=0.or.length<=0.or.length>len(token)) return
    if(scan(token,'/,*')/=0) return
    read(token,*,iostat=ios)parsed
    if(ios/=0) return
    if(.not.ieee_is_finite(parsed)) return
    if(parsed<=0.d0) return
    ! Reject trailing list-directed values instead of silently ignoring them.
    read(token,*,iostat=ios)parsed,extra
    if(ios>=0) return
    value=parsed
    status=0
  end subroutine read_positive_real
end module perfect_gas_transport
