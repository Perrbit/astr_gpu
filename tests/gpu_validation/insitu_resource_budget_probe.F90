program resource_budget_probe
  use iso_fortran_env, only: int64
  use insitu_resource_budget
  implicit none
  type(resource_budget) :: budget
  integer(int64) :: bytes
  logical :: ok
  call reserve_bytes(budget,1,1_int64,100_int64,ok)
  call require(.not.ok,'unconfigured budget')
  call configure_budget(budget,0_int64,0_int64,ok)
  call require(.not.ok,'zero limit')
  call configure_budget(budget,100_int64,20_int64,ok)
  call require(ok,'configure')
  call reserve_bytes(budget,0,1_int64,100_int64,ok)
  call require(.not.ok,'invalid slot')
  call reserve_bytes(budget,1,80_int64,99_int64,ok)
  call require(.not.ok.and.budget%current_bytes()==0,'headroom refusal is atomic')
  call reserve_bytes(budget,1,80_int64,100_int64,ok)
  call require(ok.and.budget%current_bytes()==80,'headroom equality admitted')
  call reserve_bytes(budget,1,1_int64,100_int64,ok)
  call require(.not.ok.and.budget%current_bytes()==80,'duplicate reservation')
  call reserve_bytes(budget,2,21_int64,100_int64,ok)
  call require(.not.ok.and.budget%peak_bytes()==80,'budget excess is atomic')
  call configure_budget(budget,200_int64,0_int64,ok)
  call require(.not.ok,'cannot reset live reservations')
  call reserve_bytes(budget,2,20_int64,40_int64,ok)
  call require(ok.and.budget%peak_bytes()==100,'exact budget equality admitted')
  call release_bytes(budget,1,ok)
  call require(ok.and.budget%current_bytes()==20,'release exact slot')
  call release_bytes(budget,1,ok)
  call require(.not.ok.and.budget%current_bytes()==20,'double release')
  call release_bytes(budget,2,ok)
  call require(ok.and.budget%current_bytes()==0.and.budget%peak_bytes()==100,'peak retained')
  call checked_bytes([32_int64,32_int64,32_int64,71_int64],8_int64,bytes,ok)
  call require(ok.and.bytes==18612224_int64,'statistics extent arithmetic')
  call checked_bytes([huge(bytes),2_int64],8_int64,bytes,ok)
  call require(.not.ok.and.bytes==0,'overflow rejected before multiplication')
  call checked_bytes([-1_int64,0_int64],8_int64,bytes,ok)
  call require(.not.ok,'negative extent with zero')
  call checked_bytes([huge(bytes),0_int64],8_int64,bytes,ok)
  call require(ok.and.bytes==0,'empty extent')
  call configure_budget(budget,huge(bytes),1_int64,ok)
  call require(ok,'large limit')
  call reserve_bytes(budget,1,huge(bytes),huge(bytes),ok)
  call require(.not.ok,'large free-space headroom')
  print *, 'PASS: budget admission, refusal atomicity, releases, checked byte arithmetic'
contains
  subroutine require(condition,message)
    logical,intent(in) :: condition
    character(*),intent(in) :: message
    if(.not.condition) then
      print *, 'FAIL: ',message
      stop 1
    endif
  end subroutine
end program
