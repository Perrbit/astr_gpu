!+---------------------------------------------------------------------+
!| This module is to define constants.                                 |
!+---------------------------------------------------------------------+
!| CHANGE RECORD                                                       |
!| -------------                                                       |
!| 06-02-2021  | Created by J. Fang                                    |
!+---------------------------------------------------------------------+
module constdef
  use astr_precision, only: real_kind
  !
  implicit none
  private :: real_kind
  !
  real(real_kind),parameter :: pi=4.0_real_kind*atan(1.0_real_kind),                     &
                       num1d35=1.0_real_kind/35.0_real_kind, num1d3=1.0_real_kind/3.0_real_kind, &
                       num2d3=2.0_real_kind/3.0_real_kind, num1d24=1.0_real_kind/24.0_real_kind, &
                       num4d3=4.0_real_kind/3.0_real_kind, num1d6=1.0_real_kind/6.0_real_kind, &
                       num1d12=1.0_real_kind/12.0_real_kind, num7d12=7.0_real_kind/12.0_real_kind, &
                       num7d9=7.0_real_kind/9.0_real_kind, num1d36=1.0_real_kind/36.0_real_kind, &
                       num1d60=1.0_real_kind/60.0_real_kind, num65d3=65.0_real_kind/3.0_real_kind, &
                       num20d3=20.0_real_kind/3.0_real_kind, num1d11=1.0_real_kind/11.0_real_kind, &
                       num25d12=25.0_real_kind/12.0_real_kind, num11d6=11.0_real_kind/6.0_real_kind, &
                       num1d840=1.0_real_kind/840.0_real_kind, num13d60=13.0_real_kind/60.0_real_kind, &
                       num1d30=1.0_real_kind/30.0_real_kind, num47d60=47.0_real_kind/60.0_real_kind, &
                       num5d6=5.0_real_kind/6.0_real_kind, num1d18=1.0_real_kind/18.0_real_kind, &
                       num19d18=19.0_real_kind/18.0_real_kind, num5d9=5.0_real_kind/9.0_real_kind, &
                       num9d36=9.0_real_kind/36.0_real_kind, num2d15=2.0_real_kind/15.0_real_kind, &
                       num37d60=37.0_real_kind/60.0_real_kind, num17d6=17.0_real_kind/16.0_real_kind, &
                       num1d120=1.0_real_kind/120.0_real_kind, num131d120=131.0_real_kind/120.0_real_kind, &
                       num193d256=193.0_real_kind/256.0_real_kind, num105d512=105.0_real_kind/512.0_real_kind, &
                       num15d128=15.0_real_kind/128.0_real_kind, num45d1024=45.0_real_kind/1024.0_real_kind, &
                       num5d512=5.0_real_kind/512.0_real_kind, num1d1024=1.0_real_kind/1024.0_real_kind
  !+-------------------------------------------------------------------+
  !| constants.                                                        |
  !+-------------------------------------------------------------------+
  !
end module constdef
!+---------------------------------------------------------------------+
!| The end of the module constdef.                                     |
!+---------------------------------------------------------------------+
