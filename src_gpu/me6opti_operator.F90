module me6_flow_cpp
  use iso_c_binding, only: c_ptr,c_int,c_int64_t
  use astr_precision, only: real_kind
  use commvar, only: im,jm,km,hm,ia,ja,ka,lihomo,ljhomo,lkhomo
  use parallel, only: ig0,jg0,kg0,mpirank
  use me6opti_operator, only: me6_coefficients
  use me6opti_runtime, only: prepare_me6_geometry,me6_parts,me6_peers
  use fluid_explicit, only: prepare=>fluid_me6_prepare,metrics=>fluid_me6_metrics, &
    me6_diffusion_cpp=>fluid_me6_diffusion
  use mpi
  implicit none
  private
  public :: prepare_me6_cpp,me6_diffusion_cpp,report_me6_cpp
contains
  subroutine prepare_me6_cpp(parent,mode)
    type(c_ptr),intent(in) :: parent
    integer(c_int),intent(in) :: mode
    real(real_kind) :: ci(9,6),cd(9,6),cp(9,6)
    integer(c_int) :: first(3),last(3),inner_first(3),inner_last(3),lower(3),upper(3),parts(3),periodic(3),status
    integer(c_int64_t) :: values(4)
    integer :: h,side,p,column,half,axis,offset(3),total(3)
    call prepare_me6_geometry()
    do h=1,3
    do side=1,2
      column=2*(h-1)+side;half=(2*side-3)*(2*h-1)
      do p=-4,4
        call me6_coefficients(half,p,ci(p+5,column),cd(p+5,column),cp(p+5,column))
      enddo
    enddo
    enddo
    first=-hm;last=[im,jm,km]+hm;inner_first=0;inner_last=[im,jm,km]
    offset=[ig0,jg0,kg0];total=[ia,ja,ka];periodic=merge(1_c_int,0_c_int,[lihomo,ljhomo,lkhomo])
    do axis=1,3
      if(periodic(axis)==1) cycle
      first(axis)=max(-hm,-offset(axis));last(axis)=min(last(axis),total(axis)-offset(axis))
      inner_first(axis)=max(0,4-offset(axis));inner_last(axis)=min(inner_last(axis),total(axis)-4-offset(axis))
    enddo
    lower=me6_peers(1,:);upper=me6_peers(2,:);parts=me6_parts
    status=prepare(parent,int(MPI_COMM_WORLD,c_int),lower,upper,parts,periodic,first,last,inner_first,inner_last,mode,ci,cd,cp)
    if(status/=0) then
      write(*,'(a,i0)') 'C++ ME6 preparation status=',status
      error stop 'C++ ME6 preparation failed'
    endif
    status=metrics(parent,values)
    if(status/=0) error stop 'C++ ME6 setup metrics'
    write(*,'(a,i0,a,i0,a,i0,a,i0)') 'ASTR_CPP_ME6 rank=',mpirank,' mode=',mode, &
      ' device_bytes=',values(1),' pinned_bytes=',values(2)
  end subroutine
  subroutine report_me6_cpp(parent)
    type(c_ptr),intent(in) :: parent
    integer(c_int64_t) :: values(4)
    integer(c_int) :: status
    status=metrics(parent,values)
    if(status/=0) error stop 'C++ ME6 final metrics'
    write(*,'(a,i0,a,i0,a,i0)') 'ASTR_CPP_ME6_TRAFFIC rank=',mpirank, &
      ' sent_bytes=',values(3),' host_stage_bytes=',values(4)
  end subroutine
end module
