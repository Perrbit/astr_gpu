module validation_io
  implicit none
  private
  public :: rhs_validation_requested,write_rhs_validation_snapshot, &
            write_q_validation_snapshot,write_primitive_validation_snapshot, &
            write_sensor_validation_snapshot, &
            write_compact_statistics_validation_snapshot
  logical,save :: configured=.false.,enabled=.false.
  character(len=1024),save :: prefix=''
  logical,save :: compact_configured=.false.,compact_enabled=.false.
  character(len=1024),save :: compact_prefix=''
contains
  subroutine configure_rhs_validation()
    integer :: status,length
    if(configured) return
    call get_environment_variable('ASTR_VALIDATION_RHS_PREFIX',prefix, &
                                  length=length,status=status)
    enabled=status==0.and.length>0
    if(enabled) prefix=prefix(1:length)
    configured=.true.
  end subroutine configure_rhs_validation

  logical function rhs_validation_requested()
    call configure_rhs_validation()
    rhs_validation_requested=enabled
  end function rhs_validation_requested

  subroutine configure_compact_validation()
    integer :: status,length
    if(compact_configured) return
    call get_environment_variable('ASTR_VALIDATION_COMPACT_PREFIX',compact_prefix, &
                                  length=length,status=status)
    compact_enabled=status==0.and.length>0
    if(compact_enabled) compact_prefix=compact_prefix(1:length)
    compact_configured=.true.
  end subroutine configure_compact_validation

  subroutine write_compact_statistics_validation_snapshot()
    use iso_fortran_env, only: int32,int64,real64
    use commvar, only: ia,ja,ka,im,jm,km,hm,nstep,time,lavg,feqavg, &
                       nondimen,reynolds,prandtl,const5,cp,tempconst,tempconst1
    use commarray, only: x,rho,vel,prs,tmp,dvel,dtmp,bnorm_j0
    use parallel, only: mpirank,irk,jrk,krk,ig0,jg0,kg0,isize,jsize,ksize
    implicit none
    character(len=1200) :: filename
    integer(int32) :: fixed_header(22)
    integer(int64) :: step_value
    real(real64) :: time_value,constants(6)
    integer :: unit
    logical :: has_wall

    call configure_compact_validation()
    if(.not.compact_enabled .or. nstep<=0 .or. (.not.lavg)) return
    if(mod(nstep,feqavg)/=0) return
    has_wall=jrk==0 .and. allocated(bnorm_j0)
    fixed_header=(/int(1,int32),int(z'01020304',int32),int(8,int32), &
      int(merge(1,0,has_wall),int32),int(merge(1,0,nondimen),int32), &
      int(ia+1,int32),int(ja+1,int32),int(ka+1,int32), &
      int(isize,int32),int(jsize,int32),int(ksize,int32),int(mpirank,int32), &
      int(irk,int32),int(jrk,int32),int(krk,int32), &
      int(ig0,int32),int(jg0,int32),int(kg0,int32), &
      int(im+1,int32),int(jm+1,int32),int(km+1,int32),int(hm,int32)/)
    step_value=int(nstep,int64)
    time_value=real(time,real64)
    constants=(/reynolds,prandtl,const5,cp,tempconst,tempconst1/)
    write(filename,'(A,".step",I8.8,".rank",I8.8,".bin")') &
      trim(compact_prefix),nstep,mpirank
    open(newunit=unit,file=trim(filename),access='stream',form='unformatted', &
         status='new',action='write',convert='little_endian')
    write(unit) 'ASTRCRF1'
    write(unit) fixed_header
    write(unit) step_value
    write(unit) time_value
    write(unit) constants
    write(unit) x(0:im,0:jm,0:km,1:3)
    write(unit) rho(0:im,0:jm,0:km)
    write(unit) vel(0:im,0:jm,0:km,1:3)
    write(unit) prs(0:im,0:jm,0:km)
    write(unit) tmp(0:im,0:jm,0:km)
    write(unit) dvel(0:im,0:jm,0:km,1:3,1:3)
    write(unit) dtmp(0:im,0:jm,0:km,1:3)
    if(has_wall) then
      write(unit) x(-hm:im+hm,0,0:km,1:3)
      write(unit) bnorm_j0(0:im,0:km,1:3)
    endif
    close(unit)
  end subroutine write_compact_statistics_validation_snapshot

  subroutine write_rhs_validation_snapshot(label,step_index,stage_index)
    use commvar, only: im,jm,km,numq,nstep,rkstep
    use commarray, only: qrhs
    use parallel, only: mpirank
    character(len=*),intent(in) :: label
    integer,intent(in),optional :: step_index,stage_index
    character(len=1200) :: filename
    integer :: unit,step_value,stage_value
    call configure_rhs_validation()
    step_value=nstep
    stage_value=rkstep
    if(present(step_index)) step_value=step_index
    if(present(stage_index)) stage_value=stage_index
    if(.not.enabled.or.step_value/=0.or.stage_value<1) return
    write(filename,'(A,".",A,".step",I8.8,".rk",I2.2,".rank",I8.8,".bin")') &
      trim(prefix),trim(label),step_value,stage_value,mpirank
    open(newunit=unit,file=trim(filename),access='stream',form='unformatted',status='new',action='write')
    write(unit) im,jm,km,numq
    write(unit) qrhs
    close(unit)
  end subroutine write_rhs_validation_snapshot

  subroutine write_q_validation_snapshot(label,step_index,stage_index)
    use commvar, only: im,jm,km,hm,numq,nstep,rkstep
    use commarray, only: q
    use parallel, only: mpirank
    character(len=*),intent(in) :: label
    integer,intent(in),optional :: step_index,stage_index
    character(len=1200) :: filename
    integer :: unit,step_value,stage_value
    call configure_rhs_validation()
    step_value=nstep
    stage_value=rkstep
    if(present(step_index)) step_value=step_index
    if(present(stage_index)) stage_value=stage_index
    if(.not.enabled.or.step_value/=0.or.stage_value<1) return
    write(filename,'(A,".",A,".step",I8.8,".rk",I2.2,".rank",I8.8,".bin")') &
      trim(prefix),trim(label),step_value,stage_value,mpirank
    open(newunit=unit,file=trim(filename),access='stream',form='unformatted',status='new',action='write')
    write(unit) im,jm,km,hm,numq
    write(unit) q
    close(unit)
  end subroutine write_q_validation_snapshot

  subroutine write_primitive_validation_snapshot(label,step_index,stage_index)
    use commvar, only: im,jm,km,hm,nstep,rkstep
    use commarray, only: rho,vel,prs,tmp
    use parallel, only: mpirank
    character(len=*),intent(in) :: label
    integer,intent(in),optional :: step_index,stage_index
    character(len=1200) :: filename
    integer :: unit,step_value,stage_value
    call configure_rhs_validation()
    step_value=nstep
    stage_value=rkstep
    if(present(step_index)) step_value=step_index
    if(present(stage_index)) stage_value=stage_index
    if(.not.enabled.or.step_value/=0.or.stage_value<1) return
    write(filename,'(A,".",A,".step",I8.8,".rk",I2.2,".rank",I8.8,".bin")') &
      trim(prefix),trim(label),step_value,stage_value,mpirank
    open(newunit=unit,file=trim(filename),access='stream',form='unformatted',status='new',action='write')
    write(unit) im,jm,km,hm
    write(unit) rho,vel,prs,tmp
    close(unit)
  end subroutine write_primitive_validation_snapshot

  subroutine write_sensor_validation_snapshot(label,sensor,mask,step_index,stage_index)
    use commvar, only: im,jm,km,nstep,rkstep
    use parallel, only: mpirank
    character(len=*),intent(in) :: label
    real(8),intent(in) :: sensor(:,:,:)
    integer(1),intent(in) :: mask(:,:,:)
    integer,intent(in),optional :: step_index,stage_index
    character(len=1200) :: filename
    integer :: unit,step_value,stage_value
    call configure_rhs_validation()
    step_value=nstep
    stage_value=rkstep
    if(present(step_index)) step_value=step_index
    if(present(stage_index)) stage_value=stage_index
    if(.not.enabled.or.step_value/=0.or.stage_value<1) return
    write(filename,'(A,".",A,".step",I8.8,".rk",I2.2,".rank",I8.8,".bin")') &
      trim(prefix),trim(label),step_value,stage_value,mpirank
    open(newunit=unit,file=trim(filename),access='stream',form='unformatted',status='new',action='write')
    write(unit) im,jm,km
    write(unit) sensor,mask
    close(unit)
  end subroutine write_sensor_validation_snapshot
end module validation_io
