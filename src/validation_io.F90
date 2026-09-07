module validation_io
  implicit none
  private
  public :: rhs_validation_requested,write_rhs_validation_snapshot, &
            write_q_validation_snapshot,write_primitive_validation_snapshot, &
            write_sensor_validation_snapshot
  logical,save :: configured=.false.,enabled=.false.
  character(len=1024),save :: prefix=''
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
