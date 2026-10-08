module validation_io
  use commvar, only: nstep
  implicit none
  private
  public :: rhs_validation_requested,write_rhs_validation_snapshot, &
            write_q_validation_snapshot,write_primitive_validation_snapshot, &
            write_sensor_validation_snapshot, &
            write_compact_statistics_validation_snapshot,write_scalar_validation_snapshot
  public :: sensor_detail_requested,write_sensor_detail_validation_snapshot
  logical,save :: configured=.false.,enabled=.false.
  character(len=1024),save :: prefix=''
  integer,save :: validation_step=0,validation_step_secondary=-1,validation_step_end=-1
  logical,save :: compact_configured=.false.,compact_enabled=.false.
  character(len=1024),save :: compact_prefix=''
contains
  subroutine configure_rhs_validation()
    character(len=64) :: step_buffer
    integer :: status,length,step_status,step_length,ios
    if(configured) return
    call get_environment_variable('ASTR_VALIDATION_RHS_PREFIX',prefix, &
                                  length=length,status=status)
    enabled=status==0.and.length>0
    if(enabled) prefix=prefix(1:length)
    validation_step=0
    call get_environment_variable('ASTR_VALIDATION_RHS_STEP',step_buffer, &
                                  length=step_length,status=step_status)
    if(step_status==0 .and. step_length>0) then
      read(step_buffer(1:step_length),*,iostat=ios) validation_step
      if(ios/=0 .or. validation_step<0) &
        error stop 'ASTR_VALIDATION_RHS_STEP must be a non-negative integer'
    endif
    validation_step_secondary=-1
    call get_environment_variable('ASTR_VALIDATION_RHS_STEP_SECONDARY',step_buffer, &
                                  length=step_length,status=step_status)
    if(step_status==0 .and. step_length>0) then
      read(step_buffer(1:step_length),*,iostat=ios) validation_step_secondary
      if(ios/=0 .or. validation_step_secondary<0) &
        error stop 'ASTR_VALIDATION_RHS_STEP_SECONDARY must be a non-negative integer'
    endif
    call get_environment_variable('ASTR_VALIDATION_RHS_STEP_END',step_buffer, &
                                  length=step_length,status=step_status)
    if(step_status==0 .and. step_length>0) then
      read(step_buffer(1:step_length),*,iostat=ios) validation_step_end
      if(ios/=0 .or. validation_step_end<validation_step) &
        error stop 'ASTR_VALIDATION_RHS_STEP_END must not precede the first step'
    endif
    configured=.true.
  end subroutine configure_rhs_validation

  logical function validation_snapshot_requested(step_value)
    integer,intent(in) :: step_value
    call configure_rhs_validation()
    validation_snapshot_requested=enabled.and. &
      (step_value==validation_step.or.step_value==validation_step_secondary.or. &
       (validation_step_end>=validation_step.and.step_value>=validation_step.and.step_value<=validation_step_end))
  end function validation_snapshot_requested

  logical function rhs_validation_requested()
    rhs_validation_requested=validation_snapshot_requested(nstep)
  end function rhs_validation_requested

  logical function sensor_detail_requested(stage_index)
    use commvar, only: ia,ja,ka,rkstep
    integer,intent(in),optional :: stage_index
    character(8) :: value
    integer :: status,length,stage
    call get_environment_variable('ASTR_M12_SENSOR_DETAIL',value,length=length,status=status)
    sensor_detail_requested=.false.
    if(status==1) return
    if(status/=0.or.length/=1.or.value/='1') error stop 'ASTR_M12_SENSOR_DETAIL must be exactly 1'
    if(.not.all([ia,ja,ka]==[64,32,24])) error stop 'M12 sensor detail requires approved bounded grid'
    stage=rkstep
    if(present(stage_index)) stage=stage_index
    sensor_detail_requested=rhs_validation_requested().and.stage>=1.and.stage<=3
  end function

  subroutine write_sensor_detail_validation_snapshot(gradient,pressure,stage_index)
    use commvar, only: im,jm,km,hm,rkstep,npdci,npdcj,npdck
    use parallel, only: mpirank
    real(8),intent(in) :: gradient(0:,0:,0:,:,:),pressure(-hm:,-hm:,-hm:)
    integer,intent(in),optional :: stage_index
    character(1200) :: filename
    integer :: unit,stage
    if(.not.sensor_detail_requested(stage_index)) return
    stage=rkstep
    if(present(stage_index)) stage=stage_index
    write(filename,'(a,".sensor_detail.step",i8.8,".rk",i2.2,".rank",i8.8,".bin")') &
      trim(prefix),nstep,stage,mpirank
    open(newunit=unit,file=trim(filename),access='stream',form='unformatted',status='new',action='write')
    write(unit) im,jm,km,hm,npdci,npdcj,npdck
    write(unit) gradient(0:im,0:jm,0:km,1:3,1:3)
    write(unit) pressure(-hm:im+hm,-hm:jm+hm,-hm:km+hm)
    close(unit)
  end subroutine

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
    if(.not.validation_snapshot_requested(step_value).or.stage_value<1) return
    write(filename,'(A,".",A,".step",I8.8,".rk",I2.2,".rank",I8.8,".bin")') &
      trim(prefix),trim(label),step_value,stage_value,mpirank
    open(newunit=unit,file=trim(filename),access='stream',form='unformatted',status='new',action='write')
    write(unit) im,jm,km,numq
    write(unit) qrhs
    close(unit)
  end subroutine write_rhs_validation_snapshot

  subroutine write_scalar_validation_snapshot(label,values,stage_index)
    use iso_fortran_env, only: real64
    use commvar, only: im,jm,km,hm,rkstep
    use parallel, only: mpirank
    character(len=*),intent(in) :: label
    real(real64),intent(in) :: values(-hm:im+hm,-hm:jm+hm,-hm:km+hm)
    integer,intent(in),optional :: stage_index
    character(len=1200) :: filename
    integer :: unit,stage_value
    if(.not.rhs_validation_requested()) return
    stage_value=rkstep
    if(present(stage_index)) stage_value=stage_index
    write(filename,'(A,".",A,".step",I8.8,".rk",I2.2,".rank",I8.8,".bin")') &
      trim(prefix),trim(label),nstep,stage_value,mpirank
    open(newunit=unit,file=trim(filename),access='stream',form='unformatted',status='new',action='write')
    write(unit) im,jm,km,hm
    write(unit) values
    close(unit)
  end subroutine write_scalar_validation_snapshot

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
    if(.not.validation_snapshot_requested(step_value).or.stage_value<1) return
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
    if(.not.validation_snapshot_requested(step_value).or.stage_value<1) return
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
    if(.not.validation_snapshot_requested(step_value).or.stage_value<1) return
    write(filename,'(A,".",A,".step",I8.8,".rk",I2.2,".rank",I8.8,".bin")') &
      trim(prefix),trim(label),step_value,stage_value,mpirank
    open(newunit=unit,file=trim(filename),access='stream',form='unformatted',status='new',action='write')
    write(unit) im,jm,km
    write(unit) sensor,mask
    close(unit)
  end subroutine write_sensor_validation_snapshot
end module validation_io
