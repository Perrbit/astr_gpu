module chemistry_monitor
  use iso_fortran_env, only: real64
  use mpi
  implicit none
  private
  public :: write_air5_flow_monitor
  public :: decode_air5_monitor_state
  integer,save :: stride=-1,profile_unit,wall_unit,probe_unit,profile_count=0,probe_count=0
  integer,allocatable,save :: nodes(:,:),global_nodes(:,:)
  real(real64),allocatable,save :: local(:,:),total(:,:),primitive(:,:)
contains
  subroutine decode_air5_monitor_state(state,values,status)
    use chemistry_flow_state, only: air5_conservative_to_primitive
    real(real64),intent(in) :: state(11)
    real(real64),intent(out) :: values(12)
    integer,intent(out) :: status
    real(real64) :: rho,velocity(3),temperature,tv,pressure,ys(5)
    call air5_conservative_to_primitive(state,rho,velocity,temperature,ys,tv,pressure,status)
    values=[rho,velocity,temperature,tv,pressure,ys]
  end subroutine

  subroutine write_air5_flow_monitor(sample_time)
    use commvar, only: ia,ja,ka,im,jm,km,nstep,use_gpu,flowtype,lreadgrid,nmonitor,imon
    use commarray, only: q,x
    use parallel, only: ig0,jg0,kg0,mpirank
    use chemistry_flow_state, only: air5_conservative_to_primitive
    use chemistry_transport, only: air5_transport_properties
#ifdef _CUDA
    use chemistry_monitor_gpu, only: collect_air5_monitor_gpu
#endif
    use ieee_arithmetic, only: ieee_is_finite
    real(real64),intent(in) :: sample_time
    character(len=64) :: value
    integer :: status,lo,hi,ierr,n,s,i,j,k,a,b,station
    real(real64) :: rho,vel(3),temp,tv,pressure,ys(5),mu,kt,kv,smu(5),bd(5,5),md(5)
    real(real64) :: h1,h2,w(3),tau,heat
    if(stride==-1) then
      stride=0; value=''
      call get_environment_variable('ASTR_AIR5_FLOW_MONITOR_STRIDE',value,status=status)
      if(status==0) then
        read(value,*,iostat=status) stride
        if(status/=0.or.stride<1) stride=-2
      endif
      call MPI_Allreduce(stride,lo,1,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)
      call MPI_Allreduce(stride,hi,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
      if(lo<0.or.lo/=hi) call MPI_Abort(MPI_COMM_WORLD,94,ierr)
      hi=0
      if(nmonitor>0) hi=maxval(imon(:,4))
      call MPI_Allreduce(hi,probe_count,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
      if(stride==0.and.probe_count==0) return
      if((trim(flowtype)/='air5hbl'.and.trim(flowtype)/='air5sbli')) &
        call MPI_Abort(MPI_COMM_WORLD,94,ierr)
      if(stride>0) then
        if(lreadgrid.or.ja<2) call MPI_Abort(MPI_COMM_WORLD,94,ierr)
        profile_count=3*(ja+1)+3*(ia+1)
      endif
      n=profile_count+probe_count
      allocate(nodes(3,n),global_nodes(3,n),local(15,n),total(15,n),primitive(12,n))
      s=0
      if(stride>0) then
      do station=1,3
        do j=0,ja
          s=s+1; global_nodes(:,s)=[nint(real(ia,real64)*station/4.d0),j,ka/2]
        enddo
      enddo
      do i=0,ia
        do j=0,2
          s=s+1; global_nodes(:,s)=[i,j,ka/2]
        enddo
      enddo
      endif
      nodes=-1
      do s=1,profile_count
        i=global_nodes(1,s)-ig0; j=global_nodes(2,s)-jg0; k=global_nodes(3,s)-kg0
        if(i<0.or.i>im.or.j<0.or.j>jm.or.k<0.or.k>km) cycle
        ! The upper-side rank owns duplicated interfaces; retain global endpoints.
        if(i==im.and.ig0+im<ia) cycle
        if(j==jm.and.jg0+jm<ja) cycle
        if(k==km.and.kg0+km<ka) cycle
        nodes(:,s)=[i,j,k]
      enddo
      ! Reuse readmonc/monitorsearch ownership and original probe identifiers.
      do s=1,nmonitor
        nodes(:,profile_count+imon(s,4))=imon(s,1:3)
      enddo
      if(mpirank==0) then
        if(stride>0) then
        open(newunit=profile_unit,file='monitor/air5_profiles.dat',status='new')
        write(profile_unit,'(A)') '# step time station i j x y z rho u v w T Tv p Y_N2 Y_O2 Y_N Y_O Y_NO'
        open(newunit=wall_unit,file='monitor/air5_wall.dat',status='new')
        write(wall_unit,'(A)') '# step time i x y z tau_xy q_into_gas p; three-point diagnostic, not solver flux'
        endif
        if(probe_count>0) then
          open(newunit=probe_unit,file='monitor/air5_probes.dat',status='new')
          write(probe_unit,'(A)') '# step time probe x y z rho u v w T Tv p Y_N2 Y_O2 Y_N Y_O Y_NO'
        endif
      endif
    endif
    if(profile_count==0.and.probe_count==0) return
    if(probe_count==0) then
      if(mod(nstep,stride)/=0) return
    endif
    n=size(nodes,2)
    local=0.d0
#ifdef _CUDA
    if(use_gpu) then
      call collect_air5_monitor_gpu(nodes,local)
    else
#endif
      do s=1,n
        i=nodes(1,s); j=nodes(2,s); k=nodes(3,s)
        if(i<0) cycle
        local(1:3,s)=x(i,j,k,:); local(4:14,s)=q(i,j,k,1:11); local(15,s)=1.d0
      enddo
#ifdef _CUDA
    endif
#endif
    call MPI_Reduce(local,total,15*n,MPI_DOUBLE_PRECISION,MPI_SUM,0,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,94,ierr)
    if(mpirank/=0) return
    if(any(total(15,:)/=1.d0).or..not.all(ieee_is_finite(total))) &
      call MPI_Abort(MPI_COMM_WORLD,94,ierr)
    do s=1,n
      call decode_air5_monitor_state(total(4:14,s),primitive(:,s),status)
      if(status/=0) call MPI_Abort(MPI_COMM_WORLD,94,ierr)
    enddo
    do s=1,probe_count
      a=profile_count+s
      write(probe_unit,'(I12,1X,ES25.17E3,1X,I8,15(1X,ES25.17E3))') &
        nstep,sample_time,s,total(1:3,a),primitive(:,a)
    enddo
    if(probe_count>0) flush(probe_unit)
    if(stride==0) return
    if(mod(nstep,stride)/=0) return
    do s=1,3*(ja+1)
      station=(s-1)/(ja+1)+1
      write(profile_unit,'(I12,1X,ES25.17E3,3(1X,I8),15(1X,ES25.17E3))') &
        nstep,sample_time,station,global_nodes(1:2,s),total(1:3,s),primitive(:,s)
    enddo
    do i=0,ia
      a=3*(ja+1)+3*i+1; b=a+2
      h1=total(2,a+1)-total(2,a); h2=total(2,b)-total(2,a)
      if(h1<=0.d0.or.h2<=h1) call MPI_Abort(MPI_COMM_WORLD,94,ierr)
      w=[-(h1+h2)/(h1*h2),h2/(h1*(h2-h1)),-h1/(h2*(h2-h1))]
      call air5_transport_properties(primitive(5,a),primitive(6,a),primitive(7,a),primitive(8:12,a), &
        mu,kt,kv,smu,bd,md,status)
      if(status/=0) call MPI_Abort(MPI_COMM_WORLD,94,ierr)
      tau=mu*dot_product(w,primitive(2,a:b))
      heat=-kt*dot_product(w,primitive(5,a:b))-kv*dot_product(w,primitive(6,a:b))
      if(.not.ieee_is_finite(tau).or..not.ieee_is_finite(heat)) call MPI_Abort(MPI_COMM_WORLD,94,ierr)
      write(wall_unit,'(I12,1X,ES25.17E3,1X,I8,6(1X,ES25.17E3))') &
        nstep,sample_time,i,total(1:3,a),tau,heat,primitive(7,a)
    enddo
    flush(profile_unit); flush(wall_unit)
  end subroutine
end module
