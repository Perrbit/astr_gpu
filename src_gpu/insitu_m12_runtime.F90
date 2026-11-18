module insitu_m12_cpp
  use iso_c_binding
  use iso_fortran_env, only: int64
  use mpi
  use astr_precision, only: real_kind,real_bytes
  use astr_precision_mpi, only: mpi_real_type
  use ieee_arithmetic, only: ieee_is_finite
  use checkpoint_state_io, only: checkpoint_state_require
  use fluid_runtime, only: cpp_device_context
  use insitu_sampling_cpp, only: probe_create,probe_capture,probe_metrics,probe_destroy
  use insitu_wall_cpp, only: profile_wall_create,profile_wall_capture,profile_wall_metrics,profile_wall_destroy
  implicit none
  private
  public :: m12_probe_count,m12_diagnostic_layout_supported,capture_m12_probes_cpp,release_m12_probes_cpp
  public :: capture_m12_wall_cpp,release_m12_wall_cpp
  public :: profile_insitu_cpp_supported
  integer,parameter :: m12_probe_count=72
  type(c_ptr),save :: probes=c_null_ptr
  type(c_ptr),save :: wall=c_null_ptr
contains
  logical function profile_insitu_cpp_supported() result(ok)
    use commvar, only: flowtype,ndims,numq,num_species,num_modequ,conschm,difschm,recon_schem,lchardecomp, &
      lfilter,diffterm,nondimen,lihomo,ljhomo,lkhomo,turbmode,spg_def
    use bc, only: bctype,turbinf,nscbc_farfield_policy_id,NSCBC_FARFIELD_POLICY_COMPATIBILITY, &
      nscbc_farfield_incoming_only_enabled,nscbc_farfield_sbli_shock_enabled
    use sponge_layer, only: spg_i0,spg_j0,spg_k0,spg_km
    use parallel, only: mpisize,isize,jsize,ksize
    logical :: boundary_supported
    ! Match the mature profile provider envelope; no device context exists yet here.
    boundary_supported=all(bctype==[11,21,41,50,1,1])
    if(all(bctype==[11,22,41,52,1,1])) then
      boundary_supported=nscbc_farfield_policy_id()==NSCBC_FARFIELD_POLICY_COMPATIBILITY
      if(difschm=='643c'.or.nscbc_farfield_incoming_only_enabled()) &
        boundary_supported=nscbc_farfield_incoming_only_enabled().and..not.nscbc_farfield_sbli_shock_enabled()
    endif
    ok=flowtype=='bl'.and.ndims==3.and.numq==5.and.num_species==0.and.num_modequ==0.and.turbmode=='none'.and. &
      conschm=='743e'.and.(difschm=='643e'.or.difschm=='643c').and.recon_schem==5.and.lchardecomp.and. &
      (.not.lfilter.or.difschm=='643c'.or.(all(bctype==[11,22,41,52,1,1]).and.nscbc_farfield_incoming_only_enabled())).and. &
      diffterm.and.nondimen.and..not.lihomo.and..not.ljhomo.and.lkhomo.and.turbinf=='prof'.and.boundary_supported.and. &
      spg_def=='layer'.and.all([spg_i0,spg_j0,spg_k0,spg_km]==0).and. &
      (difschm=='643c'.or.mpisize==1.or.(mpisize==2.and.max(isize,jsize,ksize)==2).or. &
       (mpisize==4.and.((isize==4.and.jsize==1.and.ksize==1).or.(isize==2.and.jsize==2.and.ksize==1).or. &
       (isize==2.and.jsize==1.and.ksize==2).or.(isize==1.and.jsize==2.and.ksize==2))))
  end function
  logical function m12_diagnostic_layout_supported() result(ok)
    use commvar, only: ia,ja,ka
    use parallel, only: isize,jsize,ksize,mpisize
    ok=(all([ia,ja,ka]==[64,32,24]).and.any(mpisize==[2,4]).and. &
         all([isize,jsize,ksize]==[mpisize,1,1])).or. &
       (all([ia,ja,ka]==[476,130,96]).and.mpisize==2.and.all([isize,jsize,ksize]==[2,1,1])).or. &
       (all([ia,ja,ka]==[2250,260,480]).and.mpisize==4.and.all([isize,jsize,ksize]==[4,1,1]))
  end function
  subroutine initialize_probes()
    use commvar, only: im,jm,km,ia,ja,ka,numq,const2,const5,const6,roinf,pinf,tinf,reynolds,prandtl,uinf,vinf,winf
    use parallel, only: ig0,jg0,kg0
    integer :: unit,status,closed,n,ijk(3),indices(3,m12_probe_count),owners(m12_probe_count),total(m12_probe_count),err
    real(real_kind) :: target(3),actual(3),scale,constants(6)
    character(256) :: header
    call checkpoint_state_require(m12_diagnostic_layout_supported().and.numq==5,MPI_COMM_WORLD,'C++ M12 probe layout')
    call checkpoint_state_require(all(ieee_is_finite([const2,const5,const6,roinf,pinf,tinf,reynolds,prandtl])).and. &
      min(const2,const5,const6,roinf,pinf,tinf,reynolds,prandtl)>0,MPI_COMM_WORLD,'C++ M12 probe reference state')
    scale=sqrt(uinf**2+vinf**2+winf**2)
    call checkpoint_state_require(ieee_is_finite(scale).and.scale>0,MPI_COMM_WORLD,'C++ M12 nonzero reference speed')
    open(newunit=unit,file='datin/insitu_probes.dat',status='old',action='read',iostat=status)
    call checkpoint_state_require(status==0,MPI_COMM_WORLD,'missing C++ M12 probe map')
    read(unit,'(a)',iostat=status) header
    call checkpoint_state_require(status==0.and.trim(header)=='ASTR_M12_PROBES_V1 72',MPI_COMM_WORLD,'C++ M12 map identity')
    owners=0
    do n=1,m12_probe_count
      read(unit,*,iostat=status) ijk,target,actual
      call checkpoint_state_require(status==0.and.all(ijk>=0).and.all(ijk<=[ia,ja,ka]).and. &
        all(ieee_is_finite([target,actual])),MPI_COMM_WORLD,'C++ M12 map entry')
      indices(:,n)=ijk-[ig0,jg0,kg0]
      if(all(indices(:,n)>=0).and.all(indices(:,n)<=[im,jm,km])) then
        if((indices(1,n)<im.or.ig0+im==ia).and.indices(3,n)<km) owners(n)=1
      endif
      if(owners(n)==0) indices(:,n)=-1
    enddo
    read(unit,'(a)',iostat=status) header
    close(unit,iostat=closed)
    call checkpoint_state_require(status<0.and.closed==0,MPI_COMM_WORLD,'C++ M12 extra map records')
    call MPI_Allreduce(owners,total,m12_probe_count,MPI_INTEGER,MPI_SUM,MPI_COMM_WORLD,err)
    call checkpoint_state_require(err==MPI_SUCCESS.and.all(total==1),MPI_COMM_WORLD,'C++ M12 unique probe owner')
    constants=[const2,const6,scale,roinf,pinf,tinf]
    status=probe_create(cpp_device_context(),int(m12_probe_count,c_int),int(indices,c_int),constants,probes)
    call checkpoint_state_require(status==0,MPI_COMM_WORLD,'allocate C++ M12 sparse device probes')
  end subroutine
  subroutine capture_m12_probes_cpp(values,ok,completed_step)
    use parallel, only: mpirank
    real(real_kind),intent(out) :: values(6,m12_probe_count)
    logical,intent(out) :: ok
    integer(int64),intent(in) :: completed_step
    real(real_kind) :: local(6,m12_probe_count)
    integer :: status,err,bad,any_bad
    integer(c_int64_t) :: before(5),after(5)
    if(.not.c_associated(probes)) call initialize_probes()
    status=probe_metrics(probes,before)
    if(status==0) status=probe_capture(probes,local,int(size(local,kind=int64)*real_bytes,c_size_t))
    bad=merge(0,1,status==0)
    call MPI_Allreduce(bad,any_bad,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,err)
    ok=err==MPI_SUCCESS.and.any_bad==0
    if(.not.ok) return
    values=0
    call MPI_Reduce(local,values,6*m12_probe_count,mpi_real_type,MPI_SUM,0,MPI_COMM_WORLD,err)
    ok=err==MPI_SUCCESS
    status=probe_metrics(probes,after)
    call checkpoint_state_require(status==0,MPI_COMM_WORLD,'C++ M12 transfer receipt')
    write(*,'(a,i0,a,i0,4(a,i0))') 'ASTR_RUNTIME_CPP_M12_PROBES rank=',mpirank,' step=',completed_step, &
      ' device_bytes=',after(1),' setup_index_h2d_bytes=',after(2),' status_d2h_bytes=',after(4)-before(4), &
      ' values_d2h_bytes=',after(5)-before(5)
  end subroutine
  subroutine release_m12_probes_cpp()
    integer :: status
    if(.not.c_associated(probes)) return
    status=probe_destroy(probes)
    call checkpoint_state_require(status==0,MPI_COMM_WORLD,'release C++ M12 probes')
  end subroutine
  subroutine capture_m12_wall_cpp(values,ok,completed_step)
    use commvar, only: im,km,ia,const2,const5,const6,roinf,uinf,vinf,winf,reynolds,prandtl,tempconst,tempconst1
    use commarray, only: x
    use parallel, only: ig0,mpirank
    real(real_kind),intent(out) :: values(3,im+1)
    logical,intent(out) :: ok
    integer(int64),intent(in) :: completed_step
    real(real_kind),allocatable :: weights(:,:)
    real(real_kind) :: material(7),reference(2)
    integer :: i,k,status
    integer(c_int64_t) :: before(5),after(5)
    if(.not.c_associated(wall)) then
      call checkpoint_state_require(m12_diagnostic_layout_supported(),MPI_COMM_WORLD,'C++ M12 wall layout')
      allocate(weights(0:im,0:km-1))
      do k=0,km-1
      do i=0,im
        weights(i,k)=.5_real_kind*(x(i,0,k+1,3)-x(i,0,k,3))
        if(k>0) then
          weights(i,k)=weights(i,k)+.5_real_kind*(x(i,0,k,3)-x(i,0,k-1,3))
        else
          weights(i,k)=weights(i,k)+.5_real_kind*(x(i,0,km,3)-x(i,0,km-1,3))
        endif
      enddo
      enddo
      material=[const2,const5,const6,reynolds,prandtl,tempconst,tempconst1]
      reference=[roinf,sqrt(uinf**2+vinf**2+winf**2)]
      status=profile_wall_create(cpp_device_context(),int(ig0,c_int),int(ia,c_int),material,reference,weights,wall)
      call checkpoint_state_require(status==0,MPI_COMM_WORLD,'allocate C++ M12 wall diagnostic')
    endif
    status=profile_wall_metrics(wall,before)
    if(status==0) status=profile_wall_capture(wall,values,int(size(values,kind=int64)*real_bytes,c_size_t))
    ok=status==0
    status=profile_wall_metrics(wall,after)
    call checkpoint_state_require(status==0,MPI_COMM_WORLD,'C++ M12 wall receipt')
    write(*,'(a,i0,a,i0,4(a,i0))') 'ASTR_RUNTIME_CPP_M12_WALL rank=',mpirank,' step=',completed_step, &
      ' device_bytes=',after(1),' setup_weights_h2d_bytes=',after(2),' status_d2h_bytes=',after(4)-before(4), &
      ' curve_d2h_bytes=',after(5)-before(5)
  end subroutine
  subroutine release_m12_wall_cpp()
    integer :: status
    if(.not.c_associated(wall)) return
    status=profile_wall_destroy(wall)
    call checkpoint_state_require(status==0,MPI_COMM_WORLD,'release C++ M12 wall diagnostic')
  end subroutine
end module
