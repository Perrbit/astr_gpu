module insitu_selected_runtime
  use mpi
  use iso_fortran_env, only: int64
  use astr_precision, only: real_kind,real_bytes
  use astr_precision_mpi, only: mpi_real_type
  use insitu_selected_statistics, only: selected_statistics,selected_external_sampler
  use insitu_sample_moments, only: sample_profile_size,sample_profile_terms,finish_sample_profile
  use insitu_spatial_statistics, only: span_node_weights
  use insitu_run_config, only: catalog_product
  use benchmark_runtime, only: insitu_clock,report_insitu_timing
  use insitu_product_schedule, only: runtime_product_state,runtime_catalog_count,runtime_catalog_state, &
    runtime_active,runtime_waiting,runtime_catalog_disable,runtime_catalog_status
#ifdef _CUDA
  use cudafor
  use iso_c_binding, only: c_char,c_int,c_size_t
  use insitu_sample_moments_gpu, only: activate_selected_moments_gpu,selected_moment_transfer_counts_gpu, &
    reduce_selected_profile_gpu,selected_profile_workspace_bytes_gpu,selected_profile_transfer_counts_gpu
#endif
  implicit none
  private
  public :: configure_selected_runtime,observe_selected_runtime,selected_runtime_info,selected_runtime_value
  public :: release_selected_runtime,selected_runtime_active
  public :: selected_runtime_sampling_info
  public :: selected_runtime_file
  public :: refresh_selected_runtime
  public :: selected_profile_result,selected_runtime_profile,selected_runtime_workspace
  public :: selected_runtime_wall_references
  public :: selected_runtime_wall_grid
  public :: selected_runtime_wall_scale
  public :: selected_runtime_cartesian
  type :: selected_profile_result
    integer :: fields(5)=0,nfields=0
    logical :: covariance=.false.
    real(real_kind) :: fixed_coordinate=0,span(2)=0,window(2)=0,actual_window(2)=0,duration=0
    integer(int64) :: segments=0
    real(real_kind),allocatable :: coordinate(:),measure(:),density(:),mean_r(:,:),mean_f(:,:), &
      covariance_r(:,:),covariance_f(:,:),density_stress(:,:),rms_r(:,:),rms_f(:,:)
  end type
  type(selected_statistics),save :: series
  integer,save :: selected_group=0
  integer(int64),save :: host_limit=0,device_limit=0,device_reserve=0
  logical,save :: first_observation=.true.
  logical,allocatable,save :: active_consumers(:)
  real(real_kind),save :: fixed_wall_references(6)=0
  logical,save :: geometry_checked=.false.,geometry_valid=.false.,geometry_rejected=.false.
  integer,save :: geometry_direction(3)=0
  character(256),save :: geometry_message=''
#ifdef _CUDA
  interface
    integer(c_int) function pci_bus_id(name,length,device) bind(C,name='cudaDeviceGetPCIBusId')
      import c_char,c_int
      character(c_char),intent(out) :: name(*)
      integer(c_int),value :: length,device
    end function
  end interface
#endif
contains
  logical function selected_runtime_cartesian(bottom_wall) result(valid)
    use commvar, only: lreadgrid
    logical,optional,intent(in) :: bottom_wall
    valid=.not.lreadgrid
    if(.not.lreadgrid) return
    valid=geometry_checked.and.geometry_valid
    if(present(bottom_wall)) then
      if(bottom_wall) valid=valid.and.geometry_direction(2)==1
    endif
  end function

  ! Validate read grids after initialization/restart, never before the device upload.
  subroutine inspect_selected_cartesian(ok,message)
    use commvar, only: ia,ja,ka,im,jm,km,hm,use_gpu,nstep
    use commarray, only: x
    use parallel, only: ig0,jg0,kg0,mpirank
    use insitu_boundary_layer, only: cartesian_block_axes,cartesian_axis_direction
#ifdef _CUDA
    use commarray_gpu, only: x_d
    use insitu_boundary_layer_gpu, only: cartesian_block_axes_gpu
#endif
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    real(real_kind),allocatable :: axes(:),local(:),global(:)
    integer :: cells(3),intervals(3),origin(3),local_offset,global_offset,d,n,nlocal,owned,status,err,ignored
    integer(int64) :: extra_host,extra_device,download
    logical :: valid
    real(8) :: started
    if(geometry_checked) then
      ok=geometry_valid;message=geometry_message;return
    endif
    started=insitu_clock();cells=[im,jm,km];intervals=[ia,ja,ka];origin=[ig0,jg0,kg0]
    n=sum(intervals)+3;nlocal=sum(cells)+3
    extra_host=(2_int64*n+nlocal)*real_bytes;extra_device=0;download=0
    if(use_gpu) extra_device=int(nlocal,int64)*real_bytes+storage_size(0)/8
    call selected_runtime_workspace(extra_host,extra_device,ok)
    message='Cartesian geometry proof exceeds configured resources'
    if(.not.ok) return
    allocate(axes(nlocal),local(n),global(n),stat=status)
    ok=collective_ok(status==0)
    if(.not.ok) return
#ifdef _CUDA
    if(use_gpu) then
      call cartesian_block_axes_gpu(x_d,cells,hm,axes,valid)
      download=extra_device
    else
#endif
      call cartesian_block_axes(x,cells,hm,axes,valid)
#ifdef _CUDA
    endif
#endif
    geometry_checked=.true.;geometry_valid=.false.
    geometry_message='Cartesian block proof failed: coordinates must be finite and separable; device workspace must be available'
    if(.not.collective_ok(valid)) then
      ok=.false.;message=geometry_message;return
    endif
    local=0;local_offset=0;global_offset=0
    do d=1,3
      if(all(pack(origin,[d/=1,d/=2,d/=3])==0)) then
        owned=cells(d)
        if(origin(d)+cells(d)==intervals(d)) owned=owned+1
        local(global_offset+origin(d)+1:global_offset+origin(d)+owned)=axes(local_offset+1:local_offset+owned)
      endif
      local_offset=local_offset+cells(d)+1;global_offset=global_offset+intervals(d)+1
    enddo
    call MPI_Allreduce(local,global,n,mpi_real_type,MPI_SUM,MPI_COMM_WORLD,err)
    if(err/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,err,ignored)
    valid=.true.;local_offset=0;global_offset=0
    do d=1,3
      geometry_direction(d)=cartesian_axis_direction(global(global_offset+1:global_offset+intervals(d)+1))
      valid=valid.and.geometry_direction(d)/=0
      valid=valid.and.all(axes(local_offset+1:local_offset+cells(d)+1)== &
        global(global_offset+origin(d)+1:global_offset+origin(d)+cells(d)+1))
      local_offset=local_offset+cells(d)+1;global_offset=global_offset+intervals(d)+1
    enddo
    geometry_valid=collective_ok(valid)
    geometry_message='Cartesian axes must be strictly monotone and agree across all MPI shared nodes'
    if(geometry_valid) geometry_message=''
    ok=geometry_valid;message=geometry_message
    write(*,'(a,i0,a,l1,a,3(i0,1x),a,i0,a,i0,a,i0)') 'ASTR_SELECTED_GEOMETRY rank=',mpirank, &
      ' cartesian=',ok,' axis_directions=',geometry_direction,' temporary_host_bytes=',extra_host, &
      ' temporary_device_bytes=',extra_device,' explicit_axis_status_d2h_bytes=',download
    call report_insitu_timing('selected_cartesian_geometry',started,nstep)
  end subroutine

  subroutine axis_coordinates(axis,lower,last,values)
    use commarray, only: x
    integer,intent(in) :: axis,lower,last
    real(real_kind),intent(inout) :: values(:)
    integer :: high,node(3),i
    if(any(pack(series%origin,[axis/=1,axis/=2,axis/=3])/=0)) return
    high=series%origin(axis)+series%cells(axis)-1
    if(high+1==series%intervals(axis)) high=high+1
    do i=max(lower,series%origin(axis)),min(last,high)
      node=0;node(axis)=i-series%origin(axis)
      values(i-lower+1)=x(node(1),node(2),node(3),axis)
    enddo
  end subroutine

  subroutine selected_runtime_wall_grid(definition,retained_bytes,distances,available,ok,message)
    use insitu_boundary_layer, only: wall_grid_spacings
    use commvar, only: nstep
    type(catalog_product),intent(in) :: definition
    integer(int64),intent(in) :: retained_bytes
    real(real_kind),allocatable,intent(out) :: distances(:,:)
    logical,allocatable,intent(out) :: available(:,:)
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    real(real_kind),allocatable :: geometry(:),global_geometry(:)
    integer(int64) :: workspace
    integer :: nx,nz,n,rows,status,err,ignored
    real(8) :: started
    started=insitu_clock();ok=.false.
    message='wall grid diagnostic workspace exceeds configured resources'
    nx=series%intervals(1);nz=series%intervals(3);n=nx+nz+4
    rows=int(definition%ranges(2,1)-definition%ranges(1,1))+1
    workspace=retained_bytes+2_int64*n*real_bytes+ &
      5_int64*rows*(real_bytes+storage_size(.true.)/8)
    call selected_runtime_workspace(workspace,0_int64,ok)
    if(.not.ok) return
    allocate(geometry(n),global_geometry(n),distances(5,rows),available(5,rows),stat=status)
    ok=collective_ok(status==0)
    if(.not.ok) return
    geometry=0
    call axis_coordinates(1,0,nx,geometry(:nx+1))
    call axis_coordinates(2,0,1,geometry(nx+2:nx+3))
    call axis_coordinates(3,0,nz,geometry(nx+4:))
    call MPI_Allreduce(geometry,global_geometry,n,mpi_real_type,MPI_SUM,MPI_COMM_WORLD,err)
    if(err/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,err,ignored)
    call wall_grid_spacings(global_geometry(:nx+1),global_geometry(nx+2:nx+3),global_geometry(nx+4:), &
      int(definition%ranges(:,[1,3])),series%periodic([1,3]),distances,available,ok)
    ok=collective_ok(ok);message='wall grid diagnostic needs finite monotone Cartesian axes and nonzero first height'
    if(ok) message=''
    call report_insitu_timing('wall_grid_geometry',started,nstep)
  end subroutine

  subroutine selected_runtime_wall_references(values)
    real(real_kind),intent(out) :: values(6)
    values=fixed_wall_references
  end subroutine

  subroutine selected_runtime_workspace(extra_host,extra_device,ok)
    integer(int64),intent(in) :: extra_host,extra_device
    logical,intent(out) :: ok
    integer(int64) :: host_bytes,device_bytes
    call series%bytes(host_bytes,device_bytes)
    call approve_allocation(host_bytes+extra_host,device_bytes+extra_device,extra_device,ok)
  end subroutine

  subroutine refresh_selected_runtime()
    integer :: n,p,status
    logical :: ok
    if(.not.series%configured.or.first_observation) return
    n=runtime_catalog_count()
    if(allocated(active_consumers)) then
      if(size(active_consumers)/=n) deallocate(active_consumers)
    endif
    if(.not.allocated(active_consumers)) allocate(active_consumers(n))
    do p=1,n
      status=runtime_catalog_status(p)
      active_consumers(p)=status==runtime_active.or.status==runtime_waiting
    enddo
    call series%set_active(active_consumers,ok)
    if(.not.ok) error stop 'selected statistics request/status size mismatch'
  end subroutine

  subroutine selected_runtime_wall_scale(definition,retained_bytes,reference,rho_wall,wall_y,ok,message)
    use insitu_boundary_layer, only: wall_reference,wall_reference_result
    type(catalog_product),intent(in) :: definition
    integer(int64),intent(in) :: retained_bytes
    type(wall_reference_result),intent(out) :: reference
    real(real_kind),intent(out) :: rho_wall,wall_y
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    type(selected_profile_result) :: wall
    type(catalog_product) :: target
    integer :: p,g,h,row,slots(5)
    real(8) :: started
    reference=wall_reference_result();rho_wall=0;wall_y=0;ok=.false.
    message='wall_units target has no active wall_profile history'
    g=series%find(definition%id);h=series%find(definition%wall_units)
    if(min(g,h)==0) return
    do p=1,size(series%requests)
      if(series%requests(p)%definition%id/=definition%wall_units) cycle
      target=series%requests(p)%definition;exit
    enddo
    if(target%view%kind/='wall_profile'.or.target%wall_reference%flags(1)/=1) then
      message='wall_units target needs a wall_profile with wall_reference enabled';return
    endif
    if(any(target%ranges(:,3)/=definition%ranges(:,3)).or. &
      definition%index<target%ranges(1,1).or.definition%index>target%ranges(2,1).or. &
      ((definition%view%sampling=='station_span').neqv.(target%view%sampling=='wall_span'))) then
      message='wall_units target must contain the station and use the same span and spatial average';return
    endif
    if(.not.collective_ok(series%matching_coverage(g,h))) then
      message='wall_units sampling histories differ; equal duration alone is insufficient';return
    endif
    started=insitu_clock()
    call selected_runtime_profile(target,wall,ok,message,retained_bytes)
    if(.not.ok) return
    slots=[findloc(wall%fields(:wall%nfields),6,dim=1),findloc(wall%fields(:wall%nfields),31,dim=1), &
      findloc(wall%fields(:wall%nfields),32,dim=1),findloc(wall%fields(:wall%nfields),33,dim=1), &
      findloc(wall%fields(:wall%nfields),34,dim=1)]
    if(any(slots==0)) then
      ok=.false.;message='wall_units target lacks compatible molecular wall dependencies';return
    endif
    row=int(definition%index-target%ranges(1,1))+1
    rho_wall=wall%density(row);wall_y=wall%fixed_coordinate
    call wall_reference([wall%mean_r(slots(2),row),0.e0_real_kind,wall%mean_r(slots(3),row)], &
      rho_wall,wall%mean_r(slots(5),row),wall%mean_r(slots(1),row),wall%mean_r(slots(4),row), &
      0.e0_real_kind,0.e0_real_kind,reference,ok)
    message='wall_units reference has no representable nonzero statistical shear scale'
    if(ok.and.reference%scale_available) message=''
    call report_insitu_timing('station_wall_reference',started,series%groups(g)%last_step)
  end subroutine

  subroutine selected_runtime_profile(definition,profile,ok,message,retained_host)
    use commvar, only: ka,nstep
    use parallel, only: mpirank
    type(catalog_product),intent(in) :: definition
    type(selected_profile_result),intent(out) :: profile
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    integer(int64),optional,intent(in) :: retained_host
    real(real_kind),allocatable :: geometry(:),global_geometry(:),span_weights(:),local_profile(:,:), &
      local(:,:),global(:,:)
    integer,allocatable :: span_nodes(:),j_nodes(:)
    real(real_kind) :: terms(24),state_values(30),rho_mean,mean_r(5),mean_f(5),cr(6),cf(6),stress(6),rr(3),rf(3),measure
    integer(int64) :: host_bytes,device_bytes,extra_host,extra_device,download,before_calls,before_down,before_up, &
      after_calls,after_down,after_up
    integer :: g,n,nf,nc,ny,nz,ny_local,nk_local,p,j,k,lo,hi,jlo,jhi,upper,idx,local_j,target,status,err,ignored,line_axis
    logical :: valid,averaged
    real(8) :: started
    ok=.false.;message='station profile has no ready selected-time history'
    if(.not.series%configured) return
    g=series%find(definition%id)
    if(g==0) return
    if(series%groups(g)%failed.or.series%groups(g)%covered<=0) return
    started=insitu_clock();nf=series%groups(g)%nfields;nc=sample_profile_size(nf,series%groups(g)%covariance)
    n=size(series%groups(g)%keys);ny_local=0;nk_local=0
    if(n>0) then
      ny_local=count(series%groups(g)%indices(3,:)==series%groups(g)%indices(3,1))
      nk_local=n/ny_local
    endif
    line_axis=2
    if(definition%view%kind=='wall_profile') line_axis=1
    jlo=int(definition%ranges(1,line_axis));jhi=int(definition%ranges(2,line_axis));ny=jhi-jlo+1
    lo=int(definition%ranges(1,3));hi=int(definition%ranges(2,3));nz=hi-lo+1
    averaged=definition%view%sampling=='station_span'.or.definition%view%sampling=='wall_span'
    call series%bytes(host_bytes,device_bytes)
    extra_host=(int(4*nc+40,int64)*ny+2_int64*(1+ny+nz)+nk_local)*real_bytes+ &
      int(nk_local+ny_local,int64)*storage_size(0)/8
    if(present(retained_host)) extra_host=extra_host+retained_host
    extra_device=0
#ifdef _CUDA
    if(series%on_device.and.n>0) then
      if(selected_profile_workspace_bytes_gpu(series%groups(g)%device)==0) &
        extra_device=(int(nk_local,int64)+int(nc,int64)*ny_local)*real_bytes
    endif
#endif
    call approve_allocation(host_bytes+extra_host,device_bytes+extra_device,extra_device,valid)
    message='station profile workspace exceeds configured resources'
    if(.not.valid) return
    allocate(geometry(1+ny+nz),global_geometry(1+ny+nz),span_nodes(nk_local),j_nodes(ny_local), &
      span_weights(nk_local),local(nc,ny),global(nc,ny),stat=status)
    if(.not.collective_ok(status==0)) return
    geometry=0;local=0;download=0
    call axis_coordinates(definition%axis,int(definition%index),int(definition%index),geometry(1:1))
    call axis_coordinates(line_axis,jlo,jhi,geometry(2:1+ny))
    call axis_coordinates(3,lo,hi,geometry(2+ny:))
    call MPI_Allreduce(geometry,global_geometry,size(geometry),mpi_real_type,MPI_SUM,MPI_COMM_WORLD,err)
    if(err/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,err,ignored)
    do k=1,nk_local
      span_nodes(k)=series%origin(3)+series%groups(g)%indices(3,(k-1)*ny_local+1)
    enddo
    do j=1,ny_local
      j_nodes(j)=series%origin(line_axis)+series%groups(g)%indices(line_axis,j)
    enddo
    call span_node_weights(global_geometry(2+ny:),lo,ka,series%periodic(3),span_nodes,averaged,span_weights,valid)
    message='station span must use finite, strictly monotone Cartesian coordinates'
    if(.not.collective_ok(valid)) return
#ifdef _CUDA
    if(series%on_device) then
      call selected_profile_transfer_counts_gpu(before_calls,before_down,before_up)
      call reduce_selected_profile_gpu(series%groups(g)%device,ny_local,span_weights,nf, &
        series%groups(g)%covariance,local_profile,valid)
      call selected_profile_transfer_counts_gpu(after_calls,after_down,after_up)
      download=after_down-before_down
    else
#endif
      allocate(local_profile(nc,ny_local),stat=status)
      valid=status==0
      if(valid) then
        local_profile=0
        do k=1,nk_local
        do j=1,ny_local
          p=(k-1)*ny_local+j
          state_values(:size(series%groups(g)%states,2))=series%groups(g)%states(p,:)
          call sample_profile_terms(size(series%groups(g)%states,2),nf,state_values,span_weights(k), &
            series%groups(g)%covariance,terms,ok)
          valid=valid.and.ok
          local_profile(:,j)=local_profile(:,j)+terms(:nc)
        enddo
        enddo
      endif
#ifdef _CUDA
    endif
#endif
    ok=.false.;message='invalid station profile moments'
    if(.not.collective_ok(valid)) return
    do j=1,ny_local
      target=j_nodes(j)
      if(series%periodic(line_axis).and.target==0.and.jlo>0.and.jhi==series%intervals(line_axis)) &
        target=series%intervals(line_axis)
      local(:,target-jlo+1)=local_profile(:,j)
    enddo
    call MPI_Allreduce(local,global,size(local),mpi_real_type,MPI_SUM,MPI_COMM_WORLD,err)
    if(err/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,err,ignored)
    if(series%periodic(line_axis).and.jlo==0.and.jhi==series%intervals(line_axis)) global(:,ny)=global(:,1)
    allocate(profile%coordinate(ny),profile%measure(ny),profile%density(ny),profile%mean_r(nf,ny),profile%mean_f(nf,ny), &
      profile%covariance_r(6,ny),profile%covariance_f(6,ny),profile%density_stress(6,ny), &
      profile%rms_r(3,ny),profile%rms_f(3,ny),stat=status)
    if(.not.collective_ok(status==0)) return
    profile%nfields=nf;profile%fields=series%groups(g)%field_ids
    profile%covariance=series%groups(g)%covariance;profile%fixed_coordinate=global_geometry(1)
    profile%span=[global_geometry(2+ny),global_geometry(1+ny+nz)];profile%coordinate=global_geometry(2:1+ny)
    profile%window=series%window
    profile%actual_window=[max(series%window(1),series%groups(g)%first_time),min(series%window(2),series%groups(g)%last_time)]
    profile%duration=series%groups(g)%covered;profile%segments=series%groups(g)%segments
    valid=.true.
    do j=1,ny
      call finish_sample_profile(nf,profile%covariance,global(:,j),measure,rho_mean,mean_r(:nf),mean_f(:nf), &
        cr,cf,stress,rr,rf,ok)
      valid=valid.and.ok
      profile%measure(j)=measure;profile%density(j)=rho_mean
      profile%mean_r(:,j)=mean_r(:nf);profile%mean_f(:,j)=mean_f(:nf)
      profile%covariance_r(:,j)=cr;profile%covariance_f(:,j)=cf;profile%density_stress(:,j)=stress
      profile%rms_r(:,j)=rr;profile%rms_f(:,j)=rf
    enddo
    ok=collective_ok(valid)
    call report_insitu_timing('station_profile_reduce',started,nstep)
    if(ok) then
      write(*,'(a,i0,a,a,a,i0,a,i0,a,i0)') 'ASTR_STATION_PROFILE rank=',mpirank,' id=',definition%id, &
        ' source_nodes=',n,' result_rows=',ny,' explicit_1d_d2h_bytes=',download
      message=''
    endif
  end subroutine

  subroutine selected_runtime_file(path,writing,identity,budget,allow_repartition)
    use checkpoint_state_io, only: checkpoint_state_identity,checkpoint_rank_words,checkpoint_state_require
    character(*),intent(in) :: path
    logical,intent(in) :: writing
    logical,optional,intent(in) :: allow_repartition
    type(checkpoint_state_identity),intent(in) :: identity
    integer(int64),intent(in) :: budget
    type(checkpoint_state_identity) :: stored
    integer(int64),allocatable :: words(:)
    integer(int64) :: host_bytes,device_bytes,temporary
    logical :: ok,permit
    permit=.false.
    if(present(allow_repartition)) permit=allow_repartition
    call checkpoint_state_require(series%configured.and..not.permit,MPI_COMM_WORLD, &
      'selected statistics restore currently requires the same MPI topology')
    if(writing) then
      call series%pack(identity%step,identity%time,budget,words,ok)
      call checkpoint_state_require(ok,MPI_COMM_WORLD,'cannot pack selected statistics within checkpoint buffer budget')
    endif
    stored=identity
    call checkpoint_rank_words(path,writing,series%intervals+1,stored,words,budget,MPI_COMM_WORLD)
    if(writing) return
    call series%restore_bytes(identity%step,identity%time,budget,words,host_bytes,device_bytes,temporary,ok)
    call checkpoint_state_require(ok,MPI_COMM_WORLD,'selected statistics layout/clock/window/backend mismatch')
    call approve_allocation(host_bytes+temporary,device_bytes,device_bytes,ok)
    call checkpoint_state_require(ok,MPI_COMM_WORLD,'selected statistics restore resource budget')
    call series%restore(identity%step,identity%time,budget,words,ok)
    call checkpoint_state_require(ok,MPI_COMM_WORLD,'cannot restore selected statistics state')
    first_observation=.true.;selected_group=0
    geometry_checked=.false.;geometry_valid=.false.;geometry_rejected=.false.;geometry_direction=0;geometry_message=''
  end subroutine

  logical function selected_runtime_active() result(active)
    active=series%configured
  end function

  subroutine configure_selected_runtime(window,host_budget,device_budget,reserve,ok)
    use ieee_arithmetic, only: ieee_is_finite
    use commvar, only: ia,ja,ka,im,jm,km,lihomo,ljhomo,lkhomo,use_gpu,nondimen, &
      roinf,uinf,vinf,winf,tinf,const5,cp,gamma,mach
    use parallel, only: ig0,jg0,kg0
    real(real_kind),intent(in) :: window(2)
    integer(int64),intent(in) :: host_budget,device_budget,reserve
    logical,intent(out) :: ok
    real(real_kind) :: velocity(3),speed,cp_reference
    call series%configure(window,[ia,ja,ka],[ig0,jg0,kg0],[im,jm,km],[lihomo,ljhomo,lkhomo],use_gpu,ok)
    host_limit=host_budget;device_limit=device_budget;device_reserve=reserve
    velocity=[uinf,vinf,winf];speed=maxval(abs(velocity))
    if(speed>0) speed=speed*sqrt(sum((velocity/speed)**2))
    cp_reference=0
    if(nondimen) then
      if(ieee_is_finite(const5).and.const5>0) cp_reference=1.e0_real_kind/const5
    else
      cp_reference=cp
    endif
    fixed_wall_references=[roinf,speed,tinf,cp_reference,gamma,mach]
    first_observation=.true.;selected_group=0
    geometry_checked=.false.;geometry_valid=.false.;geometry_rejected=.false.;geometry_direction=0;geometry_message=''
  end subroutine

  subroutine observe_selected_runtime(step,time,catalog_changed,external_sample)
    use commvar, only: hm,use_gpu,const2,const6,lreadgrid
    use commarray, only: q
#ifdef _CUDA
    use commarray_gpu, only: q_d
#endif
    integer(int64),intent(in) :: step
    real(real_kind),intent(in) :: time
    logical,intent(in) :: catalog_changed
    procedure(selected_external_sampler),optional :: external_sample
    type(catalog_product),allocatable :: definitions(:)
    logical,allocatable :: active(:)
    type(runtime_product_state) :: state
    logical :: ok,external_ok
    integer :: p,n,g
    real(8) :: started
    character(256) :: geometry_error
    if(.not.series%configured) return
    if(lreadgrid) then
      if(geometry_rejected.and..not.catalog_changed) return
      call inspect_selected_cartesian(ok,geometry_error)
      if(.not.ok) then
        call disable_selected(step,time,trim(geometry_error))
        geometry_rejected=.true.;return
      endif
    endif
    started=insitu_clock()
    if(first_observation.or.catalog_changed) then
      n=runtime_catalog_count();allocate(definitions(n),active(n))
      do p=1,n
        call runtime_catalog_state(p,state)
        definitions(p)=state%definition
        active(p)=state%status==runtime_active.or.state%status==runtime_waiting
      enddo
      call series%reconcile(definitions,active,ok,approve=approve_allocation,agree=collective_ok)
      if(.not.collective_ok(ok)) then
        call disable_selected(step,time,'cannot allocate requested selected-time statistics within configured resources')
        return
      endif
      do p=1,size(series%requests)
        if(series%requests(p)%resource_failed) call runtime_catalog_disable(p,step,real(time,kind(0.d0)), &
          'cannot allocate this selected-time support within configured resources')
      enddo
      first_observation=.false.
    endif
    call refresh_selected_runtime()
#ifdef _CUDA
    if(use_gpu) then
      call series%sample_device(step,time,hm,q_d,const2,const6,ok)
    else
#endif
      call series%sample_host(step,time,hm,q,const2,const6,ok)
#ifdef _CUDA
    endif
#endif
    if(present(external_sample)) then
      call series%sample_external(step,time,external_sample,external_ok)
      ok=ok.and.external_ok
    endif
    if(.not.collective_ok(ok)) then
      do g=1,size(series%groups)
        if(collective_ok(.not.series%groups(g)%failed)) cycle
        series%groups(g)%failed=.true.;series%groups(g)%continuous=.false.
        do p=1,size(series%requests)
          if(series%requests(p)%group/=g) cycle
          call runtime_catalog_disable(p,step,real(time,kind(0.d0)), &
            'this selected-time sample failed; its accumulated state is not published')
        enddo
      enddo
    endif
    call report_insitu_timing('selected_statistics_sample',started,step)
  end subroutine

  subroutine disable_selected(step,time,message)
    integer(int64),intent(in) :: step
    real(real_kind),intent(in) :: time
    character(*),intent(in) :: message
    integer :: p
    type(runtime_product_state) :: state
    do p=1,runtime_catalog_count()
      call runtime_catalog_state(p,state)
      if(state%version<=0) cycle
      if(state%definition%statistics_source/='selected-time') cycle
      call runtime_catalog_disable(p,step,real(time,kind(0.d0)),message)
    enddo
    first_observation=.true.
    ! Failed groups cannot later be published or reused as an intact history.
    series%groups%failed=.true.;series%groups%active=.false.
  end subroutine

  logical function collective_ok(ok) result(valid)
    logical,intent(in) :: ok
    integer :: bad,any_bad,err,ignored
    bad=merge(0,1,ok)
    call MPI_Allreduce(bad,any_bad,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,err)
    if(err/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,err,ignored)
    valid=any_bad==0
  end function

  subroutine approve_allocation(host_bytes,device_bytes,additional_device,ok)
    integer(int64),intent(in) :: host_bytes,device_bytes,additional_device
    logical,intent(out) :: ok
    integer :: node,nr,err,r
    integer(int64),allocatable :: hosts(:)
    integer(int64) :: total_host
#ifdef _CUDA
    integer :: device,status
    integer(c_size_t) :: free,total
    character(kind=c_char,len=32) :: pci
    character(kind=c_char,len=32),allocatable :: identities(:)
    integer(int64),allocatable :: devices(:),increments(:),available(:)
    integer(int64) :: total_device,total_increment,least_free
#endif
    call MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,node,err)
    call MPI_Comm_size(node,nr,err)
    allocate(hosts(nr));call MPI_Allgather(host_bytes,1,MPI_INTEGER8,hosts,1,MPI_INTEGER8,node,err)
    total_host=sum(hosts);ok=total_host<=host_limit
#ifdef _CUDA
    if(series%on_device) then
      status=cudaGetDevice(device);ok=ok.and.status==cudaSuccess
      pci=repeat(achar(0,kind=c_char),32)
      status=int(pci_bus_id(pci,32_c_int,int(device,c_int)));ok=ok.and.status==cudaSuccess
      status=cudaMemGetInfo(free,total);ok=ok.and.status==cudaSuccess
      allocate(identities(nr),devices(nr),increments(nr),available(nr))
      call MPI_Allgather(pci,32,MPI_CHARACTER,identities,32,MPI_CHARACTER,node,err)
      call MPI_Allgather(device_bytes,1,MPI_INTEGER8,devices,1,MPI_INTEGER8,node,err)
      call MPI_Allgather(additional_device,1,MPI_INTEGER8,increments,1,MPI_INTEGER8,node,err)
      least_free=int(free,int64)
      call MPI_Allgather(least_free,1,MPI_INTEGER8,available,1,MPI_INTEGER8,node,err)
      total_device=0;total_increment=0
      do r=1,nr
        if(identities(r)/=pci) cycle
        total_device=total_device+devices(r);total_increment=total_increment+increments(r)
        least_free=min(least_free,available(r))
      enddo
      ok=ok.and.total_device<=device_limit.and.least_free>=device_reserve
      if(ok) ok=total_increment<=least_free-device_reserve
    endif
#endif
    call MPI_Comm_free(node,err)
    ok=collective_ok(ok)
  end subroutine

  subroutine selected_runtime_info(definition,source,ready,ok,window,measure,count,message)
    type(catalog_product),intent(in) :: definition
    integer,intent(out) :: source
    logical,intent(out) :: ready,ok
    real(real_kind),intent(out) :: window(2),measure
    integer(int64),intent(out) :: count
    character(*),intent(out) :: message
    integer :: g
    source=0;ready=.false.;ok=.false.;window=0;measure=0;count=0
    message='selected-time requires enabled catalog_stats and a supported selected mean product'
    if(.not.series%configured) return
    g=series%find(definition%id)
    if(g==0) return
    if(series%groups(g)%failed) then
      message='selected-time history failed and cannot be published';return
    endif
    source=merge(6,5,definition%source=='favre');selected_group=g
    window=[max(series%window(1),series%groups(g)%first_time),min(series%window(2),series%groups(g)%last_time)]
    measure=series%groups(g)%covered;ready=measure>0;ok=.true.;message=''
#ifdef _CUDA
    if(series%on_device.and.ready) call activate_selected_moments_gpu(series%groups(g)%device, &
      series%groups(g)%field_ids(:series%groups(g)%nfields),series%cells,size(series%groups(g)%keys))
#endif
  end subroutine

  real(real_kind) function selected_runtime_value(i,j,k,field,favre) result(value)
    use ieee_arithmetic, only: ieee_value,ieee_quiet_nan
    integer,intent(in) :: i,j,k,field
    logical,intent(in) :: favre
    logical :: ok
    call series%value(selected_group,[i,j,k],field,favre,value,ok)
    if(.not.ok) value=ieee_value(0.e0_real_kind,ieee_quiet_nan)
  end function

  subroutine selected_runtime_sampling_info(id,requested,segments)
    character(*),intent(in) :: id
    real(real_kind),intent(out) :: requested(2)
    integer(int64),intent(out) :: segments
    integer :: g
    g=series%find(id);requested=series%window;segments=series%groups(g)%segments
  end subroutine

  subroutine release_selected_runtime()
    integer(int64) :: host_bytes,device_bytes,calls,status_bytes,index_bytes
    integer :: rank,err
    if(series%configured) then
      call series%bytes(host_bytes,device_bytes)
      call MPI_Comm_rank(MPI_COMM_WORLD,rank,err)
      print '(a,i0,a,i0,a,i0,a,i0)', 'ASTR_INSITU_SELECTED rank=',rank, &
        ' histories=',size(series%groups),' host_state_and_index_bytes=',host_bytes,' device_state_and_index_bytes=',device_bytes
#ifdef _CUDA
      if(series%on_device) then
        call selected_moment_transfer_counts_gpu(calls,status_bytes,index_bytes)
        print '(a,i0,a,i0,a,i0,a,i0,a)', 'ASTR_INSITU_SELECTED_SAMPLER rank=',rank,' kernel_calls=',calls, &
          ' status_d2h_bytes=',status_bytes,' index_h2d_bytes=',index_bytes,' flow_d2h_bytes=0'
      endif
#endif
    endif
    call series%release();selected_group=0;first_observation=.true.
    geometry_checked=.false.;geometry_valid=.false.;geometry_rejected=.false.;geometry_direction=0;geometry_message=''
    if(allocated(active_consumers)) deallocate(active_consumers)
  end subroutine
end module
