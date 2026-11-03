program sample_moments_probe
  use mpi
  use astr_precision, only: real_kind
  use insitu_time_integral, only: clipped_trapezoid
  use insitu_velocity_statistics
  use insitu_sample_moments
  use ieee_arithmetic, only: ieee_value,ieee_quiet_nan,ieee_is_finite
#ifdef _CUDA
  use cudafor
  use insitu_sample_moments_gpu, only: push_selected_moments_gpu,gather_selected_moments_gpu
  use insitu_sample_moments_gpu, only: selected_moment_device,allocate_selected_moments_gpu, &
    release_selected_moments_gpu,reduce_selected_profile_gpu,push_scattered_moments_gpu
#endif
  implicit none
  integer,parameter :: nodes=19,nfields=5
  integer :: rank,ranks,ierr,step,node,nstate
  real(real_kind),parameter :: times(3)=[0.e0_real_kind,1.e0_real_kind,2.e0_real_kind]
  real(real_kind) :: weights(2),duration,mean_rho,mean_r(5),mean_f(5),worst,difference,reference_error,rms_error
  real(real_kind) :: actual_values(35),reference_values(35)
  real(real_kind) :: actual_scalars(4),reference_scalars(4)
  real(real_kind) :: q_selected(5),decoded(5),decoded_rho,decoded_state(30),velocity_state(6)
  real(real_kind) :: q_grid(0:3,0:2,0:2,5),nan
  integer :: indices(3,nodes),ci,cj,ck
  real(real_kind) :: fields(nodes,nfields),rho(nodes),small_fields(2),small_r(2),small_f(2)
  real(real_kind) :: scalar_state(9),resumed(30),saved(30),bad_fields(5)
  real(real_kind),allocatable :: states(:,:)
  type(velocity_statistics) :: reference(nodes)
  type(velocity_statistics_result) :: actual,expected,device_result
  logical :: ok,mpi_enabled
  character(32) :: mode
#ifdef _CUDA
  real(real_kind),allocatable,device :: states_d(:,:),fields_d(:,:),rho_d(:)
  real(real_kind),allocatable,device :: scalar_states_d(:,:),scalar_fields_d(:,:)
  real(real_kind),allocatable,device :: q_d(:,:,:,:),gathered_states_d(:,:),gathered_scalar_d(:,:),velocity_states_d(:,:)
  integer,allocatable,device :: indices_d(:,:)
  real(real_kind),allocatable :: observed(:,:)
  real(real_kind) :: scalar_observed(nodes,9)
  integer,allocatable,device :: invalid_d(:)
  integer :: device_count,status,bound_device
#endif
  call get_command_argument(1,mode)
  if(len_trim(mode)>0.and.trim(mode)/='--kernel-only') error stop 'unknown moment probe mode'
  mpi_enabled=trim(mode)/='--kernel-only';rank=0;ranks=1
  if(mpi_enabled) then
    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
    call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  endif
  nstate=sample_moment_size(nfields,.true.)
  call require(nstate==30,'compact full primitive layout')
  call require(sample_moment_size(3,.true.)==24,'velocity-only layout')
  call require(sample_moment_size(2,.false.)==9,'scalar-only layout')
  allocate(states(nodes,nstate));states=0
  scalar_state=0;decoded_state=0;velocity_state=0;worst=0;reference_error=0;rms_error=0
  nan=ieee_value(0.e0_real_kind,ieee_quiet_nan)
  do node=1,nodes
    indices(:,node)=[mod(node-1,4),mod((node-1)/4,3),(node-1)/12]
  enddo
  do node=1,nodes
    call configure_velocity_statistics(reference(node),.25e0_real_kind,1.75e0_real_kind,ok)
    call require(ok,'reference configuration')
  enddo
#ifdef _CUDA
  status=cudaGetDeviceCount(device_count)
  call require(status==cudaSuccess.and.device_count>0,'CUDA device availability')
  status=cudaSetDevice(mod(rank,device_count))
  call require(status==cudaSuccess,'CUDA device binding')
  status=cudaGetDevice(bound_device)
  call require(status==cudaSuccess.and.bound_device==mod(rank,device_count),'CUDA bound device identity')
  allocate(states_d(nodes,nstate),fields_d(nodes,nfields),rho_d(nodes),observed(nodes,nstate),invalid_d(1))
  allocate(scalar_states_d(nodes,9),scalar_fields_d(nodes,2))
  allocate(q_d(0:3,0:2,0:2,5),gathered_states_d(nodes,nstate),gathered_scalar_d(nodes,9), &
    velocity_states_d(nodes,6),indices_d(3,nodes))
  gathered_states_d=0;gathered_scalar_d=0;velocity_states_d=0;indices_d=indices
  states_d=0;scalar_states_d=0
#endif
  do step=1,3
    call sample_values(step)
    q_grid=nan
    weights=0
    if(step>1) then
      call clipped_trapezoid(times(step-1),times(step),[1.e0_real_kind,0.e0_real_kind], &
        [0.e0_real_kind,1.e0_real_kind],.25e0_real_kind,1.75e0_real_kind,weights,duration,ok)
      call require(ok,'clipped endpoint weights')
    endif
    do node=1,nodes
      call push_sample_moments(states(node,:),fields(node,:),rho(node),weights,step>1,[1,2,3],ok)
      call require(ok,'compact sample update')
      call push_velocity_sample(reference(node),times(step),rho(node),fields(node,1:3),ok)
      call require(ok,'independent velocity update')
      q_selected=[rho(node),rho(node)*fields(node,1:3), &
        2.5e0_real_kind*fields(node,4)+0.5e0_real_kind*rho(node)*sum(fields(node,1:3)**2)]
      call decode_sample_primitives(q_selected,nfields,[2,3,4,5,6],3.e0_real_kind,2.5e0_real_kind,decoded,decoded_rho,ok)
      call require(ok.and.maxval(abs(decoded-fields(node,:)))<=2.e-10_real_kind,'authoritative primitive decoder')
      reference_error=max(reference_error,maxval(abs(decoded-fields(node,:))/ &
        max(1.e0_real_kind,abs(decoded),abs(fields(node,:)))))
      if(node==1) then
        call push_sample_moments(decoded_state,decoded,decoded_rho,weights,step>1,[1,2,3],ok)
        call require(ok.and.maxval(abs(decoded_state-states(1,:)))<=2.e-10_real_kind,'decoded moment reference')
        call push_sample_moments(velocity_state,fields(1,1:1),rho(1),weights,step>1,[0,0,0],ok)
        call require(ok,'velocity subset reference')
      endif
      ci=indices(1,node);cj=indices(2,node);ck=indices(3,node)
      q_grid(ci,cj,ck,:)=q_selected
    enddo
    small_fields=fields(1,4:5)
    call push_sample_moments(scalar_state,small_fields,rho(1),weights,step>1,[0,0,0],ok)
    call require(ok,'scalar-only sample update')
    if(step==2) resumed=states(1,:)
    if(step==3) then
      call push_sample_moments(resumed,fields(1,:),rho(1),weights,.true.,[1,2,3],ok)
      call require(ok.and.all(resumed==states(1,:)),'exact raw-state continuation')
    endif
#ifdef _CUDA
    fields_d=fields;rho_d=rho
    call push_selected_moments_gpu(states_d,fields_d,rho_d,weights,step>1,[1,2,3],invalid_d,ok)
    call require(ok,'device selected sample update')
    observed=states_d
    call require(all(ieee_is_finite(observed)),'finite device moments')
    difference=maxval(abs(observed-states)/max(1.e0_real_kind,abs(observed),abs(states)))
    worst=max(worst,difference)
    call require(difference<=2.e-10_real_kind,'CPU/GPU selected-state agreement')
    scalar_fields_d=fields(:,4:5)
    call push_selected_moments_gpu(scalar_states_d,scalar_fields_d,rho_d,weights,step>1, &
      [0,0,0],invalid_d,ok)
    call require(ok,'device scalar-only sample update')
    scalar_observed=scalar_states_d
    difference=maxval(abs(scalar_observed(1,:)-scalar_state)/ &
      max(1.e0_real_kind,abs(scalar_observed(1,:)),abs(scalar_state)))
    worst=max(worst,difference)
    call require(difference<=2.e-10_real_kind,'CPU/GPU scalar subset agreement')
    q_d=q_grid
    call gather_selected_moments_gpu(q_d,3,2,2,0,indices_d,[2,3,4,5,6],3.e0_real_kind,2.5e0_real_kind, &
      gathered_states_d,weights,step>1,.true.,invalid_d,ok)
    call require(ok,'authoritative device gather')
    observed=gathered_states_d
    difference=maxval(abs(observed-states)/max(1.e0_real_kind,abs(observed),abs(states)))
    worst=max(worst,difference)
    call require(difference<=2.e-10_real_kind,'indexed primitive moment agreement')
    call gather_selected_moments_gpu(q_d,3,2,2,0,indices_d,[5,6],3.e0_real_kind,2.5e0_real_kind, &
      gathered_scalar_d,weights,step>1,.false.,invalid_d,ok)
    call require(ok,'indexed scalar gather')
    scalar_observed=gathered_scalar_d
    difference=maxval(abs(scalar_observed(1,:)-scalar_state)/ &
      max(1.e0_real_kind,abs(scalar_observed(1,:)),abs(scalar_state)))
    worst=max(worst,difference)
    call require(difference<=2.e-10_real_kind,'indexed scalar means')
    q_d(:,:,:,3:5)=nan
    call gather_selected_moments_gpu(q_d,3,2,2,0,indices_d,[2],nan,nan, &
      velocity_states_d,weights,step>1,.false.,invalid_d,ok)
    call require(ok,'unrequested energy and velocity components are not read')
    observed(:,1:6)=velocity_states_d
    difference=maxval(abs(observed(1,1:6)-velocity_state)/ &
      max(1.e0_real_kind,abs(observed(1,1:6)),abs(velocity_state)))
    worst=max(worst,difference)
    call require(difference<=2.e-10_real_kind,'indexed single-field means')
#endif
  enddo
#ifdef _CUDA
  observed=gathered_states_d
#endif
  do node=1,nodes
    call read_sample_velocity(states(node,:),nfields,[1,2,3],actual,ok)
    call require(ok,'read compact velocity statistics')
    call read_velocity_statistics(reference(node),expected,ok)
    call require(ok,'read independent velocity statistics')
    rms_error=max(rms_error,maxval(abs(actual%rms_r-expected%rms_r)),maxval(abs(actual%rms_f-expected%rms_f)))
#ifdef _CUDA
    call read_sample_velocity(observed(node,:),nfields,[1,2,3],device_result,ok)
    call require(ok,'read indexed device velocity statistics')
    rms_error=max(rms_error,maxval(abs(actual%rms_r-device_result%rms_r)), &
      maxval(abs(actual%rms_f-device_result%rms_f)))
#endif
    call require(abs(actual%duration-expected%duration)<=2.e-10_real_kind,'covered duration')
    call require(maxval(abs(actual%mean_r-expected%mean_r))<=2.e-10_real_kind,'Reynolds mean')
    call require(maxval(abs(actual%mean_f-expected%mean_f))<=2.e-10_real_kind,'Favre mean')
    call require(maxval(abs(actual%covariance_r-expected%covariance_r))<=2.e-10_real_kind,'Reynolds covariance')
    call require(maxval(abs(actual%covariance_f-expected%covariance_f))<=2.e-10_real_kind,'Favre covariance')
    call require(maxval(abs(actual%density_stress-expected%density_stress))<=2.e-10_real_kind,'density stress')
    actual_values=[actual%duration,actual%mean_density,actual%mean_r,actual%mean_f, &
      reshape(actual%covariance_r,[9]),reshape(actual%covariance_f,[9]),reshape(actual%density_stress,[9])]
    reference_values=[expected%duration,expected%mean_density,expected%mean_r,expected%mean_f, &
      reshape(expected%covariance_r,[9]),reshape(expected%covariance_f,[9]),reshape(expected%density_stress,[9])]
    reference_error=max(reference_error,maxval(abs(actual_values-reference_values)/ &
      max(1.e0_real_kind,abs(actual_values),abs(reference_values))))
  enddo
  call read_sample_means(states(1,:),nfields,mean_rho,mean_r,mean_f,ok)
  call require(ok,'read primitive means')
  call require(abs(mean_r(4)-218.75e0_real_kind)<=2.e-10_real_kind,'analytic pressure mean')
  call require(abs(mean_f(4)-246.875e0_real_kind)<=2.e-10_real_kind,'analytic density-weighted pressure')
  call require(abs(mean_r(5)-318.75e0_real_kind)<=2.e-10_real_kind,'analytic temperature mean')
  call require(abs(mean_f(5)-328.125e0_real_kind)<=2.e-10_real_kind,'analytic density-weighted temperature')
  actual_scalars=[mean_r(4),mean_f(4),mean_r(5),mean_f(5)]
  reference_scalars=[218.75e0_real_kind,246.875e0_real_kind,318.75e0_real_kind,328.125e0_real_kind]
  reference_error=max(reference_error,maxval(abs(actual_scalars-reference_scalars)/ &
    max(1.e0_real_kind,abs(actual_scalars),abs(reference_scalars))))
  call read_sample_means(scalar_state,2,mean_rho,small_r,small_f,ok)
  call require(ok.and.all(small_r==mean_r(4:5)).and.all(small_f==mean_f(4:5)),'scalar subset equivalence')
  call check_profile_projection()
#ifdef _CUDA
  call check_scattered_updates()
#endif
  saved=states(1,:);bad_fields=fields(1,:)
  bad_fields(5)=ieee_value(0.e0_real_kind,ieee_quiet_nan)
  call push_sample_moments(states(1,:),bad_fields,rho(1),weights,.true.,[1,2,3],ok)
  call require(.not.ok.and.all(states(1,:)==saved),'failed update preserves prior state')
#ifdef _CUDA
  fields(1,5)=bad_fields(5);fields_d=fields
  call push_selected_moments_gpu(states_d,fields_d,rho_d,[0.e0_real_kind,0.e0_real_kind], &
    .true.,[1,2,3],invalid_d,ok)
  call require(.not.ok,'device nonfinite sample rejection')
  observed=states_d
  call require(all(observed(1,:)==saved),'failed device node preserves prior state')
  call sample_values(3)
  fields(nodes,5)=bad_fields(5);fields_d=fields
  call push_selected_moments_gpu(states_d,fields_d,rho_d,[0.e0_real_kind,0.e0_real_kind], &
    .true.,[1,2,3],invalid_d,ok,sample_count=nodes-1)
  call require(ok,'unused capacity is not sampled')
  observed=states_d
  call require(all(observed==states),'zero-weight prefix preserves moments')
  call push_selected_moments_gpu(states_d,fields_d,rho_d,weights, &
    .true.,[1,2,3],invalid_d,ok,sample_count=0)
  call require(ok,'empty device selection')
  observed=states_d
  call require(all(observed==states),'empty selection preserves prior moments')
#endif
  write(*,'(a,i0,a,es16.8,a,es16.8)') 'PASS compact primitive moments rank=',rank, &
    ' reference_error=',reference_error,' raw_rms_error=',rms_error
#ifdef _CUDA
  write(*,'(a,i0,a,i0,a,es16.8)') 'PASS device compact moments rank=',rank, &
    ' device=',bound_device,' max_scaled_error=',worst
  deallocate(states_d,fields_d,rho_d,observed,scalar_states_d,scalar_fields_d,invalid_d)
  deallocate(q_d,gathered_states_d,gathered_scalar_d,velocity_states_d,indices_d)
#endif
  if(mpi_enabled) call MPI_Finalize(ierr)
contains
#ifdef _CUDA
  subroutine check_scattered_updates()
    type(selected_moment_device) :: bank
    real(real_kind) :: expected(9,9),fields(2),density,w(2),actual(9,9)
    real(real_kind),allocatable :: packet(:,:)
    real(real_kind),allocatable,device :: packet_d(:,:)
    integer :: idx(3,9),locations(9),part,first,last,s,p,row
    logical :: valid
    locations=[8,2,4,9,1,3,5,6,7];idx=0;idx(1,:)=[(p-1,p=1,9)];expected=0
    call allocate_selected_moments_gpu(bank,idx,9,valid)
    call require(valid,'scatter state allocation')
    do s=1,2
      w=0
      if(s==2) w=.5e0_real_kind
      do part=1,2
        first=1;last=5
        if(part==2) then
          first=6;last=9
        endif
        allocate(packet(last-first+1,3),packet_d(last-first+1,3))
        do row=first,last
          p=locations(row);density=real(p+s,real_kind)
          fields=[real(p*s,real_kind),real(p-s,real_kind)]
          packet(row-first+1,:)=[density,fields]
          call push_sample_moments(expected(p,:),fields,density,w,s>1,[0,0,0],valid)
          call require(valid,'scatter CPU reference')
        enddo
        packet_d=packet
        call push_scattered_moments_gpu(bank,c_devloc(packet_d),locations(first:last),2,w,s>1,valid)
        call require(valid,'borrowed device packet and sparse statistical targets')
        deallocate(packet,packet_d)
      enddo
    enddo
    actual=bank%states
    difference=maxval(abs(actual-expected)/max(1.e0_real_kind,abs(actual),abs(expected)))
    worst=max(worst,difference)
    call require(difference<=2.e-10_real_kind,'scattered CPU/GPU endpoint moments')
    allocate(packet_d(1,3));packet_d=1
    call push_scattered_moments_gpu(bank,c_devloc(packet_d),[0],2,w,.true.,valid)
    call require(.not.valid,'invalid scatter target is rejected without an out-of-bounds access')
    actual=bank%states
    call require(all(actual==expected),'invalid scatter preserves existing target states')
    deallocate(packet_d)
    call release_selected_moments_gpu(bank)
    print '(a,i0)','PASS scattered moment updates rank=',rank
  end subroutine
#endif

  subroutine check_profile_projection()
    integer,parameter :: ny=3,nk=5,np=ny*nk
    type(velocity_statistics) :: point(np)
    type(velocity_statistics_result) :: result
    real(real_kind) :: compact(np,30),state(30),v(5),density,w(2),covered,lengths(nk),terms(24), &
      projected(24,ny),reference_terms(24,ny),zterms(24),rmean(5),fmean(5),cr(6),cf(6),stress(6), &
      rr(3),rf(3),measure,rho_mean,temporal_weights(3),snapshots(5,3),densities(3),mass
    integer :: jk(3,np),j,k,p,s,a,b,c
    integer,parameter :: pairs(2,6)=reshape([1,1,1,2,2,2,1,3,2,3,3,3],[2,6])
    logical :: valid
#ifdef _CUDA
    type(selected_moment_device) :: bank,empty
    real(real_kind),allocatable :: reduced(:,:)
#endif
    compact=0;projected=0;reference_terms=0
    lengths=[.125e0_real_kind,.5e0_real_kind,.875e0_real_kind,1.5e0_real_kind,1.e0_real_kind]
    temporal_weights=[.28125e0_real_kind,.9375e0_real_kind,.28125e0_real_kind]
    do k=1,nk
    do j=1,ny
      p=(k-1)*ny+j;jk(:,p)=[0,j,k]
      call configure_velocity_statistics(point(p),.25e0_real_kind,1.75e0_real_kind,valid)
      call require(valid,'profile reference setup')
      state=0
      do s=1,3
        density=.25e0_real_kind+real(j*k*s,real_kind)
        v=[real(j+3*k+s*s,real_kind),real(k-2*j*s,real_kind),real(2*j+k*s,real_kind), &
          real(200+j*k+7*s,real_kind),real(300+2*j+s*k,real_kind)]
        snapshots(:,s)=v;densities(s)=density
        w=0
        if(s>1) then
          call clipped_trapezoid(times(s-1),times(s),[1.e0_real_kind,0.e0_real_kind], &
            [0.e0_real_kind,1.e0_real_kind],.25e0_real_kind,1.75e0_real_kind,w,covered,valid)
          call require(valid,'profile clipped weights')
        endif
        call push_sample_moments(state,v,density,w,s>1,[1,2,3],valid)
        call require(valid,'profile compact sample')
        call push_velocity_sample(point(p),times(s),density,v(:3),valid)
        call require(valid,'profile independent sample')
      enddo
      compact(p,:)=state
      call sample_profile_terms(30,5,state,lengths(k),.true.,terms,valid)
      call require(valid,'profile weighted projection')
      projected(:,j)=projected(:,j)+terms
      call read_velocity_statistics(point(p),result,valid)
      call require(valid,'profile independent moments')
      mass=sum(temporal_weights*densities)
      zterms=0;zterms(1)=lengths(k);zterms(2)=lengths(k)*result%mean_density
      do c=1,5
        zterms(2+c)=lengths(k)*sum(temporal_weights*snapshots(c,:))/sum(temporal_weights)
        zterms(7+c)=lengths(k)*result%mean_density*sum(temporal_weights*densities*snapshots(c,:))/mass
      enddo
      do c=1,6
        a=pairs(1,c);b=pairs(2,c)
        zterms(12+c)=lengths(k)*result%covariance_r(a,b)
        zterms(18+c)=lengths(k)*result%density_stress(a,b)
      enddo
      reference_terms(:,j)=reference_terms(:,j)+zterms
    enddo
    enddo
    call require(maxval(abs(projected-reference_terms)/ &
      max(1.e0_real_kind,abs(projected),abs(reference_terms)))<=2.e-10_real_kind,'independent span reference')
    do j=1,ny
      call finish_sample_profile(5,.true.,projected(:,j),measure,rho_mean,rmean,fmean,cr,cf,stress,rr,rf,valid)
      call require(valid.and.measure==sum(lengths),'profile normalization')
      call require(maxval(abs(rr**2-cr([1,3,6])))<=2.e-10_real_kind,'profile RMS variance')
      call require(maxval(abs(cf-stress/rho_mean))<=2.e-10_real_kind,'profile Favre density consistency')
    enddo
#ifdef _CUDA
    call allocate_selected_moments_gpu(bank,jk,30,valid)
    call require(valid,'profile device workspace')
    bank%states=compact
    call reduce_selected_profile_gpu(bank,ny,lengths,5,.true.,reduced,valid)
    call require(valid,'profile device reduction')
    difference=maxval(abs(reduced-reference_terms)/max(1.e0_real_kind,abs(reduced),abs(reference_terms)))
    worst=max(worst,difference)
    call require(difference<=2.e-10_real_kind,'profile CPU/GPU reduction')
    call reduce_selected_profile_gpu(bank,ny,lengths,5,.true.,reduced,valid)
    call require(valid,'profile cached workspace reuse')
    call release_selected_moments_gpu(bank)
    call reduce_selected_profile_gpu(empty,0,[real(real_kind)::],5,.true.,reduced,valid)
    call require(valid.and.size(reduced)==0,'profile empty owner')
#endif
    print '(a,i0)','PASS station profile projection rank=',rank
  end subroutine

  subroutine sample_values(e)
    integer,intent(in) :: e
    integer :: m
    real(real_kind) :: velocity(3,3),pressure(3)
    velocity=reshape([1.e0_real_kind,0.e0_real_kind,-1.e0_real_kind, &
      3.e0_real_kind,2.e0_real_kind,1.e0_real_kind,5.e0_real_kind,-2.e0_real_kind,4.e0_real_kind],[3,3])
    pressure=[100.e0_real_kind,200.e0_real_kind,400.e0_real_kind]
    do m=1,nodes
      rho(m)=real(e,real_kind)
      fields(m,1:3)=velocity(:,e)+real(m-1+rank,real_kind)*.125e0_real_kind
      fields(m,4)=pressure(e)
      fields(m,5)=3.e0_real_kind*pressure(e)/rho(m)
    enddo
  end subroutine

  subroutine require(condition,message)
    logical,intent(in) :: condition
    character(*),intent(in) :: message
    if(condition) return
    write(*,'(a)') 'FAIL sample moments: '//message
    if(mpi_enabled) call MPI_Abort(MPI_COMM_WORLD,1,ierr)
    error stop 1
  end subroutine
end program
