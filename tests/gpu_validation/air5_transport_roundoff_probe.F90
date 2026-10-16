program air5_transport_roundoff_probe
  use iso_fortran_env, only: real64
  use chemistry_air5_data, only: air5_ru,air5_molar_mass
  use ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  use chemistry_model, only: air5_transport_species_ratio,air5_transport_roundoff_gamma
  use chemistry_model, only: chemistry_status_ok, &
    chemistry_status_infeasible_flux,chemistry_status_nonfinite
#ifdef AIR5_FLUX_GPU_PROBE
  use air5_flux_gpu_probe_wrappers, only: air5_close_species_flux,air5_close_species_flux_moment, &
    air5_preserve_zero_flux_support
#else
  use chemistry_model, only: air5_close_species_flux,air5_close_species_flux_moment, &
    air5_preserve_zero_flux_support
#endif
  implicit none
  real(real64) :: base,negative,operand_scale,theta,scaled_theta,factor
  integer :: power
  real(real64) :: high(5),lower(5),upper(5),closed(5),permuted(5),mass
  integer :: status,permutation(5),i,j,k,l,m,trial
  real(real64) :: witness(5),gradient(5),tolerance
  real(real64) :: gas(5),moment
  real(real64) :: support(5,6)
  character(4096) :: input_path,output_path
  integer :: unit

  if(command_argument_count()==2) then
    call get_command_argument(1,input_path)
    call get_command_argument(2,output_path)
    open(newunit=unit,file=trim(input_path),status='old',action='read')
    read(unit,*) mass
    read(unit,*) high
    read(unit,*) lower
    read(unit,*) upper
    close(unit)
    call air5_close_species_flux(high,lower,upper,mass,closed,status)
    open(newunit=unit,file=trim(output_path),status='new',action='write')
    write(unit,'(I0,1X,5ES25.17)') status,closed
    close(unit)
    stop
  elseif(command_argument_count()/=0) then
    error stop 'requires no arguments or a captured projection input/output pair'
  endif

  base=1.0e-38_real64
  negative=-base
  operand_scale=16.0_real64*base
  theta=air5_transport_species_ratio(base,negative,operand_scale)
  if(theta<=0.0_real64 .or. theta>=1.0_real64) error stop 'ratio must reserve roundoff'
  if(base+theta*negative<air5_transport_roundoff_gamma*operand_scale) &
    error stop 'insufficient cancellation reserve'
  do power=-100,100,100
    factor=scale(1.0_real64,power)
    scaled_theta=air5_transport_species_ratio(base*factor,negative*factor,operand_scale*factor)
    if(scaled_theta/=theta) error stop 'ratio depends on absolute species magnitude'
  enddo
  if(air5_transport_species_ratio(0.0_real64,0.0_real64,0.0_real64)/=1.0_real64) &
    error stop 'exact zero must remain unmodified'
  if(air5_transport_species_ratio(0.0_real64,-base,base)/=0.0_real64) &
    error stop 'no negative budget allowed from zero'
  if(air5_transport_species_ratio(base,-0.1_real64*base,2.0_real64*base)/=1.0_real64) &
    error stop 'unlimited safe flux must remain unlimited'
  if(air5_transport_species_ratio(base*epsilon(base),-base,base)/=0.0_real64) &
    error stop 'unresolved residual cannot fund a correction'
  write(*,'(A)') 'AIR5_TRANSPORT_ROUNDOFF_PROBE pass'

  ! Only the trace candidate violates its interval. Do not flatten the O2 flux.
  high=[0.7_real64,0.3_real64,0.0_real64,0.0_real64,-1.0e-12_real64]
  lower=[0.0_real64,0.0_real64,0.0_real64,0.0_real64,0.0_real64]
  upper=[1.0_real64,1.0_real64,0.0_real64,0.0_real64,1.0e-10_real64]
  mass=sum(high)
  call air5_close_species_flux(high,lower,upper,mass,closed,status)
  if(status/=chemistry_status_ok) error stop 'feasible species flux rejected'
  if(any(closed<lower) .or. any(closed>upper)) error stop 'flux outside budgets'
  if(abs(sum(closed)-mass)>32*epsilon(mass)) error stop 'mass flux mismatch'
  if(abs(closed(2)-high(2))>2.0e-12_real64) error stop 'major species overlimited'
  if(closed(5)/=0.0_real64) error stop 'trace flux at lower bound expected'

  do i=1,5
    permutation=cshift([1,2,3,4,5],i)
    call air5_close_species_flux(high(permutation),lower(permutation),upper(permutation), &
      mass,permuted,status)
    if(status/=chemistry_status_ok) error stop 'permuted flux rejected'
    if(maxval(abs(permuted-closed(permutation)))>32*epsilon(mass)) &
      error stop 'species ordering changes closure'
  enddo
  call air5_close_species_flux(closed,lower,upper,mass,permuted,status)
  if(status/=chemistry_status_ok .or. any(permuted/=closed)) &
    error stop 'already closed candidate changed'
  call air5_close_species_flux(high,lower,upper,3.0_real64,permuted,status)
  if(status/=chemistry_status_infeasible_flux) error stop 'infeasible flux accepted'
  if(any(permuted/=high)) error stop 'failed projection must not overwrite candidate'
  do power=-100,100,100
    factor=scale(1.0_real64,power)
    call air5_close_species_flux(high*factor,lower*factor,upper*factor,mass*factor,permuted,status)
    if(status/=chemistry_status_ok) error stop 'scaled feasible flux rejected'
    if(maxval(abs(permuted/factor-closed))>32*epsilon(mass)) error stop 'flux scale dependence'
  enddo
  call air5_close_species_flux(-high,-upper,-lower,-mass,permuted,status)
  if(status/=chemistry_status_ok .or. maxval(abs(permuted+closed))>32*epsilon(mass)) &
    error stop 'face orientation changes closure'
  lower=0.0_real64; upper=0.0_real64; high=0.0_real64
  call air5_close_species_flux(high,lower,upper,0.0_real64,closed,status)
  if(status/=chemistry_status_ok .or. any(closed/=0.0_real64)) error stop 'zero flux failure'
  lower(1)=1.0_real64
  call air5_close_species_flux(high,lower,upper,0.0_real64,closed,status)
  if(status/=chemistry_status_infeasible_flux) error stop 'reversed interval accepted'
  lower=0.0_real64; high(1)=ieee_value(0.0_real64,ieee_quiet_nan)
  call air5_close_species_flux(high,lower,upper,0.0_real64,closed,status)
  if(status/=chemistry_status_nonfinite) error stop 'NaN candidate accepted'

  ! Captured physical-filter face: feasible zero mass flux with wide species budgets.
  high=[-6.77626357803440271e-20_real64,0.0_real64,1.36718750000000006e-21_real64, &
    -1.84114728110395408e-20_real64,0.0_real64]
  lower=[-1.91749997361107399e-2_real64,-5.82499999999999945e-3_real64, &
    -2.49999999999999994e-20_real64,-2.63889255825490058e-10_real64,0.0_real64]
  upper=[1.91749997404329110e-2_real64,5.82499999999999945e-3_real64, &
    2.49999999999999994e-20_real64,2.59567085809127194e-10_real64,0.0_real64]
  do power=-100,100,100
    factor=scale(1.0_real64,power)
    call air5_close_species_flux(high*factor,lower*factor,upper*factor,0.0_real64,closed,status)
    if(status/=chemistry_status_ok) error stop 'captured trace projection rejected'
    if(any(closed<lower*factor).or.any(closed>upper*factor)) &
      error stop 'captured trace projection outside budgets'
    operand_scale=maxval(abs([high,lower,upper]))*factor
    tolerance=32*epsilon(factor)*sum(abs(high*factor/operand_scale))
    if(abs(sum(closed/operand_scale))>tolerance) error stop 'captured trace mass closure'
    if(closed(5)/=0.0_real64) error stop 'captured trace created absent species'
    call air5_close_species_flux(-high*factor,-upper*factor,-lower*factor,0.0_real64,permuted,status)
    if(status/=chemistry_status_ok) error stop 'reversed captured trace projection rejected'
    if(maxval(abs(permuted+closed))>128*epsilon(factor)*maxval(abs(high*factor))) &
      error stop 'captured trace projection depends on face orientation'
  enddo
  write(*,'(A)') 'AIR5_TRACE_PROJECTION_CONTINUATION_PROBE pass'

  ! Captured T3-P3 face: normalized closure passes before rescaling fails.
  high=[-7.46456591169233065e-23_real64,-8.44728200031146220e-24_real64, &
    5.44962107913232811e-25_real64,3.93514023170964434e-24_real64,3.48157356024232533e-24_real64]
  lower=[-1.91749999942424122e-2_real64,-5.82499999816384929e-3_real64, &
    -1.02070959651466458e-21_real64,-8.71050892605313622e-14_real64,-3.04502846226511376e-21_real64]
  upper=[1.91749999942424122e-2_real64,5.82499999816384929e-3_real64, &
    1.01774712436516967e-21_real64,8.71050892653534985e-14_real64,3.04748413095627950e-21_real64]
  mass=-2.08161917387942470e-23_real64
  do power=-100,100,100
    factor=scale(1.0_real64,power)
    call air5_close_species_flux(high*factor,lower*factor,upper*factor,mass*factor,closed,status)
    if(status/=chemistry_status_ok) error stop 'rescaled captured projection rejected'
    if(any(closed<lower*factor).or.any(closed>upper*factor)) &
      error stop 'rescaled captured projection outside budgets'
    operand_scale=max(abs(mass),maxval(abs([high,lower,upper])))*factor
    tolerance=32*epsilon(factor)*max(abs(mass*factor/operand_scale),sum(abs(high*factor/operand_scale)))
    if(abs(sum(closed/operand_scale)-mass*factor/operand_scale)>tolerance) &
      error stop 'rescaled captured mass closure'
    call air5_close_species_flux(-high*factor,-upper*factor,-lower*factor,-mass*factor,permuted,status)
    if(status/=chemistry_status_ok) error stop 'reversed rescaled captured projection rejected'
    if(maxval(abs(permuted+closed))>128*epsilon(factor)*maxval(abs(high*factor))) &
      error stop 'rescaled captured projection depends on face orientation'
  enddo
  write(*,'(A)') 'AIR5_RESCALED_PROJECTION_CONTINUATION_PROBE pass'

  do trial=1,1000
    do i=1,5
      lower(i)=sin(real(13*trial+i,real64))
      upper(i)=lower(i)+abs(cos(real(7*trial+3*i,real64)))
      witness(i)=lower(i)+0.5_real64*(upper(i)-lower(i))
      high(i)=2.0_real64*sin(real(11*trial+7*i,real64))
    enddo
    mass=sum(witness)
    call air5_close_species_flux(high,lower,upper,mass,closed,status)
    if(status/=chemistry_status_ok) error stop 'feasible signed-flux case failed'
    tolerance=64*epsilon(mass)*max(1.0_real64,sum(abs(high)))
    if(abs(sum(closed)-mass)>tolerance .or. any(closed<lower) .or. any(closed>upper)) &
      error stop 'signed flux violates constraints'
    ! KKT conditions for every feasible transfer of flux from component j to i.
    gradient=closed-high
    do i=1,5
      do j=1,5
        if(closed(i)<upper(i)-tolerance .and. closed(j)>lower(j)+tolerance) then
          if(gradient(i)-gradient(j)<-tolerance) error stop 'projection is not minimal'
        endif
      enddo
    enddo
  enddo
  write(*,'(A)') 'AIR5_SPECIES_FLUX_CLOSURE_PROBE pass (mass and supplied bounds only)'

  gas=[297.0_real64,260.0_real64,594.0_real64,520.0_real64,277.0_real64]
  high=[0.7_real64,0.3_real64,0.0_real64,0.0_real64,-1.0e-12_real64]
  lower=0.0_real64
  upper=[1.0_real64,1.0_real64,0.0_real64,0.0_real64,1.0e-10_real64]
  mass=sum(high); moment=dot_product(gas,high)
  call air5_close_species_flux_moment(high,lower,upper,mass,gas,moment,closed,status)
  if(status/=chemistry_status_ok) error stop 'joint mass/moment flux rejected'
  if(any(closed<lower) .or. any(closed>upper)) error stop 'joint flux outside bounds'
  if(abs(sum(closed)-mass)>64*epsilon(mass)) error stop 'joint mass mismatch'
  if(abs(dot_product(gas,closed)-moment)>64*epsilon(moment)*moment) &
    error stop 'joint gas moment mismatch'
  if(maxval(abs(closed(1:2)-high(1:2)))>2.0e-12_real64) error stop 'joint flux overlimited'
  if(any(closed(3:5)/=0.0_real64)) error stop 'absent species or negative trace changed'
  do i=1,5; do j=1,5; do k=1,5; do l=1,5; do m=1,5
    permutation=[i,j,k,l,m]
    if(any([(count(permutation==trial)/=1,trial=1,5)])) cycle
    call air5_close_species_flux_moment(high(permutation),lower(permutation),upper(permutation), &
      mass,gas(permutation),moment,permuted,status)
    if(status/=chemistry_status_ok) error stop 'joint permuted flux rejected'
    if(maxval(abs(permuted-closed(permutation)))>64*epsilon(mass)) &
      error stop 'joint closure depends on species ordering'
  enddo; enddo; enddo; enddo; enddo
  call air5_close_species_flux_moment(-high,-upper,-lower,-mass,gas,-moment,permuted,status)
  if(status/=chemistry_status_ok .or. maxval(abs(permuted+closed))>64*epsilon(mass)) &
    error stop 'joint orientation dependence'
  do power=-100,100,100
    factor=scale(1.0_real64,power)
    call air5_close_species_flux_moment(high*factor,lower*factor,upper*factor,mass*factor, &
      gas,moment*factor,permuted,status)
    if(status/=chemistry_status_ok .or. maxval(abs(permuted/factor-closed))>64*epsilon(mass)) &
      error stop 'joint scale dependence'
  enddo
  call air5_close_species_flux_moment(high,lower,upper,mass,gas,1.0e3_real64,permuted,status)
  if(status/=chemistry_status_infeasible_flux .or. any(permuted/=high)) &
    error stop 'infeasible moment accepted or candidate overwritten'
  write(*,'(A)') 'AIR5_SPECIES_MOMENT_FLUX_PROBE pass (two affine constraints only)'
  gas=0.0_real64
  call air5_close_species_flux_moment(high,lower,upper,mass,gas,0.0_real64,closed,status)
  if(status/=chemistry_status_ok) error stop 'zero moment row rejected'
  call air5_close_species_flux_moment(high,lower,upper,mass,gas,1.0_real64,closed,status)
  if(status/=chemistry_status_infeasible_flux) error stop 'nonzero target for zero row accepted'
  gas=2.0_real64
  call air5_close_species_flux_moment(high,lower,upper,mass,gas,2*mass,closed,status)
  if(status/=chemistry_status_ok) error stop 'redundant moment constraint rejected'
  if(any(closed<lower) .or. any(closed>upper)) error stop 'redundant row violates bounds'
  call air5_close_species_flux_moment(high,lower,upper,mass,gas, &
    ieee_value(0.0_real64,ieee_quiet_nan),closed,status)
  if(status/=chemistry_status_nonfinite) error stop 'nonfinite moment accepted'
  write(*,'(A)') 'AIR5_DEGENERATE_MOMENT_PROBE pass'
  ! Recorded transverse LLF cancellation at contact i=25,j=0,k=2.
  gas=air5_ru/air5_molar_mass
  high=[-6.61744490042422140e-24_real64,-9.92616735063633210e-24_real64, &
    0.0_real64,0.0_real64,0.0_real64]
  witness=[8.45181987420972325e-15_real64,4.15262142530430207e-15_real64, &
    0.0_real64,0.0_real64,1.98311229660493620e-27_real64]
  lower=high-witness/3.0e-8_real64; upper=high+witness/3.0e-8_real64
  mass=-3.97046694025453284e-23_real64; moment=dot_product(gas,high)
  call air5_close_species_flux_moment(high,lower,upper,mass,gas,moment,closed,status)
  if(status/=chemistry_status_ok) error stop 'recorded small transverse flux rejected'
  if(any(closed<lower) .or. any(closed>upper)) error stop 'transverse flux outside bounds'
  if(abs(sum(closed)-mass)>64*epsilon(mass)*max(abs(mass),sum(abs(high)))) &
    error stop 'transverse flux mass residual'
  write(*,'(A)') 'AIR5_TRANSVERSE_MOMENT_PROBE pass'
  ! Recorded RK2 seam: near-zero advective flux needs a finite bounded
  ! redistribution. Its summation roundoff scales with the OUTPUT operands.
  high=[-4.1943372176048245e-23_real64,-1.1888063246313867e-23_real64, &
    -6.078716994208774e-27_real64,0.0_real64,0.0_real64]
  witness=[1.5744425821605036e-10_real64,1.5729792623244496e-11_real64, &
    -1.7317405083933405e-10_real64,0.0_real64,0.0_real64]
  lower=witness-[9.679070480478613e-15_real64,2.74589500326727e-15_real64, &
    2.12002968377418e-18_real64,0.0_real64,0.0_real64]/7.5e-9_real64
  upper=witness+[9.692948310003935e-15_real64,2.7339443446462405e-15_real64, &
    2.1646756354916756e-19_real64,0.0_real64,0.0_real64]/7.5e-9_real64
  mass=-5.383751413935631e-23_real64; moment=dot_product(gas,high)
  call air5_close_species_flux_moment(high,lower,upper,mass,gas,moment,closed,status)
  if(status/=chemistry_status_ok) error stop 'recorded RK2 redistribution rejected'
  if(any(closed<lower) .or. any(closed>upper)) error stop 'RK2 redistribution outside bounds'
  if(abs(sum(closed)-mass)>64*epsilon(mass)*(abs(mass)+sum(abs(closed)))) &
    error stop 'RK2 redistribution mass residual'
  if(abs(dot_product(gas,closed)-moment)> &
    64*epsilon(moment)*(abs(moment)+sum(abs(gas*closed)))) error stop 'RK2 redistribution moment residual'
  write(*,'(A)') 'AIR5_RK2_MOMENT_PROBE pass'

  support=0.0_real64
  support(1:2,:)=1.0_real64
  lower=-1.0_real64; upper=1.0_real64
  call air5_preserve_zero_flux_support(support(:,1),support(:,2),support(:,3), &
    support(:,4),support(:,5),support(:,6),lower,upper)
  if(any(lower(3:5)/=0).or.any(upper(3:5)/=0)) error stop 'zero support not fixed'
  if(any(lower(1:2)/=-1).or.any(upper(1:2)/=1)) error stop 'bulk budget changed'
  ! A nonzero contribution in ANY input, even far below a practical floor,
  ! must prevent support locking. Endpoint states alone are insufficient.
  do i=1,6
    support(3,i)=1.0e-200_real64
    lower=-1.0_real64; upper=1.0_real64
    call air5_preserve_zero_flux_support(support(:,1),support(:,2),support(:,3), &
      support(:,4),support(:,5),support(:,6),lower,upper)
    if(lower(3)/=-1.or.upper(3)/=1) error stop 'nonzero trace support erased'
    support(3,i)=0.0_real64
  enddo
  write(*,'(A)') 'AIR5_ZERO_FLUX_SUPPORT_PROBE pass'

  ! Captured characteristic-top normal projection, step 7 / RK2, f3cced3.
  high=[8.8586932245617473e-23_real64,2.6911023746061185e-23_real64, &
    0.0_real64,0.0_real64,0.0_real64]
  lower=[-7.5846354166670633e-5_real64,-2.4967447916665852e-4_real64, &
    -3.2552083333331987e-4_real64,-3.2552083333331987e-4_real64,-3.2552083333331987e-4_real64]
  upper=[2.4967447916664925e-4_real64,7.5846354166661376e-5_real64, &
    0.0_real64,0.0_real64,0.0_real64]
  gas=[296.9450935054728_real64,259.8269568172887_real64,593.8901870109456_real64, &
    519.6539136345774_real64,277.1487539384413_real64]
  mass=1.1549795599167551e-22_real64
  moment=3.3297664283814732e-20_real64
  call air5_preserve_zero_flux_support(support(:,1),support(:,2),support(:,3), &
    support(:,4),support(:,5),support(:,6),lower,upper)
  call air5_close_species_flux_moment(high,lower,upper,mass,gas,moment,closed,status)
  if(status/=chemistry_status_ok) error stop 'captured zero-support face rejected'
  if(any(closed(3:5)/=0.0_real64)) error stop 'captured face creates absent species'
  if(any(closed<lower).or.any(closed>upper)) error stop 'captured face budget violation'
  if(abs(sum(closed)-mass)>64*epsilon(mass)*(abs(mass)+sum(abs(closed)))) &
    error stop 'captured face mass mismatch'
  if(abs(dot_product(gas,closed)-moment)>64*epsilon(moment)*(abs(moment)+sum(abs(gas*closed)))) &
    error stop 'captured face moment mismatch'
  write(*,'(A)') 'AIR5_CAPTURED_ZERO_SUPPORT_PROBE pass'
end program air5_transport_roundoff_probe
