program boundary_layer_probe
  use mpi
  use astr_precision, only: real_kind
  use insitu_boundary_layer, only: wall_transport,wall_stencil_transport,wall_reference,wall_reference_result, &
    boundary_layer_thickness,thickness_result,nonreacting_wall_material,resolve_recovery_temperature, &
    wall_grid_spacings,wall_distance_plus,wall_scaled_quantity,cartesian_block_axes,cartesian_axis_direction, &
    boundary_layer_edge,thickness_reynolds
  use index_slice_stencil, only: slice_stencil_weights
  use ieee_arithmetic, only: ieee_value,ieee_quiet_nan
#ifdef _CUDA
  use cudafor
  use insitu_boundary_layer_gpu, only: evaluate_wall_transport_gpu,evaluate_wall_stencils_gpu,cartesian_block_axes_gpu
#endif
  implicit none
  real(real_kind) :: gradient(3,3),thermal(3),shear(3),heat,nan,mu(12),kappa(12), &
    gradients(3,3,12),thermals(3,12),reference(4,12),observed(4,12),worst,absolute
  type(wall_reference_result) :: wall
  type(thickness_result) :: thickness
  real(real_kind) :: y(5),rho(5),u(5),cd(5),ct(5)
  real(real_kind) :: edge(4),edge_temperature(5),edge_pressure(5),re_theta
  real(real_kind) :: material_mu,material_kappa,recovery
  real(real_kind) :: grid_x(0:4),grid_y(0:1),grid_z(0:4),distances(5,5),wall_plus
  logical :: distance_available(5,5)
  integer :: rank,ranks,ierr,node
  logical :: ok,mpi_enabled
  character(32) :: mode
  real(real_kind) :: stencils(4,7,3,12),weights(7,0:6),metrics(3,3,12),logical_node,xi,center,h, &
    beta,coefficients(4,3),physical_delta
  integer :: offsets(3,12),d,s
#ifdef _CUDA
  real(real_kind),allocatable,device :: gradients_d(:,:,:),thermals_d(:,:),mu_d(:),kappa_d(:),values_d(:,:)
  integer,allocatable,device :: invalid_d(:)
  integer,allocatable,device :: offsets_d(:,:)
  real(real_kind),allocatable,device :: stencil_d(:,:,:,:),weights_d(:,:),metrics_d(:,:,:)
  integer :: devices,status,device
#endif
  call get_command_argument(1,mode)
  mpi_enabled=trim(mode)/='--kernel-only';rank=0;ranks=1
  call require(len_trim(mode)==0.or.trim(mode)=='--kernel-only','probe argument')
  if(mpi_enabled) then
    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
    call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  endif
  nan=ieee_value(0.e0_real_kind,ieee_quiet_nan);worst=0;absolute=0
  call nonreacting_wall_material(1.e0_real_kind,.true.,.2e0_real_kind,1.2e0_real_kind,1000.e0_real_kind, &
    nan,nan,nan,.false.,material_mu,material_kappa,ok)
  call require(ok,'unused conductivity references are not dependencies of viscosity')
  call agreement([material_mu,material_kappa],[.001e0_real_kind,0.e0_real_kind],'nondimensional Sutherland reference')
  call nonreacting_wall_material(273.15e0_real_kind,.false.,nan,nan,nan,.72e0_real_kind,nan, &
    1004.5e0_real_kind,.true.,material_mu,material_kappa,ok)
  call require(ok,'dimensional air transport')
  call agreement([material_mu,material_kappa],[1.716e-5_real_kind,1004.5e0_real_kind*1.716e-5_real_kind/.72e0_real_kind], &
    'dimensional Sutherland and molecular conduction')
  call slice_stencil_weights(weights)
  do node=1,12
    gradient=0
    gradient(1,1)=-2;gradient(2,2)=1;gradient(3,3)=3
    gradient(1,2)=3;gradient(2,1)=2
    gradient(3,2)=-4;gradient(2,3)=.5e0_real_kind
    mu(node)=real(node,real_kind)*.25e0_real_kind
    kappa(node)=2*mu(node)
    thermal=[3.e0_real_kind,-4.e0_real_kind,5.e0_real_kind]
    gradients(:,:,node)=gradient;thermals(:,node)=thermal
    call wall_transport(mu(node),kappa(node),gradient,thermal,[0.e0_real_kind,2.e0_real_kind,0.e0_real_kind], &
      shear,heat,ok)
    call require(ok,'full blowing-wall tensor')
    reference(:,node)=[5*mu(node),0.e0_real_kind,-3.5e0_real_kind*mu(node),-4*kappa(node)]
    call agreement([shear,heat],reference(:,node),'analytic shear and conductive heat')
    call wall_transport(mu(node),kappa(node),gradient,thermal,[0.e0_real_kind,-1.e0_real_kind,0.e0_real_kind], &
      shear,heat,ok)
    call require(ok,'reversed wall normal')
    call agreement([shear,heat],-reference(:,node),'normal orientation')
  enddo
  coefficients=0
  coefficients(1,1)=-2;coefficients(1,2)=3
  coefficients(2,1)=2;coefficients(2,2)=1;coefficients(2,3)=.5e0_real_kind
  coefficients(3,2)=-4;coefficients(3,3)=3
  coefficients(4,:)=[3.e0_real_kind,-4.e0_real_kind,5.e0_real_kind]
  metrics=0;beta=.3e0_real_kind;h=.025e0_real_kind
  do node=1,12
    do d=1,3
      offsets(d,node)=modulo(node-1+d,7);center=real(node+d,real_kind)*.03e0_real_kind
      metrics(d,d,node)=1.e0_real_kind/(h*(1+2*beta*center))
      do s=1,7
        logical_node=real(s-1-offsets(d,node),real_kind)*h;xi=center+logical_node
        physical_delta=(xi+beta*xi*xi)-(center+beta*center*center)
        stencils(:,s,d,node)=real(node,real_kind)+coefficients(:,d)*physical_delta+logical_node**6
      enddo
    enddo
    call wall_stencil_transport(stencils(:,:,:,node),offsets(:,node),weights,metrics(:,:,node),mu(node),kappa(node), &
      [0.e0_real_kind,1.e0_real_kind,0.e0_real_kind],shear,heat,ok)
    call require(ok,'sixth-order stretched-grid wall stencil')
    call agreement([shear,heat],reference(:,node),'all seven boundary/interior offsets and metrics')
  enddo
  gradient=0;gradient(1,2)=-2;gradient(2,1)=2
  call wall_transport(1.e0_real_kind,1.e0_real_kind,gradient,[0.e0_real_kind,0.e0_real_kind,0.e0_real_kind], &
    [0.e0_real_kind,1.e0_real_kind,0.e0_real_kind],shear,heat,ok)
  call require(ok.and.all(shear==0).and.heat==0,'rigid rotation has no viscous shear')
  call wall_transport(1.e0_real_kind,1.e0_real_kind,gradient,[0.e0_real_kind,0.e0_real_kind,0.e0_real_kind], &
    [0.e0_real_kind,0.e0_real_kind,0.e0_real_kind],shear,heat,ok)
  call require(.not.ok,'undefined normal')
  call wall_transport(nan,1.e0_real_kind,gradient,[0.e0_real_kind,0.e0_real_kind,0.e0_real_kind], &
    [0.e0_real_kind,1.e0_real_kind,0.e0_real_kind],shear,heat,ok)
  call require(.not.ok,'nonfinite material')
  call wall_reference([-3.e0_real_kind,0.e0_real_kind,4.e0_real_kind],2.e0_real_kind,.5e0_real_kind, &
    4.e0_real_kind,3.e0_real_kind,1.25e0_real_kind,2.e0_real_kind,wall,ok, &
    cp_inf=2.e0_real_kind,recovery_temperature=6.e0_real_kind)
  call require(ok.and.wall%scale_available.and.wall%stanton_available,'statistical wall references')
  call require(wall%cf_available.and.wall%cq_available,'available fixed-reference coefficients')
  call agreement([wall%shear_magnitude,wall%u_tau,wall%nu,wall%length,wall%cf,wall%cq,wall%stanton], &
    [5.e0_real_kind,sqrt(2.5e0_real_kind),.25e0_real_kind,.25e0_real_kind/sqrt(2.5e0_real_kind), &
     -1.2e0_real_kind,.3e0_real_kind,.3e0_real_kind],'signed coefficients and wall units')
  call wall_scaled_quantity(3.e0_real_kind,[wall%u_tau],wall_plus,ok)
  call require(ok,'statistical velocity wall unit')
  call agreement([wall_plus],[3.e0_real_kind/sqrt(2.5e0_real_kind)],'velocity divided by statistical u_tau')
  call wall_scaled_quantity(-.75e0_real_kind,[wall%u_tau,wall%u_tau],wall_plus,ok)
  call require(ok,'signed covariance wall unit')
  call agreement([wall_plus],[-.3e0_real_kind],'covariance keeps its sign and uses u_tau squared')
  call wall_scaled_quantity(-1.5e0_real_kind,[2.e0_real_kind,wall%u_tau,wall%u_tau],wall_plus,ok)
  call require(ok,'density-weighted stress wall unit')
  call agreement([wall_plus],[-.3e0_real_kind],'Favre density stress uses mean wall density, not local profile density')
  call wall_reference([0.e0_real_kind,0.e0_real_kind,0.e0_real_kind],2.e0_real_kind,.5e0_real_kind, &
    4.e0_real_kind,3.e0_real_kind,1.25e0_real_kind,2.e0_real_kind,wall,ok, &
    cp_inf=2.e0_real_kind,recovery_temperature=4.e0_real_kind)
  call require(ok.and..not.wall%scale_available.and..not.wall%stanton_available.and.wall%cq==.3e0_real_kind, &
    'zero shear/temperature difference only disable dependent scales')
  call wall_reference([-3.e0_real_kind,0.e0_real_kind,4.e0_real_kind],2.e0_real_kind,.5e0_real_kind, &
    4.e0_real_kind,3.e0_real_kind,1.25e0_real_kind,0.e0_real_kind,wall,ok)
  call require(ok.and.wall%scale_available.and..not.wall%cf_available.and..not.wall%cq_available, &
    'zero free-stream speed does not invalidate local viscous scales')
  call resolve_recovery_temperature(2,.9e0_real_kind,1.e0_real_kind,1.4e0_real_kind,12.e0_real_kind,recovery,ok)
  call require(ok,'explicit recovery factor')
  call agreement([recovery],[26.92e0_real_kind],'ideal-gas recovery-temperature definition')
  call resolve_recovery_temperature(2,0.e0_real_kind,1.e0_real_kind,1.4e0_real_kind,12.e0_real_kind,recovery,ok)
  call require(ok.and.recovery==1.e0_real_kind,'zero recovery-factor limit')
  grid_x=[0.e0_real_kind,.1e0_real_kind,.3e0_real_kind,.6e0_real_kind,1.e0_real_kind]
  grid_y=[2.e0_real_kind,2.02e0_real_kind]
  grid_z=[0.e0_real_kind,.2e0_real_kind,.5e0_real_kind,.9e0_real_kind,1.4e0_real_kind]
  call wall_grid_spacings(grid_x,grid_y,grid_z,reshape([0,4,1,3],[2,2]),[.false.,.true.], &
    distances,distance_available,ok)
  call require(ok.and..not.distance_available(2,1).and..not.distance_available(3,5), &
    'nonperiodic x endpoints do not invent neighboring cells')
  call agreement(distances(1,:),spread(.02e0_real_kind,1,5),'first physical off-wall distance')
  call agreement(distances(2,2:),[.1e0_real_kind,.2e0_real_kind,.3e0_real_kind,.4e0_real_kind], &
    'stretched x minus intervals')
  call agreement(distances(3,:4),[.1e0_real_kind,.2e0_real_kind,.3e0_real_kind,.4e0_real_kind], &
    'stretched x plus intervals')
  call agreement(distances(4,:),spread(.3e0_real_kind,1,5),'partial z span minimum cell')
  call agreement(distances(5,:),spread(.4e0_real_kind,1,5),'partial z span maximum cell')
  call wall_grid_spacings(grid_x,grid_y,grid_z,reshape([0,4,4,4],[2,2]),[.true.,.true.], &
    distances,distance_available,ok)
  call require(ok.and.all(distance_available),'periodic terminal node aliases have both incident cells')
  call agreement(distances(2,[1,5]),[.4e0_real_kind,.4e0_real_kind],'periodic x seam minus interval')
  call agreement(distances(3,[1,5]),[.1e0_real_kind,.1e0_real_kind],'periodic x seam plus interval')
  call agreement(distances(4,:),spread(.2e0_real_kind,1,5),'periodic z alias incident minimum')
  call agreement(distances(5,:),spread(.5e0_real_kind,1,5),'periodic z alias incident maximum')
  call wall_grid_spacings(-grid_x,-grid_y,-grid_z,reshape([0,4,2,2],[2,2]),[.false.,.false.], &
    distances,distance_available,ok)
  call require(ok,'grid distances use lengths rather than coordinate signs')
  call agreement(distances(4,:),spread(.3e0_real_kind,1,5),'interior line incident minimum')
  call agreement(distances(5,:),spread(.4e0_real_kind,1,5),'interior line incident maximum')
  call wall_distance_plus(.02e0_real_kind,.005e0_real_kind,wall_plus,ok)
  call require(ok,'positive statistical viscous length')
  call agreement([wall_plus],[4.e0_real_kind],'traditional grid wall unit')
  call wall_distance_plus(.02e0_real_kind,0.e0_real_kind,wall_plus,ok)
  call require(.not.ok,'undefined viscous length has no fabricated grid units')
  call wall_reference([.59e0_real_kind*huge(1.e0_real_kind),0.e0_real_kind,0.e0_real_kind], &
    2.e0_real_kind,.5e0_real_kind,4.e0_real_kind,3.e0_real_kind,1.e0_real_kind,1.1e0_real_kind,wall,ok)
  if(.not.wall%cf_available) then
    print '(a,2l2,4es24.16)','extreme reference: ',ok,wall%scale_available,wall%cf,wall%u_tau,wall%nu,wall%length
    print '(a,i0,2es24.16)','SCALE edge: ',2*exponent(wall%u_tau), &
      scale(.975e0_real_kind,2*exponent(wall%u_tau)), &
      scale(scale(.975e0_real_kind,exponent(wall%u_tau)),exponent(wall%u_tau))
  endif
  call require(wall%cf_available,'representable extreme coefficient remains available')
  call require(ok.and.wall%scale_available.and.wall%cf<=huge(1.e0_real_kind), &
    'coefficient multiplication order avoids a representable-result overflow')
  y=[0.e0_real_kind,.2e0_real_kind,.55e0_real_kind,1.e0_real_kind,1.5e0_real_kind]
  rho=[1.e0_real_kind,2.e0_real_kind,4.e0_real_kind,5.e0_real_kind,6.e0_real_kind]
  u=[0.e0_real_kind,.5e0_real_kind,1.e0_real_kind,2.e0_real_kind,3.e0_real_kind]
  edge_temperature=[2.e0_real_kind,3.e0_real_kind,4.e0_real_kind,5.e0_real_kind,6.e0_real_kind]
  edge_pressure=2*edge_temperature
  call boundary_layer_edge(y,rho,u,edge_temperature,edge_pressure,[2,4],edge,ok)
  call require(ok,'explicit nonuniform edge reference band')
  call agreement(edge,[123.e0_real_kind/32,161.e0_real_kind/123,65.e0_real_kind/16,65.e0_real_kind/8], &
    'length means and mass-length weighted edge velocity')
  call boundary_layer_edge(y,rho,u,edge_temperature,edge_pressure,[3,3],edge,ok)
  call require(ok,'single-node edge band')
  call agreement(edge,[4.e0_real_kind,1.e0_real_kind,4.e0_real_kind,8.e0_real_kind],'single-node direct reference')
  call thickness_reynolds(2.e0_real_kind,3.e0_real_kind,.4e0_real_kind,.2e0_real_kind,re_theta,ok)
  call require(ok,'edge-state momentum thickness Reynolds number')
  call agreement([re_theta],[12.e0_real_kind],'signed integral Reynolds scaling')
  u(3)=-1
  call boundary_layer_edge(y,rho,u,edge_temperature,edge_pressure,[3,3],edge,ok)
  call require(.not.ok,'no positive external reference is not replaced by a maximum')
  rho=1;u=y
  call boundary_layer_thickness(y,rho,u,1.e0_real_kind,1.e0_real_kind,[4,5],thickness,cd,ct,ok, &
    explicit_upper=.7e0_real_kind)
  call require(ok.and.thickness%delta_available.and.thickness%explicit_available,'linear profile thickness')
  call agreement([thickness%delta99,thickness%displacement99,thickness%displacement_explicit], &
    [.99e0_real_kind,.49995e0_real_kind,.455e0_real_kind],'crossing and partial trapezoid')
  call agreement(cd,y-.5e0_real_kind*y*y,'nonuniform cumulative displacement')
  u=[0.e0_real_kind,1.1e0_real_kind,.8e0_real_kind,1.e0_real_kind,1.e0_real_kind]
  call boundary_layer_thickness(y,rho,u,1.e0_real_kind,1.e0_real_kind,[4,5],thickness,cd,ct,ok)
  call require(ok.and.thickness%delta_available.and.thickness%multiple_crossings,'outer connected threshold component')
  call agreement([thickness%delta99],[.9775e0_real_kind],'do not select internal overshoot')
  u=[0.e0_real_kind,.99e0_real_kind,.99e0_real_kind,1.e0_real_kind,1.e0_real_kind]
  call boundary_layer_thickness(y,rho,u,1.e0_real_kind,1.e0_real_kind,[4,5],thickness,cd,ct,ok)
  call require(ok.and.thickness%delta_available,'threshold plateau')
  call agreement([thickness%delta99],[.2e0_real_kind],'inner edge of the connected threshold plateau')
  u=.75e0_real_kind*huge(1.e0_real_kind);u(:2)=-u(:2)
  call boundary_layer_thickness(y,rho,u,1.e0_real_kind,.75e0_real_kind*huge(1.e0_real_kind), &
    [4,5],thickness,cd,ct,ok)
  call require(ok.and.thickness%delta_available,'finite extreme crossing state')
  call agreement([thickness%delta99],[.54825e0_real_kind],'crossing interpolation uses scaled velocities')
  u=.4e0_real_kind
  call boundary_layer_thickness(y,rho,u,1.e0_real_kind,1.e0_real_kind,[4,5],thickness,cd,ct,ok, &
    explicit_upper=.7e0_real_kind)
  call require(ok.and..not.thickness%delta_available.and.thickness%explicit_available,'missing crossing retains explicit integral')
  call agreement([thickness%displacement_explicit,thickness%momentum_explicit], &
    [.42e0_real_kind,.168e0_real_kind],'constant integrands')
  u=[-.2e0_real_kind,.3e0_real_kind,.7e0_real_kind,1.e0_real_kind,1.e0_real_kind]
  call boundary_layer_thickness(y,rho,u,1.e0_real_kind,1.e0_real_kind,[4,5],thickness,cd,ct,ok)
  call require(ok.and.thickness%delta_available,'negative local velocity is not clipped')
  call require(ct(2)<.042e0_real_kind,'signed reverse-flow momentum contribution')
  y(3)=y(2)
  call boundary_layer_thickness(y,rho,u,1.e0_real_kind,1.e0_real_kind,[4,5],thickness,cd,ct,ok)
  call require(.not.ok,'repeated wall-normal coordinate')
  call check_integral_convergence()
  call check_cartesian_geometry()
#ifdef _CUDA
  status=cudaGetDeviceCount(devices)
  call require(status==cudaSuccess.and.devices>0,'CUDA availability')
  status=cudaSetDevice(mod(rank,devices));call require(status==cudaSuccess,'device binding')
  status=cudaGetDevice(device);call require(status==cudaSuccess,'device identity')
  allocate(gradients_d(3,3,12),thermals_d(3,12),mu_d(12),kappa_d(12),values_d(4,12),invalid_d(1))
  gradients_d=gradients;thermals_d=thermals;mu_d=mu;kappa_d=kappa
  call evaluate_wall_transport_gpu(mu_d,kappa_d,gradients_d,thermals_d, &
    [0.e0_real_kind,1.e0_real_kind,0.e0_real_kind],values_d,invalid_d,ok)
  call require(ok,'device full wall tensor')
  observed=values_d
  do node=1,12
    call agreement(observed(:,node),reference(:,node),'device tensor agreement')
  enddo
  allocate(stencil_d(4,7,3,12),offsets_d(3,12),weights_d(7,0:6),metrics_d(3,3,12))
  stencil_d=stencils;offsets_d=offsets;weights_d=weights;metrics_d=metrics
  call evaluate_wall_stencils_gpu(stencil_d,offsets_d,weights_d,metrics_d,mu_d,kappa_d, &
    [0.e0_real_kind,1.e0_real_kind,0.e0_real_kind],values_d,invalid_d,ok)
  call require(ok,'device sixth-order wall stencil')
  observed=values_d
  do node=1,12
    call agreement(observed(:,node),reference(:,node),'device stencil analytic reference')
  enddo
  deallocate(stencil_d,offsets_d,weights_d,metrics_d)
  deallocate(gradients_d,thermals_d,mu_d,kappa_d,values_d,invalid_d)
#endif
  write(*,'(a,i0,a,es16.8,a,es16.8)') 'PASS boundary layer diagnostics rank=',rank, &
    ' max_scaled_error=',worst,' max_absolute_error=',absolute
  if(mpi_enabled) call MPI_Finalize(ierr)
contains
  subroutine check_cartesian_geometry()
    integer,parameter :: cells(3)=[4,2,3],halo=2
    real(real_kind) :: coordinates(-2:6,-2:4,-2:5,3),axes(12),expected(12)
    integer :: i,j,k
    logical :: valid
#ifdef _CUDA
    real(real_kind),allocatable,device :: coordinates_d(:,:,:,:)
    integer :: status,devices
    status=cudaGetDeviceCount(devices);call require(status==cudaSuccess.and.devices>0,'Cartesian probe GPU')
    status=cudaSetDevice(mod(rank,devices));call require(status==cudaSuccess,'Cartesian probe binding')
    allocate(coordinates_d(-2:6,-2:4,-2:5,3))
#endif
    coordinates=nan
    do k=0,cells(3)
    do j=0,cells(2)
    do i=0,cells(1)
      coordinates(i,j,k,:)=[.2e0_real_kind*i*i,1.e0_real_kind+.1e0_real_kind*j*j,-.3e0_real_kind*k]
    enddo
    enddo
    enddo
    expected=[coordinates(0:4,0,0,1),coordinates(0,0:2,0,2),coordinates(0,0,0:3,3)]
    call cartesian_block_axes(coordinates,cells,halo,axes,valid)
    call require(valid,'nonuniform Cartesian physical nodes; unused halos are NaN')
    call agreement(axes,expected,'Cartesian axis packing')
    call require(cartesian_axis_direction(axes(:5))==1.and.cartesian_axis_direction(axes(9:))==-1, &
      'physical axis directions are not inferred from file-read flags')
    call require(cartesian_axis_direction([0.e0_real_kind,0.e0_real_kind])==0,'degenerate coordinate axis')
#ifdef _CUDA
    coordinates_d=coordinates
    call cartesian_block_axes_gpu(coordinates_d,cells,halo,axes,valid)
    call require(valid,'device nonuniform Cartesian geometry')
    call agreement(axes,expected,'device small one-dimensional axis packet')
#endif
    coordinates(2,1,1,1)=coordinates(2,1,1,1)+1.e-12_real_kind
    call cartesian_block_axes(coordinates,cells,halo,axes,valid)
    call require(.not.valid,'off-axis geometry is not silently rounded to Cartesian')
#ifdef _CUDA
    coordinates_d=coordinates
    call cartesian_block_axes_gpu(coordinates_d,cells,halo,axes,valid)
    call require(.not.valid,'device rejects the same off-axis geometry')
    deallocate(coordinates_d)
#endif
  end subroutine

  subroutine check_integral_convergence()
    real(real_kind),allocatable :: coordinates(:),density(:),velocity(:),displacement(:),momentum(:)
    real(real_kind) :: errors(3),expected
    integer :: level,n,j
    type(thickness_result) :: result
    logical :: valid
    expected=.99e0_real_kind**2/2.e0_real_kind-.99e0_real_kind**3/3.e0_real_kind
    do level=1,3
      n=2**(level+3)
      allocate(coordinates(n+1),density(n+1),velocity(n+1),displacement(n+1),momentum(n+1))
      do j=1,n+1
        coordinates(j)=(real(j-1,real_kind)/n)**1.4e0_real_kind
      enddo
      density=1;velocity=coordinates
      call boundary_layer_thickness(coordinates,density,velocity,1.e0_real_kind,1.e0_real_kind,[n+1,n+1], &
        result,displacement,momentum,valid)
      call require(valid.and.result%delta_available,'stretched diagnostic mesh convergence')
      errors(level)=abs(result%momentum99-expected)
      deallocate(coordinates,density,velocity,displacement,momentum)
    enddo
    call require(all(errors(:2)/errors(2:)>3.5e0_real_kind).and. &
      all(errors(:2)/errors(2:)<4.5e0_real_kind),'second-order trapezoid convergence, not sixth order')
    if(rank==0) print '(a,3es16.8)','diagnostic_grid_errors=',errors
  end subroutine

  subroutine agreement(a,b,label)
    real(real_kind),intent(in) :: a(:),b(:)
    character(*),intent(in) :: label
    real(real_kind) :: difference
    difference=maxval(abs(a-b)/max(1.e0_real_kind,abs(a),abs(b)))
    worst=max(worst,difference);absolute=max(absolute,maxval(abs(a-b)))
    call require(difference<=2.e-10_real_kind,label)
  end subroutine
  subroutine require(condition,label)
    logical,intent(in) :: condition
    character(*),intent(in) :: label
    if(condition) return
    print '(a)','FAIL boundary layer: '//label
    if(mpi_enabled) call MPI_Abort(MPI_COMM_WORLD,1,ierr)
    error stop 1
  end subroutine
end program
