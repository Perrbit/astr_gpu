program wall41_step_probe
  use iso_c_binding
  use ieee_arithmetic, only: ieee_is_finite,ieee_value,ieee_quiet_nan
  use mpi
  use device_state
  use halo_transport
  use fluid_explicit
  use fluid_step, only: fluid_prepare_stage,fluid_advance_prepared_stage,fluid_filter_explicit
  use boundary_state_cpp
  use compact_operator, only: compact_line_partition,partition_compact_line,compact_derivative_core,compact_filter_core
  use compact_line_solver, only: compact_matrix,compact_reference_factor,compact_residual, &
    factor_compact_reference,solve_compact_reference
  use astr_precision_mpi, only: mpi_real_type
  use compact_line_cpp
  use keep_operator, only: keep_split_plan,build_keep_split_plan
  use me6opti_operator, only: me6_coefficients
  implicit none
  type(c_ptr) :: state
  type(compact_distributed_cpp_plan) :: derivative_plans(3),filter_plans(3)
  type(compact_stencil_cpp) :: stencils(3)
  integer :: keep_bases,filter_components,a
  integer(c_int) :: line_mode=0
  integer(c_int64_t) :: interface_total(2)=0
  integer(c_int64_t) :: me6_total(2)=0,me6_values(4)
  integer :: residual_checks=0
  real(c_double) :: line_residual_max=0,line_error_max=0
  integer :: h(4),qh(5),rh(4),im,jm,km,hm,unit,axis,backend,debug,step,stage,steps,ierr,c,count=0, &
    rank,ranks,local_rank,split_axis,g(3),origin(3),upper(3),gradient_checks=0,boundary_type=41
  integer(c_int) :: low(3),high(3),types(3),first(3),last(3),peers(3),periodic(3)
  integer(c_int) :: higher(3),sizes(3),mode
  real(c_double),allocatable :: q(:,:,:,:),expected(:,:,:,:),metric(:,:,:,:,:),jac(:,:,:), &
    primitives(:,:,:,:),reference_primitives(:,:,:,:),rhs(:,:,:,:),reference_rhs(:,:,:,:), &
    global_q(:,:,:,:),global_primitives(:,:,:,:),global_rhs(:,:,:,:),global_metric(:,:,:,:,:),global_jac(:,:,:), &
    gradient_actual(:,:,:,:),gradient_reference(:,:,:,:),global_gradient(:,:,:,:)
  real(c_double) :: qmax=0,rmax=0,pmax=0,gmax=0,nan,force_fixed=0
  character(1024) :: root,directory,path,dump_directory
  character(80) :: suffix,arg
  character(3) :: label
  character(8) :: workspace='off'
  character(3) :: wall_axes='xyz'
  logical :: filtering,compact=.false.,compact_filter=.false.,me6=.false.,keep_convection=.true.
  character(1),parameter :: letters(3)=['x','y','z']
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',arg,status=ierr)
  if(ierr/=0) error stop 'wall step probe requires matching Open MPI launcher'
  read(arg,*) local_rank
  ierr=device_bind(int(local_rank,c_int))
  if(ierr/=0) error stop 'pre-MPI wall probe device binding failed'
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  if(ranks/=1.and.ranks/=2) error stop 'wall step probe covers NP=1/2'
  call get_command_argument(1,root);call get_command_argument(2,arg);read(arg,*) steps
  split_axis=1
  if(ranks==2) then
    call get_command_argument(3,arg);read(arg,*) split_axis
    if(split_axis<1.or.split_axis>3) error stop 'split axis must be 1/2/3'
  endif
  mode=halo_pinned
  call get_command_argument(4,arg)
  if(len_trim(arg)>0) then
    select case(trim(arg))
    case('pinned');mode=halo_pinned
    case('device');mode=halo_device
    case default;error stop 'transport must be pinned/device'
    end select
  endif
  call get_command_argument(5,arg)
  if(len_trim(arg)>0) read(arg,*) force_fixed
  call get_command_argument(6,arg)
  if(len_trim(arg)>0) workspace=trim(arg)
  if(workspace/='off'.and.workspace/='scalar'.and.workspace/='full') error stop 'filter workspace must be off/scalar/full'
  filtering=workspace/='off'
  call get_command_argument(7,dump_directory)
  call get_command_argument(8,arg)
  if(len_trim(arg)>0) then
    select case(trim(arg))
    case('pinned');line_mode=0
    case('device-aware');line_mode=1
    case default;error stop 'compact transport must be pinned/device-aware'
    end select
  endif
  call get_command_argument(9,arg)
  if(len_trim(arg)>0) then
    select case(trim(arg))
    case('explicit');compact=.false.
    case('compact');compact=.true.
    case default;error stop 'wall derivative method must be explicit/compact'
    end select
  endif
  call get_command_argument(10,arg)
  if(len_trim(arg)>0) then
    select case(trim(arg))
    case('standard');me6=.false.
    case('me6_opti');me6=.true.
    case default;error stop 'wall viscous form must be standard/me6_opti'
    end select
  endif
  compact_filter=compact
  call get_command_argument(11,arg)
  if(len_trim(arg)>0) then
    select case(trim(arg))
    case('explicit');compact_filter=.false.
    case('compact');compact_filter=.true.
    case default;error stop 'wall filter method must be explicit/compact'
    end select
  endif
  compact_filter=compact_filter.and.filtering
  call get_command_argument(12,arg)
  if(len_trim(arg)>0) wall_axes=trim(arg)
  call get_command_argument(13,arg)
  if(len_trim(arg)>0) read(arg,*) boundary_type
  if(boundary_type/=41.and.boundary_type/=42.and.boundary_type/=50.and.boundary_type/=60) &
    error stop 'boundary must be bc41/bc42/bc50/bc60'
  call get_command_argument(14,arg)
  if(len_trim(arg)>0) then
    select case(trim(arg))
    case('keep');keep_convection=.true.
    case('standard');keep_convection=.false.
    case default;error stop 'convection must be keep/standard'
    end select
  endif
  filter_components=merge(1,5,workspace=='scalar')
  nan=ieee_value(0.d0,ieee_quiet_nan)
  do axis=0,2
  if(index(wall_axes,letters(axis+1))==0) cycle
  do backend=1,2
    label='cpu'
    if(backend==2) label='gpu'
    directory=trim(root)//'/'//trim(label)//'_'//letters(axis+1)
    open(newunit=unit,file=trim(directory)//'/geometry_input.bin',status='old',access='stream',form='unformatted')
    read(unit) h
    g=h(:3);upper=g;origin=0;hm=h(4);sizes=1
    if(ranks==2) then
      if(mod(g(split_axis),2)/=0) error stop 'even global intervals required'
      upper(split_axis)=g(split_axis)/2;origin(split_axis)=rank*upper(split_axis);sizes(split_axis)=2
    endif
    im=upper(1);jm=upper(2);km=upper(3)
    allocate(q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),expected(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5), &
      metric(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3),jac(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
      primitives(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6),reference_primitives(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6), &
      rhs(0:im,0:jm,0:km,5),reference_rhs(0:im,0:jm,0:km,5), &
      global_q(-hm:g(1)+hm,-hm:g(2)+hm,-hm:g(3)+hm,5), &
      global_primitives(-hm:g(1)+hm,-hm:g(2)+hm,-hm:g(3)+hm,6), &
      global_rhs(0:g(1),0:g(2),0:g(3),5),global_metric(-hm:g(1)+hm,-hm:g(2)+hm,-hm:g(3)+hm,3,3), &
      global_jac(-hm:g(1)+hm,-hm:g(2)+hm,-hm:g(3)+hm), &
      gradient_actual(0:im,0:jm,0:km,12),gradient_reference(0:im,0:jm,0:km,12), &
      global_gradient(0:g(1),0:g(2),0:g(3),12))
    read(unit) global_metric,global_jac;close(unit)
    metric=global_metric(origin(1)-hm:origin(1)+im+hm,origin(2)-hm:origin(2)+jm+hm,origin(3)-hm:origin(3)+km+hm,:,:)
    jac=global_jac(origin(1)-hm:origin(1)+im+hm,origin(2)-hm:origin(2)+jm+hm,origin(3)-hm:origin(3)+km+hm)
    low=0;high=0
    low(axis+1)=merge(1,0,origin(axis+1)==0);high(axis+1)=merge(1,0,origin(axis+1)+upper(axis+1)==g(axis+1))
    types=3;first=low;last=upper-high
    if(low(axis+1)==1) types(axis+1)=1
    if(high(axis+1)==1) types(axis+1)=2
    if(low(axis+1)==1.and.high(axis+1)==1) types(axis+1)=4
    periodic=1;periodic(axis+1)=0;peers=rank;higher=rank
    if(ranks==2) then
      peers(split_axis)=1-rank;higher(split_axis)=1-rank
    endif
    if(low(axis+1)==1) peers(axis+1)=MPI_PROC_NULL
    if(high(axis+1)==1) higher(axis+1)=MPI_PROC_NULL
    do debug=0,1
      call check(device_state_create(int(local_rank,c_int),im,jm,km,hm,int(debug,c_int),state),'create')
      call check(fluid_prepare(state,1_c_int),'primitives')
      call check(fluid_rhs_prepare(state,1_c_int),'viscous fields')
      if(.not.compact_filter) then
        if(workspace=='scalar') call check(fluid_filter_prepare_scalar(state),'scalar filter work')
        if(workspace=='full') call check(fluid_filter_prepare(state),'full filter work')
      endif
      call check(device_transport_prepare(state,int(MPI_COMM_WORLD,c_int),peers,higher,sizes, &
        6_c_int,mode,0_c_int,periodic),'transport')
      if(compact.or.compact_filter) call prepare_lines()
      if(me6) call prepare_me6()
      call read_q('initialized',0,1,q)
      select case(axis)
      case(0)
        if(low(1)==1) q(-hm:-1,:,:,:)=nan
        if(high(1)==1) q(im+1:im+hm,:,:,:)=nan
      case(1)
        if(low(2)==1) q(:,-hm:-1,:,:)=nan
        if(high(2)==1) q(:,jm+1:jm+hm,:,:)=nan
      case(2)
        if(low(3)==1) q(:,:,-hm:-1,:)=nan
        if(high(3)==1) q(:,:,km+1:km+hm,:)=nan
      end select
      call check(device_state_upload(state,device_q,q,bytes(size(q))),'initial q')
      call check(device_state_upload(state,device_metric,metric,bytes(size(metric))),'metric')
      call check(device_state_upload(state,device_jacobian,jac,bytes(size(jac))),'Jacobian')
      call check(fluid_primitives(state,1.4d0*.3d0**2,1.d0/(1.4d0-1.d0)),'initial physical primitives')
      do step=0,steps-1
      do stage=1,3
        call check(fluid_prepare_stage(state,1.4d0*.3d0**2,1.d0/(1.4d0-1.d0),filtering,device_exchange_axis, &
          filter_operator=filter_selected,boundary_operator=walls,primitive_operator=refresh),'prepared wall stage')
        call observe_prepared()
        if(me6) then
          call check(fluid_advance_prepared_stage(state,stage,1.d-5,1.4d0*.3d0**2,1.d0/(1.4d0-1.d0), &
            3000.d0,.72d0,(1.4d0-1.d0)*.3d0**2,110.3d0/273.15d0,1.d0+110.3d0/273.15d0, &
            .true.,device_exchange_axis,device_exchange_begin,device_exchange_end,gradient,convection,diffusion, &
            integrated_viscous_operator=me6_wall,source_operator=source),'ME6 wall RK stage')
        else
          call check(fluid_advance_prepared_stage(state,stage,1.d-5,1.4d0*.3d0**2,1.d0/(1.4d0-1.d0), &
            3000.d0,.72d0,(1.4d0-1.d0)*.3d0**2,110.3d0/273.15d0,1.d0+110.3d0/273.15d0, &
            .true.,device_exchange_axis,device_exchange_begin,device_exchange_end,gradient,convection,diffusion, &
            source_operator=source),'wall RK stage')
        endif
        call check(device_state_download(state,device_q,q,bytes(size(q))),'post-update q')
        call read_q('post_update',step,stage,expected)
        call compare(q(0:im,0:jm,0:km,:),expected(0:im,0:jm,0:km,:),'post-update',qmax)
        count=count+1
      enddo
      enddo
      if(len_trim(dump_directory)>0) then
        write(path,'(a,"/wall_",a,"_",a,"_debug",i0,".rank",i8.8,".bin")') &
          trim(dump_directory),letters(axis+1),trim(label),debug,rank
        open(newunit=unit,file=trim(path),status='new',access='stream',form='unformatted')
        write(unit) int([upper,origin,g],c_int)
        write(unit) q(0:im,0:jm,0:km,:)
        close(unit)
      endif
      if(compact.or.compact_filter) call release_lines()
      if(me6) then
        call check(fluid_me6_metrics(state,me6_values),'ME6 traffic')
        me6_total=me6_total+me6_values(3:4)
      endif
      call check(device_state_destroy(state),'destroy')
    enddo
    deallocate(q,expected,metric,jac,primitives,reference_primitives,rhs,reference_rhs, &
      global_q,global_primitives,global_rhs,global_metric,global_jac,gradient_actual,gradient_reference,global_gradient)
  enddo
  enddo
  write(*,'(a,1x,i0,3(1x,es24.16e3))') 'CPP_WALL41_STEPS_PASS',count,qmax,pmax,rmax
  write(*,'(a,4(1x,i0))') 'CPP_WALL41_LAYOUT',rank,ranks,split_axis,mode
  write(*,'(a,2(1x,i0),1x,es24.16e3)') 'CPP_WALL41_GRADIENT',rank,gradient_checks,gmax
  if(compact.or.compact_filter) write(*,'(a,2(1x,i0),2(1x,i0))') 'CPP_WALL41_COMPACT_TRAFFIC',rank,line_mode,interface_total
  if(compact.or.compact_filter) write(*,'(a,2(1x,i0),2(1x,es24.16e3))') &
    'CPP_WALL41_COMPACT_RESIDUAL',rank,residual_checks,line_residual_max,line_error_max
  if(me6) write(*,'(a,4(1x,i0))') 'CPP_WALL41_ME6_TRAFFIC',rank,mode,me6_total
  call MPI_Finalize(ierr)
contains
  integer(c_int) function filter_selected(handle) result(status)
    type(c_ptr),value :: handle
    integer :: a,offset
    if(compact_filter) then
      status=0
      do a=1,3
        status=device_exchange_axis(handle,device_q,int(a-1,c_int),0_c_int)
        if(status/=0) return
        do offset=1,5,filter_components
          status=apply_compact(a,4,filter_components,offset,.false.)
          if(status/=0) return
        enddo
      enddo
    else
      status=fluid_filter_explicit(handle,device_exchange_axis,low,high)
    endif
    if(status==0) then
      call check(device_state_download(state,device_q,q,bytes(size(q))),'post-filter observation')
      call read_q('post_filter',step,stage,expected)
      call compare(q(0:im,0:jm,0:km,:),expected(0:im,0:jm,0:km,:),'post-filter',qmax)
    endif
  end function
  integer(c_int) function walls(handle) result(status)
    type(c_ptr),value :: handle
    integer :: side
    status=0
    do side=0,1
      if(side==0.and.low(axis+1)==0) cycle
      if(side==1.and.high(axis+1)==0) cycle
      if(boundary_type==41) then
        status=wall41_basic_state(handle,int(axis,c_int),int(side,c_int),1.d0,1.4d0*.3d0**2,1.d0/(1.4d0-1.d0))
      else if(boundary_type==42) then
        status=wall42_basic_state(handle,int(axis,c_int),int(side,c_int),1.4d0*.3d0**2,1.d0/(1.4d0-1.d0))
      else if(boundary_type==50) then
        status=zeroextrap_basic(handle,int(axis,c_int),int(side,c_int),1.4d0*.3d0**2,1.d0/(1.4d0-1.d0))
      else
        status=symmetry_basic(handle,int(axis,c_int),int(side,c_int),1.4d0*.3d0**2,1.d0/(1.4d0-1.d0))
      endif
      if(status/=0) return
    enddo
  end function
  integer(c_int) function refresh(handle) result(status)
    type(c_ptr),value :: handle
    status=fluid_refresh_primitives_boundary(handle,1.4d0*.3d0**2,1.d0/(1.4d0-1.d0),low,high)
  end function
  integer(c_int) function gradient(handle) result(status)
    type(c_ptr),value :: handle
    integer :: a
    if(compact) then
      do a=1,3
        status=apply_compact(a,1,4,1,.false.)
        if(status/=0) return
      enddo
    else
      status=fluid_gradient_basic_boundary(handle,low,high)
    endif
    if(status==0) call observe_gradient()
  end function
  subroutine observe_gradient()
    integer :: gh(5)
    call check(device_state_download(state,device_dvelocity,gradient_actual(:,:,:,1:9), &
      bytes(size(gradient_actual(:,:,:,1:9)))),'velocity gradient observation')
    call check(device_state_download(state,device_dtemperature,gradient_actual(:,:,:,10:12), &
      bytes(size(gradient_actual(:,:,:,10:12)))),'temperature gradient observation')
    call filename('gradients',step,stage)
    open(newunit=unit,file=trim(path),status='old',access='stream',form='unformatted')
    read(unit) gh
    if(any(gh(:3)/=g)) error stop 'gradient reference header'
    read(unit) global_gradient;close(unit)
    gradient_reference=global_gradient(origin(1):origin(1)+im,origin(2):origin(2)+jm,origin(3):origin(3)+km,:)
    call compare(gradient_actual,gradient_reference,'gradient',gmax)
    gradient_checks=gradient_checks+1
  end subroutine
  integer(c_int) function convection(handle) result(status)
    type(c_ptr),value :: handle
    integer :: a
    status=fluid_rhs_clear(handle)
    if(status/=0) return
    if(compact) then
      do a=1,3
        if(keep_convection) then
          status=apply_compact(a,2,keep_bases,1,.true.)
        else
          status=apply_compact(a,2,5,1,.false.)
        endif
        if(status/=0) return
      enddo
    else if(keep_convection) then
      status=fluid_keep_accumulate_basic(handle,1.4d0,types,first,last)
    else
      status=fluid_flux_divergence_basic(handle,0_c_int,low,high,first,last)
    endif
    if(status==0) call observe_rhs('conv')
  end function
  integer(c_int) function diffusion(handle) result(status)
    type(c_ptr),value :: handle
    integer :: a
    if(compact) then
      do a=1,3
        status=apply_compact(a,3,4,1,.false.)
        if(status/=0) return
      enddo
    else
      if(me6) then
        status=fluid_viscous_wall_band(handle,low,high,first,last)
      else
        status=fluid_flux_divergence_basic(handle,1_c_int,low,high,first,last)
      endif
    endif
  end function
  integer(c_int) function me6_wall(handle) result(status)
    type(c_ptr),value :: handle
    integer(c_int) :: a
    status=fluid_viscous_flux(handle,3000.d0,.72d0,(1.4d0-1.d0)*.3d0**2, &
      110.3d0/273.15d0,1.d0+110.3d0/273.15d0)
    if(status/=0) return
    do a=0,2
      status=device_exchange_axis(handle,device_stress,a,0_c_int)
      if(status/=0) return
      status=device_exchange_axis(handle,device_heat_flux,a,0_c_int)
      if(status/=0) return
    enddo
    status=diffusion(handle)
    if(status/=0) return
    status=fluid_me6_diffusion(handle,3000.d0,.72d0,(1.4d0-1.d0)*.3d0**2, &
      110.3d0/273.15d0,1.d0+110.3d0/273.15d0)
  end function
  subroutine prepare_me6()
    real(c_double) :: ci(9,6),cd(9,6),cp(9,6)
    integer(c_int) :: valid_first(3),valid_last(3),inner_first(3),inner_last(3)
    integer :: h,side,p,column,half,a
    do h=1,3
    do side=1,2
      column=2*(h-1)+side;half=(2*side-3)*(2*h-1)
      do p=-4,4
        call me6_coefficients(half,p,ci(p+5,column),cd(p+5,column),cp(p+5,column))
      enddo
    enddo
    enddo
    valid_first=-hm;valid_last=upper+hm;inner_first=0;inner_last=upper
    do a=1,3
      if(periodic(a)==1) cycle
      valid_first(a)=max(-hm,-origin(a));valid_last(a)=min(valid_last(a),g(a)-origin(a))
      inner_first(a)=max(0,4-origin(a));inner_last(a)=min(upper(a),g(a)-4-origin(a))
    enddo
    call check(fluid_me6_prepare(state,int(MPI_COMM_WORLD,c_int),peers,higher,sizes,periodic, &
      valid_first,valid_last,inner_first,inner_last,mode,ci,cd,cp),'ME6 physical partition')
  end subroutine
  integer(c_int) function source(handle) result(status)
    type(c_ptr),value :: handle
    status=0
    ! The existing channel driver acts only when x is homogeneous.
    if(axis/=0) status=fluid_uniform_force(handle,[force_fixed,0.d0,0.d0])
    if(status==0) then
      if(keep_convection) then
        call observe_rhs('full')
      else
        call observe_rhs('post_source')
      endif
    endif
  end function
  subroutine prepare_lines()
    type(compact_line_partition) :: part
    type(compact_matrix) :: matrix
    type(keep_split_plan) :: split
    integer :: a,line_rank,line_comm,lines
    real(c_double) :: edge(2)
    do a=1,3
      line_rank=0;line_comm=MPI_COMM_SELF
      if(sizes(a)>1) then
        line_rank=rank;line_comm=MPI_COMM_WORLD
      endif
      call partition_compact_line(g(a),line_rank,int(sizes(a)),periodic(a)==1,part)
      if(part%first/=origin(a).or.part%local_intervals/=upper(a)) error stop 'wall compact partition mismatch'
      stencils(a)%axis=a-1;stencils(a)%first=part%first;stencils(a)%total=part%intervals
      stencils(a)%periodic=periodic(a);stencils(a)%kind=1;stencils(a)%start=0;stencils(a)%component_offset=1
      stencils(a)%active_first=first;stencils(a)%active_last=last
      lines=product(upper+1)/(upper(a)+1)
      if(compact) then
        call compact_derivative_core(part,matrix,edge)
        call prepare_distributed_compact_cpp(state,matrix,edge,part,line_comm,min(5*lines,65535),line_mode,derivative_plans(a))
      endif
      if(compact_filter) then
        call compact_filter_core(part,.49d0,matrix,edge)
        call prepare_distributed_compact_cpp(state,matrix,edge,part,line_comm,min(filter_components*lines,65535),line_mode,filter_plans(a))
        call filter_fixture(stencils(a))
      endif
    enddo
    if(compact.and.keep_convection) then
      call build_keep_split_plan(split);keep_bases=split%bases
      call check(compact_keep_prepare_cpp(state,int(split%terms,c_int),int(split%bases,c_int),split%left,split%right, &
        split%component,split%derivative,split%weight),'KEEP split table')
    endif
  end subroutine
  integer(c_int) function apply_compact(a,kind,components,offset,keep) result(status)
    integer,intent(in) :: a,kind,components,offset
    logical,intent(in) :: keep
    type(compact_stencil_cpp) :: s
    type(compact_distributed_cpp_plan) :: plan
    integer :: lines,start,nrhs
    s=stencils(a);s%kind=kind;s%component_offset=offset
    if(kind==4) then
      plan=filter_plans(a)
    else
      plan=derivative_plans(a)
    endif
    lines=product(upper+1)/(upper(a)+1)
    status=0
    do start=0,lines*components-1,plan%capacity
      s%start=start;nrhs=min(plan%capacity,lines*components-start)
      if(keep) then
        status=compact_keep_rhs_cpp(state,plan%handle,s,int(nrhs,c_int),1.4d0)
      else
        status=compact_field_rhs_cpp(state,plan%handle,s,int(nrhs,c_int))
      endif
      if(status/=0) return
      call solve_distributed_compact_cpp(plan,nrhs)
      call restore_distributed_compact_cpp(plan,nrhs)
      if(step==0.and.stage==1.and.backend==1.and.debug==0.and.start==0) call check_linear_sample(plan,kind,nrhs)
      if(keep) then
        status=compact_keep_commit_cpp(state,plan%handle,s,int(nrhs,c_int),1.4d0)
      else if(me6.and.kind==3) then
        status=compact_field_commit_wall_band_cpp(state,plan%handle,s,int(nrhs,c_int))
      else
        status=compact_field_commit_cpp(state,plan%handle,s,int(nrhs,c_int))
      endif
      if(status/=0) return
    enddo
  end function
  subroutine check_linear_sample(plan,kind,nrhs)
    type(compact_distributed_cpp_plan),intent(in) :: plan
    integer,intent(in) :: kind,nrhs
    type(compact_line_partition) :: global_part
    type(compact_matrix) :: matrix
    type(compact_reference_factor) :: factor
    real(c_double),allocatable :: local_b(:,:),local_x(:,:),global_b(:,:),global_x(:,:),reference_x(:,:)
    integer,allocatable :: counts(:),displacements(:)
    real(c_double) :: edge(2),residual,error
    integer :: n,m,columns,lr,ls,j,err
    n=plan%partition%global_n;m=plan%n;columns=min(3,nrhs)
    call MPI_Comm_rank(plan%comm,lr,err);call MPI_Comm_size(plan%comm,ls,err)
    allocate(counts(ls),displacements(ls),local_b(m,columns),local_x(m,columns), &
      global_b(n,columns),global_x(n,columns),reference_x(n,columns))
    call MPI_Allgather(m,1,MPI_INTEGER,counts,1,MPI_INTEGER,plan%comm,err)
    call MPI_Allgather(plan%partition%first,1,MPI_INTEGER,displacements,1,MPI_INTEGER,plan%comm,err)
    call compact_distributed_transfer_cpp(plan,local_b,columns,1)
    call compact_distributed_transfer_cpp(plan,local_x,columns,2)
    do j=1,columns
      call MPI_Allgatherv(local_b(:,j),m,mpi_real_type,global_b(:,j),counts,displacements,mpi_real_type,plan%comm,err)
      call MPI_Allgatherv(local_x(:,j),m,mpi_real_type,global_x(:,j),counts,displacements,mpi_real_type,plan%comm,err)
    enddo
    call partition_compact_line(plan%partition%intervals,0,1,plan%partition%periodic,global_part)
    if(kind==4) then
      call compact_filter_core(global_part,.49d0,matrix,edge)
    else
      call compact_derivative_core(global_part,matrix,edge)
    endif
    matrix%cyclic=global_part%periodic
    if(matrix%cyclic) then
      matrix%lower(1)=edge(1);matrix%upper(n)=edge(2)
    endif
    residual=compact_residual(matrix,global_x,global_b)
    call factor_compact_reference(matrix,factor);call solve_compact_reference(factor,global_b,reference_x)
    error=maxval(abs(global_x-reference_x))/max(1.d0,maxval(abs(reference_x)))
    if(.not.(residual<=1.d-12).or..not.(error<=2.d-10)) then
      write(*,*) 'COMPACT_LINEAR_SAMPLE ',rank,kind,residual,error
      error stop 'actual field linear-system residual/LU mismatch'
    endif
    residual_checks=residual_checks+1;line_residual_max=max(line_residual_max,residual);line_error_max=max(line_error_max,error)
  end subroutine
  subroutine release_lines()
    integer :: a
    integer(c_int64_t) :: metrics(2)
    do a=1,3
      if(compact) then
      call compact_distributed_metrics_cpp(derivative_plans(a),metrics,.false.)
      if(line_mode==1.and.metrics(2)/=0) error stop 'device compact interface staged on host'
      if(ranks>1.and.sizes(a)>1.and.line_mode==0.and.metrics(2)<=0) error stop 'pinned compact interface not exercised'
      interface_total=interface_total+metrics
      call release_distributed_compact_cpp(derivative_plans(a))
      endif
      if(compact_filter) then
        call compact_distributed_metrics_cpp(filter_plans(a),metrics,.false.)
        if(line_mode==1.and.metrics(2)/=0) error stop 'device filter interface staged on host'
        interface_total=interface_total+metrics
        call release_distributed_compact_cpp(filter_plans(a))
      endif
    enddo
  end subroutine
  subroutine filter_fixture(s)
    type(compact_stencil_cpp),intent(inout) :: s
    real(c_double),parameter :: a=.49d0,b=.98d0
    ! The fixed fixture uses the coefficients from filter_coefficient_cal(a,*,b).
    s%c6=[(11+10*a)/32,(15+34*a)/64,(-3+6*a)/32,(1-2*a)/64]
    s%c8=[(93+70*a)/256,(7+18*a)/32,(-7+14*a)/64,(1-2*a)/32,(-1+2*a)/256]
    s%c10=[(193+126*a)/512,(105+302*a)/512,(-15+30*a)/128,(45-90*a)/1024,(-5+10*a)/512,(1-2*a)/1024]
    s%boundary(0,:)=[(63+b)/64,(3+29*b)/32,(-15+15*b)/64,(5-5*b)/16,(-15+15*b)/64,(3-3*b)/32,(-1+b)/64]
    s%boundary(1,:)=[(1+62*a)/64,(29+6*a)/32,(15+34*a)/64,(-5+10*a)/16,(15-30*a)/64,(-3+6*a)/32,(1-2*a)/64]
    s%boundary(2,:)=[(-1+2*a)/64,(3+26*a)/32,(49+30*a)/64,(5+6*a)/16,(-15+30*a)/64,(3-6*a)/32,(-1+2*a)/64]
  end subroutine
  subroutine observe_prepared()
    call check(device_state_download(state,device_q,q,bytes(size(q))),'prepared q')
    call read_q('pre_rhs',step,stage,expected)
    call compare(q(0:im,0:jm,0:km,:),expected(0:im,0:jm,0:km,:),'prepared q',qmax)
    ! Physical-side primitive ghosts are intentionally undefined. Observe the
    ! physical box plus one actually periodic face, not their allocation padding.
    call check(device_state_snapshot_axis(state,device_rho,int(mod(axis+1,3),c_int),[0_c_int,0_c_int,0_c_int], &
      [0_c_int,0_c_int,0_c_int], &
      primitives(:,:,:,1),bytes(size(jac))),'rho')
    call check(device_state_snapshot_axis(state,device_velocity,int(mod(axis+1,3),c_int),[0_c_int,0_c_int,0_c_int], &
      [0_c_int,0_c_int,0_c_int], &
      primitives(:,:,:,2:4),bytes(3*size(jac))),'velocity')
    call check(device_state_snapshot_axis(state,device_pressure,int(mod(axis+1,3),c_int),[0_c_int,0_c_int,0_c_int], &
      [0_c_int,0_c_int,0_c_int], &
      primitives(:,:,:,5),bytes(size(jac))),'pressure')
    call check(device_state_snapshot_axis(state,device_temperature,int(mod(axis+1,3),c_int),[0_c_int,0_c_int,0_c_int], &
      [0_c_int,0_c_int,0_c_int], &
      primitives(:,:,:,6),bytes(size(jac))),'temperature')
    call filename('pre_rhs_primitives',step,stage)
    open(newunit=unit,file=trim(path),status='old',access='stream',form='unformatted')
    read(unit) h
    if(any(h/=[g,hm])) error stop 'primitive reference header'
    read(unit) global_primitives;close(unit)
    reference_primitives=global_primitives(origin(1)-hm:origin(1)+im+hm, &
      origin(2)-hm:origin(2)+jm+hm,origin(3)-hm:origin(3)+km+hm,:)
    call compare(primitives(0:im,0:jm,0:km,:),reference_primitives(0:im,0:jm,0:km,:),'prepared primitives',pmax)
  end subroutine
  subroutine observe_rhs(tag)
    character(*),intent(in) :: tag
    call check(device_state_download(state,device_rhs,rhs,bytes(size(rhs))),'RHS observation')
    call filename(tag,step,stage)
    open(newunit=unit,file=trim(path),status='old',access='stream',form='unformatted')
    read(unit) rh
    if(any(rh/=[g,5])) error stop 'RHS reference header'
    read(unit) global_rhs;close(unit)
    reference_rhs=global_rhs(origin(1):origin(1)+im,origin(2):origin(2)+jm,origin(3):origin(3)+km,:)
    call compare(rhs,reference_rhs,tag,rmax)
  end subroutine
  subroutine read_q(tag,s,r,values)
    character(*),intent(in) :: tag
    integer,intent(in) :: s,r
    real(c_double),intent(out) :: values(:,:,:,:)
    call filename(tag,s,r)
    open(newunit=unit,file=trim(path),status='old',access='stream',form='unformatted')
    read(unit) qh
    if(any(qh/=[g,hm,5])) error stop 'q reference header'
    read(unit) global_q;close(unit)
    values=global_q(origin(1)-hm:origin(1)+im+hm,origin(2)-hm:origin(2)+jm+hm,origin(3)-hm:origin(3)+km+hm,:)
  end subroutine
  subroutine filename(tag,s,r)
    character(*),intent(in) :: tag
    integer,intent(in) :: s,r
    write(suffix,'(a,i8.8,a,i2.2,a)') '.step',s,'.rk',r,'.rank00000000.bin'
    path=trim(directory)//'/outdat/rhs.'//tag//trim(suffix)
  end subroutine
  subroutine compare(actual,reference,tag,maximum)
    real(c_double),intent(in) :: actual(:,:,:,:),reference(:,:,:,:)
    character(*),intent(in) :: tag
    real(c_double),intent(inout) :: maximum
    real(c_double) :: error
    if(.not.all(ieee_is_finite(actual)).or..not.all(ieee_is_finite(reference))) error stop 'nonfinite wall observation'
    do c=1,size(actual,4)
      error=maxval(abs(actual(:,:,:,c)-reference(:,:,:,c)))/max(1.d0,maxval(abs(reference(:,:,:,c))))
      if(.not.(error<=2.d-10)) then
        write(*,*) 'WALL_STEP_DIFFERENCE ',label,axis,debug,step,stage,tag,c,error, &
          maxloc(abs(actual(:,:,:,c)-reference(:,:,:,c)))
        error stop 'wall stage reference mismatch'
      endif
      maximum=max(maximum,error)
    enddo
  end subroutine
  integer(c_size_t) function bytes(n)
    integer,intent(in) :: n
    bytes=int(n,c_size_t)*8_c_size_t
  end function
  subroutine check(status,tag)
    integer,intent(in) :: status
    character(*),intent(in) :: tag
    if(status==0) return
    write(*,*) 'WALL_STEP_STATUS ',label,axis,debug,step,stage,tag,status
    call MPI_Abort(MPI_COMM_WORLD,1,ierr)
  end subroutine
end program
