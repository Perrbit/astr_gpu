module compact_runtime
  use astr_precision, only: real_kind
  use astr_precision_mpi, only: mpi_real_type
  use mpi
  use commvar, only: hm
  use compact_operator
  use derivative, only: finite_difference,fds,diff6ec
  use numerical_forms, only: compact_filter_selected,filter_scheme,convection_form,viscous_form, &
    simple_physical_boundary_axis,ldc_boundary_layout,x_open_boundary_layout
  use keep_operator, only: keep_split_plan,build_keep_split_plan,keep_split_variables, &
    keep_monomial,keep_split_accumulate
  implicit none
  private
  public :: configure_global_compact,global_compact_enabled,compact_partitions,compact_line_comms
  public :: global_compact_derivative_cpu,global_compact_filter_cpu,release_global_compact
  public :: begin_compact_geometry,end_compact_geometry
  public :: global_compact_filter_enabled
  public :: global_compact_gradient_enabled,geometry_central
  public :: global_compact_periodic_derivative_cpu
  public :: global_keep_convection_cpu
  type,extends(finite_difference) :: global_compact_central
  contains
    procedure :: central=>global_compact_derivative_cpu
  end type
  logical,save :: enabled=.false.
  logical,save :: resources_ready=.false.
  logical,save :: geometry_phase=.false.
  type(compact_line_partition),save :: compact_partitions(3)
  integer,save :: compact_line_comms(3)=MPI_COMM_NULL
  type(compact_cpu_plan),save :: derivative_plans(3),filter_plans(3)
  type(compact_cpu_plan),save :: keep_plans(3)
  type(keep_split_plan),save :: keep_split
contains
  subroutine begin_compact_geometry()
    use commvar, only: flowtype,conschm,difschm
    use bc, only: bctype
    use parallel, only: lio
    if(convection_form=='keep') then
      call configure_global_compact(bctype)
      geometry_phase=.true.
      if(lio) write(*,'(A,A)') 'KEEP GEOMETRY G1: conservative curl; derivative=',conschm
      return
    endif
    if(viscous_form=='me6_opti'.and.difschm=='643c') then
      call configure_global_compact(bctype)
      geometry_phase=.true.
      if(lio) write(*,'(A)') 'ME6 COMPACT GEOMETRY: global coupling with affine-periodic coordinate increments'
      return
    endif
    if(trim(flowtype)/='bl'.or.conschm/='743e'.or.difschm/='643c') return
    call configure_global_compact(bctype)
    geometry_phase=.true.
    if(lio) write(*,'(A)') 'COMPACT GEOMETRY: global coupling with affine-periodic coordinate increments'
  end subroutine

  function geometry_central(asolver,f,dim) result(df)
    use commtype, only: compact_scheme
    use commvar, only: conschm
    type(compact_scheme),intent(in) :: asolver
    integer,intent(in) :: dim
    real(real_kind),intent(in) :: f(-hm:dim+hm)
    real(real_kind) :: df(0:dim)
    type(global_compact_central) :: geometry_operator
    if(convection_form/='keep') then
      df=fds%central(asolver,f,dim)
    elseif(conschm=='643e') then
      df=diff6ec(f,dim,asolver%nbctype)
    elseif(conschm=='643c') then
      df=global_compact_derivative_cpu(geometry_operator,asolver,f,dim)
    else
      error stop 'KEEP geometry requires its explicit D6 or global Pade derivative'
    endif
  end function

  subroutine end_compact_geometry()
    geometry_phase=.false.
  end subroutine

  logical function global_compact_enabled() result(active)
    active=enabled
  end function

  logical function global_compact_gradient_enabled() result(active)
    use commvar, only: difschm
    active=enabled.and.difschm=='643c'
  end function

  logical function global_compact_filter_enabled() result(active)
    use commvar, only: lfilter,conschm,difschm
    active=resources_ready.and.lfilter.and.compact_filter_selected(conschm,difschm)
  end function

  subroutine configure_global_compact(boundaries)
    use commvar, only: conschm,difschm,flowtype,ndims,numq,num_species,num_modequ,turbmode, &
      rkscheme,nondimen,lihomo,ljhomo,lkhomo,lcomb,lcracon,lfilter,lrestart,im,jm,km,ia,ja,ka, &
      diffterm,recon_schem,lchardecomp
    use bc, only: turbinf,nscbc_farfield_incoming_only_enabled,nscbc_farfield_sbli_shock_enabled, &
      profile_low_order_nscbc_enabled,profile_state_boundary_enabled,low_order_incident_enabled
    use commarray, only: x
    use parallel, only: mpi_igroup,mpi_jgroup,mpi_kgroup,irk,jrk,krk,isize,jsize,ksize,ig0,jg0,kg0,lio
    integer,intent(in) :: boundaries(:)
    integer :: axis,ierr,groups(3),coords(3),ranks(3),dims(3),totals(3),first(3),i,j,k
    logical :: periodic(3),tgv,plate,keep_wall,keep_open
    if(resources_ready) return
    enabled=conschm=='643c'.or.difschm=='643c'
    if(.not.enabled.and..not.(lfilter.and.filter_scheme=='compact10')) return
    tgv=conschm=='643c'.and.difschm=='643c'.and.trim(flowtype)=='tgv'.and. &
      lihomo.and.ljhomo.and.lkhomo.and.all(boundaries==1)
    if(convection_form=='keep'.or.viscous_form=='me6_opti') tgv=all(boundaries==1)
    keep_wall=(convection_form=='keep'.or.viscous_form=='me6_opti').and. &
      (simple_physical_boundary_axis(boundaries)>0.or.x_open_boundary_layout(boundaries).or.profile_state_boundary_enabled().or. &
       (flowtype=='ldcavity'.and.ldc_boundary_layout(boundaries)))
    keep_open=profile_low_order_nscbc_enabled().or.low_order_incident_enabled()
    plate=conschm=='743e'.and.difschm=='643c'.and.trim(flowtype)=='bl'.and. &
      diffterm.and.recon_schem==5.and.lchardecomp.and.trim(turbinf)=='prof'.and. &
      .not.lihomo.and..not.ljhomo.and.lkhomo
    if(plate) then
      plate=all(boundaries==[11,21,41,50,1,1]).or. &
        (all(boundaries==[11,22,41,52,1,1]).and. &
        nscbc_farfield_incoming_only_enabled().and..not.nscbc_farfield_sbli_shock_enabled())
    endif
    if(enabled) then
      if((.not.tgv.and..not.plate.and..not.keep_wall.and..not.keep_open).or.ndims/=3.or.numq/=5.or. &
        num_species/=0.or.num_modequ/=0.or.trim(turbmode)/='none'.or.trim(rkscheme)/='rk3'.or. &
        .not.nondimen.or.lcomb.or.(lcracon.and..not.plate.and.convection_form/='keep')) &
        error stop 'Global compact requires a periodic case, KEEP wall/open boundary, or CP6 profile plate'
    else
      if((any(boundaries/=1).and..not.keep_wall.and..not.keep_open).or.ndims/=3.or.numq/=5.or.num_species/=0.or.num_modequ/=0.or.lcomb) &
        error stop 'Independent global compact filter requires a periodic field or KEEP wall/open boundary'
    endif
    if(lrestart) error stop 'Legacy compact restart is disabled; use output restore_directory'
    if(hm<2.or.(lfilter.and.hm<5)) error stop 'Insufficient halo for global compact RHS'
    if(plate) then
      do k=0,km
      do j=0,jm
      do i=0,im
        if(any(x(i,j,k,:)/=[x(i,0,0,1),x(0,j,0,2),x(0,0,k,3)])) &
          error stop 'CP6 compact plate requires separable Cartesian coordinates'
      enddo
      enddo
      enddo
      if(any(x(1:im,0,0,1)<=x(0:im-1,0,0,1)).or. &
        any(x(0,1:jm,0,2)<=x(0,0:jm-1,0,2)).or. &
        any(x(0,0,1:km,3)<=x(0,0,0:km-1,3))) &
        error stop 'CP6 compact plate requires strictly increasing coordinates; check the selected grid UDF'
    endif
    groups=[mpi_igroup,mpi_jgroup,mpi_kgroup]
    coords=[irk,jrk,krk]
    ranks=[isize,jsize,ksize]
    dims=[im,jm,km]
    totals=[ia,ja,ka]
    first=[ig0,jg0,kg0]
    periodic=[boundaries(1)==1.and.boundaries(2)==1,boundaries(3)==1.and.boundaries(4)==1, &
      boundaries(5)==1.and.boundaries(6)==1]
    do axis=1,3
      call MPI_Comm_split(groups(axis),0,coords(axis),compact_line_comms(axis),ierr)
      ! ASTR assigns remainder intervals to high ranks; use its actual bounds.
      compact_partitions(axis)=compact_line_partition(intervals=totals(axis),global_n=totals(axis), &
        first=first(axis),local_intervals=dims(axis),owned=dims(axis),rank=coords(axis), &
        ranks=ranks(axis),periodic=periodic(axis))
      if(.not.periodic(axis)) then
        compact_partitions(axis)%global_n=totals(axis)+1
        if(coords(axis)==ranks(axis)-1) compact_partitions(axis)%owned=dims(axis)+1
      endif
    enddo
    if(global_compact_gradient_enabled()) then
      deallocate(fds)
      allocate(global_compact_central::fds)
    endif
    resources_ready=.true.
    if(lio) then
      if(tgv) write(*,'(A)') 'COMPACT GLOBAL: periodic unique-node derivative/filter; CPU direct / GPU two-level PCR'
      if(keep_wall) write(*,'(A)') 'COMPACT GLOBAL: low-order physical ends; MPI interfaces remain globally coupled'
      if(keep_open) write(*,'(A)') 'COMPACT GLOBAL: KEEP physical open ends; MPI interfaces remain globally coupled'
      if(plate) write(*,'(A)') 'COMPACT GLOBAL: physical-end plate derivative; CPU direct / GPU two-level PCR'
      if(.not.enabled) write(*,'(A)') 'COMPACT GLOBAL: independent filter; existing geometry/derivative unchanged'
    endif
  end subroutine

  subroutine prepare_cpu_direction(axis,filter)
    use commvar, only: alfa_filter
    use compact_line_solver, only: compact_matrix
    integer,intent(in) :: axis
    logical,intent(in) :: filter
    type(compact_matrix) :: matrix
    real(real_kind) :: edge(2),alpha
    alpha=1.e0_real_kind/3.e0_real_kind
    if(filter) alpha=alfa_filter
    if(filter) then
      call compact_filter_core(compact_partitions(axis),alpha,matrix,edge)
    else
      call compact_derivative_core(compact_partitions(axis),matrix,edge)
    endif
    if(filter) then
      call prepare_compact_cpu(matrix,edge,compact_partitions(axis),compact_line_comms(axis),5,filter_plans(axis))
    else
      call prepare_compact_cpu(matrix,edge,compact_partitions(axis),compact_line_comms(axis),1,derivative_plans(axis))
    endif
  end subroutine

  function global_compact_derivative_cpu(this,asolver,f,dim) result(df)
    use commtype, only: compact_scheme
    class(global_compact_central),intent(in) :: this
    type(compact_scheme),intent(in) :: asolver
    integer,intent(in) :: dim
    real(real_kind),intent(in) :: f(-hm:dim+hm)
    real(real_kind) :: df(0:dim)
    df=compact_derivative_line(asolver,f,dim,geometry_phase)
  end function

  function global_compact_periodic_derivative_cpu(asolver,f,dim) result(df)
    use commtype, only: compact_scheme
    type(compact_scheme),intent(in) :: asolver
    integer,intent(in) :: dim
    real(real_kind),intent(in) :: f(-hm:dim+hm)
    real(real_kind) :: df(0:dim)
    df=compact_derivative_line(asolver,f,dim,.false.)
  end function

  function compact_derivative_line(asolver,f,dim,affine) result(df)
    use commtype, only: compact_scheme
    type(compact_scheme),intent(in) :: asolver
    integer,intent(in) :: dim
    logical,intent(in) :: affine
    real(real_kind),intent(in) :: f(-hm:dim+hm)
    real(real_kind) :: df(0:dim),values(dim+1,1),input(-hm:dim+hm,1),ends(2),slope
    real(real_kind),allocatable :: rhs(:,:),solution(:,:)
    integer :: axis,i,ierr
    axis=asolver%direction
    if(axis<1.or.axis>3) error stop 'global derivative requires an explicit direction descriptor'
    if(derivative_plans(axis)%n==0) call prepare_cpu_direction(axis,.false.)
    allocate(rhs(compact_partitions(axis)%owned,1),solution(compact_partitions(axis)%owned,1))
    input(:,1)=f
    slope=0.e0_real_kind
    if(affine.and.compact_partitions(axis)%periodic) then
      ! Coordinates and metric products can be affine-periodic, not periodic scalars.
      if(compact_partitions(axis)%rank==0) ends(1)=f(0)
      if(compact_partitions(axis)%rank==compact_partitions(axis)%ranks-1) ends(2)=f(dim)
      call MPI_Bcast(ends(1),1,mpi_real_type,0,compact_line_comms(axis),ierr)
      call MPI_Bcast(ends(2),1,mpi_real_type,compact_partitions(axis)%ranks-1, &
        compact_line_comms(axis),ierr)
      slope=(ends(2)-ends(1))/real(compact_partitions(axis)%intervals,kind=real_kind)
      do i=-hm,dim+hm
        input(i,1)=f(i)-slope*real(compact_partitions(axis)%first+i,kind=real_kind)
      enddo
    endif
    call compact_derivative_rhs(compact_partitions(axis),dim,hm,input,rhs)
    call solve_compact_cpu(derivative_plans(axis),rhs,solution,values,1)
    df=values(:,1)+slope
  end function

  subroutine global_keep_convection_cpu()
    use commvar, only: im,jm,km,gamma,is,ie,js,je,ks,ke
    use commarray, only: rho,vel,prs,jacob,dxi,qrhs
    use compact_line_solver, only: compact_matrix
    real(real_kind),allocatable :: input(:,:),rhs(:,:),owned(:,:),values(:,:)
    real(real_kind) :: variables(9),metric(3),flux(5)
    integer :: axis,dims(3),dim,lines,line,pos,point(3),b
    if(keep_split%terms==0) call build_keep_split_plan(keep_split)
    dims=[im,jm,km]
    do axis=1,3
      dim=dims(axis)
      lines=(im+1)*(jm+1)*(km+1)/(dim+1)
      if(keep_plans(axis)%n==0) then
        block
        type(compact_matrix) :: matrix
        real(real_kind) :: edge(2)
        call compact_derivative_core(compact_partitions(axis),matrix,edge)
        call prepare_compact_cpu(matrix,edge,compact_partitions(axis),compact_line_comms(axis), &
          keep_split%bases,keep_plans(axis))
        end block
      endif
      allocate(input(-hm:dim+hm,keep_split%bases),rhs(compact_partitions(axis)%owned,keep_split%bases), &
        owned(compact_partitions(axis)%owned,keep_split%bases),values(0:dim,keep_split%bases))
      do line=1,lines
        do pos=-hm,dim+hm
          if(.not.compact_partitions(axis)%periodic) then
            if(compact_partitions(axis)%first+pos<0.or. &
              compact_partitions(axis)%first+pos>compact_partitions(axis)%intervals) then
              input(pos,:)=0.e0_real_kind
              cycle
            endif
          endif
          select case(axis)
          case(1);point=[pos,mod(line-1,jm+1),(line-1)/(jm+1)]
          case(2);point=[mod(line-1,im+1),pos,(line-1)/(im+1)]
          case(3);point=[mod(line-1,im+1),(line-1)/(im+1),pos]
          end select
          metric=jacob(point(1),point(2),point(3))*dxi(point(1),point(2),point(3),axis,:)
          call keep_split_variables(rho(point(1),point(2),point(3)),vel(point(1),point(2),point(3),:), &
            prs(point(1),point(2),point(3)),metric,gamma,variables)
          do b=1,keep_split%bases
            input(pos,b)=keep_monomial(variables,keep_split%right(:,b))
          enddo
        enddo
        call compact_derivative_rhs(compact_partitions(axis),dim,hm,input,rhs)
        call solve_compact_cpu(keep_plans(axis),rhs,owned,values,keep_split%bases)
        do pos=0,dim
          select case(axis)
          case(1);point=[pos,mod(line-1,jm+1),(line-1)/(jm+1)]
          case(2);point=[mod(line-1,im+1),pos,(line-1)/(im+1)]
          case(3);point=[mod(line-1,im+1),(line-1)/(im+1),pos]
          end select
          if(any(point<[is,js,ks]).or.any(point>[ie,je,ke])) cycle
          metric=jacob(point(1),point(2),point(3))*dxi(point(1),point(2),point(3),axis,:)
          call keep_split_variables(rho(point(1),point(2),point(3)),vel(point(1),point(2),point(3),:), &
            prs(point(1),point(2),point(3)),metric,gamma,variables)
          call keep_split_accumulate(keep_split,variables,values(pos,:),flux)
          qrhs(point(1),point(2),point(3),1:5)=qrhs(point(1),point(2),point(3),1:5)+flux
        enddo
      enddo
      deallocate(input,rhs,owned,values)
    enddo
  end subroutine

  subroutine global_compact_filter_cpu(q,coefficients)
    use commvar, only: im,jm,km,numq
    use filter, only: coef6i,coef8i,coefb
    use parallel, only: dataswap
    real(real_kind),intent(inout) :: q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,numq)
    real(real_kind),intent(in) :: coefficients(0:5)
    real(real_kind),allocatable :: f(:,:),rhs(:,:),solution(:,:),values(:,:)
    integer :: dims(3),axis,dim,lines,line,i,j,k,p,offset
    dims=[im,jm,km]
    do axis=1,3
      if(filter_plans(axis)%n==0) call prepare_cpu_direction(axis,.true.)
      dim=dims(axis)
      lines=(im+1)*(jm+1)*(km+1)/(dim+1)
      call dataswap(q,direction=axis)
      allocate(f(-hm:dim+hm,numq),rhs(compact_partitions(axis)%owned,numq), &
        solution(compact_partitions(axis)%owned,numq),values(dim+1,numq))
      do line=1,lines
        select case(axis)
        case(1)
          j=mod(line-1,jm+1); k=(line-1)/(jm+1)
          f=q(:,j,k,:)
        case(2)
          i=mod(line-1,im+1); k=(line-1)/(im+1)
          f=q(i,:,k,:)
        case(3)
          i=mod(line-1,im+1); j=(line-1)/(im+1)
          f=q(i,j,:,:)
        end select
        call compact_filter_rhs_global(compact_partitions(axis),dim,hm,f,coef6i,coef8i, &
          coefficients,coefb(0:2,0:6),rhs)
        call solve_compact_cpu(filter_plans(axis),rhs,solution,values,numq)
        if(.not.compact_partitions(axis)%periodic) then
          if(compact_partitions(axis)%first==0) values(1,:)=f(0,:)
          if(compact_partitions(axis)%first+dim==compact_partitions(axis)%intervals) &
            values(dim+1,:)=f(dim,:)
        endif
        select case(axis)
        case(1)
          q(0:im,j,k,:)=values
        case(2)
          q(i,0:jm,k,:)=values
        case(3)
          q(i,j,0:km,:)=values
        end select
      enddo
      deallocate(f,rhs,solution,values)
    enddo
  end subroutine

  subroutine release_global_compact()
    integer :: axis,ierr
    if(.not.resources_ready) return
    do axis=1,3
      call release_compact_cpu(derivative_plans(axis))
      call release_compact_cpu(filter_plans(axis))
      call release_compact_cpu(keep_plans(axis))
      call MPI_Comm_free(compact_line_comms(axis),ierr)
    enddo
    enabled=.false.
    resources_ready=.false.
  end subroutine
end module
