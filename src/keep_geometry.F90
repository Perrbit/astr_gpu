module keep_geometry
  use astr_precision, only: real_kind
  use astr_precision_mpi, only: mpi_real_type
  use numerical_forms, only: convection_form,viscous_form,ldc_boundary_layout
  use commvar, only: im,jm,km,hm,ia,ja,ka,conschm,difschm,lihomo,ljhomo,lkhomo
  use commarray, only: x
  use parallel, only: ig0,jg0,kg0,lio
  use derivative, only: fds_compact_i,fds_compact_j,fds_compact_k
  use compact_runtime, only: global_compact_periodic_derivative_cpu
  use mpi
  implicit none
  private
  public :: periodic_keep_geometry_active,build_keep_periodic_area
  public :: validate_keep_wall_geometry
contains
  subroutine validate_keep_wall_geometry()
    use keep_sbp63_runtime, only: keep_sbp63_enabled
    use bc, only: bctype,profile_low_order_nscbc_enabled,profile_state_boundary_enabled,low_order_y_nonreflecting_enabled, &
      low_order_incident_enabled
    use numerical_forms, only: keep_open_boundary_selected,y_nonreflecting_boundary_layout,inviscid_incident_boundary_layout, &
      simple_physical_boundary_axis
    real(real_kind) :: local_error,global_error,scale
    integer :: i,j,k,ierr,ignored,physical_axis
    if(convection_form/='keep'.and..not.(viscous_form=='me6_opti'.and. &
      (ldc_boundary_layout(bctype).or.profile_state_boundary_enabled().or.profile_low_order_nscbc_enabled().or. &
       y_nonreflecting_boundary_layout(bctype).or.low_order_incident_enabled()))) return
    if(lihomo.and.ljhomo.and.lkhomo) return
    if(inviscid_incident_boundary_layout(bctype).and..not.low_order_incident_enabled()) &
      error stop 'KEEP incident coupling requires the existing unfiltered inviscid prof/GRCBC configuration'
    if(y_nonreflecting_boundary_layout(bctype)) then
      if(.not.low_order_y_nonreflecting_enabled()) &
        error stop 'Wall/nonreflecting-y coupling requires existing 643e/643e unfiltered TGV or prof inlet configuration'
      if(lio) write(*,'(A)') 'LOW-ORDER Y52: existing geometric normal and complete viscous-source projection'
      return
    endif
    if(keep_open_boundary_selected(bctype).and..not.profile_low_order_nscbc_enabled().and. &
      .not.low_order_incident_enabled()) &
      error stop 'KEEP open boundaries require an implemented profile boundary configuration'
    if(.not.keep_sbp63_enabled.and.conschm=='643e'.and.difschm=='643e') then
      physical_axis=simple_physical_boundary_axis(bctype)
      if(physical_axis>0) then
        if(bctype(2*physical_axis-1)==41) then
          ! Existing explicit gradients/fluxes retain all nine metric entries.
          ! Pressure still uses the original coordinate-line wall extrapolation.
          if(lio) write(*,'(A)') 'KEEP WALL: explicit full-metric low-order isothermal wall'
          return
        endif
      endif
    endif
    local_error=0.e0_real_kind
    scale=max(1.e0_real_kind,maxval(abs(x(0:im,0:jm,0:km,:))))
    do k=0,km
    do j=0,jm
    do i=0,im
      local_error=max(local_error,maxval(abs(x(i,j,k,:)-[x(i,0,0,1),x(0,j,0,2),x(0,0,k,3)]))/scale)
    enddo
    enddo
    enddo
    call mpi_allreduce(local_error,global_error,1,mpi_real_type,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call mpi_abort(MPI_COMM_WORLD,ierr,ignored)
    if(global_error>128.e0_real_kind*epsilon(1.e0_real_kind)) &
      error stop 'KEEP physical ends currently require a flat axis-aligned tensor grid'
    if(lio) then
      if(convection_form/='keep') then
        if(profile_state_boundary_enabled().or.profile_low_order_nscbc_enabled().or.low_order_incident_enabled()) then
          write(*,'(A)') 'ME6 PROFILE: existing axis-aligned tensor-grid boundary geometry'
        else
          write(*,'(A)') 'ME6 LDC: existing axis-aligned tensor-grid boundary geometry'
        endif
        return
      endif
      if(keep_sbp63_enabled) then
        write(*,'(A)') 'KEEP WALL: global SBP 6/3 rows with weak isothermal SAT; axis-aligned tensor grid'
        return
      endif
      if(keep_open_boundary_selected(bctype)) then
        write(*,'(A)') 'KEEP OPEN: low-order physical-end volume rows; existing classified NSCBC faces'
        return
      endif
      if(all(bctype==[1,1,41,41,1,1])) then
        if(conschm=='643e') write(*,'(A)') 'KEEP WALL w24: second/fourth-order nodal flux rows; sixth-order KEEP interior'
        if(conschm=='643c') write(*,'(A)') 'KEEP WALL w34: global compact split with existing third/fourth-order endpoint rows'
      else
        write(*,'(A,6(I0,1X))') 'KEEP low-order physical ends: ',bctype
      endif
    endif
  end subroutine

  logical function periodic_keep_geometry_active() result(active)
    active=convection_form=='keep'.and.conschm=='643c'.and.lihomo.and.ljhomo.and.lkhomo
  end function

  pure function cross(a,b) result(c)
    real(real_kind),intent(in) :: a(3),b(3)
    real(real_kind) :: c(3)
    c=[a(2)*b(3)-a(3)*b(2),a(3)*b(1)-a(1)*b(3),a(1)*b(2)-a(2)*b(1)]
  end function

  subroutine build_keep_periodic_area(dx,area)
    real(real_kind),intent(in) :: dx(-hm:,-hm:,-hm:,:,:)
    real(real_kind),intent(inout) :: area(-hm:,-hm:,-hm:,:,:)
    real(real_kind) :: corners(3,4),background(3,3),origin(3),dw_beta(3),dw_gamma(3),w(3)
    real(real_kind),allocatable :: product(:,:),derivative(:)
    integer :: ierr,ignored,axis,beta,gamma,side,direction,inner,dims(3),totals(3),dim,lines,line,p,point(3),c,i,j,k
    corners=0.e0_real_kind
    if(ig0==0.and.jg0==0.and.kg0==0) corners(:,1)=x(0,0,0,:)
    if(ig0+im==ia.and.jg0==0.and.kg0==0) corners(:,2)=x(im,0,0,:)
    if(ig0==0.and.jg0+jm==ja.and.kg0==0) corners(:,3)=x(0,jm,0,:)
    if(ig0==0.and.jg0==0.and.kg0+km==ka) corners(:,4)=x(0,0,km,:)
    call mpi_allreduce(MPI_IN_PLACE,corners,12,mpi_real_type,MPI_SUM,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS) call mpi_abort(MPI_COMM_WORLD,ierr,ignored)
    origin=corners(:,1)
    totals=[ia,ja,ka]
    do axis=1,3
      background(:,axis)=(corners(:,axis+1)-origin)/real(totals(axis),real_kind)
    enddo
    dims=[im,jm,km]
    ! Only physical nodes are written here; existing dataswap owns face halos.
    do axis=1,3
      beta=mod(axis,3)+1;gamma=mod(axis+1,3)+1
      do k=0,km
      do j=0,jm
      do i=0,im
        dw_beta=dx(i,j,k,:,beta)-background(:,beta)
        dw_gamma=dx(i,j,k,:,gamma)-background(:,gamma)
        area(i,j,k,axis,:)=cross(background(:,beta),background(:,gamma)) &
          +cross(background(:,beta),dw_gamma)+cross(dw_beta,background(:,gamma))
      enddo
      enddo
      enddo
      do side=1,2
        direction=beta;inner=gamma
        if(side==2) then
          direction=gamma;inner=beta
        endif
        dim=dims(direction)
        lines=(im+1)*(jm+1)*(km+1)/(dim+1)
        allocate(product(-hm:dim+hm,3),derivative(0:dim))
        do line=1,lines
          do p=-hm,dim+hm
            call line_point(direction,p,line,point)
            w=x(point(1),point(2),point(3),:)-origin &
              -background(:,1)*real(ig0+point(1),real_kind) &
              -background(:,2)*real(jg0+point(2),real_kind) &
              -background(:,3)*real(kg0+point(3),real_kind)
            product(p,:)=0.5e0_real_kind*real(3-2*side,real_kind)* &
              cross(w,dx(point(1),point(2),point(3),:,inner)-background(:,inner))
          enddo
          do c=1,3
            select case(direction)
            case(1);derivative=global_compact_periodic_derivative_cpu(fds_compact_i,product(:,c),dim)
            case(2);derivative=global_compact_periodic_derivative_cpu(fds_compact_j,product(:,c),dim)
            case(3);derivative=global_compact_periodic_derivative_cpu(fds_compact_k,product(:,c),dim)
            end select
            do p=0,dim
              call line_point(direction,p,line,point)
              area(point(1),point(2),point(3),axis,c)=area(point(1),point(2),point(3),axis,c)+derivative(p)
            enddo
          enddo
        enddo
        deallocate(product,derivative)
      enddo
    enddo
    if(lio) write(*,'(A)') 'KEEP GEOMETRY G1a: affine background plus periodic-displacement conservative curl'
  end subroutine

  subroutine line_point(axis,p,line,point)
    integer,intent(in) :: axis,p,line
    integer,intent(out) :: point(3)
    select case(axis)
    case(1);point=[p,mod(line-1,jm+1),(line-1)/(jm+1)]
    case(2);point=[mod(line-1,im+1),p,(line-1)/(im+1)]
    case(3);point=[mod(line-1,im+1),(line-1)/(im+1),p]
    end select
  end subroutine
end module
