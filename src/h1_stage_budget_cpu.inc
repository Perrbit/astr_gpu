  subroutine limit_h1_stage_cpu(dt,a,b,c,stage,origin,common_source)
    use commvar, only: im,jm,km,npdci,npdcj,npdck,gamma,nstep
    use commarray, only: q,qrhs,jacob
    use boundary_high_order, only: h1_stage_budget_candidate,boundary_budget_admissible5,boundary_budget_fraction5
    use boundary_high_order, only: h1_outlet_filter_candidate,boundary_budget_margin5,boundary_budget_roundoff_fraction5
    use bc, only: audit_h1_low_boundary_cpu
    use parallel, only: exchange_h1_candidate_faces,mpileft,mpiright,mpidown,mpiup,mpiback,mpifront,pmax
    use validation_io, only: write_rhs_validation_snapshot
    use mpi, only: MPI_PROC_NULL
    real(real_kind),intent(in) :: dt,a,b,c,origin(0:,0:,0:,:)
    real(real_kind),intent(in),optional :: common_source(0:,0:,0:,:)
    integer,intent(in) :: stage
    real(real_kind),allocatable,save :: high(:,:,:,:,:),delta(:,:,:,:,:),low(:,:,:,:), &
      residual(:,:,:,:),caps(:,:,:,:),theta_output(:,:,:,:),sendbuf(:,:),recvbuf(:,:)
    real(real_kind) :: fp(5),fm(5),lowflux(5),base(5),work(5),change(5,6),scale,theta(7)
    integer :: i,j,k,axis,m,types(3),dims(3),first(3),point(3),left(3),right(3), &
      planes(3),lower(3),upper(3),nparts,bad,limited
    logical :: prescribed,open_face,valid
    if(.not.h1_stage_budget_candidate) return
    bad=0
    do k=0,km
    do j=0,jm
    do i=0,im
      if(h1_outlet_filter_candidate) then
        valid=boundary_budget_margin5(q(i,j,k,1:5),32)
      else
        valid=boundary_budget_admissible5(q(i,j,k,1:5))
      endif
      if(valid) cycle
      if(bad==0) print *, 'H1_BUDGET_INPUT_FAILURE CPU ',nstep,stage,mpirank,[i,j,k],q(i,j,k,1:5)
      bad=1
    enddo
    enddo
    enddo
    bad=pmax(bad)
    if(bad/=0) error stop 'H1 budget requires admissible physical input states'
    dims=[im,jm,km]; types=[npdci,npdcj,npdck]
    planes=[(jm+1)*(km+1),(im+1)*(km+1),(im+1)*(jm+1)]
    lower=[mpileft,mpidown,mpiback]; upper=[mpiright,mpiup,mpifront]
    if(.not.allocated(high)) then
      allocate(high(-1:im,-1:jm,-1:km,5,3),delta(-1:im,-1:jm,-1:km,5,3), &
        low(0:im,0:jm,0:km,5),residual(0:im,0:jm,0:km,5), &
        caps(-hm:im+hm,-hm:jm+hm,-hm:km+hm,7),theta_output(0:im,0:jm,0:km,7), &
        sendbuf(10*maxval(planes),2),recvbuf(10*maxval(planes),2))
    endif
    do axis=1,3
      first=0; first(axis)=-1
      do k=first(3),km
      do j=first(2),jm
      do i=first(1),im
        point=[i,j,k]
        if(((types(axis)==1.or.types(axis)==4).and.point(axis)<0).or. &
           ((types(axis)==2.or.types(axis)==4).and.point(axis)>=dims(axis))) then
          high(i,j,k,:,axis)=0.e0_real_kind; delta(i,j,k,:,axis)=0.e0_real_kind
        else
          high(i,j,k,:,axis)=high_order_interface_flux_cpu(point,axis)
          call audit_h1_low_face_cpu(point,axis,types(axis),lowflux)
          delta(i,j,k,:,axis)=high(i,j,k,:,axis)-lowflux
        endif
      enddo
      enddo
      enddo
      call exchange_planes(axis,.false.)
    enddo
    low=0.e0_real_kind
    do k=0,km
    do j=0,jm
    do i=0,im
      do axis=1,3
        point=[i,j,k]; left=point; left(axis)=left(axis)-1
        call audit_h1_low_face_cpu(point,axis,types(axis),fp)
        call audit_h1_low_face_cpu(left,axis,types(axis),fm)
        low(i,j,k,:)=low(i,j,k,:)+fm-fp
      enddo
    enddo
    enddo
    enddo
    call audit_h1_low_boundary_cpu(low)
    if(present(common_source)) then
      call write_rhs_validation_snapshot('budget_low_conv',values=low)
      call write_rhs_validation_snapshot('budget_common',values=common_source)
      low=low+common_source
    endif
    call write_rhs_validation_snapshot('budget_low',values=low)
    caps=1.e0_real_kind; residual=0.e0_real_kind; bad=0; limited=0
    do k=0,km
    do j=0,jm
    do i=0,im
      call node_type(i,j,prescribed,open_face)
      if(prescribed) cycle
      scale=c*dt/jacob(i,j,k)
      base=(a*origin(i,j,k,:)+b*q(i,j,k,:)*jacob(i,j,k)+c*dt*low(i,j,k,:))/jacob(i,j,k)
      if(h1_outlet_filter_candidate) then
        valid=boundary_budget_margin5(base,64)
      else
        valid=boundary_budget_admissible5(base)
      endif
      if(.not.valid) then
        print *, 'H1_BUDGET_LOW_FAILURE CPU ',nstep,stage,mpirank,[i,j,k],base
        bad=1
        cycle
      endif
      call node_deltas([i,j,k],change)
      residual(i,j,k,:)=qrhs(i,j,k,:)-low(i,j,k,:)-sum(change,dim=2)
      nparts=6
      if(open_face) nparts=7
      do m=1,6
        work=real(nparts,real_kind)*scale*change(:,m)
        if(h1_outlet_filter_candidate) then
          caps(i,j,k,m)=boundary_budget_roundoff_fraction5(base,work)
        else
          caps(i,j,k,m)=boundary_budget_fraction5(base,work)
        endif
      enddo
      if(open_face) then
        work=7.e0_real_kind*scale*residual(i,j,k,:)
        if(h1_outlet_filter_candidate) then
          caps(i,j,k,7)=boundary_budget_roundoff_fraction5(base,work)
        else
          caps(i,j,k,7)=boundary_budget_fraction5(base,work)
        endif
      endif
    enddo
    enddo
    enddo
    bad=pmax(bad)
    if(bad/=0) error stop 'H1 complete low-order RK state is inadmissible'
    do axis=1,3
      call exchange_planes(axis,.true.)
    enddo
    do m=1,7
      call dataswap(caps(:,:,:,m))
    enddo
    do k=0,km
    do j=0,jm
    do i=0,im
      point=[i,j,k]; theta=1.e0_real_kind
      call node_deltas(point,change)
      do axis=1,3
        left=point; right=point; left(axis)=left(axis)-1; right(axis)=right(axis)+1
        theta(2*axis-1)=min(caps(i,j,k,2*axis-1),caps(left(1),left(2),left(3),2*axis))
        theta(2*axis)=min(caps(i,j,k,2*axis),caps(right(1),right(2),right(3),2*axis-1))
      enddo
      theta(7)=caps(i,j,k,7); theta_output(i,j,k,:)=theta
      call node_type(i,j,prescribed,open_face)
      if(prescribed) cycle
      if(any(theta<1.e0_real_kind)) limited=limited+1
      do m=1,6
        if(theta(m)<1.e0_real_kind) qrhs(i,j,k,:)=qrhs(i,j,k,:)+(theta(m)-1.e0_real_kind)*change(:,m)
      enddo
      if(theta(7)<1.e0_real_kind) qrhs(i,j,k,:)=qrhs(i,j,k,:)+ &
        (theta(7)-1.e0_real_kind)*residual(i,j,k,:)
      work=(a*origin(i,j,k,:)+b*q(i,j,k,:)*jacob(i,j,k)+c*qrhs(i,j,k,:)*dt)/jacob(i,j,k)
      if(h1_outlet_filter_candidate) then
        valid=boundary_budget_margin5(work,32)
      else
        valid=boundary_budget_admissible5(work)
      endif
      if(.not.valid) then
        print *, 'H1_BUDGET_FINAL_FAILURE CPU ',nstep,stage,mpirank,[i,j,k],work
        bad=1
      endif
    enddo
    enddo
    enddo
    bad=pmax(bad)
    if(bad/=0) error stop 'H1 complete limited RK state is inadmissible'
    call write_rhs_validation_snapshot('budget_residual',values=residual)
    call write_rhs_validation_snapshot('budget_theta',values=theta_output)
    call write_rhs_validation_snapshot('limited')
    if(lio) write(*,'(a,3(1x,i0))') 'H1_BUDGET_CPU',nstep,stage,limited
  contains
    subroutine node_type(ii,jj,fixed,open)
      integer,intent(in) :: ii,jj
      logical,intent(out) :: fixed,open
      fixed=((types(1)==1.or.types(1)==4).and.ii==0).or. &
        ((types(2)==1.or.types(2)==4).and.jj==0)
      open=((types(1)==2.or.types(1)==4).and.ii==im).or. &
        ((types(2)==2.or.types(2)==4).and.jj==jm)
    end subroutine node_type

    subroutine node_deltas(node,changes)
      integer,intent(in) :: node(3)
      real(real_kind),intent(out) :: changes(5,6)
      integer :: axis,prev(3)
      do axis=1,3
        prev=node; prev(axis)=prev(axis)-1
        changes(:,2*axis-1)=delta(prev(1),prev(2),prev(3),:,axis)
        changes(:,2*axis)=-delta(node(1),node(2),node(3),:,axis)
      enddo
    end subroutine node_deltas

    subroutine exchange_planes(axis,is_cap)
      integer,intent(in) :: axis
      logical,intent(in) :: is_cap
      integer :: n,id,lo(3),hi(3),m,offset,count
      n=planes(axis); count=10
      if(is_cap) count=7
      do id=1,n
        call plane_node(axis,id,lo,hi)
        if(.not.is_cap) hi(axis)=dims(axis)-1
        do m=1,count
          offset=id+(m-1)*n
          if(is_cap) then
            sendbuf(offset,1)=caps(lo(1),lo(2),lo(3),m)
            sendbuf(offset,2)=caps(hi(1),hi(2),hi(3),m)
          elseif(m<=5) then
            sendbuf(offset,1)=high(lo(1),lo(2),lo(3),m,axis)
            sendbuf(offset,2)=high(hi(1),hi(2),hi(3),m,axis)
          else
            sendbuf(offset,1)=delta(lo(1),lo(2),lo(3),m-5,axis)
            sendbuf(offset,2)=delta(hi(1),hi(2),hi(3),m-5,axis)
          endif
        enddo
      enddo
      call exchange_h1_candidate_faces(sendbuf,recvbuf,n*count,axis,24511)
      do id=1,n
        call plane_node(axis,id,lo,hi)
        if(.not.is_cap) then
          lo(axis)=-1; hi(axis)=dims(axis)
        endif
        do m=1,count
          offset=id+(m-1)*n
          if(is_cap) then
            if(lower(axis)/=MPI_PROC_NULL) caps(lo(1),lo(2),lo(3),m)= &
              min(caps(lo(1),lo(2),lo(3),m),recvbuf(offset,1))
            if(upper(axis)/=MPI_PROC_NULL) caps(hi(1),hi(2),hi(3),m)= &
              min(caps(hi(1),hi(2),hi(3),m),recvbuf(offset,2))
          elseif(m<=5) then
            if(lower(axis)/=MPI_PROC_NULL) high(lo(1),lo(2),lo(3),m,axis)=recvbuf(offset,1)
            if(upper(axis)/=MPI_PROC_NULL) high(hi(1),hi(2),hi(3),m,axis)=recvbuf(offset,2)
          else
            if(lower(axis)/=MPI_PROC_NULL) delta(lo(1),lo(2),lo(3),m-5,axis)=recvbuf(offset,1)
            if(upper(axis)/=MPI_PROC_NULL) delta(hi(1),hi(2),hi(3),m-5,axis)=recvbuf(offset,2)
          endif
        enddo
      enddo
    end subroutine exchange_planes

    subroutine plane_node(axis,id,lo,hi)
      integer,intent(in) :: axis,id
      integer,intent(out) :: lo(3),hi(3)
      select case(axis)
      case(1); lo=[0,mod(id-1,jm+1),(id-1)/(jm+1)]
      case(2); lo=[mod(id-1,im+1),0,(id-1)/(im+1)]
      case(3); lo=[mod(id-1,im+1),(id-1)/(im+1),0]
      end select
      hi=lo; hi(axis)=dims(axis)
    end subroutine plane_node
  end subroutine limit_h1_stage_cpu
