module keep_geometry
  use astr_precision, only: real_kind
  use astr_precision_mpi, only: mpi_real_type
  use numerical_forms, only: convection_form
  use commvar, only: im,jm,km,hm,ia,ja,ka,conschm
  use commarray, only: x
  use parallel, only: ig0,jg0,kg0,lio
  use derivative, only: fds_compact_i,fds_compact_j,fds_compact_k
  use compact_runtime, only: global_compact_periodic_derivative_cpu
  use mpi
  implicit none
  private
  public :: periodic_keep_geometry_active,build_keep_periodic_area
contains
  logical function periodic_keep_geometry_active() result(active)
    active=convection_form=='keep'.and.conschm=='643c'
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
