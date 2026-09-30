module insitu_fields
  use mpi
  use iso_fortran_env, only: real64
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: capture_sample,canonicalize_sample,derive_sample
  public :: complete_periodic_endpoints
contains
  subroutine require_fields(ok,message)
    logical,intent(in) :: ok
    character(*),intent(in) :: message
    integer :: bad,any_bad,ierr,ignored
    bad=merge(0,1,ok)
    call MPI_Allreduce(bad,any_bad,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS.or.any_bad/=0) then
      write(*,'(A)') 'ASTR INSITU FIELD ERROR: '//message
      call MPI_Abort(MPI_COMM_WORLD,1,ignored)
    endif
  end subroutine


  subroutine capture_sample(fields)
    use commvar, only: im,jm,km,numq,num_species,use_gpu
    use commarray, only: q,rho,vel,prs,tmp
#ifdef _CUDA
    use insitu_sample_gpu, only: download_sample_gpu
#endif
    real(real64),intent(out) :: fields(0:im,0:jm,0:km,11)
    call require_fields(numq==5.and.num_species==0,'sampling requires five-variable perfect gas')
#ifdef _CUDA
    if(use_gpu) then
      call download_sample_gpu(fields)
    else
#else
    call require_fields(.not.use_gpu,'GPU sampling requires CUDA build')
#endif
      fields(:,:,:,1:5)=q(0:im,0:jm,0:km,1:5)
      fields(:,:,:,6)=rho(0:im,0:jm,0:km)
      fields(:,:,:,7:9)=vel(0:im,0:jm,0:km,1:3)
      fields(:,:,:,10)=prs(0:im,0:jm,0:km)
      fields(:,:,:,11)=tmp(0:im,0:jm,0:km)
#ifdef _CUDA
    endif
#endif
    call require_fields(all(ieee_is_finite(fields)),'nonfinite sample')
  end subroutine

  subroutine canonicalize_sample(fields)
    use commvar, only: im,jm,km,const2,const6,numq,num_species
    real(real64),intent(inout) :: fields(0:im,0:jm,0:km,11)
    integer :: axis
    call require_fields(numq==5.and.num_species==0,'ownership requires five-variable perfect gas')
    call complete_periodic_endpoints(fields(:,:,:,1:5))
    call require_fields(all(fields(:,:,:,1)>0.d0),'nonpositive diagnostic density')
    fields(:,:,:,6)=fields(:,:,:,1)
    do axis=1,3
      fields(:,:,:,6+axis)=fields(:,:,:,axis+1)/fields(:,:,:,6)
    enddo
    fields(:,:,:,10)=(fields(:,:,:,5)-0.5d0*fields(:,:,:,6)* &
                      sum(fields(:,:,:,7:9)**2,dim=4))/const6
    fields(:,:,:,11)=fields(:,:,:,10)/fields(:,:,:,6)*const2
    call require_fields(all(ieee_is_finite(fields)).and.all(fields(:,:,:,10)>0.d0), &
                        'invalid canonical primitive state')
  end subroutine

  subroutine complete_periodic_endpoints(fields)
    use commvar, only: im,jm,km
    use bc, only: bctype
    use parallel, only: isize,jsize,ksize,mpileft,mpiright,mpidown,mpiup,mpiback,mpifront
    real(real64),intent(inout) :: fields(0:,0:,0:,:)
    real(real64),allocatable :: send(:),receive(:)
    integer :: axis,count,status,ierr,comm,neighbors(2,3),sizes(3),extent(3),max_count,nfields
    nfields=size(fields,4)
    call require_fields(all(shape(fields)==[im+1,jm+1,km+1,nfields]).and.nfields>0, &
                        'invalid endpoint array shape')
    call require_fields(all(bctype==1),'ownership requires periodic boundaries')
    ! Each rank owns [0,im) x [0,jm) x [0,km); periodic upper endpoints map to zero.
    sizes=[isize,jsize,ksize]
    extent=[im,jm,km]
    neighbors(:,1)=[mpileft,mpiright]
    neighbors(:,2)=[mpidown,mpiup]
    neighbors(:,3)=[mpiback,mpifront]
    max_count=nfields*max((im+1)*(jm+1),(im+1)*(km+1),(jm+1)*(km+1))
    allocate(send(max_count),receive(max_count),stat=status)
    call require_fields(status==0,'cannot allocate ownership exchange buffers')
    call MPI_Comm_dup(MPI_COMM_WORLD,comm,ierr)
    call require_fields(ierr==MPI_SUCCESS,'cannot create diagnostic communicator')
    ! Sequential face copies carry already reconciled edges into the next axis.
    do axis=1,3
      count=nfields*product(extent+1)/(extent(axis)+1)
      select case(axis)
      case(1)
        send(1:count)=reshape(fields(0,:,:,:),[count])
      case(2)
        send(1:count)=reshape(fields(:,0,:,:),[count])
      case(3)
        send(1:count)=reshape(fields(:,:,0,:),[count])
      end select
      if(sizes(axis)==1) then
        receive(1:count)=send(1:count)
      else
        call require_fields(all(neighbors(:,axis)/=MPI_PROC_NULL),'nonperiodic diagnostic neighbor')
        call MPI_Sendrecv(send,count,MPI_DOUBLE_PRECISION,neighbors(1,axis),axis, &
                          receive,count,MPI_DOUBLE_PRECISION,neighbors(2,axis),axis, &
                          comm,MPI_STATUS_IGNORE,ierr)
        call require_fields(ierr==MPI_SUCCESS,'ownership exchange failed')
      endif
      select case(axis)
      case(1)
        fields(im,:,:,:)=reshape(receive(1:count),[jm+1,km+1,nfields])
      case(2)
        fields(:,jm,:,:)=reshape(receive(1:count),[im+1,km+1,nfields])
      case(3)
        fields(:,:,km,:)=reshape(receive(1:count),[im+1,jm+1,nfields])
      end select
    enddo
    call MPI_Comm_free(comm,ierr)
    call require_fields(ierr==MPI_SUCCESS,'cannot release diagnostic communicator')
    deallocate(send,receive)
  end subroutine

  subroutine derive_sample(velocity,derived)
    use commvar, only: im,jm,km,hm,difschm
    use bc, only: bctype
    use commarray, only: dxi
    use parallel, only: dataswap
    use derivative, only: diff6ec
    real(real64),intent(in) :: velocity(0:im,0:jm,0:km,3)
    real(real64),intent(out) :: derived(0:im,0:jm,0:km,14)
    real(real64),allocatable :: work(:,:,:,:)
    real(real64) :: a(3,3)
    integer :: i,j,k,c,d,status
    call require_fields(all(bctype==1).and.hm>=3.and.trim(difschm)=='643e', &
      'diagnostics require periodic boundaries and explicit sixth-order derivatives')
    call require_fields(min(im,jm,km)>=3,'diagnostic subdomains need at least three cells per axis')
    allocate(work(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3),stat=status)
    call require_fields(status==0,'cannot allocate private diagnostic halo')
    work=0.d0
    work(0:im,0:jm,0:km,:)=velocity
    ! Blocking exchange updates only this private halo, not solver primitive arrays.
    call dataswap(work)
    call require_fields(all(work(0:im,0:jm,0:km,:)==velocity),'diagnostic exchange changed physical values')
    derived=0.d0
    do c=1,3
      do d=1,3
        do k=0,km
          do j=0,jm
            derived(:,j,k,c+3*(d-1))=diff6ec(work(:,j,k,c),im,3)*dxi(0:im,j,k,1,d)
          enddo
        enddo
        do k=0,km
          do i=0,im
            derived(i,:,k,c+3*(d-1))=derived(i,:,k,c+3*(d-1))+ &
              diff6ec(work(i,:,k,c),jm,3)*dxi(i,0:jm,k,2,d)
          enddo
        enddo
        do j=0,jm
          do i=0,im
            derived(i,j,:,c+3*(d-1))=derived(i,j,:,c+3*(d-1))+ &
              diff6ec(work(i,j,:,c),km,3)*dxi(i,j,0:km,3,d)
          enddo
        enddo
      enddo
    enddo
    do k=0,km
      do j=0,jm
        do i=0,im
          a=reshape(derived(i,j,k,1:9),[3,3])
          derived(i,j,k,10)=-0.5d0*sum(a*transpose(a))
          derived(i,j,k,11)=a(1,1)+a(2,2)+a(3,3)
          derived(i,j,k,12)=a(3,2)-a(2,3)
          derived(i,j,k,13)=a(1,3)-a(3,1)
          derived(i,j,k,14)=a(2,1)-a(1,2)
        enddo
      enddo
    enddo
    deallocate(work)
    call require_fields(all(ieee_is_finite(derived)),'nonfinite derived sample')
  end subroutine

end module
