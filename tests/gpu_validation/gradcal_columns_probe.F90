module gradcal_columns_test
  use astr_precision, only: real_kind
  use derivative, only: finite_difference,explicit_central
  use commtype, only: compact_scheme
  use commvar, only: hm
  implicit none
  integer :: calls=0
  type,extends(finite_difference) :: counted_difference
  contains
    procedure :: central=>counted_central
  end type
contains
  function counted_central(this,asolver,f,dim) result(df)
    class(counted_difference),intent(in) :: this
    type(compact_scheme),intent(in) :: asolver
    integer,intent(in) :: dim
    real(real_kind),intent(in) :: f(-hm:dim+hm)
    real(real_kind) :: df(0:dim)
    type(explicit_central) :: production
    calls=calls+1
    df=production%central(asolver,f,dim)
  end function
end module

program gradcal_columns_probe
  use mpi
  use astr_precision, only: real_kind
  use gradcal_columns_test, only: counted_difference,calls
  use derivative, only: fds,fd_scheme_initiate,fds_compact_i,fds_compact_j,fds_compact_k
  use comsolver, only: gradcal
  use commvar, only: im,jm,km,hm,ndims,num_species,num_modequ,turbmode,ltimrpt,lreport
  use commarray, only: vel,tmp,spc,dxi,dvel,dtmp,dspc,tke,omg,dtke,domg
  implicit none
  real(real_kind),parameter :: slopes(3,3)=reshape([1.0_real_kind,-2.0_real_kind,0.3_real_kind, &
    2.0_real_kind,0.5_real_kind,-0.2_real_kind,3.0_real_kind,1.0_real_kind,0.7_real_kind],[3,3])
  real(real_kind),parameter :: temperature_slopes(3)=[0.02_real_kind,0.03_real_kind,0.01_real_kind]
  integer :: ierr,i,j,k,m,a,b,mode,expected_calls,columns
  real(real_kind) :: position(3),error
  call MPI_Init(ierr)
  im=8;jm=8;km=8;ndims=3;lreport=.false.;ltimrpt=.false.
  allocate(vel(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3),tmp(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
    spc(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),dxi(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3), &
    dvel(0:im,0:jm,0:km,3,3),dtmp(0:im,0:jm,0:km,3),dspc(0:im,0:jm,0:km,5,3), &
    tke(-hm:im+hm,-hm:jm+hm,-hm:km+hm),omg(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
    dtke(0:im,0:jm,0:km,3),domg(0:im,0:jm,0:km,3))
  allocate(counted_difference::fds)
  call fd_scheme_initiate(fds_compact_i,'643e',4,im,1)
  call fd_scheme_initiate(fds_compact_j,'643e',4,jm,2)
  call fd_scheme_initiate(fds_compact_k,'643e',4,km,3)
  dxi=0.0_real_kind
  do a=1,3
    dxi(:,:,:,a,a)=1.0_real_kind
  enddo
  do k=-hm,km+hm
    do j=-hm,jm+hm
      do i=-hm,im+hm
        position=real([i,j,k],real_kind)
        vel(i,j,k,:)=matmul(slopes,position)
        tmp(i,j,k)=2.0_real_kind+dot_product(temperature_slopes,position)
        do m=1,5
          spc(i,j,k,m)=0.2_real_kind+real(m,real_kind)*0.001_real_kind*sum(position)
        enddo
        tke(i,j,k)=2.0_real_kind+2.0_real_kind*sum(position)
        omg(i,j,k)=3.0_real_kind+3.0_real_kind*sum(position)
      enddo
    enddo
  enddo
  do mode=1,3
    turbmode='none';num_species=0;num_modequ=0;columns=4
    if(mode==2) then
      num_species=5;num_modequ=1;columns=9
    elseif(mode==3) then
      turbmode='k-omega';num_modequ=2;columns=6
    endif
    calls=0
    call gradcal()
    expected_calls=3*(im+1)**2*columns
    if(calls/=expected_calls) then
      write(*,'(A,3(1X,I0))') 'GRADCAL_COLUMN_COUNT',mode,calls,expected_calls
      error stop 'gradcal differentiated an unfilled column'
    endif
    error=0.0_real_kind
    do b=1,3
      do a=1,3
        error=max(error,maxval(abs(dvel(:,:,:,a,b)-slopes(a,b))))
      enddo
      error=max(error,maxval(abs(dtmp(:,:,:,b)-temperature_slopes(b))))
      if(mode==2) then
        do m=1,5
          error=max(error,maxval(abs(dspc(:,:,:,m,b)-0.001_real_kind*m)))
        enddo
      endif
      if(mode==3) then
        error=max(error,maxval(abs(dtke(:,:,:,b)-2.0_real_kind)),maxval(abs(domg(:,:,:,b)-3.0_real_kind)))
      endif
    enddo
    if(error>1.e-11_real_kind) error stop 'gradcal affine gradient mismatch'
  enddo
  print *, 'GRADCAL_COLUMNS_PASS'
  call MPI_Finalize(ierr)
end program
