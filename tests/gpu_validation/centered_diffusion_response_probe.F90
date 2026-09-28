program centered_diffusion_response_probe
  use commvar, only: hm
  use derivative, only: diff6ec
  implicit none
  integer, parameter :: n=128, dim=n-1
  real(8), allocatable :: u(:),g(:),d(:),l(:),matrix(:,:)
  real(8) :: theta,kstar,amplitude,expected,err,pi
  integer :: i,m,unit
  character(len=1024) :: matrix_path

  allocate(u(-hm:dim+hm),g(-hm:dim+hm),d(0:dim),l(0:dim),matrix(1:dim-1,1:dim-1))
  pi=acos(-1.d0)
  do m=1,n/2
    theta=2.d0*pi*dble(m)/dble(n)
    do i=-hm,dim+hm
      if(m==n/2) then
        u(i)=(-1.d0)**i
      else
        u(i)=cos(theta*dble(modulo(i,n)))
      endif
    enddo
    d=diff6ec(u,dim,3)
    g(0:dim)=d
    do i=1,hm
      g(-i)=g(n-i)
      g(dim+i)=g(i-1)
    enddo
    l=diff6ec(g,dim,3)
    kstar=1.5d0*sin(theta)-0.3d0*sin(2.d0*theta)+sin(3.d0*theta)/30.d0
    expected=-kstar*kstar
    amplitude=dot_product(u(0:dim),l)/sum(u(0:dim)**2)
    err=maxval(abs(l-expected*u(0:dim)))
    if(err>2.d-12) error stop 'periodic response disagrees with the stencil symbol'
    if(m==1.or.m==32.or.m==60.or.m==63.or.m==64) &
      write(*,'(A,I4,3(1X,ES24.16))') 'MODE ',m,amplitude,expected,err
    if(m==n/2.and.maxval(abs(l))/=0.d0) error stop 'alternating-mode response changed'
  enddo

  ! A constant-property scalar model, not the limited AIR5 diffusion operator.
  ! Both derivative evaluations use the production physical-boundary closure.
  do m=1,dim-1
    u=0.d0
    u(m)=1.d0
    g=0.d0
    g(0:dim)=diff6ec(u,dim,4)
    l=diff6ec(g,dim,4)
    matrix(:,m)=l(1:dim-1)
  enddo
  call get_command_argument(1,matrix_path)
  if(len_trim(matrix_path)==0) error stop 'provide a new scalar matrix output path'
  open(newunit=unit,file=trim(matrix_path),status='new',action='write')
  do i=1,dim-1
    write(unit,'(*(ES25.17E3,1X))') matrix(i,:)
  enddo
  close(unit)
  write(*,'(A)') 'PASS: production diff6ec response characterized; no full-flow stability claim'
end program centered_diffusion_response_probe
