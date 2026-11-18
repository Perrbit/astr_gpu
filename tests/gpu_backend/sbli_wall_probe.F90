program sbli_wall_probe
  use iso_c_binding
  use mpi
  use ieee_arithmetic, only: ieee_is_finite
  use device_state
  use fluid_explicit
  use insitu_wall_cpp
  implicit none
  integer,parameter :: real_kind=c_device_real_kind,real_bytes=storage_size(0._real_kind)/8,hm=5,ny=8,nz=5,global_x=24
  type(c_ptr) :: parent,sample
  integer :: rank,ranks,local_rank,err,status,nx,origin,i,j,k,c,debug,scenario,first
  integer(c_int) :: lower,upper,mode
  integer(c_int64_t) :: usage(8)
  real(real_kind),allocatable :: q(:,:,:,:),original(:,:,:,:),observed(:,:,:,:),metric(:,:,:,:,:),x(:),line(:,:)
  real(real_kind) :: material(7),reference(3),fraction,rho,temp,u,expected(4),mu,weight,pressure, &
    slope_u,slope_t,tw,maximum,global_error
  character(64) :: text
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text,status=status)
  if(status/=0) error stop 'SBLI wall component requires matching Open MPI launcher'
  read(text,*) local_rank
  call check(device_bind(int(local_rank,c_int)))
  call MPI_Init(err);call MPI_Comm_rank(MPI_COMM_WORLD,rank,err);call MPI_Comm_size(MPI_COMM_WORLD,ranks,err)
  call get_command_argument(1,text);mode=0
  if(trim(text)=='device-aware') mode=1
  if(trim(text)/='pinned'.and.trim(text)/='device-aware') error stop 'explicit SBLI component transport'
  if(mod(global_x,ranks)/=0) error stop 'component x partition'
  nx=global_x/ranks;origin=rank*nx
  lower=int(rank-1,c_int);if(rank==0) lower=int(MPI_PROC_NULL,c_int)
  upper=int(rank+1,c_int);if(rank==ranks-1) upper=int(MPI_PROC_NULL,c_int)
  allocate(q(-hm:nx+hm,-hm:ny+hm,-hm:nz+hm,5),metric(-hm:nx+hm,-hm:ny+hm,-hm:nz+hm,3,3), &
    x(0:nx),line(4,0:nx))
  allocate(original,mold=q);allocate(observed,mold=q)
  material=[1.4_real_kind,.4_real_kind,2.5_real_kind,100._real_kind,.72_real_kind,.5_real_kind,1.5_real_kind]
  reference=[1._real_kind,1._real_kind,1._real_kind/material(1)]
  x=real([(origin+i,i=0,nx)],real_kind)/global_x
  first=2;maximum=0
  do debug=0,1
    call check(device_state_create(int(local_rank,c_int),int(nx,c_int),ny,nz,hm,int(debug,c_int),parent))
    call check(fluid_prepare(parent,1_c_int))
    metric=0
    metric(:,:,:,1,1)=global_x;metric(:,:,:,2,2)=2;metric(:,:,:,3,3)=10
    q=-77
    do k=0,nz
    do j=0,ny
    do i=0,nx
      rho=1+.01_real_kind*(origin+i)+.4_real_kind*k
      tw=1+.02_real_kind*(origin+i)+.5_real_kind*k
      slope_u=.02_real_kind*(k+1);slope_t=.01_real_kind*(k+1)
      temp=tw+slope_t*j+1e-6_real_kind*real(j,real_kind)**6
      u=slope_u*j+1e-5_real_kind*real(j,real_kind)**6;pressure=rho*temp/material(1)
      q(i,j,k,:)=[rho,rho*u,0._real_kind,0._real_kind,material(3)*pressure+.5_real_kind*rho*u*u]
    enddo
    enddo
    enddo
    if(upper/=MPI_PROC_NULL) q(nx,:,:,:)=-13
    original=q
    call check(device_state_upload(parent,device_q,q,bytes(size(q))))
    call check(device_state_upload(parent,device_metric,metric,bytes(size(metric))))
    do scenario=0,2
      fraction=.3_real_kind
      if(scenario==0) fraction=0
      if(scenario==2) fraction=1
      call check(sbli_wall_create(parent,int(MPI_COMM_WORLD,c_int),lower,upper,int(ranks,c_int),mode, &
        int(first,c_int),fraction,material,reference,x,sample))
      status=device_state_destroy(parent)
      if(status/=device_invalid_argument.or..not.c_associated(parent)) error stop 'SBLI sample lost its parent ownership'
      call check(sbli_wall_capture(sample,line,bytes(size(line))))
      if(.not.all(ieee_is_finite(line))) error stop 'nonfinite SBLI component line'
      do i=0,nx
        expected=0
        do k=first,first+1
          weight=1-fraction;if(k==first+1) weight=fraction
          rho=1+.01_real_kind*(origin+i)+.4_real_kind*k
          tw=1+.02_real_kind*(origin+i)+.5_real_kind*k
          slope_u=.02_real_kind*(k+1);slope_t=.01_real_kind*(k+1)
          mu=tw*sqrt(tw)*material(7)/((tw+material(6))*material(4))
          expected=expected+weight*[x(i),rho*tw,2*mu*slope_u*2,((mu/material(5))/material(2))*slope_t*2]
        enddo
        maximum=max(maximum,maxval(abs(line(:,i)-expected)/max(1._real_kind,abs(expected))))
      enddo
      if(real_bytes==8.and.maximum>2e-10_real_kind) error stop 'SBLI analytic normal slope/nonlinear interpolation'
      call check(sbli_wall_metrics(sample,usage))
      if(usage(6)/=4.or.usage(7)/=int(size(line),c_int64_t)*real_bytes) error stop 'SBLI selected download counters'
      if(mode==1.or.ranks==1) then
        if(any(usage([2,4,5])/=0)) error stop 'device/local SBLI source has host MPI staging'
      else
        if(usage(4)/=merge(70_c_int64_t*real_bytes,0_c_int64_t,lower/=MPI_PROC_NULL).or. &
          usage(5)/=merge(70_c_int64_t*real_bytes,0_c_int64_t,upper/=MPI_PROC_NULL)) error stop 'SBLI staging packet length'
      endif
      if(usage(3)/=merge(70_c_int64_t*real_bytes,0_c_int64_t,lower/=MPI_PROC_NULL)) error stop 'SBLI device packet length'
      call check(device_state_download(parent,device_q,observed,bytes(size(observed))))
      if(any(transfer(observed,[0_c_int8_t],size(observed)*real_bytes)/= &
        transfer(original,[0_c_int8_t],size(original)*real_bytes))) error stop 'SBLI observer changed solver q'
      call check(sbli_wall_destroy(sample))
    enddo
    q=original;q(0,0,first,1)=0
    call check(device_state_upload(parent,device_q,q,bytes(size(q))))
    call check(sbli_wall_create(parent,int(MPI_COMM_WORLD,c_int),lower,upper,int(ranks,c_int),mode, &
      int(first,c_int),.3_real_kind,material,reference,x,sample))
    status=sbli_wall_capture(sample,line,bytes(size(line)))
    if(status/=device_nonphysical) error stop 'SBLI source did not reject invalid consumed density collectively'
    call check(sbli_wall_destroy(sample));call check(device_state_destroy(parent))
  enddo
  call MPI_Allreduce(maximum,global_error,1,merge(MPI_DOUBLE_PRECISION,MPI_REAL,real_bytes==8),MPI_MAX,MPI_COMM_WORLD,err)
  if(rank==0) write(*,'(a,3(1x,i0),1x,es24.16)') 'CPP_SBLI_WALL_COMPONENT_PASS',real_bytes,ranks,mode,global_error
  call MPI_Finalize(err)
contains
  integer(c_size_t) function bytes(n)
    integer,intent(in) :: n
    bytes=int(n,c_size_t)*real_bytes
  end function
  subroutine check(code)
    integer,intent(in) :: code
    if(code/=0) then
      print *, 'SBLI wall component status=',code
      call MPI_Abort(MPI_COMM_WORLD,code,err)
    endif
  end subroutine
end program
