program insitu_index_probe
  use iso_c_binding
  use mpi
  use device_state
  use fluid_explicit
  use insitu_index_api
  use index_slice_stencil, only: slice_stencil_weights,slice_diagnostics
  implicit none
  integer :: i
  integer,parameter :: im=12,jm=8,km=10,hm=5,nout=(im+1)*(jm+1)*(km+1),nf=30
  integer(c_int),parameter :: raw(5)=[1,2,3,4,6],fields(nf)=[1,2,3,4,5,6,(13+i,i=0,23)]
  type(c_ptr) :: parent,values_pointer,coordinates_pointer
  type(index_region) :: region
  real(c_double),allocatable :: q(:,:,:,:),rho(:,:,:),velocity(:,:,:,:),pressure(:,:,:),temperature(:,:,:), &
    metric(:,:,:,:,:),xyz(:,:,:,:),values(:,:),coordinates(:,:),neighborhood(:,:),packet(:,:),observed(:)
  integer(c_int),allocatable :: indices(:,:),positions(:),remote_positions(:)
  real(c_double) :: weights(7,0:6),maximum=0,expected(nf),point(3),v(3),gradient(3,3),b(3,3), &
    logical_gradient(3,3),density_gradient(3),thermal(3),diagnostic(15),t,mu,kappa,vortex(2),h=.05d0
  integer :: rank,ranks,ierr,local_rank,status,j,k,n,node,frame,debug,mode,nx,nodes,owned, &
    peer,peer_owned,source,destination,begin_i,end_i,c,check_count=0,field
  integer(c_int64_t) :: usage(3)
  character(64) :: argument
  interface
    integer(c_int) function vortex_point(gradient,requested,values) bind(C,name='astr_insitu_vortex_point')
      import c_int,c_double
      real(c_double),intent(in) :: gradient(3,3)
      integer(c_int),value :: requested
      real(c_double),intent(out) :: values(2)
    end function
  end interface
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',argument,status=status)
  if(status/=0) error stop 'matching MPI launcher is required'
  read(argument,*) local_rank
  call check(device_bind(int(local_rank,c_int)),'device bind')
  call MPI_Init(ierr);call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  call get_command_argument(1,argument)
  select case(trim(argument))
  case('pinned');mode=0
  case('device-aware');mode=1
  case default;error stop 'index probe requires pinned/device-aware'
  end select
  nx=ranks*im;nodes=(nx+1)*(jm+1)*(km+1);owned=im+merge(1,0,rank==ranks-1)
  allocate(q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),rho(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
    velocity(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3),pressure(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
    temperature(-hm:im+hm,-hm:jm+hm,-hm:km+hm),metric(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3), &
    xyz(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3),values(nout,nf),coordinates(3,nout), &
    neighborhood(nodes,5),indices(3,nout),positions(nout),remote_positions(nout),packet(nout,5))
  call slice_stencil_weights(weights)
  region%first=[rank*im,0,0];region%extent=[im+1,jm+1,km+1]
  region%lower=0;region%shape=[nx+1,jm+1,km+1];region%origin=region%first
  region%intervals=[nx,jm,km];region%periodic=0;region%cartesian_support=0;region%nondimen=1
  region%tempconst=.4d0;region%tempconst1=1.4d0;region%reynolds=1000;region%prandtl=.72d0
  region%const5=.8d0;region%cp=3.5d0
  do k=-hm,km+hm
  do j=-hm,jm+hm
  do i=-hm,im+hm
    xyz(i,j,k,:)=h*[rank*im+i,j,k]
  enddo
  enddo
  enddo
  do debug=0,1
    call check(device_state_create(int(local_rank,c_int),im,jm,km,hm,int(debug,c_int),parent),'create index parent')
    call check(fluid_prepare(parent,1_c_int),'prepare source buffers')
    q=17
    call check(device_state_upload(parent,device_q,q,bytes(size(q))),'seed unchanged conservative field')
    do frame=0,1
      b=0;do c=1,3; b(c,c)=1;enddo
      if(frame==1) b=reshape([1.d0,.15d0,.05d0,.1d0,1.d0,-.1d0,0.d0,.2d0,1.d0],[3,3])
      do k=-hm,km+hm
      do j=-hm,jm+hm
      do i=-hm,im+hm
        point=h*[rank*im+i,j,k]
        call fixture(point,rho(i,j,k),velocity(i,j,k,:),pressure(i,j,k),temperature(i,j,k))
        metric(i,j,k,:,:)=b/h
      enddo
      enddo
      enddo
      call check(device_state_upload(parent,device_rho,rho,bytes(size(rho))),'upload source density')
      call check(device_state_upload(parent,device_velocity,velocity,bytes(size(velocity))),'upload source velocity')
      call check(device_state_upload(parent,device_pressure,pressure,bytes(size(pressure))),'upload source pressure')
      call check(device_state_upload(parent,device_temperature,temperature,bytes(size(temperature))),'upload source T')
      call check(device_state_upload(parent,device_metric,metric,bytes(size(metric))),'upload metric')
      call check(index_begin(parent,nodes,nout,5_c_int,raw,nf,fields,reshape(weights,[49]),1_c_int,xyz,bytes(size(xyz))), &
        'begin selected field group')
      n=0
      do k=0,km
      do j=0,jm
      do i=0,owned-1
        n=n+1;indices(:,n)=[i,j,k];positions(n)=1+rank*im+i+(nx+1)*(j+(jm+1)*k)
      enddo
      enddo
      enddo
      call check(index_fill(parent,int(n,c_int),indices,positions),'fill owned neighborhood')
      if(ranks>1) call check(index_exchange_prepare(parent,nout,int(mode,c_int)),'prepare private exchange')
      do peer=1,ranks-1
        destination=mod(rank+peer,ranks);source=mod(rank-peer+ranks,ranks)
        peer_owned=im+merge(1,0,source==ranks-1);node=0
        do k=0,km
        do j=0,jm
        do i=0,peer_owned-1
          node=node+1;remote_positions(node)=1+source*im+i+(nx+1)*(j+(jm+1)*k)
        enddo
        enddo
        enddo
        call check(index_exchange(parent,int(n,c_int),indices,int(node,c_int),remote_positions, &
          int(destination,c_int),int(source,c_int),int(MPI_COMM_WORLD,c_int)),'device private neighborhood')
      enddo
      call check(index_exchange_end(parent),'end private exchange')
      call check(index_download(parent,1_c_int,neighborhood,bytes(size(neighborhood))),'observe complete stencil')
      node=0
      do k=0,km
      do j=0,jm
      do i=0,nx
        node=node+1;point=h*[i,j,k]
        call fixture(point,t,v,mu,kappa)
        if(maxval(abs(neighborhood(node,:)-[t,v,kappa]))/=0) error stop 'neighborhood owner values differ'
      enddo
      enddo
      enddo
      call check(index_usage(parent,usage),'private byte accounting')
      if(mode==1.and.any(usage(:2)/=0)) error stop 'device neighborhood staged on host'
      if(ranks>1.and.(usage(3)/=(ranks-1)*int(n,c_int64_t)*5*8)) error stop 'private send count differs'
      if(mode==0.and.ranks>1) then
        if(usage(1)/=usage(3).or.usage(2)/=(nodes-n)*5*8) error stop 'pinned staging count differs'
      endif
      call check(index_evaluate(parent,region),'evaluate selected diagnostics')
      call check(index_view(parent,values_pointer,coordinates_pointer),'borrow packed arrays')
      if(.not.c_associated(values_pointer).or..not.c_associated(coordinates_pointer)) error stop 'missing device view'
      call check(index_download(parent,0_c_int,values,bytes(size(values))),'observe selected fields')
      call check(index_download(parent,2_c_int,coordinates,bytes(size(coordinates))),'observe coordinates')
      node=0
      do k=0,km
      do j=0,jm
      do i=0,im
        node=node+1;point=h*[rank*im+i,j,k]
        call fixture(point,expected(1),expected(2:4),expected(5),expected(6))
        logical_gradient=0
        logical_gradient(:,1)=[2*point(1),0.d0,.3d0]
        logical_gradient(:,2)=[.2d0,-.6d0*point(2),0.d0]
        logical_gradient(:,3)=[0.d0,.1d0,.8d0*point(3)]
        gradient=matmul(logical_gradient,b);density_gradient=matmul([.01d0,.02d0,.03d0],b)
        thermal=matmul([.01d0,.02d0,.03d0],b)
        call slice_diagnostics(gradient,density_gradient,diagnostic)
        expected(7:21)=diagnostic;expected(22:24)=thermal
        t=expected(6);mu=t*sqrt(t)*region%tempconst1/(t+region%tempconst)/region%reynolds
        kappa=(mu/region%prandtl)/region%const5
        expected(25:28)=[mu*(gradient(1,2)+gradient(2,1)),mu*(gradient(3,2)+gradient(2,3)),kappa*thermal(2),mu]
        call check(vortex_point(gradient,3_c_int,vortex),'existing CPU vortex reference')
        expected(29:30)=vortex
        maximum=max(maximum,maxval(abs(values(node,:)-expected)))
        if(any(transfer(coordinates(:,node),[0_c_int64_t],3)/=transfer(point,[0_c_int64_t],3))) &
          error stop 'selected coordinate bits differ'
      enddo
      enddo
      enddo
      if(maximum>2.d-10) then
        print *, 'INDEX_ANALYTIC_ERROR',maximum
        error stop 'selected field/diagnostic reference differs'
      endif
      allocate(observed(size(q)))
      call check(device_state_download(parent,device_q,observed,bytes(size(q))),'observe unmodified q')
      if(any(observed/=17)) error stop 'index consumer modified conservative field'
      deallocate(observed)
      call check(index_end(parent),'release selected group')
      check_count=check_count+1
    enddo
    ! The old material contract rejects negative Re and temperature coefficient
    ! even when their ratio accidentally gives positive viscosity.
    call check(index_begin(parent,1_c_int,1_c_int,1_c_int,[6_c_int],1_c_int,[34_c_int], &
      reshape(weights,[49]),0_c_int,xyz,bytes(size(xyz))),'begin material domain check')
    call check(index_fill(parent,1_c_int,[0_c_int,0_c_int,0_c_int],[1_c_int]),'material source')
    region%extent=1;region%lower=region%origin;region%shape=1
    region%reynolds=-1000;region%tempconst1=-1.4d0
    status=index_evaluate(parent,region)
    if(status/=device_nonphysical) error stop 'invalid material parameters canceled into an accepted viscosity'
    write(*,'(a,2(1x,i0))') 'CPP_INSITU_INDEX_DOMAIN_PASS',rank,debug
    call check(index_end(parent),'release invalid material group')
    region%extent=[im+1,jm+1,km+1];region%lower=0;region%shape=[nx+1,jm+1,km+1]
    region%reynolds=1000;region%tempconst1=1.4d0
    call check(device_state_destroy(parent),'release parent and static geometry')
  enddo
  write(*,'(a,2(1x,i0),1x,a,1x,es24.16e3)') 'CPP_INSITU_INDEX_PASS',rank,check_count,trim(argument),maximum
  call MPI_Finalize(ierr)
contains
  subroutine fixture(point,rho,v,p,t)
    real(c_double),intent(in) :: point(3)
    real(c_double),intent(out) :: rho,v(3),p,t
    rho=1+sum([.01d0,.02d0,.03d0]*point)
    v=[.5d0+point(1)**2+.2d0*point(2),-.3d0*point(2)**2+.1d0*point(3),.4d0*point(3)**2+.3d0*point(1)]
    p=1+.04d0*point(1);t=.8d0+sum([.01d0,.02d0,.03d0]*point)
  end subroutine
  integer(c_size_t) function bytes(n)
    integer,intent(in) :: n
    bytes=8_c_size_t*n
  end function
  subroutine check(code,label)
    integer(c_int),intent(in) :: code
    character(*),intent(in) :: label
    if(code==0) return
    print *, trim(label),code
    call MPI_Abort(MPI_COMM_WORLD,1,ierr)
  end subroutine
end program
