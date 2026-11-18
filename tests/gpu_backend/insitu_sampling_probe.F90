program insitu_sampling_probe
  use iso_c_binding
  use mpi
  use ieee_arithmetic, only: ieee_is_finite
  use device_state
  use fluid_explicit
  use insitu_sampling_cpp
  implicit none
  integer,parameter :: global=32,hm=5
  type :: field_packet
    real(c_double),allocatable :: values(:)
  end type
  type(field_packet) :: original(0:device_field_count-1)
  type(c_ptr) :: parent
  type(sample_plan) :: plan
  type(device_view) :: view
  integer :: ranks,rank,local_rank,status,ierr,comm,dims(3),coords(3),neighbors(2,3),extent(3),offset(3), &
    i,j,k,a,b,field,debug,config,frame,nodes,components,checks=0,n,peer,peer_coords(3),g(3),node,axis
  integer(c_int) :: lower(3),higher(3),parts(3),periodic(3),mode
  integer(c_int64_t) :: usage(9),before(9),capacity,fixed,expected_bytes,expected_peak,retained,scalar_bytes
  real(c_double),allocatable :: q(:,:,:,:),metric(:,:,:,:,:),velocity(:,:),halo(:,:,:,:),diagnostics(:,:), &
    state(:,:,:,:),global_state(:,:,:,:),send(:),gathered(:,:),actual(:),q_reference(:,:)
  real(c_double) :: h,pi,kappa,x,y,z,density,v(3),expected_v(3),gradient(3,3),logical_gradient(3,3), &
    matrix(3,3),fields(14),error,maximum=0,halo_max=0,velocity_max=0
  logical :: keep_velocity,with_halo,walls,sixth
  character(64) :: argument,transport,closure
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',argument,status=status)
  if(status/=0) error stop 'sampling probe requires the matching Open MPI launcher'
  read(argument,*) local_rank
  call check(device_bind(int(local_rank,c_int)),'bind before MPI')
  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr);call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
  call get_command_argument(1,argument);read(argument,*) dims
  if(any(dims<1).or.product(dims)/=ranks.or.any(mod(global,dims)/=0)) error stop 'sampling topology mismatch'
  call get_command_argument(2,transport)
  select case(trim(transport))
  case('pinned');mode=0
  case('device-aware');mode=1
  case default;error stop 'sample transport must be pinned/device-aware'
  end select
  call get_command_argument(3,closure)
  walls=len_trim(closure)>0;sixth=trim(closure)=='sixth'
  if(walls.and.trim(closure)/='legacy'.and..not.sixth) error stop 'sample closure must be legacy/sixth'
  call MPI_Cart_create(MPI_COMM_WORLD,3,dims,[.not.walls,.not.walls,.not.walls],.false.,comm,ierr)
  call MPI_Cart_coords(comm,rank,3,coords,ierr)
  do a=1,3
    call MPI_Cart_shift(comm,a-1,1,neighbors(1,a),neighbors(2,a),ierr)
  enddo
  extent=global/dims;offset=coords*extent;nodes=product(extent+1)
  lower=int(neighbors(1,:),c_int);higher=int(neighbors(2,:),c_int);parts=int(dims,c_int)
  periodic=merge(0_c_int,1_c_int,walls)
  pi=acos(-1.d0);h=2*pi/global;kappa=(1.5d0*sin(h)-.3d0*sin(2*h)+sin(3*h)/30)/h
  allocate(q(-hm:extent(1)+hm,-hm:extent(2)+hm,-hm:extent(3)+hm,5), &
    metric(-hm:extent(1)+hm,-hm:extent(2)+hm,-hm:extent(3)+hm,3,3), &
    global_state(0:global,0:global,0:global,4), &
    send(4*product(extent)),gathered(4*product(extent),ranks),q_reference(nodes,2))
  do debug=0,1
    call check(device_state_create(int(local_rank,c_int),int(extent(1),c_int),int(extent(2),c_int), &
      int(extent(3),c_int),hm,int(debug,c_int),parent),'create parent')
    call check(fluid_prepare(parent,1_c_int),'parent primitives/gradients')
    call check(fluid_rhs_prepare(parent,1_c_int),'parent viscous fields')
    call check(fluid_filter_prepare_scalar(parent),'parent filter scratch')
    do field=0,device_field_count-1
      status=device_state_view(parent,int(field,c_int),view)
      if(status==device_not_allocated) cycle
      call check(status,'parent field view')
      n=int(product(int(view%extent,c_int64_t)))
      allocate(original(field)%values(n))
      original(field)%values=100+field
      call check(device_state_upload(parent,int(field,c_int),original(field)%values,bytes(n)),'seed parent fields')
    enddo
    do config=1,4
      components=0
      if(config==1) components=14
      if(config==2) components=1
      keep_velocity=config/=3;with_halo=config/=4
      call check(sample_create(parent,comm,lower,higher,parts,periodic,mode,keep_velocity,with_halo, &
        int(components,c_int),sixth,plan),'prepare private sample')
      status=device_state_destroy(parent)
      if(status/=device_invalid_argument.or..not.c_associated(parent)) error stop 'sample lost parent lifetime ownership'
      do frame=0,1
        matrix=0
        do a=1,3
          matrix(a,a)=1
        enddo
        if(frame==1) matrix=reshape([1.d0,.15d0,.05d0,.1d0,1.d0,-.1d0,0.d0,.2d0,1.d0],[3,3])
        do b=1,3
        do a=1,3
          metric(:,:,:,a,b)=matrix(a,b)/h
        enddo
        enddo
        q=-77
        do k=0,extent(3)
        do j=0,extent(2)
        do i=0,extent(1)
          q(i,j,k,:)=[99.d0,100.d0,101.d0,102.d0,103.d0]
          if(frame==1) q(i,j,k,1)=0
          if((i==extent(1).and.higher(1)/=MPI_PROC_NULL).or. &
             (j==extent(2).and.higher(2)/=MPI_PROC_NULL).or. &
             (k==extent(3).and.higher(3)/=MPI_PROC_NULL)) cycle
          g=offset+[i,j,k];x=g(1)*h;y=g(2)*h;z=g(3)*h
          v=[sin(x)*cos(y)*cos(z),-cos(x)*sin(y)*cos(z),0.d0]
          if(walls) call polynomial_velocity(g,v)
          density=1
          if(frame==1) density=1+.03d0*rank+.05d0*cos(x)*sin(y)*cos(z)
          if(walls.and.frame==1) density=1+.05d0*cos(x)*sin(y)*cos(z)
          q(i,j,k,:)=[density,density*v,2.d0]
        enddo
        enddo
        enddo
        if(walls) then
          do k=0,global
          do j=0,global
          do i=0,global
            if(i/=global.and.j/=global.and.k/=global) cycle
            g=[i,j,k];call polynomial_velocity(g,v);density=1
            if(frame==1) density=1+.05d0*cos(i*h)*sin(j*h)*cos(k*h)
            global_state(i,j,k,:)=[density,density*v]
          enddo
          enddo
          enddo
        endif
        original(device_q)%values=reshape(q,[size(q)])
        original(device_metric)%values=reshape(metric,[size(metric)])
        call check(device_state_upload(parent,device_q,original(device_q)%values,bytes(size(q))),'frame q')
        call check(device_state_upload(parent,device_metric,original(device_metric)%values,bytes(size(metric))),'frame metric')
        send=reshape(q(0:extent(1)-1,0:extent(2)-1,0:extent(3)-1,1:4),[size(send)])
        call MPI_Allgather(send,size(send),MPI_DOUBLE_PRECISION,gathered,size(send),MPI_DOUBLE_PRECISION,comm,ierr)
        do peer=0,ranks-1
          call MPI_Cart_coords(comm,peer,3,peer_coords,ierr);g=peer_coords*extent
          global_state(g(1):g(1)+extent(1)-1,g(2):g(2)+extent(2)-1,g(3):g(3)+extent(3)-1,:)= &
            reshape(gathered(:,peer+1),[extent,4])
        enddo
        call check(sample_metrics(plan%handle,before),'initial sample counters')
        if(frame==0) then
          call check(sample_capture(plan),'capture through Fortran schedule')
        else
          call check(begin_sample(plan%handle),'begin staged observation')
          do axis=0,2
            call check(canonical_axis(plan%handle,int(axis,c_int)),'canonical endpoint axis')
          enddo
          call check(sample_view(plan%handle,sample_state,view),'temporary canonical state view')
          allocate(state(0:extent(1),0:extent(2),0:extent(3),4))
          call check(sample_download(plan%handle,sample_state,state,bytes(size(state))),'explicit canonical-state observation')
          do k=0,extent(3)
          do j=0,extent(2)
          do i=0,extent(1)
            g=offset+[i,j,k]
            if(.not.walls) g=modulo(g,global)
            if(any(transfer(state(i,j,k,:),[0_c_int64_t],4)/= &
              transfer(global_state(g(1),g(2),g(3),:),[0_c_int64_t],4))) error stop 'canonical q4 bits differ from unique owner'
          enddo
          enddo
          enddo
          deallocate(state)
          call check(make_velocity(plan%handle),'canonical velocity')
          if(with_halo) then
            do axis=0,2
              call check(halo_axis(plan%handle,int(axis,c_int)),'private halo axis')
            enddo
          endif
          call check(finish_sample(plan%handle),'finish staged observation')
        endif
        call check(sample_metrics(plan%handle,usage),'post-capture counters')
        if(usage(8)-before(8)/=merge(0_c_int64_t,32_c_int64_t*nodes,frame==0)) error stop 'capture made an unrequested array download'
        if(usage(9)/=frame+1) error stop 'sample generation mismatch'
        capacity=0
        do a=1,3
          if(walls.and.dims(a)==1) cycle
          retained=4_c_int64_t*(extent(mod(a,3)+1)+1)*(extent(mod(a+1,3)+1)+1)
          if(with_halo) retained=max(retained,9_c_int64_t*(extent(mod(a,3)+1)+7)*(extent(mod(a+1,3)+1)+7))
          capacity=max(capacity,retained)
        enddo
        fixed=16*capacity+4
        expected_bytes=fixed
        if(keep_velocity) expected_bytes=expected_bytes+24_c_int64_t*nodes
        if(with_halo) expected_bytes=expected_bytes+24_c_int64_t*product(int(extent+7,c_int64_t))
        expected_bytes=expected_bytes+8_c_int64_t*components*nodes
        expected_peak=max(fixed+56_c_int64_t*nodes,expected_bytes)
        if(with_halo) expected_peak=max(expected_peak,fixed+24_c_int64_t*(nodes+product(int(extent+7,c_int64_t))))
        if(usage(1)/=expected_bytes.or.usage(2)/=expected_peak) error stop 'sample allocation phase/accounting differs'
        if(usage(3)/=merge(16*capacity,0_c_int64_t,mode==0.and.ranks>1)) error stop 'pinned sample allocation policy differs'
        if(mode==1.and.any(usage(5:6)/=0)) error stop 'device sample staged on host'
        scalar_bytes=merge(8_c_int64_t,4_c_int64_t,components>0)*(frame+1)
        if(usage(7)/=scalar_bytes) error stop 'unexpected sample status download bytes'
        status=sample_view(plan%handle,sample_state,view)
        if(status/=device_not_allocated) error stop 'canonical scratch was retained after conversion'
        status=sample_view(plan%handle,sample_velocity,view)
        if(keep_velocity) then
          call check(status,'packed velocity view')
          if(view%rank/=2.or.any(view%extent(:2)/=[3,nodes]).or.view%stride(2)/=3) error stop 'packed velocity ABI'
          allocate(velocity(3,nodes))
          call check(sample_download(plan%handle,sample_velocity,velocity,bytes(size(velocity))),'velocity observation')
          do node=0,nodes-1
            call location(node,g)
            expected_v=global_state(g(1),g(2),g(3),2:4)/global_state(g(1),g(2),g(3),1)
            velocity_max=max(velocity_max,maxval(abs(velocity(:,node+1)-expected_v)))
          enddo
          deallocate(velocity)
        elseif(status/=device_not_allocated) then
          error stop 'streamline-only capture retained packed velocity'
        endif
        status=sample_view(plan%handle,sample_halo,view)
        if(with_halo) then
          call check(status,'private halo view')
          if(view%rank/=4.or.any(view%lower(:3)/=-3).or.any(view%extent(:3)/=extent+7)) error stop 'private halo ABI'
          allocate(halo(-3:extent(1)+3,-3:extent(2)+3,-3:extent(3)+3,3))
          call check(sample_download(plan%handle,sample_halo,halo,bytes(size(halo))),'halo observation')
          do k=-3,extent(3)+3
          do j=-3,extent(2)+3
          do i=-3,extent(1)+3
            g=offset+[i,j,k]
            if(walls.and.(any(g<0).or.any(g>global))) cycle
            if(.not.walls) g=modulo(g,global)
            expected_v=global_state(g(1),g(2),g(3),2:4)/global_state(g(1),g(2),g(3),1)
            halo_max=max(halo_max,maxval(abs(halo(i,j,k,:)-expected_v)))
          enddo
          enddo
          enddo
          deallocate(halo)
        elseif(status/=device_not_allocated) then
          error stop 'velocity-only capture allocated a halo'
        endif
        status=sample_view(plan%handle,sample_diagnostics,view)
        if(components>0) then
          call check(status,'diagnostic view')
          if(view%rank/=2.or.any(view%extent(:2)/=[components,nodes])) error stop 'diagnostic ABI'
          allocate(diagnostics(components,nodes))
          call check(sample_download(plan%handle,sample_diagnostics,diagnostics,bytes(size(diagnostics))),'diagnostic observation')
          if(.not.all(ieee_is_finite(diagnostics))) error stop 'nonfinite sample diagnostic'
          error=0
          do node=0,nodes-1
            call location(node,g);x=g(1)*h;y=g(2)*h;z=g(3)*h
            logical_gradient(:,1)=kappa*[cos(x)*cos(y)*cos(z),sin(x)*sin(y)*cos(z),0.d0]
            logical_gradient(:,2)=kappa*[-sin(x)*sin(y)*cos(z),-cos(x)*cos(y)*cos(z),0.d0]
            logical_gradient(:,3)=kappa*[-sin(x)*cos(y)*sin(z),cos(x)*sin(y)*sin(z),0.d0]
            if(walls) then
              logical_gradient(:,1)=[1+.2d0*y,.6d0*x,.1d0*y]
              logical_gradient(:,2)=[.2d0*x,-.4d0+.1d0*z,.1d0*x]
              logical_gradient(:,3)=[.06d0*z,.1d0*y,.2d0]
              if(sixth) then
                logical_gradient(1,1)=logical_gradient(1,1)+.06d0/(2*pi)*(x/(2*pi))**5
                logical_gradient(2,2)=logical_gradient(2,2)+.12d0/(2*pi)*(y/(2*pi))**5
                logical_gradient(3,3)=logical_gradient(3,3)+.18d0/(2*pi)*(z/(2*pi))**5
              endif
            endif
            gradient=matmul(logical_gradient,matrix)
            fields(:9)=reshape(gradient,[9]);fields(10)=-.5d0*sum(gradient*transpose(gradient))
            fields(11)=gradient(1,1)+gradient(2,2)+gradient(3,3)
            fields(12:14)=[gradient(3,2)-gradient(2,3),gradient(1,3)-gradient(3,1),gradient(2,1)-gradient(1,2)]
            if(components==14) then
              error=max(error,maxval(abs(diagnostics(:,node+1)-fields)))
              q_reference(node+1,frame+1)=diagnostics(10,node+1)
            else
              error=max(error,abs(diagnostics(1,node+1)-fields(10)))
            endif
          enddo
          maximum=max(maximum,error)
          if(components==1.and.any(transfer(diagnostics(1,:),[0_c_int64_t],nodes)/= &
            transfer(q_reference(:,frame+1),[0_c_int64_t],nodes))) error stop 'Q-only bits differ from full diagnostic'
          deallocate(diagnostics)
        elseif(status/=device_not_allocated) then
          error stop 'unrequested diagnostic array was allocated'
        endif
        do field=0,device_field_count-1
          if(.not.allocated(original(field)%values)) cycle
          allocate(actual(size(original(field)%values)))
          call check(device_state_download(parent,int(field,c_int),actual,bytes(size(actual))),'parent readonly observation')
          if(any(transfer(actual,[0_c_int64_t],size(actual))/= &
            transfer(original(field)%values,[0_c_int64_t],size(actual)))) error stop 'sample modified a solver field'
          deallocate(actual)
        enddo
        if(maximum>2.d-10.or.halo_max>2.d-10.or.velocity_max>2.d-10) error stop 'sample analytic/ownership field differs'
        checks=checks+1
      enddo
      call check(sample_destroy(plan),'release sample')
    enddo
    call check(sample_create(parent,comm,lower,higher,parts,periodic,mode,.true.,.false., &
      0_c_int,.false.,plan),'prepare collective failure observation')
    if(rank==0) q(0,0,0,1)=0
    original(device_q)%values=reshape(q,[size(q)])
    call check(device_state_upload(parent,device_q,original(device_q)%values,bytes(size(q))),'invalid owned test state')
    status=sample_capture(plan)
    if(status/=device_nonphysical) error stop 'peer capture did not report the same physical-state failure'
    call check(sample_destroy(plan),'release failed sample')
    allocate(actual(size(q)))
    call check(device_state_download(parent,device_q,actual,bytes(size(q))),'failed sample readonly observation')
    if(any(transfer(actual,[0_c_int64_t],size(actual))/= &
      transfer(original(device_q)%values,[0_c_int64_t],size(actual)))) error stop 'failed sample modified q'
    deallocate(actual)
    call check(device_state_destroy(parent),'release parent')
    do field=0,device_field_count-1
      if(allocated(original(field)%values)) deallocate(original(field)%values)
    enddo
  enddo
  write(*,'(a,2(1x,i0),2(1x,a),3(1x,es24.16e3))') 'CPP_INSITU_SAMPLING_PASS',rank,checks,trim(transport),trim(closure), &
    maximum,halo_max,velocity_max
  call MPI_Comm_free(comm,ierr);call MPI_Finalize(ierr)
contains
  subroutine location(node,g)
    integer,intent(in) :: node
    integer,intent(out) :: g(3)
    g=offset+[mod(node,extent(1)+1),mod(node/(extent(1)+1),extent(2)+1), &
      node/((extent(1)+1)*(extent(2)+1))]
    if(.not.walls) g=modulo(g,global)
  end subroutine
  subroutine polynomial_velocity(g,v)
    integer,intent(in) :: g(3)
    real(c_double),intent(out) :: v(3)
    real(c_double) :: x,y,z
    x=g(1)*h;y=g(2)*h;z=g(3)*h
    v=[.1d0+x+.2d0*x*y+.03d0*z*z,-.2d0+.3d0*x*x-.4d0*y+.1d0*z*y,.5d0+.2d0*z+.1d0*x*y]
    if(sixth) v=v+[.01d0*(x/(2*pi))**6,.02d0*(y/(2*pi))**6,.03d0*(z/(2*pi))**6]
  end subroutine
  integer(c_size_t) function bytes(n)
    integer,intent(in) :: n
    bytes=int(n,c_size_t)*8_c_size_t
  end function
  subroutine check(code,label)
    integer(c_int),intent(in) :: code
    character(*),intent(in) :: label
    if(code==0) return
    write(*,'(a,1x,a,1x,i0)') 'CPP_SAMPLE_STATUS',label,code
    error stop 'sample probe failed'
  end subroutine
end program
