program insitu_vectors_probe
  use iso_c_binding
  use iso_fortran_env, only: int64
  use mpi
  use ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  use device_state
  use fluid_explicit
  use insitu_index_api
  use insitu_sampling_cpp
  implicit none
  integer,parameter :: global=32,hm=5
  integer(c_int),parameter :: fields(5)=[3,5,4,2,1],slots(3)=[3,0,2]
  type(c_ptr) :: parent
  type(sample_plan) :: sample
  type(index_region) :: region
  type(device_view) :: values_view,coordinates_view,view
  real(c_double),allocatable :: q(:,:,:,:),rho(:,:,:),velocity(:,:,:,:),pressure(:,:,:),temperature(:,:,:), &
    xyz(:,:,:,:),halo(:,:,:,:),coordinate_halo(:,:,:,:),before_values(:,:),after_values(:,:),observed(:), &
    packed_xyz(:,:),packed_check(:,:)
  integer :: rank,ranks,local_rank,ierr,status,dims(3),coords(3),extent(3),offset(3),neighbors(2,3), &
    i,j,k,a,node,nodes,debug,mapping,frame,checks=0,g(3),wrapped(3)
  integer(c_int) :: mode
  integer(c_int64_t) :: usage(9),before(9),capacity,fixed,halo_bytes,sent_per_frame,coordinate_sent,face,internal,expected_sent
  real(c_double) :: pi,h,point(3),expected_velocity(3),expected_coordinate(3),coordinate_error=0,velocity_error=0
  character(64) :: argument,transport
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',argument,status=status)
  if(status/=0) error stop 'matching Open MPI launcher required'
  read(argument,*) local_rank
  call check(device_bind(int(local_rank,c_int)),'bind before MPI')
  call MPI_Init(ierr);call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  call get_command_argument(1,argument);read(argument,*) dims
  if(any(dims<1).or.product(dims)/=ranks.or.any(mod(global,dims)/=0)) error stop 'vector topology mismatch'
  call get_command_argument(2,transport)
  select case(trim(transport))
  case('pinned');mode=0
  case('device-aware');mode=1
  case default;error stop 'vector transport must be pinned/device-aware'
  end select
  call MPI_Cart_create(MPI_COMM_WORLD,3,dims,[.true.,.true.,.true.],.false.,i,ierr)
  block
    integer :: cart
    cart=i
    call MPI_Cart_coords(cart,rank,3,coords,ierr)
    do a=1,3
      call MPI_Cart_shift(cart,a-1,1,neighbors(1,a),neighbors(2,a),ierr)
      if(dims(a)==1) neighbors(:,a)=MPI_PROC_NULL
    enddo
    extent=global/dims;offset=coords*extent;nodes=product(extent+1)
    pi=acos(-1.d0);h=2*pi/global
    allocate(q(-hm:extent(1)+hm,-hm:extent(2)+hm,-hm:extent(3)+hm,5), &
      rho(-hm:extent(1)+hm,-hm:extent(2)+hm,-hm:extent(3)+hm), &
      velocity(-hm:extent(1)+hm,-hm:extent(2)+hm,-hm:extent(3)+hm,3), &
      pressure(-hm:extent(1)+hm,-hm:extent(2)+hm,-hm:extent(3)+hm), &
      temperature(-hm:extent(1)+hm,-hm:extent(2)+hm,-hm:extent(3)+hm), &
      xyz(-hm:extent(1)+hm,-hm:extent(2)+hm,-hm:extent(3)+hm,3),before_values(nodes,5),after_values(nodes,5), &
      halo(-3:extent(1)+3,-3:extent(2)+3,-3:extent(3)+3,3), &
      coordinate_halo(-3:extent(1)+3,-3:extent(2)+3,-3:extent(3)+3,3),packed_xyz(3,nodes),packed_check(3,nodes))
    capacity=0;sent_per_frame=0;coordinate_sent=0
    do a=1,3
      face=9_c_int64_t*(extent(mod(a,3)+1)+7)*(extent(mod(a+1,3)+1)+7)
      capacity=max(capacity,face)
      if(dims(a)>1) then
        internal=merge(1,0,offset(a)>0)+merge(1,0,offset(a)+extent(a)<global)
        sent_per_frame=sent_per_frame+(2+internal)*face*8
        coordinate_sent=coordinate_sent+internal*face*8
      endif
    enddo
    fixed=16*capacity+4;halo_bytes=24_c_int64_t*product(int(extent+7,c_int64_t))
    region%first=offset;region%extent=extent+1;region%lower=0;region%shape=1
    region%origin=offset;region%intervals=global;region%periodic=1;region%cartesian_support=0;region%nondimen=1
    region%tempconst=.4d0;region%tempconst1=1.4d0;region%reynolds=1600;region%prandtl=.72d0
    region%const5=1;region%cp=3.5d0
    do debug=0,1
    do mapping=1,2
      call check(device_state_create(int(local_rank,c_int),int(extent(1),c_int),int(extent(2),c_int), &
        int(extent(3),c_int),hm,int(debug,c_int),parent),'create source parent')
      call check(fluid_prepare(parent,0_c_int),'prepare source primitive buffers')
      q=77;rho=1;temperature=2
      call check(device_state_upload(parent,device_q,q,bytes(size(q))),'seed independent q')
      call check(device_state_upload(parent,device_rho,rho,bytes(size(rho))),'seed density')
      call check(device_state_upload(parent,device_temperature,temperature,bytes(size(temperature))),'seed temperature')
      xyz=-991
      do k=0,extent(3)
      do j=0,extent(2)
      do i=0,extent(1)
        call position(offset+[i,j,k],mapping,xyz(i,j,k,:))
      enddo
      enddo
      enddo
      call check(sample_create(parent,cart,int(neighbors(1,:),c_int),int(neighbors(2,:),c_int), &
        int(dims,c_int),[1_c_int,1_c_int,1_c_int],mode,.false.,.true.,0_c_int,.false.,sample),'create vector transport')
      do frame=0,1
        ! Pressure is not a selected velocity column; its NaN must not reject the vector.
        pressure=ieee_value(0.d0,ieee_quiet_nan)
        do k=-hm,extent(3)+hm
        do j=-hm,extent(2)+hm
        do i=-hm,extent(1)+hm
          velocity(i,j,k,:)=real(offset+[i,j,k]+[1,2,3],c_double)/64+real(frame,c_double)/128
        enddo
        enddo
        enddo
        call check(device_state_upload(parent,device_velocity,velocity,bytes(size(velocity))),'source velocity')
        call check(device_state_upload(parent,device_pressure,pressure,bytes(size(pressure))),'unused pressure')
        call check(index_begin(parent,0_c_int,int(nodes,c_int),0_c_int,[integer(c_int) ::],5_c_int,fields, &
          spread(0.d0,1,49),1_c_int,xyz,bytes(size(xyz))),'prepare strided source')
        call check(index_evaluate(parent,region),'pack selected fields/coordinates')
        call check(index_source(parent,0_c_int,values_view),'borrow SoA values')
        call check(index_source(parent,1_c_int,coordinates_view),'borrow interleaved coordinates')
        if(values_view%stride(1)/=1.or.values_view%stride(2)/=nodes.or. &
          coordinates_view%stride(1)/=3.or.coordinates_view%stride(2)/=1) error stop 'source view strides differ'
        call check(index_download(parent,0_c_int,before_values,bytes(size(before_values))),'source readonly observation')
        call check(sample_metrics(sample%handle,before),'capture counters before')
        call check(sample_capture_vector(sample,values_view,slots),'capture selected velocity')
        call check(sample_capture_coordinates(sample%handle,coordinates_view,int(offset,c_int), &
          [global,global,global]),'extend physical coordinate halo')
        call check(sample_metrics(sample%handle,usage),'capture counters after')
        if(usage(1)/=fixed+2*halo_bytes.or. &
          usage(2)/=fixed+2*halo_bytes+merge(24_c_int64_t*nodes,0_c_int64_t,frame>0)) &
          error stop 'unexpected vector/q scratch retained'
        if(usage(3)/=merge(16*capacity,0_c_int64_t,mode==0.and.ranks>1)) error stop 'packet host allocation differs'
        expected_sent=sent_per_frame*(frame+1)+coordinate_sent*frame
        if(usage(4)/=expected_sent) error stop 'coordinate end faces wrapped or MPI count differs'
        if(mode==1.and.any(usage(5:6)/=0)) error stop 'device source staged on host'
        if(mode==0.and.any(usage(5:6)/=expected_sent)) error stop 'pinned face staging differs'
        if(usage(7)/=8*(frame+1)+4*frame.or.usage(8)/=before(8).or.usage(9)/=frame+1) &
          error stop 'capture scalar/array/identity differs'
        status=sample_view(sample%handle,sample_state,view)
        if(status/=device_not_allocated) error stop 'vector capture consulted a q snapshot'
        status=sample_view(sample%handle,sample_velocity,view)
        if(status/=device_not_allocated) error stop 'vector capture duplicated the packed field'
        call check(sample_download(sample%handle,sample_halo,halo,bytes(size(halo))),'velocity halo observation')
        call check(sample_download(sample%handle,sample_coordinates,coordinate_halo,bytes(size(coordinate_halo))), &
          'coordinate halo observation')
        do k=-3,extent(3)+3
        do j=-3,extent(2)+3
        do i=-3,extent(1)+3
          g=offset+[i,j,k];wrapped=g
          where(wrapped<0.or.wrapped>global) wrapped=modulo(wrapped,global)
          expected_velocity=real(wrapped+[1,2,3],c_double)/64+real(frame,c_double)/128
          expected_coordinate=0
          if(all(g>=0).and.all(g<=global)) call position(g,mapping,expected_coordinate)
          velocity_error=max(velocity_error,maxval(abs(halo(i,j,k,:)-expected_velocity)))
          coordinate_error=max(coordinate_error,maxval(abs(coordinate_halo(i,j,k,:)-expected_coordinate)))
        enddo
        enddo
        enddo
        if(velocity_error/=0.or.coordinate_error/=0) error stop 'vector/physical coordinate halo owner differs'
        node=0
        do k=0,extent(3)
        do j=0,extent(2)
        do i=0,extent(1)
          node=node+1;packed_xyz(:,node)=xyz(i,j,k,:)
        enddo
        enddo
        enddo
        call check(sample_upload_coordinates(sample%handle,packed_xyz,bytes(size(packed_xyz)), &
          int(offset,c_int),[global,global,global]),'upload saved physical coordinates')
        call check(sample_download(sample%handle,sample_coordinates,halo,bytes(size(halo))),'uploaded geometry halo')
        call check(sample_download(sample%handle,sample_geometry,packed_check,bytes(size(packed_check))), &
          'uploaded packed geometry')
        if(any(transfer(packed_check,[0_c_int64_t],size(packed_check))/= &
          transfer(packed_xyz,[0_c_int64_t],size(packed_xyz)))) error stop 'saved coordinate upload changed words'
        if(any(transfer(halo,[0_c_int64_t],size(halo))/= &
          transfer(coordinate_halo,[0_c_int64_t],size(coordinate_halo)))) error stop 'saved coordinate halo differs'
        call check(index_download(parent,0_c_int,after_values,bytes(size(after_values))),'source post observation')
        if(any(transfer(before_values,[0_c_int64_t],size(before_values))/= &
          transfer(after_values,[0_c_int64_t],size(after_values)))) error stop 'capture modified selected source'
        allocate(observed(size(q)))
        call check(device_state_download(parent,device_q,observed,bytes(size(q))),'parent q post observation')
        if(any(observed/=77)) error stop 'vector/coordinate capture modified parent q'
        deallocate(observed)
        call check(index_end(parent),'release source group')
        checks=checks+1
      enddo
      if(rank==0) velocity(0,0,0,1)=ieee_value(0.d0,ieee_quiet_nan)
      call check(device_state_upload(parent,device_velocity,velocity,bytes(size(velocity))),'invalid selected test column')
      call check(index_begin(parent,0_c_int,int(nodes,c_int),0_c_int,[integer(c_int) ::],5_c_int,fields, &
        spread(0.d0,1,49),0_c_int,xyz,bytes(size(xyz))),'prepare invalid vector source')
      call check(index_evaluate(parent,region),'pack invalid selected test source')
      call check(index_source(parent,0_c_int,values_view),'borrow invalid vector test view')
      status=sample_capture_vector(sample,values_view,slots)
      if(status/=device_nonphysical) error stop 'selected NaN did not fail collectively'
      write(*,'(a,3(1x,i0))') 'CPP_INSITU_VECTOR_REJECT_PASS',rank,debug,mapping
      call check(index_end(parent),'release failed vector source')
      call check(sample_destroy(sample),'release vector/coordinate transport')
      call check(device_state_destroy(parent),'release parent')
    enddo
    enddo
    call MPI_Comm_free(cart,ierr)
  end block
  write(*,'(a,2(1x,i0),1x,a,2(1x,es24.16e3))') 'CPP_INSITU_VECTOR_PASS',rank,checks,trim(transport), &
    velocity_error,coordinate_error
  call MPI_Finalize(ierr)
contains
  subroutine position(g,mapping,value)
    integer,intent(in) :: g(3),mapping
    real(c_double),intent(out) :: value(3)
    real(c_double) :: p(3)
    p=real(g,c_double)*h
    if(mapping==1) then
      value=[p(1)+.15d0*sin(p(1))*sin(p(2)),p(2)+.15d0*sin(p(2))*sin(p(3)),p(3)+.15d0*sin(p(3))*sin(p(1))]
    else
      value=[p(1),p(2)+.15d0*sin(p(1))*sin(p(3)),p(3)]
    endif
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
