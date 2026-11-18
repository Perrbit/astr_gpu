program insitu_wall_probe
  use iso_c_binding
  use mpi
  use ieee_arithmetic, only: ieee_is_finite
  use device_state
  use fluid_explicit
  use insitu_wall_cpp
  implicit none
  integer,parameter :: real_kind=c_device_real_kind,real_bytes=storage_size(0._real_kind)/8,hm=5,global=24
  type(c_ptr) :: parent,wall,restored
  type(device_view) :: view
  integer :: rank,ranks,local_rank,status,ierr,comm,dims(3),coords(3),neighbors(2,3),extent(3),offset(3), &
    i,j,k,a,b,w,first,nw,nodes,layers,curve,frame,debug,unit,header(9)
  integer(c_int) :: lower(3),upper(3),sizes(3),periodic(3),mode
  integer(c_int64_t) :: before(10),after(10)
  real(real_kind),allocatable :: q(:,:,:,:),original(:,:,:,:),observed(:,:,:,:),metric(:,:,:,:,:), &
    geometry(:,:,:,:),fields(:,:,:,:),line(:,:),initial(:,:,:,:),packed(:,:,:,:),roundtrip(:,:,:,:),means(:,:,:,:)
  real(real_kind) :: material(7),h,x,y,z,rho,v(3),temperature,pressure,mu,expected(4),error,maximum,all_error,kappa, &
    weights(2),expected_mean(3),central,statistic_error
  logical :: packet
  character(1400) :: arg,source,output
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',arg,status=status)
  if(status/=0) error stop 'wall probe requires matching Open MPI'
  read(arg,*) local_rank
  call check(device_bind(int(local_rank,c_int)),'bind before MPI')
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  call get_command_argument(1,arg);packet=trim(arg)=='packet'
  if(packet) then
    if(ranks/=1) error stop 'frozen packet is a single-rank component comparison'
    call get_command_argument(2,source);call get_command_argument(3,output)
    open(newunit=unit,file=trim(source),access='stream',form='unformatted',status='old');read(unit) header
    extent=header(1:3);if(header(4)/=hm.or.header(5)/=real_bytes) error stop 'wall packet type/halo'
    nw=header(6);first=header(7);layers=header(8);curve=header(9)
    dims=1;offset=0
  else
    read(arg,*) dims
    call get_command_argument(2,arg)
    mode=0;if(trim(arg)=='device-aware') mode=1
    if(trim(arg)/='pinned'.and.trim(arg)/='device-aware') error stop 'wall transport selection'
    call get_command_argument(3,arg);curve=0;layers=3
    if(trim(arg)=='curve') curve=1
    if(trim(arg)=='sixth') layers=8
    if(trim(arg)/='curve'.and.trim(arg)/='legacy'.and.trim(arg)/='sixth') error stop 'wall closure selection'
    if(product(dims)/=ranks.or.any(mod(global,dims)/=0)) error stop 'wall topology'
  endif
  call MPI_Cart_create(MPI_COMM_WORLD,3,dims,[.true.,.false.,.true.],.false.,comm,ierr)
  call MPI_Cart_coords(comm,rank,3,coords,ierr)
  do a=1,3
    call MPI_Cart_shift(comm,a-1,1,neighbors(1,a),neighbors(2,a),ierr)
  enddo
  lower=int(neighbors(1,:),c_int);upper=int(neighbors(2,:),c_int);sizes=int(dims,c_int);periodic=[1,0,1]
  if(.not.packet) then
    extent=global/dims;offset=coords*extent;nw=0;first=extent(2)
    if(offset(2)==0) then
      first=0;nw=nw+1
    endif
    if(offset(2)+extent(2)==global) nw=nw+1
  else
    mode=0
  endif
  nodes=(extent(1)+1)*(extent(3)+1)*nw
  allocate(q(-hm:extent(1)+hm,-hm:extent(2)+hm,-hm:extent(3)+hm,5), &
    metric(-hm:extent(1)+hm,-hm:extent(2)+hm,-hm:extent(3)+hm,3,3), &
    geometry(0:extent(1),0:extent(3),nw,9),fields(0:extent(1),0:extent(3),nw,4),line(6,0:extent(1)))
  allocate(original,mold=q);allocate(observed,mold=q)
  allocate(initial(0:extent(1),0:extent(3),nw,3),packed(0:extent(1),0:extent(3),nw,34), &
    roundtrip(0:extent(1),0:extent(3),nw,34),means(0:extent(1),0:extent(3),nw,4))
  if(packet) then
    read(unit) material,q,metric,geometry;close(unit)
  endif
  h=2*acos(-1._real_kind)/global
  kappa=(1.5_real_kind*sin(h)-.3_real_kind*sin(2*h)+sin(3*h)/30)/h
  maximum=0
  do debug=0,1
    call check(device_state_create(int(local_rank,c_int),int(extent(1),c_int),int(extent(2),c_int), &
      int(extent(3),c_int),hm,int(debug,c_int),parent),'create parent')
    call check(fluid_prepare(parent,1_c_int),'prepare primitives')
    if(.not.packet) then
      material=[1._real_kind,2._real_kind,2._real_kind,100._real_kind,.72_real_kind,.5_real_kind,1.5_real_kind]
      metric=0
      do a=1,3
        metric(:,:,:,a,a)=1/h
      enddo
      do w=1,nw
        j=first;if(w==2) j=extent(2)
        do k=0,extent(3)
        do i=0,extent(1)
          geometry(i,k,w,1:3)=real(offset+[i,j,k],real_kind)*h
          geometry(i,k,w,4:6)=[0._real_kind,real(merge(-1,1,j==extent(2)),real_kind),0._real_kind]
          geometry(i,k,w,7:9)=[1._real_kind,0._real_kind,0._real_kind]
        enddo
        enddo
      enddo
    endif
    call check(device_state_upload(parent,device_metric,metric,bytes(size(metric))),'upload static metric')
    call check(wall_create(parent,int(comm,c_int),lower,upper,sizes,periodic,mode,int(first,c_int),int(nw,c_int), &
      int(layers,c_int),int(curve,c_int),material,geometry,wall),'create private wall')
    status=device_state_destroy(parent)
    if(status/=device_invalid_argument.or..not.c_associated(parent)) error stop 'wall parent ownership'
    do frame=0,merge(0,1,packet)
      if(.not.packet) then
        q=-77
        do k=0,extent(3)
        do j=0,extent(2)
        do i=0,extent(1)
          x=(offset(1)+i)*h;y=(offset(2)+j)*h;z=(offset(3)+k)*h
          rho=1+.05_real_kind*cos(x)*cos(z)
          v=[.1_real_kind+.03_real_kind*y+.02_real_kind*sin(x)+.01_real_kind*cos(z), &
            .025_real_kind*sin(x),.02_real_kind*cos(z)]
          temperature=2+.04_real_kind*y+.01_real_kind*sin(x)
          if(frame==0) then
            rho=2;v=[.125_real_kind,0._real_kind,0._real_kind];temperature=2
          endif
          pressure=rho*temperature/material(1)
          q(i,j,k,:)=[rho,rho*v,material(3)*pressure+.5_real_kind*rho*sum(v*v)]
          ! Canonical upper x/z values are supplied by the private sample, not the solver.
          if(i==extent(1).or.k==extent(3)) q(i,j,k,:)=-13
        enddo
        enddo
        enddo
      endif
      original=q
      call check(device_state_upload(parent,device_q,q,bytes(size(q))),'upload frame')
      call check(wall_metrics(wall,before),'capture counters before')
      call check(wall_capture(wall),'capture wall')
      call check(wall_metrics(wall,after),'capture counters after')
      if(after(8)/=before(8).or.after(7)-before(7)/=4*(1+curve).or.after(9)-before(9)/=1) &
        error stop 'implicit wall field download or scalar/capture accounting'
      if(mode==1.or.ranks==1) then
        if(after(5)/=0.or.after(6)/=0) error stop 'device/local wall has host staging'
      endif
      call check(wall_view(wall,0_c_int,view),'wall view')
      if(view%element_bytes/=real_bytes.or.any(view%extent(1:4)/=[extent(1)+1,extent(3)+1,nw,4])) &
        error stop 'wall view shape/kind'
      call check(wall_download(wall,fields,bytes(size(fields))),'explicit wall observation')
      if(.not.all(ieee_is_finite(fields))) error stop 'nonfinite wall quantities'
      weights=0
      if(frame==1) weights=[.25_real_kind,.75_real_kind]
      call check(wall_statistics_push(wall,weights,int(frame,c_int)),'unit-density wall moments')
      if(frame==0) initial=fields(:,:,:,1:3)
      if(packet) then
        write(arg,'(a,".",i1,".bin")') trim(output),debug
        open(newunit=unit,file=trim(arg),access='stream',form='unformatted',status='new')
        write(unit) fields;close(unit)
      else
        do w=1,nw
          j=first;if(w==2) j=extent(2)
          do k=0,extent(3)
          do i=0,extent(1)
            x=(offset(1)+i)*h;y=(offset(2)+j)*h
            rho=1+.05_real_kind*cos(x)*cos((offset(3)+k)*h);temperature=2+.04_real_kind*y+.01_real_kind*sin(x)
            if(frame==0) then
              rho=2;temperature=2
            endif
            pressure=rho*temperature/material(1)
            mu=temperature*sqrt(temperature)*material(7)/(temperature+material(6))/material(4)
            expected=[pressure,mu*.03_real_kind,-((mu/material(5))/material(2))*.04_real_kind, &
              real(merge(-1,1,j==extent(2)),real_kind)]
            expected(2:3)=expected(2:3)*expected(4)
            if(curve==1) then
              expected(2)=mu*(.03_real_kind+.025_real_kind*kappa*cos(x))*expected(4)
            endif
            if(frame==0) expected(2:3)=0
            error=maxval(abs(fields(i,k,w,:)-expected)/max(1._real_kind,abs(expected)))
            maximum=max(maximum,error)
          enddo
          enddo
          call check(wall_line(wall,0_c_int,int(w-1,c_int),line,bytes(size(line))),'selected wall line')
          if(any(line(1:3,:)/=transpose(geometry(:,0,w,1:3))).or. &
            any(line(4:6,:)/=transpose(fields(:,0,w,1:3)))) error stop 'selected wall line differs from view'
        enddo
        if(real_bytes==8.and.maximum>2e-10_real_kind) error stop 'analytic wall FP64 threshold'
      endif
      call check(device_state_download(parent,device_q,observed,bytes(size(observed))),'parent after observation')
      if(any(transfer(observed,[0_c_int8_t],size(observed)*real_bytes)/= &
        transfer(original,[0_c_int8_t],size(original)*real_bytes))) error stop 'wall changed parent q'
    enddo
    call check(wall_statistics_pack(wall,packed,bytes(size(packed)),[1._real_kind,2._real_kind], &
      real(merge(0,2,packet),real_kind)),'wall wire state')
    if(.not.packet) then
      statistic_error=0
      do w=1,nw
      do k=0,extent(3)
      do i=0,extent(1)
        expected_mean=.25_real_kind*initial(i,k,w,:)+.75_real_kind*fields(i,k,w,1:3)
        statistic_error=max(statistic_error,maxval(abs(packed(i,k,w,10:12)-expected_mean)))
        if(any(packed(i,k,w,[1,2,3,4,8,9,34])/=[1._real_kind,2._real_kind,2._real_kind, &
          1._real_kind,1._real_kind,1._real_kind,1._real_kind])) error stop 'wall wire metadata/duration'
        if(any(packed(i,k,w,5:7)/=fields(i,k,w,1:3)).or. &
          any(packed(i,k,w,10:12)/=packed(i,k,w,13:15)).or. &
          any(packed(i,k,w,16:24)/=packed(i,k,w,25:33))) error stop 'wall unit-density wire identity'
        do b=1,3
        do a=1,3
          central=.1875_real_kind*(fields(i,k,w,a)-initial(i,k,w,a))*(fields(i,k,w,b)-initial(i,k,w,b))
          statistic_error=max(statistic_error,abs(packed(i,k,w,15+a+3*(b-1))-central))
        enddo
        enddo
      enddo
      enddo
      enddo
      if(real_bytes==8.and.statistic_error>2e-10_real_kind) error stop 'wall moments independent two-point formula'
      maximum=max(maximum,statistic_error)
      call check(wall_create(parent,int(comm,c_int),lower,upper,sizes,periodic,mode,int(first,c_int),int(nw,c_int), &
        int(layers,c_int),int(curve,c_int),material,geometry,restored),'restored wall owner')
      call check(wall_statistics_restore(restored,packed,bytes(size(packed))),'restore wire state')
      call check(wall_statistics_pack(restored,roundtrip,bytes(size(roundtrip)),[1._real_kind,2._real_kind], &
        2._real_kind),'restored wire state')
      if(any(transfer(packed,[0_c_int8_t],size(packed)*real_bytes)/= &
        transfer(roundtrip,[0_c_int8_t],size(roundtrip)*real_bytes))) error stop 'wall state roundtrip differs'
      call check(wall_capture(restored),'capture mean destination geometry')
      call check(wall_statistics_mean(restored,wall,1._real_kind),'device mean from original history')
      call check(wall_download(restored,means,bytes(size(means))),'original wall means')
      call check(wall_statistics_mean(wall,restored,1._real_kind),'device mean from restored history')
      call check(wall_download(wall,fields,bytes(size(fields))),'restored wall means')
      if(any(transfer(fields,[0_c_int8_t],size(fields)*real_bytes)/= &
        transfer(means,[0_c_int8_t],size(means)*real_bytes))) error stop 'wall restored means differ'
      call check(wall_destroy(restored),'release restored wall owner')
    endif
    if(nw>0) then
      q=original;q(0,first,0,1)=0
      call check(device_state_upload(parent,device_q,q,bytes(size(q))),'explicit invalid wall input')
    endif
    status=wall_capture(wall)
    if(status/=device_nonphysical) error stop 'invalid wall state was not rejected collectively'
    status=wall_view(wall,0_c_int,view)
    if(status/=device_not_allocated) error stop 'failed wall capture exposed old fields'
    call check(device_state_upload(parent,device_q,original,bytes(size(original))),'restore probe input')
    q=original
    call check(wall_destroy(wall),'release wall');if(c_associated(wall)) error stop 'wall release handle'
    call check(wall_create(parent,int(comm,c_int),lower,upper,sizes,periodic,mode,int(first,c_int),0_c_int, &
      int(layers,c_int),int(curve,c_int),material,geometry,wall),'empty-wall rank metadata')
    call check(wall_capture(wall),'empty-wall collective capture')
    call check(wall_view(wall,0_c_int,view),'empty-wall view')
    if(view%extent(3)/=0) error stop 'empty-wall ownership was lost'
    call check(wall_destroy(wall),'release empty wall')
    call check(device_state_destroy(parent),'release parent')
  enddo
  call MPI_Allreduce(maximum,all_error,1,merge(MPI_DOUBLE_PRECISION,MPI_REAL,real_bytes==8),MPI_MAX,comm,ierr)
  if(rank==0) write(*,'(a,5(1x,i0),1x,es24.16)') 'CPP_WALL_SAMPLE_PASS',real_bytes,ranks,dims,all_error
  call MPI_Comm_free(comm,ierr);call MPI_Finalize(ierr)
contains
  integer(c_size_t) function bytes(n)
    integer,intent(in) :: n
    bytes=int(n,c_size_t)*real_bytes
  end function
  subroutine check(code,message)
    integer,intent(in) :: code
    character(*),intent(in) :: message
    if(code/=0) then
      print *,trim(message),' status=',code
      call MPI_Abort(MPI_COMM_WORLD,code,ierr)
    endif
  end subroutine
end program
