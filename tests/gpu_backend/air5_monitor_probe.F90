program air5_monitor_probe
  use iso_fortran_env,only:real64,int64
  use iso_c_binding,only:c_ptr,c_null_ptr,c_int,c_size_t
  use ieee_arithmetic,only:ieee_value,ieee_quiet_nan
  use mpi
  use commvar
  use commarray
  use parallel
  use chemistry_monitor,only:write_air5_flow_monitor
#ifdef ASTR_AIR5_MONITOR_REFERENCE
  use cudafor
  use commarray_gpu
  use chemistry_monitor_gpu,only:collect_air5_monitor_gpu
#else
  use device_state
  use chemistry_device_properties,only:air5_device_prepare
  use chemistry_device_statistics,only:air5_monitor_prepare,air5_monitor_collect
#endif
  implicit none
  real(real64),allocatable :: frames(:,:,:,:,:),samples(:),back(:,:,:,:)
  real(real64) :: xyz(3,6),packet(15,6),expected(15,6)
  integer :: raw_nodes(3,6),probes(3,4),local(3),coord(3),topology(3),rank,ranks,cart,device,hm, &
    nframes,frame,unit,output,point,i,j,k,d,found,status,ierr
  character(1024) :: inputs,mode,backend,text,path
  logical :: periods(3),owned
#ifndef ASTR_AIR5_MONITOR_REFERENCE
  type(c_ptr) :: context=c_null_ptr
#endif
  call get_command_argument(1,inputs);call get_command_argument(2,mode);call get_command_argument(3,backend)
  do i=1,3
    call get_command_argument(i+3,text);read(text,*)topology(i)
  enddo
  use_gpu=trim(backend)=='gpu';lreadgrid=trim(mode)=='curved-probes'
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text,status=status)
  if(status/=0)error stop 'local rank unavailable'
  read(text,*)rank;device=modulo(rank,2)
  if(use_gpu)then
#ifdef ASTR_AIR5_MONITOR_REFERENCE
    status=cudaSetDevice(device);if(status/=0)error stop 'reference monitor GPU binding'
#else
    call check(device_bind(int(device,c_int)))
#endif
  endif
  call MPI_Init(ierr);call check(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call check(ierr);mpirank=rank
  call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr);call check(ierr)
  if(product(topology)/=ranks)error stop 'monitor topology'
  periods=.false.
  call MPI_Cart_create(MPI_COMM_WORLD,3,topology,periods,.false.,cart,ierr);call check(ierr)
  call MPI_Cart_coords(cart,rank,3,coord,ierr);call check(ierr)
  write(path,'(a,"/input.rank",i4.4,".bin")')trim(inputs),rank
  open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='old')
  read(unit)im,jm,km,hm,nframes
  allocate(q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,11),back(-hm:im+hm,-hm:jm+hm,-hm:km+hm,11), &
    x(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3), &
    frames(-hm:im+hm,-hm:jm+hm,-hm:km+hm,11,nframes),samples(nframes))
  read(unit)samples,x,frames;close(unit)
  ia=im*topology(1);ja=jm*topology(2);ka=km*topology(3)
  ig0=im*coord(1);jg0=jm*coord(2);kg0=km*coord(3)
  probes(:,1)=[0,0,0];probes(:,2)=[ia/2,ja/2,ka/2];probes(:,3)=[ia,ja,ka];probes(:,4)=[ia/2,0,ka/2]
  nmonitor=0
  allocate(imon(4,4));imon=0
  if(trim(mode)/='profiles'.and.trim(mode)/='none')then
    do point=1,4
      local=probes(:,point)-[ig0,jg0,kg0]
      owned=all(local>=0).and.all(local<=[im,jm,km])
      owned=owned.and..not.(local(1)==im.and.ig0+im<ia)
      owned=owned.and..not.(local(2)==jm.and.jg0+jm<ja)
      owned=owned.and..not.(local(3)==km.and.kg0+km<ka)
      if(.not.owned)cycle
      nmonitor=nmonitor+1;imon(nmonitor,:)=[local,point]
    enddo
    ! The production consumer uses imon(:,4), so keep only actual rows.
    if(nmonitor<4)imon=imon(1:nmonitor,:)
  else
    imon=imon(1:0,:)
  endif
  if(use_gpu)then
#ifdef ASTR_AIR5_MONITOR_REFERENCE
    allocate(q_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,11),x_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3))
    x_d=x
#else
    call check(device_state_create(int(device,c_int),int(im,c_int),int(jm,c_int),int(km,c_int),int(hm,c_int),1_c_int,context))
    call check(air5_device_prepare(context))
#endif
  endif
  if(trim(mode)=='raw')then
    raw_nodes(:,1)=[0,0,0];raw_nodes(:,2)=[im,jm,km];raw_nodes(:,3)=[im/2,jm/2,km/2]
    raw_nodes(:,4)=raw_nodes(:,3);raw_nodes(:,5)=[-1,-99,-99];raw_nodes(:,6)=[1,1,1]
    xyz=ieee_value(0.d0,ieee_quiet_nan)
    do point=1,6
      i=raw_nodes(1,point);j=raw_nodes(2,point);k=raw_nodes(3,point)
      if(i>=0)xyz(:,point)=x(i,j,k,:)
    enddo
#ifndef ASTR_AIR5_MONITOR_REFERENCE
    if(use_gpu)call check(air5_monitor_prepare(context,6_c_int,raw_nodes,xyz))
#endif
    write(path,'("raw.rank",i4.4,".bin")')rank
    open(newunit=output,file=trim(path),access='stream',form='unformatted',status='new')
    write(output)6,nframes
  endif
  do frame=1,nframes
    nstep=frame-1;q=frames(:,:,:,:,frame)
    if(use_gpu)then
#ifdef ASTR_AIR5_MONITOR_REFERENCE
      q_d=q
#else
      call check(device_state_upload(context,device_q,q,int(size(q),c_size_t)*8_c_size_t))
#endif
    endif
    if(trim(mode)=='raw')then
      expected=0.d0
      do point=1,6
        i=raw_nodes(1,point);j=raw_nodes(2,point);k=raw_nodes(3,point)
        if(i<0)cycle
        expected(1:3,point)=x(i,j,k,:);expected(4:14,point)=q(i,j,k,:);expected(15,point)=1.d0
      enddo
      if(use_gpu)then
#ifdef ASTR_AIR5_MONITOR_REFERENCE
        call collect_air5_monitor_gpu(raw_nodes,packet)
#else
        call check(air5_monitor_collect(context,packet,int(size(packet),c_size_t)*8_c_size_t))
#endif
      else
        packet=expected
      endif
      if(any(transfer(packet,[0_int64],size(packet))/=transfer(expected,[0_int64],size(expected))))error stop 'raw monitor packet words'
      write(output)packet
    else
      if(use_gpu)q=ieee_value(0.d0,ieee_quiet_nan)
#ifdef ASTR_AIR5_MONITOR_REFERENCE
      call write_air5_flow_monitor(samples(frame))
#else
      call write_air5_flow_monitor(samples(frame),context)
#endif
    endif
    if(use_gpu)then
#ifdef ASTR_AIR5_MONITOR_REFERENCE
      back=q_d
#else
      call check(device_state_download(context,device_q,back,int(size(back),c_size_t)*8_c_size_t))
#endif
      if(any(transfer(back,[0_int64],size(back))/=transfer(frames(:,:,:,:,frame),[0_int64],size(back)))) &
        error stop 'monitor changed conservative field words'
    endif
  enddo
  if(trim(mode)=='raw')close(output)
#ifndef ASTR_AIR5_MONITOR_REFERENCE
  if(use_gpu)then
    if(trim(mode)=='none')then
      status=air5_monitor_collect(context,packet,int(size(packet),c_size_t)*8_c_size_t)
      if(status/=device_not_allocated)error stop 'disabled monitor allocated device state'
    endif
    call check(device_state_destroy(context))
  endif
#endif
  write(*,'(a,i0,3a)')'AIR5_MONITOR_PASS rank=',rank,' mode=',trim(mode),' packets/fields checked'
  call MPI_Comm_free(cart,ierr);call check(ierr);call MPI_Finalize(ierr)
contains
  subroutine check(code)
    integer,intent(in) :: code
    if(code/=0)then
      write(*,*)'monitor device/MPI API failure',code
      error stop
    endif
  end subroutine
end program
