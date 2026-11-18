program native_diagnostics_probe
  use iso_c_binding
  use ieee_arithmetic, only: ieee_is_finite
  use mpi
  use astr_precision, only: real_kind,real_bytes,real_to_word
  use astr_precision_mpi, only: mpi_real_type
  use device_state
  use fluid_explicit
  implicit none
  integer,parameter :: total=16,hm=5
  integer :: rank,ranks,local_rank,ierr,status,cart,topology(3),coord(3),upper(3),origin(3)
  integer :: im,jm,km,i,j,k,c,d,fixture,debug,nondim,checks=0
  integer(c_int64_t) :: keys(4),expected_keys(4),global_keys(4),local_keys(4),key
  type(c_ptr) :: state
  real(real_kind),parameter :: sentinel=-12345.5_real_kind
  real(real_kind) :: rates(5),values(6),statistics(15),expected(13),local_rates(4),global_rates(4),point_rates(4)
  real(real_kind) :: profile(4,0:total),expected_profile(4,0:total),temperature,sound,u,v,w,error
  real(real_kind),allocatable :: q(:,:,:,:),vel(:,:,:,:),tmp(:,:,:),metric(:,:,:,:,:),observed(:,:,:,:)
  character(128) :: text
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text,status=status)
  if(status/=0) error stop 'native diagnostics requires Open MPI local rank'
  read(text,*) local_rank
  call check(device_bind(int(local_rank,c_int)))
  call MPI_Init(ierr);call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  call get_command_argument(1,text);read(text,*) topology
  if(product(topology)/=ranks.or.any(mod(total,topology)/=0)) error stop 'native diagnostics topology'
  call MPI_Cart_create(MPI_COMM_WORLD,3,topology,[.true.,.true.,.true.],.false.,cart,ierr);call check(ierr)
  call MPI_Cart_coords(cart,rank,3,coord,ierr);call check(ierr)
  upper=total/topology;origin=coord*upper;im=upper(1);jm=upper(2);km=upper(3)
  allocate(q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),observed(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5))
  allocate(vel(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3),tmp(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
  allocate(metric(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3))
  do fixture=0,1
    metric=0;metric(:,:,:,1,1)=2;metric(:,:,:,2,2)=4;metric(:,:,:,3,3)=8
    tmp=4
    do k=-hm,km+hm
    do j=-hm,jm+hm
    do i=-hm,im+hm
      u=.25_real_kind;v=-.5_real_kind;w=1
      if(fixture==1) then
        u=real(i+origin(1),real_kind)/8
        v=real(total-i-origin(1),real_kind)/8
        w=real(k+origin(3),real_kind)/8
      endif
      vel(i,j,k,:)=[u,v,w]
      q(i,j,k,:)=[2._real_kind,2*u,2*v,2*w,8._real_kind+u*u+v*v+w*w]
    enddo
    enddo
    enddo
    do debug=0,1
      call check(device_state_create(int(local_rank,c_int),int(im,c_int),int(jm,c_int),int(km,c_int), &
        hm,int(debug,c_int),state))
      call check(fluid_prepare(state,1_c_int))
      call check(device_state_upload(state,device_q,q,bytes(size(q))))
      call check(fluid_refresh_primitives(state,2._real_kind,2._real_kind))
      ! Exact primitive construction above gives rho=2, pressure=T=4.
      call check(device_state_upload(state,device_metric,metric,bytes(size(metric))))
      do nondim=0,1
        rates=sentinel;values=sentinel;profile=sentinel;statistics=sentinel
        local_rates=0;expected_keys=huge(key);expected_profile=0
        sound=2
        if(nondim==1) sound=1
        do k=0,km
        do j=0,jm
        do i=0,im
          point_rates(1)=2*(abs(vel(i,j,k,1))+sound)
          point_rates(2)=4*(abs(vel(i,j,k,2))+sound)
          point_rates(3)=8*(abs(vel(i,j,k,3))+sound)
          point_rates(4)=sum(point_rates(1:3))
          key=(int(k+origin(3),c_int64_t)*(total+1)+j+origin(2))*(total+1)+i+origin(1)
          do c=1,4
            if(point_rates(c)>local_rates(c)) then
              local_rates(c)=point_rates(c);expected_keys(c)=key
            elseif(point_rates(c)==local_rates(c)) then
              expected_keys(c)=min(expected_keys(c),key)
            endif
          enddo
          expected_profile(:,j+origin(2))=max(expected_profile(:,j+origin(2)),point_rates)
        enddo
        enddo
        enddo
        call check(fluid_cfl(state,int(nondim,c_int),2._real_kind,2._real_kind,.5_real_kind,rates(2:4)))
        call check(fluid_cfl_diagnostics(state,int(nondim,c_int),2._real_kind,2._real_kind,.5_real_kind, &
          [total,total,total],origin,total+1,values(2:5),keys,profile))
        if(any(rates([1,5])/=sentinel).or.any(values([1,6])/=sentinel)) error stop 'native CFL buffer width'
        if(any(rates(2:4)/=local_rates(1:3)).or.any(values(2:5)/=local_rates)) error stop 'native CFL analytic maxima'
        if(any(keys/=expected_keys)) error stop 'native CFL deterministic local tie keys'
        if(any(profile(:,origin(2):origin(2)+jm)/=expected_profile(:,origin(2):origin(2)+jm))) &
          error stop 'native CFL plane profile'
        if(any(profile(:,:origin(2)-1)/=sentinel).or.any(profile(:,origin(2)+jm+1:)/=sentinel)) &
          error stop 'native CFL profile overwrote another rank selection'
        call MPI_Allreduce(local_rates,global_rates,4,MPI_REAL_TYPE,MPI_MAX,cart,ierr);call check(ierr)
        local_keys=huge(key)
        where(local_rates==global_rates) local_keys=keys
        call MPI_Allreduce(local_keys,global_keys,4,MPI_INTEGER8,MPI_MIN,cart,ierr);call check(ierr)
        if(fixture==1.and.nondim==1) then
          if(any(global_rates/=[6._real_kind,12._real_kind,24._real_kind,38._real_kind])) &
            error stop 'native global directional maxima versus nodal-sum maximum'
          if(any(global_keys/=[16_c_int64_t,0_c_int64_t,4624_c_int64_t,4624_c_int64_t])) &
            error stop 'native global tie keys'
        endif
        checks=checks+1
      enddo
      call check(fluid_gradient(state))
      call check(fluid_statistics(state,2._real_kind,1._real_kind,2._real_kind,statistics(2:14)))
      if(any(statistics([1,15])/=sentinel).or..not.all(ieee_is_finite(statistics))) &
        error stop 'native statistics buffer width or finite state'
      if(fixture==0) then
        expected(1:3)=[2.625_real_kind*im*jm*km,0._real_kind,0._real_kind]
        expected(4:8)=[2._real_kind,.5_real_kind,-1._real_kind,2._real_kind,9.3125_real_kind]
        expected(9:13)=expected(4:8)
        if(any(real_to_word(statistics(2:14))/=real_to_word(expected))) error stop 'native constant statistics'
        call constant_wall_integrals()
      endif
      call check(device_state_download(state,device_q,observed,bytes(size(observed))))
      if(any(real_to_word(observed)/=real_to_word(q))) error stop 'diagnostics modified authoritative q'
      print *, 'CPP_NATIVE_DIAGNOSTICS ',rank,real_bytes,fixture,debug,checks,statistics(2:4)
      call check(device_state_destroy(state))
    enddo
  enddo
  ! Failure remains latched; diagnostics must not turn a bad temperature into a valid rate.
  call check(device_state_create(int(local_rank,c_int),int(im,c_int),int(jm,c_int),int(km,c_int),hm,0_c_int,state))
  call check(fluid_prepare(state,0_c_int));tmp=-1
  call check(device_state_upload(state,device_velocity,vel,bytes(size(vel))))
  call check(device_state_upload(state,device_temperature,tmp,bytes(size(tmp))))
  call check(device_state_upload(state,device_metric,metric,bytes(size(metric))))
  status=fluid_cfl(state,1_c_int,2._real_kind,2._real_kind,.5_real_kind,rates(2:4))
  if(status/=device_invalid_spectrum) error stop 'native CFL failure status'
  tmp=4;call check(device_state_upload(state,device_temperature,tmp,bytes(size(tmp))))
  if(fluid_check(state)/=device_invalid_spectrum) error stop 'native CFL failure was cleared'
  call check(device_state_destroy(state))
  print *, 'CPP_NATIVE_DIAGNOSTICS_PASS ',rank,real_bytes,checks
  call MPI_Comm_free(cart,ierr);call check(ierr);call MPI_Finalize(ierr)
contains
  subroutine constant_wall_integrals()
    real(real_kind),allocatable :: coordinate(:,:,:),volume(:,:,:),area(:,:,:)
    real(real_kind) :: channel(5),flat(7),cells
    allocate(coordinate(0:im,0:jm,0:km),volume(0:im-1,0:jm-1,0:km-1),area(0:im-1,0:0,0:km-1))
    do j=0,jm
      coordinate(:,j,:)=real(j+origin(2),real_kind)
    enddo
    volume=1;area=1;cells=real(im*jm*km,real_kind)
    call check(fluid_channel_geometry_prepare(state))
    call check(device_state_upload(state,device_coordinate_y,coordinate,bytes(size(coordinate))))
    channel=sentinel
    call check(fluid_channel_statistics(state,2._real_kind,1._real_kind,2._real_kind,1_c_int,1_c_int,channel(2:4)))
    if(any(channel([1,5])/=sentinel).or.any(channel(2:4)/=[.5_real_kind*cells,0._real_kind,cells])) &
      error stop 'native constant channel integral or buffer width'
    call check(fluid_flat_geometry_prepare(state,1_c_int))
    call check(device_state_upload(state,device_cell_volume,volume,bytes(size(volume))))
    call check(device_state_upload(state,device_wall_cell_area,area,bytes(size(area))))
    flat=sentinel
    call check(fluid_flat_statistics(state,2._real_kind,1._real_kind,2._real_kind,1._real_kind,1._real_kind, &
      1_c_int,flat(2:6)))
    if(any(flat([1,7])/=sentinel).or.any(flat(2:6)/= &
      [.5_real_kind*cells,0._real_kind,cells,0._real_kind,real(im*km,real_kind)])) &
      error stop 'native constant plate integral or buffer width'
    print *, 'CPP_NATIVE_DIAGNOSTICS_WALL ',rank,real_bytes,debug
  end subroutine
  integer(c_size_t) function bytes(n)
    integer,intent(in) :: n
    bytes=int(n,c_size_t)*real_bytes
  end function
  subroutine check(code)
    integer,intent(in) :: code
    if(code==0) return
    print *, 'NATIVE_DIAGNOSTICS_STATUS ',rank,fixture,debug,code
    call MPI_Abort(MPI_COMM_WORLD,1,ierr)
  end subroutine
end program
