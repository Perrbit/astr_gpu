program flat_statistics_probe
  use iso_c_binding
  use mpi
  use device_state
  use fluid_explicit
  use production_statistics_cpp
  implicit none
  integer,parameter :: g(3)=[16,16,16],hm=5
  type(c_ptr) :: state
  type(device_view) :: before,after
  integer :: rank,ranks,local_rank,ierr,axis,upper(3),origin(3),i,j,k,debug,repeat,cases=0,diagnostic
  integer(c_int) :: wall
  real(c_double),allocatable :: q(:,:,:,:),observed(:,:,:,:),p(:,:,:,:),gradient(:,:,:,:,:), &
    dtemp(:,:,:,:),volumes(:,:,:),areas(:,:,:),coordinates(:,:,:,:),normal(:,:,:),wall_moments(:,:)
  real(c_float),allocatable :: diagnostic_gradient(:,:,:,:,:),diagnostic_thermal(:,:,:,:)
  real(c_double) :: local(5),total(5),expected(5),previous(5),maximum=0,error,y,den,t,u(3),dx,dy,dz,wall_expected(3)
  character(80) :: text
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text,status=ierr)
  if(ierr/=0) error stop 'flat statistics probe needs matching MPI launcher'
  read(text,*) local_rank
  call check(device_bind(int(local_rank,c_int)))
  call MPI_Init(ierr);call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  call get_command_argument(1,text);read(text,*) axis
  diagnostic=0;call get_command_argument(2,text)
  if(len_trim(text)>0) read(text,*) diagnostic
  upper=g;upper(axis)=g(axis)/ranks;origin=0;origin(axis)=rank*upper(axis);wall=merge(1,0,origin(2)==0)
  allocate(q(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,5), &
    observed(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,5), &
    p(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,6), &
    gradient(0:upper(1),0:upper(2),0:upper(3),3,3),dtemp(0:upper(1),0:upper(2),0:upper(3),3), &
    volumes(1:upper(1),1:upper(2),1:upper(3)),areas(1:upper(1),1,1:upper(3)))
  allocate(coordinates(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,3), &
    normal(0:upper(1),0:upper(3),3),wall_moments(0:upper(1),3))
  do k=-hm,upper(3)+hm
  do j=-hm,upper(2)+hm
  do i=-hm,upper(1)+hm
    y=(real(origin(2)+j,c_double)/g(2))**2;den=2;t=1+.5d0*y;u=[3*y,0.d0,.25d0]
    p(i,j,k,:)=[den,u,den*t/5.6d0,t]
    q(i,j,k,:)=[den,den*u,p(i,j,k,5)*2.5d0+.5d0*den*sum(u*u)]
    coordinates(i,j,k,:)=[10.d0*(origin(1)+i)/g(1),y,.25d0*(origin(3)+k)/g(3)]
  enddo
  enddo
  enddo
  dx=10.d0/g(1);dz=.25d0/g(3)
  do k=1,upper(3)
  do j=1,upper(2)
    dy=(real(origin(2)+j,c_double)/g(2))**2-(real(origin(2)+j-1,c_double)/g(2))**2
    volumes(:,j,k)=dx*dy*dz
  enddo
  enddo
  areas=dx*dz;gradient=0;gradient(:,:,:,1,2)=3;dtemp=0;dtemp(:,:,:,2)=.5d0
  normal=0;normal(:,:,2)=1
  expected=[7.5d0,7.5d0/1000.d0,.0625d0*product(g),1.25d0/(1000.d0*.72d0*1.6d0),2.5d0]
  if(diagnostic==1) then
    allocate(diagnostic_gradient,mold=real(gradient,c_float));allocate(diagnostic_thermal,mold=real(dtemp,c_float))
    diagnostic_gradient=0;diagnostic_gradient(:,:,:,1,2)=7.25_c_float
    diagnostic_thermal=0;diagnostic_thermal(:,:,:,2)=.375_c_float
    expected(2)=2.5d0*7.25d0/1000.d0;expected(4)=2.5d0*.375d0/(1000.d0*.72d0*1.6d0)
  endif
  do debug=0,1
    call check(device_state_create(int(local_rank,c_int),upper(1),upper(2),upper(3),hm,int(debug,c_int),state))
    call check(fluid_prepare(state,1_c_int));call check(fluid_rhs_prepare(state,0_c_int))
    if(device_state_view(state,device_cell_volume,before)/=device_not_allocated) error stop 'unrequested flat geometry'
    call check(fluid_flat_geometry_prepare(state,wall));call check(device_state_view(state,device_cell_volume,before))
    call check(fluid_flat_geometry_prepare(state,wall));call check(device_state_view(state,device_cell_volume,after))
    if(.not.c_associated(before%data,after%data)) error stop 'flat geometry storage was not reused'
    if(product(before%extent)/=product(upper)) error stop 'flat cell volumes contain duplicate nodes'
    if(wall==0.and.device_state_view(state,device_wall_cell_area,after)/=device_not_allocated) error stop 'nonwall rank has wall storage'
    call check(device_state_upload(state,device_cell_volume,volumes,bytes(size(volumes))))
    if(wall==1) call check(device_state_upload(state,device_wall_cell_area,areas,bytes(size(areas))))
    call check(device_state_upload(state,device_q,q,bytes(size(q))))
    call check(device_state_upload(state,device_rho,p(:,:,:,1),bytes(size(p(:,:,:,1)))))
    call check(device_state_upload(state,device_velocity,p(:,:,:,2:4),bytes(3*size(p(:,:,:,1)))))
    call check(device_state_upload(state,device_pressure,p(:,:,:,5),bytes(size(p(:,:,:,1)))))
    call check(device_state_upload(state,device_temperature,p(:,:,:,6),bytes(size(p(:,:,:,1)))))
    call check(device_state_upload(state,device_dvelocity,gradient,bytes(size(gradient))))
    call check(device_state_upload(state,device_dtemperature,dtemp,bytes(size(dtemp))))
    if(diagnostic==1) then
      call check(fluid_prepare_diagnostic_gradient(state))
      call check(device_state_upload_f32(state,device_diagnostic_dvelocity,diagnostic_gradient,int(size(gradient),c_size_t)*4))
      call check(device_state_upload_f32(state,device_diagnostic_dtemperature,diagnostic_thermal,int(size(dtemp),c_size_t)*4))
    endif
    do repeat=1,2
      call check(fluid_flat_statistics(state,1000.d0,110.3d0/273.15d0,1.d0+110.3d0/273.15d0,.72d0,1.6d0,wall,local))
      if(repeat==2) then
        if(any(transfer(local,[0_c_int64_t],5)/=transfer(previous,[0_c_int64_t],5))) error stop 'flat repeat changed bits'
      endif
      previous=local
      call MPI_Allreduce(local,total,5,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr);call check(ierr)
      error=maxval(abs(total-expected)/max(1.d0,abs(expected)))
      if(.not.(error<=2.d-10)) then
        write(*,*) 'FLAT_STATISTICS_DIFFER',rank,total,expected,error
        error stop 'flat statistics differ from analytic volume and wall integrals'
      endif
      maximum=max(maximum,error);cases=cases+1
    enddo
    call check(production_geometry_cpp(state,coordinates,normal,upper(1),upper(2),upper(3),hm,g(1),origin(1),wall==1))
    if(wall==1) then
      wall_expected=[2.d0/5.6d0,merge(7.25d0,3.d0,diagnostic==1)/1000.d0, &
        merge(.375d0,.5d0,diagnostic==1)/(1000.d0*.72d0*1.6d0)]*(upper(3)*dz)
      do repeat=1,2
        call check(production_wall_cpp(state,merge(.25d0,0.d0,repeat==2),int(repeat-1,c_int),1_c_int, &
          [1000.d0,.72d0,1.6d0,3.5d0,110.3d0/273.15d0,1.d0+110.3d0/273.15d0]))
        call check(production_transfer_cpp(state,production_wall_endpoint,0_c_int,wall_moments,bytes(size(wall_moments))))
        error=maxval(abs(wall_moments-spread(wall_expected,1,upper(1)+1)))
        if(.not.(error<=2.d-10)) error stop 'wall diagnostic endpoint differs from analytic integral'
        maximum=max(maximum,error)
      enddo
      call check(production_transfer_cpp(state,production_wall_sum,0_c_int,wall_moments,bytes(size(wall_moments))))
      error=maxval(abs(wall_moments-.25d0*spread(wall_expected,1,upper(1)+1)))
      if(.not.(error<=2.d-10)) error stop 'wall diagnostic time integral differs from analytic value'
      maximum=max(maximum,error)
    endif
    call check(device_state_download(state,device_q,observed,bytes(size(q))))
    if(any(transfer(observed,[0_c_int64_t],size(q))/=transfer(q,[0_c_int64_t],size(q)))) error stop 'flat statistics changed q'
    call check(device_state_destroy(state))
  enddo
  write(*,'(a,2(1x,i0),1x,es24.16e3)') 'CPP_FLAT_STATISTICS_PASS',rank,cases,maximum
  call MPI_Finalize(ierr)
contains
  integer(c_size_t) function bytes(n)
    integer,intent(in) :: n
    bytes=int(n,c_size_t)*8
  end function
  subroutine check(status)
    integer,intent(in) :: status
    if(status==0) return
    write(*,*) 'FLAT_STATISTICS_STATUS',rank,status
    error stop 'flat statistics operation failed'
  end subroutine
end program
