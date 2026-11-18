program channel_probe
  use iso_c_binding
  use ieee_arithmetic, only: ieee_is_finite
  use mpi
  use device_state
  use fluid_explicit
  implicit none
  type(c_ptr) :: state
  type(device_view) :: coordinates_before,coordinates_after
  integer :: h(4),g(3),upper(3),origin(3),hm,unit,rank,ranks,local_rank,split_axis, &
    backend,step,debug,repeat,n,c,i,j,k,ierr,cases=0,force_cases=0
  real(c_double),allocatable :: gq(:,:,:,:),gp(:,:,:,:),gx(:,:,:,:),gg(:,:,:,:,:),gj(:,:,:), &
    q(:,:,:,:),p(:,:,:,:),x(:,:,:,:),grad(:,:,:,:,:),jac(:,:,:),rhs(:,:,:,:),expected(:,:,:,:),after(:,:,:,:)
  real(c_double) :: reference(3),local(3),global(3),old(3),force(3),maximum=0,force_max=0,error
  character(1024) :: root,path
  character(80) :: text,suffix
  character(3) :: label
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text,status=ierr)
  if(ierr/=0) error stop 'channel probe needs matching Open MPI launcher'
  read(text,*) local_rank
  ierr=device_bind(int(local_rank,c_int))
  if(ierr/=0) error stop 'channel device binding failed'
  call MPI_Init(ierr);call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  if(ranks/=1.and.ranks/=2) error stop 'channel probe covers NP=1/2'
  call get_command_argument(1,root)
  split_axis=1
  if(ranks==2) then
    call get_command_argument(2,text);read(text,*) split_axis
    if(split_axis<1.or.split_axis>3) error stop 'channel split axis must be 1/2/3'
  endif
  do backend=1,2
    label='cpu'
    if(backend==2) label='gpu'
    do step=1,3
      write(suffix,'(a,i8.8,a)') '.step',step,'.bin'
      path=trim(root)//'/'//trim(label)//trim(suffix)
      open(newunit=unit,file=trim(path),status='old',access='stream',form='unformatted')
      read(unit) h;g=h(:3);hm=h(4);upper=g;origin=0
      if(ranks==2) then
        if(mod(g(split_axis),2)/=0) error stop 'even channel intervals required'
        upper(split_axis)=g(split_axis)/2;origin(split_axis)=rank*upper(split_axis)
      endif
      allocate(gq(-hm:g(1)+hm,-hm:g(2)+hm,-hm:g(3)+hm,5),gp(-hm:g(1)+hm,-hm:g(2)+hm,-hm:g(3)+hm,6), &
        gx(0:g(1),0:g(2),0:g(3),3),gg(0:g(1),0:g(2),0:g(3),3,3),gj(-hm:g(1)+hm,-hm:g(2)+hm,-hm:g(3)+hm), &
        q(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,5),p(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,6), &
        x(0:upper(1),0:upper(2),0:upper(3),3),grad(0:upper(1),0:upper(2),0:upper(3),3,3), &
        jac(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm),rhs(0:upper(1),0:upper(2),0:upper(3),5), &
        expected(0:upper(1),0:upper(2),0:upper(3),5),after(-hm:upper(1)+hm,-hm:upper(2)+hm,-hm:upper(3)+hm,5))
      read(unit) gq,gp,gx,gg,gj,reference;close(unit)
      q=gq(origin(1)-hm:origin(1)+upper(1)+hm,origin(2)-hm:origin(2)+upper(2)+hm,origin(3)-hm:origin(3)+upper(3)+hm,:)
      p=gp(origin(1)-hm:origin(1)+upper(1)+hm,origin(2)-hm:origin(2)+upper(2)+hm,origin(3)-hm:origin(3)+upper(3)+hm,:)
      jac=gj(origin(1)-hm:origin(1)+upper(1)+hm,origin(2)-hm:origin(2)+upper(2)+hm,origin(3)-hm:origin(3)+upper(3)+hm)
      x=gx(origin(1):origin(1)+upper(1),origin(2):origin(2)+upper(2),origin(3):origin(3)+upper(3),:)
      grad=gg(origin(1):origin(1)+upper(1),origin(2):origin(2)+upper(2),origin(3):origin(3)+upper(3),:,:)
      do debug=0,1
        call check(device_state_create(int(local_rank,c_int),int(upper(1),c_int),int(upper(2),c_int), &
          int(upper(3),c_int),hm,int(debug,c_int),state))
        call check(fluid_prepare(state,1_c_int));call check(fluid_rhs_prepare(state,0_c_int))
        if(device_state_view(state,device_coordinate_y,coordinates_before)/=device_not_allocated) &
          error stop 'coordinates allocated without a consumer'
        call check(fluid_channel_geometry_prepare(state))
        call check(device_state_view(state,device_coordinate_y,coordinates_before))
        if(coordinates_before%rank/=3.or.coordinates_before%extent(4)/=1) error stop 'channel geometry stores unused coordinates'
        call check(device_state_upload(state,device_q,q,bytes(size(q))))
        call check(device_state_upload(state,device_velocity,p(:,:,:,2:4),bytes(3*size(jac))))
        call check(device_state_upload(state,device_temperature,p(:,:,:,6),bytes(size(jac))))
        call check(device_state_upload(state,device_coordinate_y,x(:,:,:,2),bytes(size(x(:,:,:,2)))))
        call check(device_state_upload(state,device_dvelocity,grad,bytes(size(grad))))
        call check(device_state_upload(state,device_jacobian,jac,bytes(size(jac))))
        do repeat=1,2
          call check(fluid_channel_statistics(state,3000.d0,110.3d0/273.15d0,1.d0+110.3d0/273.15d0, &
            int(merge(1,0,origin(2)==0),c_int),int(merge(1,0,origin(2)+upper(2)==g(2)),c_int),local))
          if(.not.all(ieee_is_finite(local))) error stop 'nonfinite channel partial'
          if(repeat==2) then
            if(any(transfer(local,[0_c_int64_t],3)/=transfer(old,[0_c_int64_t],3))) error stop 'channel repeat changed bits'
          endif
          old=local
          call MPI_Allreduce(local,global,3,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr)
          if(ierr/=MPI_SUCCESS) error stop 'channel scalar reduction'
          global(1:2)=global(1:2)/real(g(1)*g(3),c_double)
          global(3)=sqrt(global(3)/real(product(g),c_double))
          error=maxval(abs(global-reference)/max(1.d0,abs(reference)))
          if(.not.(error<=2.d-10)) then
            write(*,*) label,step,split_axis,global,reference,error
            error stop 'channel scalars differ from frozen flowstate'
          endif
          maximum=max(maximum,error);cases=cases+1
        enddo
        call check(device_state_view(state,device_coordinate_y,coordinates_after))
        if(.not.c_associated(coordinates_before%data,coordinates_after%data)) error stop 'coordinate pointer changed'
        do n=0,1
          force=real(n,c_double)*[1.d-4,-2.d-4,3.d-4]
          rhs=7;expected=rhs
          do k=0,upper(3)
          do j=0,upper(2)
          do i=0,upper(1)
            expected(i,j,k,2:4)=expected(i,j,k,2:4)+force*jac(i,j,k)
            expected(i,j,k,5)=expected(i,j,k,5)+(force(1)*(q(i,j,k,2)/q(i,j,k,1))+ &
              force(2)*(q(i,j,k,3)/q(i,j,k,1))+force(3)*(q(i,j,k,4)/q(i,j,k,1)))*jac(i,j,k)
          enddo
          enddo
          enddo
          call check(device_state_upload(state,device_rhs,rhs,bytes(size(rhs))))
          call check(fluid_uniform_force(state,force))
          call check(device_state_download(state,device_rhs,rhs,bytes(size(rhs))))
          if(.not.all(ieee_is_finite(rhs))) error stop 'nonfinite force RHS'
          if(any(transfer(rhs(:,:,:,1),[0_c_int64_t],size(rhs(:,:,:,1)))/= &
            transfer(expected(:,:,:,1),[0_c_int64_t],size(rhs(:,:,:,1))))) error stop 'force changed density RHS'
          do c=1,5
            error=maxval(abs(rhs(:,:,:,c)-expected(:,:,:,c)))/max(1.d0,maxval(abs(expected(:,:,:,c))))
            if(.not.(error<=2.d-10)) error stop 'force differs from momentum/energy work identity'
            force_max=max(force_max,error)
          enddo
          force_cases=force_cases+1
        enddo
        call check(device_state_download(state,device_q,after,bytes(size(after))))
        if(any(transfer(q,[0_c_int64_t],size(q))/=transfer(after,[0_c_int64_t],size(after)))) &
          error stop 'statistics/force changed q'
        call check(device_state_destroy(state))
      enddo
      deallocate(gq,gp,gx,gg,gj,q,p,x,grad,jac,rhs,expected,after)
    enddo
  enddo
  write(*,'(a,2(1x,i0),1x,es24.16e3)') 'CPP_CHANNEL_SCALARS_PASS',rank,cases,maximum
  write(*,'(a,2(1x,i0),1x,es24.16e3)') 'CPP_CHANNEL_FORCE_PASS',rank,force_cases,force_max
  call MPI_Finalize(ierr)
contains
  integer(c_size_t) function bytes(n)
    integer,intent(in) :: n
    bytes=int(n,c_size_t)*8_c_size_t
  end function
  subroutine check(status)
    integer,intent(in) :: status
    if(status==0) return
    write(*,*) 'CHANNEL_PROBE_STATUS ',rank,backend,step,debug,status
    call MPI_Abort(MPI_COMM_WORLD,1,ierr)
  end subroutine
end program
