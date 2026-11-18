program wall41_probe
  use iso_c_binding
  use ieee_arithmetic, only: ieee_is_finite
  use device_state
  use fluid_explicit
  use boundary_state_cpp
  implicit none
  type(c_ptr) :: state
  real(c_double),allocatable :: q(:,:,:,:),p(:,:,:,:),seed(:,:,:,:),ps(:,:,:,:),actual(:,:,:,:),pa(:,:,:,:),rhs(:,:,:,:),expected_rhs(:,:,:,:)
  real(c_double),allocatable :: metric(:,:,:,:,:)
  logical,allocatable :: selected(:,:,:)
  integer :: h(5),ph(4),im,jm,km,hm,unit,axis,backend,stage,debug,mask,side,i,j,k,c,count=0,boundary_type=41
  real(c_double) :: error,maximum=0,scale
  character(1024) :: root,path
  character(80) :: suffix
  character(3) :: label
  character(3) :: wall_axes='xyz'
  character(80) :: arg
  character(1),parameter :: letters(3)=['x','y','z']
  call get_command_argument(1,root)
  call get_command_argument(2,arg)
  if(len_trim(arg)>0) read(arg,*) boundary_type
  if(boundary_type/=41.and.boundary_type/=42.and.boundary_type/=50.and.boundary_type/=60) &
    error stop 'boundary must be bc41/bc42/bc50/bc60'
  call get_command_argument(3,arg)
  if(len_trim(arg)>0) wall_axes=trim(arg)
  do axis=0,2
  if(index(wall_axes,letters(axis+1))==0) cycle
  do backend=1,2
    label='cpu'
    if(backend==2) label='gpu'
    do stage=1,3
      write(suffix,'(a,i2.2,a)') '.step00000000.rk',stage,'.rank00000000.bin'
      path=trim(root)//'/'//trim(label)//'_'//letters(axis+1)//'/outdat/rhs.pre_rhs'//trim(suffix)
      open(newunit=unit,file=trim(path),status='old',access='stream',form='unformatted')
      read(unit) h
      im=h(1);jm=h(2);km=h(3);hm=h(4)
      if(h(5)/=5) error stop 'wall probe needs five components'
      allocate(q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),p(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6), &
        seed(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),ps(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6), &
        actual(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),pa(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6), &
        rhs(0:im,0:jm,0:km,5),expected_rhs(0:im,0:jm,0:km,5),selected(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
      read(unit) q;close(unit)
      path=trim(root)//'/'//trim(label)//'_'//letters(axis+1)//'/outdat/rhs.pre_rhs_primitives'//trim(suffix)
      open(newunit=unit,file=trim(path),status='old',access='stream',form='unformatted')
      read(unit) ph
      if(any(ph/=h(:4))) error stop 'wall primitive layout'
      read(unit) p;close(unit)
      if(boundary_type==60) then
        allocate(metric(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3))
        path=trim(root)//'/'//trim(label)//'_'//letters(axis+1)//'/geometry_input.bin'
        open(newunit=unit,file=trim(path),status='old',access='stream',form='unformatted')
        read(unit) ph
        if(any(ph/=h(:4))) error stop 'symmetry metric layout'
        read(unit) metric;close(unit)
      endif
      do debug=0,1
        call check(device_state_create(0_c_int,im,jm,km,hm,int(debug,c_int),state))
        call check(fluid_prepare(state,0_c_int));call check(fluid_rhs_prepare(state,0_c_int))
        if(boundary_type==60) call check(device_state_upload(state,device_metric,metric,bytes(size(metric))))
        do mask=1,3
          selected=.false.
          do side=0,1
            if(.not.btest(mask,side)) cycle
            select case(axis)
            case(0);selected(merge(im,0,side==1),0:jm,0:km)=.true.
            case(1);selected(0:im,merge(jm,0,side==1),0:km)=.true.
            case(2);selected(0:im,0:jm,merge(km,0,side==1))=.true.
            end select
          enddo
          seed=q;ps=p
          do c=1,5
            where(selected) seed(:,:,:,c)=-137.d0
          enddo
          do c=1,6
            where(selected) ps(:,:,:,c)=-29.d0
          enddo
          call check(device_state_upload(state,device_q,seed,bytes(size(seed))))
          call check(device_state_upload(state,device_rho,ps(:,:,:,1),bytes(size(selected))))
          call check(device_state_upload(state,device_velocity,ps(:,:,:,2:4),bytes(3*size(selected))))
          call check(device_state_upload(state,device_pressure,ps(:,:,:,5),bytes(size(selected))))
          call check(device_state_upload(state,device_temperature,ps(:,:,:,6),bytes(size(selected))))
          rhs=7;expected_rhs=7
          call check(device_state_upload(state,device_rhs,rhs,bytes(size(rhs))))
          do side=0,1
            if(.not.btest(mask,side)) cycle
            if(boundary_type==41) then
              call check(wall41_basic_state(state,int(axis,c_int),int(side,c_int),1.d0,1.4d0*.3d0**2,1.d0/(1.4d0-1.d0)))
            else if(boundary_type==42) then
              call check(wall42_basic_state(state,int(axis,c_int),int(side,c_int),1.4d0*.3d0**2,1.d0/(1.4d0-1.d0)))
            else if(boundary_type==50) then
              call check(zeroextrap_basic(state,int(axis,c_int),int(side,c_int),1.4d0*.3d0**2,1.d0/(1.4d0-1.d0)))
            else
              call check(symmetry_basic(state,int(axis,c_int),int(side,c_int),1.4d0*.3d0**2,1.d0/(1.4d0-1.d0)))
            endif
          enddo
          call check(device_state_download(state,device_q,actual,bytes(size(actual))))
          call check(device_state_download(state,device_rho,pa(:,:,:,1),bytes(size(selected))))
          call check(device_state_download(state,device_velocity,pa(:,:,:,2:4),bytes(3*size(selected))))
          call check(device_state_download(state,device_pressure,pa(:,:,:,5),bytes(size(selected))))
          call check(device_state_download(state,device_temperature,pa(:,:,:,6),bytes(size(selected))))
          call check(device_state_download(state,device_rhs,rhs,bytes(size(rhs))))
          if(boundary_type==50.or.boundary_type==60) then
            do c=1,5
              where(selected(0:im,0:jm,0:km)) expected_rhs(:,:,:,c)=0
            enddo
          endif
          if(any(transfer(rhs,[0_c_int64_t],size(rhs))/=transfer(expected_rhs,[0_c_int64_t],size(expected_rhs)))) &
            error stop 'boundary command RHS footprint differs'
          error=0
          do c=1,5
            if(.not.all(ieee_is_finite(pack(actual(:,:,:,c),selected)))) error stop 'nonfinite wall q'
            scale=max(1.d0,maxval(abs(q(:,:,:,c)),mask=selected))
            error=max(error,maxval(abs(actual(:,:,:,c)-q(:,:,:,c)),mask=selected)/scale)
          enddo
          do c=1,6
            if(.not.all(ieee_is_finite(pack(pa(:,:,:,c),selected)))) error stop 'nonfinite wall primitive'
            scale=max(1.d0,maxval(abs(p(:,:,:,c)),mask=selected))
            error=max(error,maxval(abs(pa(:,:,:,c)-p(:,:,:,c)),mask=selected)/scale)
          enddo
          if(.not.(error<=2.d-10)) error stop 'wall state differs from real CPU/old GPU closure'
          do k=-hm,km+hm
          do j=-hm,jm+hm
          do i=-hm,im+hm
            if(selected(i,j,k)) cycle
            if(any(transfer(actual(i,j,k,:),[0_c_int64_t],5)/=transfer(seed(i,j,k,:),[0_c_int64_t],5)).or. &
              any(transfer(pa(i,j,k,:),[0_c_int64_t],6)/=transfer(ps(i,j,k,:),[0_c_int64_t],6))) &
              error stop 'wall state changed interior/other face/halo'
          enddo
          enddo
          enddo
          maximum=max(maximum,error);count=count+1
        enddo
        call check(device_state_destroy(state))
      enddo
      deallocate(q,p,seed,ps,actual,pa,rhs,expected_rhs,selected)
      if(allocated(metric)) deallocate(metric)
    enddo
  enddo
  enddo
  if(boundary_type==60) call check_oblique_projection()
  write(*,'(a,i0,a,1x,i0,1x,es24.16e3)') 'CPP_WALL',boundary_type,'_PASS',count,maximum
contains
  subroutine check_oblique_projection()
    real(c_double),parameter :: normals(3,3)=reshape([2.d0,1.d0,-2.d0,-1.d0,2.d0,1.d0,2.d0,0.d0,2.d0],[3,3])
    real(c_double),parameter :: scales(3)=[-2.d0,1.d0,3.d0],velocity(3)=[.3d0,-.2d0,.1d0]
    real(c_double) :: n(3),projector(3,3),v(3),target_q(5),target_p(6),local_error,largest=0,normal_error,normal_max=0
    integer :: a,b,sc,checks=0
    im=4;jm=5;km=6;hm=3
    allocate(q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),p(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6), &
      seed(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),ps(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6), &
      actual(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),pa(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6), &
      metric(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3), &
      rhs(0:im,0:jm,0:km,5),expected_rhs(0:im,0:jm,0:km,5),selected(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
    p(:,:,:,1)=2;p(:,:,:,5)=3;p(:,:,:,6)=1.5d0*1.4d0*.3d0**2
    do c=1,3
      p(:,:,:,c+1)=velocity(c)
    enddo
    q(:,:,:,1)=2;q(:,:,:,2:4)=2*p(:,:,:,2:4)
    q(:,:,:,5)=3.d0/.4d0+sum(velocity**2)
    do axis=0,2
      n=normals(axis+1,:)/sqrt(sum(normals(axis+1,:)**2))
      projector=0
      do a=1,3
        projector(a,a)=1
      enddo
      do b=1,3
      do a=1,3
        projector(a,b)=projector(a,b)-n(a)*n(b)
      enddo
      enddo
      v=matmul(projector,velocity)
      target_q=[2.d0,2*v,3.d0/.4d0+sum(v**2)]
      target_p=[2.d0,v,3.d0,1.5d0*1.4d0*.3d0**2]
      do debug=0,1
        call check(device_state_create(0_c_int,im,jm,km,hm,int(debug,c_int),state))
        call check(fluid_prepare(state,0_c_int));call check(fluid_rhs_prepare(state,0_c_int))
        do side=0,1
        do sc=1,size(scales)
          selected=.false.
          select case(axis)
          case(0);selected(merge(im,0,side==1),0:jm,0:km)=.true.
          case(1);selected(0:im,merge(jm,0,side==1),0:km)=.true.
          case(2);selected(0:im,0:jm,merge(km,0,side==1))=.true.
          end select
          do b=1,3
          do a=1,3
            metric(:,:,:,a,b)=normals(a,b)
          enddo
          enddo
          metric(:,:,:,axis+1,:)=scales(sc)*metric(:,:,:,axis+1,:)
          seed=q;ps=p;rhs=7;expected_rhs=7
          do c=1,5
            where(selected) seed(:,:,:,c)=-137
            where(selected(0:im,0:jm,0:km)) expected_rhs(:,:,:,c)=0
          enddo
          do c=1,6
            where(selected) ps(:,:,:,c)=-29
          enddo
          call check(device_state_upload(state,device_metric,metric,bytes(size(metric))))
          call check(device_state_upload(state,device_q,seed,bytes(size(seed))))
          call check(device_state_upload(state,device_rho,ps(:,:,:,1),bytes(size(selected))))
          call check(device_state_upload(state,device_velocity,ps(:,:,:,2:4),bytes(3*size(selected))))
          call check(device_state_upload(state,device_pressure,ps(:,:,:,5),bytes(size(selected))))
          call check(device_state_upload(state,device_temperature,ps(:,:,:,6),bytes(size(selected))))
          call check(device_state_upload(state,device_rhs,rhs,bytes(size(rhs))))
          call check(symmetry_basic(state,int(axis,c_int),int(side,c_int),1.4d0*.3d0**2,1.d0/.4d0))
          call check(device_state_download(state,device_q,actual,bytes(size(actual))))
          call check(device_state_download(state,device_rho,pa(:,:,:,1),bytes(size(selected))))
          call check(device_state_download(state,device_velocity,pa(:,:,:,2:4),bytes(3*size(selected))))
          call check(device_state_download(state,device_pressure,pa(:,:,:,5),bytes(size(selected))))
          call check(device_state_download(state,device_temperature,pa(:,:,:,6),bytes(size(selected))))
          call check(device_state_download(state,device_rhs,rhs,bytes(size(rhs))))
          local_error=0
          do c=1,5
            if(.not.all(ieee_is_finite(pack(actual(:,:,:,c),selected)))) error stop 'nonfinite projected q'
            local_error=max(local_error,maxval(abs(actual(:,:,:,c)-target_q(c)),mask=selected)/max(1.d0,abs(target_q(c))))
          enddo
          do c=1,6
            if(.not.all(ieee_is_finite(pack(pa(:,:,:,c),selected)))) error stop 'nonfinite projected primitive'
            local_error=max(local_error,maxval(abs(pa(:,:,:,c)-target_p(c)),mask=selected)/max(1.d0,abs(target_p(c))))
          enddo
          normal_error=maxval(abs(pa(:,:,:,2)*n(1)+pa(:,:,:,3)*n(2)+pa(:,:,:,4)*n(3)),mask=selected)
          if(.not.(local_error<=2.d-10).or..not.(normal_error<=2.d-10)) error stop 'oblique normal projection differs'
          if(any(transfer(rhs,[0_c_int64_t],size(rhs))/=transfer(expected_rhs,[0_c_int64_t],size(expected_rhs)))) &
            error stop 'projection RHS footprint differs'
          do k=-hm,km+hm
          do j=-hm,jm+hm
          do i=-hm,im+hm
            if(selected(i,j,k)) cycle
            if(any(transfer(actual(i,j,k,:),[0_c_int64_t],5)/=transfer(seed(i,j,k,:),[0_c_int64_t],5)).or. &
              any(transfer(pa(i,j,k,:),[0_c_int64_t],6)/=transfer(ps(i,j,k,:),[0_c_int64_t],6))) &
              error stop 'projection changed other nodes/halo'
          enddo
          enddo
          enddo
          largest=max(largest,local_error);normal_max=max(normal_max,normal_error);checks=checks+1
        enddo
        enddo
        call check(device_state_destroy(state))
      enddo
    enddo
    deallocate(q,p,seed,ps,actual,pa,metric,rhs,expected_rhs,selected)
    write(*,'(a,1x,i0,2(1x,es24.16e3))') 'CPP_SYMMETRY_PROJECTION_PASS',checks,largest,normal_max
  end subroutine
  integer(c_size_t) function bytes(n)
    integer,intent(in) :: n
    bytes=int(n,c_size_t)*8_c_size_t
  end function
  subroutine check(status)
    integer,intent(in) :: status
    if(status==0) return
    print *, 'wall41 device status=',status
    error stop 1
  end subroutine
end program
