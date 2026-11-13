program rhs_probe
  use iso_c_binding
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use mpi
  use hdf5
  use device_state
  use device_halo
  use halo_transport
  use fluid_explicit
  use fluid_step, only: fluid_advance_stage
  implicit none
  integer :: ierr,rank,status,local_rank,debug,stage,unit,im,jm,km,hm,nq,n,a,b,m,i,j,k,step_index,rk
  integer(c_int) :: header(5),pheader(4)
  character(1200) :: directory,path
  character(80) :: suffix,name,text
  type(c_ptr) :: handle
  real(c_double),allocatable :: q(:,:,:,:),primitives(:,:,:,:),metric(:,:,:,:,:),jac(:,:,:), &
    geometry(:,:,:,:),rhs(:,:,:,:),reference(:,:,:,:),scratch(:,:,:,:),expected(:,:,:,:)
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text,status=status)
  if(status/=0) error stop 'RHS probe requires Open MPI local rank'
  read(text,*) local_rank
  status=device_bind(int(local_rank,c_int))
  if(status/=0) error stop 'RHS probe pre-MPI binding failed'
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
  call get_command_argument(1,directory)
  call get_command_argument(2,text)
  read(text,*) stage
  write(suffix,'(a,i2.2,a)') '.step00000000.rk',stage,'.rank00000000.bin'
  path=trim(directory)//'/rhs.pre_rhs'//trim(suffix)
  open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='old',action='read')
  read(unit) header
  im=header(1);jm=header(2);km=header(3);hm=header(4);nq=header(5)
  if(nq/=5) call fail('five frozen components required')
  allocate(q(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5),primitives(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6), &
    metric(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3),jac(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
    geometry(0:im,0:jm,0:km,13),rhs(0:im,0:jm,0:km,5),reference(0:im,0:jm,0:km,5), &
    scratch(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6),expected(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6))
  read(unit) q
  close(unit)
  open(newunit=unit,file=trim(directory)//'/rhs.pre_rhs_primitives'//trim(suffix), &
    access='stream',form='unformatted',status='old',action='read')
  read(unit) pheader
  if(any(pheader/=header(1:4))) call fail('primitive header mismatch')
  read(unit) primitives
  close(unit)
  call read_geometry()
  do k=-hm,km+hm
  do j=-hm,jm+hm
  do i=-hm,im+hm
    jac(i,j,k)=geometry(modulo(i,im),modulo(j,jm),modulo(k,km),4)
    m=4
    do b=1,3
    do a=1,3
      m=m+1
      metric(i,j,k,a,b)=geometry(modulo(i,im),modulo(j,jm),modulo(k,km),m)
    enddo
    enddo
  enddo
  enddo
  enddo
  ! Preserve distinct endpoint metric values from the frozen CPU resource.
  jac(0:im,0:jm,0:km)=geometry(:,:,:,4)
  m=4
  do b=1,3
  do a=1,3
    m=m+1
    metric(0:im,0:jm,0:km,a,b)=geometry(:,:,:,m)
  enddo
  enddo
  do debug=0,1
    call check(device_state_create(int(local_rank,c_int),int(im,c_int),int(jm,c_int),int(km,c_int), &
      int(hm,c_int),int(debug,c_int),handle),'create')
    call check(fluid_prepare(handle,1_c_int),'prepare primitives and gradients')
    call check(fluid_rhs_prepare(handle,1_c_int),'prepare RHS')
    call check(device_transport_prepare(handle,int(MPI_COMM_SELF,c_int),[0_c_int,0_c_int,0_c_int], &
      [0_c_int,0_c_int,0_c_int],[1_c_int,1_c_int,1_c_int],6_c_int,halo_pinned,0_c_int),'prepare local transport')
    call local_halo_check()
    call upload(device_q,q)
    call check(device_state_upload(handle,device_rho,primitives(:,:,:,1),bytes(size(jac))),'rho upload')
    call upload(device_velocity,primitives(:,:,:,2:4))
    call check(device_state_upload(handle,device_pressure,primitives(:,:,:,5),bytes(size(jac))),'pressure upload')
    call check(device_state_upload(handle,device_temperature,primitives(:,:,:,6),bytes(size(jac))),'temperature upload')
    call check(device_state_upload(handle,device_metric,metric,bytes(size(metric))),'metric upload')
    call check(device_state_upload(handle,device_jacobian,jac,bytes(size(jac))),'Jacobian upload')
    call read_q('pre_rhs',1,scratch(:,:,:,1:5))
    call upload(device_q,scratch(:,:,:,1:5))
    call check(fluid_rk_save(handle),'save stage-one origin')
    call upload(device_q,q)
    call check(fluid_convection(handle),'convection')
    call check(device_state_download(handle,device_rhs,rhs,bytes(size(rhs))),'convection download')
    call read_rhs('conv')
    call compare_rhs('convection')
    call check(fluid_gradient(handle),'gradient')
    scratch=0.d0
    call upload(device_stress,scratch)
    call upload(device_heat_flux,scratch(:,:,:,1:3))
    call check(fluid_viscous_flux(handle,1600.d0,.72d0,(1.4d0-1.d0)*.1d0**2, &
      110.3d0/273.15d0,1.d0+110.3d0/273.15d0),'viscous flux')
    do a=0,2
      call check(device_periodic_axis(handle,device_stress,int(a,c_int),0_c_int),'stress halo')
      call check(device_periodic_axis(handle,device_heat_flux,int(a,c_int),0_c_int),'heat flux halo')
    enddo
    call check(fluid_viscous_divergence(handle),'viscous divergence')
    call check(device_state_download(handle,device_rhs,rhs,bytes(size(rhs))),'full RHS download')
    call read_rhs('full')
    call compare_rhs('full_rhs')
    select case(stage)
    case(1)
      call check(fluid_rk_update(handle,1.d-3,1.d0,0.d0,1.d0),'RK first stage')
    case(2)
      call check(fluid_rk_update(handle,1.d-3,.75d0,.25d0,.25d0),'RK second stage')
    case(3)
      call check(fluid_rk_update(handle,1.d-3,1.d0/3.d0,2.d0/3.d0,2.d0/3.d0),'RK third stage')
    end select
    call check(device_state_download(handle,device_q,scratch(:,:,:,1:5),bytes(size(q))),'RK state download')
    call read_q('post_update',stage,expected(:,:,:,1:5))
    call compare_state('rk_state')
    if(stage==1) then
      call check(fluid_filter_prepare(handle),'filter workspace prepare')
      call read_q('initialized',1,scratch(:,:,:,1:5))
      call upload(device_q,scratch(:,:,:,1:5))
      do a=0,2
        call check(device_periodic_axis(handle,device_q,int(a,c_int),0_c_int),'input filter halo')
      enddo
      call check(fluid_filter_axis(handle,0_c_int),'filter x')
      call check(device_periodic_axis(handle,device_filter_work,1_c_int,0_c_int),'filter y halo')
      call check(fluid_filter_axis(handle,1_c_int),'filter y')
      call check(device_periodic_axis(handle,device_q,2_c_int,0_c_int),'filter z halo')
      call check(fluid_filter_axis(handle,2_c_int),'filter z')
      call check(fluid_filter_commit(handle),'filter commit')
      do a=0,2
        call check(device_periodic_axis(handle,device_q,int(a,c_int),1_c_int),'filtered endpoint reconciliation')
      enddo
      do a=0,2
        call check(device_periodic_axis(handle,device_q,int(a,c_int),0_c_int),'filtered final halo')
      enddo
      call check(device_state_download(handle,device_q,scratch(:,:,:,1:5),bytes(size(q))),'filtered state download')
      call read_q('pre_rhs',1,expected(:,:,:,1:5))
      call compare_state('filtered_state')
      call read_q('initialized',1,scratch(:,:,:,1:5))
      call upload(device_q,scratch(:,:,:,1:5))
      do step_index=1,4
        do rk=1,3
          call check(fluid_advance_stage(handle,rk,1.d-3,1.4d0*.1d0**2,1.d0/(1.4d0-1.d0), &
            1600.d0,.72d0,(1.4d0-1.d0)*.1d0**2,110.3d0/273.15d0,1.d0+110.3d0/273.15d0, &
            .true.,.true.,device_exchange_axis,device_exchange_begin,device_exchange_end),'coupled periodic RK stage')
        enddo
        if(mod(step_index,2)==0) then
          call check(device_state_download(handle,device_q,scratch(:,:,:,1:5),bytes(size(q))),'trajectory download')
          call read_checkpoint(step_index)
          write(name,'(a,i0)') 'trajectory_step',step_index
          call compare_rhs(trim(name))
        endif
      enddo
    endif
    call check(device_state_destroy(handle),'destroy')
  enddo
  call MPI_Finalize(ierr)
contains
  subroutine fail(message)
    character(*),intent(in) :: message
    write(*,'(a,i0,2a)') 'RHS FAIL rank=',rank,': ',message
    call MPI_Abort(MPI_COMM_WORLD,1,ierr)
    error stop 1
  end subroutine
  subroutine check(code,message)
    integer,intent(in) :: code
    character(*),intent(in) :: message
    if(code/=0) then
      write(*,'(a,i0)') 'device status=',code
      call fail(message)
    endif
  end subroutine
  integer(c_size_t) function bytes(count)
    integer,intent(in) :: count
    bytes=int(count,c_size_t)*c_sizeof(0.d0)
  end function
  subroutine upload(field,values)
    integer(c_int),intent(in) :: field
    real(c_double),contiguous,intent(in) :: values(:,:,:,:)
    call check(device_state_upload(handle,field,values,bytes(size(values))),'field upload')
  end subroutine
  subroutine read_geometry()
    integer(hid_t) :: file,dataset
    integer(hsize_t) :: dims(3)
    call h5open_f(status)
    call check(status,'HDF5 initialize')
    call h5fopen_f(trim(directory)//'/new/resources/geometry.h5',H5F_ACC_RDONLY_F,file,status)
    call check(status,'frozen geometry open')
    dims=int([im+1,jm+1,km+1],hsize_t)
    do n=1,13
      write(name,'(a,i4.4)') 'q',n
      call h5dopen_f(file,trim(name),dataset,status)
      call check(status,'geometry dataset')
      call h5dread_f(dataset,H5T_NATIVE_DOUBLE,geometry(:,:,:,n),dims,status)
      call check(status,'geometry read')
      call h5dclose_f(dataset,status)
      call check(status,'geometry dataset close')
    enddo
    call h5fclose_f(file,status)
    call check(status,'geometry close')
    call h5close_f(status)
    call check(status,'HDF5 finalize')
  end subroutine
  subroutine read_checkpoint(step)
    integer,intent(in) :: step
    integer(hid_t) :: file,dataset
    integer(hsize_t) :: dims(3)
    character(1200) :: filename
    write(filename,'(a,a,i12.12,a)') trim(directory),'/new/checkpoints/step',step,'/state.h5'
    call h5open_f(status)
    call check(status,'checkpoint reader initialize')
    call h5fopen_f(trim(filename),H5F_ACC_RDONLY_F,file,status)
    call check(status,'frozen checkpoint open')
    dims=int([im+1,jm+1,km+1],hsize_t)
    do n=1,5
      write(name,'(a,i4.4)') 'q',n
      call h5dopen_f(file,trim(name),dataset,status)
      call check(status,'checkpoint q dataset')
      call h5dread_f(dataset,H5T_NATIVE_DOUBLE,reference(:,:,:,n),dims,status)
      call check(status,'checkpoint q read')
      call h5dclose_f(dataset,status)
      call check(status,'checkpoint dataset close')
    enddo
    call h5fclose_f(file,status)
    call check(status,'checkpoint reader close')
    call h5close_f(status)
    call check(status,'checkpoint reader finalize')
    rhs=scratch(0:im,0:jm,0:km,1:5)
  end subroutine
  subroutine read_rhs(label)
    character(*),intent(in) :: label
    integer :: h(4)
    open(newunit=unit,file=trim(directory)//'/rhs.'//label//trim(suffix),access='stream', &
      form='unformatted',status='old',action='read')
    read(unit) h
    if(any(h/=[im,jm,km,5])) call fail('RHS header')
    read(unit) reference
    close(unit)
  end subroutine
  subroutine read_q(label,rk,values)
    character(*),intent(in) :: label
    integer,intent(in) :: rk
    real(c_double),contiguous,intent(out) :: values(:,:,:,:)
    integer :: h(5)
    character(80) :: tag
    write(tag,'(a,i2.2,a)') '.step00000000.rk',rk,'.rank00000000.bin'
    open(newunit=unit,file=trim(directory)//'/rhs.'//label//trim(tag),access='stream', &
      form='unformatted',status='old',action='read')
    read(unit) h
    if(any(h/=header)) call fail('q snapshot header')
    read(unit) values
    close(unit)
  end subroutine
  subroutine compare_state(label)
    character(*),intent(in) :: label
    rhs=scratch(0:im,0:jm,0:km,1:5)
    reference=expected(0:im,0:jm,0:km,1:5)
    call compare_rhs(label)
  end subroutine
  subroutine compare_rhs(label)
    character(*),intent(in) :: label
    real(c_double) :: error
    if(.not.all(ieee_is_finite(rhs)).or..not.all(ieee_is_finite(reference))) call fail('nonfinite RHS')
    do n=1,5
      error=maxval(abs(rhs(:,:,:,n)-reference(:,:,:,n)))/max(1.d0,maxval(abs(reference(:,:,:,n))))
      write(*,'(a,i0,a,i0,3a,i0,1x,es24.16)') 'RHS rank=',rank,' debug=',debug,' ',label,'_',n,error
      if(.not.(error<=2.d-10)) call fail(label)
    enddo
  end subroutine
  subroutine local_halo_check()
    real(c_double),allocatable :: masked(:,:,:,:)
    do n=1,6
    do k=-hm,km+hm
    do j=-hm,jm+hm
    do i=-hm,im+hm
      scratch(i,j,k,n)=real(i+32*j+1024*k+32768*n,c_double)/8.d0
    enddo
    enddo
    enddo
    enddo
    expected=scratch
    ! Independent host contract: original qswap endpoint reconciliation, followed
    ! by its final dataswap pass. Compare all halo, endpoint and corner bits.
    expected(0,0:jm,0:km,:)=.5d0*(expected(0,0:jm,0:km,:)+expected(im,0:jm,0:km,:))
    expected(im,0:jm,0:km,:)=expected(0,0:jm,0:km,:)
    expected(0:im,0,0:km,:)=.5d0*(expected(0:im,0,0:km,:)+expected(0:im,jm,0:km,:))
    expected(0:im,jm,0:km,:)=expected(0:im,0,0:km,:)
    expected(0:im,0:jm,0,:)=.5d0*(expected(0:im,0:jm,0,:)+expected(0:im,0:jm,km,:))
    expected(0:im,0:jm,km,:)=expected(0:im,0:jm,0,:)
    expected(-hm:-1,0:jm,0:km,:)=expected(im-hm:im-1,0:jm,0:km,:)
    expected(im+1:im+hm,0:jm,0:km,:)=expected(1:hm,0:jm,0:km,:)
    expected(0:im,-hm:-1,0:km,:)=expected(0:im,jm-hm:jm-1,0:km,:)
    expected(0:im,jm+1:jm+hm,0:km,:)=expected(0:im,1:hm,0:km,:)
    expected(0:im,0:jm,-hm:-1,:)=expected(0:im,0:jm,km-hm:km-1,:)
    expected(0:im,0:jm,km+1:km+hm,:)=expected(0:im,0:jm,1:hm,:)
    call upload(device_stress,scratch)
    do a=0,2
      call check(device_periodic_axis(handle,device_stress,int(a,c_int),1_c_int),'periodic reconcile')
    enddo
    do a=0,2
      call check(device_periodic_axis(handle,device_stress,int(a,c_int),0_c_int),'final halo refresh')
    enddo
    call check(device_state_download(handle,device_stress,scratch,bytes(size(scratch))),'periodic download')
    if(any(transfer(scratch,[0_c_int64_t],size(scratch))/=transfer(expected,[0_c_int64_t],size(expected)))) &
      call fail('local periodic endpoint and halo bits differ')
    write(*,'(a,i0,a,i0,a)') 'RHS rank=',rank,' debug=',debug,' periodic endpoint/halo/corners exact'
    ! Qualify the observation copy independently of halo-to-peer comparisons.
    expected(4,3,3,2)=transfer(int(z'8000000000000000',c_int64_t),0.d0)
    call upload(device_stress,expected)
    allocate(masked(-hm:im+hm,-hm:jm+hm,-hm:km+hm,6))
    do a=0,2
      masked=expected
      do k=-hm,km+hm
      do j=-hm,jm+hm
      do i=-hm,im+hm
        if((a/=0.and.(i<0.or.i>im)).or.(a/=1.and.(j<0.or.j>jm)).or.(a/=2.and.(k<0.or.k>km))) &
          masked(i,j,k,:)=0.d0
      enddo
      enddo
      enddo
      call check(device_state_snapshot_axis(handle,device_stress,int(a,c_int),scratch,bytes(size(scratch))), &
        'six-component observation copy')
      if(any(transfer(scratch,[0_c_int64_t],size(scratch))/=transfer(masked,[0_c_int64_t],size(masked)))) &
        call fail('observation values, signed zero or padding bits differ')
    enddo
    deallocate(masked)
    write(*,'(a,i0,a,i0,a)') 'RHS rank=',rank,' debug=',debug,' six-component observation copies exact'
  end subroutine
end program
