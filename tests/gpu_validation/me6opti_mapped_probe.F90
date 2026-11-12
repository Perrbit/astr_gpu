program me6opti_mapped_probe
  use astr_precision, only: real_kind
  use me6opti_operator, only: me6_mapped_direction
  use hdf5
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
#ifdef TEST_CUDA
  use cudafor
  use me6opti_operator_gpu
#endif
  implicit none
  integer,parameter :: hm=5
  real(real_kind),parameter :: pi=acos(-1.e0_real_kind)
  real(real_kind),parameter :: ua(3)=[0.2e0_real_kind,0.13e0_real_kind,-0.17e0_real_kind]
  real(real_kind),parameter :: ub(3)=[0.07e0_real_kind,-0.03e0_real_kind,0.04e0_real_kind]
  real(real_kind),allocatable :: coordinates(:,:,:,:),metrics(:,:,:,:,:),jac(:,:,:)
  real(real_kind) :: v(-hm:hm,-hm:hm,-hm:hm,3),t(-hm:hm,-hm:hm,-hm:hm)
  real(real_kind) :: mu(-hm:hm,-hm:hm,-hm:hm),kappa(-hm:hm,-hm:hm,-hm:hm)
  real(real_kind) :: g(-hm:hm,-hm:hm,-hm:hm,3,3),jlocal(-hm:hm,-hm:hm,-hm:hm)
  real(real_kind) :: phase,wave(3),u(3),du(3),ddu(3),grad(3,3),stress(3,3)
  real(real_kind) :: momentum(3),energy,heat,mut,mup,kt,kp,td,tdd,w2,trace,actual(4),part(4),error(5),full_energy
  integer(hid_t) :: file,dataset,space
  integer(hsize_t) :: shape(3),maximum(3)
  integer :: hdferr,last(3),node(3),point(3),i,j,k,a,b,sample,orientation,trial
  character(1024) :: path
  character(5) :: key
#ifdef TEST_CUDA
  real(real_kind),device,allocatable :: vd(:,:,:,:),tdv(:,:,:),md(:,:,:),kd(:,:,:),gd(:,:,:,:,:),jd(:,:,:),rd(:,:,:,:)
  real(real_kind) :: device_rhs(0:0,0:0,0:0,5),scaled_difference,gpu_error
  integer :: status
  type(dim3) :: block
  allocate(vd(-hm:hm,-hm:hm,-hm:hm,3),tdv(-hm:hm,-hm:hm,-hm:hm),md(-hm:hm,-hm:hm,-hm:hm), &
    kd(-hm:hm,-hm:hm,-hm:hm),gd(-hm:hm,-hm:hm,-hm:hm,3,3),jd(-hm:hm,-hm:hm,-hm:hm),rd(0:0,0:0,0:0,5))
  call prepare_me6_gpu()
  gpu_error=0
#endif
  call get_command_argument(1,path)
  if(len_trim(path)==0) error stop 'G2 mapped probe requires actual geometry.h5'
  call h5open_f(hdferr);call check_hdf()
  call h5fopen_f(trim(path),H5F_ACC_RDONLY_F,file,hdferr);call check_hdf()
  call h5dopen_f(file,'q0004',dataset,hdferr);call check_hdf()
  call h5dget_space_f(dataset,space,hdferr);call check_hdf()
  call h5sget_simple_extent_dims_f(space,shape,maximum,hdferr)
  if(hdferr<0) error stop 'G2 geometry extent read failed'
  last=int(shape)-1
  allocate(coordinates(0:last(1),0:last(2),0:last(3),3),metrics(0:last(1),0:last(2),0:last(3),3,3), &
    jac(0:last(1),0:last(2),0:last(3)))
  call h5dread_f(dataset,H5T_NATIVE_DOUBLE,jac,shape,hdferr);call check_hdf()
  call h5sclose_f(space,hdferr);call check_hdf()
  call h5dclose_f(dataset,hdferr);call check_hdf()
  do a=1,3
    write(key,'(A,I4.4)') 'q',a
    call h5dopen_f(file,key,dataset,hdferr);call check_hdf()
    call h5dread_f(dataset,H5T_NATIVE_DOUBLE,coordinates(:,:,:,a),shape,hdferr);call check_hdf()
    call h5dclose_f(dataset,hdferr);call check_hdf()
    do b=1,3
      write(key,'(A,I4.4)') 'q',4+a+3*(b-1)
      call h5dopen_f(file,key,dataset,hdferr);call check_hdf()
      call h5dread_f(dataset,H5T_NATIVE_DOUBLE,metrics(:,:,:,a,b),shape,hdferr);call check_hdf()
      call h5dclose_f(dataset,hdferr);call check_hdf()
    enddo
  enddo
  call h5fclose_f(file,hdferr);call check_hdf()
  call h5close_f(hdferr);call check_hdf()
  if(.not.all(ieee_is_finite(jac)).or.any(jac<=0.e0_real_kind).or. &
    .not.all(ieee_is_finite(metrics))) error stop 'G2 invalid saved geometry'
  error=0
  do orientation=1,2
    wave=[1.e0_real_kind,2.e0_real_kind,-1.e0_real_kind]
    if(orientation==2) wave=[-2.e0_real_kind,1.e0_real_kind,1.e0_real_kind]
    w2=sum(wave*wave)
    do sample=0,15
      node=modulo([sample,3*sample+2,5*sample+1],16)*last/16
      do k=-hm,hm
      do j=-hm,hm
      do i=-hm,hm
        point=modulo(node+[i,j,k],last)
        phase=0.13e0_real_kind+dot_product(wave,coordinates(point(1),point(2),point(3),:))
        v(i,j,k,:)=ua*sin(phase)+ub*cos(2*phase)
        t(i,j,k)=1.e0_real_kind+0.1e0_real_kind*cos(phase+0.2e0_real_kind)
        mu(i,j,k)=1.e0_real_kind+0.1e0_real_kind*cos(phase+0.3e0_real_kind)
        kappa(i,j,k)=0.7e0_real_kind+0.07e0_real_kind*sin(phase+0.4e0_real_kind)
        g(i,j,k,:,:)=metrics(point(1),point(2),point(3),:,:)
        jlocal(i,j,k)=jac(point(1),point(2),point(3))
      enddo
      enddo
      enddo
      phase=0.13e0_real_kind+dot_product(wave,coordinates(node(1),node(2),node(3),:))
      u=ua*sin(phase)+ub*cos(2*phase)
      du=ua*cos(phase)-2*ub*sin(2*phase)
      ddu=-ua*sin(phase)-4*ub*cos(2*phase)
      mut=mu(0,0,0);mup=-0.1e0_real_kind*sin(phase+0.3e0_real_kind)
      kt=kappa(0,0,0);kp=0.07e0_real_kind*cos(phase+0.4e0_real_kind)
      td=-0.1e0_real_kind*sin(phase+0.2e0_real_kind)
      tdd=-0.1e0_real_kind*cos(phase+0.2e0_real_kind)
      do b=1,3
        grad(:,b)=du*wave(b)
      enddo
      trace=dot_product(du,wave)
      stress=mut*(grad+transpose(grad))
      do a=1,3
        stress(a,a)=stress(a,a)-2*mut*trace/3.e0_real_kind
      enddo
      momentum=mut*(w2*ddu+wave*dot_product(wave,ddu)/3.e0_real_kind) &
        +mup*(w2*du+wave*dot_product(wave,du)/3.e0_real_kind)
      energy=dot_product(u,momentum)+sum(stress*grad)
      heat=w2*(kt*tdd+kp*td)
      do trial=1,2
        if(trial==2) v=0
        actual=0
        do a=1,3
          call me6_mapped_direction(a,[0,0,0],hm,v,t,mu,kappa,g,jlocal,part)
          actual=actual+part/jlocal(0,0,0)
        enddo
        if(.not.all(ieee_is_finite(actual))) error stop 'G2 nonfinite manufactured result'
#ifdef TEST_CUDA
        vd=v;tdv=t;md=mu;kd=kappa;gd=g;jd=jlocal;rd=0.e0_real_kind
        do a=1,3
          select case(a)
          case(1);block=dim3(512,1,1)
          case(2);block=dim3(32,16,1)
          case(3);block=dim3(64,1,8)
          end select
          call me6_mapped_direction_kernel<<<dim3(1,1,1),block>>>(0,0,0,hm,a,0,0,0,0,0,0,vd,tdv,md,kd,gd,jd,rd)
          status=cudaGetLastError()
          if(status/=cudaSuccess) then
            write(*,'(A,I0,A,A)') 'G2 launch status=',status,' ',trim(cudaGetErrorString(status))
            error stop 'G2 mapped kernel launch failed'
          endif
          status=cudaDeviceSynchronize()
          if(status/=cudaSuccess) error stop 'G2 mapped kernel failed'
        enddo
        device_rhs=rd
        if(.not.all(ieee_is_finite(device_rhs))) error stop 'G2 nonfinite device result'
        if(device_rhs(0,0,0,1)/=0.e0_real_kind) error stop 'G2 changed density RHS'
        device_rhs=device_rhs/jlocal(0,0,0)
        scaled_difference=maxval(abs(device_rhs(0,0,0,2:5)-actual)/ &
          max(1.e0_real_kind,abs(actual),abs(device_rhs(0,0,0,2:5))))
        gpu_error=max(gpu_error,scaled_difference)
        if(gpu_error>2.e-10_real_kind) then
          write(*,'(A,3I5,A,ES24.16)') 'G2 orientation/sample/trial=',orientation,sample,trial,' scaled=',gpu_error
          write(*,'(A,4ES24.16)') 'G2 CPU=',actual
          write(*,'(A,4ES24.16)') 'G2 GPU=',device_rhs(0,0,0,2:5)
          error stop 'G2 CPU/GPU mapped component mismatch'
        endif
        actual=device_rhs(0,0,0,2:5)
#endif
        if(trial==1) then
          error(1:3)=max(error(1:3),abs(actual(1:3)-momentum))
          full_energy=actual(4)
        else
          error(4)=max(error(4),abs(actual(4)-heat))
          error(5)=max(error(5),abs(full_energy-actual(4)-energy))
        endif
      enddo
    enddo
  enddo
  write(*,'(A,3I6,A,5ES24.16)') 'G2 actual geometry intervals=',last,' momentum/heat/work errors=',error
#ifdef TEST_CUDA
  write(*,'(A,ES24.16)') 'G2 CPU/GPU scaled difference=',gpu_error
  call release_me6_gpu()
  deallocate(vd,tdv,md,kd,gd,jd,rd)
#endif
contains
  subroutine check_hdf()
    if(hdferr/=0) error stop 'G2 geometry HDF5 read failed'
  end subroutine
end program
