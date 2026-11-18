program nscbc_filter_probe
  use iso_c_binding
  use mpi
#if defined(REFERENCE_CPU) || defined(REFERENCE_CUDA)
  use commvar, only: im,jm,km,hm
  use filter, only: spafilter6exp,filter_coefficient_explicit
#else
  use device_state
  use fluid_explicit
  use halo_transport
  use boundary_state_cpp, only: nscbc_outlet_prepare,nscbc_outlet_filter
#ifdef TEST_TOP_FILTER
  use boundary_state_cpp, only: nscbc_top_filter_prepare,nscbc_top_filter
#endif
#endif
#ifdef REFERENCE_CUDA
  use cudafor
  use commarray_gpu, only: q_d,qwork_d
  use boundary_gpu, only: nscbc_outflow_filter_y_kernel,nscbc_copy_outflow_filter_y_kernel, &
    nscbc_outflow_filter_y_physical_kernel,nscbc_copy_outflow_physical_filter_kernel,nscbc_outflow_plane_d
#ifdef TEST_TOP_FILTER
  use boundary_gpu, only: nscbc_farfield_y_upper_filter_x_kernel,nscbc_farfield_y_upper_filter_z_kernel, &
    nscbc_farfield_y_upper_filter_copy_kernel
#endif
#endif
  implicit none
  integer,parameter :: g(3)=[16,16,16],halo=5
#ifdef TEST_TOP_FILTER
  integer,parameter :: records=12,fixture_records=6
#else
  integer,parameter :: records=6,fixture_records=3
#endif
  integer :: rank,ranks,axis=1,local_rank,ierr,fixture,trial,debug,a,i,j,k,c,unit,cart,case_count=0,headers(3)
  integer(c_int) :: dims(3),upper(3),origin(3),periodic(3),lower(3),higher(3),low(3),high(3),ntype,mode
  logical :: periods(3),owner
  real(c_double),allocatable :: global_q(:,:,:,:),reference_q(:,:,:,:),q(:,:,:,:),actual(:,:,:,:), &
    primitive(:,:,:,:),metrics(:,:,:,:,:),expected(:,:,:,:)
  real(c_double) :: y,z,r,v(3),p,error,maximum=0,theta
  character(1024) :: root,path
  character(80) :: text
#if !defined(REFERENCE_CPU) && !defined(REFERENCE_CUDA)
  type(c_ptr) :: state
#endif
#ifdef REFERENCE_CUDA
  type(dim3) :: block,grid
#endif
#if !defined(REFERENCE_CPU) && !defined(REFERENCE_CUDA)
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text,status=ierr)
  if(ierr/=0) error stop 'filter probe needs matching Open MPI launcher'
  read(text,*) local_rank
  call check(device_bind(int(local_rank,c_int)))
#endif
  call MPI_Init(ierr);call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  call get_command_argument(1,root)
  allocate(global_q(-halo:g(1)+halo,-halo:g(2)+halo,-halo:g(3)+halo,5), &
    reference_q(-halo:g(1)+halo,-halo:g(2)+halo,-halo:g(3)+halo,5))
#if defined(REFERENCE_CPU) || defined(REFERENCE_CUDA)
  if(ranks/=1) error stop 'freeze filter reference at NP=1'
  im=g(1);jm=g(2);km=g(3)
  call filter_coefficient_explicit()
#ifdef REFERENCE_CUDA
  ierr=cudaSetDevice(0);if(ierr/=cudaSuccess) error stop 'reference filter GPU bind'
  allocate(q_d,mold=global_q);allocate(qwork_d,mold=global_q)
  allocate(nscbc_outflow_plane_d(0:jm,0:km,5))
  path=trim(root)//'/gpu.bin'
#else
  path=trim(root)//'/cpu.bin'
#endif
  open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='new')
  write(unit) g
  do fixture=0,1
    do trial=1,3
      call initialize(fixture,trial)
      reference_q=global_q
#ifdef REFERENCE_CUDA
      q_d=global_q;block=dim3(256,1,1);grid=dim3(((jm+1)*(km+1)*5+255)/256,1,1)
#ifdef TEST_TOP_FILTER
      grid=dim3(((im+1)*(km+1)*5+255)/256,1,1)
      call nscbc_farfield_y_upper_filter_x_kernel<<<grid,block>>>(im,jm,km,merge(3,4,fixture==0),1)
      call nscbc_farfield_y_upper_filter_copy_kernel<<<grid,block>>>(im,jm,km,1)
#else
      if(fixture==0) then
        call nscbc_outflow_filter_y_kernel<<<grid,block>>>(im,jm,km,halo,1)
        call nscbc_copy_outflow_filter_y_kernel<<<grid,block>>>(im,jm,km,1)
      else
        call nscbc_outflow_filter_y_physical_kernel<<<grid,block>>>(im,jm,km,halo,4,1)
        call nscbc_copy_outflow_physical_filter_kernel<<<grid,block>>>(im,jm,km,1)
      endif
#endif
      ierr=cudaGetLastError();if(ierr/=cudaSuccess) error stop 'reference filter launch'
      ierr=cudaDeviceSynchronize();if(ierr/=cudaSuccess) error stop 'reference filter execution'
      reference_q=q_d
#else
#ifdef TEST_TOP_FILTER
      do c=1,5
      do k=0,km
        reference_q(0:im,jm,k,c)=spafilter6exp(global_q(:,jm,k,c),merge(3,4,fixture==0),im)
      enddo
      enddo
#else
      do c=1,5
      do k=0,km
        reference_q(im,0:jm,k,c)=spafilter6exp(global_q(im,:,k,c),merge(3,4,fixture==0),jm)
        if(fixture==1) reference_q(im,0,k,c)=global_q(im,0,k,c)
      enddo
      enddo
#endif
#endif
      write(unit) global_q,reference_q
#ifdef TEST_TOP_FILTER
      ! Independently prepare the second filter's periodic z donors from the
      ! x-filtered face. The new backend must obtain these through real halo.
      global_q=reference_q
      do k=1,halo
        global_q(:,jm,-k,:)=global_q(:,jm,km-k,:)
        global_q(:,jm,km+k,:)=global_q(:,jm,k,:)
      enddo
      reference_q=global_q
#ifdef REFERENCE_CUDA
      q_d=global_q
      call nscbc_farfield_y_upper_filter_z_kernel<<<grid,block>>>(im,jm,km,3,1)
      call nscbc_farfield_y_upper_filter_copy_kernel<<<grid,block>>>(im,jm,km,1)
      ierr=cudaGetLastError();if(ierr/=cudaSuccess) error stop 'top z reference filter launch'
      ierr=cudaDeviceSynchronize();if(ierr/=cudaSuccess) error stop 'top z reference filter execution'
      reference_q=q_d
#else
      do c=1,5
      do i=0,im
        reference_q(i,jm,0:km,c)=spafilter6exp(global_q(i,jm,:,c),3,km)
      enddo
      enddo
#endif
      write(unit) global_q,reference_q
#endif
    enddo
  enddo
  close(unit)
  print *, 'NSCBC_FILTER_FROZEN ',rank,records
#else
  call get_command_argument(2,text);read(text,*) axis
  call get_command_argument(3,text);mode=halo_pinned
  if(trim(text)=='device') mode=halo_device
  if((ranks/=1.and.ranks/=2).or.axis<1.or.axis>3) error stop 'filter probe topology'
  dims=1;dims(axis)=ranks;upper=g;upper(axis)=g(axis)/ranks;origin=0;origin(axis)=rank*upper(axis)
#ifdef TEST_TOP_FILTER
  owner=origin(2)+upper(2)==g(2)
#else
  owner=origin(1)+upper(1)==g(1)
#endif
  allocate(q(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,5), &
    actual(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,5), &
    expected(0:upper(1),0:upper(2),0:upper(3),5), &
    primitive(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,6), &
    metrics(-halo:upper(1)+halo,-halo:upper(2)+halo,-halo:upper(3)+halo,3,3))
  metrics=0
  do c=1,3
    metrics(:,:,:,c,c)=1
  enddo
  do fixture=0,1
#ifdef TEST_TOP_FILTER
    periodic=[1,0,1]
    if(fixture==1) periodic(1)=0
#else
    periodic=[0,1,1]
    if(fixture==1) periodic(2)=0
#endif
    low=merge(1,0,periodic==0.and.origin==0)
    high=merge(1,0,periodic==0.and.origin+upper==g)
    periods=periodic/=0
    call MPI_Cart_create(MPI_COMM_WORLD,3,dims,periods,.false.,cart,ierr);call check(ierr)
    do a=1,3
      call MPI_Cart_shift(cart,a-1,1,lower(a),higher(a),ierr);call check(ierr)
    enddo
    ntype=3
    if(fixture==1) then
#ifdef TEST_TOP_FILTER
      if(origin(1)==0) ntype=1
      if(origin(1)+upper(1)==g(1)) ntype=2
      if(origin(1)==0.and.upper(1)==g(1)) ntype=4
#else
      if(origin(2)==0) ntype=1
      if(origin(2)+upper(2)==g(2)) ntype=2
      if(origin(2)==0.and.upper(2)==g(2)) ntype=4
#endif
    endif
    do debug=0,1
      call check(device_state_create(int(local_rank,c_int),upper(1),upper(2),upper(3),halo,int(debug,c_int),state))
      call check(fluid_prepare(state,0_c_int))
      call check(fluid_initialize_exterior(state,low,high))
      call check(device_state_upload(state,device_metric,metrics,bytes(size(metrics))))
      call check(device_transport_prepare(state,int(cart,c_int),lower,higher,dims,5_c_int,mode,0_c_int,periodic))
      if(owner) then
#ifdef TEST_TOP_FILTER
        call check(nscbc_top_filter_prepare(state));call check(nscbc_top_filter_prepare(state))
#else
        call check(nscbc_outlet_prepare(state));call check(nscbc_outlet_prepare(state))
#endif
      endif
      do c=1,2
        path=trim(root)//'/cpu.bin'
        if(c==2) path=trim(root)//'/gpu.bin'
        open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='old')
        read(unit) headers
        if(any(headers/=g)) error stop 'filter reference shape'
        do trial=1,records
          read(unit) global_q,reference_q
          if((fixture==0.and.trial>fixture_records).or.(fixture==1.and.trial<=fixture_records)) cycle
          q=global_q(origin(1)-halo:origin(1)+upper(1)+halo, &
            origin(2)-halo:origin(2)+upper(2)+halo,origin(3)-halo:origin(3)+upper(3)+halo,:)
          expected=reference_q(origin(1):origin(1)+upper(1),origin(2):origin(2)+upper(2), &
            origin(3):origin(3)+upper(3),:)
          ! Poison the exchanged outlet halo to prove that the filter consumes
          ! the production transport, not the host's precomputed halo values.
#ifdef TEST_TOP_FILTER
          if(modulo(trial-1,2)==0) then
#endif
          call poison_face_halos()
          call check(device_state_upload(state,device_q,q,bytes(size(q))))
          call exchange_all()
#ifdef TEST_TOP_FILTER
          endif
          if(owner) then
            if(modulo(trial-1,2)==0) then
              call check(nscbc_top_filter(state,0_c_int,ntype))
            else
              call check(nscbc_top_filter(state,2_c_int,3_c_int))
            endif
          endif
#else
          if(owner) call check(nscbc_outlet_filter(state,ntype))
#endif
          call exchange_all()
          call check(device_state_download(state,device_q,actual,bytes(size(actual))))
#ifdef TEST_TOP_FILTER
          do j=0,upper(2)
            if(origin(2)+j==g(2)) cycle
            if(any(transfer(actual(0:upper(1),j,0:upper(3),:),[0_c_int64_t],(upper(1)+1)*(upper(3)+1)*5)/= &
              transfer(q(0:upper(1),j,0:upper(3),:),[0_c_int64_t],(upper(1)+1)*(upper(3)+1)*5))) &
              error stop 'top filter changed nontop state'
          enddo
#else
          do i=0,upper(1)
            if(origin(1)+i==g(1)) cycle
            if(any(transfer(actual(i,0:upper(2),0:upper(3),:),[0_c_int64_t],(upper(2)+1)*(upper(3)+1)*5)/= &
              transfer(q(i,0:upper(2),0:upper(3),:),[0_c_int64_t],(upper(2)+1)*(upper(3)+1)*5))) &
              error stop 'outlet filter changed nonoutlet state'
          enddo
#endif
          do a=1,5
            error=maxval(abs(actual(0:upper(1),0:upper(2),0:upper(3),a)-expected(:,:,:,a)))/ &
              max(1.d0,maxval(abs(expected(:,:,:,a))))
            if(.not.(error<=2.d-10)) then
              print *, 'NSCBC_FILTER_MISMATCH ',rank,fixture,trial,c,a,error
              error stop 'filter state differs from original'
            endif
            maximum=max(maximum,error)
          enddo
          call check(fluid_refresh_primitives_boundary(state,1.4d0,2.5d0,low,high))
          call check(device_state_download(state,device_rho,primitive(:,:,:,1),bytes(size(primitive(:,:,:,1)))))
          call check(device_state_download(state,device_pressure,primitive(:,:,:,2),bytes(size(primitive(:,:,:,2)))))
          call check(device_state_download(state,device_temperature,primitive(:,:,:,3),bytes(size(primitive(:,:,:,3)))))
          call check(device_state_download(state,device_velocity,primitive(:,:,:,4:6),bytes(size(primitive(:,:,:,4:6)))))
          do k=-halo,upper(3)+halo
          do j=-halo,upper(2)+halo
          do i=-halo,upper(1)+halo
            if(count([i,j,k]<0.or.[i,j,k]>upper)>1) cycle
            if(any(periodic==0.and.(origin+[i,j,k]<0.or.origin+[i,j,k]>g))) cycle
            r=actual(i,j,k,1);p=(actual(i,j,k,5)-.5d0*sum(actual(i,j,k,2:4)**2)/r)/2.5d0
            error=max(abs(primitive(i,j,k,1)-r),abs(primitive(i,j,k,2)-p),abs(primitive(i,j,k,3)-1.4d0*p/r), &
              maxval(abs(primitive(i,j,k,4:6)-actual(i,j,k,2:4)/r)))
            if(.not.(error<=2.d-10)) error stop 'filter primitive cache phase'
            maximum=max(maximum,error)
          enddo
          enddo
          enddo
          case_count=case_count+1
        enddo
        close(unit)
      enddo
      call check(device_state_destroy(state))
    enddo
    call MPI_Comm_free(cart,ierr);call check(ierr)
  enddo
#ifdef TEST_TOP_FILTER
  print *, 'CPP_NSCBC_TOP_FILTER_PASS ',rank,case_count,maximum
#else
  print *, 'CPP_NSCBC_FILTER_PASS ',rank,case_count,maximum
#endif
#endif
  call MPI_Finalize(ierr)
contains
  subroutine initialize(boundary,number)
    integer,intent(in) :: boundary,number
    do k=-halo,g(3)+halo
    do j=-halo,g(2)+halo
    do i=-halo,g(1)+halo
#ifdef TEST_TOP_FILTER
      y=real(i,c_double)/g(1)
      if(boundary==0) y=real(modulo(i,g(1)),c_double)/g(1)
#else
      y=real(j,c_double)/g(2)
      if(boundary==0) y=real(modulo(j,g(2)),c_double)/g(2)
#endif
      z=real(modulo(k,g(3)),c_double)/g(3)
      theta=2*acos(-1.d0)
      r=1+.03d0*sin(theta*y)+.02d0*cos(3*theta*y)*cos(theta*z)
      v=[.4d0+.01d0*sin(2*theta*y),.02d0*cos(theta*y),.01d0*sin(theta*z)]
      p=1.d0/1.4d0+.01d0*cos(2*theta*y)
      if(number==1) then
        r=1;v=[.4d0,.02d0,.01d0];p=1.d0/1.4d0
      elseif(number==3) then
        r=1+.01d0*y**4+.02d0*cos(theta*z)
        if(boundary==0) r=1+.01d0*cos(7*theta*y)+.02d0*cos(theta*z)
      endif
      global_q(i,j,k,1)=r;global_q(i,j,k,2:4)=r*v
      global_q(i,j,k,5)=p/0.4d0+.5d0*r*sum(v*v)
    enddo
    enddo
    enddo
  end subroutine
#if !defined(REFERENCE_CPU) && !defined(REFERENCE_CUDA)
  subroutine poison_face_halos()
    integer :: direction,b,d,side,layer,ii,jj,point(3)
    do direction=1,3
      b=modulo(direction,3)+1;d=modulo(direction+1,3)+1
      do side=0,1
        if(periodic(direction)==0) then
          if(side==0.and.lower(direction)==MPI_PROC_NULL) cycle
          if(side==1.and.higher(direction)==MPI_PROC_NULL) cycle
        endif
        do layer=1,halo
        do jj=0,upper(d)
        do ii=0,upper(b)
          point=0;point(b)=ii;point(d)=jj
          point(direction)=merge(-layer,upper(direction)+layer,side==0)
          q(point(1),point(2),point(3),:)=-777.d0
        enddo
        enddo
        enddo
      enddo
    enddo
  end subroutine
  subroutine exchange_all()
    integer :: direction
    do direction=0,2
      call check(device_exchange_axis(state,device_q,int(direction,c_int),1_c_int))
    enddo
    do direction=0,2
      call check(device_exchange_axis(state,device_q,int(direction,c_int),0_c_int))
    enddo
  end subroutine
  subroutine check(code)
    integer(c_int),intent(in) :: code
    if(code/=0) then
      print *, 'NSCBC_FILTER_DEVICE_STATUS ',code
      error stop 'filter device command'
    endif
  end subroutine
  integer(c_size_t) function bytes(count)
    integer,intent(in) :: count
    bytes=int(count,c_size_t)*c_sizeof(0.d0)
  end function
#endif
end program
