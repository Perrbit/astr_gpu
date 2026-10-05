module insitu_fields
  use mpi
  use iso_fortran_env, only: real64
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: capture_sample,canonicalize_sample,derive_sample
  public :: complete_periodic_endpoints
  public :: capture_canonical_velocity
  public :: capture_index_plane
  public :: capture_channel_walls
  public :: insitu_owned_counts,insitu_node_weights,complete_owned_endpoints
  public :: capture_channel_statistics
  public :: capture_air5_walls
  public :: insitu_volume_weights,release_insitu_geometry
  public :: nonreacting_wall_candidate
  real(real64),allocatable,save :: curve_volume_weights(:,:,:)
contains
  logical function nonreacting_wall_candidate() result(supported)
    use commvar, only: flowtype,lreadgrid
    use bc, only: bctype
    supported=trim(flowtype)=='channel'.or.(trim(flowtype)=='tgv'.and.lreadgrid.and. &
      all(bctype==[1,1,41,41,1,1]))
  end function

  subroutine release_insitu_geometry()
    if(allocated(curve_volume_weights)) deallocate(curve_volume_weights)
  end subroutine

  subroutine insitu_volume_weights(weights)
    use commvar, only: im,jm,km,ia,ja,ka,lreadgrid
    use commarray, only: x
    use bc, only: bctype
    use parallel, only: ig0,jg0,kg0,mpileft,mpiright,mpidown,mpiup,mpiback,mpifront,mpirank
    use insitu_geometry, only: hex_volume
    real(real64),intent(out) :: weights(0:im,0:jm,0:km)
    real(real64) :: points(3,0:1,0:1,0:1),volume,local,total
    real(real64),allocatable :: send(:),receive(:)
    integer :: i,j,k,a,b,c,axis,neighbors(2,3),count,ierr,comm,status,origin(3),cells(3),global(3)
    logical :: ok,valid
    call require_fields(lreadgrid,'CURVE weights require an explicit physical grid')
    if(.not.allocated(curve_volume_weights)) then
      allocate(curve_volume_weights(0:im,0:jm,0:km), &
        send(max((im+1)*(jm+1),(im+1)*(km+1),(jm+1)*(km+1))), &
        receive(max((im+1)*(jm+1),(im+1)*(km+1),(jm+1)*(km+1))),stat=status)
      call require_fields(status==0,'cannot allocate CURVE geometric weights')
      curve_volume_weights=0.d0; valid=.true.
      cells_loop: do k=0,km-1
      do j=0,jm-1
      do i=0,im-1
        do c=0,1
        do b=0,1
        do a=0,1
          points(:,a,b,c)=x(i+a,j+b,k+c,1:3)
        enddo
        enddo
        enddo
        call hex_volume(points,volume,ok)
        if(.not.ok) then
          valid=.false.; exit cells_loop
        endif
        curve_volume_weights(i:i+1,j:j+1,k:k+1)=curve_volume_weights(i:i+1,j:j+1,k:k+1)+volume/8.d0
      enddo
      enddo
      enddo cells_loop
      call require_fields(valid,'CURVE cell has nonpositive or invalid Gauss Jacobian')
      origin=[ig0,jg0,kg0]; cells=[im,jm,km]; global=[ia,ja,ka]
      neighbors(:,1)=[mpileft,mpiright]; neighbors(:,2)=[mpidown,mpiup]; neighbors(:,3)=[mpiback,mpifront]
      call MPI_Comm_dup(MPI_COMM_WORLD,comm,ierr)
      call require_fields(ierr==MPI_SUCCESS,'cannot create geometric weight communicator')
      ! Send upper-node contributions to the next owner's lower plane; axis order carries corners.
      do axis=1,3
        if(cells(axis)==global(axis).and.all(bctype(2*axis-1:2*axis)==1)) &
          neighbors(:,axis)=mpirank
        if(origin(axis)==0.and.bctype(2*axis-1)/=1) neighbors(1,axis)=MPI_PROC_NULL
        if(origin(axis)+cells(axis)==global(axis).and.bctype(2*axis)/=1) neighbors(2,axis)=MPI_PROC_NULL
        select case(axis)
        case(1)
          count=(jm+1)*(km+1); send(1:count)=reshape(curve_volume_weights(im,:,:),[count])
        case(2)
          count=(im+1)*(km+1); send(1:count)=reshape(curve_volume_weights(:,jm,:),[count])
        case(3)
          count=(im+1)*(jm+1); send(1:count)=reshape(curve_volume_weights(:,:,km),[count])
        end select
        receive(1:count)=0.d0
        call MPI_Sendrecv(send,count,MPI_DOUBLE_PRECISION,neighbors(2,axis),axis, &
          receive,count,MPI_DOUBLE_PRECISION,neighbors(1,axis),axis,comm,MPI_STATUS_IGNORE,ierr)
        call require_fields(ierr==MPI_SUCCESS,'geometric weight contribution exchange failed')
        select case(axis)
        case(1)
          curve_volume_weights(0,:,:)=curve_volume_weights(0,:,:)+reshape(receive(1:count),[jm+1,km+1])
          if(neighbors(2,axis)/=MPI_PROC_NULL) curve_volume_weights(im,:,:)=0.d0
        case(2)
          curve_volume_weights(:,0,:)=curve_volume_weights(:,0,:)+reshape(receive(1:count),[im+1,km+1])
          if(neighbors(2,axis)/=MPI_PROC_NULL) curve_volume_weights(:,jm,:)=0.d0
        case(3)
          curve_volume_weights(:,:,0)=curve_volume_weights(:,:,0)+reshape(receive(1:count),[im+1,jm+1])
          if(neighbors(2,axis)/=MPI_PROC_NULL) curve_volume_weights(:,:,km)=0.d0
        end select
      enddo
      call MPI_Comm_free(comm,ierr)
      call require_fields(ierr==MPI_SUCCESS,'cannot release geometric weight communicator')
      call require_fields(all(ieee_is_finite(curve_volume_weights)).and.all(curve_volume_weights>=0.d0), &
        'invalid assembled physical geometric weights')
      local=sum(curve_volume_weights)
      call MPI_Allreduce(local,total,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr)
      call require_fields(ierr==MPI_SUCCESS.and.ieee_is_finite(total).and.total>0.d0,'invalid geometric volume')
      if(mpirank==0) write(*,'(a,es24.16)') 'ASTR_INSITU_CURVE_VOLUME=',total
    endif
    weights=curve_volume_weights
  end subroutine

  subroutine capture_air5_walls(coordinates,fields,owned,host_budget,device_budget,reserve,download_bytes)
    use iso_fortran_env, only: int64
    use commvar, only: im,jm,km,ia,ja,ka,numq,num_species,num_modequ,use_gpu,flowtype,lreadgrid, &
      nondimen,difschm,lcomb
    use commarray, only: q,x,dxi
    use parallel, only: ig0,jg0,mpisize,isize,ksize,mpileft,mpiright,mpiback,mpifront
    use bc, only: bctype
#ifdef ASTR_AIR5_CHEMISTRY
    use chemistry_monitor, only: decode_air5_monitor_state
    use chemistry_transport, only: air5_transport_properties
#endif
#ifdef _CUDA
    use insitu_sample_gpu, only: download_wall_state_gpu
#endif
    real(real64),allocatable,intent(out) :: coordinates(:,:,:,:),fields(:,:,:,:)
    logical,allocatable,intent(out) :: owned(:,:,:)
    integer(int64),intent(in) :: host_budget,device_budget,reserve
    integer(int64),intent(out) :: download_bytes
#ifdef ASTR_AIR5_CHEMISTRY
    real(real64),allocatable :: state(:,:,:,:),send(:),receive(:)
    real(real64) :: primitive(12,3),mu,kt,kv,smu(5),binary(5,5),diffusion(5),metric,derivative(3)
    integer :: nw,i,k,m,c,status,local_status,ierr,comm,count,axis,sizes(2),neighbors(2,2)
    integer(int64) :: nodes,bytes
    call require_fields(trim(flowtype)=='air5hbl'.and.numq==11.and.num_species==5.and.num_modequ==1.and. &
      .not.nondimen.and.lcomb.and..not.lreadgrid.and.all(bctype==[11,50,41,51,1,1]).and. &
      trim(difschm)=='643e'.and.max(ia,ja,ka)<=32.and.mpisize<=2.and.jm>=2, &
      'AIR5 wall candidate requires dimensional Cartesian HBL <=32 NP=1/2 and two interior layers')
    nw=merge(1,0,jg0==0)
    nodes=int(im+1,int64)*(km+1)*nw
    bytes=nodes*(54*8+storage_size(.false.)/8)+int(max(im+1,km+1),int64)*nw*33*8*4
    call require_fields(bytes<=host_budget,'AIR5 wall host budget')
    allocate(state(0:im,0:km,nw,33),coordinates(0:im,0:km,nw,3),fields(0:im,0:km,nw,18), &
      owned(0:im,0:km,nw),send(max(1,max(im+1,km+1)*nw*33)), &
      receive(max(1,max(im+1,km+1)*nw*33)),stat=status)
    call require_fields(status==0,'cannot allocate private AIR5 wall buffers')
    state=0.d0; fields=0.d0; coordinates=0.d0; download_bytes=0
#ifdef _CUDA
    if(use_gpu) then
      call download_wall_state_gpu([(0,m=1,nw)],state,device_budget,reserve)
      download_bytes=nodes*33*8
    else
#else
    call require_fields(.not.use_gpu,'AIR5 GPU capture requires CUDA build')
#endif
      if(nw>0) then
        do m=0,2
          state(:,:,1,11*m+1:11*m+11)=q(0:im,m,0:km,1:11)
        enddo
      endif
#ifdef _CUDA
    endif
#endif
    neighbors(:,1)=[mpileft,mpiright]; neighbors(:,2)=[mpiback,mpifront]; sizes=[isize,ksize]
    call MPI_Comm_dup(MPI_COMM_WORLD,comm,ierr)
    call require_fields(ierr==MPI_SUCCESS,'cannot create AIR5 wall endpoint communicator')
    do axis=1,2
      ierr=MPI_SUCCESS
      if(nw>0.and..not.(axis==1.and.sizes(axis)==1)) then
        if(axis==1) then
          count=(km+1)*33; send(1:count)=reshape(state(0,:,:,:),[count])
        else
          count=(im+1)*33; send(1:count)=reshape(state(:,0,:,:),[count])
        endif
        if(sizes(axis)==1) then
          receive(1:count)=send(1:count)
        else
          call MPI_Sendrecv(send,count,MPI_DOUBLE_PRECISION,neighbors(1,axis),axis,receive,count, &
            MPI_DOUBLE_PRECISION,neighbors(2,axis),axis,comm,MPI_STATUS_IGNORE,ierr)
        endif
        if(sizes(axis)==1.or.neighbors(2,axis)/=MPI_PROC_NULL) then
          if(axis==1) then
            state(im,:,:,:)=reshape(receive(1:count),[km+1,nw,33])
          else
            state(:,km,:,:)=reshape(receive(1:count),[im+1,nw,33])
          endif
        endif
      endif
      call require_fields(ierr==MPI_SUCCESS,'AIR5 wall endpoint exchange failed')
    enddo
    call MPI_Comm_free(comm,ierr)
    call require_fields(ierr==MPI_SUCCESS,'cannot release AIR5 wall endpoint communicator')
    owned=.false.
    if(nw>0) then
      owned(0:im-1,0:km-1,:)=.true.
      if(ig0+im==ia) owned(im,0:km-1,:)=.true.
    endif
    local_status=0
    do k=0,km
    do i=0,im
      if(nw==0) cycle
      do m=1,3
        c=11*(m-1)
        call decode_air5_monitor_state(state(i,k,1,c+1:c+11),primitive(:,m),status)
        local_status=max(local_status,status)
      enddo
      if(local_status/=0) cycle
      call air5_transport_properties(primitive(5,1),primitive(6,1),primitive(7,1),primitive(8:12,1), &
        mu,kt,kv,smu,binary,diffusion,status)
      local_status=max(local_status,status)
      if(local_status/=0) cycle
      metric=dxi(i,0,k,2,2)
      if(.not.ieee_is_finite(metric).or.metric<=0.d0) then
        local_status=1; cycle
      endif
      derivative=(-0.5d0*primitive([2,5,6],3)+2.d0*primitive([2,5,6],2)- &
        1.5d0*primitive([2,5,6],1))*metric
      coordinates(i,k,1,:)=x(i,0,k,:)
      ! Static noncatalytic wall: J_s dot n=0 and viscous work is zero.
      fields(i,k,1,:)=[primitive(:,1),mu*derivative(1),-kt*derivative(2),-kv*derivative(3), &
        0.d0,-kt*derivative(2)-kv*derivative(3),1.d0]
    enddo
    enddo
    call require_fields(local_status==0.and.all(ieee_is_finite(fields)).and. &
      all(ieee_is_finite(coordinates)),'invalid completed AIR5 wall fields')
#else
    download_bytes=0
    call require_fields(.false.,'AIR5 wall capture requires ASTR_WITH_AIR5_CHEMISTRY')
#endif
  end subroutine

  function insitu_owned_counts() result(extent)
    use commvar, only: im,jm,km,ja,flowtype
    use parallel, only: jg0
    integer :: extent(3)
    extent=[im,jm,km]
    if(nonreacting_wall_candidate().and.jg0+jm==ja) extent(2)=jm+1
  end function

  subroutine insitu_node_weights(weights)
    use commvar, only: im,jm,km,flowtype,ja
    use commarray, only: x
    use parallel, only: jg0,mpidown,mpiup
    real(real64),intent(out) :: weights(0:jm)
    real(real64) :: area,left,right,send,receive
    integer :: j,ierr,comm
    area=(x(1,0,0,1)-x(0,0,0,1))*(x(0,0,1,3)-x(0,0,0,3))
    weights=area*(x(0,1,0,2)-x(0,0,0,2))
    if(trim(flowtype)=='channel') then
      ! Exchange the preceding physical coordinate, not a solver metric halo.
      send=x(0,jm-1,0,2); receive=x(0,0,0,2)
      call MPI_Comm_dup(MPI_COMM_WORLD,comm,ierr)
      call require_fields(ierr==MPI_SUCCESS,'cannot create statistics coordinate communicator')
      call MPI_Sendrecv(send,1,MPI_DOUBLE_PRECISION,mpiup,1,receive,1,MPI_DOUBLE_PRECISION, &
        mpidown,1,comm,MPI_STATUS_IGNORE,ierr)
      call require_fields(ierr==MPI_SUCCESS,'statistics coordinate exchange failed')
      call MPI_Comm_free(comm,ierr)
      call require_fields(ierr==MPI_SUCCESS,'cannot release statistics coordinate communicator')
      do j=0,jm
        left=x(0,j,0,2); right=left
        if(j>0) left=x(0,j-1,0,2)
        if(j==0.and.jg0>0) left=receive
        if(j<jm) right=x(0,j+1,0,2)
        if(j==jm.and.jg0+jm/=ja) then
          ! This duplicate interface is not owned and has zero integration weight.
          weights(j)=0.d0
        else
          weights(j)=area*0.5d0*(right-left)
        endif
      enddo
    endif
    call require_fields(all(ieee_is_finite(weights)).and.all(weights>=0.d0).and.area>0.d0, &
      'invalid physical statistics node weights')
  end subroutine

  subroutine capture_channel_statistics(fields)
    use commvar, only: im,jm,km,use_gpu,numq,num_species,flowtype
    use commarray, only: q
    real(real64),intent(out) :: fields(0:im,0:jm,0:km,11)
    call require_fields(.not.use_gpu.and.nonreacting_wall_candidate().and.numq==5.and.num_species==0, &
      'channel statistics host capture requires the CPU five-variable solver')
    fields=0.d0
    fields(:,:,:,1:5)=q(0:im,0:jm,0:km,1:5)
    call complete_owned_endpoints(fields(:,:,:,1:5))
    fields(:,:,:,6)=fields(:,:,:,1)
    call require_fields(all(ieee_is_finite(fields)).and.all(fields(:,:,:,6)>0.d0), &
      'invalid completed channel statistics state')
    fields(:,:,:,7)=fields(:,:,:,2)/fields(:,:,:,6)
    fields(:,:,:,8)=fields(:,:,:,3)/fields(:,:,:,6)
    fields(:,:,:,9)=fields(:,:,:,4)/fields(:,:,:,6)
    call require_fields(all(ieee_is_finite(fields)),'invalid channel statistics velocity')
  end subroutine

  subroutine capture_channel_walls(coordinates,fields,owned,host_budget,device_budget,reserve,download_bytes)
    use iso_fortran_env, only: int64
    use commvar, only: im,jm,km,ia,ja,ka,numq,num_species,use_gpu,flowtype,lreadgrid,nondimen, &
      difschm,const2,const5,const6,reynolds,prandtl
    use commarray, only: q,x,dxi
    use parallel, only: jg0,isize,ksize,mpileft,mpiright,mpiback,mpifront,mpisize
    use bc, only: bctype
    use fludyna, only: miucal
    use insitu_geometry, only: wall_frame
#ifdef _CUDA
    use insitu_sample_gpu, only: download_wall_state_gpu
#endif
    real(real64),allocatable,intent(out) :: coordinates(:,:,:,:),fields(:,:,:,:)
    logical,allocatable,intent(out) :: owned(:,:,:)
    integer(int64),intent(in) :: host_budget,device_budget,reserve
    integer(int64),intent(out) :: download_bytes
    real(real64),allocatable :: state(:,:,:,:),send(:),receive(:)
    real(real64),allocatable :: wall_primitive(:,:,:,:)
    real(real64) :: velocity(3,3),temperature(3),pressure(3),rho,mu,metric,du,dt
    real(real64) :: first(3),second(3),inward(3),normal(3),tangent(3),gradient(3,3),stress(3,3),dv(3)
    real(real64) :: directional(3,3),thermal(3)
    real(real64),parameter :: coefficients(3)=[0.75d0,-0.15d0,1.d0/60.d0]
    logical :: frame_ok
    integer :: offset
    integer :: walls(2),nw,i,k,w,m,j,side,d,c,axis,count,status,ierr,comm
    integer :: neighbors(2,2),sizes(2)
    integer(int64) :: nodes,bytes
    call require_fields(nonreacting_wall_candidate().and.numq==5.and.num_species==0.and.nondimen.and. &
      all(bctype==[1,1,41,41,1,1]).and.trim(difschm)=='643e'.and. &
      max(ia,ja,ka)<=32.and.min(im,km)>=1.and.jm>=2.and.mpisize<=2, &
      'wall candidate requires bc41 channel or CURVE TGV <=32, NP=1/2 and two local interior layers')
    call require_fields(all(ieee_is_finite([const2,const5,const6,reynolds,prandtl])).and. &
      min(const2,const5,const6,reynolds,prandtl)>0.d0,'invalid wall material constants')
    nw=0
    if(jg0==0) then
      nw=nw+1; walls(nw)=0
    endif
    if(jg0+jm==ja) then
      nw=nw+1; walls(nw)=jm
    endif
    nodes=int(im+1,int64)*(km+1)*nw
    ! Includes endpoint packing/reshape temporaries as well as retained outputs.
    bytes=nodes*(22*8+storage_size(.false.)/8)+int(max(im+1,km+1),int64)*nw*15*8*4
    if(lreadgrid) bytes=bytes+int(im+7,int64)*(km+7)*nw*4*8+ &
      int(max(im+1,km+1),int64)*nw*4*3*8*4
    call require_fields(bytes<=host_budget,'wall host budget')
    allocate(state(0:im,0:km,nw,15),coordinates(0:im,0:km,nw,3),fields(0:im,0:km,nw,4), &
      owned(0:im,0:km,nw),send(max(1,max(im+1,km+1)*nw*15)), &
      receive(max(1,max(im+1,km+1)*nw*15)),stat=status)
    call require_fields(status==0,'cannot allocate private wall buffers')
    fields=0.d0; coordinates=0.d0
    download_bytes=0
#ifdef _CUDA
    if(use_gpu) then
      call download_wall_state_gpu(walls(1:nw),state,device_budget,reserve)
      download_bytes=nodes*15*8
    else
#else
    call require_fields(.not.use_gpu,'wall GPU capture requires CUDA build')
#endif
      do w=1,nw
        side=1; if(walls(w)==jm) side=-1
        do m=0,2
          j=walls(w)+side*m
          state(:,:,w,5*m+1:5*m+5)=q(0:im,j,0:km,1:5)
        enddo
      enddo
#ifdef _CUDA
    endif
#endif
    neighbors(:,1)=[mpileft,mpiright]; neighbors(:,2)=[mpiback,mpifront]
    sizes=[isize,ksize]
    call MPI_Comm_dup(MPI_COMM_WORLD,comm,ierr)
    call require_fields(ierr==MPI_SUCCESS,'cannot create wall endpoint communicator')
    do axis=1,2
      count=0
      if(nw>0) then
        if(axis==1) then
          count=(km+1)*nw*15; send(1:count)=reshape(state(0,:,:,:),[count])
        else
          count=(im+1)*nw*15; send(1:count)=reshape(state(:,0,:,:),[count])
        endif
      endif
      ierr=MPI_SUCCESS
      if(count>0) then
        if(sizes(axis)==1) then
          receive(1:count)=send(1:count)
        else
          call MPI_Sendrecv(send,count,MPI_DOUBLE_PRECISION,neighbors(1,axis),axis, &
            receive,count,MPI_DOUBLE_PRECISION,neighbors(2,axis),axis,comm,MPI_STATUS_IGNORE,ierr)
        endif
        if(axis==1) then
          state(im,:,:,:)=reshape(receive(1:count),[km+1,nw,15])
        else
          state(:,km,:,:)=reshape(receive(1:count),[im+1,nw,15])
        endif
      endif
      call require_fields(ierr==MPI_SUCCESS,'wall periodic endpoint exchange failed')
    enddo
    call MPI_Comm_free(comm,ierr)
    call require_fields(ierr==MPI_SUCCESS,'cannot release wall endpoint communicator')
    call require_fields(all(ieee_is_finite(state)),'nonfinite wall state')
    if(lreadgrid) call prepare_curve_wall_primitive(state,wall_primitive)
    owned=.false.
    owned(0:im-1,0:km-1,:)=.true.
    status=0
    do w=1,nw
      side=1; if(walls(w)==jm) side=-1
      j=walls(w)
      do k=0,km
      do i=0,im
        do m=1,3
          c=5*(m-1)
          rho=state(i,k,w,c+1)
          if(rho<=0.d0) then
            status=1; cycle
          endif
          velocity(:,m)=state(i,k,w,c+2:c+4)/rho
          pressure(m)=(state(i,k,w,c+5)-0.5d0*rho*sum(velocity(:,m)**2))/const6
          temperature(m)=pressure(m)/rho*const2
        enddo
        if(status/=0) cycle
        metric=dxi(i,j,k,2,2)
        if(.not.all(ieee_is_finite(temperature)).or.minval(temperature)<=0.d0.or. &
          .not.ieee_is_finite(metric).or.metric<=0.d0) then
          status=1; cycle
        endif
        mu=miucal(temperature(1))/reynolds
        du=(-0.5d0*velocity(1,3)+2.d0*velocity(1,2)-1.5d0*velocity(1,1))*metric
        dt=(-0.5d0*temperature(3)+2.d0*temperature(2)-1.5d0*temperature(1))*metric
        coordinates(i,k,w,:)=x(i,j,k,1:3)
        fields(i,k,w,:)=[pressure(1),mu*du,-((mu/prandtl)/const5)*dt,real(side,real64)]
        if(lreadgrid) then
          first=0.d0; second=0.d0
          do offset=1,3
            first=first+coefficients(offset)*(x(i+offset,j,k,1:3)-x(i-offset,j,k,1:3))
            second=second+coefficients(offset)*(x(i,j,k+offset,1:3)-x(i,j,k-offset,1:3))
          enddo
          inward=x(i,j+side,k,1:3)-x(i,j,k,1:3)
          call wall_frame(first,second,inward,normal,tangent,frame_ok)
          if(.not.frame_ok.or..not.all(ieee_is_finite(dxi(i,j,k,2,:)))) then
            status=1; cycle
          endif
          dv=real(side,real64)*(-0.5d0*velocity(:,3)+2.d0*velocity(:,2)-1.5d0*velocity(:,1))
          directional=0.d0; directional(:,2)=dv; thermal=0.d0
          thermal(2)=real(side,real64)*(-0.5d0*temperature(3)+2.d0*temperature(2)-1.5d0*temperature(1))
          do offset=1,3
            directional(:,1)=directional(:,1)+coefficients(offset)*( &
              wall_primitive(i+offset,k,w,1:3)-wall_primitive(i-offset,k,w,1:3))
            directional(:,3)=directional(:,3)+coefficients(offset)*( &
              wall_primitive(i,k+offset,w,1:3)-wall_primitive(i,k-offset,w,1:3))
            thermal(1)=thermal(1)+coefficients(offset)*(wall_primitive(i+offset,k,w,4)-wall_primitive(i-offset,k,w,4))
            thermal(3)=thermal(3)+coefficients(offset)*(wall_primitive(i,k+offset,w,4)-wall_primitive(i,k-offset,w,4))
          enddo
          do d=1,3
            gradient(:,d)=matmul(directional,dxi(i,j,k,:,d))
          enddo
          stress=mu*(gradient+transpose(gradient))
          do d=1,3
            stress(d,d)=stress(d,d)-(2.d0/3.d0)*mu*(gradient(1,1)+gradient(2,2)+gradient(3,3))
          enddo
          fields(i,k,w,2)=dot_product(tangent,matmul(stress,normal))
          fields(i,k,w,3)=-((mu/prandtl)/const5)*dot_product(matmul(thermal,dxi(i,j,k,:,:)),normal)
          fields(i,k,w,4)=normal(2)
        endif
      enddo
      enddo
    enddo
    call require_fields(status==0.and.all(ieee_is_finite(fields)).and.all(ieee_is_finite(coordinates)), &
      'invalid wall primitive state or diagnostic')
    if(lreadgrid) call record_curve_wall_integrals(coordinates,fields,walls(1:nw))
  end subroutine

  subroutine record_curve_wall_integrals(coordinates,fields,walls)
    use commvar, only: im,km
    use parallel, only: jg0,mpirank
    use insitu_geometry, only: quad_area
    real(real64),intent(in) :: coordinates(0:,0:,:,:),fields(0:,0:,:,:)
    integer,intent(in) :: walls(:)
    real(real64) :: quad(3,0:1,0:1),area,local(4,2),global(4,2)
    integer :: i,k,w,a,b,side,ierr
    logical :: ok,valid
    local=0.d0; valid=.true.
    do w=1,size(walls)
      side=2; if(jg0+walls(w)==0) side=1
      do k=0,km-1
      do i=0,im-1
        do b=0,1
        do a=0,1
          quad(:,a,b)=coordinates(i+a,k+b,w,:)
        enddo
        enddo
        call quad_area(quad,area,ok)
        if(.not.ok) then
          valid=.false.; cycle
        endif
        local(1,side)=local(1,side)+area
        ! Each cell is owned once; its equal corner masses include periodic seam endpoints.
        do b=0,1
        do a=0,1
          local(2:4,side)=local(2:4,side)+(area/4.d0)*fields(i+a,k+b,w,1:3)
        enddo
        enddo
      enddo
      enddo
    enddo
    call require_fields(valid.and.all(ieee_is_finite(local)),'invalid CURVE wall area/integral')
    call MPI_Allreduce(local,global,8,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr)
    call require_fields(ierr==MPI_SUCCESS.and.all(ieee_is_finite(global)).and.all(global(1,:)>0.d0), &
      'invalid global CURVE wall area/integral')
    if(mpirank==0) then
      do side=1,2
        write(*,'(a,i0,a,es24.16,a,3(es24.16,1x))') &
          'ASTR_INSITU_CURVE_WALL wall=',side,' area=',global(1,side), &
          ' area_means=',global(2:4,side)/global(1,side)
      enddo
    endif
  end subroutine

  subroutine prepare_curve_wall_primitive(state,primitive)
    use commvar, only: im,km,const2,const6
    use parallel, only: isize,ksize,mpileft,mpiright,mpiback,mpifront
    real(real64),intent(in) :: state(0:,0:,:,:)
    real(real64),allocatable,intent(out) :: primitive(:,:,:,:)
    real(real64),allocatable :: send(:),receive(:)
    real(real64) :: rho,velocity(3),pressure
    integer :: i,k,w,nw,status,axis,count,comm,ierr,s,neighbors(2,2),sizes(2)
    logical :: valid
    nw=size(state,3)
    allocate(primitive(-3:im+3,-3:km+3,nw,4), &
      send(max(1,3*max(im+1,km+1)*nw*4)),receive(max(1,3*max(im+1,km+1)*nw*4)),stat=status)
    call require_fields(status==0,'cannot allocate CURVE tangential wall halo')
    primitive=0.d0; valid=.true.
    do w=1,nw
    do k=0,km
    do i=0,im
      rho=state(i,k,w,1)
      if(rho<=0.d0) then
        valid=.false.; cycle
      endif
      velocity=state(i,k,w,2:4)/rho
      pressure=(state(i,k,w,5)-0.5d0*rho*sum(velocity**2))/const6
      primitive(i,k,w,1:3)=velocity
      primitive(i,k,w,4)=pressure/rho*const2
    enddo
    enddo
    enddo
    call require_fields(valid.and.all(ieee_is_finite(primitive)).and. &
      all(primitive(0:im,0:km,:,4)>0.d0),'invalid CURVE wall primitive state')
    sizes=[isize,ksize]; neighbors(:,1)=[mpileft,mpiright]; neighbors(:,2)=[mpiback,mpifront]
    call MPI_Comm_dup(MPI_COMM_WORLD,comm,ierr)
    call require_fields(ierr==MPI_SUCCESS,'cannot create wall tangential communicator')
    do axis=1,2
      if(sizes(axis)==1) then
        do s=1,3
          if(axis==1) then
            primitive(-s,0:km,:,:)=primitive(im-s,0:km,:,:)
            primitive(im+s,0:km,:,:)=primitive(s,0:km,:,:)
          else
            primitive(0:im,-s,:,:)=primitive(0:im,km-s,:,:)
            primitive(0:im,km+s,:,:)=primitive(0:im,s,:,:)
          endif
        enddo
        cycle
      endif
      count=3*merge(km+1,im+1,axis==1)*nw*4
      if(count==0) cycle
      if(axis==1) then
        send(1:count)=reshape(primitive(1:3,0:km,:,:),[count])
      else
        send(1:count)=reshape(primitive(0:im,1:3,:,:),[count])
      endif
      call MPI_Sendrecv(send,count,MPI_DOUBLE_PRECISION,neighbors(1,axis),2*axis, &
        receive,count,MPI_DOUBLE_PRECISION,neighbors(2,axis),2*axis,comm,MPI_STATUS_IGNORE,ierr)
      call require_fields(ierr==MPI_SUCCESS,'CURVE wall positive tangential halo exchange')
      if(axis==1) then
        primitive(im+1:im+3,0:km,:,:)=reshape(receive(1:count),[3,km+1,nw,4])
        send(1:count)=reshape(primitive(im-3:im-1,0:km,:,:),[count])
      else
        primitive(0:im,km+1:km+3,:,:)=reshape(receive(1:count),[im+1,3,nw,4])
        send(1:count)=reshape(primitive(0:im,km-3:km-1,:,:),[count])
      endif
      call MPI_Sendrecv(send,count,MPI_DOUBLE_PRECISION,neighbors(2,axis),2*axis+1, &
        receive,count,MPI_DOUBLE_PRECISION,neighbors(1,axis),2*axis+1,comm,MPI_STATUS_IGNORE,ierr)
      call require_fields(ierr==MPI_SUCCESS,'CURVE wall negative tangential halo exchange')
      if(axis==1) then
        primitive(-3:-1,0:km,:,:)=reshape(receive(1:count),[3,km+1,nw,4])
      else
        primitive(0:im,-3:-1,:,:)=reshape(receive(1:count),[im+1,3,nw,4])
      endif
    enddo
    call MPI_Comm_free(comm,ierr)
    call require_fields(ierr==MPI_SUCCESS,'cannot release wall tangential communicator')
  end subroutine

  subroutine capture_index_plane(axis,index,coordinates,velocity,host_budget,device_budget,reserve)
    use iso_fortran_env, only: int64
    use commvar, only: im,jm,km,numq,num_species,use_gpu
    use commarray, only: x
    use parallel, only: ig0,jg0,kg0,isize,jsize,ksize, &
      mpileft,mpiright,mpidown,mpiup,mpiback,mpifront
    use bc, only: bctype
#ifdef _CUDA
    use insitu_sample_gpu, only: download_plane_state_gpu
#endif
    integer,intent(in) :: axis,index
    real(real64),allocatable,intent(out) :: coordinates(:,:,:),velocity(:,:,:)
    integer(int64),intent(in) :: host_budget,device_budget,reserve
    real(real64),allocatable :: state(:,:,:),send(:),receive(:)
    integer :: dims(3),origin(3),sizes(3),neighbors(2,3),tangent(2),ni,nj,local_index
    integer :: status,ierr,comm,d,c,a,b,ijk(3),count
    logical :: owns
    call require_fields(use_gpu.and.numq==5.and.num_species==0.and.all(bctype==1), &
      'index-plane sampling requires periodic five-variable GPU solver')
    call require_fields(axis>=1.and.axis<=3.and.index>=0,'invalid index-plane selection')
    dims=[im,jm,km]; origin=[ig0,jg0,kg0]; sizes=[isize,jsize,ksize]
    neighbors(:,1)=[mpileft,mpiright]; neighbors(:,2)=[mpidown,mpiup]; neighbors(:,3)=[mpiback,mpifront]
    tangent=pack([1,2,3],[1,2,3]/=axis)
    local_index=index-origin(axis)
    ! Only the upper-side rank owns a plane on a partition interface.
    owns=local_index>=0.and.local_index<dims(axis)
    ni=0; nj=0
    if(owns) then
      ni=dims(tangent(1))+1; nj=dims(tangent(2))+1
    endif
    call require_fields(int(ni,int64)*nj*10*8+int(max(ni,nj),int64)*8*8<=host_budget, &
      'index-plane host budget')
    allocate(state(0:ni-1,0:nj-1,4),coordinates(0:ni-1,0:nj-1,3), &
      velocity(0:ni-1,0:nj-1,3),send(max(1,4*max(ni,nj))),receive(max(1,4*max(ni,nj))),stat=status)
    call require_fields(status==0,'cannot allocate index-plane sample')
#ifdef _CUDA
    call download_plane_state_gpu(axis,local_index,tangent,state,device_budget,reserve)
#else
    call require_fields(.false.,'index-plane sampling requires CUDA build')
#endif
    call MPI_Comm_dup(MPI_COMM_WORLD,comm,ierr)
    call require_fields(ierr==MPI_SUCCESS,'cannot create index-plane communicator')
    ! Reconcile the two tangential endpoints without downloading any 3-D face.
    do d=1,2
      ierr=MPI_SUCCESS
      if(owns) then
        if(d==1) then
          count=nj*4; send(1:count)=reshape(state(0,:,:),[count])
        else
          count=ni*4; send(1:count)=reshape(state(:,0,:),[count])
        endif
        if(sizes(tangent(d))==1) then
          receive(1:count)=send(1:count)
        else
          call MPI_Sendrecv(send,count,MPI_DOUBLE_PRECISION,neighbors(1,tangent(d)),d, &
            receive,count,MPI_DOUBLE_PRECISION,neighbors(2,tangent(d)),d,comm,MPI_STATUS_IGNORE,ierr)
        endif
        if(d==1) then
          state(ni-1,:,:)=reshape(receive(1:count),[nj,4])
        else
          state(:,nj-1,:)=reshape(receive(1:count),[ni,4])
        endif
      endif
      call require_fields(ierr==MPI_SUCCESS,'index-plane endpoint exchange failed')
    enddo
    call MPI_Comm_free(comm,ierr)
    call require_fields(ierr==MPI_SUCCESS,'cannot release index-plane communicator')
    call require_fields(all(ieee_is_finite(state)).and.all(state(:,:,1)>0.d0),'invalid index-plane state')
    do c=1,3
      velocity(:,:,c)=state(:,:,c+1)/state(:,:,1)
    enddo
    do b=0,nj-1
      do a=0,ni-1
        ijk(axis)=local_index; ijk(tangent(1))=a; ijk(tangent(2))=b
        coordinates(a,b,:)=x(ijk(1),ijk(2),ijk(3),1:3)
      enddo
    enddo
    call require_fields(all(ieee_is_finite(velocity)).and.all(ieee_is_finite(coordinates)), &
      'nonfinite index-plane payload')
  end subroutine

  subroutine require_fields(ok,message)
    logical,intent(in) :: ok
    character(*),intent(in) :: message
    integer :: bad,any_bad,ierr,ignored
    bad=merge(0,1,ok)
    call MPI_Allreduce(bad,any_bad,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)
    if(ierr/=MPI_SUCCESS.or.any_bad/=0) then
      write(*,'(A)') 'ASTR INSITU FIELD ERROR: '//message
      call MPI_Abort(MPI_COMM_WORLD,1,ignored)
    endif
  end subroutine


  subroutine capture_sample(fields)
    use commvar, only: im,jm,km,numq,num_species,use_gpu
    use commarray, only: q,rho,vel,prs,tmp
#ifdef _CUDA
    use insitu_sample_gpu, only: download_sample_gpu
#endif
    real(real64),intent(out) :: fields(0:im,0:jm,0:km,11)
    call require_fields(numq==5.and.num_species==0,'sampling requires five-variable perfect gas')
#ifdef _CUDA
    if(use_gpu) then
      call download_sample_gpu(fields)
    else
#else
    call require_fields(.not.use_gpu,'GPU sampling requires CUDA build')
#endif
      fields(:,:,:,1:5)=q(0:im,0:jm,0:km,1:5)
      fields(:,:,:,6)=rho(0:im,0:jm,0:km)
      fields(:,:,:,7:9)=vel(0:im,0:jm,0:km,1:3)
      fields(:,:,:,10)=prs(0:im,0:jm,0:km)
      fields(:,:,:,11)=tmp(0:im,0:jm,0:km)
#ifdef _CUDA
    endif
#endif
    call require_fields(all(ieee_is_finite(fields)),'nonfinite sample')
  end subroutine

  subroutine capture_canonical_velocity(velocity)
    use commvar, only: im,jm,km,numq,num_species,use_gpu
#ifdef _CUDA
    use insitu_sample_gpu, only: download_velocity_state_gpu
#endif
    real(real64),intent(out) :: velocity(0:im,0:jm,0:km,3)
    real(real64),allocatable :: state(:,:,:,:)
    integer :: status,c
    call require_fields(use_gpu.and.numq==5.and.num_species==0,'selected sampling requires five-variable GPU solver')
    allocate(state(0:im,0:jm,0:km,4),stat=status)
    call require_fields(status==0,'cannot allocate private velocity state')
#ifdef _CUDA
    call download_velocity_state_gpu(state)
#else
    call require_fields(.false.,'selected sampling requires CUDA build')
#endif
    call complete_periodic_endpoints(state)
    call require_fields(all(ieee_is_finite(state)).and.all(state(:,:,:,1)>0.d0),'invalid selected velocity state')
    do c=1,3
      velocity(:,:,:,c)=state(:,:,:,c+1)/state(:,:,:,1)
    enddo
    call require_fields(all(ieee_is_finite(velocity)),'nonfinite selected velocity')
  end subroutine

  subroutine canonicalize_sample(fields)
    use commvar, only: im,jm,km,const2,const6,numq,num_species
    real(real64),intent(inout) :: fields(0:im,0:jm,0:km,11)
    integer :: axis
    call require_fields(numq==5.and.num_species==0,'ownership requires five-variable perfect gas')
    call complete_periodic_endpoints(fields(:,:,:,1:5))
    call require_fields(all(fields(:,:,:,1)>0.d0),'nonpositive diagnostic density')
    fields(:,:,:,6)=fields(:,:,:,1)
    do axis=1,3
      fields(:,:,:,6+axis)=fields(:,:,:,axis+1)/fields(:,:,:,6)
    enddo
    fields(:,:,:,10)=(fields(:,:,:,5)-0.5d0*fields(:,:,:,6)* &
                      sum(fields(:,:,:,7:9)**2,dim=4))/const6
    fields(:,:,:,11)=fields(:,:,:,10)/fields(:,:,:,6)*const2
    call require_fields(all(ieee_is_finite(fields)).and.all(fields(:,:,:,10)>0.d0), &
                        'invalid canonical primitive state')
  end subroutine

  subroutine complete_periodic_endpoints(fields)
    use bc, only: bctype
    real(real64),intent(inout) :: fields(0:,0:,0:,:)
    call require_fields(all(bctype==1),'ownership requires periodic boundaries')
    call complete_owned_endpoints(fields)
  end subroutine

  subroutine complete_owned_endpoints(fields)
    use commvar, only: im,jm,km
    use bc, only: bctype
    use parallel, only: isize,jsize,ksize,mpileft,mpiright,mpidown,mpiup,mpiback,mpifront
    real(real64),intent(inout) :: fields(0:,0:,0:,:)
    real(real64),allocatable :: send(:),receive(:)
    integer :: axis,count,status,ierr,comm,neighbors(2,3),sizes(3),extent(3),max_count,nfields
    nfields=size(fields,4)
    call require_fields(all(shape(fields)==[im+1,jm+1,km+1,nfields]).and.nfields>0, &
                        'invalid endpoint array shape')
    call require_fields(all(bctype==1).or.all(bctype==[1,1,41,41,1,1]), &
      'ownership requires periodic or bc41 channel boundaries')
    ! Retain physical upper walls; other upper endpoints are copies of the next owner.
    sizes=[isize,jsize,ksize]
    extent=[im,jm,km]
    neighbors(:,1)=[mpileft,mpiright]
    neighbors(:,2)=[mpidown,mpiup]
    neighbors(:,3)=[mpiback,mpifront]
    max_count=nfields*max((im+1)*(jm+1),(im+1)*(km+1),(jm+1)*(km+1))
    allocate(send(max_count),receive(max_count),stat=status)
    call require_fields(status==0,'cannot allocate ownership exchange buffers')
    call MPI_Comm_dup(MPI_COMM_WORLD,comm,ierr)
    call require_fields(ierr==MPI_SUCCESS,'cannot create diagnostic communicator')
    ! Sequential face copies carry already reconciled edges into the next axis.
    do axis=1,3
      if(sizes(axis)==1.and.bctype(2*axis)/=1) cycle
      count=nfields*product(extent+1)/(extent(axis)+1)
      select case(axis)
      case(1)
        send(1:count)=reshape(fields(0,:,:,:),[count])
      case(2)
        send(1:count)=reshape(fields(:,0,:,:),[count])
      case(3)
        send(1:count)=reshape(fields(:,:,0,:),[count])
      end select
      if(sizes(axis)==1) then
        receive(1:count)=send(1:count)
      else
        call MPI_Sendrecv(send,count,MPI_DOUBLE_PRECISION,neighbors(1,axis),axis, &
                          receive,count,MPI_DOUBLE_PRECISION,neighbors(2,axis),axis, &
                          comm,MPI_STATUS_IGNORE,ierr)
        call require_fields(ierr==MPI_SUCCESS,'ownership exchange failed')
      endif
      if(sizes(axis)>1.and.neighbors(2,axis)==MPI_PROC_NULL) cycle
      select case(axis)
      case(1)
        fields(im,:,:,:)=reshape(receive(1:count),[jm+1,km+1,nfields])
      case(2)
        fields(:,jm,:,:)=reshape(receive(1:count),[im+1,km+1,nfields])
      case(3)
        fields(:,:,km,:)=reshape(receive(1:count),[im+1,jm+1,nfields])
      end select
    enddo
    call MPI_Comm_free(comm,ierr)
    call require_fields(ierr==MPI_SUCCESS,'cannot release diagnostic communicator')
    deallocate(send,receive)
  end subroutine

  subroutine derive_sample(velocity,derived)
    use commvar, only: im,jm,km,hm,difschm
    use bc, only: bctype
    use commarray, only: dxi
    use parallel, only: dataswap,mpirank
    use derivative, only: diff6ec
    real(real64),intent(in) :: velocity(0:im,0:jm,0:km,3)
    real(real64),intent(out) :: derived(0:im,0:jm,0:km,14)
    real(real64),allocatable :: work(:,:,:,:)
    real(real64) :: a(3,3)
    real(real64) :: halo_started,halo_seconds,gradient_started,gradient_seconds,q_started,q_seconds
    integer :: i,j,k,c,d,status
    call require_fields(all(bctype==1).and.hm>=3.and.trim(difschm)=='643e', &
      'diagnostics require periodic boundaries and explicit sixth-order derivatives')
    call require_fields(min(im,jm,km)>=3,'diagnostic subdomains need at least three cells per axis')
    allocate(work(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3),stat=status)
    call require_fields(status==0,'cannot allocate private diagnostic halo')
    work=0.d0
    work(0:im,0:jm,0:km,:)=velocity
    ! Blocking exchange updates only this private halo, not solver primitive arrays.
    halo_started=MPI_Wtime()
    call dataswap(work)
    halo_seconds=MPI_Wtime()-halo_started
    write(*,'(a,i0,a,es16.8)') 'ASTR_INSITU_HALO_TIMING rank=',mpirank,' seconds=',halo_seconds
    call require_fields(all(work(0:im,0:jm,0:km,:)==velocity),'diagnostic exchange changed physical values')
    gradient_started=MPI_Wtime()
    derived=0.d0
    do c=1,3
      do d=1,3
        do k=0,km
          do j=0,jm
            derived(:,j,k,c+3*(d-1))=diff6ec(work(:,j,k,c),im,3)*dxi(0:im,j,k,1,d)
          enddo
        enddo
        do k=0,km
          do i=0,im
            derived(i,:,k,c+3*(d-1))=derived(i,:,k,c+3*(d-1))+ &
              diff6ec(work(i,:,k,c),jm,3)*dxi(i,0:jm,k,2,d)
          enddo
        enddo
        do j=0,jm
          do i=0,im
            derived(i,j,:,c+3*(d-1))=derived(i,j,:,c+3*(d-1))+ &
              diff6ec(work(i,j,:,c),km,3)*dxi(i,j,0:km,3,d)
          enddo
        enddo
      enddo
    enddo
    gradient_seconds=MPI_Wtime()-gradient_started
    q_started=MPI_Wtime()
    do k=0,km
      do j=0,jm
        do i=0,im
          a=reshape(derived(i,j,k,1:9),[3,3])
          derived(i,j,k,10)=-0.5d0*sum(a*transpose(a))
          derived(i,j,k,11)=a(1,1)+a(2,2)+a(3,3)
          derived(i,j,k,12)=a(3,2)-a(2,3)
          derived(i,j,k,13)=a(1,3)-a(3,1)
          derived(i,j,k,14)=a(2,1)-a(1,2)
        enddo
      enddo
    enddo
    q_seconds=MPI_Wtime()-q_started
    write(*,'(a,i0,a,2es16.8)') 'ASTR_INSITU_DERIVATIVE_TIMING rank=',mpirank, &
      ' gradient q_div_curl=',gradient_seconds,q_seconds
    deallocate(work)
    call require_fields(all(ieee_is_finite(derived)),'nonfinite derived sample')
  end subroutine

end module
