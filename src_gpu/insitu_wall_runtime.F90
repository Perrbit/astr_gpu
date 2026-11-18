module insitu_wall_runtime_cpp
  use iso_c_binding
  use iso_fortran_env, only: int64
  use mpi
  use astr_precision, only: real_kind,real_bytes
  use astr_precision_mpi, only: mpi_real_type
  use ieee_arithmetic, only: ieee_is_finite
  use commvar, only: im,jm,km,hm,ia,ja,ka,numq,ndims,use_gpu,lreadgrid,conschm, &
    const2,const5,const6,reynolds,prandtl,tempconst,tempconst1,lihomo,ljhomo,lkhomo
  use commarray, only: x
  use parallel, only: ig0,jg0,mpirank,mpisize,isize,jsize,ksize, &
    mpileft,mpiright,mpidown,mpiup,mpiback,mpifront
  use boundary_high_order, only: high_order_wall_enabled
  use insitu_geometry, only: wall_frame
  use fluid_runtime, only: cpp_device_context,cpp_wall_forcing_active
  use device_state, only: device_memory_info
  use insitu_wall_cpp
  implicit none
  private
  public :: catalog_wall_cpp_supported,begin_catalog_wall_cpp,end_catalog_wall_cpp,catalog_wall_line_cpp
  public :: prepare_wall_statistics_cpp,capture_and_push_wall_statistics_cpp
  public :: air5_wall_span_cpp
  public :: pack_wall_statistics_cpp,restore_wall_statistics_cpp,release_wall_statistics_cpp
  public :: sbli_insitu_cpp_supported,catalog_sbli_wall_cpp_supported,capture_catalog_sbli_wall_cpp
  public :: capture_wall_packet_cpp
  type(c_ptr),save :: catalog=c_null_ptr,history=c_null_ptr
  logical,save :: sbli_grid_checked=.false.,sbli_grid_rectilinear=.false.
contains
  logical function sbli_insitu_cpp_supported() result(ok)
    use commvar, only: flowtype,difschm,recon_schem,lchardecomp,lfilter,diffterm,nondimen,turbmode, &
      num_species,num_modequ,lcomb,ia,ka
    use bc, only: bctype,turbinf,nscbc_farfield_sbli_shock_enabled
    use sponge_layer, only: spg_i0,spg_im,spg_j0,spg_jm,spg_k0,spg_km
    use case_model_options, only: get_case_option
    character(32) :: precision
    integer :: status
    call get_case_option('gpu_precision_mode',precision,status=status)
    if(status==1) precision='fp64'
    ok=use_gpu.and.lreadgrid.and.real_bytes==8.and. &
      precision=='fp64'.and. &
      flowtype=='bl'.and.ndims==3.and.numq==5.and.num_species==0.and.num_modequ==0.and..not.lcomb.and. &
      conschm=='743e'.and.difschm=='643e'.and.recon_schem==5.and.lchardecomp.and..not.lfilter.and. &
      diffterm.and.nondimen.and..not.lihomo.and..not.ljhomo.and.lkhomo.and.turbmode=='none'.and. &
      turbinf=='prof'.and.all(bctype==[11,22,41,52,1,1]).and.nscbc_farfield_sbli_shock_enabled().and. &
      all([spg_i0,spg_im,spg_j0,spg_jm,spg_k0,spg_km]==0).and.min(ia,ja)>=6.and.ka>=5.and.hm>=3.and. &
      all([isize,jsize,ksize]==[mpisize,1,1]).and.any(mpisize==[1,2,4])
  end function
  logical function catalog_sbli_wall_cpp_supported() result(ok)
    integer :: i,j,k
    ok=.false.
    if(.not.sbli_insitu_cpp_supported().or..not.c_associated(cpp_device_context()).or.cpp_wall_forcing_active()) return
    if(.not.sbli_grid_checked) then
      sbli_grid_rectilinear=all(x(1:im,0,0,1)>x(0:im-1,0,0,1)).and. &
        all(x(0,1:jm,0,2)>x(0,0:jm-1,0,2)).and.all(x(0,0,1:km,3)>x(0,0,0:km-1,3))
      do k=0,km
      do j=0,jm
      do i=0,im
        sbli_grid_rectilinear=sbli_grid_rectilinear.and.all(ieee_is_finite(x(i,j,k,:))).and. &
          x(i,j,k,1)==x(i,0,0,1).and.x(i,j,k,2)==x(0,j,0,2).and.x(i,j,k,3)==x(0,0,k,3)
      enddo
      enddo
      enddo
      sbli_grid_checked=.true.
    endif
    ok=sbli_grid_rectilinear
  end function
  subroutine capture_catalog_sbli_wall_cpp(mode,step,time,host_budget,device_budget,reserve,values,download,ok)
    use commvar, only: ia,roinf,uinf,pinf
    use parallel, only: ig0
    character(*),intent(in) :: mode
    integer(int64),intent(in) :: step,host_budget,device_budget,reserve
    real(real_kind),intent(in) :: time
    real(real_kind),allocatable,intent(out) :: values(:,:)
    integer(int64),intent(out) :: download
    logical,intent(out) :: ok
    type(c_ptr) :: sample
    real(real_kind),allocatable :: wall(:,:),local(:,:),global(:,:)
    real(real_kind) :: bounds(6),fraction,mid_z,material(7),reference(3)
    integer(c_int) :: transport,status
    integer(c_size_t) :: free,total
    integer(int64) :: host_bytes,device_bytes,usage(8)
    integer :: first,i,err,allocerr
    download=0;sample=c_null_ptr;transport=-1
    if(mode=='pinned') transport=0
    if(mode=='device-aware') transport=1
    ok=agreed(catalog_sbli_wall_cpp_supported().and.transport>=0);if(.not.ok) return
    host_bytes=(4_int64*(im+1)+14_int64*(ia+1))*real_bytes
    if(transport==0.and.mpisize>1) host_bytes=host_bytes+140*real_bytes
    device_bytes=(75_int64*(im+1)+140)*real_bytes+4
    free=0;total=0;status=device_memory_info(cpp_device_context(),free,total)
    ok=agreed(status==0.and.host_bytes<=host_budget.and.device_bytes<=device_budget.and. &
      reserve>=0.and.int(free,int64)>=reserve.and.device_bytes<=int(free,int64)-reserve)
    if(.not.ok) return
    allocate(wall(4,0:im),local(4,0:ia),global(4,0:ia),values(6,ia+1),stat=allocerr)
    ok=agreed(allocerr==0);if(.not.ok) return
    bounds=[x(0,0,0,1),x(im,0,0,1),x(0,0,0,2),x(0,jm,0,2),x(0,0,0,3),x(0,0,km,3)]
    mid_z=.5_real_kind*(bounds(5)+bounds(6))
    do first=0,km-1
      if(x(0,0,first+1,3)>=mid_z) exit
    enddo
    fraction=(mid_z-x(0,0,first,3))/(x(0,0,first+1,3)-x(0,0,first,3))
    material=[const2,const5,const6,reynolds,prandtl,tempconst,tempconst1];reference=[roinf,uinf,pinf]
    status=sbli_wall_create(cpp_device_context(),int(MPI_COMM_WORLD,c_int),int(mpileft,c_int),int(mpiright,c_int), &
      int(isize,c_int),transport,int(first,c_int),fraction,material,reference,x(0:im,0,0,1),sample)
    ok=agreed(status==0);if(.not.ok) return
    status=sbli_wall_capture(sample,wall,int(size(wall,kind=int64)*real_bytes,c_size_t))
    ok=agreed(status==0)
    call require_status(sbli_wall_metrics(sample,usage),'SBLI wall transfer receipt')
    call require_status(sbli_wall_destroy(sample),'release SBLI wall capture')
    if(.not.ok) return
    local=0;global=0
    do i=0,im
      if(i==im.and.ig0+im<ia) cycle
      local(:,ig0+i)=wall(:,i)
    enddo
    call MPI_Reduce(local,global,4*(ia+1),mpi_real_type,MPI_SUM,0,MPI_COMM_WORLD,err)
    call require_status(err,'SBLI wall line reduction')
    values(1,:)=global(1,:);values(2,:)=bounds(3);values(3,:)=mid_z;values(4:6,:)=global(2:4,:)
    download=usage(7)
    write(*,'(a,i0,a,i0,4(a,i0),a,es24.16)') 'ASTR_RUNTIME_CPP_SBLI_WALL_SOURCE rank=',mpirank,' step=',step, &
      ' curve_d2h_bytes=',usage(7),' private_d2h_bytes=',usage(4),' private_h2d_bytes=',usage(5), &
      ' mpi_sent_bytes=',usage(3),' fraction=',fraction
  end subroutine
  logical function agreed(ok) result(all_ok)
    logical,intent(in) :: ok
    integer :: bad,total,err
    bad=merge(0,1,ok)
    call MPI_Allreduce(bad,total,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,err)
    if(err/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,err,bad)
    all_ok=total==0
  end function
  subroutine require_status(code,where)
    integer,intent(in) :: code
    character(*),intent(in) :: where
    integer :: err
    if(agreed(code==0)) return
    print *, 'C++ wall rank=',mpirank,trim(where),' status=',code
    call MPI_Abort(MPI_COMM_WORLD,1,err)
  end subroutine
  logical function catalog_wall_cpp_supported() result(ok)
    use commvar, only: flowtype,num_species,num_modequ,lcomb,nondimen,difschm
    use bc, only: bctype
    ok=use_gpu.and.c_associated(cpp_device_context()).and.ndims==3.and.hm>=3.and.trim(conschm)=='643e'
    if(numq==11) then
      ok=ok.and.real_bytes==8.and.flowtype=='air5hbl'.and.lcomb.and.num_species==5.and.num_modequ==1.and. &
        .not.nondimen.and..not.lreadgrid.and.all([ia,ja,ka]==16).and.mpisize<=2.and.jm>=2.and. &
        all(bctype==[11,50,41,51,1,1]).and.difschm=='643e'
    else
      ok=ok.and.numq==5
      if(lreadgrid) ok=ok.and.min(im,km)>=3.and.lihomo.and.lkhomo
      if(high_order_wall_enabled) ok=ok.and.jm>=7
    endif
  end function
  subroutine wall_layout(nw,first_y)
    integer,intent(out) :: nw,first_y
    nw=merge(1,0,jg0==0);first_y=merge(0,jm,jg0==0)
    if(numq==5.and.jg0+jm==ja) nw=nw+1
  end subroutine
  subroutine prepare_handle(handle,mode,statistics,host_budget,device_budget,reserve,ok)
    type(c_ptr),intent(out) :: handle
    character(*),intent(in) :: mode
    logical,intent(in) :: statistics
    integer(int64),intent(in) :: host_budget,device_budget,reserve
    logical,intent(out) :: ok
    real(real_kind),allocatable :: geometry(:,:,:,:)
    real(real_kind),parameter :: weights(3)=[.75_real_kind,-.15_real_kind,1._real_kind/60._real_kind]
    real(real_kind) :: first(3),second(3),inward(3),normal(3),tangent(3),material(7)
    integer(c_int) :: low(3),high(3),sizes(3),periodic(3),transport,status
    integer(c_size_t) :: free,total
    integer(int64) :: nodes,capacity,host_bytes,device_bytes,allocation_bytes,usage(10)
    integer :: nw,first_y,layers,i,j,k,w,s,side,allocerr,state_fields,nfields,groups
    logical :: frame_ok,valid,curved
    handle=c_null_ptr;transport=-1
    if(mode=='pinned') transport=0
    if(mode=='device-aware') transport=1
    curved=lreadgrid.and..not.high_order_wall_enabled
    ok=agreed(catalog_wall_cpp_supported().and.transport>=0);if(.not.ok) return
    call wall_layout(nw,first_y)
    layers=3;if(high_order_wall_enabled.and.numq==5) layers=8
    state_fields=layers*merge(11,5,numq==11)
    nfields=merge(18,4,numq==11);groups=merge(6,1,numq==11)
    nodes=int(im+1,int64)*(km+1)*nw
    capacity=int(max(state_fields,12),int64)*nw*max(im+1,km+1)
    host_bytes=9*nodes*real_bytes
    if(transport==0.and.mpisize>1) host_bytes=host_bytes+2*capacity*real_bytes
    device_bytes=((state_fields+9+nfields+merge(30*groups,0,statistics))*nodes+6_int64*(im+1)+2*capacity)*real_bytes+4
    if(curved) device_bytes=device_bytes+4_int64*(im+7)*(km+7)*nw*real_bytes
    allocation_bytes=device_bytes
    if(.not.statistics.and.c_associated(history)) then
      call require_status(wall_metrics(history,usage),'retained wall history bytes')
      device_bytes=device_bytes+usage(1);host_bytes=host_bytes+usage(3)
    endif
    free=0;total=0
    status=device_memory_info(cpp_device_context(),free,total)
    ok=agreed(status==0.and.host_bytes<=host_budget.and.device_bytes<=device_budget.and. &
      reserve>=0.and.int(free,int64)>=reserve.and.allocation_bytes<=int(free,int64)-reserve)
    if(.not.ok) return
    allocate(geometry(0:im,0:km,nw,9),stat=allocerr)
    ok=agreed(allocerr==0);if(.not.ok) return
    valid=.true.
    do w=1,nw
      j=first_y;if(w==2) j=jm
      side=1;if(j==jm) side=-1
      do k=0,km
      do i=0,im
        normal=[0._real_kind,real(side,real_kind),0._real_kind];tangent=[1._real_kind,0._real_kind,0._real_kind]
        if(curved) then
          first=0;second=0
          do s=1,3
            first=first+weights(s)*(x(i+s,j,k,:)-x(i-s,j,k,:))
            second=second+weights(s)*(x(i,j,k+s,:)-x(i,j,k-s,:))
          enddo
          inward=x(i,j+side,k,:)-x(i,j,k,:)
          call wall_frame(first,second,inward,normal,tangent,frame_ok)
          valid=valid.and.frame_ok
        endif
        geometry(i,k,w,:)=[x(i,j,k,:),normal,tangent]
      enddo
      enddo
    enddo
    ok=agreed(valid.and.all(ieee_is_finite(geometry)));if(.not.ok) return
    material=[const2,const5,const6,reynolds,prandtl,tempconst,tempconst1]
    low=int([mpileft,mpidown,mpiback],c_int);high=int([mpiright,mpiup,mpifront],c_int)
    sizes=int([isize,jsize,ksize],c_int);periodic=merge(1_c_int,0_c_int,[lihomo,ljhomo,lkhomo])
    status=wall_create(cpp_device_context(),int(MPI_COMM_WORLD,c_int),low,high,sizes,periodic,transport, &
      int(first_y,c_int),int(nw,c_int),int(layers,c_int),merge(1_c_int,0_c_int,curved),material,geometry,handle)
    ok=status==0
  end subroutine
  subroutine prepare_wall_statistics_cpp(mode,host_budget,device_budget,reserve)
    character(*),intent(in) :: mode
    integer(int64),intent(in) :: host_budget,device_budget,reserve
    character(16) :: selected
    logical :: ok
    if(c_associated(history)) return
    selected=mode;if(selected=='') selected='pinned'
    call prepare_handle(history,selected,.true.,host_budget,device_budget,reserve,ok)
    call require_status(merge(0,1,ok),'prepare device wall history')
  end subroutine
  subroutine capture_wall_packet_cpp(mode,host_budget,device_budget,reserve,coordinates,fields,owned,download,success)
    character(*),intent(in) :: mode
    integer(int64),intent(in) :: host_budget,device_budget,reserve
    real(real_kind),allocatable,intent(out) :: coordinates(:,:,:,:),fields(:,:,:,:)
    logical,allocatable,intent(out) :: owned(:,:,:)
    integer(int64),intent(out) :: download
    logical,optional,intent(out) :: success
    type(c_ptr) :: sample
    integer(int64) :: usage(10)
    integer :: nw,first_y,w,j,i,k,status,nfields
    integer(int64) :: nodes,bytes
    logical :: ok
    download=0
    call wall_layout(nw,first_y)
    nfields=merge(18,4,numq==11)
    call prepare_handle(sample,mode,.false.,host_budget,device_budget,reserve,ok)
    if(present(success)) then
      success=ok
      if(.not.ok)return
    endif
    call require_status(merge(0,1,ok),'prepare wall sample packet')
    call require_status(wall_metrics(sample,usage),'wall sample packet workspace')
    nodes=int(im+1,int64)*(km+1)*nw
    bytes=nodes*((3+nfields)*real_bytes+storage_size(.false.)/8)+usage(3)
    status=1
    if(bytes<=host_budget) allocate(coordinates(0:im,0:km,nw,3),fields(0:im,0:km,nw,nfields), &
      owned(0:im,0:km,nw),stat=status)
    if(agreed(status==0)) then
      status=wall_capture(sample)
      if(status==0.and.nodes>0) status=wall_download(sample,fields,int(nodes*nfields*real_bytes,c_size_t))
    else
      status=1
    endif
    ok=agreed(status==0)
    if(.not.ok) then
      call require_status(wall_destroy(sample),'release failed wall sample packet')
      if(present(success)) then
        success=.false.;return
      endif
      call require_status(1,'assemble wall sample packet')
    endif
    do w=1,nw
      j=first_y;if(w==2) j=jm
      do k=0,km
      do i=0,im
        coordinates(i,k,w,:)=x(i,j,k,:)
      enddo
      enddo
    enddo
    owned=.false.;owned(0:im-1,0:km-1,:)=.true.
    if(numq==11.and.ig0+im==ia) owned(im,0:km-1,:)=.true.
    call require_status(wall_metrics(sample,usage),'wall sample packet transfer receipt')
    download=usage(8)
    call require_status(wall_destroy(sample),'release wall sample packet')
  end subroutine
  subroutine capture_and_push_wall_statistics_cpp(mode,weights,previous,host_budget,device_budget,reserve)
    character(*),intent(in) :: mode
    real(real_kind),intent(in) :: weights(2)
    logical,intent(in) :: previous
    integer(int64),intent(in) :: host_budget,device_budget,reserve
    integer(int64) :: before(10),after(10)
    call prepare_wall_statistics_cpp(mode,host_budget,device_budget,reserve)
    call require_status(wall_metrics(history,before),'before wall statistics')
    call require_status(wall_capture(history),'device wall statistics sample')
    call require_status(wall_statistics_push(history,weights,merge(1_c_int,0_c_int,previous)),'push device wall statistics')
    call require_status(wall_metrics(history,after),'after wall statistics')
    write(*,'(a,i0,5(a,i0))') 'ASTR_INSITU_CPP_WALL_STATS rank=',mpirank, &
      ' staging_d2h_bytes=',after(5)-before(5),' staging_h2d_bytes=',after(6)-before(6), &
      ' status_d2h_bytes=',after(7)-before(7),' field_download_bytes=',after(8)-before(8), &
      ' face_exchange_bytes=',after(4)-before(4)
  end subroutine
  subroutine air5_wall_span_cpp(weights,profile)
    real(real_kind),contiguous,intent(in) :: weights(:)
    real(real_kind),contiguous,intent(out) :: profile(:,:)
    integer(c_int) :: owned_x
    integer(int64) :: nodes
    owned_x=int(im+merge(1,0,ig0+im==ia),c_int)
    call require_status(wall_air5_span(history,weights,owned_x,profile),'AIR5 wall spanwise shear integral')
    nodes=merge(int(im+1,int64),0_int64,jg0==0)
    write(*,'(a,i0,3(a,i0))') 'ASTR_INSITU_CPP_WALL_SEPARATION rank=',mpirank, &
      ' profile_download_bytes=',3*nodes*real_bytes,' weight_upload_bytes=',merge(km*real_bytes,0,jg0==0), &
      ' status_download_bytes=',merge(4,0,jg0==0)
  end subroutine
  subroutine pack_wall_statistics_cpp(values,window,time)
    real(real_kind),contiguous,intent(out) :: values(:,:,:,:)
    real(real_kind),intent(in) :: window(2),time
    call require_status(wall_statistics_pack(history,values,int(size(values,kind=int64)*real_bytes,c_size_t),window,time), &
      'pack device wall statistics')
  end subroutine
  subroutine restore_wall_statistics_cpp(values)
    real(real_kind),contiguous,intent(in) :: values(:,:,:,:)
    call require_status(wall_statistics_restore(history,values,int(size(values,kind=int64)*real_bytes,c_size_t)), &
      'restore device wall statistics')
  end subroutine
  subroutine release_wall_statistics_cpp()
    call require_status(wall_destroy(history),'release wall history')
  end subroutine
  subroutine begin_catalog_wall_cpp(mode,mean,duration,host_budget,device_budget,reserve,ok)
    character(*),intent(in) :: mode
    logical,intent(in) :: mean
    real(real_kind),intent(in) :: duration
    integer(int64),intent(in) :: host_budget,device_budget,reserve
    logical,intent(out) :: ok
    integer(c_int) :: status
    call prepare_handle(catalog,mode,.false.,host_budget,device_budget,reserve,ok)
    if(.not.ok) return
    status=wall_capture(catalog)
    if(status==0.and.mean) then
      ok=agreed(c_associated(history))
      if(ok) then
        status=wall_statistics_mean(catalog,history,duration)
      else
        status=1
      endif
    endif
    ok=status==0
    if(.not.ok) call end_catalog_wall_cpp()
  end subroutine
  subroutine catalog_wall_line_cpp(k,w,line,download,ok)
    integer,intent(in) :: k,w
    real(real_kind),contiguous,intent(out) :: line(:,:)
    integer(int64),intent(out) :: download
    logical,intent(out) :: ok
    line=0;download=0;ok=.true.
    if(k<0.or.w<1) return
    ok=wall_line(catalog,int(k,c_int),int(w-1,c_int),line,int(size(line,kind=int64)*real_bytes,c_size_t))==0
    if(ok) download=size(line,kind=int64)*real_bytes
  end subroutine
  subroutine end_catalog_wall_cpp()
    call require_status(wall_destroy(catalog),'release catalog wall capture')
  end subroutine
end module
