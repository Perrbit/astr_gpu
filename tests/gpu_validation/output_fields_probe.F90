program output_fields_probe
  use iso_fortran_env, only: int64,real64
  use mpi
  use ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  use commvar, only: im,jm,km,hm,numq,num_species,difschm,lihomo,ljhomo,lkhomo,ia,ja,ka
  use commarray, only: rho,vel,prs,tmp,x,dxi
  use bc, only: bctype
  use parallel, only: isize,jsize,ksize,mpileft,mpiright,mpidown,mpiup,mpiback,mpifront,mpitag
#ifdef ASTR_AIR5_CHEMISTRY
  use commarray, only: tve,spc
#endif
  use output_fields, only: write_basic_output,pack_basic_output_cpu,write_slice_output_xdmf,write_output_series_xdmf
  use output_fields, only: begin_derived_output_cpu,pack_derived_output_cpu,end_derived_output_cpu
  use checkpoint_state_io, only: checkpoint_state_identity
#ifdef TEST_FIELDS_GPU
  use commarray_gpu, only: rho_d,vel_d,prs_d,tmp_d,dxi_d
  use commarray_gpu, only: tve_d,spc_d
  use output_fields_gpu, only: begin_basic_output_gpu,pack_basic_output_gpu,end_basic_output_gpu
  use output_fields_gpu, only: begin_derived_output_gpu,pack_derived_output_gpu,end_derived_output_gpu
  use cudafor, only: cudaSetDevice,cudaSuccess
  use halo_exchange_gpu, only: solution_pipeline_active
#endif
  implicit none
  character(1024) :: path,prefix,geometry_path,geometry_source
  character(32) :: argument,label,fault,units
  character(1),parameter :: tags(3)=['i','j','k']
  integer :: ierr,rank,ranks,components,decomposition,global_shape(3),cells(3),origin(3),axis,index
  integer :: i,j,k,m,backend,unit,status,requested_components
  integer :: derived(14),derived_count,derived_budget
  integer,parameter :: numeric_cells=10
  integer(int64) :: peak,bytes,download,global_bytes,host_workspace,device_workspace,capacity,provider_budget
  real(real64) :: encoded,point(3),pi
  real(real64),allocatable :: before_velocity(:,:,:,:),after_velocity(:,:,:,:)
  type(checkpoint_state_identity) :: identity
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)
  call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr)
  call get_command_argument(1,prefix)
  call get_command_argument(2,argument); read(argument,*) components
  call get_command_argument(3,argument); read(argument,*) decomposition
  call get_command_argument(4,fault)
  global_shape=[9,7,5]; cells=global_shape-1; origin=0
  if(fault(1:8)=='numeric_') then
    global_shape=numeric_cells+1; cells=global_shape-1
  endif
  if(trim(fault)=='series_repeat') then
    if(decomposition==2) then
      open(newunit=unit,status='scratch')
      close(unit)
    endif
    do m=1,50
      write(label,'("/series",i3.3,".xdmf")') m
      call write_output_series_xdmf(trim(prefix)//trim(label),trim(prefix)//'/series.frames', &
        global_shape,[1,2,3],[4,3,2],components,'si',MPI_COMM_WORLD)
    enddo
    call MPI_Finalize(ierr)
    stop
  endif
  if(trim(fault)=='series_volume'.or.trim(fault)=='series_planes') then
    if(trim(fault)=='series_volume') then
      call write_output_series_xdmf(trim(prefix)//'/series.xdmf',trim(prefix)//'/series.frames', &
        global_shape,[0],[0],components,'si',MPI_COMM_WORLD)
    else
      call write_output_series_xdmf(trim(prefix)//'/series.xdmf',trim(prefix)//'/series.frames', &
        global_shape,[1,2,3],[4,3,2],components,'si',MPI_COMM_WORLD)
    endif
    print *, 'OUTPUT_SERIES_PASS'
    call MPI_Finalize(ierr)
    stop
  endif
  cells(decomposition)=cells(decomposition)/ranks
  origin(decomposition)=rank*cells(decomposition)
  im=cells(1); jm=cells(2); km=cells(3)
  allocate(rho(-hm:im+hm,-hm:jm+hm,-hm:km+hm),vel(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3), &
    prs(-hm:im+hm,-hm:jm+hm,-hm:km+hm),tmp(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
    x(0:im,0:jm,0:km,3))
#ifdef ASTR_AIR5_CHEMISTRY
  if(components==12) allocate(tve(-hm:im+hm,-hm:jm+hm,-hm:km+hm),spc(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5))
#endif
  do k=-hm,km+hm
    do j=-hm,jm+hm
      do i=-hm,im+hm
        point=real(origin+[i,j,k],real64)
        encoded=100*point(1)+10*point(2)+point(3)+rank/1024.d0
        rho(i,j,k)=encoded+1/16.d0
        do m=1,3
          vel(i,j,k,m)=encoded+(m+1)/16.d0
        enddo
        prs(i,j,k)=encoded+5/16.d0; tmp(i,j,k)=encoded+6/16.d0
#ifdef ASTR_AIR5_CHEMISTRY
        if(components==12) then
          tve(i,j,k)=encoded+7/16.d0
          do m=1,5
            spc(i,j,k,m)=encoded+(m+7)/16.d0
          enddo
        endif
#endif
      enddo
    enddo
  enddo
  do k=0,km
    do j=0,jm
      do i=0,im
        point=real(origin+[i,j,k],real64)
        x(i,j,k,:)=[point(1)+point(2)/32,point(2)+point(1)/16,point(3)+point(1)*point(2)/64]
      enddo
    enddo
  enddo
  identity=checkpoint_state_identity(12_int64,0.012d0,0.001d0,0.001d0)
  if(fault(1:8)=='numeric_') then
    if(components/=6) error stop 'numeric fixture only supports perfect gas'
    numq=5; num_species=0; difschm='643e'; bctype=1
    lihomo=.true.; ljhomo=.true.; lkhomo=.true.; ia=numeric_cells; ja=numeric_cells; ka=numeric_cells
    isize=1; jsize=1; ksize=1; mpitag=100
    mpileft=rank; mpiright=rank; mpidown=rank; mpiup=rank; mpiback=rank; mpifront=rank
    select case(decomposition)
    case(1)
      isize=ranks; mpileft=mod(rank+ranks-1,ranks); mpiright=mod(rank+1,ranks)
    case(2)
      jsize=ranks; mpidown=mod(rank+ranks-1,ranks); mpiup=mod(rank+1,ranks)
    case(3)
      ksize=ranks; mpiback=mod(rank+ranks-1,ranks); mpifront=mod(rank+1,ranks)
    end select
    pi=acos(-1.d0)
    allocate(dxi(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3))
    dxi=0.d0
    do m=1,3
      dxi(:,:,:,m,m)=real(numeric_cells,real64)/((m+1)*pi)
    enddo
    vel=ieee_value(0.d0,ieee_quiet_nan)
    do k=0,km
      do j=0,jm
        do i=0,im
          point=real(origin+[i,j,k],real64)*(2.d0*pi/numeric_cells)
          vel(i,j,k,:)=[sin(point(1))+0.25d0*cos(point(2))+0.125d0*sin(2*point(3)), &
            0.5d0*cos(point(1))+sin(2*point(2))+0.25d0*cos(point(3)), &
            0.125d0*sin(2*point(1))+0.5d0*cos(2*point(2))+sin(point(3))]
          x(i,j,k,:)=point*[1.d0,1.5d0,2.d0]
        enddo
      enddo
    enddo
    allocate(before_velocity(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3), &
      after_velocity(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3))
    before_velocity=vel
    derived=[(m,m=1,14)]; derived_count=14
    provider_budget=2_int64*1024*1024
    select case(trim(fault))
    case('numeric_gradient')
      derived_count=9
    case('numeric_curl')
      derived(:3)=[12,13,14]; derived_count=3
    case('numeric_q')
      derived(:2)=[10,11]; derived_count=2
    case('numeric_budget','numeric_gpu_budget')
      provider_budget=1
    case('numeric_boundary','numeric_gpu_boundary')
      bctype(1)=41
    case('numeric_scheme','numeric_gpu_scheme')
      difschm='642c'
    end select
    capacity=2160_int64/8/(components+derived_count+3)*(components+derived_count)
#ifdef TEST_FIELDS_GPU
    status=cudaSetDevice(rank)
    if(status/=cudaSuccess) error stop 'numeric fixture GPU selection'
    allocate(rho_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm),vel_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3), &
      prs_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm),tmp_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm), &
      dxi_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3,3))
    rho_d=rho; vel_d=vel; prs_d=prs; tmp_d=tmp; dxi_d=dxi
    if(trim(fault)=='numeric_gpu_busy'.and.rank==0) solution_pipeline_active=.true.
#endif
    do backend=0,1
#ifndef TEST_FIELDS_GPU
      if(backend==1) cycle
#endif
      if(fault(9:12)=='gpu_'.and.backend==0) cycle
      do axis=0,3
        if(trim(fault)=='numeric_slices'.and.axis==0) cycle
        index=0
        if(axis>0) index=numeric_cells/2
        write(label,'("/",i0,"_",i0)') backend,axis
        path=trim(prefix)//trim(label)
        device_workspace=0; download=0
        if(backend==0) then
          call begin_derived_output_cpu(capacity,provider_budget,MPI_COMM_WORLD,derived(:derived_count),host_workspace)
          call write_basic_output(trim(path),global_shape,origin,cells,axis,index,components,x, &
            pack_derived_output_cpu,identity,2160_int64,2160_int64,MPI_COMM_WORLD,'si',peak,bytes, &
            derived_indices=derived(:derived_count))
          call end_derived_output_cpu()
          after_velocity=vel
        else
#ifdef TEST_FIELDS_GPU
          call begin_derived_output_gpu(capacity,provider_budget,provider_budget,MPI_COMM_WORLD, &
            derived(:derived_count),host_workspace,device_workspace)
          call write_basic_output(trim(path),global_shape,origin,cells,axis,index,components,x, &
            pack_derived_output_gpu,identity,2160_int64,2160_int64,MPI_COMM_WORLD,'si',peak,bytes, &
            derived_indices=derived(:derived_count))
          call end_derived_output_gpu(download)
          if(download/=bytes) error stop 'private derivative transfer mismatch'
          ! Full test-observer copy is separate from the production tile counter.
          after_velocity=vel_d
#endif
        endif
        do m=1,3
          do k=-hm,km+hm
            do j=-hm,jm+hm
              do i=-hm,im+hm
                if(transfer(after_velocity(i,j,k,m),0_int64)/=transfer(before_velocity(i,j,k,m),0_int64)) &
                  error stop 'output changed solver physical velocity or stale halo'
              enddo
            enddo
          enddo
        enddo
        if(host_workspace+peak>2_int64*1024*1024.or.device_workspace+capacity*8>2_int64*1024*1024) &
          error stop 'numeric controlled budget'
        if(8_int64*(size(rho,kind=int64)+size(vel,kind=int64)+size(prs,kind=int64)+size(tmp,kind=int64)+ &
          size(x,kind=int64)+size(dxi,kind=int64)+size(before_velocity,kind=int64)+ &
          size(after_velocity,kind=int64))+host_workspace+peak>2_int64*1024*1024) &
          error stop 'numeric total controlled host arrays'
        call MPI_Reduce(bytes,global_bytes,1,MPI_INTEGER8,MPI_SUM,0,MPI_COMM_WORLD,ierr)
        write(label,'("/budget_rank",i0,".txt")') rank
        open(newunit=unit,file=trim(path)//trim(label),status='new',iostat=status)
        if(status/=0) error stop 'numeric budget file'
        write(unit,'(5(i0,1x))') peak,bytes,download,host_workspace,device_workspace+capacity*8
        close(unit)
      enddo
    enddo
    print *, 'OUTPUT_PRIVATE_DERIVATIVES_PASS'
    call MPI_Finalize(ierr)
    stop
  endif
  if(fault(1:8)=='derived_') then
    derived=[(m,m=1,14)]; derived_count=14; derived_budget=2160
    units='si'
    call get_command_argument(5,argument)
    if(len_trim(argument)>0) units=argument
    select case(trim(fault))
    case('derived_gradient')
      derived_count=9
    case('derived_curl')
      derived(:3)=[12,13,14]; derived_count=3
    case('derived_q','derived_rank_mismatch')
      derived(:2)=[10,11]; derived_count=2
      if(trim(fault)=='derived_rank_mismatch'.and.rank==1) derived(2)=12
    case('derived_empty')
      derived_count=0
    case('derived_invalid')
      derived(1)=15; derived_count=1
    case('derived_duplicate')
      derived(:2)=[10,10]; derived_count=2
    case('derived_unsorted')
      derived(:2)=[12,11]; derived_count=2
    case('derived_budget')
      derived_budget=1
    end select
    if(trim(fault)=='derived_series_volume'.or.trim(fault)=='derived_series_planes') then
      if(trim(fault)=='derived_series_volume') then
        call write_output_series_xdmf(trim(prefix)//'/series.xdmf',trim(prefix)//'/series.frames', &
          global_shape,[0],[0],components,trim(units),MPI_COMM_WORLD,derived(:derived_count))
      else
        call write_output_series_xdmf(trim(prefix)//'/series.xdmf',trim(prefix)//'/series.frames', &
          global_shape,[1,2,3],[4,3,2],components,trim(units),MPI_COMM_WORLD,derived(:derived_count))
      endif
    else if(trim(fault)=='derived_grouped'.or.trim(fault)=='derived_group_mismatch'.or. &
            trim(fault)=='derived_shared_grouped') then
      path=trim(prefix)//'/grouped'
      do axis=1,3
        index=(global_shape(axis)-1)/2
        write(label,'(a,i12.12)') tags(axis),index
        if(trim(fault)=='derived_group_mismatch'.and.axis==2) then
          derived(:2)=[10,11]; derived_count=2
        endif
        geometry_source='data.h5'
        if(trim(fault)=='derived_shared_grouped') then
          geometry_path=trim(prefix)//'/geometry'
          call write_basic_output(trim(geometry_path),global_shape,origin,cells,axis,index,components,x, &
            pack_basic_output_cpu,identity,2160_int64,2160_int64,MPI_COMM_WORLD,trim(units),peak,bytes, &
            group_name=trim(label),append=axis>1,geometry_only=.true.)
          geometry_source='../geometry/data.h5'
        endif
        call write_basic_output(trim(path),global_shape,origin,cells,axis,index,components,x, &
          pack_derived_fixture,identity,int(derived_budget,int64),int(derived_budget,int64), &
          MPI_COMM_WORLD,trim(units),peak,bytes,group_name=trim(label),append=axis>1, &
          geometry_file=trim(geometry_source), &
          derived_indices=derived(:derived_count))
      enddo
      call write_slice_output_xdmf(trim(path)//'/data.xdmf',global_shape,[1,2,3], &
        (global_shape-1)/2,components,identity,trim(units),MPI_COMM_WORLD, &
        geometry_file=trim(geometry_source), &
        derived_indices=derived(:derived_count))
    else
      do axis=0,3
        index=0
        if(axis>0) index=(global_shape(axis)-1)/2
        write(label,'("/0_",i0)') axis
        path=trim(prefix)//trim(label)
        geometry_source='data.h5'
        if(trim(fault)=='derived_shared') then
          write(label,'("/geometry_",i0)') axis
          geometry_path=trim(prefix)//trim(label)
          call write_basic_output(trim(geometry_path),global_shape,origin,cells,axis,index,components,x, &
            pack_basic_output_cpu,identity,2160_int64,2160_int64,MPI_COMM_WORLD,trim(units),peak,bytes, &
            geometry_only=.true.)
          write(geometry_source,'("../geometry_",i0,"/data.h5")') axis
        endif
        call write_basic_output(trim(path),global_shape,origin,cells,axis,index,components,x, &
          pack_derived_fixture,identity,int(derived_budget,int64),int(derived_budget,int64), &
          MPI_COMM_WORLD,trim(units),peak,bytes,geometry_only=trim(fault)=='derived_geometry', &
          geometry_file=trim(geometry_source), &
          derived_indices=derived(:derived_count))
        if(peak>derived_budget) error stop 'derived fixture budget exceeded'
        call MPI_Reduce(bytes,global_bytes,1,MPI_INTEGER8,MPI_SUM,0,MPI_COMM_WORLD,ierr)
        if(rank==0) then
          open(newunit=unit,file=trim(path)//'/budget.txt',status='new',iostat=status)
          if(status/=0) error stop 'derived budget file'
          write(unit,'(2(i0,1x))') peak,global_bytes
          close(unit)
        endif
      enddo
    endif
    print *, 'OUTPUT_DERIVED_LAYOUT_PASS'
    call MPI_Finalize(ierr)
    stop
  endif
  if(trim(fault)=='geometry_write'.or.trim(fault)=='geometry_verify') then
    do axis=0,3
      index=0
      if(axis>0) index=(global_shape(axis)-1)/2
      write(label,'("/geometry_",i0)') axis
      path=trim(prefix)//trim(label)
      call write_basic_output(trim(path),global_shape,origin,cells,axis,index,components,x, &
        pack_basic_output_cpu,identity,2160_int64,2160_int64,MPI_COMM_WORLD,'si',peak,bytes, &
        geometry_only=.true.,verify_geometry=trim(fault)=='geometry_verify')
      if(peak>2160.or.bytes/=0) error stop 'geometry validation budget or flow download'
    enddo
    print *, 'OUTPUT_GEOMETRY_PASS'
    call MPI_Finalize(ierr)
    stop
  endif
#ifdef TEST_FIELDS_GPU
  allocate(rho_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm),vel_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3), &
    prs_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm),tmp_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm))
  rho_d=rho; vel_d=vel; prs_d=prs; tmp_d=tmp
  if(components==12) then
    allocate(tve_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm),spc_d(-hm:im+hm,-hm:jm+hm,-hm:km+hm,5))
    tve_d=tve; spc_d=spc
  endif
#endif
  identity=checkpoint_state_identity(12_int64,0.012d0,0.001d0,0.001d0)
  if(trim(fault)=='grouped'.or.trim(fault)=='grouped_clock') then
    do backend=0,1
#ifndef TEST_FIELDS_GPU
      if(backend==1) cycle
#endif
      write(label,'("/",i0,"_grouped")') backend
      path=trim(prefix)//trim(label)
#ifdef TEST_FIELDS_GPU
      if(backend==1) call begin_basic_output_gpu(2160_int64/8/(components+3)*components,2160_int64,MPI_COMM_WORLD)
#endif
      do axis=1,3
        index=(global_shape(axis)-1)/2
        write(label,'(a,i12.12)') tags(axis),index
        if(trim(fault)=='grouped_clock'.and.axis==2) identity%time=0.013d0
        if(backend==0) then
          call write_basic_output(trim(path),global_shape,origin,cells,axis,index,components,x, &
            pack_basic_output_cpu,identity,2160_int64,2160_int64,MPI_COMM_WORLD,'si',peak,bytes, &
            group_name=trim(label),append=axis>1)
        else
#ifdef TEST_FIELDS_GPU
          call write_basic_output(trim(path),global_shape,origin,cells,axis,index,components,x, &
            pack_basic_output_gpu,identity,2160_int64,2160_int64,MPI_COMM_WORLD,'si',peak,bytes, &
            group_name=trim(label),append=axis>1)
#endif
        endif
      enddo
#ifdef TEST_FIELDS_GPU
      if(backend==1) call end_basic_output_gpu(download)
#endif
      call write_slice_output_xdmf(trim(path)//'/data.xdmf',global_shape,[1,2,3], &
        (global_shape-1)/2,components,identity,'si',MPI_COMM_WORLD)
    enddo
    print *, 'OUTPUT_GROUPED_FIELDS_PASS'
    call MPI_Finalize(ierr)
    stop
  endif
  if(len_trim(fault)>0) then
    path=trim(prefix)//'/0_0'; requested_components=components; axis=0; index=0; peak=2160
    units='si'
    select case(trim(fault))
    case('budget')
      peak=1
    case('index')
      axis=1; index=global_shape(1)
    case('nan')
      if(rank==0) rho(0,0,0)=ieee_value(0.d0,ieee_quiet_nan)
    case('units')
      if(rank==1) units='dimensionless'
    case('components')
      if(rank==1) requested_components=6
    case default
      error stop 'unknown rejection fixture'
    end select
    bytes=peak
    call write_basic_output(trim(path),global_shape,origin,cells,axis,index,requested_components,x, &
      pack_basic_output_cpu,identity,bytes,bytes,MPI_COMM_WORLD,trim(units),peak,download)
    error stop 'rejection fixture unexpectedly passed'
  endif
  do axis=0,3
    index=0
    if(axis>0) index=(global_shape(axis)-1)/2
    do backend=0,1
#ifndef TEST_FIELDS_GPU
      if(backend==1) cycle
#endif
      write(label,'("/",i0,"_",i0)') backend,axis
      path=trim(prefix)//trim(label)
      if(backend==0) then
        call write_basic_output(trim(path),global_shape,origin,cells,axis,index,components,x, &
          pack_basic_output_cpu,identity,2160_int64,2160_int64,MPI_COMM_WORLD,'si',peak,bytes)
        download=0
      else
#ifdef TEST_FIELDS_GPU
        call begin_basic_output_gpu(2160_int64/8/(components+3)*components,2160_int64,MPI_COMM_WORLD)
        call write_basic_output(trim(path),global_shape,origin,cells,axis,index,components,x, &
          pack_basic_output_gpu,identity,2160_int64,2160_int64,MPI_COMM_WORLD,'si',peak,bytes)
        call end_basic_output_gpu(download)
        if(download/=bytes) error stop 'GPU transfer count mismatch'
#endif
      endif
      if(peak>2160) error stop 'controlled budget exceeded'
      call MPI_Reduce(bytes,global_bytes,1,MPI_INTEGER8,MPI_SUM,0,MPI_COMM_WORLD,ierr)
      if(rank==0) then
        open(newunit=unit,file=trim(path)//'/budget.txt',status='new',iostat=status)
        if(status/=0) error stop 'budget file'
        write(unit,'(3(i0,1x))') peak,global_bytes,backend
        close(unit)
      endif
    enddo
  enddo
  print *, 'OUTPUT_FIELDS_PASS'
  call MPI_Finalize(ierr)
contains
  subroutine pack_derived_fixture(start,extent,total_components,values)
    integer,intent(in) :: start(3),extent(3),total_components
    real(real64),intent(out) :: values(:)
    integer :: ii,jj,kk,c,node,nodes
    real(real64) :: value
    nodes=product(extent)
    if(total_components/=components+derived_count) error stop 'derived packer component contract'
    call pack_basic_output_cpu(start,extent,components,values(:nodes*components))
    node=0
    do kk=start(3),start(3)+extent(3)-1
      do jj=start(2),start(2)+extent(2)-1
        do ii=start(1),start(1)+extent(1)-1
          node=node+1
          value=100.d0*(origin(1)+ii)+10.d0*(origin(2)+jj)+(origin(3)+kk)+rank/1024.d0
          do c=1,derived_count
            values((components+c-1)*nodes+node)=value+derived(c)/32.d0
          enddo
        enddo
      enddo
    enddo
    if(trim(fault)=='derived_nan'.and.rank==0) &
      values(components*nodes+1)=ieee_value(0.d0,ieee_quiet_nan)
  end subroutine
end program
