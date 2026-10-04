module output_fields
  use iso_fortran_env, only: int64,real64,iostat_end
  use ieee_arithmetic, only: ieee_is_finite
  use mpi
  use hdf5
  use checkpoint_state_io, only: checkpoint_state_identity,checkpoint_state_require
  implicit none
  private
  public :: write_basic_output,pack_basic_output_cpu,basic_output_names,write_slice_output_xdmf
  public :: write_output_series_xdmf
  public :: derived_output_names,agree_derived_selection
  public :: begin_derived_output_cpu,pack_derived_output_cpu,end_derived_output_cpu
  public :: derived_output_workspace_cpu
  public :: private_derivative_layout_valid
  public :: output_basic_components
  real(real64),allocatable,save :: private_velocity(:,:,:,:)
  integer,save :: selected_cpu(14)=0
  integer,save :: private_types_cpu(3)=3
  integer(int64),save :: derived_capacity_cpu=0
  character(32),parameter :: basic_output_names(12)=[character(32) :: &
    'density','velocity_x','velocity_y','velocity_z','pressure','temperature', &
    'vibrational_temperature','mass_fraction_N2','mass_fraction_O2','mass_fraction_N', &
    'mass_fraction_O','mass_fraction_NO']
  character(32),parameter :: derived_output_names(14)=[character(32) :: &
    'velocity_gradient_xx','velocity_gradient_yx','velocity_gradient_zx', &
    'velocity_gradient_xy','velocity_gradient_yy','velocity_gradient_zy', &
    'velocity_gradient_xz','velocity_gradient_yz','velocity_gradient_zz', &
    'Q_rs','velocity_divergence','vorticity_x','vorticity_y','vorticity_z']
  abstract interface
    subroutine field_packer(start,extent,components,values)
      import real64
      integer,intent(in) :: start(3),extent(3),components
      real(real64),intent(out) :: values(:)
    end subroutine
  end interface
contains
  integer function output_basic_components() result(components)
    use commvar, only: numq
    components=6
    if(numq==11) components=12
  end function

  logical function private_derivative_layout_valid() result(valid)
    use commvar, only: im,jm,km,hm,difschm,numq,num_species,lihomo,ljhomo,lkhomo,npdci,npdcj,npdck
    use bc, only: bctype
    integer :: types(3)
    logical :: supported_state
    types=[npdci,npdcj,npdck]
    supported_state=numq==5.and.num_species==0
#ifdef ASTR_AIR5_CHEMISTRY
    supported_state=supported_state.or.(numq==11.and.num_species==5)
#endif
    valid=supported_state.and.hm>=3.and.min(im,jm,km)>=max(hm,5).and. &
      trim(difschm)=='643e'.and.all(types>=1).and.all(types<=4).and. &
      (lihomo.eqv.all(bctype(1:2)==1)).and.(ljhomo.eqv.all(bctype(3:4)==1)).and. &
      (lkhomo.eqv.all(bctype(5:6)==1))
  end function

  pure real(real64) function output_derivative6(fm3,fm2,fm1,f0,fp1,fp2,fp3,node,dim,ntype) result(value)
    use constdef, only: num1d60,num2d3,num1d12
    real(real64),intent(in) :: fm3,fm2,fm1,f0,fp1,fp2,fp3
    integer,intent(in) :: node,dim,ntype
    logical :: left,right
    ! Same closure as derivative::diff6ec; only the private output field is read.
    left=ntype==1.or.ntype==4; right=ntype==2.or.ntype==4
    if(left.and.node==0) then
      value=-0.5d0*fp2+2.d0*fp1-1.5d0*f0
    elseif((left.and.node==1).or.(right.and.node==dim-1)) then
      value=0.5d0*(fp1-fm1)
    elseif((left.and.node==2).or.(right.and.node==dim-2)) then
      value=num2d3*(fp1-fm1)-num1d12*(fp2-fm2)
    elseif(right.and.node==dim) then
      value=0.5d0*fm2-2.d0*fm1+1.5d0*f0
    else
      value=0.75d0*(fp1-fm1)-0.15d0*(fp2-fm2)+num1d60*(fp3-fm3)
    endif
  end function

  subroutine require(condition,comm,message)
    logical,intent(in) :: condition
    integer,intent(in) :: comm
    character(*),intent(in) :: message
    call checkpoint_state_require(condition,comm,'field output: '//message)
  end subroutine

  subroutine pack_basic_output_cpu(start,extent,components,values)
    use commarray, only: rho,vel,prs,tmp
#ifdef ASTR_AIR5_CHEMISTRY
    use commarray, only: tve,spc
#endif
    integer,intent(in) :: start(3),extent(3),components
    real(real64),intent(out) :: values(:)
    integer :: i,j,k,n,nodes,m
    nodes=product(extent); n=0
    do k=start(3),start(3)+extent(3)-1
      do j=start(2),start(2)+extent(2)-1
        do i=start(1),start(1)+extent(1)-1
          n=n+1
          values(n)=rho(i,j,k)
          do m=1,3
            values(m*nodes+n)=vel(i,j,k,m)
          enddo
          values(4*nodes+n)=prs(i,j,k)
          values(5*nodes+n)=tmp(i,j,k)
#ifdef ASTR_AIR5_CHEMISTRY
          if(components==12) then
            values(6*nodes+n)=tve(i,j,k)
            do m=1,5
              values((6+m)*nodes+n)=spc(i,j,k,m)
            enddo
          endif
#endif
        enddo
      enddo
    enddo
  end subroutine

  subroutine derived_output_workspace_cpu(workspace_bytes)
    use commvar, only: im,jm,km,hm
    use parallel, only: isize,jsize,ksize
    integer(int64),intent(out) :: workspace_bytes
    integer(int64) :: face,halo_bytes,private_bytes
    face=0
    if(isize>1) face=max(face,int(jm+1,int64)*int(km+1,int64))
    if(jsize>1) face=max(face,int(im+1,int64)*int(km+1,int64))
    if(ksize>1) face=max(face,int(im+1,int64)*int(jm+1,int64))
    halo_bytes=4_int64*hm*face*3*8
    private_bytes=product(int([im,jm,km]+2*hm+1,int64))*3*8
    workspace_bytes=private_bytes+halo_bytes
  end subroutine

  subroutine begin_derived_output_cpu(capacity,budget,comm,indices,workspace_bytes)
    use commvar, only: im,jm,km,hm,npdci,npdcj,npdck
    use commarray, only: vel
    use parallel, only: dataswap
    integer(int64),intent(in) :: capacity,budget
    integer,intent(in) :: comm,indices(:)
    integer(int64),intent(out) :: workspace_bytes
    integer :: status
    call require(comm==MPI_COMM_WORLD.and..not.allocated(private_velocity),comm,'private CPU output lifecycle')
    call agree_derived_selection(indices,selected_cpu,comm)
    call require(count(selected_cpu>0)>0.and.private_derivative_layout_valid(),comm, &
      'private derivatives require consistent explicit 643e layout')
    private_types_cpu=[npdci,npdcj,npdck]
    call derived_output_workspace_cpu(workspace_bytes)
    call require(capacity>0.and.capacity<=huge(status).and.budget>=workspace_bytes.and. &
      capacity<=(budget-workspace_bytes)/8,comm,'private CPU derivative budget')
    allocate(private_velocity(-hm:im+hm,-hm:jm+hm,-hm:km+hm,3),stat=status)
    call require(status==0,comm,'private CPU velocity allocation')
    private_velocity=0.d0
    private_velocity(0:im,0:jm,0:km,:)=vel(0:im,0:jm,0:km,:)
    call dataswap(private_velocity)
    call require(all(private_velocity(0:im,0:jm,0:km,:)==vel(0:im,0:jm,0:km,:)), &
      comm,'private exchange changed physical velocity')
    derived_capacity_cpu=capacity
  end subroutine

  subroutine pack_derived_output_cpu(start,extent,components,values)
    use commvar, only: im,jm,km
    use commarray, only: dxi
    integer,intent(in) :: start(3),extent(3),components
    real(real64),intent(out) :: values(:)
    real(real64) :: gradient(3,3),directional(3),fields(14)
    integer :: i,j,k,n,nodes,c,d,m,nselected,ierr,basic
    logical :: valid
    nselected=count(selected_cpu>0)
    basic=output_basic_components()
    nodes=product(extent)
    valid=allocated(private_velocity).and.all(extent>0).and.all(start>=0).and. &
      all(start+extent<=[im,jm,km]+1).and.components==basic+nselected.and. &
      size(values)==nodes*components.and.int(size(values),int64)<=derived_capacity_cpu
    ! The writer skips empty ranks, so callback rejection must not enter a collective.
    if(.not.valid) then
      write(*,'(a)') 'field output: private CPU derivative tile shape'
      call MPI_Abort(MPI_COMM_WORLD,71,ierr)
      error stop 'private CPU derivative tile rejected'
    endif
    call pack_basic_output_cpu(start,extent,basic,values(:nodes*basic))
    n=0
    do k=start(3),start(3)+extent(3)-1
      do j=start(2),start(2)+extent(2)-1
        do i=start(1),start(1)+extent(1)-1
          n=n+1
          do c=1,3
            directional(1)=output_derivative6(private_velocity(i-3,j,k,c),private_velocity(i-2,j,k,c), &
              private_velocity(i-1,j,k,c),private_velocity(i,j,k,c),private_velocity(i+1,j,k,c), &
              private_velocity(i+2,j,k,c),private_velocity(i+3,j,k,c),i,im,private_types_cpu(1))
            directional(2)=output_derivative6(private_velocity(i,j-3,k,c),private_velocity(i,j-2,k,c), &
              private_velocity(i,j-1,k,c),private_velocity(i,j,k,c),private_velocity(i,j+1,k,c), &
              private_velocity(i,j+2,k,c),private_velocity(i,j+3,k,c),j,jm,private_types_cpu(2))
            directional(3)=output_derivative6(private_velocity(i,j,k-3,c),private_velocity(i,j,k-2,c), &
              private_velocity(i,j,k-1,c),private_velocity(i,j,k,c),private_velocity(i,j,k+1,c), &
              private_velocity(i,j,k+2,c),private_velocity(i,j,k+3,c),k,km,private_types_cpu(3))
            do d=1,3
              gradient(c,d)=directional(1)*dxi(i,j,k,1,d)+directional(2)*dxi(i,j,k,2,d)+ &
                directional(3)*dxi(i,j,k,3,d)
              fields(c+3*(d-1))=gradient(c,d)
            enddo
          enddo
          fields(10)=-0.5d0*sum(gradient*transpose(gradient))
          fields(11)=gradient(1,1)+gradient(2,2)+gradient(3,3)
          fields(12)=gradient(3,2)-gradient(2,3)
          fields(13)=gradient(1,3)-gradient(3,1)
          fields(14)=gradient(2,1)-gradient(1,2)
          do m=1,nselected
            values((basic-1+m)*nodes+n)=fields(selected_cpu(m))
          enddo
        enddo
      enddo
    enddo
  end subroutine

  subroutine end_derived_output_cpu()
    if(allocated(private_velocity)) deallocate(private_velocity)
    selected_cpu=0; derived_capacity_cpu=0
  end subroutine

  subroutine write_basic_output(path,global_shape,origin,cells,axis,index,components, &
                                coordinates,pack,identity,buffer_bytes,host_budget,comm,units, &
                                peak_bytes,field_bytes,group_name,append,geometry_only,geometry_file,verify_geometry, &
                                derived_indices)
    character(*),intent(in) :: path,units
    integer,intent(in) :: global_shape(3),origin(3),cells(3),axis,index,components,comm
    real(real64),intent(in) :: coordinates(0:,0:,0:,:)
    procedure(field_packer) :: pack
    type(checkpoint_state_identity),intent(in) :: identity
    integer(int64),intent(in) :: buffer_bytes,host_budget
    integer(int64),intent(out) :: peak_bytes,field_bytes
    character(*),optional,intent(in) :: group_name
    logical,optional,intent(in) :: append
    logical,optional,intent(in) :: geometry_only
    character(*),optional,intent(in) :: geometry_file
    logical,optional,intent(in) :: verify_geometry
    integer,optional,intent(in) :: derived_indices(:)
    real(real64),allocatable :: values(:),xyz(:,:)
    integer(int64) :: capacity,local_nodes,local_tiles,max_tiles,tile,ntiles(3),tile_grid(3),header(11),agreed(11)
    integer :: owned(3),first(3),extent(3),block(3),start(3),global_start(3),dim,position,m,i,j,k,n,err,rank
    integer :: selected_dims(3),ndims,d,nodes,status,packed_components,space_rank,verify_flag,agreed_verify
    integer :: selected_derived(14),nfields
    integer(hid_t) :: file,location,access,xfer,native,sets(28),spaces(28),mem,integer_set,integer_space,dtype
    integer(hsize_t) :: shape3(3),shape4(4),counts(4),offsets(4),memory_shape(1),actual_dims(4),max_dims(4)
    character(32) :: name
    character(16) :: agreed_units
    character(64) :: group,agreed_group
    integer :: append_flag,agreed_append
    logical :: participates,adding,valid_group,only_geometry,write_coordinates,selected_set,verifying,equal_type,equal_values
    character(1024) :: geometry_source,agreed_geometry
    integer :: geometry_flag,agreed_geometry_flag
    call MPI_Comm_rank(comm,rank,err)
    call require(err==MPI_SUCCESS,comm,'rank')
    call require(all(global_shape>1).and.all(cells>0).and.all(origin>=0).and. &
      all(origin+cells<global_shape).and.axis>=0.and.axis<=3,comm,'partition layout')
    call require(components==6.or.components==12,comm,'basic component count')
#ifndef ASTR_AIR5_CHEMISTRY
    call require(components==6,comm,'AIR5 output is not built')
#endif
    call agree_derived_selection(derived_indices,selected_derived,comm)
    nfields=components+count(selected_derived>0)
    call require(units=='si'.or.units=='dimensionless',comm,'unit declaration')
    call require(all(shape(coordinates)==[cells+1,3]),comm,'coordinate shape')
    call require(identity%step>=0.and.ieee_is_finite(identity%time).and.identity%time>=0.and. &
      ieee_is_finite(identity%dt_used).and.identity%dt_used>=0.and. &
      ieee_is_finite(identity%dt_next).and.identity%dt_next>0,comm,'clock')
    call require(identity%step==0.or.identity%dt_used>0,comm,'missing completed-step dt')
    header=[1_int64,identity%step,transfer(identity%time,0_int64), &
      transfer(identity%dt_used,0_int64),transfer(identity%dt_next,0_int64),int(axis,int64),int(index,int64), &
      int(global_shape,int64),int(components,int64)]
    agreed=header
    call MPI_Bcast(agreed,11,MPI_INTEGER8,0,comm,err)
    call require(err==MPI_SUCCESS.and.all(agreed==header),comm,'clock or slice differs between ranks')
    agreed_units=units
    call MPI_Bcast(agreed_units,16,MPI_CHARACTER,0,comm,err)
    call require(err==MPI_SUCCESS.and.trim(agreed_units)==units,comm,'units differ between ranks')
    only_geometry=.false.; geometry_source='data.h5'
    verifying=.false.
    if(present(verify_geometry)) verifying=verify_geometry
    if(present(geometry_only)) only_geometry=geometry_only
    if(present(geometry_file)) geometry_source=geometry_file
    write_coordinates=geometry_source=='data.h5'
    call require(.not.only_geometry.or.write_coordinates,comm,'geometry writer cannot reference other geometry')
    call require(.not.verifying.or.only_geometry,comm,'verification requires geometry-only mode')
    call require(.not.only_geometry.or.nfields==components,comm,'geometry cannot contain derived fields')
    verify_flag=merge(1,0,verifying); agreed_verify=verify_flag
    call MPI_Bcast(agreed_verify,1,MPI_INTEGER,0,comm,err)
    call require(err==MPI_SUCCESS.and.verify_flag==agreed_verify,comm,'geometry verification differs between ranks')
    call require(len_trim(geometry_source)>0.and.scan(trim(geometry_source),'&<>"')==0, &
      comm,'invalid geometry reference')
    agreed_geometry=geometry_source; geometry_flag=merge(1,0,only_geometry); agreed_geometry_flag=geometry_flag
    call MPI_Bcast(agreed_geometry,1024,MPI_CHARACTER,0,comm,err)
    call require(err==MPI_SUCCESS.and.agreed_geometry==geometry_source,comm,'geometry reference differs between ranks')
    call MPI_Bcast(agreed_geometry_flag,1,MPI_INTEGER,0,comm,err)
    call require(err==MPI_SUCCESS.and.geometry_flag==agreed_geometry_flag,comm,'geometry role differs between ranks')
    packed_components=nfields
    if(only_geometry) packed_components=0
    if(verifying) packed_components=1
    group=''; adding=.false.
    if(present(group_name)) group=group_name
    if(present(append)) adding=append
    call require(.not.adding.or.len_trim(group)>0,comm,'append requires a named group')
    valid_group=.true.
    if(present(group_name)) valid_group=len_trim(group_name)>0.and.len_trim(group_name)<=64.and. &
      verify(trim(group_name),'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_')==0
    call require(valid_group,comm,'invalid field group name')
    agreed_group=group; append_flag=merge(1,0,adding); agreed_append=append_flag
    call MPI_Bcast(agreed_group,64,MPI_CHARACTER,0,comm,err)
    call require(err==MPI_SUCCESS.and.agreed_group==group,comm,'field group differs between ranks')
    call MPI_Bcast(agreed_append,1,MPI_INTEGER,0,comm,err)
    call require(err==MPI_SUCCESS.and.agreed_append==append_flag,comm,'append mode differs between ranks')
    if(axis>0) call require(index>=0.and.index<global_shape(max(1,axis)),comm,'slice index')
    owned=cells
    where(origin+cells==global_shape-1) owned=owned+1
    first=0
    participates=.true.
    if(axis>0) then
      participates=index>=origin(axis).and.index<origin(axis)+owned(axis)
      first(axis)=index-origin(axis)
      owned(axis)=1
    endif
    local_nodes=product(int(owned,int64))
    capacity=min(buffer_bytes,host_budget)/8/(packed_components+3)
    call require(capacity>0,comm,'host budget cannot hold one basic node')
    capacity=min(capacity,local_nodes,int(huge(nodes)/(packed_components+3),int64))
    block=1
    do d=1,3
      block(d)=int(min(int(owned(d),int64),capacity/product(int(block,int64))))
    enddo
    capacity=product(int(block,int64))
    peak_bytes=capacity*8*(packed_components+3)
    allocate(values(int(capacity)*packed_components),xyz(int(capacity),3),stat=status)
    call require(status==0,comm,'packing allocation')
    ntiles=(int(owned,int64)+block-1)/block
    local_tiles=0
    if(participates) local_tiles=product(ntiles)
    call MPI_Allreduce(local_tiles,max_tiles,1,MPI_INTEGER8,MPI_MAX,comm,err)
    call require(err==MPI_SUCCESS,comm,'tile count')
    ndims=3
    selected_dims=[1,2,3]
    if(axis>0) then
      ndims=2; position=0
      do d=1,3
        if(d==axis) cycle
        position=position+1; selected_dims(position)=d
      enddo
    endif
    shape3(:ndims)=int(global_shape(selected_dims(:ndims)),hsize_t)
    shape4(1)=3; shape4(2:ndims+1)=shape3(:ndims)
    call h5open_f(err)
    call require(err==0,comm,'HDF5 initialize')
    call h5pcreate_f(H5P_FILE_ACCESS_F,access,err)
    call require(err==0,comm,'file access property')
    call h5pset_fapl_mpio_f(access,comm,MPI_INFO_NULL,err)
    call require(err==0,comm,'parallel file access')
    if(verifying) then
      call h5fopen_f(trim(path)//'/data.h5',H5F_ACC_RDONLY_F,file,err,access_prp=access)
    else if(adding) then
      call h5fopen_f(trim(path)//'/data.h5',H5F_ACC_RDWR_F,file,err,access_prp=access)
    else
      call h5fcreate_f(trim(path)//'/data.h5',H5F_ACC_EXCL_F,file,err,access_prp=access)
    endif
    call require(err==0,comm,'exclusive create or grouped append')
    if(verifying) then
      call verify_text_attribute(file,'units',units,comm)
      call verify_text_attribute(file,'center','Node',comm)
      call verify_text_attribute(file,'phase','completed_step',comm)
    else if(.not.adding) then
      call text_attribute(file,'units',units,comm)
      call text_attribute(file,'center','Node',comm)
      call text_attribute(file,'phase','completed_step',comm)
    endif
    if(.not.only_geometry) call derived_layout_attribute(file,adding,selected_derived,comm)
    if(len_trim(group)>0.and..not.verifying) call grouped_frame_identity(file,adding,[header(1:5),header(8:11), &
      int(merge(1,0,units=='si'),int64)],comm)
    location=file
    if(len_trim(group)>0) then
      if(verifying) then
        call h5gopen_f(file,trim(group),location,err)
      else
        call h5gcreate_f(file,trim(group),location,err)
      endif
      call require(err==0,comm,'open or create field group')
    endif
    call h5pclose_f(access,err)
    call require(err==0,comm,'close access property')
    call h5pcreate_f(H5P_DATASET_XFER_F,xfer,err)
    call require(err==0,comm,'transfer property')
    call h5pset_dxpl_mpio_f(xfer,H5FD_MPIO_COLLECTIVE_F,err)
    call require(err==0,comm,'collective transfer property')
    native=h5kind_to_type(real64,H5_REAL_KIND)
    do m=1,nfields+2
      selected_set=(m<=nfields+1.and..not.only_geometry).or.(m==nfields+2.and.write_coordinates)
      if(.not.selected_set) cycle
      name=''
      if(m<=nfields) name=output_field_name(m,components,selected_derived)
      if(m==nfields+1) name='velocity'
      if(m==nfields+2) name='coordinates'
      if(verifying) then
        call h5dopen_f(location,trim(name),sets(m),err)
        call require(err==0,comm,'open geometry dataset')
        call h5dget_type_f(sets(m),dtype,err)
        call require(err==0,comm,'geometry datatype')
        call h5tequal_f(dtype,H5T_IEEE_F64LE,equal_type,err)
        call require(err==0.and.equal_type,comm,'geometry must be FP64 little-endian')
        call h5tclose_f(dtype,err)
        call require(err==0,comm,'close geometry datatype')
        call h5dget_space_f(sets(m),spaces(m),err)
        call require(err==0,comm,'geometry dataspace')
        call h5sget_simple_extent_ndims_f(spaces(m),space_rank,err)
        call require(err==0.and.space_rank==ndims+1,comm,'geometry dataset rank')
        call h5sget_simple_extent_dims_f(spaces(m),actual_dims,max_dims,err)
        call require(err>=0.and.all(actual_dims(:ndims+1)==shape4(:ndims+1)),comm,'geometry dataset shape')
        cycle
      else if(m<=nfields) then
        call h5screate_simple_f(ndims,shape3(:ndims),spaces(m),err)
      else
        call h5screate_simple_f(ndims+1,shape4(:ndims+1),spaces(m),err)
      endif
      call require(err==0,comm,'create space '//trim(name))
      call h5dcreate_f(location,trim(name),H5T_IEEE_F64LE,spaces(m),sets(m),err)
      call require(err==0,comm,'create dataset '//trim(name))
      if(m>components.and.m<=nfields) &
        call derived_quantity_attributes(sets(m),selected_derived(m-components),units,comm)
    enddo
    memory_shape=11
    if(verifying) then
      call h5dopen_f(location,'metadata',integer_set,err)
      call require(err==0,comm,'open geometry metadata')
      call h5dget_space_f(integer_set,integer_space,err)
      call require(err==0,comm,'geometry metadata space')
      call h5sget_simple_extent_ndims_f(integer_space,space_rank,err)
      call require(err==0.and.space_rank==1,comm,'geometry metadata rank')
      call h5sget_simple_extent_dims_f(integer_space,actual_dims,max_dims,err)
      call require(err>=0.and.actual_dims(1)==11,comm,'geometry metadata shape')
      call h5dget_type_f(integer_set,dtype,err)
      call require(err==0,comm,'geometry metadata type')
      call h5tequal_f(dtype,H5T_STD_I64LE,equal_type,err)
      call require(err==0.and.equal_type,comm,'geometry metadata must be int64')
      call h5tclose_f(dtype,err)
      call require(err==0,comm,'close geometry metadata type')
      call h5dread_f(integer_set,h5kind_to_type(int64,H5_INTEGER_KIND),agreed,memory_shape,err)
      call require(err==0.and.agreed(1)==header(1).and.all(agreed(6:11)==header(6:11)),comm, &
        'shared geometry layout mismatch')
    else
      call h5screate_simple_f(1,memory_shape,integer_space,err)
      call require(err==0,comm,'metadata space')
      call h5dcreate_f(location,'metadata',H5T_STD_I64LE,integer_space,integer_set,err)
      call require(err==0,comm,'metadata dataset')
      if(rank==0) then
        call h5dwrite_f(integer_set,h5kind_to_type(int64,H5_INTEGER_KIND),header,memory_shape,err)
      else
        err=0
      endif
      call require(err==0,comm,'metadata write')
    endif
    call h5dclose_f(integer_set,err)
    call require(err==0,comm,'close metadata dataset')
    call h5sclose_f(integer_space,err)
    call require(err==0,comm,'close metadata space')
    field_bytes=0
    do tile=0,max_tiles-1
      nodes=0; start=0; extent=1
      if(tile<local_tiles) then
        tile_grid=[mod(tile,ntiles(1)),mod(tile/ntiles(1),ntiles(2)),tile/(ntiles(1)*ntiles(2))]
        start=first+int(tile_grid)*block
        extent=min(block,owned-int(tile_grid)*block)
        nodes=product(extent)
        if(.not.only_geometry) call pack(start,extent,nfields,values(:nodes*nfields))
        n=0
        do k=start(3),start(3)+extent(3)-1
          do j=start(2),start(2)+extent(2)-1
            do i=start(1),start(1)+extent(1)-1
              n=n+1; xyz(n,:)=coordinates(i,j,k,:)
            enddo
          enddo
        enddo
        if(.not.only_geometry) field_bytes=field_bytes+int(nodes,int64)*nfields*8
      endif
      equal_values=all_finite_xyz(xyz,nodes)
      if(.not.only_geometry) equal_values=equal_values.and.all_finite(values(:nodes*nfields))
      call require(equal_values,comm,'nonfinite basic fields')
      global_start=origin+start
      memory_shape=max(1,nodes)
      call h5screate_simple_f(1,memory_shape,mem,err)
      call require(err==0,comm,'tile memory space')
      if(nodes==0) then
        call h5sselect_none_f(mem,err)
      else
        err=0
      endif
      call require(err==0,comm,'tile memory selection')
      do m=1,nfields
        if(only_geometry) exit
        call transfer_component(sets(m),spaces(m),values((m-1)*nodes+1:m*nodes),0)
      enddo
      do m=1,3
        if(.not.only_geometry) &
          call transfer_component(sets(nfields+1),spaces(nfields+1),values(m*nodes+1:(m+1)*nodes),m)
        if(verifying) then
          call transfer_component(sets(nfields+2),spaces(nfields+2),values(:nodes),m)
          equal_values=.true.
          do n=1,nodes
            if(transfer(values(n),0_int64)/=transfer(xyz(n,m),0_int64)) equal_values=.false.
          enddo
          call require(equal_values,comm,'shared geometry coordinate mismatch')
        else if(write_coordinates) then
          call transfer_component(sets(nfields+2),spaces(nfields+2),xyz(:nodes,m),m)
        endif
      enddo
      call h5sclose_f(mem,err)
      call require(err==0,comm,'close tile memory space')
    enddo
    do m=1,nfields+2
      selected_set=(m<=nfields+1.and..not.only_geometry).or.(m==nfields+2.and.write_coordinates)
      if(.not.selected_set) cycle
      call h5dclose_f(sets(m),err)
      call require(err==0,comm,'close field dataset')
      call h5sclose_f(spaces(m),err)
      call require(err==0,comm,'close field space')
    enddo
    call h5pclose_f(xfer,err)
    call require(err==0,comm,'close transfer property')
    if(len_trim(group)>0) then
      call h5gclose_f(location,err)
      call require(err==0,comm,'close field group')
    endif
    call h5fclose_f(file,err)
    call require(err==0,comm,'close field file')
    err=0
    if(rank==0.and.len_trim(group)==0.and..not.only_geometry) &
      call write_xdmf(trim(path)//'/data.xdmf',int(shape3(:ndims)),components,identity,units,err, &
        trim(geometry_source),derived_indices)
    if(rank/=0) err=0
    call require(err==0,comm,'write XDMF')
  contains
    subroutine transfer_component(set,space,data,vector_component)
      integer(hid_t),intent(in) :: set,space
      real(real64),intent(inout) :: data(:)
      integer,intent(in) :: vector_component
      integer :: rank_dims,shift
      real(real64) :: empty(1)
      shift=merge(1,0,vector_component>0); rank_dims=ndims+shift
      offsets(shift+1:rank_dims)=int(global_start(selected_dims(:ndims)),hsize_t)
      counts(shift+1:rank_dims)=int(extent(selected_dims(:ndims)),hsize_t)
      if(shift==1) then
        offsets(1)=vector_component-1; counts(1)=1
      endif
      if(nodes>0) then
        call h5sselect_hyperslab_f(space,H5S_SELECT_SET_F,offsets(:rank_dims),counts(:rank_dims),err)
      else
        call h5sselect_none_f(space,err)
      endif
      call require(err==0,comm,'field file selection')
      empty=0
      if(verifying.and.nodes>0) then
        call h5dread_f(set,native,data,memory_shape,err,mem_space_id=mem,file_space_id=space,xfer_prp=xfer)
      else if(verifying) then
        call h5dread_f(set,native,empty,memory_shape,err,mem_space_id=mem,file_space_id=space,xfer_prp=xfer)
      else if(nodes>0) then
        call h5dwrite_f(set,native,data,memory_shape,err,mem_space_id=mem,file_space_id=space,xfer_prp=xfer)
      else
        call h5dwrite_f(set,native,empty,memory_shape,err,mem_space_id=mem,file_space_id=space,xfer_prp=xfer)
      endif
      call require(err==0,comm,'field tile transfer')
    end subroutine
  end subroutine

  subroutine verify_text_attribute(file,name,expected,comm)
    integer(hid_t),intent(in) :: file
    character(*),intent(in) :: name,expected
    integer,intent(in) :: comm
    integer(hid_t) :: attribute,dtype,space
    integer(size_t) :: length
    integer(hsize_t) :: dims(1)
    integer :: err,rank_dims,type_class
    character(:),allocatable :: actual
    call h5aopen_f(file,name,attribute,err)
    call require(err==0,comm,'missing geometry attribute '//name)
    call h5aget_type_f(attribute,dtype,err)
    call require(err==0,comm,'geometry attribute datatype')
    call h5tget_class_f(dtype,type_class,err)
    call require(err==0.and.type_class==H5T_STRING_F,comm,'geometry attribute must be a string')
    call h5aget_space_f(attribute,space,err)
    call require(err==0,comm,'geometry attribute space')
    call h5sget_simple_extent_ndims_f(space,rank_dims,err)
    call require(err==0.and.rank_dims==0,comm,'geometry attribute must be scalar')
    call h5tget_size_f(dtype,length,err)
    call require(err==0.and.length==len(expected),comm,'geometry attribute length')
    allocate(character(int(length)) :: actual)
    dims=1
    call h5aread_f(attribute,dtype,actual,dims,err)
    call require(err==0.and.actual==expected,comm,'geometry attribute mismatch '//name)
    call h5tclose_f(dtype,err)
    call require(err==0,comm,'close geometry attribute datatype')
    call h5aclose_f(attribute,err)
    call require(err==0,comm,'close geometry attribute')
    call h5sclose_f(space,err)
    call require(err==0,comm,'close geometry attribute space')
  end subroutine

  subroutine text_attribute(file,name,text,comm)
    integer(hid_t),intent(in) :: file
    character(*),intent(in) :: name,text
    integer,intent(in) :: comm
    integer(hid_t) :: dtype,space,attribute
    integer(hsize_t) :: dims(1)
    integer :: err
    dims=1
    call h5tcopy_f(H5T_FORTRAN_S1,dtype,err)
    call require(err==0,comm,'attribute type')
    call h5tset_size_f(dtype,int(len_trim(text),size_t),err)
    call require(err==0,comm,'attribute string size')
    call h5screate_f(H5S_SCALAR_F,space,err)
    call require(err==0,comm,'attribute space')
    call h5acreate_f(file,name,dtype,space,attribute,err)
    call require(err==0,comm,'create attribute '//name)
    call h5awrite_f(attribute,dtype,trim(text),dims,err)
    call require(err==0,comm,'write attribute '//name)
    call h5aclose_f(attribute,err)
    call require(err==0,comm,'close attribute')
    call h5sclose_f(space,err)
    call require(err==0,comm,'close attribute space')
    call h5tclose_f(dtype,err)
    call require(err==0,comm,'close attribute type')
  end subroutine

  subroutine grouped_frame_identity(file,adding,expected,comm)
    integer(hid_t),intent(in) :: file
    logical,intent(in) :: adding
    integer(int64),intent(in) :: expected(10)
    integer,intent(in) :: comm
    integer(int64) :: actual(10)
    integer(hsize_t) :: dims(1)
    integer(hid_t) :: space,set,native
    integer :: err,rank
    dims=10; native=h5kind_to_type(int64,H5_INTEGER_KIND)
    call MPI_Comm_rank(comm,rank,err)
    call require(err==MPI_SUCCESS,comm,'grouped frame rank')
    if(adding) then
      call h5dopen_f(file,'frame_identity',set,err)
      call require(err==0,comm,'grouped frame identity')
      call h5dread_f(set,native,actual,dims,err)
      call require(err==0.and.all(actual==expected),comm,'grouped frame clock/layout/unit mismatch')
    else
      call h5screate_simple_f(1,dims,space,err)
      call require(err==0,comm,'grouped identity space')
      call h5dcreate_f(file,'frame_identity',H5T_STD_I64LE,space,set,err)
      call require(err==0,comm,'grouped identity create')
      if(rank==0) then
        call h5dwrite_f(set,native,expected,dims,err)
      else
        err=0
      endif
      call require(err==0,comm,'grouped identity write')
      call h5sclose_f(space,err)
      call require(err==0,comm,'grouped identity space close')
    endif
    call h5dclose_f(set,err)
    call require(err==0,comm,'grouped identity close')
  end subroutine

  logical function all_finite(values) result(valid)
    real(real64),intent(in) :: values(:)
    integer :: i
    valid=.false.
    do i=1,size(values)
      if(.not.ieee_is_finite(values(i))) return
    enddo
    valid=.true.
  end function

  logical function all_finite_xyz(values,nodes) result(valid)
    real(real64),intent(in) :: values(:,:)
    integer,intent(in) :: nodes
    integer :: i,m
    valid=.false.
    do m=1,3
      do i=1,nodes
        if(.not.ieee_is_finite(values(i,m))) return
      enddo
    enddo
    valid=.true.
  end function

  subroutine normalize_derived_selection(indices,selection,valid)
    integer,optional,intent(in) :: indices(:)
    integer,intent(out) :: selection(14)
    logical,intent(out) :: valid
    integer :: n,m
    selection=0; valid=.true.
    if(.not.present(indices)) return
    n=size(indices)
    valid=n<=size(selection)
    if(.not.valid) return
    valid=all(indices>=1.and.indices<=14)
    if(.not.valid) return
    do m=2,n
      if(indices(m)<=indices(m-1)) valid=.false.
    enddo
    if(valid) selection(:n)=indices
  end subroutine

  subroutine agree_derived_selection(indices,selection,comm)
    integer,optional,intent(in) :: indices(:)
    integer,intent(out) :: selection(14)
    integer,intent(in) :: comm
    integer :: agreed(14),err
    logical :: valid
    call normalize_derived_selection(indices,selection,valid)
    call require(valid,comm,'invalid derived field selection')
    agreed=selection
    call MPI_Bcast(agreed,size(agreed),MPI_INTEGER,0,comm,err)
    call require(err==MPI_SUCCESS.and.all(agreed==selection),comm,'derived fields differ between ranks')
  end subroutine

  pure function output_field_name(component,basic_components,selection) result(name)
    integer,intent(in) :: component,basic_components,selection(14)
    character(32) :: name
    if(component<=basic_components) then
      name=basic_output_names(component)
    else
      name=derived_output_names(selection(component-basic_components))
    endif
  end function

  subroutine derived_layout_attribute(file,adding,selection,comm)
    integer(hid_t),intent(in) :: file
    logical,intent(in) :: adding
    integer,intent(in) :: selection(14),comm
    character(64) :: declaration
    logical :: exists
    integer :: err
    write(declaration,'(14(i0,1x))') selection
    if(adding) then
      call h5aexists_f(file,'derived_layout',exists,err)
      call require(err==0.and.(exists.eqv.any(selection>0)),comm,'grouped derived layout mismatch')
      if(exists) call verify_text_attribute(file,'derived_layout',trim(declaration),comm)
    else if(any(selection>0)) then
      call text_attribute(file,'derived_layout',trim(declaration),comm)
    endif
  end subroutine

  subroutine derived_quantity_attributes(set,index,units,comm)
    integer(hid_t),intent(in) :: set
    integer,intent(in) :: index,comm
    character(*),intent(in) :: units
    character(16) :: quantity_units
    character(80),parameter :: definitions(14)=[character(80) :: &
      'du_x/dx','du_y/dx','du_z/dx','du_x/dy','du_y/dy','du_z/dy', &
      'du_x/dz','du_y/dz','du_z/dz','-0.5*tr(A*A); A(i,j)=du_i/dx_j; full strain', &
      'tr(A); A(i,j)=du_i/dx_j','du_z/dy-du_y/dz','du_x/dz-du_z/dx','du_y/dx-du_x/dy']
    quantity_units='dimensionless'
    if(units=='si') then
      quantity_units='s^-1'
      if(index==10) quantity_units='s^-2'
    endif
    call text_attribute(set,'quantity_units',trim(quantity_units),comm)
    call text_attribute(set,'definition',trim(definitions(index)),comm)
    call text_attribute(set,'coordinate_space','physical',comm)
    if(index==10) then
      call text_attribute(set,'time_dimension_exponent','-2',comm)
    else
      call text_attribute(set,'time_dimension_exponent','-1',comm)
    endif
  end subroutine

  subroutine write_xdmf(path,shape,components,identity,units,err,geometry_file,derived_indices)
    character(*),intent(in) :: path,units
    integer,intent(in) :: shape(:),components
    type(checkpoint_state_identity),intent(in) :: identity
    integer,intent(out) :: err
    character(*),optional,intent(in) :: geometry_file
    integer,optional,intent(in) :: derived_indices(:)
    integer :: unit,closed
    open(newunit=unit,file=path,status='new',action='write',iostat=err)
    if(err/=0) return
    write(unit,'(a)',iostat=err) '<?xml version="1.0"?>', &
      '<Xdmf Version="3.0"><Domain><Grid Name="series" GridType="Collection" CollectionType="Temporal">'
    if(err==0) call write_grid_xml(unit,shape,components,identity,units,'frame','',err, &
      geometry_file,derived_indices=derived_indices)
    if(err==0) write(unit,'(a)',iostat=err) '</Grid></Domain></Xdmf>'
    close(unit,iostat=closed)
    if(closed/=0) err=closed
  end subroutine

  subroutine write_slice_output_xdmf(path,global_shape,axes,indices,components,identity,units,comm, &
                                    geometry_file,derived_indices)
    character(*),intent(in) :: path,units
    integer,intent(in) :: global_shape(3),axes(:),indices(:),components,comm
    type(checkpoint_state_identity),intent(in) :: identity
    character(*),optional,intent(in) :: geometry_file
    integer,optional,intent(in) :: derived_indices(:)
    integer :: rank,err,closed,unit,n,d,p,shape(2)
    integer :: selected_derived(14)
    character(1),parameter :: tags(3)=['i','j','k']
    character(32) :: label
    character(128) :: clock
    call MPI_Comm_rank(comm,rank,err)
    call require(err==MPI_SUCCESS.and.size(axes)==size(indices).and.size(axes)>0,comm,'slice XML layout')
    call require(all(axes>=1).and.all(axes<=3),comm,'slice XML axes')
    call require(all(indices>=0).and.all(indices<global_shape(axes)),comm,'slice XML indices')
    call agree_derived_selection(derived_indices,selected_derived,comm)
    err=0; closed=0
    if(rank==0) then
      open(newunit=unit,file=path,status='new',action='write',iostat=err)
      if(err==0) then
        write(clock,'(es24.16)') identity%time
        write(unit,'(a)',iostat=err) '<?xml version="1.0"?>', &
          '<Xdmf Version="3.0"><Domain><Grid Name="series" GridType="Collection" CollectionType="Temporal">', &
          '<Grid Name="slices" GridType="Collection" CollectionType="Spatial">', &
          '<Time Value="'//trim(adjustl(clock))//'"/>'
        do n=1,size(axes)
          if(err/=0) exit
          p=0
          do d=1,3
            if(d==axes(n)) cycle
            p=p+1; shape(p)=global_shape(d)
          enddo
          write(label,'(a,i12.12)') tags(axes(n)),indices(n)
          call write_grid_xml(unit,shape,components,identity,units,trim(label),trim(label)//'/',err, &
            geometry_file,derived_indices=derived_indices)
        enddo
        if(err==0) write(unit,'(a)',iostat=err) '</Grid></Grid></Domain></Xdmf>'
        close(unit,iostat=closed)
      endif
    endif
    call require(err==0.and.closed==0,comm,'write grouped slice XDMF')
  end subroutine

  subroutine write_output_series_xdmf(path,records,global_shape,axes,indices,components,units,comm,derived_indices)
    character(*),intent(in) :: path,records,units
    integer,intent(in) :: global_shape(3),axes(:),indices(:),components,comm
    integer,optional,intent(in) :: derived_indices(:)
    integer :: selected_derived(14)
    integer :: rank,err,unit,source,closed,status,n,d,p,tokens,shape(2)
    integer(int64) :: previous_step,step
    real(real64) :: previous_time,time
    type(checkpoint_state_identity) :: identity
    character(32) :: name,label
    character(128) :: magic,clock
    character(256) :: row
    character(1201) :: data_reference,geometry_reference
    logical :: in_token,lineage
    character(1),parameter :: tags(3)=['i','j','k']
    call MPI_Comm_rank(comm,rank,err)
    call require(err==MPI_SUCCESS.and.size(axes)>0.and.size(axes)==size(indices),comm,'series layout')
    call require((size(axes)==1.and.axes(1)==0).or.all(axes>=1.and.axes<=3),comm,'series axes')
    call require(all(global_shape>0).and.(components==6.or.components==12),comm,'series shape/components')
    call agree_derived_selection(derived_indices,selected_derived,comm)
    do n=1,size(axes)
      if(axes(n)==0) cycle
      call require(indices(n)>=0.and.indices(n)<global_shape(axes(n)),comm,'series plane index')
    enddo
    err=0; closed=0
    if(rank==0) then
      open(newunit=source,file=records,status='old',action='read',iostat=err)
      if(err==0) then
        magic=''
        read(source,'(a)',iostat=err) magic
        lineage=magic=='ASTR_NATIVE_LINEAGE_1'
        if(err==0.and.magic/='ASTR_FRAME_SERIES_1'.and..not.lineage) err=1
        if(err==0) then
          open(newunit=unit,file=path,status='new',action='write',iostat=err)
          if(err==0) then
            write(unit,'(a)',iostat=err) '<?xml version="1.0"?>', &
              '<Xdmf Version="3.0"><Domain><Grid Name="series" GridType="Collection" CollectionType="Temporal">'
            previous_step=-1; previous_time=-1
            do while(err==0)
              read(source,'(a)',iostat=status) row
              if(status==iostat_end) exit
              if(status/=0) then
                err=1; exit
              endif
              ! Avoid internal-read EOF: NVHPC 26.1 propagates it to external NEWUNIT=-99.
              tokens=0; in_token=.false.
              do p=1,len_trim(row)
                if(row(p:p)==' '.or.row(p:p)==achar(9)) then
                  in_token=.false.
                else
                  if(.not.in_token) tokens=tokens+1
                  in_token=.true.
                endif
              enddo
              if(tokens/=2.or.scan(row,',/''"')/=0) then
                err=1; exit
              endif
              read(row,*,iostat=status) step,time
              if(status/=0) then
                err=1; exit
              endif
              if(step<0.or.step<=previous_step.or..not.ieee_is_finite(time).or.time<0.or.time<=previous_time) then
                err=1; exit
              endif
              identity=checkpoint_state_identity(step,time,0.0_real64,1.0_real64)
              write(name,'("step",i12.12)') step
              data_reference=trim(name)//'/data.h5'
              geometry_reference='../resources/data.h5'
              if(lineage) then
                read(source,'(a)',iostat=status) data_reference
                if(status==0) read(source,'(a)',iostat=status) geometry_reference
                if(status/=0.or..not.valid_lineage_reference(data_reference).or. &
                  .not.valid_lineage_reference(geometry_reference)) then
                  err=1; exit
                endif
              endif
              if(axes(1)==0) then
                call write_grid_xml(unit,global_shape,components,identity,units,trim(name),'',err, &
                  trim(geometry_reference),trim(data_reference),derived_indices)
              else
                write(clock,'(es24.16)') time
                write(unit,'(a)',iostat=err) '<Grid Name="'//trim(name)// &
                  '" GridType="Collection" CollectionType="Spatial">', &
                  '<Time Value="'//trim(adjustl(clock))//'"/>'
                do n=1,size(axes)
                  if(err/=0) exit
                  p=0
                  do d=1,3
                    if(d==axes(n)) cycle
                    p=p+1; shape(p)=global_shape(d)
                  enddo
                  write(label,'(a,i12.12)') tags(axes(n)),indices(n)
                  call write_grid_xml(unit,shape,components,identity,units,trim(label),trim(label)//'/',err, &
                    trim(geometry_reference),trim(data_reference),derived_indices)
                enddo
                if(err==0) write(unit,'(a)',iostat=err) '</Grid>'
              endif
              previous_step=step; previous_time=time
            enddo
            if(err==0) write(unit,'(a)',iostat=err) '</Grid></Domain></Xdmf>'
            close(unit,iostat=closed)
          endif
        endif
        close(source,iostat=status)
        if(status/=0) err=status
      endif
    endif
    call require(err==0.and.closed==0,comm,'write completed-frame time series')
  end subroutine

  pure logical function valid_lineage_reference(value) result(valid)
    character(*),intent(in) :: value
    integer :: i
    valid=len_trim(value)>0.and.len_trim(value)<=1200
    if(.not.valid) return
    valid=value(1:1)/='/'.and.index(value,':')==0
    do i=1,len_trim(value)
      if(iachar(value(i:i))<32.or.iachar(value(i:i))==127) valid=.false.
    enddo
  end function

  pure function xml_text(value) result(escaped)
    character(*),intent(in) :: value
    character(:),allocatable :: escaped
    integer :: i
    escaped=''
    do i=1,len_trim(value)
      select case(value(i:i))
      case('&')
        escaped=escaped//'&amp;'
      case('<')
        escaped=escaped//'&lt;'
      case('>')
        escaped=escaped//'&gt;'
      case default
        escaped=escaped//value(i:i)
      end select
    enddo
  end function

  subroutine write_grid_xml(unit,shape,components,identity,units,label,prefix,err,geometry_file,data_file,derived_indices)
    integer,intent(in) :: unit,shape(:),components
    type(checkpoint_state_identity),intent(in) :: identity
    character(*),intent(in) :: units,label,prefix
    integer,intent(out) :: err
    character(*),optional,intent(in) :: geometry_file
    character(*),optional,intent(in) :: data_file
    integer,optional,intent(in) :: derived_indices(:)
    integer :: m,selection(14),nfields
    logical :: valid
    character(32) :: field_name
    character(128) :: dimensions,clock
    character(16) :: topology
    character(1200) :: geometry_source,data_source
    character(:),allocatable :: geometry_xml,data_xml
    geometry_source='data.h5'
    if(present(geometry_file)) geometry_source=geometry_file
    data_source='data.h5'
    if(present(data_file)) data_source=data_file
    geometry_xml=xml_text(geometry_source)
    data_xml=xml_text(data_source)
    call normalize_derived_selection(derived_indices,selection,valid)
    if(.not.valid) then
      err=1; return
    endif
    nfields=components+count(selection>0)
    write(dimensions,'(*(i0,1x))') shape(size(shape):1:-1)
    write(clock,'(es24.16)') identity%time
    write(topology,'(i0,"DSMesh")') size(shape)
    write(unit,'(a)',iostat=err) '<Grid Name="'//label//'" GridType="Uniform">', &
      '<Time Value="'//trim(adjustl(clock))//'"/>', &
      '<Information Name="units" Value="'//trim(units)//'"/>', &
      '<Topology TopologyType="'//trim(topology)//'" Dimensions="'//trim(dimensions)//'"/>', &
      '<Geometry GeometryType="XYZ">', &
      '<DataItem Dimensions="'//trim(dimensions)//' 3" NumberType="Float" Precision="8" Format="HDF">', &
      geometry_xml//':/'//prefix//'coordinates</DataItem></Geometry>'
    do m=1,nfields+1
      if(err/=0) exit
      if(m<=nfields) then
        field_name=output_field_name(m,components,selection)
        write(unit,'(a)',iostat=err) '<Attribute Name="'//trim(field_name)// &
          '" AttributeType="Scalar" Center="Node">', &
          '<DataItem Dimensions="'//trim(dimensions)//'" NumberType="Float" Precision="8" Format="HDF">', &
          data_xml//':/'//prefix//trim(field_name)//'</DataItem></Attribute>'
      else
        write(unit,'(a)',iostat=err) '<Attribute Name="velocity" AttributeType="Vector" Center="Node">', &
          '<DataItem Dimensions="'//trim(dimensions)//' 3" NumberType="Float" Precision="8" Format="HDF">', &
          data_xml//':/'//prefix//'velocity</DataItem></Attribute>'
      endif
    enddo
    if(err==0) write(unit,'(a)',iostat=err) '</Grid>'
  end subroutine
end module
