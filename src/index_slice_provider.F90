module index_slice_provider
  use mpi
  use iso_fortran_env, only: int64
  use astr_precision, only: real_kind,real_bytes
  use astr_precision_mpi, only: mpi_real_type
  use benchmark_runtime, only: insitu_clock,report_insitu_timing
  use commvar, only: ia,ja,ka,im,jm,km,use_gpu
  use parallel, only: ig0,jg0,kg0
  use commarray, only: x,dxi
  use insitu_run_config, only: catalog_product
  use index_slice_source, only: slice_source_value_cpu
  use index_slice_stencil, only: slice_stencil_start,slice_stencil_weights,slice_diagnostics
#ifdef _CUDA
  use index_slice_gpu
#endif
  implicit none
  private
  public :: prepare_index_slice
contains
  subroutine prepare_index_slice(definition,source,count,values,xyz,first,extent,ok,message, &
                                 neighborhood_bytes,download_bytes,exchange_bytes,compute_seconds)
    type(catalog_product),intent(in) :: definition
    integer,intent(in) :: source
    integer(int64),intent(in) :: count
    real(real_kind),allocatable,intent(out) :: values(:,:),xyz(:,:)
    integer,intent(out) :: first(3),extent(3)
    logical,intent(out) :: ok
    character(*),intent(out) :: message
    integer(int64),intent(out) :: neighborhood_bytes,download_bytes,exchange_bytes
    real(real_kind),intent(out) :: compute_seconds
    integer :: intervals(3),origin(3),cells(3),owned_upper(3),bounds(2,3),box(2,3),shape(3),nsend,nrecv,capacity
    integer,allocatable :: boxes(:,:,:),owners(:,:,:),indices(:,:),positions(:),recv_indices(:,:),recv_positions(:)
    real(real_kind),allocatable :: neighborhood(:,:),send_values(:,:),recv_values(:,:)
    integer :: raw_ids(12),nraw,nfields,nout,nn,rank,ranks,comm,err,status,d,m,i,j,k,n,p,dest,src,local(3)
    integer :: point(3),sample(3),stencil(3),offset(3),pos,c,s,field,slot,header(2,3)
    logical :: selected(12),periodic(3),derived,valid
    real(real_kind) :: weights(7,0:6),directional(4,3),gradient(3,3),density_gradient(3),diagnostic(15),started
    real(8) :: phase_started
    integer :: step
    block
      use commvar, only: nstep
      step=nstep
    end block
    phase_started=insitu_clock()
    call MPI_Comm_dup(MPI_COMM_WORLD,comm,err)
    call mpi_ok(err)
    call MPI_Comm_rank(comm,rank,err);call mpi_ok(err)
    call MPI_Comm_size(comm,ranks,err);call mpi_ok(err)
    intervals=[ia,ja,ka];origin=[ig0,jg0,kg0];cells=[im,jm,km]
    block
      use commvar, only: lihomo,ljhomo,lkhomo
      periodic=[lihomo,ljhomo,lkhomo]
    end block
    bounds=int(definition%ranges);bounds(:,definition%axis)=int(definition%index)
    first=max(origin,bounds(1,:));owned_upper=origin+cells-1
    where(origin+cells==intervals) owned_upper=owned_upper+1
    extent=max(0,min(owned_upper,bounds(2,:))-first+1)
    if(any(extent==0)) extent=0
    nfields=size(definition%fields)
    derived=any(definition%fields>=13)
    ok=.false.;message='';neighborhood_bytes=0;download_bytes=0;exchange_bytes=0;compute_seconds=0
    call agree_allocation(product(int(extent,int64))<=int(huge(nout),int64)/nfields,valid)
    if(.not.valid) then
      message='selected slice exceeds addressable packing indices'
      call MPI_Comm_free(comm,err);call mpi_ok(err);return
    endif
    nout=product(extent)
    if(derived.and.any(intervals<6)) then
      message='sixth-order slice diagnostics need at least seven global nodes in each direction'
      call MPI_Comm_free(comm,err);call mpi_ok(err);return
    endif
    selected=.false.
    do m=1,nfields
      field=definition%fields(m)
      if(field<=12.and.source/=0) selected(field)=.true.
      if(field>=13.and.field<=21) selected(2+mod(field-13,3))=.true.
      if(field==22.or.field==23) selected(2:4)=.true.
      if(field==24) selected(3:4)=.true.
      if(field==25) selected([2,4])=.true.
      if(field==26) selected(2:3)=.true.
      if(field==27) selected(1)=.true.
    enddo
    nraw=0
    do m=1,12
      if(.not.selected(m)) cycle
      nraw=nraw+1;raw_ids(nraw)=m
    enddo
    call slice_stencil_weights(weights)
    box(1,:)=first;box(2,:)=first+extent-1
    if(derived.and.nout>0) then
      do d=1,3
        box(1,d)=slice_stencil_start(first(d),intervals(d),periodic(d))
        box(2,d)=slice_stencil_start(first(d)+extent(d)-1,intervals(d),periodic(d))+6
      enddo
    endif
    if(nout==0.or.nraw==0) then
      box(1,:)=0;box(2,:)=-1
    endif
    shape=max(0,box(2,:)-box(1,:)+1)
    call agree_allocation(product(int(shape,int64))<=int(huge(nn),int64)/max(1,nraw),valid)
    if(.not.valid) then
      message='slice stencil neighborhood exceeds addressable packing indices'
      call MPI_Comm_free(comm,err);call mpi_ok(err);return
    endif
    nn=product(shape)
    allocate(values(nout,nfields),xyz(3,nout),boxes(2,3,ranks),owners(2,3,ranks),stat=status)
    call agree_allocation(status==0,valid)
    if(.not.valid) then
      message='cannot allocate selected two-dimensional slice';goto 900
    endif
    header(1,:)=origin;header(2,:)=owned_upper
    ! A periodic final node is an alias for stencil reads, not another source owner.
    where(periodic.and.origin+cells==intervals) header(2,:)=intervals-1
    call MPI_Allgather(box,6,MPI_INTEGER,boxes,6,MPI_INTEGER,comm,err);call mpi_ok(err)
    call MPI_Allgather(header,6,MPI_INTEGER,owners,6,MPI_INTEGER,comm,err);call mpi_ok(err)
    capacity=0
    do p=1,ranks
      capacity=max(capacity,product(max(0,boxes(2,:,p)-boxes(1,:,p)+1)))
    enddo
    allocate(indices(3,capacity),positions(capacity),recv_indices(3,capacity),recv_positions(capacity),stat=status)
    call agree_allocation(status==0,valid)
    if(.not.valid) then
      message='cannot allocate slice neighborhood index lists';goto 900
    endif
    n=0
    do k=0,extent(3)-1
    do j=0,extent(2)-1
    do i=0,extent(1)-1
      n=n+1;local=first-origin+[i,j,k];xyz(:,n)=x(local(1),local(2),local(3),:)
    enddo
    enddo
    enddo
    neighborhood_bytes=int(nn,int64)*nraw*real_bytes
#ifdef _CUDA
    if(use_gpu) then
      call allocate_index_slice_gpu(nn,nout,raw_ids(:nraw),definition%fields,weights,valid)
    else
#endif
      allocate(neighborhood(nn,nraw),stat=status);valid=status==0
#ifdef _CUDA
    endif
#endif
    call agree_allocation(valid,valid)
    if(.not.valid) then
      message='cannot allocate slice stencil neighborhood';goto 900
    endif
    call slice_source_list(boxes(:,:,rank+1),owners(:,:,rank+1),intervals,periodic,indices,positions,nsend)
#ifdef _CUDA
    if(use_gpu) then
      call fill_index_slice_gpu(indices(:,:nsend),positions(:nsend),source,count,valid)
    else
#endif
      do n=1,nsend
        do m=1,nraw
          neighborhood(positions(n),m)=slice_source_value_cpu(indices(1,n),indices(2,n),indices(3,n),raw_ids(m),source)
        enddo
      enddo
      valid=.true.
#ifdef _CUDA
    endif
#endif
    call agree_allocation(valid,valid)
    if(.not.valid) then
      message='cannot pack local slice stencil';goto 900
    endif
    do p=1,ranks-1
      if(nraw==0) exit
      dest=mod(rank+p,ranks);src=mod(rank-p+ranks,ranks)
      call slice_source_list(boxes(:,:,dest+1),owners(:,:,rank+1),intervals,periodic,indices,positions,nsend)
      call slice_source_list(boxes(:,:,rank+1),owners(:,:,src+1),intervals,periodic,recv_indices,recv_positions,nrecv)
      allocate(send_values(nsend,nraw),recv_values(nrecv,nraw),stat=status)
      call agree_allocation(status==0,valid)
      if(.not.valid) then
        message='cannot allocate private slice communication strip';goto 900
      endif
#ifdef _CUDA
      if(use_gpu) then
        call sample_remote_slice_gpu(indices(:,:nsend),source,count,send_values,valid)
      else
#endif
        do n=1,nsend
          do m=1,nraw
            send_values(n,m)=slice_source_value_cpu(indices(1,n),indices(2,n),indices(3,n),raw_ids(m),source)
          enddo
        enddo
        valid=.true.
#ifdef _CUDA
      endif
#endif
      call agree_allocation(valid,valid)
      if(.not.valid) then
        message='cannot pack private slice communication strip';goto 900
      endif
      call MPI_Sendrecv(send_values,size(send_values),mpi_real_type,dest,1, &
        recv_values,size(recv_values),mpi_real_type,src,1,comm,MPI_STATUS_IGNORE,err)
      call mpi_ok(err)
      exchange_bytes=exchange_bytes+int(size(send_values)+size(recv_values),int64)*real_bytes
#ifdef _CUDA
      if(use_gpu) then
        call receive_index_slice_gpu(recv_positions(:nrecv),recv_values,valid)
      else
#endif
        do m=1,nraw
          neighborhood(recv_positions(:nrecv),m)=recv_values(:,m)
        enddo
        valid=.true.
#ifdef _CUDA
      endif
#endif
      call agree_allocation(valid,valid)
      if(.not.valid) then
        message='cannot store private slice communication strip';goto 900
      endif
      deallocate(send_values,recv_values)
    enddo
    call report_insitu_timing('index_slice_neighborhood',phase_started,step)
    started=MPI_Wtime()
    phase_started=insitu_clock()
#ifdef _CUDA
    if(use_gpu) then
      call evaluate_index_slice_gpu(first,extent,box(1,:),shape,origin,intervals,periodic,source,count)
      compute_seconds=real(MPI_Wtime()-started,real_kind)
      call report_insitu_timing('index_slice_diagnostics',phase_started,step)
      phase_started=insitu_clock()
      call download_index_slice_gpu(values,valid)
      call report_insitu_timing('index_slice_field_download',phase_started,step)
      download_bytes=int(size(values),int64)*real_bytes
    else
#endif
      n=0
      do k=0,extent(3)-1
      do j=0,extent(2)-1
      do i=0,extent(1)-1
        n=n+1;point=first+[i,j,k];local=point-origin
        directional=0;gradient=0;density_gradient=0
        if(derived) then
          do d=1,3
            stencil(d)=slice_stencil_start(point(d),intervals(d),periodic(d))
          enddo
          offset=point-stencil
          do d=1,3
            do s=0,6
              sample=point-box(1,:);sample(d)=stencil(d)+s-box(1,d)
              pos=1+sample(1)+shape(1)*(sample(2)+shape(2)*sample(3))
              do m=1,nraw
                field=raw_ids(m)
                if(field>4) cycle
                directional(field,d)=directional(field,d)+weights(s+1,offset(d))*neighborhood(pos,m)
              enddo
            enddo
          enddo
          do d=1,3
            do c=1,3
              gradient(c,d)=sum(directional(c+1,:)*dxi(local(1),local(2),local(3),:,d))
            enddo
            density_gradient(d)=sum(directional(1,:)*dxi(local(1),local(2),local(3),:,d))
          enddo
        endif
        call slice_diagnostics(gradient,density_gradient,diagnostic)
        sample=point-box(1,:);pos=1+sample(1)+shape(1)*(sample(2)+shape(2)*sample(3))
        do m=1,nfields
          field=definition%fields(m)
          if(field>=13) then
            values(n,m)=diagnostic(field-12)
          elseif(source==0) then
            values(n,m)=slice_source_value_cpu(local(1),local(2),local(3),field,source)
          else
            slot=findloc(raw_ids(:nraw),field,dim=1)
            values(n,m)=neighborhood(pos,slot)
          endif
        enddo
      enddo
      enddo
      enddo
      compute_seconds=real(MPI_Wtime()-started,real_kind);valid=.true.
      call report_insitu_timing('index_slice_diagnostics',phase_started,step)
#ifdef _CUDA
    endif
#endif
    call agree_allocation(valid,ok)
    if(.not.ok) message='cannot download selected slice fields'
900 continue
#ifdef _CUDA
    if(use_gpu) call release_index_slice_gpu()
#endif
    call MPI_Comm_free(comm,err);call mpi_ok(err)
  contains
    subroutine mpi_ok(code)
      integer,intent(in) :: code
      integer :: ierr
      if(code/=MPI_SUCCESS) call MPI_Abort(MPI_COMM_WORLD,code,ierr)
    end subroutine

    subroutine agree_allocation(local_ok,global_ok)
      logical,value :: local_ok
      logical,intent(out) :: global_ok
      integer :: bad,total
      bad=merge(0,1,local_ok)
      call MPI_Allreduce(bad,total,1,MPI_INTEGER,MPI_MAX,comm,err);call mpi_ok(err)
      global_ok=total==0
    end subroutine

  end subroutine

  subroutine slice_source_list(box,owner,intervals,periodic,idx,positions,number)
    integer,intent(in) :: box(2,3),owner(2,3),intervals(3)
    logical,intent(in) :: periodic(3)
    integer,intent(out) :: idx(:,:),positions(:),number
    integer :: a,b,c,global(3),canonical(3),d,position
    number=0;position=0
      do c=box(1,3),box(2,3)
      do b=box(1,2),box(2,2)
      do a=box(1,1),box(2,1)
        position=position+1;global=[a,b,c];canonical=global
        do d=1,3
          if(periodic(d)) canonical(d)=modulo(global(d),intervals(d))
        enddo
        if(any(canonical<owner(1,:)).or.any(canonical>owner(2,:))) cycle
        number=number+1
        idx(:,number)=canonical-owner(1,:);positions(number)=position
      enddo
      enddo
      enddo
  end subroutine
end module
