program compact_filter_probe
  use iso_c_binding, only: c_ptr,c_int,c_size_t
  use mpi
  use astr_precision, only: real_kind,real_bytes,real_to_word
  use astr_precision_mpi, only: mpi_real_type
  use ieee_arithmetic, only: ieee_is_finite
  use device_state
  use halo_transport
  use compact_operator, only: compact_line_partition,partition_compact_line,compact_filter_core,compact_filter_rhs_global
  use compact_line_solver, only: compact_matrix,compact_reference_factor,factor_compact_reference,solve_compact_reference
  use compact_line_cpp
  implicit none
  integer,parameter :: n=16,hm=5
  real(real_kind),parameter :: alpha=.49_real_kind
  real(real_kind),parameter :: base(5)=[2._real_kind,.125_real_kind,.25_real_kind,.5_real_kind,16._real_kind]
  type(c_ptr) :: state
  type(compact_distributed_cpp_plan) :: plans(3)
  type(compact_line_partition) :: part,whole
  type(compact_matrix) :: matrix,full
  type(compact_reference_factor) :: factor
  type(compact_stencil_cpp) :: stencil
  real(real_kind) :: coefficient(0:5),edge(2),pi,theta(3),error
  real(real_kind),allocatable :: global(:,:,:,:),reference(:,:,:,:),initial(:,:,:,:),out(:,:,:,:),gold(:,:,:,:), &
    actual(:,:,:,:),line_input(:,:),line_rhs(:,:),line_solution(:,:),source(:,:,:,:),zero(:,:,:,:)
  integer :: ierr,rank,ranks,local_rank,axis,cart,dims(3),coords(3),u(3),origin(3),low(3),high(3), &
    line_comm(3),line_rank,line_size,i,j,k,c,d,g(3),debug,mode,components,lines,start,nrhs,offset, &
    point(3),row,l,m,sample,unit,header(12),column
  integer(c_int) :: transport
  character(1024) :: argument,directory,path
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',argument,status=ierr)
  if(ierr/=0) error stop 'compact filter requires Open MPI local rank'
  read(argument,*) local_rank
  call check(device_bind(int(local_rank,c_int)),'pre-MPI binding')
  call MPI_Init(ierr);call check(ierr,'MPI initialization')
  call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr);call check(ierr,'rank')
  call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr);call check(ierr,'size')
  call get_command_argument(1,argument);read(argument,*) axis
  if(axis<1.or.axis>3.or.(ranks/=1.and.ranks/=2)) error stop 'compact filter fixture decomposition'
  call get_command_argument(2,argument)
  select case(trim(argument))
  case('pinned');transport=0
  case('device-aware');transport=1
  case default;error stop 'compact filter transport'
  end select
  call get_command_argument(3,directory)
  if(len_trim(directory)==0) error stop 'compact filter output directory required'
  dims=1;dims(axis)=ranks;u=n/dims
  call MPI_Cart_create(MPI_COMM_WORLD,3,dims,[.true.,.true.,.true.],.false.,cart,ierr)
  call check(ierr,'Cartesian communicator')
  call MPI_Cart_coords(cart,rank,3,coords,ierr);call check(ierr,'coordinates')
  origin=coords*u
  do d=1,3
    call MPI_Cart_shift(cart,d-1,1,low(d),high(d),ierr);call check(ierr,'neighbors')
    call MPI_Cart_sub(cart,[d==1,d==2,d==3],line_comm(d),ierr);call check(ierr,'line communicator')
  enddo
  ! The existing src/filter.F90 compact10 coefficients, not the identity fixture.
  coefficient=[(193+126*alpha)/512,(105+302*alpha)/512,(-15+30*alpha)/128, &
    (45-90*alpha)/1024,(-5+10*alpha)/512,(1-2*alpha)/1024]
  pi=acos(-1._real_kind)
  allocate(global(0:n-1,0:n-1,0:n-1,5),reference(0:n-1,0:n-1,0:n-1,5), &
    initial(-hm:u(1)+hm,-hm:u(2)+hm,-hm:u(3)+hm,5),out(0:u(1),0:u(2),0:u(3),5), &
    gold(0:u(1),0:u(2),0:u(3),5),actual(0:u(1),0:u(2),0:u(3),5), &
    line_input(-hm:n-1+hm,5*n*n),line_rhs(n,5*n*n),line_solution(n,5*n*n))
  allocate(source,mold=global);allocate(zero,mold=initial);zero=0
  if(rank==0) then
    do k=0,n-1
    do j=0,n-1
    do i=0,n-1
      theta=2*pi*real([i,j,k],real_kind)/n
      do c=1,5
        global(i,j,k,c)=base(c)+.03_real_kind*c*(sin(theta(1))+cos(5*theta(2))) &
          +.01_real_kind*c*cos(theta(1)+2*theta(2)+3*theta(3))+.005_real_kind*c*(-1)**(i+j+k)
      enddo
    enddo
    enddo
    enddo
  endif
  call MPI_Bcast(global,size(global),mpi_real_type,0,cart,ierr);call check(ierr,'canonical field')
  call partition_compact_line(n,0,1,.true.,whole)
  call compact_filter_core(whole,alpha,full,edge)
  full%cyclic=.true.;full%lower(1)=alpha;full%upper(n)=alpha
  call factor_compact_reference(full,factor)
  do k=-hm,u(3)+hm
  do j=-hm,u(2)+hm
  do i=-hm,u(1)+hm
    g=modulo(origin+[i,j,k],n)
    initial(i,j,k,:)=global(g(1),g(2),g(3),:)
  enddo
  enddo
  enddo
  do debug=0,1
  do mode=1,3
    components=merge(5,1,mode==1)
    call check(device_state_create(int(local_rank,c_int),u(1),u(2),u(3),hm,int(debug,c_int),state),'context')
    call check(device_transport_prepare(state,int(cart,c_int),int(low,c_int),int(high,c_int), &
      int(dims,c_int),5_c_int,transport,0_c_int,[1_c_int,1_c_int,1_c_int]),'halo setup')
    do d=1,3
      call MPI_Comm_rank(line_comm(d),line_rank,ierr);call check(ierr,'line rank')
      call MPI_Comm_size(line_comm(d),line_size,ierr);call check(ierr,'line size')
      call partition_compact_line(n,line_rank,line_size,.true.,part)
      call compact_filter_core(part,alpha,matrix,edge)
      lines=product(u+1)/(u(d)+1)
      nrhs=components*lines
      if(mode==3) nrhs=min(nrhs,137)
      call prepare_distributed_compact_cpp(state,matrix,edge,part,line_comm(d),nrhs,transport,plans(d))
    enddo
    do sample=1,2
      if(sample==1) then
        reference=global
        call check(device_state_upload(state,device_q,initial,bytes(size(initial))),'field input')
      else
        reference=0
        call check(device_state_upload(state,device_q,zero,bytes(size(zero))),'zero input')
      endif
      do d=1,3
        source=reference;column=0
        do c=1,5
        do m=0,n*n-1
          column=column+1
          do row=-hm,n-1+hm
            call node(d,modulo(row,n),m,n,n,point)
            line_input(row,column)=source(point(1),point(2),point(3),c)
          enddo
        enddo
        enddo
        call compact_filter_rhs_global(whole,n,hm,line_input,[0._real_kind,0._real_kind,0._real_kind,0._real_kind], &
          [0._real_kind,0._real_kind,0._real_kind,0._real_kind,0._real_kind],coefficient, &
          reshape([(0._real_kind,l=1,21)],[3,7]),line_rhs)
        call solve_compact_reference(factor,line_rhs,line_solution)
        column=0
        do c=1,5
        do m=0,n*n-1
          column=column+1
          do row=0,n-1
            call node(d,row,m,n,n,point)
            reference(point(1),point(2),point(3),c)=line_solution(row+1,column)
          enddo
        enddo
        enddo
        call check(device_exchange_axis(state,device_q,int(d-1,c_int),0_c_int),'axis halo')
        stencil%axis=d-1;stencil%kind=4;stencil%periodic=1;stencil%total=n
        stencil%first=plans(d)%partition%first;stencil%c10=coefficient
        lines=product(u+1)/(u(d)+1)
        do offset=1,5,components
          stencil%component_offset=offset
          do start=0,components*lines-1,plans(d)%capacity
            stencil%start=start;nrhs=min(plans(d)%capacity,components*lines-start)
            call check(compact_field_rhs_cpp(state,plans(d)%handle,stencil,int(nrhs,c_int)),'filter RHS')
            call solve_distributed_compact_cpp(plans(d),nrhs)
            call restore_distributed_compact_cpp(plans(d),nrhs)
            call check(compact_field_commit_cpp(state,plans(d)%handle,stencil,int(nrhs,c_int)),'filter commit')
          enddo
        enddo
        call check(device_state_download(state,device_q,initial,bytes(size(initial))),'filter observation')
        out=initial(0:u(1),0:u(2),0:u(3),:)
        if(.not.all(ieee_is_finite(out))) error stop 'nonfinite actual compact10 field'
        if(sample==2.and.any(out/=0)) error stop 'compact10 zero field invariant'
        error=0
        do c=1,5
        do k=0,u(3)
        do j=0,u(2)
        do i=0,u(1)
          g=modulo(origin+[i,j,k],n)
          error=max(error,abs(out(i,j,k,c)-reference(g(1),g(2),g(3),c))/max(1._real_kind,maxval(abs(reference(:,:,:,c)))))
        enddo
        enddo
        enddo
        enddo
        if(real_bytes==8.and.error>2.e-10_real_kind) error stop 'actual compact10 FP64 reference criterion'
        write(*,'(a,6(1x,i0),1x,es24.16e3)') 'CPP_COMPACT10',rank,axis,debug,mode,sample,d,error
        if(sample==1.and.d==3) actual=out
      enddo
    enddo
    if(debug==0.and.mode==1) then
      gold=actual
    elseif(any(real_to_word(actual)/=real_to_word(gold))) then
      error stop 'compact10 workspace/batch/sync changed words'
    endif
    write(path,'(a,"/rank",i8.8,".debug",i0,".mode",i0,".bin")') trim(directory),rank,debug,mode
    open(newunit=unit,file=trim(path),access='stream',form='unformatted',status='new')
    header=[real_bytes,rank,axis,debug,u,origin,n,mode]
    write(unit) header,actual
    close(unit)
    do d=1,3
      call release_distributed_compact_cpp(plans(d))
    enddo
    call check(device_state_destroy(state),'release context')
    ! initial was used as the download buffer; reconstruct the unchanged fixture.
    do k=-hm,u(3)+hm
    do j=-hm,u(2)+hm
    do i=-hm,u(1)+hm
      g=modulo(origin+[i,j,k],n);initial(i,j,k,:)=global(g(1),g(2),g(3),:)
    enddo
    enddo
    enddo
  enddo
  enddo
  do d=1,3
    call MPI_Comm_free(line_comm(d),ierr);call check(ierr,'release line communicator')
  enddo
  call MPI_Comm_free(cart,ierr);call check(ierr,'release Cartesian communicator')
  write(*,'(a,2(1x,i0))') 'CPP_COMPACT10_PASS',rank,real_bytes
  call MPI_Finalize(ierr)
contains
  subroutine node(direction,position,line,nx,ny,point)
    integer,intent(in) :: direction,position,line,nx,ny
    integer,intent(out) :: point(3)
    select case(direction)
    case(1);point=[position,mod(line,ny),line/ny]
    case(2);point=[mod(line,nx),position,line/nx]
    case(3);point=[mod(line,nx),line/nx,position]
    end select
  end subroutine
  integer(c_size_t) function bytes(count)
    integer,intent(in) :: count
    bytes=int(count,c_size_t)*real_bytes
  end function
  subroutine check(status,label)
    integer(c_int),intent(in) :: status
    character(*),intent(in) :: label
    if(status==0) return
    write(*,*) 'CPP_COMPACT10_STATUS ',label,status
    error stop 'compact10 component failure'
  end subroutine
end program
