program air5_chemistry_phase_probe
  use iso_c_binding
  use mpi
  use device_state
  use chemistry_device_properties,only:air5_device_prepare
  use chemistry_device_coupling
  implicit none
  integer,parameter :: im=8,jm=7,km=6,h=5
  integer,parameter :: fields(9)=[device_q,device_air5_carry,device_air5_history,device_rho, &
    device_velocity,device_temperature,device_air5_tv,device_pressure,device_air5_species]
  type saved_field
    real(c_double),allocatable :: values(:)
  end type
  type(saved_field) :: baseline(9)
  type(c_ptr) :: handle(2)
  type(device_view) :: view
  type(air5_device_options) :: options
  type(air5_device_report) :: local,reference
  type(air5_chemistry_global_report) :: global
  real(c_double) :: base(11),initial(-h:im+h,-h:jm+h,-h:km+h,11)
  real(c_double) :: low(0:im,0:jm,0:km,11),history(0:im,0:jm,0:km)
  real(c_double),allocatable :: expected(:),actual(:),poison(:)
  real(c_double),allocatable :: all_minimums(:,:),all_maximums(:,:)
  integer(c_int),allocatable :: all_counts(:,:),all_status(:)
  integer(c_int64_t),allocatable :: all_activity(:,:)
  integer(c_int) :: first(3,2),last(3,2),regions,physics,returned,devices,device
  integer :: rank,ranks,cart,topology(3),coord(3),i,j,k,c,n,u,status,ierr,test,side,field
  logical :: collect,packed,activity,periods(3)
  character(1024) :: input,text

  call get_command_argument(1,input)
  do i=1,3
    call get_command_argument(i+1,text);read(text,*)topology(i)
  enddo
  call get_environment_variable('OMPI_COMM_WORLD_LOCAL_RANK',text,status=status)
  if(status/=0)error stop 'missing local rank'
  read(text,*)rank
  call check(device_count(devices))
  if(devices<1)error stop 'no visible device'
  device=modulo(int(rank,c_int),devices)
  call check(device_bind(device))
  call MPI_Init(ierr);call check(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD,ranks,ierr);call check(ierr)
  if(product(topology)/=ranks)error stop 'topology size'
  periods=.false.
  call MPI_Cart_create(MPI_COMM_WORLD,3,topology,periods,.false.,cart,ierr);call check(ierr)
  call MPI_Comm_rank(cart,rank,ierr);call check(ierr)
  call MPI_Cart_coords(cart,rank,3,coord,ierr);call check(ierr)
  open(newunit=u,file=trim(input),access='stream',form='unformatted',status='old')
  read(u)base;close(u)
  allocate(all_minimums(3,ranks),all_maximums(2,ranks),all_counts(4,ranks),all_status(ranks),all_activity(2,ranks))
  first(:,1)=[merge(1,0,coord(1)==0),1,0];last(:,1)=[im,jm-1,km]
  first(:,2)=[merge(1,0,coord(1)==0),jm,0];last(:,2)=[im,jm,km]
  regions=merge(2_c_int,1_c_int,coord(2)==topology(2)-1)
  do test=1,9
    collect=test/=3.and.test/=9;packed=test/=1.and.test/=6
    activity=test==4
    options=air5_device_options(1_c_int,1_c_int,0_c_int,0_c_int,1_c_int,32_c_int, &
      int(modulo(test,2),c_int),1e-9_c_double,1e-13_c_double,0._c_double)
    if(activity)options%activity_threshold=1e30_c_double
    if(test==5)then
      options%source_mode=1;options%source_only=1
    endif
    if(test==6)options%source_mode=2
    if(test==7)options%source_mode=3
    initial=-321._c_double
    do c=1,11;do k=0,km;do j=0,jm;do i=0,im
      initial(i,j,k,c)=base(c)*(1._c_double+real(rank,c_double)/16._c_double+real(i+2*j+3*k,c_double)/1024._c_double)
    enddo;enddo;enddo;enddo
    if(test>=8.and.rank==ranks-1)initial(first(1,1),1,0,1)=-1._c_double
    low=0._c_double;low(0,0,0,1)=-0._c_double;history=1e-10_c_double
    do side=1,2
      call check(device_state_create(device,int(im,c_int),int(jm,c_int),int(km,c_int), &
        int(h,c_int),int(modulo(test,2),c_int),handle(side)))
      call check(air5_device_prepare(handle(side)));call check(air5_chemistry_prepare(handle(side),options))
      call check(device_state_upload(handle(side),device_q,initial,int(size(initial),c_size_t)*8_c_size_t))
      call check(device_state_upload(handle(side),device_air5_carry,low,int(size(low),c_size_t)*8_c_size_t))
      call check(device_state_upload(handle(side),device_air5_history,history,int(size(history),c_size_t)*8_c_size_t))
      do field=4,size(fields)
        call check(device_state_view(handle(side),int(fields(field),c_int),view));allocate(poison(product(view%extent)))
        poison=-987._c_double
        call check(device_state_upload(handle(side),int(fields(field),c_int),poison,int(size(poison),c_size_t)*8_c_size_t))
        deallocate(poison)
      enddo
    enddo
    ! Characterization reference: existing begin/region/report calls, before
    ! adding any MPI policy. Global reference uses Allgather and host extrema.
    call check(air5_chemistry_begin(handle(1)))
    do i=1,regions
      call check(air5_chemistry_region(handle(1),first(:,i),last(:,i),1e-10_c_double,merge(1,0,collect)))
    enddo
    call check(air5_chemistry_report(handle(1),reference))
    call MPI_Allgather(reference%status,1,MPI_INTEGER,all_status,1,MPI_INTEGER,cart,ierr);call check(ierr)
    call MPI_Allgather(reference%max_counts,4,MPI_INTEGER,all_counts,4,MPI_INTEGER,cart,ierr);call check(ierr)
    call MPI_Allgather([reference%skipped,reference%total],2,MPI_INTEGER8,all_activity,2,MPI_INTEGER8,cart,ierr);call check(ierr)
    call MPI_Allgather(transfer(reference%minimum_bits,[0._c_double,0._c_double,0._c_double]),3, &
      MPI_DOUBLE_PRECISION,all_minimums,3,MPI_DOUBLE_PRECISION,cart,ierr);call check(ierr)
    call MPI_Allgather(transfer(reference%maximum_bits,[0._c_double,0._c_double]),2, &
      MPI_DOUBLE_PRECISION,all_maximums,2,MPI_DOUBLE_PRECISION,cart,ierr);call check(ierr)
    returned=air5_chemistry_apply_regions(handle(2),regions,first,last,1e-10_c_double,collect,packed,activity, &
      int(cart,c_int),local,global)
    call require(returned==maxval(all_status).and.global%status==returned,'global status')
    call require(global%collected.eqv.(collect.and.returned==0),'diagnostic validity')
    call require(local%status==reference%status,'local status')
    if(test<8)call require(returned==0,'unexpected valid-state failure')
    if(test>=8)then
      call require(returned/=0,'injected rank failure disappeared')
      call require(all(local%failure==reference%failure),'local failure identity')
      call words(local%failure_q,reference%failure_q,'failure input q')
      call words(local%failure_carry,reference%failure_carry,'failure input carry')
    endif
    if(collect.and.returned==0)then
      call require(all(global%max_counts==maxval(all_counts,dim=2)),'global work counts')
      call words(global%minimums,minval(all_minimums,dim=2),'global minima')
      call words(global%maximums,maxval(all_maximums,dim=2),'global maxima')
      if(activity)then
        call require(global%skipped==sum(all_activity(1,:)).and.global%total==sum(all_activity(2,:)),'global activity')
        call require(global%skipped==global%total.and.global%total>0,'test activity bypass')
      endif
      n=product(last(:,1)-first(:,1)+1)
      if(regions==2)n=n+product(last(:,2)-first(:,2)+1)
      call require(local%total==int(n,c_int64_t),'local active boxes')
    endif
    do field=1,size(fields)
      call check(device_state_view(handle(1),int(fields(field),c_int),view));n=product(view%extent)
      allocate(expected(n),actual(n))
      call check(device_state_download(handle(1),int(fields(field),c_int),expected,int(n,c_size_t)*8_c_size_t))
      call check(device_state_download(handle(2),int(fields(field),c_int),actual,int(n,c_size_t)*8_c_size_t))
      call words(actual,expected,'phase changed device math/commit')
      if(test==1)then
        allocate(baseline(field)%values(n));baseline(field)%values=actual
      elseif(test<=3)then
        call words(actual,baseline(field)%values,'packed/quiet changed device state')
      endif
      deallocate(expected,actual)
    enddo
    do side=1,2
      call check(device_state_destroy(handle(side)))
    enddo
  enddo
  write(*,'(a,i0,a,i0)')'AIR5_CHEMISTRY_PHASE_PASS rank=',rank, &
    ' cases=9 global_failures=2 device_words=exact device=',device
  call MPI_Comm_free(cart,ierr);call check(ierr)
  call MPI_Finalize(ierr)
contains
  subroutine check(code)
    integer,intent(in) :: code
    if(code/=0)then
      write(*,*)'chemistry phase API/MPI failure',code
      call MPI_Abort(MPI_COMM_WORLD,code,ierr)
    endif
  end subroutine
  subroutine require(valid,label)
    logical,intent(in) :: valid
    character(*),intent(in) :: label
    if(valid)return
    write(*,*)'chemistry phase assertion rank/test=',rank,test,trim(label)
    call MPI_Abort(MPI_COMM_WORLD,99,ierr)
  end subroutine
  subroutine words(a,b,label)
    real(c_double),intent(in) :: a(:),b(:)
    character(*),intent(in) :: label
    call require(size(a)==size(b),label)
    call require(all(transfer(a,[0_c_int64_t],size(a))==transfer(b,[0_c_int64_t],size(b))),label)
  end subroutine
end program
